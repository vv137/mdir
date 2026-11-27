/* Distributed electrostatic PME [Essmann1995, Liang2026]. */
#include "../include/mdir/Driver/ProlateSpectrum.h"
#include "mdrt_dist_stats.h"
#include "mdrt_pencils.h"
#ifdef MDIR_ENABLE_MPI
#include "mdrt_mpi.h"
#include <mpi.h>
MPI_Request mdrtMPIStartChargeExchange(int64_t *, int *, int *, int64_t *, int *, int *);
#endif
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void mdrtDistPhase(int64_t);
extern int64_t mdrtDistWord(double, double, int64_t);
extern int64_t mdrtDistNarrowWord(__int128, int64_t);
struct Rows {
  int64_t *base, *data, offset, size, stride;
};
struct Field {
  double *base, *data;
  int64_t offset, n[3], s[3];
};
struct Table {
  double *base, *data;
  int64_t offset, n[2], s[2];
};
struct Results {
  double *base, *data;
  int64_t offset, n, s;
};
struct Window {
  int64_t *first;
  double *w[3], *dw[3];
};
/* A periodic box is stored contiguously starting at its lower corner.
   Constraint groups can straddle a periodic face, so the lower corner is
   chosen from the occupied stencils each step. */
struct Subgrid {
  int start[3], len[3];
};
struct Particle {
  int64_t first[3];
  double q;
};
struct DomainMesh {
  struct Subgrid box;
  struct Particle *atoms;
  double *weights, *derivatives;
  void *points;
};
struct Exchange {
  int *counts, *ns, *nr, *os, *or;
  int *domains;
  size_t sent, received;
};

static void fail(const char *s) {
#ifdef MDIR_ENABLE_MPI
  if (mdrtMPIActive())
    mdrtMPIAbort(s);
#endif
  fprintf(stderr, "distributed PME: %s\n", s);
  abort();
}
static void *alloc(size_t n, size_t size) {
  void *p = calloc(n ? n : 1, size);
  if (!p)
    fail("insufficient memory for the PME mesh");
  return p;
}
static double *field(struct Field *v, int64_t d, int64_t r) {
  return v->data + v->offset + d * v->s[0] + r * v->s[1];
}
static double *table(struct Table *t, int row) {
  return t->data + t->offset + row * t->s[0];
}
static double series(const double *a, int n, double t) {
  double b = 0, c = 0;
  for (int j = n - 1; j > 0; --j) {
    double next = (2 * t * b - c) + a[j];
    c = b;
    b = next;
  }
  return (t * b - c) + a[0];
}
static inline double precision(double x, int mixed) {
  return mixed ? (double)(float)x : x;
}

static void spline(double t, int order, double *w, double *dw, int mixed) {
  t = precision(t, mixed);
  w[0] = precision(1 - t, mixed);
  w[1] = t;
  for (int k = 3; k <= order; ++k) {
    if (k == order) {
      dw[0] = -w[0];
      for (int j = 1; j < order; ++j)
        dw[j] = precision(w[j - 1] - w[j], mixed);
    }
    double inv = precision(1.0 / (k - 1), mixed);
    w[k - 1] = precision(precision(inv * t, mixed) * w[k - 2], mixed);
    for (int j = 1; j < k - 1; ++j) {
      int i = k - 1 - j;
      double lo = precision(precision(t + j, mixed) * w[i - 1], mixed);
      double hi = precision(precision(k - j - t, mixed) * w[i], mixed);
      w[i] = precision(inv * precision(lo + hi, mixed), mixed);
    }
    w[0] = precision(precision(inv * precision(1 - t, mixed), mixed) * w[0],
                     mixed);
  }
}
static void window(struct Window *w, const double *x, int stride,
                   const double box[3], const int n[3], int order,
                   struct Table *t, int mixed) {
  for (int a = 0; a < 3; ++a) {
    double s = x[a * stride] / box[a], u = (s - floor(s)) * n[a];
    if (t->n[0] == 5) {
      u += n[a] - (order % 2) * 0.5;
      double f = floor(u), v = 2 * (u - f) - 1;
      w->first[a] = ((int)f - order / 2 + 1) % n[a];
      int terms = (int)table(t, 4)[0];
      for (int k = 0; k < order; ++k) {
        w->w[a][k] =
            precision(series(table(t, 3) + k * terms, terms, v), mixed);
        w->dw[a][k] = precision(
            series(table(t, 3) + (order + k) * terms, terms, v), mixed);
      }
    } else {
      double f = floor(u);
      w->first[a] = ((int)f + n[a] - order + 1) % n[a];
      spline(u - f, order, w->w[a], w->dw[a], mixed);
    }
  }
}
static struct Window particleWindow(struct DomainMesh *local, size_t i,
                                    int order) {
  struct Window w = {.first = local->atoms[i].first};
  for (int a = 0; a < 3; ++a) {
    size_t offset = (3 * i + a) * order;
    w.w[a] = local->weights + offset;
    w.dw[a] = local->derivatives + offset;
  }
  return w;
}
static int wrap(int64_t x, int n) { return x >= n ? x - n : x; }
static size_t volume(struct Subgrid g) {
  return (size_t)g.len[0] * g.len[1] * g.len[2];
}

/* Remove the largest empty cyclic interval on each axis. */
static struct Subgrid footprint(struct Particle *p, size_t count,
                                const int n[3], int order, int threaded) {
  struct Subgrid g = {0};
  size_t extent = (size_t)n[0]+n[1]+n[2];
  size_t width = extent > 1024 ? extent : 1024;
  size_t blocks = count ? 1+(count-1)/width : 1;
  char *occupied = alloc(blocks,extent);
  #pragma omp parallel if(threaded && count > 512)
  {
    #pragma omp for schedule(static)
    for (size_t b = 0; b < blocks; ++b) {
      size_t end = (b+1)*width < count ? (b+1)*width : count;
      for (size_t i = b*width; i < end; ++i) {
        if (!p[i].q) continue;
        size_t base = b*extent;
        for (int a = 0; a < 3; ++a) {
          occupied[base+p[i].first[a]] = 1;
          base += n[a];
        }
      }
    }
    #pragma omp for schedule(static)
    for (size_t j = 0; j < extent; ++j)
      for (size_t b = 1; b < blocks; ++b) occupied[j] |= occupied[b*extent+j];
    #pragma omp single
    {
      size_t base = 0;
      for (int a = 0; a < 3; ++a) {
        const char *used = occupied+base;
        base += n[a];
        int last = -1, first = -1, gap = 0;
        for (int j = 0; j < n[a]; ++j)
          if (used[j]) {
            if (first < 0)
              first = j;
            else if (j - last - 1 > gap) {
              gap = j - last - 1;
              g.start[a] = j;
            }
            last = j;
          }
        if (first < 0)
          continue;
        int cyclic = (n[a] - 1 - last) + first;
        if (cyclic >= gap) {
          gap = cyclic;
          g.start[a] = first;
        }
        g.len[a] = n[a] - (gap >= order ? gap - order + 1 : 0);
        if (g.len[a] == n[a])
          g.start[a] = 0;
      }
    }
    #pragma omp for schedule(static) nowait
    for (size_t i = 0; i < count; ++i)
      if (p[i].q)
        for (int a = 0; a < 3; ++a) {
          int x = p[i].first[a] - g.start[a];
          p[i].first[a] = x < 0 ? x + n[a] : x;
        }
  }
  free(occupied);
  return g;
}

static inline void addCharge(int64_t *point, int64_t word) {
  if (__builtin_add_overflow(*point, word, point))
    fail("PME charge sum outside signed Q40 range");
}

static inline void spreadFixedPlane(const struct DomainMesh *local,
                                    struct Subgrid g, int x, int order,
                                    int mixed, int64_t *grid,
                                    const size_t *offset, const size_t *ids) {
  int nx = g.len[0], ny = g.len[1], nz = g.len[2];
  for (int j = 0; j < order; ++j) {
    int start = x - j;
    if (start < 0)
      start += nx;
    for (size_t h = offset[start]; h < offset[start + 1]; ++h) {
      size_t id = ids[h];
      const struct Particle *atom = local->atoms + id;
      const double *w = local->weights + id * (3 * order);
      double qx = precision(precision(atom->q, mixed) * 0x1p40 * w[j], mixed);
      for (int k = 0; k < order; ++k) {
        int y = wrap((int64_t)atom->first[1] + k, ny);
        size_t row = ((size_t)x * ny + y) * nz;
        double qxy = mixed ? (float)qx * (float)w[order + k] : qx * w[order + k];
        for (int l = 0; l < order; ++l) {
          size_t at = row + wrap((int64_t)atom->first[2] + l, nz);
          double v = mixed ? (float)qxy * (float)w[2 * order + l] : qxy * w[2 * order + l];
          double word = nearbyint(v);
          if (!(word >= -0x1p63 && word < 0x1p63))
            fail("PME assignment outside signed Q40 range");
          addCharge(grid + at, (int64_t)word);
        }
      }
    }
  }
}
/* Group the starts by x. A thread owns an output plane and visits the
   order incident bins, avoiding atomics and private copies of the mesh. */
static void spreadFixed(const struct DomainMesh *local, size_t count,
                        struct Subgrid g, int order, int mixed, int threaded,
                        int64_t *grid) {
  const struct Particle *p = local->atoms;
  int nx = g.len[0];
  if (!nx) return;
  size_t *offset = alloc((size_t)nx + 1, sizeof *offset);
  size_t width = nx > 512 ? (size_t)nx : 512;
  size_t blocks = count ? 1+(count-1)/width : 1;
  size_t *next = alloc(blocks,(size_t)nx*sizeof *next), *ids = alloc(count,sizeof *ids);
  #pragma omp parallel if(threaded && count > 512)
  {
    #pragma omp for schedule(static)
    for (size_t b = 0; b < blocks; ++b) {
      size_t end = (b+1)*width < count ? (b+1)*width : count;
      for (size_t i = b*width; i < end; ++i)
        if (p[i].q) ++next[b*nx+p[i].first[0]];
    }
    #pragma omp single
    {
      for (int x = 0; x < nx; ++x) {
        size_t pos = offset[x];
        for (size_t b = 0; b < blocks; ++b) {
          size_t n = next[b*nx+x];
          next[b*nx+x] = pos;
          pos += n;
        }
        offset[x+1] = pos;
      }
    }
    #pragma omp for schedule(static)
    for (size_t b = 0; b < blocks; ++b) {
      size_t end = (b+1)*width < count ? (b+1)*width : count;
      for (size_t i = b*width; i < end; ++i)
        if (p[i].q) ids[next[b*nx+p[i].first[0]]++] = i;
    }
    #pragma omp for schedule(static) nowait
    for (int x = 0; x < nx; ++x)
      spreadFixedPlane(local, g, x, order, mixed, grid, offset, ids);
  }
  free(next);
  free(offset);
  free(ids);
}
/* Mixed inverse grid values, weights and derivatives are already rounded
   to binary32. The ordered gradient fold also rounds each sum to binary32.
   Keep each rounded product separate from its gradient addition. */
static void gatherMixed(const struct Window *w, struct Subgrid g, int order,
                        const double *potential, double grad[3]) {
  float gx = 0, gy = 0, gz = 0;
  float wz[8], dwz[8];
  for (int l = 0; l < order; ++l) {
    wz[l] = w->w[2][l];
    dwz[l] = w->dw[2][l];
  }
  int firstZ = w->first[2];
  int contiguous = g.len[2] - firstZ;
  if (contiguous > order) contiguous = order;
  for (int j = 0; j < order; ++j) {
    int x = wrap(w->first[0] + j, g.len[0]);
    float wx = w->w[0][j], dwx = w->dw[0][j];
    for (int k = 0; k < order; ++k) {
      int y = wrap(w->first[1] + k, g.len[1]);
      size_t row = ((size_t)x * g.len[1] + y) * g.len[2];
      float wy = w->w[1][k], dwy = w->dw[1][k];
      float xy = wx * wy, dx = dwx * wy, dy = wx * dwy;
      float values[8], tx[8], ty[8], tz[8];
      for (int l = 0; l < contiguous; ++l)
        values[l] = potential[row + firstZ + l];
      for (int l = contiguous; l < order; ++l)
        values[l] = potential[row + l - contiguous];
      #pragma omp simd
      for (int l = 0; l < order; ++l) {
        tx[l] = values[l] * (dx * wz[l]);
        ty[l] = values[l] * (dy * wz[l]);
        tz[l] = values[l] * (xy * dwz[l]);
      }
      for (int l = 0; l < order; ++l) {
        gx += tx[l];
        gy += ty[l];
        gz += tz[l];
      }
    }
  }
  grad[0] = gx;
  grad[1] = gy;
  grad[2] = gz;
}

static void gather(const struct Window *w, struct Subgrid g, int order,
                   const double *potential, double grad[3]) {
  for (int j = 0; j < order; ++j) {
    int x = wrap((int64_t)w->first[0] + j, g.len[0]);
    for (int k = 0; k < order; ++k) {
      int y = wrap((int64_t)w->first[1] + k, g.len[1]);
      size_t row = ((size_t)x * g.len[1] + y) * g.len[2];
      double xy = w->w[0][j] * w->w[1][k];
      double dx = w->dw[0][j] * w->w[1][k];
      double dy = w->w[0][j] * w->dw[1][k];
      for (int l = 0; l < order; ++l) {
        double v = potential[row + wrap((int64_t)w->first[2] + l, g.len[2])];
        double deriv[3] = {dx * w->w[2][l], dy * w->w[2][l], xy * w->dw[2][l]};
        for (int a = 0; a < 3; ++a)
          grad[a] += v * deriv[a];
      }
    }
  }
}

enum Move {
  Count,
  PackCharge,
  SumCharge,
  PackPotential,
  UnpackPotential,
  AddLocalCharge,
  CopyLocalPotential
};

struct Segment {
  int local, global, len;
};
/* Cut a periodic interval at the cell boundary and clip both pieces to
   the mesh owner's interval. The partitions may be unequal in width. */
static int segments(int start, int len, int period, int lo, int hi,
                    struct Segment parts[2]) {
  int count = 0;
  for (int pass = 0; pass < 2; ++pass) {
    int64_t origin = (int64_t)start - (pass ? period : 0);
    int64_t left = origin > lo ? origin : lo;
    int64_t right = origin + len < hi ? origin + len : hi;
    if (right > left)
      parts[count++] = (struct Segment){left - origin, left, right - left};
  }
  return count;
}
static void sumRow(int64_t *to, size_t dst, const int64_t *from,
                   size_t src, int len) {
  for (int k = 0; k < len; ++k)
    addCharge(to + dst + k, from[src + k]);
}
static inline void transferRow(struct Subgrid g, struct PmeBox b, const int n[3],
    enum Move op, void *field, void *packet, struct Segment span[3][2],
    const int parts[3], size_t base[2][2][2], int row, int col) {
  const size_t width = sizeof(int64_t);
  int i = row >= span[0][0].len, j = col >= span[1][0].len;
  int u = row - (i ? span[0][0].len : 0), v = col - (j ? span[1][0].len : 0);
  for (int k = 0; k < parts[2]; ++k) {
    struct Segment x = span[0][i], y = span[1][j], z = span[2][k];
    size_t local =
        ((size_t)(x.local + u) * g.len[1] + y.local + v) * g.len[2] +
        z.local;
    size_t pencil =
        ((size_t)(x.global + u - b.lo[0]) * (b.hi[1] - b.lo[1]) +
         y.global + v - b.lo[1]) *
            n[2] +
        z.global;
    size_t atPacket = base[i][j][k] + ((size_t)u * y.len + v) * z.len;
    switch (op) {
    case PackCharge:
    case PackPotential: {
      size_t at = op == PackCharge ? local : pencil;
      memcpy((char *)packet + atPacket * width,
             (char *)field + at * width, z.len * width);
      break;
    }
    case SumCharge:
      sumRow(field, pencil, packet, atPacket, z.len);
      break;
    case AddLocalCharge:
      sumRow(packet, pencil, field, local, z.len);
      break;
    case UnpackPotential:
    case CopyLocalPotential: {
      size_t at = op == CopyLocalPotential ? pencil : atPacket;
      memcpy((double *)field + local, (double *)packet + at,
             z.len * sizeof(double));
      break;
    }
    case Count:
      break;
    }
  }
}

static size_t transfer(struct Subgrid g, struct PmeBox b, const int n[3],
                       enum Move op, void *field, void *packet,
                       int threaded) {
  struct Segment span[3][2];
  int parts[3];
  size_t count = 1;
  for (int a = 0; a < 3; ++a) {
    parts[a] = segments(g.start[a], g.len[a], n[a], b.lo[a], b.hi[a], span[a]);
    if (!parts[a])
      return 0;
    count *= span[a][0].len + (parts[a] == 2 ? span[a][1].len : 0);
  }
  if (op == Count)
    return count;
  size_t offset = 0, base[2][2][2];
  for (int i = 0; i < parts[0]; ++i)
    for (int j = 0; j < parts[1]; ++j)
      for (int k = 0; k < parts[2]; ++k) {
        base[i][j][k] = offset;
        offset += (size_t)span[0][i].len*span[1][j].len*span[2][k].len;
      }
  int nx = span[0][0].len + (parts[0] == 2 ? span[0][1].len : 0);
  int ny = span[1][0].len + (parts[1] == 2 ? span[1][1].len : 0);
  if (threaded && (size_t)nx * ny >= 128) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < nx; ++row)
      for (int col = 0; col < ny; ++col)
        transferRow(g, b, n, op, field, packet, span, parts, base, row, col);
  } else {
    for (int row = 0; row < nx; ++row)
      for (int col = 0; col < ny; ++col)
        transferRow(g, b, n, op, field, packet, span, parts, base, row, col);
  }
  return count;
}
static int append(size_t *total, size_t count) {
  if (count > INT_MAX - *total)
    fail("PME overlap exceeds MPI count range");
  *total += count;
  return (int)count;
}
/* A rank can own several particle domains. Keep each subgrid compact,
   concatenate its rows with the other domains destined for this pencil. */
static size_t moveDomains(struct DomainMesh *domains, int first, int past,
                          struct PmeBox pencil, const int n[3], enum Move op,
                          void *mesh, void *packet, int threaded) {
  size_t count = 0, width = sizeof(int64_t);
  for (int d = first; d < past; ++d) {
    void *field =
        op == PackCharge || op == UnpackPotential ? domains[d].points : mesh;
    count += transfer(domains[d].box, pencil, n, op, field,
                      packet ? (char *)packet + count * width : NULL, threaded);
  }
  return count;
}
static struct Exchange schedule(struct DomainMesh *domains, int nd, int first,
                                int past, const int n[3],
                                struct PmePencils *mesh, int peers, int rank) {
  struct Exchange e = {0};
  e.counts = alloc(peers, 4 * sizeof(int));
  e.ns = e.counts;
  e.nr = e.ns + peers;
  e.os = e.nr + peers;
  e.or = e.os + peers;
  e.domains = alloc((size_t)peers + 1, sizeof *e.domains);
  for (int r = 0; r < peers; ++r)
    e.domains[r + 1] = e.domains[r] + nd / peers + (r < nd % peers);
#ifdef MDIR_ENABLE_MPI
  if (peers > 1) {
    double *metadata = alloc(nd, 6 * sizeof(double));
    for (int d = first; d < past; ++d)
      for (int a = 0; a < 3; ++a) {
        metadata[6 * d + a] = domains[d].box.start[a];
        metadata[6 * d + a + 3] = domains[d].box.len[a];
      }
    for (int r = 0; r < peers; ++r) {
      e.ns[r] = (e.domains[r + 1] - e.domains[r]) * 6;
      e.os[r] = e.domains[r] * 6;
    }
    mdrtMPIShareBlocks(metadata, e.ns, e.os, 0);
    for (int d = 0; d < nd; ++d)
      for (int a = 0; a < 3; ++a) {
        domains[d].box.start[a] = metadata[6 * d + a];
        domains[d].box.len[a] = metadata[6 * d + a + 3];
      }
    free(metadata);
  }
#endif
  for (int r = 0; r < peers; ++r) {
    e.os[r] = e.sent;
    e.or[r] = e.received;
    if (r == rank) {
      e.ns[r] = e.nr[r] = 0;
      continue;
    }
    e.ns[r] = append(&e.sent, moveDomains(domains, first, past,
                                          mdrtPmeRealBoxAt(mesh, r), n, Count,
                                          NULL, NULL, 0));
    e.nr[r] =
        append(&e.received,
               moveDomains(domains, e.domains[r], e.domains[r + 1],
                           mdrtPmeRealBox(mesh), n, Count, NULL, NULL, 0));
  }
  return e;
}
static size_t boxSize(struct PmeBox b) {
  return (size_t)(b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) *
         (b.hi[2] - b.lo[2]);
}
/* D440. Hermite interpolation of the cosine transform below the transition.
   The endpoint expansion above it is exact for the finite polynomial. */
static void polynomial(double k, const double *value, const double *slope,
                       double *v, double *kdv) {
  if (k < SpectrumTransition) {
    double scaled = k * SpectrumSamplesPerUnit;
    int i = (int)scaled;
    double u = scaled - i, delta = value[i + 1] - value[i];
    double quadratic = 3 * delta - 2 * slope[i] - slope[i + 1];
    double cubic = slope[i] + slope[i + 1] - 2 * delta;
    *v = value[i] + u * (slope[i] + u * (quadratic + u * cubic));
    *kdv = scaled * (slope[i] + u * (2 * quadratic + 3 * u * cubic));
    return;
  }
  double inv = 1 / k, ratio = SpectrumTransition * inv, x = ratio * ratio;
  double a = 0, b = 0, da = 0, db = 0;
  for (int i = SpectrumTerms - 1; i >= SpectrumKnots; --i) {
    da = a + x * da;
    db = b + x * db;
    a = value[i] + x * a;
    b = slope[i] + x * b;
  }
  a *= inv;
  b *= inv * inv;
  da = -(a + 2 * x * inv * da);
  db = -2 * (b + x * inv * inv * db);
  double sn = sin(k), cs = cos(k);
  *v = sn * a + cs * b;
  *kdv = k * (cs * a - sn * b) + (sn * da + cs * db);
}
/* The Gaussian influence factor separates by axis. Round the factors
   at the same point as the ordinary mixed influence calculation. */
struct MixedMode {
  double wave;
  float square, factor;
};

static void spectrum(const int n[3], const double box[3], struct PmeBox b,
                     double *fft, struct Table *t, double beta, double coulomb,
                     int fixed, int mixed, int threaded, double *sums, __int128 words[10]) {
  const double pi = acos(-1),
               pref = coulomb / (pi * (box[0] * box[1] * box[2]));
  const double exponent = pi * pi / (beta * beta);
  struct MixedMode *axisModes[3] = {NULL, NULL, NULL};
  int mixedGaussian = mixed && t->n[0] == 3;
  if (mixedGaussian)
    for (int a = 0; a < 3; ++a) {
      axisModes[a] = alloc(b.hi[a]-b.lo[a], sizeof *axisModes[a]);
      for (int k = b.lo[a]; k < b.hi[a]; ++k) {
        double wave = (a != 2 && k > n[a]/2 ? k-n[a] : k) / box[a];
        double square = wave * wave;
        double factor = exp(-exponent * square) * table(t, a)[k];
        axisModes[a][k-b.lo[a]] = (struct MixedMode){
            wave, (float)square, (float)(a == 0 ? factor * pref : factor)};
      }
    }
  size_t planes = b.hi[0]-b.lo[0];
  double (*planeSums)[10] = fixed ? NULL : alloc(planes,sizeof *planeSums);
  __int128 (*planeWords)[10] = fixed ? alloc(planes,sizeof *planeWords) : NULL;
  #pragma omp parallel for schedule(static) if(threaded && boxSize(b) >= 1024)
  for (int x = b.lo[0]; x < b.hi[0]; ++x) {
    int plane = x-b.lo[0];
    size_t i = 2*(size_t)plane*(b.hi[1]-b.lo[1])*(b.hi[2]-b.lo[2]);
    for (int y = b.lo[1]; y < b.hi[1]; ++y)
      for (int z = b.lo[2]; z < b.hi[2]; ++z, i += 2) {
        struct MixedMode mx, my, mz;
        double m[3];
        if (mixedGaussian) {
          mx = axisModes[0][plane];
          my = axisModes[1][y-b.lo[1]];
          mz = axisModes[2][z-b.lo[2]];
          m[0] = mx.wave;
          m[1] = my.wave;
          m[2] = mz.wave;
        } else {
          m[0] = (x > n[0]/2 ? x-n[0] : x) / box[0];
          m[1] = (y > n[1]/2 ? y-n[1] : y) / box[1];
          m[2] = z / box[2];
        }
        double s = m[0]*m[0] + m[1]*m[1] + m[2]*m[2];
        double g = 0, dg = 0;
        if (s > 0 && mixedGaussian) {
          float inverse = 1.0f / (mx.square + my.square + mz.square);
          g = ((mx.factor * my.factor) * mz.factor) * inverse;
          dg = -2 * g * (inverse + (float)exponent);
        } else if (s > 0) {
          double mods = table(t, 0)[x] * table(t, 1)[y] * table(t, 2)[z];
          double scale = pref * mods / s;
          if (t->n[0] == 5) {
            const double *radial = table(t, 4);
            double u = radial[3] * sqrt(s);
            if (u < 1) {
              int terms = (int)radial[2];
              double v = 2 * u * u - 1;
              double kernel, derivative;
              if (radial[3] < 0)
                polynomial(-u, radial + 4, radial + 4 + terms, &kernel,
                           &derivative);
              else {
                kernel = series(radial + 4, terms, v);
                derivative = series(radial + 4 + terms, terms, v);
              }
              g = scale * kernel;
              dg = (scale * derivative - 2 * g) / s;
            }
          } else {
            g = scale * exp(-exponent * s);
            dg = -2 * g * (1 / s + exponent);
          }
        }
        g = precision(g, mixed);
        double re = fft[i], im = fft[i + 1];
        double power =
            (z == 0 || 2 * z == n[2] ? 0.5 : 1) * (re * re + im * im);
        fft[i] = mixed ? (float)re * (float)g : re * g;
        fft[i + 1] = mixed ? (float)im * (float)g : im * g;
        double term[10] = {power * g};
        for (int a = 0; a < 3; ++a)
          for (int d = 0; d < 3; ++d)
            term[1 + 3 * a + d] = power * ((a == d ? g : 0) + dg * m[a] * m[d]);
        for (int k = 0; k < 10; ++k) {
          if (fixed)
            planeWords[plane][k] += mdrtDistWord(term[k], 0x1p32, DistPairEnergy);
          else
            planeSums[plane][k] += term[k];
        }
      }
  }
  for (size_t plane = 0; plane < planes; ++plane)
    for (int k = 0; k < 10; ++k) {
      if (fixed) words[k] += planeWords[plane][k];
      else sums[k] += planeSums[plane][k];
    }
  free(planeWords);
  free(planeSums);
  for (int a = 0; a < 3; ++a)
    free(axisModes[a]);
}
void _mlir_ciface_mdrtDistSpatialPME(struct Rows *rows, int64_t first, int64_t past,
                              struct Field *x, struct Field *q, struct Table *t,
                              struct Field *f, struct Results *result,
                              double lx, double ly, double lz, double beta,
                              double coulomb, int64_t nx, int64_t ny,
                              int64_t nz, int64_t order, int64_t px, int64_t py,
                              int64_t mpi, int64_t fixed, int64_t mixed,
                              int64_t threaded, int64_t gx, int64_t gy, int64_t gz) {
  if (nx < 1 || ny < 1 || nz < 1 || nx > INT_MAX || ny > INT_MAX ||
      nz > INT_MAX || nx * ny > INT64_MAX / nz || order < 3 || order > 8)
    fail("unsupported PME mesh size or assignment order");
  const int n[3] = {nx, ny, nz};
  const double box[3] = {lx, ly, lz};
  int peers = 1, rank = 0;
#ifdef MDIR_ENABLE_MPI
  if (mpi) {
    peers = mdrtMPISize();
    rank = mdrtMPIRank();
  }
#endif
  struct PmePencils *mesh =
      mdrtPmeSpatialPencils(nx, ny, nz, px, py, order, mpi, mixed, gx, gy, gz);
  struct PmeBox realBox = mdrtPmeRealBox(mesh),
                modes = mdrtPmeSpectrumBox(mesh);
  mdrtDistPhase(DistSpread);
  if (rows->size > INT_MAX / 6)
    fail("PME domain count exceeds MPI metadata range");
  struct DomainMesh *domains = alloc(rows->size, sizeof *domains);
  size_t width = sizeof(int64_t);
  for (int64_t d = first; d < past; ++d) {
    int64_t num = rows->data[rows->offset + d * rows->stride];
    struct DomainMesh *local = domains + d;
    local->atoms = alloc(num, sizeof *local->atoms);
    local->weights = alloc(num, 3 * order * sizeof(double));
    local->derivatives = alloc(num, 3 * order * sizeof(double));
#pragma omp parallel for schedule(static) if (threaded)
    for (int64_t r = 0; r < num; ++r) {
      struct Particle *p = local->atoms + r;
      p->q = *field(q, d, r);
      if (p->q) {
        struct Window w = particleWindow(local, r, order);
        window(&w, field(x, d, r), x->s[2], box, n, order, t, mixed);
      }
    }
    local->box = footprint(local->atoms, num, n, order, threaded);
    local->points = alloc(volume(local->box), width);
    spreadFixed(local, num, local->box, order, mixed, threaded,
                  local->points);
  }
  mdrtDistPhase(-DistSpread);
  mdrtDistPhase(DistOverlap);
  struct Exchange e =
      schedule(domains, rows->size, first, past, n, mesh, peers, rank);
  /* Off-rank payload sent by this rank. Metadata is the six box values
     delivered to each peer. The reverse exchange sends the incoming rows. */
  mdrtDistMeshBytes(DistOverlapBytes, (int64_t)e.sent * width +
      (int64_t)e.received * sizeof(double) + (past-first)*6*(peers-1)*sizeof(double));
  void *send = alloc(e.sent, width), *recv = alloc(e.received, width);
  for (int r = 0; r < peers; ++r)
    if (r != rank)
      moveDomains(domains, first, past, mdrtPmeRealBoxAt(mesh, r), n,
                  PackCharge, NULL, (char *)send + width * e.os[r],
                  threaded);

#ifdef MDIR_ENABLE_MPI
  MPI_Request request = MPI_REQUEST_NULL;
  if (peers > 1)
    request = mdrtMPIStartChargeExchange(send, e.ns, e.os, recv, e.nr, e.or);
#endif
  size_t size = boxSize(realBox);
  double *real = alloc(size, sizeof *real);
  int64_t *grid = alloc(size, sizeof *grid);
  for (int d = first; d < past; ++d)
    transfer(domains[d].box, realBox, n, AddLocalCharge,
             domains[d].points, grid, threaded);
#ifdef MDIR_ENABLE_MPI
  if (peers > 1) MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
  for (int r = 0; r < peers; ++r)
    if (r != rank)
      moveDomains(domains, e.domains[r], e.domains[r + 1], realBox, n,
                  SumCharge, grid, (char *)recv + width * e.or[r],
                  threaded);
#pragma omp parallel for schedule(static) if (threaded)
  for (size_t i = 0; i < size; ++i)
    real[i] = mixed ? (float)grid[i] * 0x1p-40 : (double)grid[i] * 0x1p-40;
  free(grid);
  mdrtDistPhase(-DistOverlap);
  double *fft = alloc(boxSize(modes), 2 * sizeof(double));
  mdrtDistPhase(DistFFT);
  mdrtPmeForward(mesh, real, fft);
  mdrtDistPhase(-DistFFT);
  mdrtDistPhase(DistSolve);
  double sums[10] = {0};
  __int128 words[10] = {0};
  spectrum(n, box, modes, fft, t, beta, coulomb, fixed, mixed, threaded, sums, words);
#ifdef MDIR_ENABLE_MPI
  if (peers > 1) {
    int *counts = alloc(peers, 2 * sizeof(int)), *offsets = counts + peers;
    size_t width = fixed ? sizeof(__int128) : sizeof(double);
    char *all = alloc(peers, 10 * width);
    for (int r = 0; r < peers; ++r) {
      counts[r] = 10;
      offsets[r] = r * 10;
    }
    memcpy(all + rank * 10 * width, fixed ? (void *)words : (void *)sums,
           10 * width);
    mdrtMPIShareBlocks(all, counts, offsets, fixed);
    memset(sums, 0, sizeof sums);
    memset(words, 0, sizeof words);
    for (int r = 0; r < peers; ++r)
      for (int k = 0; k < 10; ++k)
        if (fixed)
          words[k] += ((__int128 *)all)[r * 10 + k];
        else
          sums[k] += ((double *)all)[r * 10 + k];
    free(all);
    free(counts);
  }
#endif
  for (int k = 0; k < 10; ++k)
    result->data[result->offset + k * result->s] =
        fixed ? mdrtDistNarrowWord(words[k], DistPairEnergy) * 0x1p-32 : sums[k];
  mdrtDistPhase(-DistSolve);
  mdrtDistPhase(DistFFT);
  mdrtPmeBackward(mesh, fft, real);
  mdrtDistPhase(-DistFFT);
  mdrtDistPhase(DistOverlap);
  /* Reverse the schedule. Both buffers can hold at least one double per point.
   */
  double *out = recv, *in = send;
  for (int r = 0; r < peers; ++r)
    if (r != rank)
      moveDomains(domains, e.domains[r], e.domains[r + 1], realBox, n,
                  PackPotential, real, out + e.or[r], threaded);
#ifdef MDIR_ENABLE_MPI
  if (peers > 1)
    mdrtMPIExchange(out, e.nr, e.or, in, e.ns, e.os, 0);
#endif
  for (int r = 0; r < peers; ++r)
    if (r != rank)
      moveDomains(domains, first, past, mdrtPmeRealBoxAt(mesh, r), n,
                  UnpackPotential, NULL, in + e.os[r], threaded);
  for (int d = first; d < past; ++d)
    transfer(domains[d].box, realBox, n, CopyLocalPotential,
             domains[d].points, real, threaded);
  mdrtDistPhase(-DistOverlap);
  mdrtDistPhase(DistGather);
  for (int64_t d = first; d < past; ++d) {
    int64_t num = rows->data[rows->offset + d * rows->stride];
#pragma omp parallel for schedule(static) if (threaded)
    for (int64_t r = 0; r < num; ++r) {
      const struct Particle *p = domains[d].atoms + r;
      double grad[3] = {0};
      if (p->q) {
        struct Window w = particleWindow(domains + d, r, order);
        if (mixed)
          gatherMixed(&w, domains[d].box, order, domains[d].points, grad);
        else
          gather(&w, domains[d].box, order, domains[d].points, grad);
      }
      for (int a = 0; a < 3; ++a) {
        double v = precision((-p->q * (n[a] / box[a])) * grad[a], mixed);
        field(f, d, r)[a * f->s[2]] =
            fixed ? mdrtDistWord(v, 0x1p40, DistPairForce) * 0x1p-40 : v;
      }
    }
    free(domains[d].atoms);
    free(domains[d].weights);
    free(domains[d].derivatives);
    free(domains[d].points);
  }
  mdrtDistPhase(-DistGather);
  free(domains);
  free(send);
  free(recv);
  free(fft);
  free(real);
  free(e.domains);
  free(e.counts);
  mdrtPmeFreePencils(mesh);
}

void _mlir_ciface_mdrtDistPME(struct Rows *rows, int64_t first, int64_t past,
                              struct Field *x, struct Field *q, struct Table *t,
                              struct Field *f, struct Results *result,
                              double lx, double ly, double lz, double beta,
                              double coulomb, int64_t nx, int64_t ny,
                              int64_t nz, int64_t order, int64_t px, int64_t py,
                              int64_t mpi, int64_t fixed, int64_t mixed,
                              int64_t threaded) {
  _mlir_ciface_mdrtDistSpatialPME(rows, first, past, x, q, t, f, result, lx, ly, lz,
      beta, coulomb, nx, ny, nz, order, px, py, mpi, fixed, mixed, threaded, 0, 0, 0);
}
