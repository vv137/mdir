/* Distributed electrostatic PME [Essmann1995, Liang2026]. Each rank stores
   the union of the assignment stencils of its owned particles, including
   the order-1 overlap. Indexing this subgrid sparsely also covers particles
   owned with a constraint group on the far side of a periodic face. */
#include "../include/mdir/Driver/ProlateSpectrum.h"
#include "mdrt_dist_stats.h"
#include "mdrt_pencils.h"
#ifdef MDIR_ENABLE_MPI
#include "mdrt_mpi.h"
#endif
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void mdrtDistPhase(int64_t);
extern int64_t mdrtDistWord(double, double);
extern int64_t mdrtDistNarrowWord(__int128);
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
  int first[3];
  double w[3][8], dw[3][8];
};
struct Cell {
  int64_t key;
  __int128 charge;
  double potential;
};
struct Subgrid {
  struct Cell *cells;
  size_t count, capacity;
};
struct Packet {
  int64_t key;
  __int128 charge;
};
struct Exchange {
  int peers, *ns, *nr, *os, *or;
  int sent, received;
  struct Packet *send, *recv;
  struct Cell **source;
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
static void spline(double t, int order, double *w, double *dw) {
  w[0] = 1 - t;
  w[1] = t;
  for (int k = 3; k <= order; ++k) {
    if (k == order) {
      dw[0] = -w[0];
      for (int j = 1; j < order; ++j)
        dw[j] = w[j - 1] - w[j];
    }
    double inv = 1.0 / (k - 1);
    w[k - 1] = inv * t * w[k - 2];
    for (int j = 1; j < k - 1; ++j) {
      int i = k - 1 - j;
      w[i] = inv * ((t + j) * w[i - 1] + (k - j - t) * w[i]);
    }
    w[0] *= inv * (1 - t);
  }
}
static void window(struct Window *w, const double *x, int stride,
                   const double box[3], const int n[3], int order,
                   struct Table *t) {
  for (int a = 0; a < 3; ++a) {
    double s = x[a * stride] / box[a], u = (s - floor(s)) * n[a];
    if (t->n[0] == 5) {
      u += n[a] - (order % 2) * 0.5;
      double f = floor(u), v = 2 * (u - f) - 1;
      w->first[a] = ((int)f - order / 2 + 1) % n[a];
      int terms = (int)table(t, 4)[0];
      for (int k = 0; k < order; ++k) {
        w->w[a][k] = series(table(t, 3) + k * terms, terms, v);
        w->dw[a][k] = series(table(t, 3) + (order + k) * terms, terms, v);
      }
    } else {
      double f = floor(u);
      w->first[a] = ((int)f + n[a] - order + 1) % n[a];
      spline(u - f, order, w->w[a], w->dw[a]);
    }
  }
}
static struct Cell *lookup(struct Subgrid *g, int64_t key) {
  uint64_t h = (uint64_t)key * UINT64_C(11400714819323198485);
  size_t i = (h ^ (h >> 32)) & (g->capacity - 1);
  while (g->cells[i].key && g->cells[i].key != key)
    i = (i + 1) & (g->capacity - 1);
  return g->cells + i;
}
static void grow(struct Subgrid *g) {
  size_t old = g->capacity;
  struct Cell *cells = g->cells;
  g->capacity = old ? old * 2 : 256;
  g->cells = alloc(g->capacity, sizeof *cells);
  for (size_t i = 0; i < old; ++i)
    if (cells[i].key)
      *lookup(g, cells[i].key) = cells[i];
  free(cells);
}
static void stencil(struct Subgrid *g, const struct Window *w, const int n[3],
                    int order, double q, double gradient[3]) {
  for (int j = 0; j < order; ++j) {
    int x = (w->first[0] + j) % n[0];
    for (int k = 0; k < order; ++k) {
      int y = (w->first[1] + k) % n[1];
      int64_t row = ((int64_t)x * n[1] + y) * n[2];
      double xy = w->w[0][j] * w->w[1][k];
      for (int l = 0; l < order; ++l) {
        int z = (w->first[2] + l) % n[2];
        int64_t key = row + z + 1;
        if (!gradient && g->count * 2 >= g->capacity)
          grow(g);
        struct Cell *c = lookup(g, key);
        if (gradient) {
          double v = c->potential;
          gradient[0] += v * w->dw[0][j] * w->w[1][k] * w->w[2][l];
          gradient[1] += v * w->w[0][j] * w->dw[1][k] * w->w[2][l];
          gradient[2] += v * xy * w->dw[2][l];
        } else {
          if (!c->key) {
            c->key = key;
            ++g->count;
          }
          double word =
              nearbyint(((q * 0x1p40 * w->w[0][j]) * w->w[1][k]) * w->w[2][l]);
          if (!(word >= -0x1p63 && word < 0x1p63))
            fail("PME assignment outside signed Q40 range");
          c->charge += (int64_t)word;
        }
      }
    }
  }
}
static int owner(int64_t key, const int n[3], int px, int py) {
  int64_t row = (key - 1) / n[2];
  int64_t x = row / n[1], y = row % n[1];
  return ((x + 1) * px - 1) / n[0] * py + ((y + 1) * py - 1) / n[1];
}
static int prefix(int *counts, int *offsets, int size) {
  int total = 0;
  for (int r = 0; r < size; ++r) {
    if (counts[r] > INT_MAX - total)
      fail("PME overlap exceeds MPI count range");
    offsets[r] = total;
    total += counts[r];
  }
  return total;
}
static struct Exchange exchange(struct Subgrid *g, const int n[3], int px,
                                int py, int peers) {
  struct Exchange e = {.peers = peers};
  e.ns = alloc(peers, 4 * sizeof(int));
  e.nr = e.ns + peers;
  e.os = e.nr + peers;
  e.or = e.os + peers;
  if (g->count > INT_MAX)
    fail("PME overlap exceeds MPI count range");
  for (size_t i = 0; i < g->capacity; ++i)
    if (g->cells[i].key)
      ++e.ns[owner(g->cells[i].key, n, px, py)];
  if (peers == 1)
    e.nr[0] = e.ns[0];
#ifdef MDIR_ENABLE_MPI
  else
    mdrtMPIExchangeCounts(e.ns, e.nr);
#endif
  e.sent = prefix(e.ns, e.os, peers);
  e.received = prefix(e.nr, e.or, peers);
  e.send = alloc(e.sent, sizeof *e.send);
  e.recv = alloc(e.received, sizeof *e.recv);
  e.source = alloc(e.sent, sizeof *e.source);
  int *next = alloc(peers, sizeof(int));
  memcpy(next, e.os, peers * sizeof(int));
  for (size_t i = 0; i < g->capacity; ++i) {
    struct Cell *c = g->cells + i;
    if (!c->key)
      continue;
    int at = next[owner(c->key, n, px, py)]++;
    e.send[at] = (struct Packet){c->key, c->charge};
    e.source[at] = c;
  }
  free(next);
  if (peers == 1)
    memcpy(e.recv, e.send, e.sent * sizeof *e.recv);
#ifdef MDIR_ENABLE_MPI
  else
    mdrtMPIExchangeRecords(e.send, e.ns, e.os, e.recv, e.nr, e.or,
                           sizeof *e.send);
#endif
  return e;
}
static size_t boxSize(struct PmeBox b) {
  return (size_t)(b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) *
         (b.hi[2] - b.lo[2]);
}
static size_t localIndex(struct PmeBox b, int64_t key, const int n[3]) {
  int64_t row = (key - 1) / n[2];
  return ((row / n[1] - b.lo[0]) * (b.hi[1] - b.lo[1]) + row % n[1] - b.lo[1]) *
             n[2] +
         (key - 1) % n[2];
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
static void spectrum(const int n[3], const double box[3], struct PmeBox b,
                     double *fft, struct Table *t, double beta, double coulomb,
                     int fixed, double *sums, __int128 words[10]) {
  const double pi = acos(-1),
               pref = coulomb / (pi * (box[0] * box[1] * box[2]));
  size_t i = 0;
  for (int x = b.lo[0]; x < b.hi[0]; ++x)
    for (int y = b.lo[1]; y < b.hi[1]; ++y)
      for (int z = b.lo[2]; z < b.hi[2]; ++z, i += 2) {
        double m[3] = {(x > n[0] / 2 ? x - n[0] : x) / box[0],
                       (y > n[1] / 2 ? y - n[1] : y) / box[1], z / box[2]};
        double s = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
        double g = 0, dg = 0;
        if (s > 0) {
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
            double a = pi * pi / (beta * beta);
            g = scale * exp(-a * s);
            dg = -2 * g * (1 / s + a);
          }
        }
        double re = fft[i], im = fft[i + 1];
        double power =
            (z == 0 || 2 * z == n[2] ? 0.5 : 1) * (re * re + im * im);
        fft[i] = re * g;
        fft[i + 1] = im * g;
        double term[10] = {power * g};
        for (int a = 0; a < 3; ++a)
          for (int d = 0; d < 3; ++d)
            term[1 + 3 * a + d] = power * ((a == d ? g : 0) + dg * m[a] * m[d]);
        for (int k = 0; k < 10; ++k) {
          if (fixed)
            words[k] += mdrtDistWord(term[k], 0x1p32);
          else
            sums[k] += term[k];
        }
      }
}
void _mlir_ciface_mdrtDistPME(struct Rows *rows, int64_t first, int64_t past,
                              struct Field *x, struct Field *q, struct Table *t,
                              struct Field *f, struct Results *result,
                              double lx, double ly, double lz, double beta,
                              double coulomb, int64_t nx, int64_t ny,
                              int64_t nz, int64_t order, int64_t px, int64_t py,
                              int64_t mpi, int64_t fixed) {
  if (nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
      nx * ny > INT64_MAX / nz || order < 3 || order > 8)
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
  struct PmePencils *mesh = mdrtPmePencils(nx, ny, nz, px, py, order, mpi);
  struct PmeBox realBox = mdrtPmeRealBox(mesh),
                modes = mdrtPmeSpectrumBox(mesh);
  struct Subgrid local = {0};
  mdrtDistPhase(DistSpread);
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < rows->data[rows->offset + d * rows->stride]; ++r) {
      double charge = *field(q, d, r);
      if (!charge)
        continue;
      struct Window w = {0};
      window(&w, field(x, d, r), x->s[2], box, n, order, t);
      stencil(&local, &w, n, order, charge, NULL);
    }
  mdrtDistPhase(-DistSpread);
  mdrtDistPhase(DistOverlap);
  struct Exchange e = exchange(&local, n, px, py, peers);
  size_t size = boxSize(realBox);
  __int128 *grid = alloc(size, sizeof *grid);
  size_t *indices = alloc(e.received, sizeof *indices);
  for (int i = 0; i < e.received; ++i) {
    indices[i] = localIndex(realBox, e.recv[i].key, n);
    grid[indices[i]] += e.recv[i].charge;
  }
  double *real = alloc(size, sizeof *real);
  for (size_t i = 0; i < size; ++i)
    real[i] = mdrtDistNarrowWord(grid[i]) * 0x1p-40;
  free(grid);
  free(e.send);
  free(e.recv);
  mdrtDistPhase(-DistOverlap);
  double *fft = alloc(boxSize(modes), 2 * sizeof(double));
  mdrtDistPhase(DistFFT);
  mdrtPmeForward(mesh, real, fft);
  mdrtDistPhase(-DistFFT);
  mdrtDistPhase(DistSolve);
  double sums[10] = {0};
  __int128 words[10] = {0};
  spectrum(n, box, modes, fft, t, beta, coulomb, fixed, sums, words);
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
        fixed ? mdrtDistNarrowWord(words[k]) * 0x1p-32 : sums[k];
  mdrtDistPhase(-DistSolve);
  mdrtDistPhase(DistFFT);
  mdrtPmeBackward(mesh, fft, real);
  mdrtDistPhase(-DistFFT);
  mdrtDistPhase(DistOverlap);
  double *out = alloc(e.received, sizeof(double)),
         *in = alloc(e.sent, sizeof(double));
  for (int i = 0; i < e.received; ++i)
    out[i] = real[indices[i]];
  if (peers == 1)
    memcpy(in, out, e.sent * sizeof(double));
#ifdef MDIR_ENABLE_MPI
  else
    mdrtMPIExchange(out, e.nr, e.or, in, e.ns, e.os, 0);
#endif
  for (int i = 0; i < e.sent; ++i)
    e.source[i]->potential = in[i];
  mdrtDistPhase(-DistOverlap);
  mdrtDistPhase(DistGather);
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < rows->data[rows->offset + d * rows->stride]; ++r) {
      double grad[3] = {0}, charge = *field(q, d, r);
      if (charge) {
        struct Window w = {0};
        window(&w, field(x, d, r), x->s[2], box, n, order, t);
        stencil(&local, &w, n, order, charge, grad);
      }
      for (int a = 0; a < 3; ++a) {
        double v = -charge * grad[a] * n[a] / box[a];
        field(f, d, r)[a * f->s[2]] =
            fixed ? mdrtDistWord(v, 0x1p40) * 0x1p-40 : v;
      }
    }
  mdrtDistPhase(-DistGather);
  free(out);
  free(in);
  free(indices);
  free(fft);
  free(real);
  free(e.source);
  free(e.ns);
  free(local.cells);
  mdrtPmeFreePencils(mesh);
}
