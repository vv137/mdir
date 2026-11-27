/* A separable pencil FFT for an orthorhombic charge mesh. A transpose
   moves the intersections of two partitions. Keeping global x,y,z order
   inside every message makes packing and unpacking exact inverses. */
#include "mdrt_pencils.h"
#include "mdrt_dist_stats.h"
extern void mdrtDistPhase(int64_t);
#include "pocketfft.h"
#ifdef MDIR_ENABLE_MPI
#include "mdrt_mpi.h"
#endif
#include <limits.h>
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct PmePencils {
  int n[3], px, py, rank, peers, mixed;
  rfft_plan rz;
  cfft_plan cy, cx;
  double *a, *b;
  void *send, *recv;
  int *schedule;
};

static void fail(const char *message) {
#ifdef MDIR_ENABLE_MPI
  if (mdrtMPIActive())
    mdrtMPIAbort(message);
#endif
  fprintf(stderr, "distributed PME: %s\n", message);
  abort();
}

static void *allocate(size_t n, size_t width) {
  void *p = calloc(n ? n : 1, width);
  if (!p)
    fail("insufficient memory for FFT pencils");
  return p;
}

/* Only an idle set is cached. Concurrent callers own separate sets,
   and a grid change evicts the old plans when the new set is returned. */
struct PlanSet {
  int n[3];
  rfft_plan rz;
  cfft_plan cy, cx;
};
static struct PlanSet idle;
static pthread_mutex_t planLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t planCleanup = PTHREAD_ONCE_INIT;

static void clearIdle(void) {
  if (idle.rz) {
    destroy_rfft_plan(idle.rz);
    destroy_cfft_plan(idle.cy);
    destroy_cfft_plan(idle.cx);
    memset(&idle,0,sizeof idle);
  }
}

static void registerPlanCleanup(void) { atexit(clearIdle); }

static void acquirePlans(struct PmePencils *p) {
  pthread_once(&planCleanup,registerPlanCleanup);
  pthread_mutex_lock(&planLock);
  if (idle.rz && !memcmp(idle.n,p->n,sizeof idle.n)) {
    p->rz = idle.rz;
    p->cy = idle.cy;
    p->cx = idle.cx;
    memset(&idle,0,sizeof idle);
  } else {
    p->rz = make_rfft_plan(p->n[2]);
    p->cy = make_cfft_plan(p->n[1]);
    p->cx = make_cfft_plan(p->n[0]);
  }
  pthread_mutex_unlock(&planLock);
  if (!p->rz || !p->cy || !p->cx) fail("cannot create pocketfft pencil plans");
}

static void cut(struct PmeBox *b, int axis, int parts, int part) {
  int n = b->hi[axis];
  b->lo[axis] = (int64_t)n * part / parts;
  b->hi[axis] = (int64_t)n * (part + 1) / parts;
}

/* z: [x/P,y/Q,z]; y: [x/P,y,h/Q]; x: [x,y/P,h/Q]. */
static struct PmeBox bounds(const struct PmePencils *p, int axis, int rank) {
  struct PmeBox b = {{0, 0, 0}, {0, 0, 0}};
  if (rank >= p->px * p->py)
    return b;
  memcpy(b.hi, p->n, sizeof b.hi);
  b.hi[2] = p->n[2] / 2 + 1;
  cut(&b, axis == 0 ? 1 : 0, p->px, rank / p->py);
  cut(&b, axis == 2 ? 1 : 2, p->py, rank % p->py);
  return b;
}

struct PmeBox mdrtPmeRealBoxAt(const struct PmePencils *p, int rank) {
  struct PmeBox b = bounds(p, 2, rank);
  if (rank < p->px * p->py)
    b.hi[2] = p->n[2];
  return b;
}

struct PmeBox mdrtPmeRealBox(const struct PmePencils *p) {
  return mdrtPmeRealBoxAt(p, p->rank);
}

struct PmeBox mdrtPmeSpectrumBox(const struct PmePencils *p) {
  return bounds(p, 0, p->rank);
}

static int volume(struct PmeBox b) {
  int v = 1;
  for (int a = 0; a < 3; ++a) {
    int len = b.hi[a] - b.lo[a];
    if (len && v > INT_MAX / 2 / len)
      fail("FFT pencil exceeds MPI count range");
    v *= len;
  }
  return v;
}

static size_t indexOf(struct PmeBox b, int x, int y, int z) {
  return ((size_t)(x - b.lo[0]) * (b.hi[1] - b.lo[1]) + y - b.lo[1]) *
             (b.hi[2] - b.lo[2]) +
         z - b.lo[2];
}

static struct PmeBox intersect(struct PmeBox b, struct PmeBox c) {
  for (int a = 0; a < 3; ++a) {
    if (c.lo[a] > b.lo[a])
      b.lo[a] = c.lo[a];
    if (c.hi[a] < b.hi[a])
      b.hi[a] = c.hi[a];
    if (b.hi[a] < b.lo[a])
      b.hi[a] = b.lo[a];
  }
  return b;
}

static void transfer(struct PmeBox b, struct PmeBox part, double *field,
                     void *buffer, int offset, int mixed, int pack) {
  int width = 2 * (part.hi[2] - part.lo[2]);
  if (!width)
    return;
  for (int x = part.lo[0]; x < part.hi[0]; ++x)
    for (int y = part.lo[1]; y < part.hi[1]; ++y) {
      double *row = field + 2 * indexOf(b, x, y, part.lo[2]);
      if (mixed) {
        float *packet = (float *)buffer + offset;
        if (pack)
          for (int k = 0; k < width; ++k)
            packet[k] = (float)row[k];
        else
          for (int k = 0; k < width; ++k)
            row[k] = packet[k];
      } else {
        double *packet = (double *)buffer + offset;
        if (pack)
          memcpy(packet, row, width * sizeof(double));
        else
          memcpy(row, packet, width * sizeof(double));
      }
      offset += width;
    }
}

static void transpose(struct PmePencils *p, int from, int to, double *src,
                      double *dst) {
  struct PmeBox old = bounds(p, from, p->rank), next = bounds(p, to, p->rank);
  if (p->peers == 1) {
    memcpy(dst, src, 2 * volume(old) * sizeof(double));
    return;
  }
#ifdef MDIR_ENABLE_MPI
  mdrtDistPhase(DistTranspose);
  int *ns = p->schedule, *nr = ns + p->peers;
  int *os = nr + p->peers, *or = os + p->peers;
  int sent = 0, received = 0;
  for (int r = 0; r < p->peers; ++r) {
    struct PmeBox out = intersect(old, bounds(p, to, r));
    struct PmeBox in = intersect(next, bounds(p, from, r));
    ns[r] = 2 * volume(out);
    nr[r] = 2 * volume(in);
    os[r] = sent;
    or[r] = received;
    transfer(old, out, src, p->send, sent, p->mixed, 1);
    sent += ns[r];
    received += nr[r];
  }
  size_t width = p->mixed ? sizeof(float) : sizeof(double);
  mdrtDistMeshBytes(DistTransposeBytes, (int64_t)(sent - ns[p->rank]) * width);
  if (p->mixed)
    mdrtMPIExchangeFloats(p->send, ns, os, p->recv, nr, or);
  else
    mdrtMPIExchange(p->send, ns, os, p->recv, nr, or, 0);
  for (int r = 0; r < p->peers; ++r)
    transfer(next, intersect(next, bounds(p, from, r)), dst, p->recv, or[r],
             p->mixed, 0);
  mdrtDistPhase(-DistTranspose);
#endif
}

struct PmePencils *mdrtPmePencils(int nx, int ny, int nz, int px, int py,
                                  int order, int mpi, int mixed) {
  struct PmePencils *p = allocate(1, sizeof *p);
  p->peers = 1;
  p->mixed = mixed;
#ifdef MDIR_ENABLE_MPI
  if (mpi) {
    p->rank = mdrtMPIRank();
    p->peers = mdrtMPISize();
  }
#else
  if (mpi)
    fail("MPI transport unavailable");
#endif
  if (px < 1 || py < 1 || px > p->peers / py)
    fail("PME decomposition exceeds the transport rank count");
  if (order < 2 || nx / px < order - 1 || ny / py < order - 1 || nz < order)
    fail("real-space PME pencil is smaller than the assignment overlap");
  if (ny < px || nz / 2 + 1 < py)
    fail("empty transposed PME pencil");
  p->n[0] = nx;
  p->n[1] = ny;
  p->n[2] = nz;
  p->px = px;
  p->py = py;
  int largest = 0;
  for (int a = 0; a < 3; ++a) {
    int points = volume(bounds(p, a, p->rank));
    if (points > largest)
      largest = points;
  }
  p->a = allocate(2 * (size_t)largest, sizeof(double));
  p->b = allocate(2 * (size_t)largest, sizeof(double));
  size_t wireWidth = mixed ? sizeof(float) : sizeof(double);
  p->send = allocate(2 * (size_t)largest, wireWidth);
  p->recv = allocate(2 * (size_t)largest, wireWidth);
  p->schedule = allocate(p->peers, 4 * sizeof(int));
  acquirePlans(p);
  return p;
}

/* pocketfft performs a line in binary64. Ordinary CPU PME rounds on storing
   each axis into its grid, including the real inverse. */
static void roundLine(const struct PmePencils *p, double *line, int count) {
  if (p->mixed)
    for (int i = 0; i < count; ++i)
      line[i] = (float)line[i];
}

static void complexLines(struct PmePencils *p, int axis, double *data,
                         int forward) {
  struct PmeBox b = bounds(p, axis, p->rank);
  if (!volume(b))
    return;
  int plane = 1 - axis, len = p->n[axis];
  #pragma omp parallel if(volume(b) >= 4096)
  {
    double *line = allocate(len,2*sizeof *line);
    #pragma omp for collapse(2) schedule(static)
    for (int j = b.lo[plane]; j < b.hi[plane]; ++j)
      for (int z = b.lo[2]; z < b.hi[2]; ++z) {
        size_t start = 2 * indexOf(b, axis == 0 ? 0 : j, axis == 0 ? j : 0, z);
        size_t stride = 2 * (size_t)(b.hi[2] - b.lo[2]);
        if (axis == 0)
          stride *= b.hi[1] - b.lo[1];
        for (int k = 0; k < len; ++k)
          memcpy(line + 2 * k, data + start + k * stride, 2 * sizeof(double));
        cfft_plan plan = axis == 0 ? p->cx : p->cy;
        if (forward ? cfft_forward(plan, line, 1)
                    : cfft_backward(plan, line, 1))
          fail("pocketfft complex transform failed");
        roundLine(p,line, 2 * len);
        for (int k = 0; k < len; ++k)
          memcpy(data + start + k * stride, line + 2 * k, 2 * sizeof(double));
      }
    free(line);
  }
}

void mdrtPmeForward(struct PmePencils *p, const double *real,
                    double *spectrum) {
  int n = p->n[2], h = n / 2 + 1, rows = volume(bounds(p, 2, p->rank)) / h;
  #pragma omp parallel if((size_t)rows*n >= 4096)
  {
    double *line = allocate(n,sizeof *line);
    #pragma omp for schedule(static)
    for (int row = 0; row < rows; ++row) {
      memcpy(line, real + (size_t)row * n, n * sizeof(double));
      if (rfft_forward(p->rz, line, 1))
        fail("pocketfft real transform failed");
      roundLine(p,line, n);
      double *out = p->a + 2 * (size_t)row * h;
      out[0] = line[0];
      out[1] = 0;
      for (int z = 1; z < h; ++z) {
        out[2 * z] = line[2 * z - 1];
        out[2 * z + 1] = 2 * z < n ? line[2 * z] : 0;
      }
    }
    free(line);
  }
  transpose(p, 2, 1, p->a, p->b);
  complexLines(p, 1, p->b, 1);
  transpose(p, 1, 0, p->b, spectrum);
  complexLines(p, 0, spectrum, 1);
}

void mdrtPmeBackward(struct PmePencils *p, const double *spectrum,
                     double *real) {
  memcpy(p->a, spectrum, 2 * volume(bounds(p, 0, p->rank)) * sizeof(double));
  /* The ordinary mixed inverse rounds after y, then x, then z.  Matching
     that order requires one redistribution before the first inverse line. */
  int first = p->mixed ? 1 : 0, second = 1 - first;
  double *src = p->a, *dst = p->b;
  if (p->mixed) {
    transpose(p, 0, 1, src, dst);
    src = p->b;
    dst = p->a;
  }
  complexLines(p, first, src, 0);
  transpose(p, first, second, src, dst);
  complexLines(p, second, dst, 0);
  transpose(p, second, 2, dst, src);
  int n = p->n[2], h = n / 2 + 1, rows = volume(bounds(p, 2, p->rank)) / h;
  #pragma omp parallel if((size_t)rows*n >= 4096)
  {
    double *line = allocate(n,sizeof *line);
    #pragma omp for schedule(static)
    for (int row = 0; row < rows; ++row) {
      const double *in = src + 2 * (size_t)row * h;
      line[0] = in[0];
      for (int z = 1; z < h; ++z) {
        line[2 * z - 1] = in[2 * z];
        if (2 * z < n)
          line[2 * z] = in[2 * z + 1];
      }
      if (rfft_backward(p->rz, line, 1))
        fail("pocketfft real inverse failed");
      roundLine(p,line, n);
      memcpy(real + (size_t)row * n, line, n * sizeof(double));
    }
    free(line);
  }
}

void mdrtPmeFreePencils(struct PmePencils *p) {
  pthread_mutex_lock(&planLock);
  clearIdle();
  memcpy(idle.n,p->n,sizeof idle.n);
  idle.rz = p->rz;
  idle.cy = p->cy;
  idle.cx = p->cx;
  pthread_mutex_unlock(&planLock);
  free(p->a);
  free(p->b);
  free(p->send);
  free(p->recv);
  free(p->schedule);
  free(p);
}
