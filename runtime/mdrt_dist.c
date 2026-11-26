/* The in-process transport of md_dist (D275, docs/distributed-ml2.md,
   Sections 6 and 9): every domain in one process, its rows in buffers of the
   compiled program, and a transfer a copy between them.

   A layout and a halo are handles to the structures below. The buffers of
   distributed fields are memref<D x rows x k x f64>: domain, row,
   component. Every loop runs in a fixed order (domains, then particles in
   the order of the set, then images), so a run gives the same rows and the
   same sums to the bit for a given grid.

   MDRT_DIST_TRACE set to a nonempty value prints the owned and ghost rows
   of each domain when a halo is built, and the rows each transfer moves:
   the schedule as it runs. */

#include "mdrt_dist_stats.h"
#ifdef MDIR_ENABLE_NCCL
#include "mdrt_nccl.h"
#endif
#include "mdrt_host_matrix.h"
#ifdef MDIR_ENABLE_MPI
#include "mdrt_mpi.h"
#endif
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Memref1I {
  int64_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
struct Memref2I {
  int64_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct Memref1D {
  double *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
struct Memref2D {
  double *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct Memref3D {
  double *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};

static double *at2(struct Memref2D *m, int64_t i, int64_t j) {
  return m->aligned + m->offset + i * m->strides[0] + j * m->strides[1];
}
static double *at3(struct Memref3D *m, int64_t d, int64_t r, int64_t c) {
  return m->aligned + m->offset + d * m->strides[0] + r * m->strides[1] +
         c * m->strides[2];
}

static _Thread_local struct DistObserver *observer;

void mdrtDistObserve(struct DistObserver *value) { observer = value; }

int mdrtDistObserving(void) { return observer != NULL; }

void mdrtDistPhase(int64_t phase) {
  if (observer)
    observer->phase(observer->context, phase);
}

static int tracing(void) {
  const char *value = getenv("MDRT_DIST_TRACE");
  return value && *value;
}

static void fail(const char *message) {
#ifdef MDIR_ENABLE_MPI
  if (mdrtMPIActive()) mdrtMPIAbort(message);
#endif
  fprintf(stderr, "mdrt_dist: %s\n", message);
  fflush(stderr);
  abort();
}

/* Layouts and halos count their references (D399): the
   op that makes one holds the first, a md_dist.state holds one while it
   carries it, and a halo holds one of its layout. */
struct Layout {
  int mpi;
  int64_t references;
  int64_t grid[3], domains, count, stride;
  double edges[3];
  struct ConstraintGroup {
    int32_t atoms[4];
    int size;
  } *groups;
  size_t groupCount;
  int groupsFrozen;
  double (*migratedPositions)[3];
  int32_t *groupCenters; /* center + 1; zero for a free atom */
#ifdef MDIR_ENABLE_MPI
  int64_t *migrantOffsets, *migrantIds; /* Old rank, then particle identity. */
#endif
#ifdef MDIR_ENABLE_NCCL
  struct TuplePlan *tuplePlans;
  struct Layout *identityOwner; /* Borrow immutable numberings across stationary epochs. */
  uint32_t *faceMasks;
  unsigned *localFaceMasks; /* Device snapshot in this epoch's owned order. */
  int selfFaces, pairFaces;
  double faceReach;
  int *deviceMigrationMap;
  int64_t incomingRows;
#endif
  int64_t *owner; /* [count]: the domain that owns each particle */
  int64_t *row;   /* [count]: its owned row there */
  int64_t *rows;  /* [domains]: the number of owned rows */
  int64_t *ids;   /* [domains * stride]: the particle of each owned row */
};

struct DomainMatrix {
  struct DomainMatrix *next;
  int64_t domain, centers, rows, raw, selected;
  double cutoff, skin, box[3];
  double *reference, *shifts;
  int32_t *counts, *selectedCounts;
  int valid;
};

#ifdef MDIR_ENABLE_MPI
struct TransferRow { int64_t domain, ghost; int peer; };
struct TransferPlan {
  int *counts;
  struct TransferRow *send, *receive, *local;
  size_t sends, receives, locals;
};
#endif

struct Halo {
  int64_t references;
  struct Layout *layout;
  double reach;
  int64_t *firstGhost, *nextGhost;
  struct DomainMatrix *matrices;
#ifdef MDIR_ENABLE_MPI
  struct TransferPlan plans[2];
#endif
#ifdef MDIR_ENABLE_NCCL
  struct TuplePlan *tuplePlans;
  double *tupleCoordinates;
  struct Memref3D tupleInput;
  struct NCCLPlan *ncclPlan;
  struct DistDeviceMatrix deviceMatrix;
  double deviceCutoff;
  struct ScalarHaloField *scalarFields;
#endif
  int64_t capacity; /* the most ghosts of one domain */
  int64_t *rows;    /* [domains] */
  /* [domains * capacity] each: the source's owner, its owned row, its
     identity, and the image n (3 per ghost). */
  struct Edge { int64_t from, to, rows; } *edges;
  size_t edgeCount;
  int64_t *sourceDomain, *sourceRow, *ids, *image;
};

/* Input tuple members use particle-set indices even after a reorder. The
   symmetric key removes both orientations from the cutoff relation. */
struct Exclusions {
  size_t count;
  uint64_t pairs[];
};

struct Members {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};

static int compareKeys(const void *lhs, const void *rhs) {
  uint64_t a = *(const uint64_t *)lhs, b = *(const uint64_t *)rhs;
  return (a > b) - (a < b);
}

int64_t _mlir_ciface_mdrtDistExclusions(struct Members *members) {
  size_t count = members->sizes[0];
  if (count > (SIZE_MAX - sizeof(struct Exclusions)) / sizeof(uint64_t))
    fail("exclusion table too large");
  struct Exclusions *e = malloc(sizeof *e + count * sizeof *e->pairs);
  if (!e) fail("out of memory building exclusions");
  e->count = count;
  for (size_t r = 0; r < count; ++r) {
    const int32_t *row = members->aligned + members->offset + r * members->strides[0];
    uint32_t i = row[0], j = row[members->strides[1]];
    e->pairs[r] = i < j ? ((uint64_t)i << 32) | j : ((uint64_t)j << 32) | i;
  }
  qsort(e->pairs, count, sizeof *e->pairs, compareKeys);
  return (int64_t)(intptr_t)e;
}

/* Map both ends of each excluded pair to the packed owned/image rows.
   An identity can have several images in a thin domain. */
void _mlir_ciface_mdrtDistLocalExclusions(struct Members *result, int64_t halo,
                                           int64_t domain, int64_t exclusions) {
  const struct Halo *h = (const struct Halo *)(intptr_t)halo;
  const struct Layout *l = h->layout;
  const struct Exclusions *e = (const struct Exclusions *)(intptr_t)exclusions;
  int64_t owned = l->rows[domain], rows = owned + h->rows[domain];
  if (rows > INT32_MAX) fail("too many local MATRIX rows");
  int32_t *head = malloc((l->count + 1) * sizeof(*head));
  int32_t *next = malloc((rows + 1) * sizeof(*next));
  if (!head || !next) fail("out of memory indexing domain exclusions");
  for (int64_t i = 0; i < l->count; ++i) head[i] = -1;
  for (int64_t i = 0; i < rows; ++i) {
    int64_t id = i < owned ? l->ids[domain*l->stride + i] :
                             h->ids[domain*h->capacity + i-owned];
    next[i] = head[id];
    head[id] = (int32_t)i;
  }
  int32_t *pairs = NULL;
  int64_t count = 0;
  for (int pass = 0; pass < 2; ++pass) {
    count = 0;
    for (size_t k = 0; k < e->count; ++k) {
      uint32_t a = e->pairs[k] >> 32, b = (uint32_t)e->pairs[k];
      if (a >= l->count || b >= l->count) fail("invalid excluded particle ID");
      for (int32_t i = head[a]; i >= 0; i = next[i])
        for (int32_t j = head[b]; j >= 0; j = next[j]) {
          if (count == INT32_MAX) fail("too many local excluded pairs");
          if (pass) { pairs[2*count] = i; pairs[2*count+1] = j; }
          ++count;
        }
    }
    if (!pass) {
      pairs = malloc((count ? 2*count : 1) * sizeof(*pairs));
      if (!pairs) fail("out of memory mapping domain exclusions");
    }
  }
  free(next);
  free(head);
  *result = (struct Members){pairs, pairs, 0, {count, 2}, {2, 1}};
}

int64_t mdrtDistPairAllowed(int64_t handle, int64_t i, int64_t j) {
  const struct Exclusions *e = (const struct Exclusions *)(intptr_t)handle;
  uint64_t key = ((uint64_t)i << 32) | (uint32_t)j;
  return !bsearch(&key, e->pairs, e->count, sizeof key, compareKeys);
}

void mdrtDistFreeExclusions(int64_t handle) {
  free((void *)(intptr_t)handle);
}

/* Word conversion is defined even for a singular pair or huge finite input. */
int64_t mdrtDistWord(double x, double scale) {
  double w = round(x * scale);
  if (!(w >= -0x1p63 && w < 0x1p63))
    fail("deterministic fixed-point term outside signed 64-bit range");
  return (int64_t)w;
}

void mdrtDistCheckWordCount(int64_t count) {
  if (count < 0 || count > INT64_C(4294967296))
    fail("deterministic pair count exceeds the signed 128-bit accumulation bound");
}

typedef __int128 DistWide;
/* Bucket the local rows by their current Cartesian image. Each center
   covers three adjacent x bins in each of nine (y,z) slabs. The caller
   still evaluates the original distance, cutoff and identity predicate. */
static void pairCells(int64_t d, int64_t owned, int64_t count, int64_t centers,
    double cutoff, struct Memref3D *xo, struct Memref3D *xg,
    struct Memref3D *shifts, int64_t shifted, struct Memref1I *order,
    struct Memref1I *spans) {
  if (!centers)
    return;
  int64_t *rows = order->aligned + order->offset;
  int64_t *ranges = spans->aligned + spans->offset;
  double lo[3] = {INFINITY, INFINITY, INFINITY};
  double hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  double magnitude = cutoff;
  for (int64_t r = 0; r < count; ++r)
    for (int k = 0; k < 3; ++k) {
      double x = r < owned ? *at3(xo, d, r, k) : *at3(xg, d, r-owned, k);
      double shift = shifted && r >= owned ? *at3(shifts, d, r-owned, k) : 0;
      magnitude = fmax(magnitude, fmax(fabs(x), fabs(shift)));
      x += shift;
      if (!isfinite(x))
        fail("non-finite pair cell coordinate");
      lo[k] = fmin(lo[k], x);
      hi[k] = fmax(hi[k], x);
    }
  /* Include the rounding error of binning x + image when the pair uses
     (center - x) - image. Coarsen unusually sparse boxes to bound storage. */
  double width = cutoff + (32 * DBL_EPSILON) * magnitude;
  int64_t shape[3];
  for (;;) {
    double bins = 1;
    for (int k = 0; k < 3; ++k) {
      double n = floor((hi[k]/width) - (lo[k]/width)) + 1;
      if (n > count) { bins = INFINITY; break; }
      shape[k] = (int64_t)n;
      bins *= n;
    }
    if (bins <= 4.0 * count)
      break;
    width *= 2;
  }
  int64_t nbin = shape[0] * shape[1] * shape[2];
  int64_t *offset = calloc(nbin + 1, sizeof *offset);
  int64_t *cell = malloc(count * sizeof *cell);
  if (!offset || !cell)
    fail("out of memory building pair cells");
  for (int64_t r = 0; r < count; ++r) {
    int64_t bin = 0;
    for (int k = 2; k >= 0; --k) {
      double x = r < owned ? *at3(xo,d,r,k) : *at3(xg,d,r-owned,k);
      if (shifted && r >= owned)
        x += *at3(shifts,d,r-owned,k);
      bin = bin * shape[k] + (int64_t)floor(x/width - lo[k]/width);
    }
    cell[r] = bin;
    ++offset[bin+1];
  }
  for (int64_t b = 1; b <= nbin; ++b)
    offset[b] += offset[b-1];
  /* Fill backward. offset[b+1] then points at the start of bin b. */
  for (int64_t r = count; r-- > 0;)
    rows[--offset[cell[r]+1]] = r;
  for (int64_t i = 0; i < centers; ++i) {
    int64_t x = cell[i] % shape[0], yz = cell[i] / shape[0];
    int64_t y = yz % shape[1], z = yz / shape[1];
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy) {
        int64_t begin = 0, end = 0;
        if (y+dy >= 0 && y+dy < shape[1] && z+dz >= 0 && z+dz < shape[2]) {
          int64_t slab = ((z+dz)*shape[1] + y+dy)*shape[0];
          int64_t first = slab + (x > 0 ? x-1 : x);
          int64_t last = slab + (x+1 < shape[0] ? x+1 : x);
          begin = offset[first+1];
          end = last+1 == nbin ? count : offset[last+2];
        }
        *ranges++ = begin;
        *ranges++ = end;
      }
  }
  free(cell);
  free(offset);
}


void _mlir_ciface_mdrtDistPairCells(int64_t d, int64_t owned, int64_t count,
    double cutoff, struct Memref3D *xo, struct Memref3D *xg,
    struct Memref3D *shifts, int64_t shifted, struct Memref1I *order,
    struct Memref1I *spans) {
  pairCells(d, owned, count, owned, cutoff, xo, xg, shifts, shifted, order, spans);
}

void _mlir_ciface_mdrtDistMidpointCells(int64_t d, int64_t owned, int64_t count,
    double cutoff, struct Memref3D *xo, struct Memref3D *xg,
    struct Memref3D *shifts, int64_t shifted, struct Memref1I *order,
    struct Memref1I *spans) {
  pairCells(d, owned, count, count, cutoff, xo, xg, shifts, shifted, order, spans);
}

DistWide mdrtDistAddWide(DistWide a, DistWide b) {
  DistWide result;
  if (__builtin_add_overflow(a, b, &result))
    fail("deterministic fixed-point sum exceeds signed 128-bit range");
  return result;
}

int64_t mdrtDistNarrowWord(DistWide value) {
  if (value < INT64_MIN || value > INT64_MAX)
    fail("deterministic fixed-point result exceeds signed 64-bit range");
  return (int64_t)value;
}

void _mlir_ciface_mdrtDistImageShifts(int64_t handle, struct Memref3D *shifts) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (!h || !shifts || shifts->sizes[0] != h->layout->domains ||
      shifts->sizes[2] != 3)
    fail("deterministic pair geometry requires a complete halo image field");
  for (int64_t d = 0; d < h->layout->domains; ++d)
    if (shifts->sizes[1] < h->rows[d])
      fail("missing halo rows for deterministic pair geometry");
  for (int64_t d = 0; d < h->layout->domains; ++d)
    for (int64_t g = 0; g < h->rows[d]; ++g)
      for (int c = 0; c < 3; ++c)
        *at3(shifts, d, g, c) =
            (double)h->image[3 * (d * h->capacity + g) + c] * h->layout->edges[c];
}

struct Memref3W {
  DistWide *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};

static DistWide *wordAt(struct Memref3W *m, int64_t d, int64_t r, int64_t c) {
  return m->aligned + m->offset + d*m->strides[0] + r*m->strides[1] + c*m->strides[2];
}

/* The cell of the grid that fractional coordinate `u` along an axis of `g`
   divisions falls in. */
static int64_t division(double u, int64_t g) {
  int64_t a = (int64_t)floor(u * (double)g);
  if (a < 0)
    a = 0;
  if (a >= g)
    a = g - 1;
  return a;
}

/* xi is the untranslated position of the lesser identity. Form its midpoint
   before choosing a grid. Of all common translations of this pair, use
   exactly the one that puts the midpoint in the primary cell. */
int64_t mdrtDistMidpointOwns(int64_t handle, int64_t d, int64_t i,
    double x, double y, double z, double dx, double dy, double dz) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  struct Layout *l = h->layout;
  double midpoint[3] = {x - 0.5*dx, y - 0.5*dy, z - 0.5*dz};
  int64_t owner = 0;
  for (int k = 2; k >= 0; --k) {
    double u = midpoint[k] / l->edges[k];
    double image = floor(u);
    int64_t n = i < l->rows[d] ? 0 :
        h->image[3*(d*h->capacity + i-l->rows[d]) + k];
    if (image != -(double)n)
      return 0;
    owner = owner*l->grid[k] + division(u - image, l->grid[k]);
  }
  return owner == d;
}

/* The rank visits its domains serially. The ordinary compiled builder and
   row kernels parallelize the work inside each visit. The halo fixes row
   identities and images, but current explicit-image positions still have
   to satisfy the Verlet displacement bound. */
int64_t mdrtDistMatrix(int64_t handle, int64_t domain, int64_t centers,
                       int64_t rows, double cutoff, double skin, int64_t width) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (!h || domain < 0 || domain >= h->layout->domains ||
      rows != h->layout->rows[domain] + h->rows[domain] ||
      (centers != h->layout->rows[domain] && centers != rows))
    fail("domain matrix extents do not match its halo");
  struct DomainMatrix *m = h->matrices;
  while (m && !(m->domain == domain && m->centers == centers &&
                m->cutoff == cutoff && m->skin == skin))
    m = m->next;
  if (!m) {
    m = calloc(1, sizeof *m);
    if (!m) fail("out of memory allocating a domain matrix");
    m->next = h->matrices;
    h->matrices = m;
    m->domain = domain;
    m->rows = rows;
    m->centers = centers;
    m->cutoff = cutoff;
    m->skin = skin;
    m->raw = mdrtHostMatrixAllocate(centers, width);
    m->counts = calloc(rows ? rows : 1, sizeof *m->counts);
    m->reference = calloc(rows ? rows : 1, 3*sizeof *m->reference);
    if (!m->counts || !m->reference) fail("out of memory allocating domain rows");
  }
  if (rows != m->rows) fail("local row count changed in a live halo");
  return (int64_t)(intptr_t)m;
}

int64_t _mlir_ciface_mdrtDistMatrixRebuild(int64_t matrix, struct Memref2D *x) {
  struct DomainMatrix *m = (struct DomainMatrix *)(intptr_t)matrix;
  double limit = 0.25*m->skin*m->skin;
  if (m->valid) {
    int64_t i = 0;
    for (; i < m->rows; ++i) {
      double r2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        double dx = *at2(x, i, c) - m->reference[3*i+c];
        r2 += dx*dx;
      }
      if (!(r2 <= limit)) break;
    }
    if (i == m->rows) return 0;
  }
  double lo[3] = {INFINITY, INFINITY, INFINITY};
  double hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (int64_t i = 0; i < m->rows; ++i)
    for (int c = 0; c < 3; ++c) {
      double value = *at2(x, i, c);
      m->reference[3*i+c] = value;
      lo[c] = fmin(lo[c], value);
      hi[c] = fmax(hi[c], value);
    }
  /* The periodic search in this enclosing box agrees with a plain search
     through the build radius. Each explicit halo image occurs only once
     at its own position in the list. */
  for (int c = 0; c < 3; ++c)
    m->box[c] = (m->rows ? hi[c]-lo[c] : 0.0) + 2*(m->cutoff + m->skin);
  m->valid = 1;
  return 1;
}

void _mlir_ciface_mdrtDistMatrixImages(struct Memref2D *r, int64_t handle, int64_t matrix) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  struct DomainMatrix *m = (struct DomainMatrix *)(intptr_t)matrix;
  if (!m->shifts) {
    m->shifts = calloc(m->rows ? m->rows : 1, 3 * sizeof *m->shifts);
    if (!m->shifts) fail("out of memory allocating local image translations");
    int64_t d = m->domain, n = h->layout->rows[d];
    for (int64_t g = 0; g < h->rows[d]; ++g)
      for (int c = 0; c < 3; ++c)
        m->shifts[3*(n+g)+c] = h->image[3*(d*h->capacity+g)+c]*h->layout->edges[c];
  }
  r->allocated = r->aligned = m->shifts;
  r->offset = 0;
  r->sizes[0] = m->rows;
  r->sizes[1] = r->strides[0] = 3;
  r->strides[1] = 1;
}

double mdrtDistMatrixEdge(int64_t matrix, int64_t axis) {
  return ((struct DomainMatrix *)(intptr_t)matrix)->box[axis];
}

int64_t mdrtDistMatrixHandle(int64_t matrix, int64_t selected) {
  struct DomainMatrix *m = (struct DomainMatrix *)(intptr_t)matrix;
  return selected ? m->selected : m->raw;
}

struct RowCounts {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};

void _mlir_ciface_mdrtDistMatrixCounts(struct RowCounts *r, int64_t matrix, int64_t selected) {
  struct DomainMatrix *m = (struct DomainMatrix *)(intptr_t)matrix;
  r->allocated = r->aligned = selected ? m->selectedCounts : m->counts;
  r->offset = 0;
  r->sizes[0] = selected ? m->centers : m->rows;
  r->strides[0] = 1;
}

int64_t mdrtDistSelectMatrix(int64_t matrix, int64_t exclusions, int64_t rule) {
  if (!rule && !exclusions) return 0;
  struct DomainMatrix *m = (struct DomainMatrix *)(intptr_t)matrix;
  struct HostBuffer2 src;
  _mlir_ciface_mdrtHostMatrixEntries(&src, m->raw);
  if (!m->selected) {
    m->selected = mdrtHostMatrixAllocate(m->centers, src.sizes[1]);
    m->selectedCounts = calloc(m->centers ? m->centers : 1, sizeof *m->selectedCounts);
    if (!m->selectedCounts) fail("out of memory allocating filtered row counts");
  }
  mdrtHostMatrixGrow(m->selected, src.sizes[1]);
  return 1;
}


static size_t firstKey(const struct Exclusions *e, uint64_t key) {
  size_t first = 0, end = e->count;
  while (first < end) {
    size_t middle = first + (end-first)/2;
    if (e->pairs[middle] < key) first = middle+1;
    else end = middle;
  }
  return first;
}

void _mlir_ciface_mdrtDistSelectRows(int64_t handle, int64_t matrix,
    int64_t exclusions, int64_t rule, int64_t first, int64_t end,
    struct Memref3D *owned, struct Memref3D *ghost) {
  struct DomainMatrix *m = (struct DomainMatrix *)(intptr_t)matrix;
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  struct Layout *l = h->layout;
  int64_t d = m->domain, n = l->rows[d];
  struct HostBuffer2 src, dst;
  _mlir_ciface_mdrtHostMatrixEntries(&src, m->raw);
  _mlir_ciface_mdrtHostMatrixEntries(&dst, m->selected);
  struct Exclusions *e = (struct Exclusions *)(intptr_t)exclusions;
  for (int64_t i = first; i < end; ++i) {
    int64_t gi = d*h->capacity + i-n;
    int64_t idi = i<n ? l->ids[d*l->stride+i] : h->ids[gi];
    uint64_t prefix = (uint64_t)idi << 32;
    size_t lo = 0, hi = 0;
    if (e && rule) {
      lo = firstKey(e, prefix);
      hi = firstKey(e, prefix + (UINT64_C(1)<<32));
    }
    int32_t count = 0;
    for (int32_t k = 0; k < m->counts[i]; ++k) {
      int32_t j = ((int32_t *)src.aligned)[i*src.strides[0]+k];
      int64_t gj = d*h->capacity + j-n;
      int64_t idj = j<n ? l->ids[d*l->stride+j] : h->ids[gj];
      if (rule && idi >= idj) continue;
      if (e && rule) {
        uint64_t key = prefix | (uint32_t)idj;
        size_t t = lo;
        while (t < hi && e->pairs[t] < key) ++t;
        if (t < hi && e->pairs[t] == key) continue;
      }
      if (e && !rule && !mdrtDistPairAllowed(exclusions, idi < idj ? idi : idj,
                                             idi < idj ? idj : idi)) continue;
      if (rule == 2) {
        double xi[3], dx[3];
        for (int c = 0; c < 3; ++c) {
          xi[c] = i<n ? *at3(owned,d,i,c) : *at3(ghost,d,i-n,c);
          double xj = j<n ? *at3(owned,d,j,c) : *at3(ghost,d,j-n,c);
          double a = i<n ? 0.0 : h->image[3*gi+c]*l->edges[c];
          double b = j<n ? 0.0 : h->image[3*gj+c]*l->edges[c];
          dx[c] = (xi[c]-xj)-(b-a);
        }
        if (!mdrtDistMidpointOwns(handle,d,i,xi[0],xi[1],xi[2],dx[0],dx[1],dx[2]))
          continue;
      }
      ((int32_t *)dst.aligned)[i*dst.strides[0]+count++] = j;
    }
    m->selectedCounts[i] = count;
  }
}

/* Constraints in a group share the periodic image of the center atom.
   The arithmetic mean is evaluated in topology order before division. */
static int64_t groupOwner(const struct Layout *l, const struct ConstraintGroup *g,
                          const double points[4][3]) {
  int64_t owner = 0;
  for (int k = 2; k >= 0; --k) {
    double sum = 0, length = l->edges[k];
    for (int a = 1; a < g->size; ++a) {
      double delta = points[a][k] - points[0][k];
      sum += delta - length * floor(delta / length + 0.5);
    }
    double u = (points[0][k] + sum / g->size) / length;
    owner = owner * l->grid[k] + division(u - floor(u), l->grid[k]);
  }
  return owner;
}

static void numberOwnedRows(struct Layout *l) {
  memset(l->rows,0,l->domains*sizeof *l->rows);
  for (int64_t j = 0; j < l->count; ++j) ++l->rows[l->owner[j]];
  int64_t cap = 1;
  for (int64_t d = 0; d < l->domains; ++d)
    if (l->rows[d] > cap) cap = l->rows[d];
  if (l->stride != cap) {
    l->ids = realloc(l->ids,l->domains*cap*sizeof *l->ids);
    if (!l->ids) fail("out of memory indexing owned rows");
    l->stride = cap;
  }
  memset(l->rows,0,l->domains*sizeof *l->rows);
  for (int64_t j = 0; j < l->count; ++j) {
    int64_t d = l->owner[j], r = l->rows[d]++;
    l->ids[d * cap + r] = j;
    l->row[j] = r;
  }
}

void _mlir_ciface_mdrtDistGroupOwners(int64_t handle, struct Memref2D *positions,
                                      struct Members *members) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  size_t count = members->sizes[0];
  int64_t size = members->sizes[1];
  if (!count || l->groupsFrozen) return;
  if (size < 2 || size > 4 || count > (size_t)l->count ||
      count > SIZE_MAX / sizeof *l->groups - l->groupCount)
    fail("invalid constraint group dimensions");
  if (!l->groupCenters)
    l->groupCenters = calloc(l->count, sizeof *l->groupCenters);
  l->groups = realloc(l->groups, (l->groupCount + count) * sizeof *l->groups);
  if (!l->groupCenters || !l->groups) fail("out of memory partitioning constraint groups");
  for (size_t row = 0; row < count; ++row) {
    struct ConstraintGroup *g = &l->groups[l->groupCount++];
    g->size = size;
    double points[4][3];
    for (int a = 0; a < size; ++a) {
      int32_t atom = members->aligned[members->offset + row * members->strides[0] + a * members->strides[1]];
      if (atom < 0 || atom >= l->count) fail("invalid atom in constraint group");
      g->atoms[a] = atom;
      if (l->groupCenters[atom]) {
        char message[192];
        snprintf(message, sizeof message, "coupled constraint groups: atom %lld occurs at centers %lld and %lld",
                 (long long)atom + 1, (long long)l->groupCenters[atom], (long long)g->atoms[0] + 1);
        fail(message);
      }
      l->groupCenters[atom] = g->atoms[0] + 1;
      for (int k = 0; k < 3; ++k)
        points[a][k] = *at2(positions, atom, k);
    }
    int64_t owner = groupOwner(l, g, points);
    for (int a = 0; a < size; ++a)
      l->owner[g->atoms[a]] = owner;
  }
  numberOwnedRows(l);
}

int64_t _mlir_ciface_mdrtDistPartition(struct Memref2D *x, int64_t gx,
                                       int64_t gy, int64_t gz, double lx,
                                       double ly, double lz) {
  struct Layout *l = calloc(1, sizeof *l);
  if (!l)
    fail("out of memory");
  l->references = 1;
  l->grid[0] = gx;
  l->grid[1] = gy;
  l->grid[2] = gz;
  l->domains = gx * gy * gz;
  l->count = x->sizes[0];
  l->edges[0] = lx;
  l->edges[1] = ly;
  l->edges[2] = lz;
  int64_t n = l->count, cap = n > 0 ? n : 1;
  l->owner = malloc(sizeof(int64_t) * cap);
  l->row = malloc(sizeof(int64_t) * cap);
  l->rows = calloc(l->domains, sizeof(int64_t));
  if (!l->owner || !l->row || !l->rows)
    fail("out of memory");
  for (int64_t j = 0; j < n; ++j) {
    int64_t cell[3];
    for (int k = 0; k < 3; ++k) {
      double u = *at2(x, j, k) / l->edges[k];
      cell[k] = division(u - floor(u), l->grid[k]);
    }
    int64_t d = cell[0] + gx * (cell[1] + gy * cell[2]);
    l->owner[j] = d;
  }
  numberOwnedRows(l);
  return (int64_t)(intptr_t)l;
}

#ifdef MDIR_ENABLE_MPI
int64_t _mlir_ciface_mdrtDistMPIPartition(struct Memref2D *x, int64_t gx,
    int64_t gy, int64_t gz, double lx, double ly, double lz) {
  mdrtMPIPlaceDomains((int64_t[]){gx, gy, gz}, (double[]){lx, ly, lz});
  mdrtMPIDomainBound(gx * gy * gz, 0);
  int64_t handle = _mlir_ciface_mdrtDistPartition(x, gx, gy, gz, lx, ly, lz);
  ((struct Layout *)(intptr_t)handle)->mpi = 1;
  return handle;
}

/* The lowering passes a dense allocation. Transfer only live rows.
   Replicated ownership metadata keeps record order independent of rank count. */
static void shareRows(struct Layout *l, const int64_t *rows, struct Memref3D *field) {
  if (!l->mpi)
    return;
  if (field->strides[2] != 1 || field->strides[1] != field->sizes[2])
    mdrtMPIAbort("expected contiguous field rows");
  for (int64_t d = 0; d < l->domains; ++d)
    if (rows[d])
      mdrtMPIBroadcast(at3(field, d, 0, 0), rows[d] * field->sizes[2],
                       mdrtMPIOwner(l->domains, d));
}

/* Packed GPU-driver states share immutable owned rows. Operations that
   change field values produce new storage. MPI can populate a remote
   domain's scratch rows without changing this rank's owned values. */
struct StateField {
  size_t refs;
#ifdef MDIR_ENABLE_NCCL
  double *device;
  size_t bytes;
  int64_t domain;
  int deviceNewer;
  unsigned char *hostValid;
#endif
  double data[];
};

void _mlir_ciface_mdrtDistCreateField(struct Memref3D *out, int64_t domains,
                                      int64_t rows, int64_t width) {
  size_t limit = (SIZE_MAX-sizeof(struct StateField))/sizeof(double);
  if (domains < 1 || rows < 0 || width < 1 || (uint64_t)width > limit ||
      (uint64_t)rows > limit/width/domains)
    fail("invalid distributed field size");
  size_t count = (size_t)domains*rows*width;
  struct StateField *field = malloc(sizeof *field + count*sizeof(double));
  if (!field) fail("out of memory allocating distributed field");
  *field = (struct StateField){.refs = 1};
  *out = (struct Memref3D){(double *)field, field->data, 0,
                           {domains,rows,width}, {rows*width,width,1}};
}

void _mlir_ciface_mdrtDistRetainField(struct Memref3D *m) {
  ++((struct StateField *)m->allocated)->refs;
}

void _mlir_ciface_mdrtDistReleaseField(struct Memref3D *m) {
  struct StateField *field = (struct StateField *)m->allocated;
  if (--field->refs) return;
#ifdef MDIR_ENABLE_NCCL
  free(field->hostValid);
  if (field->device) mdrtNCCLFreePositions(field->device);
#endif
  free(field);
}

#ifdef MDIR_ENABLE_NCCL
static double *deviceRows(struct Memref3D *m, int64_t domain, int64_t rows, int output) {
  struct StateField *field = (struct StateField *)m->allocated;
  if (rows < 0 || rows > m->sizes[1]) fail("invalid distributed device extent");
  size_t bytes = (size_t)rows*m->sizes[2]*sizeof(double);
  if (!field->device) {
    field->device = mdrtNCCLAllocate(bytes);
    field->domain = domain;
    field->bytes = bytes;
    if (!output) mdrtNCCLUpload(field->device, at3(m,domain,0,0), bytes);
  } else if (field->domain != domain || field->bytes != bytes) {
    fail("distributed field changed its local row extent");
  }
  if (output) {
    free(field->hostValid);
    field->hostValid = NULL;
    field->deviceNewer = 1;
  }
  return field->device;
}

void _mlir_ciface_mdrtDistHostRows(struct Memref3D *m) {
  struct StateField *field = (struct StateField *)m->allocated;
  if (field->deviceNewer) {
    mdrtNCCLDownload(at3(m,field->domain,0,0),field->device,field->bytes);
    free(field->hostValid);
    field->hostValid = NULL;
    field->deviceNewer = 0;
  }
}

/* Materialize the union of tuple reads over the lifetime of an immutable
   device field. The full-slice flag stays dirty for other consumers. */
static void downloadSubset(struct Memref3D *m, int64_t domain,
                           const int *rows, int64_t count) {
  struct StateField *field=(void *)m->allocated;
  if (!field->deviceNewer || !count) return;
  int64_t width=m->sizes[2], extent=field->bytes/(width*sizeof(double));
  if (field->domain != domain) fail("tuple rows belong to another domain");
  if (!field->hostValid) {
    field->hostValid=calloc(extent ? extent : 1,1);
    if (!field->hostValid) fail("out of memory recording host tuple rows");
  }
  int *missing=malloc(count*sizeof *missing);
  if (!missing) fail("out of memory indexing host tuple reads");
  int64_t n=0;
  for (int64_t i=0; i<count; ++i) {
    int row=rows[i];
    if (row<0 || row>=extent) fail("tuple row exceeds device field extent");
    if (field->hostValid[row]) continue;
    field->hostValid[row]=1;
    missing[n++]=row;
  }
  if (n) {
    double *wire=malloc((size_t)n*width*sizeof *wire);
    if (!wire) fail("out of memory staging host tuple reads");
    mdrtNCCLDownloadRows(field->device,missing,n,width,wire);
    for (int64_t i=0; i<n; ++i)
      memcpy(at3(m,domain,missing[i],0),wire+i*width,width*sizeof *wire);
    free(wire);
  }
  free(missing);
}

void _mlir_ciface_mdrtDistHostTupleField(int64_t layout, struct Memref3D *field,
                                         struct Memref2I *tuples, int64_t ghost) {
  struct StateField *state=(void *)field->allocated;
  if (!state->deviceNewer) return;
  struct Layout *l=(void *)(intptr_t)layout;
  int64_t d=mdrtMPIDomainBound(l->domains,0), owned=l->rows[d];
  int64_t arity=tuples->sizes[1]-1, count=tuples->sizes[0];
  if (!count) return;
  int *rows=malloc((size_t)count*arity*sizeof *rows);
  if (!rows) fail("out of memory mapping host tuple reads");
  int64_t n=0;
  for (int64_t t=0; t<count; ++t) {
    int64_t *record=tuples->aligned+tuples->offset+t*tuples->strides[0];
    if (record[0]!=d) continue;
    for (int64_t a=1; a<=arity; ++a) {
      int64_t row=record[a*tuples->strides[1]];
      if ((row>=owned) != ghost) continue;
      row-=ghost ? owned : 0;
      if (row<0 || row>INT_MAX) fail("host tuple row exceeds index range");
      rows[n++]=(int)row;
    }
  }
  downloadSubset(field,d,rows,n);
  free(rows);
}

void _mlir_ciface_mdrtDistScalarDeviceView(struct Memref1D *out, int64_t d, int64_t n,
                                           struct Memref3D *field) {
  if (field->sizes[2] != 1) fail("expected scalar device rows");
  double *data = deviceRows(field,d,n,0);
  mdrtNCCLRowsReady();
  *out = (struct Memref1D){data,data,0,{n},{1}};
}

void _mlir_ciface_mdrtDistVectorDeviceView(struct Memref2D *out, int64_t d, int64_t n,
                                           struct Memref3D *field) {
  if (field->sizes[2] != 3) fail("expected vector device rows");
  double *data = deviceRows(field,d,n,0);
  mdrtNCCLRowsReady();
  *out = (struct Memref2D){data,data,0,{n,3},{3,1}};
}

void _mlir_ciface_mdrtDistAdvance(int64_t layout, struct Memref3D *x, struct Memref3D *v,
    struct Memref3D *mass, struct Memref3D *out, double dt, int64_t kick) {
  struct Layout *l = (struct Layout *)(intptr_t)layout;
  int64_t d = mdrtMPIDomainBound(l->domains,0), n = l->rows[d];
  mdrtNCCLAdvance(deviceRows(x,d,n,0),deviceRows(v,d,n,0),
      kick ? deviceRows(mass,d,n,0) : NULL,deviceRows(out,d,n,1),n,dt);
}

void _mlir_ciface_mdrtDistImportForces(struct Memref3D *field, int64_t domain,
    int64_t rows, int64_t start, struct Memref2D *forces) {
  if (forces->sizes[1] != 3 || start < 0 || rows > forces->sizes[0]-start)
    fail("invalid distributed force slice");
  mdrtNCCLImport(deviceRows(field,domain,rows,1),at2(forces,start,0),rows*3*sizeof(double));
}

int64_t _mlir_ciface_mdrtDistDeviceCovered(int64_t halo, struct Memref3D *ref,
    struct Memref3D *current, double cutoff) {
  struct Halo *h = (struct Halo *)(intptr_t)halo;
  int64_t d = mdrtMPIDomainBound(h->layout->domains,0), n = h->layout->rows[d];
  double half = 0.5*(h->reach-cutoff);
  if (!(cutoff > 0 && half > 0)) fail("coverage requires a positive cutoff and skin");
  int flag = mdrtNCCLCoverage(deviceRows(current,d,n,0),deviceRows(ref,d,n,0),n,half*half);
  if (flag == 2) fail("coverage requires finite displacement");
  return mdrtMPIAllTrue(!flag);
}
#endif

void _mlir_ciface_mdrtMPICopyState(struct Memref3D *source, struct Memref3D *dest) {
  int64_t domains = source->sizes[0];
  int64_t first = mdrtMPIDomainBound(domains, 0), end = mdrtMPIDomainBound(domains, 1);
  size_t length = (end-first)*source->sizes[1]*source->sizes[2];
  memcpy(at3(dest,first,0,0),at3(source,first,0,0),length*sizeof(double));
}

void _mlir_ciface_mdrtMPICollectRows(int64_t handle, struct Memref3D *field) {
  struct Layout *layout = (struct Layout *)(intptr_t)handle;
  shareRows(layout, layout->rows, field);
}

int64_t _mlir_ciface_mdrtMPICheckpointRows(int64_t handle, struct Memref3D *field) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  for (int64_t d = 0; d < l->domains; ++d)
    if (l->rows[d])
      mdrtMPIRowsToRoot(at3(field, d, 0, 0), l->rows[d] * field->sizes[2],
                        mdrtMPIOwner(l->domains, d));
  return mdrtMPIRank() == 0 ? l->domains : 0;
}

struct Memref2W {
  DistWide *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};

void _mlir_ciface_mdrtMPIShareWords(struct Memref2W *partial) {
  for (int64_t d = 0; d < partial->sizes[0]; ++d)
    mdrtMPIBroadcastWords(partial->aligned + partial->offset + d * partial->strides[0],
                          partial->sizes[1], mdrtMPIOwner(partial->sizes[0], d));
}

void _mlir_ciface_mdrtMPISharePartials(struct Memref2D *partial) {
  if (partial->strides[1] == 1 &&
      mdrtMPIGatherDomainPartials(at2(partial, 0, 0), partial->sizes[0],
                                  partial->strides[0], partial->sizes[1]))
    return;
  for (int64_t d = 0; d < partial->sizes[0]; ++d)
    mdrtMPIBroadcast(at2(partial, d, 0), partial->sizes[1],
                     mdrtMPIOwner(partial->sizes[0], d));
}
#else
#define shareRows(layout, rows, field) ((void)0)
#endif

/* The owned rows of each domain, and the particle of each. */
void _mlir_ciface_mdrtDistLayoutRows(int64_t handle, struct Memref1I *rows,
                                     struct Memref2I *ids) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  int64_t cap = l->stride;
  for (int64_t d = 0; d < l->domains; ++d) {
    rows->aligned[rows->offset + d * rows->strides[0]] = l->rows[d];
    for (int64_t r = 0; r < l->rows[d]; ++r)
      ids->aligned[ids->offset + d * ids->strides[0] + r * ids->strides[1]] =
          l->ids[d * cap + r];
  }
}

/* A state's retained handles own these immutable tables until unpack's
   block exits. Return dense views without materializing the same rows. */
void _mlir_ciface_mdrtDistRowView(struct Memref1I *out, int64_t handle, int64_t ghosts) {
  struct Halo *h = ghosts ? (struct Halo *)(intptr_t)handle : NULL;
  struct Layout *l = h ? h->layout : (struct Layout *)(intptr_t)handle;
  *out = (struct Memref1I){.aligned = h ? h->rows : l->rows,
                           .sizes = {l->domains}, .strides = {1}};
  out->allocated = out->aligned;
}

void _mlir_ciface_mdrtDistIdView(struct Memref2I *out, int64_t handle, int64_t ghosts) {
  struct Halo *h = ghosts ? (struct Halo *)(intptr_t)handle : NULL;
  struct Layout *l = h ? h->layout : (struct Layout *)(intptr_t)handle;
  int64_t stride = h ? h->capacity : l->stride;
  *out = (struct Memref2I){.aligned = h ? h->ids : l->ids,
                           .sizes = {l->domains,stride}, .strides = {stride,1}};
  out->allocated = out->aligned;
}

void mdrtDistRetainLayout(int64_t handle) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  if (!l || l->references <= 0)
    fail("retain of a released layout");
  ++l->references;
}

static int64_t ownedCapacity(const struct Layout *l) {
  return l->stride;
}

int64_t mdrtDistOwnedCapacity(int64_t handle) {
  return ownedCapacity((struct Layout *)(intptr_t)handle);
}

int64_t mdrtDistLayoutCount(int64_t handle) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  if (!l || l->references <= 0)
    fail("use of a released layout");
  return l->count;
}

#ifdef MDIR_ENABLE_NCCL
static void freeTuplePlans(struct TuplePlan *);
#endif
void mdrtDistReleaseLayout(int64_t handle) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  if (!l || l->references <= 0)
    fail("release of a released layout");
  if (--l->references > 0)
    return;
#ifdef MDIR_ENABLE_MPI
  free(l->migrantOffsets);
  free(l->migrantIds);
#endif
#ifdef MDIR_ENABLE_NCCL
  freeTuplePlans(l->tuplePlans);
  if (l->deviceMigrationMap) mdrtNCCLFreePositions((double *)l->deviceMigrationMap);
#endif
  free(l->migratedPositions);
  free(l->groups);
  free(l->groupCenters);
#ifdef MDIR_ENABLE_NCCL
  free(l->faceMasks);
  if (l->localFaceMasks) mdrtNCCLFreePositions((double *)l->localFaceMasks);
  if (l->identityOwner) {
    mdrtDistReleaseLayout((int64_t)(intptr_t)l->identityOwner);
  } else
#endif
  {
    free(l->owner);
    free(l->row);
    free(l->rows);
    free(l->ids);
  }
  free(l);
}

/* A migration is a new ownership epoch. Prove that the source's identity
   table is a bijection before any field can be copied into that epoch. */
static void checkIdentities(struct Layout *l) {
  int64_t n = l->count, cap = n > 0 ? n : 1;
  unsigned char *seen = calloc((size_t)cap, 1);
  if (!seen)
    fail("out of memory");
  int64_t visited = 0;
  for (int64_t d = 0; d < l->domains; ++d) {
    if (l->rows[d] < 0 || l->rows[d] > n)
      fail("invalid number of owned rows in a layout");
    for (int64_t r = 0; r < l->rows[d]; ++r) {
      int64_t j = l->ids[d * l->stride + r];
      if (j < 0 || j >= n || seen[j])
        fail("duplicate or invalid global particle identity in a layout");
      if (l->owner[j] != d || l->row[j] != r)
        fail("layout identity and owner/row tables disagree");
      seen[j] = 1;
      ++visited;
    }
  }
  if (visited != n)
    fail("missing global particle identity in a layout");
  free(seen);
}

/* The particles whose owner changes, by the number of axes of the grid
   along which the domain changes: across a face (1), an edge (2), or a
   corner (3); and the totals of all migrations of the process. */
static int64_t migratedTotals[4];

static void traceMigration(struct Layout *old, struct Layout *next) {
  int64_t moved[4] = {0, 0, 0, 0};
  for (int64_t j = 0; j < old->count; ++j) {
    int64_t a = old->owner[j], b = next->owner[j];
    if (a == b)
      continue;
    int64_t ca[3] = {a % old->grid[0], (a / old->grid[0]) % old->grid[1],
                     a / (old->grid[0] * old->grid[1])};
    int64_t cb[3] = {b % old->grid[0], (b / old->grid[0]) % old->grid[1],
                     b / (old->grid[0] * old->grid[1])};
    int axes = 0;
    for (int k = 0; k < 3; ++k)
      axes += ca[k] != cb[k];
    ++moved[axes];
    ++migratedTotals[axes];
  }
  fprintf(stderr,
          "mdrt_dist: migrate: %lld faces, %lld edges, %lld corners; "
          "total %lld faces, %lld edges, %lld corners\n",
          (long long)moved[1], (long long)moved[2], (long long)moved[3],
          (long long)migratedTotals[1], (long long)migratedTotals[2],
          (long long)migratedTotals[3]);
}

#ifdef MDIR_ENABLE_MPI
/* Redistribute is verified against the migration that produced its new
   layout. All fields in that migration have the same crossing identities. */
static void indexMigrants(struct Layout *old, struct Layout *next) {
  int peers = mdrtMPISize();
  int *owners = malloc(old->domains*sizeof *owners);
  int64_t *offsets = calloc((size_t)peers+1,sizeof *offsets);
  int64_t *cursor = malloc(peers*sizeof *cursor);
  if (!owners || !offsets || !cursor) fail("out of memory indexing migration");
  for (int64_t d = 0; d < old->domains; ++d) owners[d] = mdrtMPIOwner(old->domains,d);
  for (int64_t j = 0; j < old->count; ++j) {
    int from = owners[old->owner[j]];
    if (from != owners[next->owner[j]]) ++offsets[from+1];
  }
  for (int p = 0; p < peers; ++p) offsets[p+1] += offsets[p];
  next->migrantOffsets = offsets;
  if (offsets[peers]) {
    next->migrantIds = malloc(offsets[peers]*sizeof *next->migrantIds);
    if (!next->migrantIds) fail("out of memory recording migrating rows");
    memcpy(cursor,offsets,peers*sizeof *cursor);
    for (int64_t j = 0; j < old->count; ++j) {
      int from = owners[old->owner[j]];
      if (from != owners[next->owner[j]]) next->migrantIds[cursor[from]++] = j;
    }
  }
  free(owners);
  free(cursor);
}

static void shareMigrants(struct Layout *old, struct Layout *next, struct Memref3D *source) {
  int peers = mdrtMPISize(), rank = mdrtMPIRank();
  int64_t width = source->sizes[2], *offsets = next->migrantOffsets;
  for (int p = 0; p < peers; ++p) {
    int64_t rows = offsets[p+1]-offsets[p];
    if (!rows) continue;
    double *payload = malloc(rows*width*sizeof *payload);
    if (!payload) fail("out of memory sharing migration payload");
    if (p == rank)
      for (int64_t r = 0; r < rows; ++r) {
        int64_t j = next->migrantIds[offsets[p]+r];
        for (int64_t c = 0; c < width; ++c)
          payload[r*width+c] = *at3(source,old->owner[j],old->row[j],c);
      }
    mdrtMPIBroadcast(payload,rows*width,p);
    if (p != rank)
      for (int64_t r = 0; r < rows; ++r) {
        int64_t j = next->migrantIds[offsets[p]+r];
        for (int64_t c = 0; c < width; ++c)
          *at3(source,old->owner[j],old->row[j],c) = payload[r*width+c];
      }
    free(payload);
  }
}
#endif

static int64_t migrate(int64_t oldHandle, struct Memref3D *positions,
                       double lx, double ly, double lz, int keepPositions) {
  struct Layout *old = (struct Layout *)(intptr_t)oldHandle;
  if (!old || !positions || positions->sizes[0] != old->domains ||
      positions->sizes[1] < ownedCapacity(old) || positions->sizes[2] != 3)
    fail("migration positions do not cover the old layout");
  checkIdentities(old);
  shareRows(old, old->rows, positions);
  double edges[3] = {lx, ly, lz};
  for (int k = 0; k < 3; ++k)
    if (!(isfinite(edges[k]) && edges[k] > 0.0))
      fail("migration requires positive finite cell edges");
  struct Layout *next = calloc(1, sizeof *next);
  if (!next)
    fail("out of memory");
  next->references = 1;
  next->mpi = old->mpi;
  next->domains = old->domains;
  next->count = old->count;
  for (int k = 0; k < 3; ++k) {
    next->grid[k] = old->grid[k];
    next->edges[k] = edges[k];
  }
  int64_t n = next->count, cap = n > 0 ? n : 1;
  next->owner = malloc(sizeof(int64_t) * cap);
  next->row = malloc(sizeof(int64_t) * cap);
  next->rows = calloc(next->domains, sizeof(int64_t));
  if (!next->owner || !next->row || !next->rows)
    fail("out of memory");
  if (keepPositions) {
    next->migratedPositions = malloc(cap*sizeof *next->migratedPositions);
    if (!next->migratedPositions) fail("out of memory retaining migrated positions");
  }
  for (int64_t j = 0; j < n; ++j) {
    int64_t cell[3];
    for (int k = 0; k < 3; ++k) {
      double coordinate = *at3(positions, old->owner[j], old->row[j], k);
      if (!isfinite(coordinate))
        fail("migration requires finite positions");
      double u = coordinate / edges[k];
      if (!isfinite(u))
        fail("migration coordinate exceeds cell range");
      double image = floor(u);
      cell[k] = division(u - image, next->grid[k]);
      if (keepPositions) next->migratedPositions[j][k] = coordinate - edges[k]*image;
    }
    int64_t d = cell[0] + next->grid[0] *
                            (cell[1] + next->grid[1] * cell[2]);
    next->owner[j] = d;
  }
  if (old->groupCount) {
    next->groups = malloc(old->groupCount * sizeof *next->groups);
    if (!next->groups) fail("out of memory migrating constraint groups");
    next->groupCount = old->groupCount;
    for (size_t r = 0; r < old->groupCount; ++r) {
      struct ConstraintGroup *g = &next->groups[r];
      *g = old->groups[r];
      double points[4][3];
      for (int a = 0; a < g->size; ++a) {
        int32_t j = g->atoms[a];
        for (int k = 0; k < 3; ++k)
          points[a][k] = *at3(positions, old->owner[j], old->row[j], k);
      }
      int64_t owner = groupOwner(next, g, points);
      for (int a = 0; a < g->size; ++a)
        next->owner[g->atoms[a]] = owner;
    }
  }
  numberOwnedRows(next);
#ifdef MDIR_ENABLE_MPI
  if (next->mpi) indexMigrants(old,next);
#endif
  if (observer)
    for (int64_t j = 0; j < n; ++j)
      if (old->owner[j] != next->owner[j])
        observer->migration(observer->context, old->owner[j], next->owner[j]);
  checkIdentities(next);
  if (tracing())
    traceMigration(old, next);
  return (int64_t)(intptr_t)next;
}

int64_t _mlir_ciface_mdrtDistMigrate(int64_t old, struct Memref3D *x, double lx,
                                     double ly, double lz) {
  return migrate(old,x,lx,ly,lz,0);
}

int64_t _mlir_ciface_mdrtDistMigratePositions(int64_t old, struct Memref3D *x, double lx,
                                              double ly, double lz) {
  return migrate(old,x,lx,ly,lz,1);
}

#ifdef MDIR_ENABLE_MPI
void _mlir_ciface_mdrtDistMigratedPositions(int64_t handle, struct Memref3D *dest) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->rows[d]; ++r) {
      int64_t j = l->ids[d*l->stride+r];
      for (int c = 0; c < 3; ++c) *at3(dest,d,r,c) = l->migratedPositions[j][c];
    }
}
#endif

void _mlir_ciface_mdrtDistRedistribute(int64_t oldHandle, int64_t newHandle,
                                       struct Memref3D *source,
                                       struct Memref3D *destination,
                                       int64_t image) {
  struct Layout *old = (struct Layout *)(intptr_t)oldHandle;
  struct Layout *next = (struct Layout *)(intptr_t)newHandle;
  if (!old || !next || !source || !destination ||
      old->count != next->count ||
      old->domains != next->domains || source == destination ||
      source->aligned == destination->aligned ||
      source->sizes[0] != old->domains ||
      destination->sizes[0] != next->domains ||
      source->sizes[1] < ownedCapacity(old) ||
      destination->sizes[1] < ownedCapacity(next) ||
      source->sizes[2] != destination->sizes[2] ||
      (image && source->sizes[2] != 3))
    fail("redistribution requires matching, distinct field buffers");
#ifdef MDIR_ENABLE_MPI
  if (next->mpi) shareMigrants(old,next,source);
#endif
  int64_t components = source->sizes[2];
  int64_t first = 0, past = next->domains;
#ifdef MDIR_ENABLE_MPI
  if (next->mpi) {
    first = mdrtMPIDomainBound(next->domains, 0);
    past = mdrtMPIDomainBound(next->domains, 1);
  }
#endif
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < next->rows[d]; ++r) {
      int64_t j = next->ids[d*next->stride+r];
      for (int64_t k = 0; k < components; ++k) {
        double value = *at3(source, old->owner[j], old->row[j], k);
        if (image) {
          if (!isfinite(value))
            fail("redistributed positions must be finite");
          value -= next->edges[k] * floor(value / next->edges[k]);
        }
        *at3(destination,d,r,k) = value;
      }
    }
}

#ifdef MDIR_ENABLE_NCCL
/* Each rank owns one domain. Local survivors are gathered on device. A
   negative map entry selects an arriving row in sender/identity order. */
static void mapDeviceMigrants(struct Layout *old, struct Layout *next, int64_t d) {
  if (next->deviceMigrationMap) return;
  int64_t n = next->rows[d];
  if (n > INT_MAX || old->rows[d] > INT_MAX) fail("device migration exceeds row index range");
  int *map = malloc((n ? n : 1)*sizeof *map);
  if (!map) fail("out of memory mapping device migration");
  for (int64_t r = 0; r < n; ++r) {
    int64_t j = next->ids[d*next->stride+r];
    if (old->owner[j] == d) map[r] = (int)old->row[j];
  }
  for (int64_t i = 0; i < next->migrantOffsets[mdrtMPISize()]; ++i) {
    int64_t j = next->migrantIds[i];
    if (next->owner[j] == d) map[next->row[j]] = -1-(int)next->incomingRows++;
  }
  next->deviceMigrationMap = mdrtNCCLUploadIndices(map,n);
  free(map);
}

void _mlir_ciface_mdrtDistRedistributeDevice(struct Memref3D *result,
    int64_t oldHandle, int64_t nextHandle, struct Memref3D *source) {
  struct Layout *old = (void *)(intptr_t)oldHandle;
  struct Layout *next = (void *)(intptr_t)nextHandle;
  int peers = mdrtMPISize(), rank = mdrtMPIRank();
  int64_t d = mdrtMPIDomainBound(old->domains,0);
  int64_t *offsets = next->migrantOffsets;
  if (!offsets || old->domains != peers || next->domains != peers)
    fail("device redistribution requires one domain per rank");
  if (!offsets[peers]) {
    // Both layouts order each domain's particles by global identity.
    *result = *source;
    _mlir_ciface_mdrtDistRetainField(result);
    return;
  }
  int64_t width = source->sizes[2];
  mapDeviceMigrants(old,next,d);
  double *owned = deviceRows(source,d,old->rows[d],0);
  double *incoming = malloc((1+(size_t)next->incomingRows)*width*sizeof(double));
  if (!incoming) fail("out of memory receiving device migrants");
  int64_t received = 0;
  for (int p = 0; p < peers; ++p) {
    int64_t count = offsets[p+1]-offsets[p];
    if (!count) continue;
    double *wire = malloc((size_t)count*width*sizeof(double));
    if (!wire) fail("out of memory sharing device migrants");
    if (p == rank) {
      int *rows = malloc(count*sizeof *rows);
      if (!rows) fail("out of memory mapping outgoing migrants");
      for (int64_t r = 0; r < count; ++r)
        rows[r] = old->row[next->migrantIds[offsets[p]+r]];
      mdrtNCCLDownloadRows(owned,rows,count,width,wire);
      free(rows);
    }
    mdrtMPIBroadcast(wire,count*width,p);
    for (int64_t r = 0; r < count; ++r) {
      int64_t j = next->migrantIds[offsets[p]+r];
      if (next->owner[j] == d)
        memcpy(incoming+(received++)*width,wire+r*width,width*sizeof(double));
    }
    free(wire);
  }
  if (received != next->incomingRows) fail("inconsistent migration row map");
  _mlir_ciface_mdrtDistCreateField(result,next->domains,next->stride,width);
  mdrtNCCLReorder(owned, incoming, received, next->deviceMigrationMap,
      deviceRows(result,d,next->rows[d],1),next->rows[d],width);
  free(incoming);
}
#endif

#ifdef MDIR_ENABLE_NCCL
/* All rows outside the local face layer retain their owner and cannot be
   ghosts. Inside that layer the GPU returns the new owner and a bit for
   each receiving neighbor, including the local domain's periodic images.
   The host exchanges these records in place of particle coordinates. */
static int compareIdentities(const void *a, const void *b) {
  int64_t left=*(const int64_t *)a, right=*(const int64_t *)b;
  return (left>right)-(left<right);
}

static void borrowNumbering(struct Layout *old, struct Layout *next) {
  next->identityOwner=old->identityOwner ? old->identityOwner : old;
  mdrtDistRetainLayout((int64_t)(intptr_t)next->identityOwner);
  next->owner=old->owner;
  next->row=old->row;
  next->rows=old->rows;
  next->ids=old->ids;
  next->stride=old->stride;
}

int64_t _mlir_ciface_mdrtDistDeviceEpoch(int64_t oldHandle,
    struct Memref3D *positions, double lx, double ly, double lz, double reach, int64_t pairOnly) {
  struct Layout *old = (void *)(intptr_t)oldHandle;
  double edges[3] = {lx,ly,lz};
  double margin = 64*DBL_EPSILON*fmax(lx,fmax(ly,lz));
  int64_t d = mdrtMPIDomainBound(old->domains,0), rows = old->rows[d];
  int eligible = old->groupCount == 0 && old->count <= INT_MAX && old->domains <= INT_MAX;
  for (int k=0; k<3; ++k)
    eligible = eligible && edges[k] == old->edges[k] && 2*(reach+margin)<edges[k]/old->grid[k];
  if (!eligible) {
    _mlir_ciface_mdrtDistHostRows(positions);
    return migrate(oldHandle,positions,lx,ly,lz,1);
  }
  struct Layout *next = calloc(1,sizeof *next);
  if (!next) fail("out of memory creating device epoch");
  next->references=1;
  next->mpi=old->mpi;
  next->count=old->count;
  next->domains=old->domains;
  next->faceReach=reach;
  memcpy(next->edges,edges,sizeof edges);
  memcpy(next->grid,old->grid,sizeof next->grid);
  next->migrantOffsets=calloc(mdrtMPISize()+1,sizeof *next->migrantOffsets);
  if (!next->migrantOffsets) fail("out of memory indexing device migrants");
  next->pairFaces = pairOnly;
  if (pairOnly && old->domains==1) {
    next->selfFaces=1;
    borrowNumbering(old,next);
    if (tracing()) traceMigration(old,next);
    return (int64_t)(intptr_t)next;
  }
  int64_t localCount;
  int64_t *local = mdrtNCCLFaceRecords(deviceRows(positions,d,rows,0),rows,d,
      old->grid,edges,reach+margin,old->localFaceMasks,&localCount);
  int peers = mdrtMPISize();
  int64_t *offsets = malloc(((size_t)peers + 1) * sizeof(*offsets));
  if (!offsets)
    fail("out of memory indexing face records");
  struct FaceExchange *faces = mdrtMPIBeginFaces(2 * localCount, offsets);
  for (int64_t i=0; i<localCount; ++i) {
    int64_t r=local[2*i];
    if (r<0 || r>=rows) fail("invalid position classifying device face rows");
    local[2*i] = old->ids[d*old->stride+r];
  }
  size_t capacity = old->count ? old->count : 1;
  next->faceMasks=calloc(capacity,sizeof *next->faceMasks);
  if (!next->faceMasks) fail("out of memory indexing device face rows");
  if (old->localFaceMasks)
    memcpy(next->faceMasks, old->faceMasks, old->count * sizeof(*next->faceMasks));
  int64_t *allRecords = mdrtMPIFinishFaces(faces, local, offsets);
  free(local);
  int64_t moved=0;
  for (int peer = 0; peer < peers; ++peer) {
    int64_t count = (offsets[peer + 1] - offsets[peer]) / 2;
    int64_t *records = allRecords + offsets[peer];
    next->migrantOffsets[peer+1]=moved;
    int64_t arrivals=0;
    for (int64_t i=0; i<count; ++i) {
      int64_t j=records[2*i], owner=records[2*i+1]>>32;
      next->faceMasks[j]=(uint32_t)records[2*i+1];
      if (owner==old->owner[j]) continue;
      if (!next->owner) {
        next->owner=malloc(capacity*sizeof *next->owner);
        if (!next->owner) fail("out of memory recording new domain owners");
        memcpy(next->owner,old->owner,old->count*sizeof *next->owner);
      }
      // Compact in place after reading this row's identity and mask.
      records[arrivals++]=j;
      next->owner[j]=owner;
      if (observer) observer->migration(observer->context,old->owner[j],owner);
    }
    if (arrivals) {
      next->migrantIds=realloc(next->migrantIds,(moved+arrivals)*sizeof *next->migrantIds);
      if (!next->migrantIds) fail("out of memory recording device migrants");
      qsort(records,arrivals,sizeof *records,compareIdentities);
      memcpy(next->migrantIds+moved,records,arrivals*sizeof *records);
      moved+=arrivals;
      next->migrantOffsets[peer+1]=moved;
    }
  }
  free(allRecords);
  free(offsets);
  if (next->owner) {
    next->row=malloc(capacity*sizeof *next->row);
    next->rows=calloc(next->domains,sizeof *next->rows);
    if (!next->row || !next->rows) fail("out of memory numbering device migrants");
    numberOwnedRows(next);

  } else {
    borrowNumbering(old,next);

  }
  // Classification on the next rebuild reads masks in the new owned order.
  // The identity-indexed host snapshot also retains rows that did not move.
  size_t bytes = (size_t)next->rows[d] * sizeof(unsigned);
  unsigned *masks = malloc(bytes ? bytes : sizeof(*masks));
  if (!masks) fail("out of memory storing previous face classifications");
  for (int64_t r = 0; r < next->rows[d]; ++r)
    masks[r] = next->faceMasks[next->ids[d * next->stride + r]];
  next->localFaceMasks = (unsigned *)mdrtNCCLAllocate(bytes);
  mdrtNCCLUpload((double *)next->localFaceMasks, (const double *)masks, bytes);
  free(masks);
  if (tracing()) traceMigration(old,next);
  return (int64_t)(intptr_t)next;
}

void _mlir_ciface_mdrtDistEpochPositions(struct Memref3D *result,
    int64_t oldHandle, int64_t newHandle, struct Memref3D *source) {
  struct Layout *next=(void *)(intptr_t)newHandle;
  if (!next->faceMasks && !next->selfFaces) {
    _mlir_ciface_mdrtDistCreateField(result,next->domains,next->stride,3);
    _mlir_ciface_mdrtDistMigratedPositions(newHandle,result);
    return;
  }
  int64_t d=mdrtMPIDomainBound(next->domains,0), rows=next->rows[d];
  double *input;
  if (next->identityOwner) {
    _mlir_ciface_mdrtDistCreateField(result,next->domains,next->stride,3);
    input=deviceRows(source,d,rows,0);
  } else {
    _mlir_ciface_mdrtDistRedistributeDevice(result,oldHandle,newHandle,source);
    input=deviceRows(result,d,rows,0);
  }
  mdrtNCCLWrapRows(input,deviceRows(result,d,rows,1),rows,next->edges);
}
#endif

/* The box of domain `d`: its region of the grid, widened to hold its owned
   positions, so that every owned particle lies in it whatever the rounding
   of its division (the coverage argument of Section 4.3 needs x_i in it). */
static void domainBox(struct Layout *l, const double (*points)[3], int64_t d,
                      double lo[3], double hi[3], int midpoint) {
  int64_t cell[3] = {d % l->grid[0], (d / l->grid[0]) % l->grid[1],
                     d / (l->grid[0] * l->grid[1])};
  for (int k = 0; k < 3; ++k) {
    lo[k] = l->edges[k] * (double)cell[k] / (double)l->grid[k];
    hi[k] = l->edges[k] * (double)(cell[k] + 1) / (double)l->grid[k];
  }
  /* At midpoint ownership the interaction center must lie in the grid
     cell. The reach bounds every participating site from that center.
     An owned site across a face of the periodic box cannot enlarge it. */
  if (midpoint) return;
  for (int64_t r = 0; r < l->rows[d]; ++r)
    for (int k = 0; k < 3; ++k) {
      double v = points[l->ids[d*l->stride+r]][k];
      if (v < lo[k])
        lo[k] = v;
      if (v > hi[k])
        hi[k] = v;
    }
}

static double outsideBox(double y, double lo, double hi) {
  return y < lo ? lo-y : (y > hi ? y-hi : 0.0);
}

/* Bin consecutive particle identities together. A bin's bounds reject
   entire image ranges without reordering the surviving rows. */
enum { GhostBinSize = 32 };
struct GhostBin { double lo[3], hi[3]; };
struct GhostRow { int64_t id, image[3]; };
struct GhostList { struct GhostRow *data; size_t size, capacity; double lo[3], hi[3]; };

static void visitGhosts(struct Halo *h, const double (*points)[3],
                        const struct GhostBin *bins, int64_t d,
                        const double lo[3], const double hi[3],
                        struct GhostList *list) {
  struct Layout *l = h->layout;
  const double reach2 = h->reach*h->reach;
  int64_t span[3];
  for (int c = 0; c < 3; ++c)
    span[c] = (int64_t)ceil(h->reach/l->edges[c]);
  for (int64_t first = 0; first < l->count; first += GhostBinSize) {
    const struct GhostBin *bin = &bins[first/GhostBinSize];
    int64_t low[3], high[3];
    int c = 0;
    for (; c < 3; ++c) {
      double edge = l->edges[c];
      low[c] = -span[c];
      high[c] = span[c];
      while (low[c] <= high[c] && bin->hi[c]+low[c]*edge < lo[c]-h->reach)
        ++low[c];
      while (low[c] <= high[c] && bin->lo[c]+high[c]*edge > hi[c]+h->reach)
        --high[c];
      if (low[c] > high[c]) break;
    }
    if (c != 3) continue;
    int64_t end = first+GhostBinSize < l->count ? first+GhostBinSize : l->count;
    for (int64_t j = first; j < end; ++j) {
      const double *x = points[j];
      for (int64_t a = low[0]; a <= high[0]; ++a) {
        double dx = outsideBox(x[0]+a*l->edges[0],lo[0],hi[0]);
        double dx2 = dx*dx;
        if (!(dx2 < reach2)) continue;
        for (int64_t b = low[1]; b <= high[1]; ++b) {
          double dy = outsideBox(x[1]+b*l->edges[1],lo[1],hi[1]);
          double dxy2 = dx2+dy*dy;
          if (!(dxy2 < reach2)) continue;
          for (int64_t z = low[2]; z <= high[2]; ++z) {
            if (l->owner[j] == d && a == 0 && b == 0 && z == 0) continue;
            double dz = outsideBox(x[2]+z*l->edges[2],lo[2],hi[2]);
            if (!(dxy2+dz*dz < reach2)) continue;
            if (list->size == list->capacity) {
              list->capacity = list->capacity ? 2*list->capacity : 64;
              list->data = realloc(list->data,list->capacity*sizeof *list->data);
              if (!list->data) fail("out of memory recording ghost rows");
            }
            list->data[list->size++] = (struct GhostRow){j,{a,b,z}};
          }
        }
      }
    }
  }
}

struct HaloBuild {
  struct Halo *halo;
  double (*points)[3];
  struct GhostBin *bins;
  struct GhostList *lists;
  int midpoint;
};

int64_t _mlir_ciface_mdrtDistBeginHalo(int64_t layout, struct Memref3D *x,
                                       double reach, int64_t midpoint, int64_t migrated) {
  struct Halo *h = calloc(1, sizeof *h);
  if (!h)
    fail("out of memory");
  h->references = 1;
  h->layout = (struct Layout *)(intptr_t)layout;
  mdrtDistRetainLayout(layout);
  h->reach = reach;
  if (!migrated) shareRows(h->layout, h->layout->rows, x);
  int64_t domains = h->layout->domains;
  h->rows = calloc(domains, sizeof(int64_t));
  if (!h->rows)
    fail("out of memory");
  struct Layout *l = h->layout;
  size_t length = l->count ? l->count : 1;
  double (*points)[3] = migrated ? l->migratedPositions : malloc(length*sizeof *points);
  struct GhostBin *bins = malloc(((length+GhostBinSize-1)/GhostBinSize)*sizeof *bins);
  struct GhostList *lists = calloc(domains,sizeof *lists);
  if (!points || !bins || !lists) fail("out of memory binning halo positions");
  for (int64_t j = 0; j < l->count; ++j) {
    struct GhostBin *bin = &bins[j/GhostBinSize];
    for (int c = 0; c < 3; ++c) {
      if (!migrated) points[j][c] = *at3(x,l->owner[j],l->row[j],c);
      double v = points[j][c];
      if (j%GhostBinSize == 0 || v < bin->lo[c]) bin->lo[c] = v;
      if (j%GhostBinSize == 0 || v > bin->hi[c]) bin->hi[c] = v;
    }
  }
  struct HaloBuild *build = malloc(sizeof *build);
  if (!build) fail("out of memory starting halo construction");
  *build = (struct HaloBuild){h,points,bins,lists,midpoint};
  return (int64_t)(intptr_t)build;
}

/* Each domain owns its output rows. A threaded caller visits independent
   domains in parallel while each domain retains its identity/image order. */
void _mlir_ciface_mdrtDistVisitHalo(int64_t handle, int64_t domain, struct Memref3D *x) {
  struct HaloBuild *build = (struct HaloBuild *)(intptr_t)handle;
  struct Halo *h = build->halo;
  double *lo = build->lists[domain].lo, *hi = build->lists[domain].hi;
  domainBox(h->layout,build->points,domain,lo,hi,build->midpoint);
  visitGhosts(h,build->points,build->bins,domain,lo,hi,&build->lists[domain]);
  h->rows[domain] = build->lists[domain].size;
}

static int64_t finishHaloMetadata(struct Halo *h, struct GhostList *lists) {
  int64_t domains = h->layout->domains;
  if (observer) {
    // Count neighbor multiplicities once per halo, including periodic
    // self-images. Later transfers scale these counts by their field width.
    int64_t *counts = calloc(domains, sizeof *counts);
    if (!counts) fail("out of memory counting halo imports");
    for (int64_t d = 0; d < domains; ++d) {
      const double *lo = lists[d].lo, *hi = lists[d].hi;
      observer->domain(observer->context, d, h->layout->rows[d], h->rows[d], lo, hi);
      for (int64_t g = 0; g < h->rows[d]; ++g)
        ++counts[h->sourceDomain[d * h->capacity + g]];
      size_t neighbors = 0;
      for (int64_t e = 0; e < domains; ++e)
        neighbors += counts[e] != 0;
      if (neighbors) {
        h->edges = realloc(h->edges, (h->edgeCount + neighbors) * sizeof *h->edges);
        if (!h->edges) fail("out of memory counting halo edges");
      }
      for (int64_t e = 0; e < domains; ++e)
        if (counts[e]) {
          h->edges[h->edgeCount++] = (struct Edge){e, d, counts[e]};
          counts[e] = 0;
        }
    }
  }
  free(lists);
  if (tracing())
    for (int64_t d = 0; d < domains; ++d)
      fprintf(stderr,
              "mdrt_dist: halo reach %g: domain %lld owns %lld rows, "
              "holds %lld ghosts\n",
              h->reach, (long long)d, (long long)h->layout->rows[d],
              (long long)h->rows[d]);
  return (int64_t)(intptr_t)h;
}

int64_t _mlir_ciface_mdrtDistFinishHalo(int64_t handle, struct Memref3D *x) {
  struct HaloBuild *build = (struct HaloBuild *)(intptr_t)handle;
  struct Halo *h = build->halo;
  struct Layout *l = h->layout;
  struct GhostList *lists = build->lists;
  int64_t domains = l->domains;
  if (build->points != l->migratedPositions) free(build->points);
  free(build->bins);
  free(build);
  for (int64_t d = 0; d < domains; ++d)
    if (h->rows[d] > h->capacity) h->capacity = h->rows[d];
  int64_t cap = h->capacity > 0 ? h->capacity : 1;
  h->capacity = cap;
  h->sourceDomain = malloc(sizeof(int64_t) * domains * cap);
  h->sourceRow = malloc(sizeof(int64_t) * domains * cap);
  h->ids = malloc(sizeof(int64_t) * domains * cap);
  h->image = malloc(sizeof(int64_t) * 3 * domains * cap);
  if (!h->sourceDomain || !h->sourceRow || !h->ids || !h->image)
    fail("out of memory");
  for (int64_t d = 0; d < domains; ++d) {
    for (int64_t r = 0; r < h->rows[d]; ++r) {
      struct GhostRow *row = &lists[d].data[r];
      int64_t g = d*cap+r, j = row->id;
      h->ids[g] = j;
      h->sourceDomain[g] = l->owner[j];
      h->sourceRow[g] = l->row[j];
      memcpy(&h->image[3*g],row->image,sizeof row->image);
    }
    free(lists[d].data);
  }
  return finishHaloMetadata(h, lists);
}

#ifdef MDIR_ENABLE_NCCL
struct FaceDestination {
  int64_t domain, image[3], count;
};

// A directed pair traversal consumes only this rank's imports and exports.
// Count logical edges for reporting, then retain just the incident rows in
// destination/identity/image order, as in the complete halo.
static void buildPairFacePlan(struct Halo *h, struct FaceDestination *routes,
                              const struct GhostList *boxes) {
  struct Layout *l = h->layout;
  int64_t local = mdrtMPIDomainBound(l->domains, 0);
  int64_t *counts = calloc(l->domains, sizeof(*counts));
  int64_t *offsets = calloc(l->domains + 1, sizeof(*offsets));
  uint32_t *incident = calloc(l->domains, sizeof(*incident));
  if (!counts || !offsets || !incident) fail("out of memory indexing pair faces");
  for (int64_t d = 0; d < l->domains; ++d)
    for (int bit = 0; bit < 27; ++bit)
      if (d == local || routes[27*d + bit].domain == local)
        incident[d] |= 1u << bit;
  for (int64_t j = 0; j < l->count; ++j) {
    uint32_t mask = l->faceMasks[j];
    if (!mask) continue;
    struct FaceDestination *row = routes + 27*l->owner[j];
    while (mask) {
      int bit = __builtin_ctz(mask);
      ++row[bit].count;
      mask &= mask - 1;
    }
  }
  if (observer) {
    h->edges = malloc(27*l->domains*sizeof(*h->edges));
    if (!h->edges) fail("out of memory counting pair face transfers");
  }
  for (int64_t d = 0; d < l->domains; ++d)
    for (int bit = 0; bit < 27; ++bit) {
      struct FaceDestination *route = &routes[27*d + bit];
      if (!route->count) continue;
      h->rows[route->domain] += route->count;
      if (incident[d] & (1u << bit))
        counts[route->domain] += route->count;
      if (observer)
        h->edges[h->edgeCount++] = (struct Edge){d, route->domain, route->count};
    }

  h->capacity = 1;
  for (int64_t d = 0; d < l->domains; ++d) {
    if (h->rows[d] > h->capacity) h->capacity = h->rows[d];
    offsets[d+1] = offsets[d] + counts[d];
    counts[d] = 0;
    if (observer)
      observer->domain(observer->context, d, l->rows[d], h->rows[d], boxes[d].lo, boxes[d].hi);
  }
  size_t n = offsets[l->domains] ? offsets[l->domains] : 1;
  int64_t *from = malloc(n*sizeof(*from)), *source = malloc(n*sizeof(*source));
  int64_t *image = malloc(3*n*sizeof(*image));
  if (!from || !source || !image) fail("out of memory packing pair faces");
  for (int64_t j = 0; j < l->count; ++j) {
    uint32_t mask = l->faceMasks[j];
    if (!mask) continue;
    int64_t d = l->owner[j];
    mask &= incident[d];
    while (mask) {
      int bit = 31 - __builtin_clz(mask);
      mask ^= 1u << bit;
      struct FaceDestination *route = &routes[27*d + bit];
      int64_t slot = offsets[route->domain] + counts[route->domain]++;
      from[slot] = d;
      source[slot] = l->row[j];
      memcpy(image + 3*slot, route->image, sizeof(route->image));
    }
  }
  h->ncclPlan = mdrtNCCLPlan(l->rows[local], l->domains, counts, 0, offsets,
                             from, source, image, l->edges);
  free(image);
  free(source);
  free(from);
  free(incident);
  free(offsets);
  free(counts);
}

int64_t _mlir_ciface_mdrtDistDeviceEpochHalo(int64_t layout, struct Memref3D *x, double reach) {
  struct Layout *l=(void *)(intptr_t)layout;
  if (l->selfFaces) {
    if (reach!=l->faceReach) fail("local face plan belongs to a different halo reach");
    struct Halo *h=calloc(1,sizeof *h);
    if (!h) fail("out of memory building local faces");
    h->references=1;
    h->layout=l;
    h->reach=reach;
    mdrtDistRetainLayout(layout);
    h->rows=malloc(sizeof *h->rows);
    if (!h->rows) fail("out of memory counting local faces");
    double margin=64*DBL_EPSILON*fmax(l->edges[0],fmax(l->edges[1],l->edges[2]));
    h->ncclPlan=mdrtNCCLSelfPlan(deviceRows(x,0,l->rows[0],0),l->rows[0],
                                 l->edges,reach+margin,h->rows);
    h->capacity=h->rows[0] ? h->rows[0] : 1;
    if (observer) {
      double lo[3]={0,0,0};
      observer->domain(observer->context,0,l->rows[0],h->rows[0],lo,l->edges);
      if (h->rows[0]) {
        h->edges=malloc(sizeof *h->edges);
        if (!h->edges) fail("out of memory counting local halo edges");
        h->edges[h->edgeCount++]=(struct Edge){0,0,h->rows[0]};
      }
    }
    if (tracing())
      fprintf(stderr,"mdrt_dist: halo reach %g: domain 0 owns %lld rows, holds %lld ghosts\n",
              reach,(long long)l->rows[0],(long long)h->rows[0]);
    return (int64_t)(intptr_t)h;
  }
  if (!l->faceMasks) {
    int64_t build=_mlir_ciface_mdrtDistBeginHalo(layout,x,reach,0,1);
    for (int64_t d=0; d<l->domains; ++d) _mlir_ciface_mdrtDistVisitHalo(build,d,x);
    return _mlir_ciface_mdrtDistFinishHalo(build,x);
  }
  if (l->faceReach!=reach) fail("device face map belongs to a different halo reach");
  struct Halo *h=calloc(1,sizeof *h);
  struct GhostList *lists=calloc(l->domains,sizeof *lists);
  if (!h || !lists) fail("out of memory building device face halo");
  h->references=1;
  h->layout=l;
  h->reach=reach;
  h->rows=calloc(l->domains,sizeof *h->rows);
  if (!h->rows) fail("out of memory counting device faces");
  mdrtDistRetainLayout(layout);
  // Each direction selects one receiving domain and periodic translation,
  // independently of the row identity. Resolve it once per domain.
  struct FaceDestination *routes=calloc(l->domains*27,sizeof *routes);
  if (!routes) fail("out of memory routing device face rows");
  for (int64_t d=0; d<l->domains; ++d) {
    int64_t code=d;
    for (int k=0; k<3; ++k) {
      int64_t cell=code%l->grid[k];
      code/=l->grid[k];
      lists[d].lo[k]=l->edges[k]*cell/l->grid[k];
      lists[d].hi[k]=l->edges[k]*(cell+1)/l->grid[k];
    }
    for (int bit=0; bit<27; ++bit) {
      struct FaceDestination *route=&routes[d*27+bit];
      int relative[3]={bit/9-1,(bit/3)%3-1,bit%3-1};
      int64_t from=d, stride=1;
      for (int k=0; k<3; ++k) {
        int64_t grid=l->grid[k], dest=from%grid+relative[k];
        from/=grid;
        if (dest<0) { dest+=grid; route->image[k]=1; }
        if (dest>=grid) { dest-=grid; route->image[k]=-1; }
        route->domain+=dest*stride;
        stride*=grid;
      }
    }
  }
  if (l->pairFaces && !tracing()) {
    buildPairFacePlan(h, routes, lists);
    free(routes);
    free(lists);
    return (int64_t)(intptr_t)h;
  }
  int64_t cap=1;
  for (int pass=0; pass<2; ++pass) {
    for (int64_t j=0; j<l->count; ++j) {
      uint32_t mask=l->faceMasks[j];
      if (!mask) continue;
      const struct FaceDestination *domainRoutes=routes+27*l->owner[j];
      // Descending direction bits give ascending periodic images.
      while (mask) {
        int bit=31-__builtin_clz(mask);
        mask ^= 1u<<bit;
        const struct FaceDestination *dest=domainRoutes+bit;
        int64_t d=dest->domain;
        if (!pass) {
          ++h->rows[d];
        } else {
          int64_t r=d*cap+lists[d].size++;
          h->sourceDomain[r]=l->owner[j];
          h->sourceRow[r]=l->row[j];
          h->ids[r]=j;
          memcpy(h->image+3*r,dest->image,sizeof dest->image);
        }
      }
    }
    if (!pass) {
      for (int64_t d=0; d<l->domains; ++d)
        if (h->rows[d]>cap) cap=h->rows[d];
      h->capacity=cap;
      size_t slots=(size_t)cap*l->domains;
      h->sourceDomain=malloc(slots*sizeof *h->sourceDomain);
      h->sourceRow=malloc(slots*sizeof *h->sourceRow);
      h->ids=malloc(slots*sizeof *h->ids);
      h->image=malloc(3*slots*sizeof *h->image);
      if (!h->sourceDomain || !h->sourceRow || !h->ids || !h->image)
        fail("out of memory indexing device faces");
    }
  }
  free(routes);
  return finishHaloMetadata(h,lists);
}
#endif

static int64_t buildHalo(int64_t layout, struct Memref3D *x, double reach, int midpoint) {
  int64_t build = _mlir_ciface_mdrtDistBeginHalo(layout,x,reach,midpoint,0);
  const struct Layout *l = (struct Layout *)(intptr_t)layout;
  for (int64_t d = 0; d < l->domains; ++d)
    _mlir_ciface_mdrtDistVisitHalo(build,d,x);
  return _mlir_ciface_mdrtDistFinishHalo(build,x);
}

int64_t _mlir_ciface_mdrtDistHalo(int64_t layout, struct Memref3D *x, double reach) {
  return buildHalo(layout,x,reach,0);
}

int64_t _mlir_ciface_mdrtDistMidpointHalo(int64_t layout, struct Memref3D *x, double reach) {
  return buildHalo(layout,x,reach,1);
}

int64_t mdrtDistHaloCapacity(int64_t handle) {
  return ((struct Halo *)(intptr_t)handle)->capacity;
}

/* A team-wide Verlet skin predicate (Verlet, Phys. Rev. 159, 98, 1967).
   Each global identity is visited once
   through the halo's ownership layout; even domains with no rows take part
   in the decision. The current coordinate payload is independent of the
   halo's stored membership and must be refreshed separately. */
int64_t _mlir_ciface_mdrtDistCovered(int64_t handle,
                                     struct Memref3D *reference,
                                     struct Memref3D *current,
                                     double cutoff) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (!h || !reference || !current ||
      !(isfinite(cutoff) && cutoff > 0.0 && cutoff < h->reach))
    fail("coverage requires a halo and a cutoff below its reach");
  struct Layout *l = h->layout;
  if (reference->sizes[0] != l->domains ||
      current->sizes[0] != l->domains ||
      reference->sizes[1] < ownedCapacity(l) || current->sizes[1] < ownedCapacity(l) ||
      reference->sizes[2] != 3 || current->sizes[2] != 3)
    fail("coverage positions do not span the halo layout");
  double halfSkin = (h->reach - cutoff) * 0.5;
  double limit2 = halfSkin * halfSkin;
  int64_t first = 0, past = l->domains;
#ifdef MDIR_ENABLE_MPI
  if (l->mpi) {
    first = mdrtMPIDomainBound(l->domains, 0);
    past = mdrtMPIDomainBound(l->domains, 1);
  }
#endif
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->rows[d]; ++r) {
      double displacement2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        double a = *at3(reference, d, r, k);
        double b = *at3(current, d, r, k);
        if (!isfinite(a) || !isfinite(b))
          fail("coverage requires finite positions");
        /* Halo rows retain explicit images. A wrap changes that image, so
           minimum-image displacement would incorrectly preserve coverage. */
        double delta = b - a;
        if (!isfinite(delta))
          fail("coverage requires finite displacement");
        displacement2 += delta * delta;
      }
      if (!(displacement2 < limit2))
        {
#ifdef MDIR_ENABLE_MPI
          if (l->mpi)
            return mdrtMPIAllTrue(0);
#endif
          return 0;
        }
    }
#ifdef MDIR_ENABLE_MPI
  if (l->mpi)
    return mdrtMPIAllTrue(1);
#endif
  return 1;
}

/* The partition checked disjointness before assigning a row to each
   member. Verify the common owner again at the epoch of the solve. */
void _mlir_ciface_mdrtDistConstraintRows(int64_t handle, struct Memref3D *positions,
                                         struct Members *members, struct Memref2I *out) {
  const struct Layout *l = (struct Layout *)(intptr_t)handle;
  (void)positions;
  for (int64_t t = 0; t < members->sizes[0]; ++t) {
    int64_t *record = out->aligned + out->offset + t*out->strides[0];
    for (int64_t a = 0; a < members->sizes[1]; ++a) {
      int32_t j = members->aligned[members->offset + t*members->strides[0] + a*members->strides[1]];
      if (j < 0 || j >= l->count) fail("constraint member outside particle set");
      if (a == 0) record[0] = l->owner[j];
      if (record[0] != l->owner[j]) fail("constraint group split across domains");
      record[(a+1)*out->strides[1]] = l->row[j];
    }
  }
}

#ifdef MDIR_ENABLE_NCCL
void _mlir_ciface_mdrtDistCompactTupleIds(struct Memref1I *out,
                                          struct Memref2I *schedule,
                                          int64_t domain) {
  int64_t count = schedule->sizes[0];
  int64_t *ids = malloc((count ? count : 1) * sizeof *ids);
  if (!ids)
    fail("out of memory compacting the local tuple schedule");
  int64_t n = 0;
  for (int64_t i = 0; i < count; ++i)
    if (schedule->aligned[schedule->offset + i * schedule->strides[0]] ==
        domain)
      ids[n++] = i;
  *out = (struct Memref1I){ids, ids, 0, {n}, {1}};
}

void _mlir_ciface_mdrtDistShareTupleMembers(int64_t layout, struct Memref3D *positions,
                                             struct Members *members) {
  struct Layout *l=(void *)(intptr_t)layout;
  unsigned char *needed=calloc(l->count ? l->count : 1,1);
  if (!needed) fail("out of memory recording tuple members");
  for (int64_t t=0; t<members->sizes[0]; ++t) {
    for (int64_t a=0; a<members->sizes[1]; ++a) {
      int32_t j=members->aligned[members->offset+t*members->strides[0]+a*members->strides[1]];
      if (j<0 || j>=l->count) fail("tuple member outside particle set");
      needed[j]=1;
    }
  }
  int *rows=malloc(l->stride*sizeof *rows);
  double *wire=malloc(l->stride*3*sizeof *wire);
  if (!rows || !wire) fail("out of memory exchanging tuple coordinates");
  for (int64_t d=0; d<l->domains; ++d) {
    int64_t n=0;
    for (int64_t r=0; r<l->rows[d]; ++r)
      if (needed[l->ids[d*l->stride+r]]) {
        if (r>INT_MAX) fail("tuple coordinates exceed device row range");
        rows[n++]=(int)r;
      }
    if (!n) continue;
    int root=mdrtMPIOwner(l->domains,d);
    if (root==mdrtMPIRank()) {
      downloadSubset(positions,d,rows,n);
      if (mdrtMPISize()==1) continue;
      for (int64_t i=0; i<n; ++i)
        memcpy(wire+3*i,at3(positions,d,rows[i],0),3*sizeof *wire);
    }
    mdrtMPIBroadcast(wire,3*n,root);
    if (root!=mdrtMPIRank())
      for (int64_t i=0; i<n; ++i)
        memcpy(at3(positions,d,rows[i],0),wire+3*i,3*sizeof *wire);
  }
  free(needed);
  free(rows);
  free(wire);
}
#endif

/* Index the imported images by identity. An atom may have several images
   on one domain, including periodic images of a locally owned atom. */
static void tupleRows(int64_t handle, struct Memref3D *positions,
                      struct Members *members, struct Memref2I *out, int local) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  struct Layout *l = h->layout;
  int64_t count = members->sizes[0], arity = members->sizes[1];
  if (arity < 2 || out->sizes[0] != count || out->sizes[1] != 1 + arity)
    fail("invalid tuple ownership schedule shape");
  if (count == 0) return;
  int64_t begin = 0, end = l->domains;
#ifdef MDIR_ENABLE_MPI
  if (local) {
    begin = mdrtMPIDomainBound(l->domains,0);
    end = mdrtMPIDomainBound(l->domains,1);

  }
#endif
  if (!h->firstGhost) {
    h->firstGhost = malloc(l->domains*l->count*sizeof *h->firstGhost);
    h->nextGhost = malloc(l->domains*h->capacity*sizeof *h->nextGhost);
    if (!h->firstGhost || !h->nextGhost) fail("out of memory indexing tuple images");
    for (int64_t d = 0; d < l->domains; ++d) {
      int64_t *heads = &h->firstGhost[d*l->count];
      for (int64_t j = 0; j < l->count; ++j) heads[j] = -1;
      for (int64_t r = 0; r < h->rows[d]; ++r) {
        int64_t slot = d*h->capacity+r;
        int64_t *head = &heads[h->ids[slot]];
        h->nextGhost[slot] = *head;
        *head = r;
      }
    }
  }
  int64_t *ids = malloc(arity*sizeof *ids);
  int64_t (*image)[3] = calloc(arity,sizeof *image);
  if (!ids || !image) fail("out of memory locating tuple members");
  for (int64_t t = 0; t < count; ++t) {
    int64_t anchor = l->count;
    for (int64_t a = 0; a < arity; ++a) {
      ids[a] = members->aligned[members->offset + t*members->strides[0] + a*members->strides[1]];
      if (ids[a] < 0 || ids[a] >= l->count) fail("tuple member outside particle set");
      if (ids[a] < anchor) anchor = ids[a];
    }
    int64_t *record = out->aligned + out->offset + t*out->strides[0];
    int64_t d = 0;
    for (int k = 2; k >= 0; --k) {
      double origin = *at3(positions, l->owner[anchor], l->row[anchor], k);
      double lo = 0, hi = 0, length = l->edges[k];
      for (int64_t a = 0; a < arity; ++a) {
        double delta = *at3(positions, l->owner[ids[a]], l->row[ids[a]], k) - origin;
        image[a][k] = -(int64_t)floor(delta / length + 0.5);
        delta += (double)image[a][k] * length;
        lo = fmin(lo, delta);
        hi = fmax(hi, delta);
      }
      double u = (origin + 0.5*(lo + hi)) / length;
      double translation = floor(u);
      d = d*l->grid[k] + division(u - translation, l->grid[k]);
      for (int64_t a = 0; a < arity; ++a)
        image[a][k] -= (int64_t)translation;
    }
    record[0] = d;
    if (d < begin || d >= end) continue;
    for (int64_t a = 0; a < arity; ++a) {
      int64_t j = ids[a], row = -1;
      if (l->owner[j] == d && !image[a][0] && !image[a][1] && !image[a][2])
        row = l->row[j];
      for (int64_t g = h->firstGhost[d*l->count+j]; row < 0 && g >= 0; g = h->nextGhost[d*h->capacity+g]) {
        int64_t *n = &h->image[3*(d*h->capacity + g)];
        if (n[0] == image[a][0] && n[1] == image[a][1] && n[2] == image[a][2])
          row = l->rows[d] + g;
      }
      if (row < 0) {
        char message[192];
        snprintf(message, sizeof message, "missing tuple owner coverage: tuple %lld domain %lld lacks atom %lld",
                 (long long)t+1, (long long)d, (long long)j+1);
        fail(message);
      }
      record[(a+1)*out->strides[1]] = row;
    }
  }
  free(image);
  free(ids);
}

void _mlir_ciface_mdrtDistTupleRows(int64_t handle, struct Memref3D *positions,
                                    struct Members *members, struct Memref2I *out) {
  tupleRows(handle,positions,members,out,0);
}

#ifdef MDIR_ENABLE_MPI
void _mlir_ciface_mdrtDistLocalTupleRows(int64_t handle, struct Memref3D *positions,
                                         struct Members *members, struct Memref2I *out) {
  tupleRows(handle,positions,members,out,1);
}
#endif

/* The ghost rows of each domain, and the particle each is an image of. */
void _mlir_ciface_mdrtDistHaloRows(int64_t handle, struct Memref1I *rows,
                                   struct Memref2I *ids) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  for (int64_t d = 0; d < h->layout->domains; ++d) {
    rows->aligned[rows->offset + d * rows->strides[0]] = h->rows[d];
    for (int64_t g = 0; g < h->rows[d]; ++g)
      ids->aligned[ids->offset + d * ids->strides[0] + g * ids->strides[1]] =
          h->ids[d * h->capacity + g];
  }
}

static int64_t movedRows(struct Halo *h) {
  int64_t moved = 0;
  for (int64_t d = 0; d < h->layout->domains; ++d)
    for (int64_t g = 0; g < h->rows[d]; ++g)
      moved += h->sourceDomain[d * h->capacity + g] != d;
  return moved;
}

#ifdef MDIR_ENABLE_MPI
/* Pack and consume a peer's stream in the same (domain, ghost, component)
   order on both sides. For the transpose, interleave local and remote
   contributions in that order too, preserving the oracle's sum. */
static struct TransferPlan *transferPlan(struct Halo *h, int transpose) {
  struct TransferPlan *plan = &h->plans[transpose];
  if (plan->counts) return plan;
  int peers = mdrtMPISize(), rank = mdrtMPIRank();
  int64_t domains = h->layout->domains;
  plan->counts = calloc((size_t)peers * 2, sizeof(int));
  int *owners = malloc(domains * sizeof *owners);
  if (!owners || !plan->counts) mdrtMPIAbort("out of memory planning halo");
  for (int64_t d = 0; d < domains; ++d) owners[d] = mdrtMPIOwner(domains, d);
  for (int pass = 0; pass < 2; ++pass) {
    size_t ns = 0, nr = 0, nl = 0;
    for (int64_t d = 0; d < domains; ++d)
      for (int64_t g = 0; g < h->rows[d]; ++g) {
        int owner = owners[h->sourceDomain[d*h->capacity+g]];
        int from = transpose ? owners[d] : owner, to = transpose ? owner : owners[d];
        if (from == rank && to != rank) {
          if (pass) plan->send[ns] = (struct TransferRow){d, g, to};
          else if (plan->counts[to] == INT_MAX) mdrtMPIAbort("halo count exceeds MPI int");
          else ++plan->counts[to];
          ++ns;
        }
        if (to != rank) continue;
        if (!transpose && from == rank) {
          if (pass) plan->local[nl] = (struct TransferRow){d, g, from};
          ++nl;
        } else {
          if (pass) plan->receive[nr] = (struct TransferRow){d, g, from};
          else if (from != rank) {
            if (plan->counts[peers+from] == INT_MAX) mdrtMPIAbort("halo count exceeds MPI int");
            ++plan->counts[peers+from];
          }
          ++nr;
        }
      }
    if (!pass) {
      plan->sends = ns; plan->receives = nr; plan->locals = nl;
      plan->send = malloc(ns * sizeof *plan->send);
      plan->receive = malloc(nr * sizeof *plan->receive);
      plan->local = malloc(nl * sizeof *plan->local);
      if ((ns && !plan->send) || (nr && !plan->receive) || (nl && !plan->local))
        mdrtMPIAbort("out of memory storing halo schedule");
    }
  }
  free(owners);
  return plan;
}

#ifdef MDIR_ENABLE_NCCL
int64_t mdrtDistDeviceMatrixSlot(int64_t halo, double cutoff) {
  struct Halo *h = (struct Halo *)(intptr_t)halo;
  if (h->deviceCutoff && h->deviceCutoff != cutoff) return 0;
  h->deviceCutoff = cutoff;
  return (int64_t)(intptr_t)&h->deviceMatrix;
}

static struct NCCLPlan *devicePlan(struct Halo *h) {
  if (!h->ncclPlan)
    h->ncclPlan = mdrtNCCLPlan(h->layout->rows[mdrtMPIDomainBound(h->layout->domains,0)], h->layout->domains, h->rows, h->capacity, NULL,
        h->sourceDomain, h->sourceRow, h->image, h->layout->edges);
  return h->ncclPlan;
}

/* Scalar pair inputs are immutable values. A halo retains the last
   input at each copy site. Decide reuse collectively, since all ranks
   must participate in a miss. */
struct ScalarHaloField {
  struct ScalarHaloField *next;
  struct Memref3D input;
  int64_t site;
  void *data;
};

static void discardScalar(struct ScalarHaloField *s) {
  if (!s->data) return;
  mdrtNCCLFreePositions(s->data);
  _mlir_ciface_mdrtDistReleaseField(&s->input);
}

static void *scalarHalo(struct Halo *h, struct Memref3D *owned, int64_t site,
                        int narrow, int64_t *count) {
  struct ScalarHaloField **p = &h->scalarFields;
  while (*p && (*p)->site != site) p = &(*p)->next;
  if (!*p) {
    *p = calloc(1,sizeof **p);
    if (!*p) fail("out of memory recording scalar halo");
    (*p)->site = site;
  }
  struct ScalarHaloField *s = *p;
  int64_t d = mdrtMPIDomainBound(h->layout->domains,0), n = h->layout->rows[d];
  *count = n+h->rows[d];
  if (mdrtMPIAllTrue(s->data && s->input.allocated == owned->allocated))
    return s->data;
  discardScalar(s);
  s->input = *owned;
  _mlir_ciface_mdrtDistRetainField(owned);
  if (observer)
    for (size_t i=0; i<h->edgeCount; ++i) {
      struct Edge e=h->edges[i];
      observer->transfer(observer->context,e.from,e.to,e.rows*sizeof(double),0);
    }
  double *packed = mdrtNCCLPackedRows(devicePlan(h),deviceRows(owned,d,n,0),n,1,0);
  if (narrow) {
    s->data = mdrtNCCLAllocate((size_t)*count*sizeof(int32_t));
    mdrtNCCLNarrow(packed,s->data,*count);
    mdrtNCCLFreePositions(packed);
  } else {
    s->data = packed;
  }
  return s->data;
}

void _mlir_ciface_mdrtDistDeviceScalar(struct Memref1D *out, int64_t handle,
    struct Memref3D *owned, int64_t site) {
  int64_t n;
  double *value = scalarHalo((void *)(intptr_t)handle,owned,site,0,&n);
  *out = (struct Memref1D){value,value,0,{n},{1}};
}

struct DeviceKinds { int32_t *allocated, *aligned; int64_t offset, rows, stride; };
/* The epoch fixes row identities and periodic images. Keep the immutable
   topology adjacency; the device selects its active incidences at the
   current coordinates in the original tuple order. */
struct TuplePlan {
  struct TuplePlan *next;
  int64_t site, extent;
  int count, arity, rows, built;
  int *topology, *locations, *particles, *starts, *entries;
  int *mapped, *blocks, *csr;
};

static void freeTuplePlans(struct TuplePlan *p) {
  while (p) {
    struct TuplePlan *next = p->next;
    int *buffers[] = {p->topology,p->locations,p->particles,p->starts,p->entries,
                      p->mapped,p->blocks,p->csr};
    for (size_t i=0; i<sizeof buffers/sizeof *buffers; ++i)
      mdrtNCCLFreePositions((double *)buffers[i]);
    free(p);
    p=next;
  }
}

static int *tupleInts(size_t count) {
  int *values=calloc(count ? count : 1,sizeof(*values));
  if (!values) fail("out of memory indexing distributed tuples");
  return values;
}

static struct TuplePlan *newTuplePlan(struct Layout *l, struct Halo *h,
    struct Members *members, int64_t site, int64_t d) {
  int64_t count=members->sizes[0], arity=members->sizes[1];
  int64_t owned=l->rows[d], rows=owned+(h ? h->rows[d] : 0);
  if (arity<2 || arity>INT_MAX-2 || count<0 || count>INT_MAX/arity ||
      rows>INT_MAX-256 || l->count>INT_MAX/3 || rows>INT_MAX/5 ||
      l->stride>INT_MAX/l->domains)
    fail("distributed tuple map exceeds device index range");
  int64_t memberships=count*arity;
  struct TuplePlan *p=calloc(1,sizeof(*p));
  if (!p) fail("out of memory allocating distributed tuple plan");
  p->site=site; p->count=count; p->arity=arity; p->rows=rows;
  p->extent=rows+2+(arity+2)*(memberships+1);
  int *locations=tupleInts(3*l->count), *particles=tupleInts(5*rows);
  for (int64_t j=0; j<l->count; ++j) {
    locations[3*j]=(int)(mdrtMPIOwner(l->domains,l->owner[j])*l->stride+l->row[j]);
    locations[3*j+1]=-1;
    locations[3*j+2]=(int)l->owner[j];
  }
  for (int r=rows; r-- > 0;) {
    int64_t g=d*(h ? h->capacity : 0)+r-owned;
    int j=r<owned ? l->ids[d*l->stride+r] : h->ids[g];
    particles[5*r]=j;
    particles[5*r+1]=locations[3*j+1];
    locations[3*j+1]=r;
    if (r>=owned)
      for (int k=0; k<3; ++k) particles[5*r+2+k]=h->image[3*g+k];
  }
  int *topology=tupleInts(memberships), *starts=tupleInts(l->count+1);
  for (int t=0;t<count;++t)
    for (int a=0;a<arity;++a) {
      int j=members->aligned[members->offset+t*members->strides[0]+a*members->strides[1]];
      if (j<0 || j>=l->count) fail("tuple member outside particle set");
      topology[t*arity+a]=j;
      ++starts[j+1];
    }
  for (int j=0;j<l->count;++j) starts[j+1]+=starts[j];
  int *cursor=tupleInts(l->count), *entries=tupleInts(memberships);
  memcpy(cursor,starts,l->count*sizeof(*cursor));
  for (int m=0;m<memberships;++m) entries[cursor[topology[m]]++]=m;
  p->locations=mdrtNCCLUploadIndices(locations,3*l->count);
  p->particles=mdrtNCCLUploadIndices(particles,5*rows);
  p->topology=mdrtNCCLUploadIndices(topology,memberships);
  p->starts=mdrtNCCLUploadIndices(starts,l->count+1);
  p->entries=mdrtNCCLUploadIndices(entries,memberships);
  free(locations); free(particles); free(topology); free(starts); free(entries); free(cursor);
  p->mapped=(int *)mdrtNCCLAllocate(memberships*sizeof(int));
  p->blocks=(int *)mdrtNCCLAllocate(((rows+255)/256)*sizeof(int));
  p->csr=(int *)mdrtNCCLAllocate(p->extent*sizeof(int));
  return p;
}

static double *tupleCoordinates(struct Halo *h, struct Memref3D *x, int64_t d) {
  struct Layout *l=h->layout;
  if (h->tupleInput.allocated!=x->allocated) {
    if (!h->tupleCoordinates)
      h->tupleCoordinates=mdrtNCCLAllocate(3*l->stride*l->domains*sizeof(double));
    if (h->tupleInput.allocated) _mlir_ciface_mdrtDistReleaseField(&h->tupleInput);
    h->tupleInput=*x;
    _mlir_ciface_mdrtDistRetainField(x);
    mdrtNCCLGatherCoordinates(h->tupleCoordinates,deviceRows(x,d,l->rows[d],0),l->rows[d],l->stride);
  }
  return h->tupleCoordinates;
}

void _mlir_ciface_mdrtDistTupleIncidence(struct DeviceKinds *out,int64_t handle,
    struct Memref3D *x,struct Members *members,int64_t site,int64_t group,void *stream) {
  struct Halo *h=group ? NULL : (void *)(intptr_t)handle;
  struct Layout *l=group ? (void *)(intptr_t)handle : h->layout;
  int64_t d=mdrtMPIDomainBound(l->domains,0);
  struct TuplePlan **slot=group ? &l->tuplePlans : &h->tuplePlans;
  while (*slot && (*slot)->site!=site) slot=&(*slot)->next;
  if (!*slot) *slot=newTuplePlan(l,h,members,site,d);
  struct TuplePlan *p=*slot;
  if (!group || !p->built) {
    double *coordinates=group ? NULL : tupleCoordinates(h,x,d);
    mdrtNCCLBuildTupleRows(p->topology,p->count,p->arity,p->locations,p->particles,coordinates,
        l->grid,l->edges,d,group,p->starts,p->entries,p->rows,p->mapped,p->blocks,p->csr);
    p->built=1;
  }
  mdrtNCCLWaitForRows(stream);
  *out=(struct DeviceKinds){p->csr,p->csr,0,p->extent,1};
}

static void residentHalo(struct Halo *h,struct Memref3D *owned,struct Memref3D *ghosts,
                         int image,int reverse,struct Memref3D *initial) {
  int64_t d=mdrtMPIDomainBound(h->layout->domains,0), n=h->layout->rows[d];
  int64_t k=owned->sizes[2];
  double *x=deviceRows(owned,d,n,reverse), *g=deviceRows(ghosts,d,h->rows[d],!reverse);
  if (initial) mdrtNCCLJoin(x,deviceRows(initial,d,n,0),n*k,NULL,0);
  if (observer)
    for (size_t i=0;i<h->edgeCount;++i) {
      struct Edge e=h->edges[i];
      observer->transfer(observer->context,reverse ? e.to : e.from,reverse ? e.from : e.to,
                         e.rows*k*sizeof(double),reverse);
    }
  mdrtNCCLResidentHalo(devicePlan(h),x,n,g,k,image,reverse);
}

void _mlir_ciface_mdrtDistResidentCopy(int64_t handle,struct Memref3D *owned,
                                        struct Memref3D *ghosts,int64_t image) {
  residentHalo((void *)(intptr_t)handle,owned,ghosts,image,0,NULL);
}

void _mlir_ciface_mdrtDistResidentSum(int64_t handle,struct Memref3D *initial,
                                       struct Memref3D *ghosts,struct Memref3D *out) {
  residentHalo((void *)(intptr_t)handle,out,ghosts,0,1,initial);
}

static double *joinTupleFields(int64_t d,int64_t n,int64_t g,struct Memref3D *owned,
                                struct Memref3D *ghost,int width,void *stream) {
  if (owned->sizes[2]!=width || ghost->sizes[2]!=width) fail("invalid tuple field width");
  double *result=mdrtNCCLAllocate((n+g)*width*sizeof(double));
  mdrtNCCLJoin(result,deviceRows(owned,d,n,0),n*width,deviceRows(ghost,d,g,0),g*width);
  mdrtNCCLWaitForRows(stream);
  return result;
}
void _mlir_ciface_mdrtDistTupleVectors(struct Memref2D *out,int64_t d,int64_t n,int64_t g,
    struct Memref3D *owned,struct Memref3D *ghost,void *stream) {
  double *data=joinTupleFields(d,n,g,owned,ghost,3,stream);
  *out=(struct Memref2D){data,data,0,{n+g,3},{3,1}};
}
void _mlir_ciface_mdrtDistTupleScalars(struct Memref1D *out,int64_t d,int64_t n,int64_t g,
    struct Memref3D *owned,struct Memref3D *ghost,void *stream) {
  double *data=joinTupleFields(d,n,g,owned,ghost,1,stream);
  *out=(struct Memref1D){data,data,0,{n+g},{1}};
}
void _mlir_ciface_mdrtDistTupleOwnedVectors(struct Memref2D *out,int64_t d,int64_t n,
    struct Memref3D *field,void *stream) {
  if (field->sizes[2]!=3) fail("invalid tuple vector field");
  double *data=deviceRows(field,d,n,0);
  mdrtNCCLWaitForRows(stream);
  *out=(struct Memref2D){data,data,0,{n,3},{3,1}};
}
void _mlir_ciface_mdrtDistTupleOwnedScalars(struct Memref1D *out,int64_t d,int64_t n,
    struct Memref3D *field,void *stream) {
  if (field->sizes[2]!=1) fail("invalid tuple scalar field");
  double *data=deviceRows(field,d,n,0);
  mdrtNCCLWaitForRows(stream);
  *out=(struct Memref1D){data,data,0,{n},{1}};
}
void _mlir_ciface_mdrtDistImportTupleForces(struct Memref3D *field,int64_t domain,
    int64_t rows,int64_t start,struct Memref2D *forces,void *stream) {
  if (forces->sizes[1]!=3 || start<0 || rows>forces->sizes[0]-start)
    fail("invalid distributed force slice");
  mdrtNCCLImportOnStream(deviceRows(field,domain,rows,1),at2(forces,start,0),
                         rows*3*sizeof(double),stream);
}
void _mlir_ciface_mdrtDistUploadTupleScalars(struct Memref1D *out,struct Memref1D *src) {
  if (src->strides[0]!=1) fail("expected contiguous tuple parameters");
  size_t bytes=src->sizes[0]*sizeof(double);
  double *data=mdrtNCCLAllocate(bytes);
  mdrtNCCLUpload(data,src->aligned+src->offset,bytes);
  *out=(struct Memref1D){data,data,0,{src->sizes[0]},{1}};
}
void _mlir_ciface_mdrtDistUploadTupleVectors(struct Memref2D *out,struct Memref2D *src) {
  if (src->strides[1]!=1 || src->strides[0]!=src->sizes[1]) fail("expected contiguous tuple parameters");
  size_t bytes=src->sizes[0]*src->sizes[1]*sizeof(double);
  double *data=mdrtNCCLAllocate(bytes);
  mdrtNCCLUpload(data,src->aligned+src->offset,bytes);
  *out=(struct Memref2D){data,data,0,{src->sizes[0],src->sizes[1]},{src->sizes[1],1}};
}
void _mlir_ciface_mdrtDistReleaseTupleVectors(struct Memref2D *view) {
  mdrtNCCLFreePositions(view->allocated);
}
void _mlir_ciface_mdrtDistReleaseTupleScalars(struct Memref1D *view) {
  mdrtNCCLFreePositions(view->allocated);
}

void _mlir_ciface_mdrtDistDeviceKinds(struct DeviceKinds *out, int64_t handle,
    struct Memref3D *owned, int64_t site) {
  int64_t n;
  int32_t *value = scalarHalo((void *)(intptr_t)handle,owned,site,1,&n);
  *out = (struct DeviceKinds){value,value,0,n,1};
}

static void packPositions(struct Memref2D *result, int64_t handle,
                          struct Memref3D *owned, int asynchronous) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  int64_t d = mdrtMPIDomainBound(h->layout->domains, 0);
  if (owned->sizes[2] != 3 || owned->strides[1] != 3 || owned->strides[2] != 1)
    fail("NCCL positions require contiguous triples");
  if (observer)
    for (size_t i = 0; i < h->edgeCount; ++i) {
      struct Edge edge = h->edges[i];
      observer->transfer(observer->context, edge.from, edge.to, edge.rows*3*sizeof(double), 0);
    }
  int64_t n = h->layout->rows[d];
  double *source = deviceRows(owned,d,n,0);
  double *data = asynchronous ? mdrtNCCLBeginPositions(devicePlan(h),source,n)
                              : mdrtNCCLPackedRows(devicePlan(h),source,n,3,1);
  *result = (struct Memref2D){data, data, 0, {h->layout->rows[d]+h->rows[d], 3}, {3, 1}};
}

void _mlir_ciface_mdrtDistDevicePositions(struct Memref2D *result, int64_t halo,
                                          struct Memref3D *owned) {
  packPositions(result,halo,owned,0);
}

void _mlir_ciface_mdrtDistBeginPositions(struct Memref2D *result, int64_t halo,
                                         struct Memref3D *owned) {
  packPositions(result,halo,owned,1);
}

int64_t mdrtDistFirstPairPhase(int64_t halo, double cutoff) {
  struct Halo *h = (void *)(intptr_t)halo;
  return !h->layout->groupCount && h->deviceMatrix.data && h->deviceCutoff == cutoff ? 0 : 2;
}

void _mlir_ciface_mdrtDistPairPositions(struct Memref2D *out, int64_t halo,
    struct Memref2D *full, struct Memref3D *owned, int64_t phase, void *stream) {
  struct Halo *h = (void *)(intptr_t)halo;
  int64_t d = mdrtMPIDomainBound(h->layout->domains,0), n = h->layout->rows[d];
  double *p = mdrtNCCLPairPositions(devicePlan(h),full->aligned,
                                    deviceRows(owned,d,n,0),n,phase,stream);
  *out = (struct Memref2D){p,p,0,{full->sizes[0],3},{3,1}};
}

void _mlir_ciface_mdrtDistCenterMask(struct DeviceKinds *out, int64_t halo, int64_t phase) {
  struct Halo *h = (void *)(intptr_t)halo;
  int64_t d = mdrtMPIDomainBound(h->layout->domains,0);
  int64_t rows = h->layout->rows[d]+h->rows[d];
  int *p = mdrtNCCLCenterMask(devicePlan(h),rows,phase);
  *out = (struct DeviceKinds){p,p,0,rows,1};
}

void _mlir_ciface_mdrtDistPairPlaces(struct DeviceKinds *out, int64_t halo,
    int64_t phase, int64_t slot, struct DeviceKinds *order, void *stream) {
  struct Halo *h = (void *)(intptr_t)halo;
  int64_t count;
  int *places = mdrtNCCLPairPlaces(devicePlan(h), order->aligned + order->offset,
      order->rows, phase, slot && h->deviceMatrix.data, stream, &count);
  *out = (struct DeviceKinds){places, places, 0, count, 1};
}

void _mlir_ciface_mdrtDistWriteCenters(struct Memref3D *out, int64_t d,
    int64_t n, struct Memref2D *forces, struct DeviceKinds *mask, void *stream) {
  mdrtNCCLWriteCenters(deviceRows(out,d,n,1),forces->aligned,mask->aligned,n,stream);
}

void _mlir_ciface_mdrtDistReleaseDevicePositions(struct Memref2D *rows) {
  mdrtNCCLFreePositions(rows->allocated);
}
#endif

static void exchangeHalo(struct Halo *h, struct Memref3D *owned,
                         struct Memref3D *ghosts, int image, int transpose,
                         struct Memref3W *ownedWords, struct Memref3W *ghostWords) {
#ifdef MDIR_ENABLE_NCCL
  if (mdrtNCCLActive()) {
    if (ownedWords || ghostWords) fail("NCCL requires floating-point halo payloads");
    if (owned->strides[2] != 1 || ghosts->strides[2] != 1 ||
        owned->strides[1] != owned->sizes[2] || ghosts->strides[1] != ghosts->sizes[2])
      fail("NCCL halo fields require contiguous rows");
    int64_t d = mdrtMPIDomainBound(h->layout->domains, 0);
    mdrtNCCLHalo(devicePlan(h), at3(owned,d,0,0), h->layout->rows[d],
                 at3(ghosts,d,0,0), owned->sizes[2], image, transpose);
    return;
  }
#endif
  const int peers = mdrtMPISize(), rank = mdrtMPIRank();
  struct TransferPlan *plan = transferPlan(h, transpose);
  int *counts = calloc((size_t)peers * 6, sizeof(int));
  if (!counts) mdrtMPIAbort("out of memory packing halo counts");
  int *ns = counts, *nr = ns + peers, *os = nr + peers, *or = os + peers;
  int *ws = or + peers, *wr = ws + peers;
  int64_t components = ownedWords ? ownedWords->sizes[2] : owned->sizes[2];
  for (int p = 0; p < 2*peers; ++p) {
    if (components && plan->counts[p] > INT_MAX / components)
      mdrtMPIAbort("halo count exceeds MPI int");
    counts[p] = plan->counts[p] * components;
  }
  int sendSize = 0, recvSize = 0;
  for (int p = 0; p < peers; ++p) {
    os[p] = ws[p] = sendSize;
    or[p] = wr[p] = recvSize;
    if (ns[p] > INT_MAX - sendSize || nr[p] > INT_MAX - recvSize)
      mdrtMPIAbort("halo displacement exceeds MPI int");
    sendSize += ns[p];
    recvSize += nr[p];
  }
  size_t bytes = ownedWords ? sizeof(DistWide) : sizeof(double);
  void *send = malloc((size_t)(sendSize ? sendSize : 1) * bytes);
  void *recv = malloc((size_t)(recvSize ? recvSize : 1) * bytes);
  if (!send || !recv) mdrtMPIAbort("out of memory packing halo rows");
  for (int pass = 0; pass < 3; ++pass) {
    if (pass == 1 && peers > 1) {
      if (!transpose) mdrtDistPhase(DistInter);
      mdrtMPIExchange(send, ns, os, recv, nr, or, ownedWords != NULL);
      if (!transpose) mdrtDistPhase(-DistInter);
    }
    size_t length = pass == 0 ? plan->sends : pass == 1 ? plan->receives : plan->locals;
    if (!length) continue;
    struct TransferRow *rows = pass == 0 ? plan->send : pass == 1 ? plan->receive : plan->local;
    int phase = pass == 0 ? DistPack : pass == 1 ? DistInter : DistIntra;
    if (!transpose) mdrtDistPhase(phase);
    for (size_t k = 0; k < length; ++k) {
      int64_t d = rows[k].domain, g = rows[k].ghost, i = d*h->capacity+g;
      int64_t dom = h->sourceDomain[i], row = h->sourceRow[i];
      int peer = rows[k].peer;
      for (int64_t c = 0; c < components; ++c) {
        if (ownedWords) {
          DistWide *from = wordAt(ghostWords, d, g, c);
          if (!pass) ((DistWide *)send)[ws[peer]++] = *from;
          else {
            DistWide term = peer == rank ? *from : ((DistWide *)recv)[wr[peer]++];
            DistWide *to = wordAt(ownedWords, dom, row, c);
            *to = mdrtDistAddWide(*to, term);
          }
          continue;
        }
        double *a = at3(owned, dom, row, c), *b = at3(ghosts, d, g, c);
        if (!pass) ((double *)send)[ws[peer]++] = transpose ? *b : *a;
        else {
          double v = peer == rank ? (transpose ? *b : *a) : ((double *)recv)[wr[peer]++];
          if (transpose) *a += v;
          else {
            if (image) v += (double)h->image[3*i+c] * h->layout->edges[c];
            *b = v;
          }
        }
      }
    }
    if (!transpose) mdrtDistPhase(-phase);
  }
  free(send); free(recv); free(counts);
}
#endif

void _mlir_ciface_mdrtDistHaloSumWords(int64_t handle, struct Memref3W *owned,
    struct Memref3W *ghosts, struct Memref3D *out) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  int64_t components = owned->sizes[2];
  if (observer)
    for (size_t i = 0; i < h->edgeCount; ++i) {
      struct Edge edge = h->edges[i];
      observer->transfer(observer->context, edge.to, edge.from,
                         edge.rows * components * (int64_t)sizeof(DistWide), 1);
    }
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi)
    exchangeHalo(h, NULL, NULL, 0, 1, owned, ghosts);
  else
#endif
  for (int64_t d = 0; d < h->layout->domains; ++d)
    for (int64_t g = 0; g < h->rows[d]; ++g) {
      int64_t slot = d * h->capacity + g;
      for (int64_t c = 0; c < components; ++c) {
        DistWide *to = wordAt(owned, h->sourceDomain[slot], h->sourceRow[slot], c);
        *to = mdrtDistAddWide(*to, *wordAt(ghosts, d, g, c));
      }
    }
  int64_t first = 0, past = h->layout->domains;
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi) {
    first = mdrtMPIDomainBound(past, 0);
    past = mdrtMPIDomainBound(past, 1);
  }
#endif
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < h->layout->rows[d]; ++r)
      for (int64_t c = 0; c < components; ++c)
        *at3(out, d, r, c) = (double)mdrtDistNarrowWord(*wordAt(owned, d, r, c)) * 0x1p-40;
}

/* G: (G a)_g = a_{s(g)}, with `image` x_{s(g)} + n_g L. */
void _mlir_ciface_mdrtDistHaloCopy(int64_t handle, struct Memref3D *owned,
                                   struct Memref3D *ghosts, int64_t image) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (observer)
    for (size_t i = 0; i < h->edgeCount; ++i) {
      struct Edge edge = h->edges[i];
      observer->transfer(observer->context, edge.from, edge.to,
                         edge.rows * owned->sizes[2] * (int64_t)sizeof(double), 0);
    }
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi) {
    exchangeHalo(h, owned, ghosts, image, 0, NULL, NULL);
    return;
  }
#endif
  int64_t k = owned->sizes[2];
  for (int64_t d = 0; d < h->layout->domains; ++d)
    for (int64_t g = 0; g < h->rows[d]; ++g) {
      int64_t s = d * h->capacity + g;
      for (int64_t c = 0; c < k; ++c) {
        double v = *at3(owned, h->sourceDomain[s], h->sourceRow[s], c);
        if (image)
          v += (double)h->image[3 * s + c] * h->layout->edges[c];
        *at3(ghosts, d, g, c) = v;
      }
    }
  if (tracing())
    fprintf(stderr,
            "mdrt_dist: halo_copy of %lld components: %lld rows between "
            "domains\n",
            (long long)k, (long long)movedRows(h));
}

/* a + G^T b: each ghost's value added at the owned row of its source, the
   ghosts in a fixed order. */
void _mlir_ciface_mdrtDistHaloSum(int64_t handle, struct Memref3D *ghosts,
                                  struct Memref3D *owned) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (observer)
    for (size_t i = 0; i < h->edgeCount; ++i) {
      struct Edge edge = h->edges[i];
      observer->transfer(observer->context, edge.to, edge.from,
                         edge.rows * owned->sizes[2] * (int64_t)sizeof(double), 1);
    }
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi) {
    exchangeHalo(h, owned, ghosts, 0, 1, NULL, NULL);
    return;
  }
#endif
  int64_t k = owned->sizes[2];
  for (int64_t d = 0; d < h->layout->domains; ++d)
    for (int64_t g = 0; g < h->rows[d]; ++g) {
      int64_t s = d * h->capacity + g;
      for (int64_t c = 0; c < k; ++c)
        *at3(owned, h->sourceDomain[s], h->sourceRow[s], c) +=
            *at3(ghosts, d, g, c);
    }
  if (tracing())
    fprintf(stderr,
            "mdrt_dist: halo_sum of %lld components: %lld rows between "
            "domains\n",
            (long long)k, (long long)movedRows(h));
}

void mdrtDistRetainHalo(int64_t handle) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (!h || h->references <= 0)
    fail("retain of a released halo");
  ++h->references;
}

/* The cell a state is used with must be the one its halo's images and its
   layout's regions were made in (D399); this slice has
   no change of the cell on domains. */
void mdrtDistCheckHaloCell(int64_t handle, double lx, double ly, double lz) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (!h || h->references <= 0)
    fail("use of a released halo");
  double edges[3] = {lx, ly, lz};
  for (int k = 0; k < 3; ++k)
    if (!(edges[k] == h->layout->edges[k]))
      fail("a state is used with another cell than its halo's: a change of "
           "the cell on domains needs a new layout");
}

void mdrtDistReleaseHalo(int64_t handle) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (!h || h->references <= 0)
    fail("release of a released halo");
  if (--h->references > 0)
    return;
  mdrtDistReleaseLayout((int64_t)(intptr_t)h->layout);
  while (h->matrices) {
    struct DomainMatrix *m = h->matrices;
    h->matrices = m->next;
    mdrtHostMatrixRelease(m->raw);
    mdrtHostMatrixRelease(m->selected);
    free(m->counts);
    free(m->selectedCounts);
    free(m->reference);
    free(m->shifts);
    free(m);
  }
#ifdef MDIR_ENABLE_NCCL
  freeTuplePlans(h->tuplePlans);
  if (h->tupleInput.allocated) _mlir_ciface_mdrtDistReleaseField(&h->tupleInput);
  if (h->tupleCoordinates) mdrtNCCLFreePositions(h->tupleCoordinates);
  while (h->scalarFields) {
    struct ScalarHaloField *s = h->scalarFields;
    h->scalarFields = s->next;
    discardScalar(s);
    free(s);
  }
  if (h->deviceMatrix.data) h->deviceMatrix.release(h->deviceMatrix.data);
  mdrtNCCLFreePlan(h->ncclPlan);
#endif
  free(h->rows);
  free(h->edges);
  free(h->firstGhost);
  free(h->nextGhost);
#ifdef MDIR_ENABLE_MPI
  for (int i = 0; i < 2; ++i) {
    free(h->plans[i].counts);
    free(h->plans[i].send);
    free(h->plans[i].receive);
    free(h->plans[i].local);
  }
#endif
  free(h->sourceDomain);
  free(h->sourceRow);
  free(h->ids);
  free(h->image);
  free(h);
}

/* A native integrator passes replicated particle coordinates at each force
   evaluation. Keep the layout and explicit images for the Verlet epoch.
   The caller scopes this object to one loop with invariant group relations. */
struct TrajectoryEpoch {
  struct Layout *layout;
  struct Halo *halo;
  double *reference;
};

int64_t mdrtDistEpochCreate(void) {
  struct TrajectoryEpoch *e = calloc(1, sizeof *e);
  if (!e) fail("out of memory creating a trajectory epoch");
  return (int64_t)(intptr_t)e;
}

void mdrtDistEpochDestroy(int64_t cache) {
  struct TrajectoryEpoch *e = (struct TrajectoryEpoch *)(intptr_t)cache;
  if (e->halo) mdrtDistReleaseHalo((int64_t)(intptr_t)e->halo);
  if (e->layout) mdrtDistReleaseLayout((int64_t)(intptr_t)e->layout);
  free(e->reference);
  free(e);
}

int64_t _mlir_ciface_mdrtDistEpochPartition(struct Memref2D *x, int64_t gx,
    int64_t gy, int64_t gz, double lx, double ly, double lz,
    int64_t cache, double skin, int64_t mpi) {
  struct TrajectoryEpoch *e = (struct TrajectoryEpoch *)(intptr_t)cache;
  double box[] = {lx, ly, lz};
  int64_t n = x->sizes[0];
  if (e->layout) {
    struct Layout *l = e->layout;
    int valid = l->count == n && l->edges[0] == lx && l->edges[1] == ly && l->edges[2] == lz;
    for (int64_t j = 0; valid && j < n; ++j) {
      double r2 = 0;
      for (int c = 0; c < 3; ++c) {
        double dx = *at2(x,j,c) - e->reference[3*j+c];
        dx -= box[c]*floor(dx/box[c] + 0.5);
        r2 += dx*dx;
      }
      valid = r2 <= 0.25*skin*skin;
    }
    if (valid) {
      l->groupsFrozen = 1;
      mdrtDistRetainLayout((int64_t)(intptr_t)l);
      return (int64_t)(intptr_t)l;
    }
    if (e->halo) mdrtDistReleaseHalo((int64_t)(intptr_t)e->halo);
    e->halo = NULL;
    mdrtDistReleaseLayout((int64_t)(intptr_t)e->layout);
    e->layout = NULL;
  }
  e->reference = realloc(e->reference, (n ? n : 1)*3*sizeof(double));
  if (!e->reference) fail("out of memory recording trajectory positions");
  for (int64_t j = 0; j < n; ++j)
    for (int c = 0; c < 3; ++c)
      e->reference[3*j+c] = *at2(x,j,c);
  int64_t handle;
#ifdef MDIR_ENABLE_MPI
  if (mpi) handle = _mlir_ciface_mdrtDistMPIPartition(x,gx,gy,gz,lx,ly,lz);
  else
#endif
    handle = _mlir_ciface_mdrtDistPartition(x,gx,gy,gz,lx,ly,lz);
  e->layout = (struct Layout *)(intptr_t)handle;
  mdrtDistRetainLayout(handle);
  return handle;
}

void _mlir_ciface_mdrtDistEpochImages(int64_t cache, int64_t layout,
                                      struct Memref3D *x) {
  struct TrajectoryEpoch *e = (struct TrajectoryEpoch *)(intptr_t)cache;
  if (!e->halo) return;
  struct Layout *l = (struct Layout *)(intptr_t)layout;
  int64_t first = 0, past = l->domains;
#ifdef MDIR_ENABLE_MPI
  if (l->mpi) {
    first = mdrtMPIDomainBound(l->domains, 0);
    past = mdrtMPIDomainBound(l->domains, 1);
  }
#endif
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->rows[d]; ++r) {
      int64_t j = l->ids[d*l->stride+r];
      for (int c = 0; c < 3; ++c) {
        double len = l->edges[c], ref = e->reference[3*j+c];
        ref -= len*floor(ref/len);
        double *v = at3(x,d,r,c);
        *v -= len*floor((*v-ref)/len + 0.5);
      }
    }
}

int64_t _mlir_ciface_mdrtDistEpochHalo(int64_t layout, struct Memref3D *x,
                                       double reach, int64_t cache, int64_t midpoint) {
  struct TrajectoryEpoch *e = (struct TrajectoryEpoch *)(intptr_t)cache;
  if (!e->halo) {
    e->halo = (struct Halo *)(intptr_t)buildHalo(layout,x,reach,midpoint);
    e->halo->layout->groupsFrozen = 1;
  }
  int64_t handle = (int64_t)(intptr_t)e->halo;
  mdrtDistRetainHalo(handle);
  return handle;
}

/* The precondition under which the relation of md_dist is that of
   md.neighborhood: at most one image of a particle within the cutoff of
   another, a cutoff less than half the least edge. */
void mdrtDistCheckCutoff(double cutoff, double lx, double ly, double lz) {
  double least = lx < ly ? (lx < lz ? lx : lz) : (ly < lz ? ly : lz);
  if (!(2.0 * cutoff < least))
    fail("the cutoff is not less than half the least edge of the cell");
}
