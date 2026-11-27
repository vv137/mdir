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
#include "mdrt_host_matrix.h"
#include "mdrt_midpoint.h"
#ifdef MDIR_ENABLE_MPI
#include "mdrt_mpi.h"
#endif
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef MDIR_ENABLE_MPI
#include <time.h>
#endif

struct Memref1I {
  int64_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
struct Memref2I {
  int64_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
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
  return &m->aligned[m->offset + i * m->strides[0] + j * m->strides[1]];
}
static double *at3(struct Memref3D *m, int64_t d, int64_t r, int64_t c) {
  return m->aligned + m->offset + d * m->strides[0] + r * m->strides[1] +
         c * m->strides[2];
}

static _Thread_local struct DistObserver *observer;

#ifdef MDIR_ENABLE_MPI
static void resetBalance(void);
#endif

void mdrtDistObserve(struct DistObserver *value) {
#ifdef MDIR_ENABLE_MPI
  resetBalance();
#endif
  observer = value;
}

int mdrtDistObserving(void) { return observer != NULL; }

void mdrtDistMeshBytes(enum DistMeshTraffic kind, int64_t bytes) {
  if (observer)
    observer->meshBytes(observer->context, kind, bytes);
}

#ifdef MDIR_ENABLE_MPI
struct BalanceClock {
  double force, wait, started, waitStarted, tuples, tupleStarted;
  int steps, phase;
};
static _Thread_local struct BalanceClock *balanceClock;

static void measureBalance(int64_t phase) {
  struct BalanceClock *b = balanceClock;
  if (!b) return;
  if (phase == DistStep) ++b->steps;
  if (phase != DistPairs && phase != -DistPairs &&
      phase != DistOther && phase != -DistOther &&
      phase != DistWait && phase != -DistWait &&
      phase != DistTupleWork && phase != -DistTupleWork) return;
  struct timespec stamp;
  clock_gettime(CLOCK_MONOTONIC, &stamp);
  double now = stamp.tv_sec+1.e-9*stamp.tv_nsec;
  if (phase == DistTupleWork) b->tupleStarted = now;
  else if (phase == -DistTupleWork) b->tuples += now-b->tupleStarted;
  else if (phase == DistWait) b->waitStarted = now;
  else if (phase == -DistWait) {
    double elapsed = now-b->waitStarted;
    b->wait += elapsed;
    if (b->phase) b->force -= elapsed;
  } else if (phase > 0) {
    b->started = now;
    b->phase = phase;
  } else if (b->phase == -phase) {
    b->force += now-b->started;
    b->phase = 0;
  }
}
#endif

void mdrtDistPhase(int64_t phase) {
#ifdef MDIR_ENABLE_MPI
  measureBalance(phase);
#endif
  if (observer && phase != DistTupleWork && phase != -DistTupleWork)
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
  int64_t idBase, *globalIds;
  int64_t references;
  int64_t grid[3], domains, count, stride;
  double edges[3];
  double *cuts[3];
#ifdef MDIR_ENABLE_MPI
  struct TupleDonation {
    int destination;
    double fraction;
  } *donations;
#endif
  struct ConstraintGroup {
    int32_t atoms[4];
    int size;
    int32_t relationCenter;
    int64_t tuple;
  } *groups, *localGroups;
  size_t groupCount;
  int groupsFrozen, identityGroups;
  struct Layout *groupSource;
  int64_t *centerGroup; /* Group index plus one at each center. */
  double (*migratedPositions)[3];
  int32_t *groupCenters; /* center + 1; zero for a free atom */
#ifdef MDIR_ENABLE_MPI
  struct InitialAtom *initial;
  struct Collection *collection;
  struct TupleSchedule *groupSchedules;
  int *migrationCounts; /* Send/receive counts, then displacements. */
  int64_t *sendSlots, *sourceSlots;
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
  struct MidpointBounds *bounds;
  struct MidpointSite { double position[3], shift[3]; } *midpointSites;
  int32_t *counts, *selectedCounts;
  int valid;
  int64_t candidateRule, candidates;
  int32_t *candidateCounts;
  uint64_t exclusionVersion;
  int candidatesValid, midpointList;
  int buffered;
  struct BufferedRow { int32_t stable, selected; } *bufferedRows;
  double *pruneReference;
  int32_t maxRawCount;
};

struct TransferRow { int64_t domain, ghost, offset; int peer; int64_t source,row; };
struct GatherRow { int64_t slot, offset; };

#ifdef MDIR_ENABLE_MPI
struct TransferPlan {
  int *counts;
  struct TransferRow *send, *receive, *local;
  size_t sends, receives, locals, onNode;
};
#endif

struct GhostLink { int64_t domain,ghost,source,row; };
struct Halo {
  int64_t idBase;
  int64_t references;
  struct Layout *layout;
  double reach;
  struct DomainMatrix *matrices;
  int64_t gatherSlots;
  unsigned char *needsPeer;
  size_t *gatherOffsets;
  struct GatherRow *gatherRows;
#ifdef MDIR_ENABLE_MPI
  struct TransferPlan plans[2];
  double tupleMargin;
  struct Exclusions *exclusions;
  uint64_t directoryVersion;
  struct GhostLink *links;
  int64_t *linkOffsets;
#endif
  int64_t capacity; /* the most ghosts of one domain */
  int64_t *rows;    /* [domains] */
  /* [domains * capacity] each: the source's owner, its owned row, its
     identity, and the image n (3 per ghost). */
  struct Edge {
    int64_t from, to, rows;
  } *edges;
  size_t edgeCount;
  int64_t *sourceDomain, *sourceRow, *ids, *image;
};

/* Input tuple members use particle-set indices even after a reorder. The
   symmetric key removes both orientations from the cutoff relation. */
/* Matrix entries depend only on this immutable relation and their local
   rows. A new relation may reuse a released relation's address. */
static _Atomic uint64_t lastExclusionVersion;
struct Exclusions {
  uint64_t version;
  int64_t directoryExtent;
  size_t references, count;
  uint64_t pairs[];
};

static const int64_t *domainIds(const struct Layout *l, int64_t d) {
  return l->globalIds ? l->globalIds+d*l->stride : l->ids+(d-l->idBase)*l->stride;
}

struct Members {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};

static int compareKeys(const void *lhs, const void *rhs) {
  uint64_t a = *(const uint64_t *)lhs, b = *(const uint64_t *)rhs;
  return (a > b) - (a < b);
}

static struct Exclusions *newExclusions(size_t count) {
  if (count > (SIZE_MAX - sizeof(struct Exclusions)) / sizeof(uint64_t))
    fail("exclusion table too large");
  struct Exclusions *e = malloc(sizeof *e + count * sizeof *e->pairs);
  if (!e) fail("out of memory building exclusions");
  e->version = atomic_fetch_add_explicit(&lastExclusionVersion, 1, memory_order_relaxed) + 1;
  if (!e->version) fail("exclusion relation version exhausted");
  e->directoryExtent = 0;
  e->references = 1;
  e->count = count;
  return e;
}

int64_t _mlir_ciface_mdrtDistExclusions(struct Members *members) {
  size_t count = members->sizes[0];
  struct Exclusions *e = newExclusions(count);
  for (size_t r = 0; r < count; ++r) {
    const int32_t *row = members->aligned + members->offset + r * members->strides[0];
    uint32_t i = row[0], j = row[members->strides[1]];
    e->pairs[r] = i < j ? ((uint64_t)i << 32) | j : ((uint64_t)j << 32) | i;
  }
  qsort(e->pairs, count, sizeof *e->pairs, compareKeys);
  return (int64_t)(intptr_t)e;
}

int64_t mdrtDistPairAllowed(int64_t handle, int64_t i, int64_t j) {
  const struct Exclusions *e = (const struct Exclusions *)(intptr_t)handle;
  uint64_t key = ((uint64_t)i << 32) | (uint32_t)j;
  return !bsearch(&key, e->pairs, e->count, sizeof key, compareKeys);
}

void mdrtDistFreeExclusions(int64_t handle) {
  struct Exclusions *e = (void *)(intptr_t)handle;
  if (e && !--e->references) free(e);
}

static const char *const quantityNames[DistQuantityCount] = {
    "pair energy or virial",
    "force or gathered field",
    "particle sum",
    "kinetic energy",
    "momentum",
    "sum of F^2/m (the kinetic correction)"};

static const char *quantityName(int64_t quantity) {
  return quantity >= 0 && quantity < DistQuantityCount ? quantityNames[quantity]
                                                       : "quantity";
}

/* Word conversion is defined even for a singular pair or huge finite input:
   the rounded value is checked before the cast. */
int64_t mdrtDistWord(double x, double scale, int64_t quantity) {
  double w = round(x * scale);
  if (!(w >= -0x1p63 && w < 0x1p63)) {
    char message[160];
    snprintf(message, sizeof message,
             "deterministic fixed-point %s term outside signed 64-bit range",
             quantityName(quantity));
    fail(message);
  }
  return (int64_t)w;
}

/* At most 2^32 particles: N(N-1) directed visits of words below 2^63 then
   bound every subtotal below 2^127. */
void mdrtDistCheckWordCount(int64_t count) {
  if (count < 0 || count > INT64_C(4294967296))
    fail("deterministic pair count exceeds the signed 128-bit accumulation "
         "bound");
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
      double x = r < owned ? *at3(xo, d, r, k) : *at3(xg, d, r - owned, k);
      double shift = shifted && r >= owned ? *at3(shifts, d, r - owned, k) : 0;
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
      double n = floor((hi[k] / width) - (lo[k] / width)) + 1;
      if (n > count) {
        bins = INFINITY;
        break;
      }
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
      double x = r < owned ? *at3(xo, d, r, k) : *at3(xg, d, r - owned, k);
      if (shifted && r >= owned)
        x += *at3(shifts, d, r - owned, k);
      bin = bin * shape[k] + (int64_t)floor(x / width - lo[k] / width);
    }
    cell[r] = bin;
    ++offset[bin + 1];
  }
  for (int64_t b = 1; b <= nbin; ++b)
    offset[b] += offset[b - 1];
  /* Fill backward. offset[b+1] then points at the start of bin b. */
  for (int64_t r = count; r-- > 0;)
    rows[--offset[cell[r] + 1]] = r;
  for (int64_t i = 0; i < centers; ++i) {
    int64_t x = cell[i] % shape[0], yz = cell[i] / shape[0];
    int64_t y = yz % shape[1], z = yz / shape[1];
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy) {
        int64_t begin = 0, end = 0;
        if (y + dy >= 0 && y + dy < shape[1] && z + dz >= 0 &&
            z + dz < shape[2]) {
          int64_t slab = ((z + dz) * shape[1] + y + dy) * shape[0];
          int64_t first = slab + (x > 0 ? x - 1 : x);
          int64_t last = slab + (x + 1 < shape[0] ? x + 1 : x);
          begin = offset[first + 1];
          end = last + 1 == nbin ? count : offset[last + 2];
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

int64_t mdrtDistNarrowWord(DistWide value, int64_t quantity) {
  if (value < INT64_MIN || value > INT64_MAX) {
    char message[160];
    snprintf(message, sizeof message,
             "deterministic fixed-point %s exceeds signed 64-bit range",
             quantityName(quantity));
    fail(message);
  }
  return (int64_t)value;
}

void _mlir_ciface_mdrtDistImageShifts(int64_t handle, struct Memref3D *shifts) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (!h || !shifts || shifts->sizes[0] != h->layout->domains ||
      shifts->sizes[2] != 3)
    fail("deterministic pair geometry requires a complete halo image field");
  int64_t first = 0, past = h->layout->domains;
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi) {
    first = mdrtMPIDomainBound(h->layout->domains,0);
    past = mdrtMPIDomainBound(h->layout->domains,1);
  }
#endif
  for (int64_t d = first; d < past; ++d)
    if (shifts->sizes[1] < h->rows[d])
      fail("missing halo rows for deterministic pair geometry");
  #pragma omp parallel for collapse(2) schedule(static) if((past-first)*h->capacity > 512)
  for (int64_t d = first; d < past; ++d)
    for (int64_t g = 0; g < h->capacity; ++g) {
      if (g >= h->rows[d]) continue;
      for (int c = 0; c < 3; ++c)
        *at3(shifts,d,g,c) =
            (double)h->image[3*((d-h->idBase)*h->capacity+g)+c] * h->layout->edges[c];
    }
}

struct Memref3W {
  DistWide *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};

static DistWide *wordAt(struct Memref3W *m, int64_t d, int64_t r, int64_t c) {
  return m->aligned + m->offset + d * m->strides[0] + r * m->strides[1] +
         c * m->strides[2];
}

static int64_t layoutDivision(const struct Layout *l, int axis, double u) {
  return cutDivision(u,l->grid[axis],l->cuts[axis]);
}

static void copyCuts(struct Layout *to, double *const from[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    if (!from[axis]) continue;
    size_t bytes = (to->grid[axis]+1)*sizeof(double);
    to->cuts[axis] = malloc(bytes);
    if (!to->cuts[axis]) fail("out of memory copying domain boundaries");
    memcpy(to->cuts[axis],from[axis],bytes);
  }
}

/* xi is the untranslated position of the lesser identity. Form its midpoint
   before choosing a grid. Of all common translations of this pair, use
   exactly the one that puts the midpoint in the primary cell. */
struct ImageView {
  int64_t *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};

void _mlir_ciface_mdrtDistHaloImageWords(struct ImageView *view, int64_t handle) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  *view = (struct ImageView){h->image,h->image,-3*h->idBase*h->capacity,{h->layout->domains,h->capacity,3},
                             {3*h->capacity,3,1}};
}

/* The rank visits its domains serially. The ordinary compiled builder and
   row kernels parallelize the work inside each visit. The halo fixes row
   identities and images, but current explicit-image positions still have
   to satisfy the Verlet displacement bound. */
int64_t mdrtDistMatrix(int64_t handle, int64_t domain, int64_t centers,
                       int64_t rows, double cutoff, double skin, int64_t width, int64_t midpoint) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (!h || domain < 0 || domain >= h->layout->domains ||
      rows != h->layout->rows[domain] + h->rows[domain] ||
      (centers != h->layout->rows[domain] && centers != rows))
    fail("domain matrix extents do not match its halo");
  struct DomainMatrix *m = h->matrices;
  while (m && !(m->domain == domain && m->centers == centers &&
                m->cutoff == cutoff && m->skin == skin && m->midpointList == midpoint))
    m = m->next;
  if (!m) {
    m = calloc(1, sizeof *m);
    if (!m) fail("out of memory allocating a domain matrix");
    m->next = h->matrices;
    h->matrices = m;
    m->domain = domain;
    m->rows = rows;
    m->centers = centers;
    m->midpointList = midpoint;
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
  m->candidatesValid = 0;
  return 1;
}

/* Packed coordinates already include the halo translation. Within this
   open Cartesian box the common-image shifts and periodic cell searches
   of a full-system list are unnecessary. Only the lower identity can
   own the center of a midpoint pair. */
void _mlir_ciface_mdrtDistBuildMidpoint(int64_t halo, int64_t matrix, struct Memref2D *x) {
  struct Halo *h = (struct Halo *)(intptr_t)halo;
  struct DomainMatrix *m = (struct DomainMatrix *)(intptr_t)matrix;
  struct Layout *l = h->layout;
  int64_t n = m->rows, centers = m->centers, d = m->domain, owned = l->rows[d];
  if (!centers) return;
  if (n > INT32_MAX) fail("too many explicit images in a pair list");
  double reach = m->cutoff + m->skin + 3.e-6*(m->box[0]+m->box[1]+m->box[2]);
  double limit2 = reach*reach;
  int64_t *order = malloc(n*sizeof *order);
  int64_t (*spans)[18] = malloc(centers*sizeof *spans);
  struct SortedSite { double x[3]; int32_t row, id; };
  struct SortedSite *sites = malloc(n*sizeof *sites);
  if (!order || !spans || !sites) fail("out of memory binning midpoint rows");
  struct Memref3D positions = {x->allocated,x->aligned,x->offset,{1,n,3},{0,x->strides[0],x->strides[1]}};
  struct Memref1I permutation = {order,order,0,{n},{1}};
  struct Memref1I ranges = {&spans[0][0],&spans[0][0],0,{18*centers},{1}};
  pairCells(0,n,n,centers,reach,&positions,NULL,NULL,0,&permutation,&ranges);
  const int64_t *ownerIds = domainIds(l,d);
  const int64_t *ghostIds = h->ids+(d-h->idBase)*h->capacity;
  #pragma omp parallel for schedule(static) if(n > 1024)
  for (int64_t q = 0; q < n; ++q) {
    int64_t j = order[q];
    sites[q].row = j;
    sites[q].id = j < owned ? ownerIds[j] : ghostIds[j-owned];
    for (int k = 0; k < 3; ++k) sites[q].x[k] = *at2(x,j,k);
  }
  for (;;) {
    struct HostBuffer2 entries;
    _mlir_ciface_mdrtHostMatrixEntries(&entries,m->raw);
    int32_t largest = 0;
    #pragma omp parallel for schedule(static,8) reduction(max:largest) if(centers > 256)
    for (int64_t i = 0; i < centers; ++i) {
      int32_t *row = (int32_t *)entries.aligned+i*entries.strides[0];
      int64_t id = i < owned ? ownerIds[i] : ghostIds[i-owned];
      double xi = *at2(x,i,0), yi = *at2(x,i,1), zi = *at2(x,i,2);
      int32_t count = 0;
      for (int slab = 0; slab < 9; ++slab)
        for (int64_t q = spans[i][2*slab]; q < spans[i][2*slab+1]; ++q) {
          if (sites[q].id <= id) continue;
          double dx = xi-sites[q].x[0], dy = yi-sites[q].x[1], dz = zi-sites[q].x[2];
          if (dx*dx+dy*dy+dz*dz > limit2) continue;
          if (count < entries.sizes[1]) row[count] = sites[q].row;
          ++count;
        }
      m->counts[i] = count;
      if (count > largest) largest = count;
    }
    if (largest <= entries.sizes[1]) {
      m->maxRawCount = largest > 0 ? largest : 1;
      break;
    }
    mdrtHostMatrixGrow(m->raw,largest);
  }
  free(sites); free(spans); free(order);
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
        m->shifts[3*(n+g)+c] = h->image[3*((d-h->idBase)*h->capacity+g)+c]*h->layout->edges[c];
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

static size_t firstKey(const struct Exclusions *e, uint64_t key) {
  size_t first = 0, end = e->count;
  while (first < end) {
    size_t middle = first + (end-first)/2;
    if (e->pairs[middle] < key) first = middle+1;
    else end = middle;
  }
  return first;
}

static void rowRanges(const struct Layout *l, const int64_t cell[3],
                      const struct MidpointRange common[3][3], const int64_t image[3],
                      struct MidpointRange ranges[3]) {
  for (int c = 0; c < 3; ++c)
    ranges[c] = image[c] >= -1 && image[c] <= 1 ? common[c][image[c]+1] :
        midpointCutRange(l->edges[c],image[c],l->grid[c],cell[c],l->cuts[c]);
}

static inline double distanceToRow(const double center[3], struct Memref2D *x, int64_t row) {
  double r2 = 0.0;
  for (int c = 0; c < 3; ++c) {
    double dx = center[c] - *at2(x,row,c);
    r2 += dx*dx;
  }
  return r2;
}

/* An omitted stable-owner neighbor cannot enter the physical cutoff while
   each explicit position has moved by at most half the inner skin. */
static int refreshPruneReference(struct DomainMatrix *m, struct Memref2D *x,
                                 double skin, int rebuild) {
  double limit = 0.25*skin*skin;
  if (!rebuild) {
    #pragma omp parallel for schedule(static) reduction(|:rebuild) if(m->rows > 512)
    for (int64_t i = 0; i < m->rows; ++i)
      rebuild |= !(distanceToRow(m->pruneReference+3*i,x,i) <= limit);
  }
  if (rebuild) {
    #pragma omp parallel for schedule(static) if(m->rows > 512)
    for (int64_t i = 0; i < m->rows; ++i)
      for (int c = 0; c < 3; ++c)
        m->pruneReference[3*i+c] = *at2(x,i,c);
  }
  return rebuild;
}

int64_t _mlir_ciface_mdrtDistFilterMatrix(int64_t handle, int64_t matrix,
    int64_t exclusions, int64_t rule, struct Memref3D *owned, struct Memref3D *ghost,
    struct Memref2D *packed, int64_t prune) {
  if (!rule && !exclusions) return 0;
  struct DomainMatrix *m = (struct DomainMatrix *)(intptr_t)matrix;
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  struct Layout *l = h->layout;
  int64_t d = m->domain, n = l->rows[d];
  struct HostBuffer2 src, dst;
  _mlir_ciface_mdrtHostMatrixEntries(&src,m->raw);
  int64_t width = m->maxRawCount ? m->maxRawCount : src.sizes[1];
  if (!m->selected) {
    m->selected = mdrtHostMatrixAllocate(m->centers,width);
    m->selectedCounts = calloc(m->centers ? m->centers : 1,sizeof *m->selectedCounts);
    if (!m->selectedCounts) fail("out of memory allocating selected row counts");
  }
  mdrtHostMatrixGrow(m->selected,width);
  _mlir_ciface_mdrtHostMatrixEntries(&dst,m->selected);
  struct HostBuffer2 candidates = dst;
  int32_t *counts = m->selectedCounts;
  if (rule == 2) {
    if (!m->candidates) {
      m->candidates = mdrtHostMatrixAllocate(m->centers,width);
      m->candidateCounts = calloc(m->centers ? m->centers : 1,sizeof *m->candidateCounts);
      if (!m->candidateCounts) fail("out of memory allocating midpoint row counts");
    }
    mdrtHostMatrixGrow(m->candidates,width);
    _mlir_ciface_mdrtHostMatrixEntries(&candidates,m->candidates);
    counts = m->candidateCounts;
  }
  const struct Exclusions *e = (const struct Exclusions *)(intptr_t)exclusions;
  uint64_t version = e ? e->version : 0;
  int64_t cell[3] = {d%l->grid[0], (d/l->grid[0])%l->grid[1], d/(l->grid[0]*l->grid[1])};
  double cutoff2 = m->cutoff*m->cutoff;
  double outer2 = cutoff2*(1.0 + 16.0*FLT_EPSILON);
  prune = prune && cutoff2 > FLT_MIN && outer2 < FLT_MAX;
  double innerSkin = 0.25*m->skin;
  double radius = m->cutoff + innerSkin;
  double buffered2 = radius*radius*(1.0 + 32.0*FLT_EPSILON);
  int buffered = rule == 2 && prune && innerSkin > 0 && isfinite(buffered2);
  int rebuild = !m->candidatesValid || m->candidateRule != rule ||
                version != m->exclusionVersion || buffered != m->buffered;
  if (buffered && !m->bufferedRows) {
    m->bufferedRows = calloc(m->centers ? m->centers : 1,sizeof *m->bufferedRows);
    m->pruneReference = calloc(m->rows ? m->rows : 1,3*sizeof *m->pruneReference);
    if (!m->bufferedRows || !m->pruneReference) fail("out of memory buffering midpoint neighbors");
  }
  double coordinateBound = 1.0;
  struct MidpointRange common[3][3];
  if (rule == 2) {
    for (int c = 0; c < 3; ++c)
      for (int image = -1; image <= 1; ++image)
        common[c][image+1] = midpointCutRange(l->edges[c],image,l->grid[c],cell[c],l->cuts[c]);
    if (!m->bounds) {
      m->bounds = calloc(m->rows ? m->rows : 1, sizeof *m->bounds);
      m->midpointSites = calloc(m->rows ? m->rows : 1, sizeof *m->midpointSites);
      if (!m->bounds || !m->midpointSites)
        fail("out of memory bounding midpoint candidates");
      rebuild = 1;
    }
    /* The ambiguous-neighbor scan reads these unshifted positions many
       times. Populate them while checking the bounds, once per row. */
    #pragma omp parallel for reduction(|:rebuild) schedule(static) if(m->rows > 512)
    for (int64_t i = 0; i < m->rows; ++i)
      for (int c = 0; c < 3; ++c) {
        double x = i<n ? *at3(owned,d,i,c) : *at3(ghost,d,i-n,c);
        m->midpointSites[i].position[c] = x;
        rebuild |= !(x >= m->bounds[i].low[c] && x <= m->bounds[i].high[c]);
      }
    if (rebuild) {
      #pragma omp parallel for schedule(static) reduction(max:coordinateBound) if(m->rows > 512)
      for (int64_t i = 0; i < m->rows; ++i)
        for (int c = 0; c < 3; ++c) {
          double x = m->midpointSites[i].position[c];
          m->bounds[i].low[c] = x - 0.5*m->skin;
          m->bounds[i].high[c] = x + 0.5*m->skin;
          int64_t image = i<n ? 0 : h->image[3*((d-h->idBase)*h->capacity+i-n)+c];
          m->bounds[i].image[c] = image;
          m->bounds[i].shift[c] = image*l->edges[c];
          m->midpointSites[i].shift[c] = m->bounds[i].shift[c];
          coordinateBound = fmax(coordinateBound, fabs(m->bounds[i].low[c]));
          coordinateBound = fmax(coordinateBound, fabs(m->bounds[i].high[c]));
          coordinateBound = fmax(coordinateBound, fabs(m->bounds[i].shift[c]));
        }
    }
  }
  if (rebuild) {
    /* Each intermediate has magnitude below eight times the coordinate
       bound. Include the additions at both enclosing endpoints. For
       extreme scales, defer the decision to the per-evaluation test. */
    double padding = midpointPadding(coordinateBound);
    #pragma omp parallel for schedule(static,8) if(m->centers > 256)
    for (int64_t i = 0; i < m->centers; ++i) {
      int64_t idi = i<n ? domainIds(l,d)[i] : h->ids[(d-h->idBase)*h->capacity+i-n];
      uint64_t prefix = (uint64_t)idi << 32;
      size_t lo = e && rule ? firstKey(e,prefix) : 0;
      size_t hi = e && rule ? firstKey(e,prefix+(UINT64_C(1)<<32)) : 0;
      struct MidpointRange ranges[3];
      if (rule == 2) rowRanges(l,cell,common,m->bounds[i].image,ranges);
      int32_t count = 0, uncertain = 0;
      int32_t *candidateRow = (int32_t *)candidates.aligned+i*candidates.strides[0];
      int32_t *scratch = (int32_t *)dst.aligned+i*dst.strides[0];
      for (int32_t k = 0; k < m->counts[i]; ++k) {
        int32_t j = ((int32_t *)src.aligned)[i*src.strides[0]+k];
        int64_t idj = j<n ? domainIds(l,d)[j] : h->ids[(d-h->idBase)*h->capacity+j-n];
        if (rule && idi >= idj) continue;
        if (e && rule) {
          uint64_t key = prefix | (uint32_t)idj;
          size_t t = lo;
          while (t < hi && e->pairs[t] < key) ++t;
          if (t < hi && e->pairs[t] == key) continue;
        }
        if (e && !rule && !mdrtDistPairAllowed(exclusions,idi<idj ? idi : idj,
                                                idi<idj ? idj : idi)) continue;
        if (rule == 2) {
          int membership = classifyMidpoint(&m->bounds[i],&m->bounds[j],ranges,padding);
          if (membership < 0) continue;
          if (buffered && !membership) {
            scratch[uncertain++] = j;
            continue;
          }
          if (membership > 0 && !buffered) j = ~j;
        }
        candidateRow[count++] = j;
      }
      if (buffered) {
        m->bufferedRows[i].stable = count;
        memcpy(candidateRow+count,scratch,uncertain*sizeof *scratch);
      }
      counts[i] = count+uncertain;
    }
    m->exclusionVersion = version;
    m->candidateRule = rule;
    m->buffered = buffered;
    m->candidatesValid = 1;
  }
  if (rule != 2) return 1;

  /* The packed coordinates are the same explicit images the pair kernel
     reads. Leave a guard shell for rounding the displacement, products,
     and sum to binary32. The kernel still applies its own cutoff. */
  int refresh = buffered && refreshPruneReference(m,packed,innerSkin,rebuild);
  #pragma omp parallel for schedule(static,8) if(m->centers > 256)
  for (int64_t i = 0; i < m->centers; ++i) {
    int32_t *selected = (int32_t *)dst.aligned+i*dst.strides[0];
    const int32_t *neighbors = (int32_t *)candidates.aligned+i*candidates.strides[0];
    double center[3];
    for (int c = 0; c < 3; ++c) center[c] = *at2(packed,i,c);
    int32_t begin = 0, count = 0;
    if (buffered) {
      begin = m->bufferedRows[i].stable;
      if (refresh) {
        for (int32_t k = 0; k < begin; ++k)
          if (!(distanceToRow(center,packed,neighbors[k]) > buffered2))
            selected[count++] = neighbors[k];
        m->bufferedRows[i].selected = count;
      } else {
        count = m->bufferedRows[i].selected;
      }
      if (begin == counts[i]) {
        m->selectedCounts[i] = count;
        continue;
      }
    }
    const struct MidpointSite *centerSite = m->midpointSites+i;
    struct MidpointRange ranges[3];
    rowRanges(l,cell,common,m->bounds[i].image,ranges);
    for (int32_t k = begin; k < counts[i]; ++k) {
      int32_t j = neighbors[k];
      if (prune) {
        int32_t neighbor = j < 0 ? ~j : j;
        if (distanceToRow(center,packed,neighbor) > outer2) continue;
      }
      if (j < 0) {
        selected[count++] = ~j;
        continue;
      }
      int keep = 1;
      const struct MidpointSite *neighborSite = m->midpointSites+j;
      for (int c = 0; c < 3; ++c) {
        double xi = centerSite->position[c];
        double dx = (xi-neighborSite->position[c])-
                    (neighborSite->shift[c]-centerSite->shift[c]);
        double midpoint = xi-0.5*dx;
        if (!(midpoint >= ranges[c].lower && midpoint < ranges[c].upper)) {
          keep = 0;
          break;
        }
      }
      if (keep) selected[count++] = j;
    }
    m->selectedCounts[i] = count;
  }
  return 1;
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
    owner = owner * l->grid[k] + layoutDivision(l,k,u-floor(u));
  }
  return owner;
}

static void numberRows(struct Layout *l, const int64_t *ids, int64_t count) {
  /* Each block owns its counters. The scan assigns disjoint ranges of
     each domain in global identity order, independent of thread count. */
  const int64_t width = l->domains > 2048 ? l->domains : 2048;
  const int64_t blocks = (count + width - 1) / width;
  int64_t *offset = calloc((size_t)(blocks ? blocks : 1), l->domains * sizeof *offset);
  if (!offset) fail("out of memory numbering owned rows");
  #pragma omp parallel for schedule(static) if(blocks > 1)
  for (int64_t b = 0; b < blocks; ++b) {
    int64_t end = (b+1)*width < count ? (b+1)*width : count;
    for (int64_t j = b*width; j < end; ++j)
      ++offset[b*l->domains + l->owner[ids ? ids[j] : j]];
  }
  int64_t cap = 1;
  for (int64_t d = 0; d < l->domains; ++d) {
    int64_t sum = 0;
    for (int64_t b = 0; b < blocks; ++b) {
      int64_t *v = &offset[b*l->domains+d], count = *v;
      *v = sum;
      sum += count;
    }
    l->rows[d] = sum;
    if (sum > cap) cap = sum;
  }
#ifdef MDIR_ENABLE_MPI
  if (ids) {
    mdrtMPIShareDomainCounts(l->rows, l->domains);
    for (int64_t d = 0; d < l->domains; ++d)
      if (l->rows[d] > cap) cap = l->rows[d];
  }
#endif
  if (l->stride != cap) {
    l->ids = realloc(l->ids,l->domains*cap*sizeof *l->ids);
    if (!l->ids) fail("out of memory indexing owned rows");
    l->stride = cap;
  }
  #pragma omp parallel for schedule(static) if(blocks > 1)
  for (int64_t b = 0; b < blocks; ++b) {
    int64_t end = (b+1)*width < count ? (b+1)*width : count;
    for (int64_t j = b*width; j < end; ++j) {
      int64_t id = ids ? ids[j] : j;
      int64_t d = l->owner[id], r = offset[b*l->domains+d]++;
      l->ids[d * cap + r] = id;
      l->row[id] = r;
    }
  }
  free(offset);
}

#ifdef MDIR_ENABLE_MPI
static void materializeRows(struct Layout *l);
#else
#define materializeRows(l) ((void)0)
#endif

#ifdef MDIR_ENABLE_MPI
static void releaseCollection(struct Collection *p);
#endif

#ifdef MDIR_ENABLE_MPI
static void releaseGroupSchedules(struct TupleSchedule *head);
#endif
void mdrtDistReleaseLayout(int64_t handle);

#ifdef MDIR_ENABLE_MPI
/* Extending a registered group relation is a topology update. Rebuild
   the mutable registry from owned centers before adding the new relation. */
static void assignGroups(struct Layout *, struct Memref2D *, struct Members *, int);

static void collectGroups(struct Layout *l) {
  if (!l->localGroups) return;
  l->groups = malloc(l->groupCount*sizeof *l->groups);
  l->centerGroup = calloc(l->count,sizeof *l->centerGroup);
  l->groupCenters = calloc(l->count,sizeof *l->groupCenters);
  if (!l->groups || !l->centerGroup || !l->groupCenters) fail("out of memory extending group registry");
  size_t used = 0;
  for (int64_t d = 0; d < l->domains; ++d) {
    int owner = mdrtMPIOwner(l->domains,d);
    int64_t entries = 0;
    if (owner == mdrtMPIRank())
      for (int64_t r = 0; r < l->rows[d]; ++r) {
        struct ConstraintGroup g = l->localGroups[(d-l->idBase)*l->stride+r];
        if (g.size > 0) l->groups[used+entries++] = g;
      }
    mdrtMPIBroadcastIndices(&entries,1,owner);
    for (int64_t i = 0; i < entries; ++i) {
      struct ConstraintGroup *g = l->groups+used++;
      int64_t record[7] = {0};
      if (owner == mdrtMPIRank()) {
        record[0] = g->size;
        record[5] = g->relationCenter;
        record[6] = g->tuple;
        for (int a = 0; a < g->size; ++a) record[a+1] = g->atoms[a];
      }
      mdrtMPIBroadcastIndices(record,7,owner);
      g->relationCenter = record[5];
      g->tuple = record[6];
      g->size = record[0];
      for (int a = 0; a < g->size; ++a) {
        g->atoms[a] = record[a+1];
        l->groupCenters[g->atoms[a]] = record[1]+1;
      }
      l->centerGroup[g->atoms[0]] = used;
    }
  }
  free(l->localGroups);
  l->localGroups = NULL;
  free(l->ids);
  l->ids = l->globalIds;
  l->globalIds = NULL;
  l->idBase = 0;
}
#endif

void _mlir_ciface_mdrtDistGroupOwners(int64_t handle, struct Memref2D *positions,
                                      struct Members *members) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  size_t count = members->sizes[0];
  int64_t size = members->sizes[1];
  if (!count || l->groupsFrozen) return;
#ifdef MDIR_ENABLE_MPI
  if (l->initial) { assignGroups(l,positions,members,0); return; }
#endif
  materializeRows(l);
#ifdef MDIR_ENABLE_MPI
  collectGroups(l);
#endif
  if (size < 2 || size > 4 || count > (size_t)l->count ||
      count > SIZE_MAX / sizeof *l->groups - l->groupCount)
    fail("invalid constraint group dimensions");
  if (l->groupSource) {
    struct Layout *source = l->groupSource;
    l->groups = malloc(l->groupCount*sizeof *l->groups);
    l->centerGroup = malloc(l->count*sizeof *l->centerGroup);
    l->groupCenters = malloc(l->count*sizeof *l->groupCenters);
    if (!l->groups || !l->centerGroup || !l->groupCenters)
      fail("out of memory extending migrated constraint groups");
    memcpy(l->groups,source->groups,l->groupCount*sizeof *l->groups);
    memcpy(l->centerGroup,source->centerGroup,l->count*sizeof *l->centerGroup);
    memcpy(l->groupCenters,source->groupCenters,l->count*sizeof *l->groupCenters);
    l->groupSource = NULL;
    mdrtDistReleaseLayout((int64_t)(intptr_t)source);
  }
  if (!l->groupCenters)
    l->groupCenters = calloc(l->count, sizeof *l->groupCenters);
  if (!l->centerGroup) l->centerGroup = calloc(l->count, sizeof *l->centerGroup);
  l->groups = realloc(l->groups, (l->groupCount + count) * sizeof *l->groups);
  if (!l->groupCenters || !l->centerGroup || !l->groups) fail("out of memory partitioning constraint groups");
  for (size_t row = 0; row < count; ++row) {
    struct ConstraintGroup *g = &l->groups[l->groupCount++];
    g->size = size;
    g->tuple = row;
    g->relationCenter = members->aligned[members->offset];
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
    }
    l->centerGroup[g->atoms[0]] = l->groupCount;
  }
  #pragma omp parallel for schedule(static) if(count > 256)
  for (size_t row = 0; row < count; ++row) {
    struct ConstraintGroup *g = &l->groups[l->groupCount-count+row];
    double points[4][3];
    for (int a = 0; a < size; ++a)
      for (int k = 0; k < 3; ++k)
        points[a][k] = *at2(positions, g->atoms[a], k);
    int64_t owner = groupOwner(l, g, points);
    for (int a = 0; a < size; ++a)
      l->owner[g->atoms[a]] = owner;
  }
  numberRows(l, NULL, l->count);
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
      cell[k] = layoutDivision(l,k,u-floor(u));
    }
    int64_t d = cell[0] + gx * (cell[1] + gy * cell[2]);
    l->owner[j] = d;
  }
  numberRows(l, NULL, l->count);
  return (int64_t)(intptr_t)l;
}

#ifdef MDIR_ENABLE_MPI
int64_t _mlir_ciface_mdrtDistMPIPartition(struct Memref2D *x, int64_t gx,
    int64_t gy, int64_t gz, double lx, double ly, double lz) {
  mdrtMPIDomainBound(gx * gy * gz, 0);
  int64_t handle = _mlir_ciface_mdrtDistPartition(x, gx, gy, gz, lx, ly, lz);
  ((struct Layout *)(intptr_t)handle)->mpi = 1;
  return handle;
}

/* The lowering passes a dense allocation. Transfer only live rows.
   Replicated ownership metadata keeps record order independent of rank count. */
static void shareRows(struct Layout *l, const int64_t *rows, struct Memref3D *field) {
  if (!l->mpi || mdrtMPISize() == 1)
    return;
  materializeRows(l);
  if (field->strides[2] != 1 || field->strides[1] != field->sizes[2])
    mdrtMPIAbort("expected contiguous field rows");
  // Gather compact live rows once. The owner of each domain fills its
  // original interval, so all consumers see the same doubles in row order.
  int peers = mdrtMPISize(), rank = mdrtMPIRank();
  int64_t width = field->sizes[2];
  if (l->count && width > 0 && l->count <= INT_MAX / width) {
    int *counts = calloc(2*(size_t)peers, sizeof *counts);
    double *packed = malloc((size_t)l->count*width*sizeof *packed);
    if (!counts || !packed) fail("out of memory collecting domain rows");
    int *offsets = counts + peers;
    int offset = 0;
    for (int64_t d = 0; d < l->domains; ++d) {
      int owner = mdrtMPIOwner(l->domains,d);
      int length = rows[d]*width;
      if (owner == rank && length)
        memcpy(packed+offset, at3(field,d,0,0), length*sizeof *packed);
      counts[owner] += length;
      offset += length;
    }
    for (int p = 1; p < peers; ++p) offsets[p] = offsets[p-1]+counts[p-1];
    mdrtMPIShareBlocks(packed,counts,offsets,0);
    offset = 0;
    for (int64_t d = 0; d < l->domains; ++d) {
      int length = rows[d]*width;
      if (mdrtMPIOwner(l->domains,d) != rank && length)
        memcpy(at3(field,d,0,0), packed+offset, length*sizeof *packed);
      offset += length;
    }
    free(packed);
    free(counts);
    return;
  }
  for (int64_t d = 0; d < l->domains; ++d)
    if (rows[d])
      mdrtMPIBroadcast(at3(field, d, 0, 0), rows[d] * field->sizes[2],
                       mdrtMPIOwner(l->domains, d));
}

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
  materializeRows(l);
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
  if (partial->strides[0] == partial->sizes[1] && partial->strides[1] == 1 &&
      mdrtMPIShareDomainPartials(partial->aligned + partial->offset, partial->sizes[0],
                                 partial->sizes[1], 1))
    return;
  for (int64_t d = 0; d < partial->sizes[0]; ++d)
    mdrtMPIBroadcastWords(partial->aligned + partial->offset + d * partial->strides[0],
                          partial->sizes[1], mdrtMPIOwner(partial->sizes[0], d));
}

void _mlir_ciface_mdrtMPISharePartials(struct Memref2D *partial) {
  if (partial->strides[0] == partial->sizes[1] && partial->strides[1] == 1 &&
      mdrtMPIShareDomainPartials(partial->aligned + partial->offset, partial->sizes[0],
                                 partial->sizes[1], 0))
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
  materializeRows(l);
  for (int64_t d = 0; d < l->domains; ++d) {
    rows->aligned[rows->offset + d * rows->strides[0]] = l->rows[d];
    for (int64_t r = 0; r < l->rows[d]; ++r)
      ids->aligned[ids->offset + d * ids->strides[0] + r * ids->strides[1]] =
          domainIds(l,d)[r];
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
                           .offset = -(h ? h->idBase : l->idBase)*stride,
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

void mdrtDistReleaseLayout(int64_t handle) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  if (!l || l->references <= 0)
    fail("release of a released layout");
  if (--l->references > 0)
    return;
#ifdef MDIR_ENABLE_MPI
  free(l->initial);
  releaseCollection(l->collection);
  releaseGroupSchedules(l->groupSchedules);
  free(l->donations);
  free(l->migrationCounts);
  free(l->sendSlots);
  free(l->sourceSlots);
#endif
  for (int axis = 0; axis < 3; ++axis) free(l->cuts[axis]);
  free(l->migratedPositions);
  free(l->localGroups);
  if (l->groupSource) {
    mdrtDistReleaseLayout((int64_t)(intptr_t)l->groupSource);
  } else {
    free(l->groups);
    free(l->groupCenters);
    free(l->centerGroup);
  }
  free(l->owner);
  free(l->row);
  free(l->rows);
  free(l->globalIds);
  free(l->ids);
  free(l);
}

/* A migration is a new ownership epoch. Prove that the source's identity
   table is a bijection before any field can be copied into that epoch. */
static void checkIdentities(struct Layout *l) {
  int64_t visited = 0, first = 0, past = l->domains;
#ifdef MDIR_ENABLE_MPI
  if (l->mpi) {
    first = mdrtMPIDomainBound(l->domains, 0);
    past = mdrtMPIDomainBound(l->domains, 1);
  }
#endif
  for (int64_t d = 0; d < l->domains; ++d) {
    if (l->rows[d] < 0 || l->rows[d] > l->stride)
      fail("invalid number of owned rows in a layout");
    visited += l->rows[d];
  }
  if (visited != l->count) fail("missing global particle identity in a layout");
  // The inverse map proves uniqueness. Each rank checks its own interval.
  int invalid = 0;
  #pragma omp parallel for collapse(2) reduction(|:invalid) schedule(static) if(l->count > 1024)
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->stride; ++r) {
      if (r >= l->rows[d]) continue;
      int64_t j = domainIds(l,d)[r];
      invalid |= j < 0 || j >= l->count ||
                 (l->owner && (l->owner[j] != d || l->row[j] != r)) ||
                 (r && domainIds(l,d)[r-1] >= j);
    }
  if (invalid) fail("layout identity and owner/row tables disagree");
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
/* The input slots are offsets in the old domain field. A negative slot in
   the output schedule denotes an incoming record in peer order. */
struct Crossing { int64_t id, domain; struct ConstraintGroup group; };

static struct ConstraintGroup rowGroup(struct Layout *l, int64_t d, int64_t r, int64_t id) {
  if (l->localGroups) return l->localGroups[(d-l->idBase)*l->stride+r];
  if (l->groupCenters && l->groupCenters[id]) {
    if (l->centerGroup[id]) return l->groups[l->centerGroup[id]-1];
    return (struct ConstraintGroup){.size=-1};
  }
  return (struct ConstraintGroup){0};
}
struct Arrival { int64_t domain, id, slot; };

static int compareArrival(const void *a, const void *b) {
  const struct Arrival *x = a, *y = b;
  if (x->domain != y->domain) return x->domain < y->domain ? -1 : 1;
  return (x->id > y->id) - (x->id < y->id);
}

static int64_t findOwnedRow(const struct Layout *l, int64_t d, int64_t id) {
  if (l->row) return l->row[id];
  const int64_t *ids = domainIds(l,d);
  int64_t lo = 0, hi = l->rows[d];
  while (lo < hi) {
    int64_t mid = lo+(hi-lo)/2;
    if (ids[mid] < id) lo = mid+1;
    else hi = mid;
  }
  if (lo == l->rows[d] || ids[lo] != id) fail("missing owned constraint member");
  return lo;
}

static void fillLocalField(struct Layout *old, struct Layout *next,
    struct Memref3D *source, struct Memref3D *dest, int image) {
  int peers = mdrtMPISize();
  int *ns = next->migrationCounts, *nr = ns+peers, *os = nr+peers, *or = os+peers;
  int sends = os[peers-1]+ns[peers-1], receives = or[peers-1]+nr[peers-1];
  int64_t width = source->sizes[2];
  if (width < 1 || width > INT_MAX/(int)sizeof(double)) fail("invalid migration field width");
  double *out = malloc((sends ? (size_t)sends : 1)*width*sizeof *out);
  double *in = malloc((receives ? (size_t)receives : 1)*width*sizeof *in);
  if (!out || !in) fail("out of memory transferring migrating fields");
  for (int i = 0; i < sends; ++i) {
    int64_t slot = next->sendSlots[i];
    for (int64_t k = 0; k < width; ++k)
      out[i*width+k] = *at3(source,slot/old->stride,slot%old->stride,k);
  }
  mdrtMPIExchangeRecords(out,ns,os,in,nr,or,width*sizeof(double));
  int64_t first = mdrtMPIDomainBound(next->domains,0), past = mdrtMPIDomainBound(next->domains,1);
  #pragma omp parallel for collapse(2) schedule(static) if((past-first)*next->stride > 1024)
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < next->stride; ++r) {
      if (r >= next->rows[d]) continue;
      int64_t local = (d-first)*next->stride+r, slot = next->sourceSlots[local];
      for (int64_t k = 0; k < width; ++k) {
        double v = slot < 0 ? in[(-1-slot)*width+k] :
                   *at3(source,slot/old->stride,slot%old->stride,k);
        if (image) {
          if (!isfinite(v)) fail("redistributed positions must be finite");
          v -= next->edges[k]*floor(v/next->edges[k]);
        }
        if (dest) *at3(dest,d,r,k) = v;
        else next->migratedPositions[local][k] = v;
      }
    }
  free(in);
  free(out);
}

/* Build the inverse identity maps only for consumers that require an
   explicitly global view. Migration and ordinary redistributions use the
   old local rows and the receive schedule directly. */
static void materializeRows(struct Layout *l) {
  if (!l->mpi || l->owner) return;
  int64_t n = l->count ? l->count : 1;
  l->owner = malloc(n*sizeof *l->owner);
  l->row = malloc(n*sizeof *l->row);
  if (!l->owner || !l->row) fail("out of memory gathering layout identities");
  l->globalIds = malloc(l->domains*l->stride*sizeof *l->globalIds);
  if (!l->globalIds) fail("out of memory gathering global rows");
  int64_t past = mdrtMPIDomainBound(l->domains,1);
  for (int64_t d = 0; d < l->domains; ++d) {
    int64_t *ids = l->globalIds+d*l->stride;
    if (d >= l->idBase && d < past)
      memcpy(ids,l->ids+(d-l->idBase)*l->stride,l->rows[d]*sizeof *ids);
    mdrtMPIBroadcastIndices(ids,l->rows[d],mdrtMPIOwner(l->domains,d));
    #pragma omp parallel for schedule(static) if(l->rows[d] > 2048)
    for (int64_t r = 0; r < l->rows[d]; ++r) {
      l->owner[ids[r]] = d;
      l->row[ids[r]] = r;
    }
  }
}

static void migrateLocal(struct Layout *old, struct Layout *next, struct Memref3D *x,
                         int64_t first, int64_t past, int keepPositions) {
  int peers = mdrtMPISize(), rank = mdrtMPIRank();
  int64_t capacity = (past-first)*old->stride;
  int64_t *targets = malloc((capacity ? capacity : 1)*sizeof *targets);
  next->migrationCounts = calloc(4*(size_t)peers,sizeof(int));
  if (!targets || !next->migrationCounts) fail("out of memory planning migration");
  int *ns = next->migrationCounts, *nr = ns+peers, *os = nr+peers, *or = os+peers;
  #pragma omp parallel for collapse(2) schedule(static) if(capacity > 1024)
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < old->stride; ++r) {
      if (r >= old->rows[d]) continue;
      int64_t id = domainIds(old,d)[r], cell[3];
      struct ConstraintGroup group = rowGroup(old,d,r,id);
      if (group.size) {
        if (group.size < 0) continue;
        const struct ConstraintGroup *g = &group;
        int64_t rows[4];
        double points[4][3];
        for (int a = 0; a < g->size; ++a) {
          rows[a] = findOwnedRow(old,d,g->atoms[a]);
          for (int k = 0; k < 3; ++k) {
            points[a][k] = *at3(x,d,rows[a],k);
            if (!isfinite(points[a][k]/next->edges[k])) fail("migration requires finite coordinates");
          }
        }
        int64_t domain = groupOwner(next,g,points);
        for (int a = 0; a < g->size; ++a) targets[(d-first)*old->stride+rows[a]] = domain;
      } else {
        for (int k = 0; k < 3; ++k) {
          double u = *at3(x,d,r,k)/next->edges[k];
          if (!isfinite(u)) fail("migration requires finite coordinates");
          cell[k] = layoutDivision(next,k,u-floor(u));
        }
        targets[(d-first)*old->stride+r] = cell[0]+next->grid[0]*(cell[1]+next->grid[1]*cell[2]);
      }
    }
  int64_t total = 0;
  for (int64_t d = first; d < past; ++d) {
    total += old->rows[d];
    for (int64_t r = 0; r < old->rows[d]; ++r) {
      int peer = mdrtMPIOwner(next->domains,targets[(d-first)*old->stride+r]);
      if (peer != rank) {
        if (ns[peer] == INT_MAX) fail("migration peer count exceeds MPI range");
        ++ns[peer];
      }
    }
  }
  mdrtMPIExchangeCounts(ns,nr);
  int sends = 0, receives = 0;
  for (int p = 0; p < peers; ++p) {
    os[p] = sends; or[p] = receives;
    if (ns[p] > INT_MAX-sends || nr[p] > INT_MAX-receives)
      fail("migration displacement exceeds MPI range");
    sends += ns[p]; receives += nr[p];
  }
  int64_t n = total-sends+receives;
  struct Arrival *rows = malloc((n ? n : 1)*sizeof *rows);
  struct Crossing *out = malloc((sends ? (size_t)sends : 1)*sizeof *out);
  struct Crossing *in = malloc((receives ? (size_t)receives : 1)*sizeof *in);
  next->sendSlots = malloc((sends ? (size_t)sends : 1)*sizeof *next->sendSlots);
  int *cursor = malloc(peers*sizeof *cursor);
  if (!rows || !out || !in || !next->sendSlots || !cursor) fail("out of memory routing migrants");
  memcpy(cursor,os,peers*sizeof *cursor);
  int64_t used = 0;
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < old->rows[d]; ++r) {
      int64_t slot = d*old->stride+r, id = domainIds(old,d)[r], target = targets[(d-first)*old->stride+r];
      int peer = mdrtMPIOwner(next->domains,target);
      if (rank == peer) rows[used++] = (struct Arrival){target,id,slot};
      else {
        int i = cursor[peer]++;
        out[i] = (struct Crossing){id,target,rowGroup(old,d,r,id)};
        next->sendSlots[i] = slot;
      }
    }
  mdrtMPIExchangeRecords(out,ns,os,in,nr,or,sizeof *in);
  for (int i = 0; i < receives; ++i)
    rows[used++] = (struct Arrival){in[i].domain,in[i].id,-1-(int64_t)i};
  free(cursor); free(out);
  qsort(rows,n,sizeof *rows,compareArrival);
  for (int64_t i = 0; i < n; ++i) ++next->rows[rows[i].domain];
  mdrtMPIShareDomainCounts(next->rows,next->domains);
  next->stride = 1;
  for (int64_t d = 0; d < next->domains; ++d)
    if (next->rows[d] > next->stride) next->stride = next->rows[d];
  next->idBase = first;
  size_t slots = (past > first ? (past-first)*next->stride : 1);
  next->ids = malloc(slots*sizeof *next->ids);
  next->sourceSlots = malloc(slots*sizeof *next->sourceSlots);
  if (!next->sourceSlots || !next->ids) fail("out of memory indexing migrated rows");
  if (old->groupCount) {
    next->localGroups = malloc(slots*sizeof *next->localGroups);
    if (!next->localGroups) fail("out of memory retaining owned constraint groups");
    next->groupCount = old->groupCount;
  }
  int64_t i = 0;
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < next->rows[d]; ++r, ++i) {
      next->ids[(d-first)*next->stride+r] = rows[i].id;
      int64_t local = (d-first)*next->stride+r, slot = rows[i].slot;
      next->sourceSlots[local] = slot;
      if (next->localGroups)
        next->localGroups[local] = slot < 0 ? in[-1-slot].group :
            rowGroup(old,slot/old->stride,slot%old->stride,rows[i].id);
    }
  free(rows);
  free(in);
  if (keepPositions) {
    next->migratedPositions = malloc(slots*sizeof *next->migratedPositions);
    if (!next->migratedPositions) fail("out of memory retaining local positions");
    fillLocalField(old,next,x,NULL,1);
  }
  if (observer) {
    /* A rank counts departures from its old rows. Share the domain totals
       at the reporting interface, independently of particle identities. */
    int64_t *moves = calloc(next->domains,2*sizeof *moves);
    if (!moves) fail("out of memory counting domain crossings");
    for (int64_t d = first; d < past; ++d)
      for (int64_t r = 0; r < old->rows[d]; ++r) {
        int64_t target = targets[(d-first)*old->stride+r];
        if (target != d) { ++moves[2*d+1]; ++moves[2*target]; }
      }
    mdrtMPISumMigration(moves,next->domains);
    for (int64_t d = 0; d < next->domains; ++d)
      observer->movementCounts(observer->context,d,moves[2*d],moves[2*d+1]);
    free(moves);
  }
  free(targets);
}

#endif

static int64_t migrate(int64_t oldHandle, struct Memref3D *positions,
                       double lx, double ly, double lz, int keepPositions) {
  struct Layout *old = (struct Layout *)(intptr_t)oldHandle;
  if (!old || !positions || positions->sizes[0] != old->domains ||
      positions->sizes[1] < ownedCapacity(old) || positions->sizes[2] != 3)
    fail("migration positions do not cover the old layout");
  checkIdentities(old);
  int64_t first = 0, past = old->domains;
#ifdef MDIR_ENABLE_MPI
  if (old->mpi) {
    first = mdrtMPIDomainBound(old->domains, 0);
    past = mdrtMPIDomainBound(old->domains, 1);
  }
#endif
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
  copyCuts(next,old->cuts);
  int64_t n = next->count, cap = n > 0 ? n : 1;
  next->rows = calloc(next->domains,sizeof *next->rows);
  if (!next->rows) fail("out of memory counting domains");
#ifdef MDIR_ENABLE_MPI
  if (old->mpi && mdrtMPISize() > 1) {
    migrateLocal(old,next,positions,first,past,keepPositions);
    checkIdentities(next);
    if (tracing()) {
      materializeRows(old);
      materializeRows(next);
      traceMigration(old,next);
    }
    return (int64_t)(intptr_t)next;
  }
#endif
  if (old->groupCount) {
    next->groupSource = old->groupSource ? old->groupSource : old;
    mdrtDistRetainLayout((int64_t)(intptr_t)next->groupSource);
    next->groupCount = old->groupCount;
    next->groups = old->groups;
    next->centerGroup = old->centerGroup;
    next->groupCenters = old->groupCenters;
  }
  next->owner = malloc(cap*sizeof *next->owner);
  next->row = malloc(cap*sizeof *next->row);
  if (!next->owner || !next->row) fail("out of memory numbering particles");
  if (keepPositions) {
    next->migratedPositions = malloc(cap*sizeof *next->migratedPositions);
    if (!next->migratedPositions) fail("out of memory retaining migrated positions");
  }
  int invalid = 0;
  #pragma omp parallel if(n > 1024) reduction(|:invalid)
  {
    #pragma omp for collapse(2) schedule(static)
    for (int64_t d = first; d < past; ++d)
      for (int64_t r = 0; r < old->stride; ++r) {
        if (r >= old->rows[d]) continue;
        int64_t j = domainIds(old,d)[r], cell[3];
        for (int k = 0; k < 3; ++k) {
          double coordinate = *at3(positions,d,r,k), u = coordinate/edges[k];
          if (!isfinite(u)) { invalid = 1; u = 0; }
          double image = floor(u);
          cell[k] = layoutDivision(next,k,u-image);
          if (keepPositions) next->migratedPositions[j][k] = coordinate-edges[k]*image;
        }
        if (!old->groupCenters || !old->groupCenters[j]) {
          next->owner[j] = cell[0] + next->grid[0]*(cell[1] + next->grid[1]*cell[2]);
        } else if (old->centerGroup[j]) {
          struct ConstraintGroup *g = &old->groups[old->centerGroup[j]-1];
          double points[4][3];
          for (int a = 0; a < g->size; ++a)
            for (int k = 0; k < 3; ++k)
              points[a][k] = *at3(positions,d,old->row[g->atoms[a]],k);
          int finite = 1;
          for (int a = 0; a < g->size; ++a)
            for (int k = 0; k < 3; ++k) finite &= isfinite(points[a][k]/edges[k]);
          if (!finite) { invalid = 1; continue; }
          int64_t owner = groupOwner(next,g,points);
          for (int a = 0; a < g->size; ++a) next->owner[g->atoms[a]] = owner;
        }
      }
  }
  if (invalid) fail("migration requires finite positions within cell range");
  numberRows(next,NULL,next->count);
  if (observer)
    for (int64_t j = 0; j < n; ++j)
      if (old->owner[j] != next->owner[j]) {
        observer->movementCounts(observer->context,old->owner[j],0,1);
        observer->movementCounts(observer->context,next->owner[j],1,0);
      }
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
  #pragma omp parallel for collapse(2) schedule(static) if(l->count > 1024)
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->stride; ++r) {
      if (r >= l->rows[d]) continue;
      int64_t j = l->sourceSlots ? (d-first)*l->stride+r : domainIds(l,d)[r];
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
  if (next->sourceSlots) {
    fillLocalField(old,next,source,destination,image);
    return;
  }
#endif
  int64_t components = source->sizes[2];
  int64_t first = 0, past = next->domains;
#ifdef MDIR_ENABLE_MPI
  if (next->mpi) {
    first = mdrtMPIDomainBound(next->domains, 0);
    past = mdrtMPIDomainBound(next->domains, 1);
  }
#endif
  #pragma omp parallel for collapse(2) schedule(static) if(next->count > 1024)
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < next->stride; ++r) {
      if (r >= next->rows[d]) continue;
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

/* The box of domain `d`: its region of the grid, widened to hold its owned
   positions, so that every owned particle lies in it whatever the rounding
   of its division (the coverage argument of Section 4.3 needs x_i in it). */
static void domainBox(struct Layout *l, const double (*points)[3], int64_t d,
                      double lo[3], double hi[3], int midpoint) {
  int64_t cell[3] = {d % l->grid[0], (d / l->grid[0]) % l->grid[1],
                     d / (l->grid[0] * l->grid[1])};
  for (int k = 0; k < 3; ++k) {
    if (l->cuts[k]) {
      lo[k] = l->edges[k]*l->cuts[k][cell[k]];
      hi[k] = l->edges[k]*l->cuts[k][cell[k]+1];
    } else {
      lo[k] = l->edges[k] * (double)cell[k] / (double)l->grid[k];
      hi[k] = l->edges[k] * (double)(cell[k] + 1) / (double)l->grid[k];
    }
  }
  /* At midpoint ownership the interaction center must lie in the grid
     cell. The reach bounds every participating site from that center.
     An owned site across a face of the periodic box cannot enlarge it. */
  if (midpoint) return;
  for (int64_t r = 0; r < l->rows[d]; ++r)
    for (int k = 0; k < 3; ++k) {
      double v = points[domainIds(l,d)[r]][k];
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

struct GhostSite { int64_t id, domain, row; double x[3]; };
static void visitGhosts(struct Halo *h, const double (*points)[3],
                        const struct GhostBin *bins, int64_t d,
                        const double lo[3], const double hi[3],
                        struct GhostList *list, const struct GhostSite *sites, int64_t count) {
  struct Layout *l = h->layout;
  const double reach2 = h->reach*h->reach;
  int64_t span[3];
  for (int c = 0; c < 3; ++c)
    span[c] = (int64_t)ceil(h->reach/l->edges[c]);
  for (int64_t first = 0; first < count; first += GhostBinSize) {
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
    int64_t end = first+GhostBinSize < count ? first+GhostBinSize : count;
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
            if ((sites ? sites[j].domain : l->owner[j]) == d && a == 0 && b == 0 && z == 0) continue;
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
  struct GhostSite *sites;
  int64_t count, first, past;
};

#ifdef MDIR_ENABLE_MPI
struct Route {
  int *send, *recv, *sendOffset, *recvOffset, *cursor;
  int sends, receives;
};

static struct Route beginRoute(void) {
  int peers = mdrtMPISize();
  int *v = calloc(5*(size_t)peers,sizeof *v);
  if (!v) fail("out of memory counting routed records");
  return (struct Route){v,v+peers,v+2*peers,v+3*peers,v+4*peers,0,0};
}

static void countRoute(struct Route *r, int peer) {
  if (r->send[peer] == INT_MAX) fail("too many records for an MPI peer");
  ++r->send[peer];
}

static void finishRoute(struct Route *r) {
  mdrtMPIExchangeCounts(r->send,r->recv);
  for (int p = 0; p < mdrtMPISize(); ++p) {
    if (r->send[p] > INT_MAX-r->sends || r->recv[p] > INT_MAX-r->receives)
      fail("record displacement exceeds MPI range");
    r->cursor[p] = r->sendOffset[p] = r->sends;
    r->recvOffset[p] = r->receives;
    r->sends += r->send[p];
    r->receives += r->recv[p];
  }
}


struct InitialAtom { int64_t id, domain; struct ConstraintGroup group; };

int64_t mdrtMPIIdentityBound(int64_t n, int64_t end) {
  int peers = mdrtMPISize(), p = mdrtMPIRank()+end;
  return p*(n/peers)+(p < n%peers ? p : n%peers);
}

/* Topology input is replicated. Sort only this rank's tuple interval, then
   send each key to the directory rank of its smaller endpoint. */
int64_t _mlir_ciface_mdrtDistExclusionShard(struct Members *members) {
  int peers = mdrtMPISize();
  int64_t n = members->sizes[0], lo = mdrtMPIIdentityBound(n,0), hi = mdrtMPIIdentityBound(n,1);
  struct Members slice = *members;
  slice.offset += lo*slice.strides[0];
  slice.sizes[0] = hi-lo;
  struct Exclusions *local = (void *)(intptr_t)_mlir_ciface_mdrtDistExclusions(&slice);
  if (peers == 1) return (int64_t)(intptr_t)local;
  int last = local->count ? (int)(local->pairs[local->count-1] >> 32) : -1;
  int64_t extent = 1 + (int64_t)mdrtMPIMax(last);
  struct Route r = beginRoute();
  for (size_t i = 0; i < local->count; ++i)
    countRoute(&r,mdrtMPIOwner(extent,local->pairs[i] >> 32));
  finishRoute(&r);
  struct Exclusions *shard = newExclusions(r.receives);
  shard->directoryExtent = extent;
  mdrtMPIExchangeRecords(local->pairs,r.send,r.sendOffset,shard->pairs,r.recv,r.recvOffset,
                         sizeof *local->pairs);
  qsort(shard->pairs,shard->count,sizeof *shard->pairs,compareKeys);
  free(local); free(r.send);
  return (int64_t)(intptr_t)shard;
}

/* The local candidate list needs keys only when both endpoints occur in
   the owned or halo rows. The halo keeps that projection across steps. */
int64_t mdrtDistHaloExclusions(int64_t halo, int64_t handle) {
  struct Halo *h = (struct Halo *)(intptr_t)halo;
  struct Layout *l = h->layout;
  struct Exclusions *shard = (struct Exclusions *)(intptr_t)handle;
  int peers = mdrtMPISize();
  if (peers == 1) {
    ++shard->references;
    return handle;
  }
  if (h->exclusions && h->directoryVersion == shard->version) {
    ++h->exclusions->references;
    return (int64_t)(intptr_t)h->exclusions;
  }
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  size_t n = 0;
  for (int64_t d = first; d < past; ++d) n += l->rows[d]+h->rows[d];
  uint64_t *ids = malloc((n ? n : 1)*sizeof *ids);
  if (!ids) fail("out of memory indexing local exclusion rows");
  size_t k = 0;
  for (int64_t d = first; d < past; ++d) {
    for (int64_t i = 0; i < l->rows[d]; ++i) ids[k++] = domainIds(l,d)[i];
    for (int64_t i = 0; i < h->rows[d]; ++i) ids[k++] = h->ids[(d-h->idBase)*h->capacity+i];
  }
  qsort(ids,n,sizeof *ids,compareKeys);
  size_t unique = 0;
  for (size_t i = 0; i < n; ++i)
    if (!i || ids[i] != ids[i-1]) ids[unique++] = ids[i];
  struct Route request = beginRoute();
  for (size_t i = 0; i < unique && ids[i] < (uint64_t)shard->directoryExtent; ++i)
    countRoute(&request,mdrtMPIOwner(shard->directoryExtent,ids[i]));
  finishRoute(&request);
  uint64_t *recv = malloc((request.receives ? (size_t)request.receives : 1)*sizeof *recv);
  if (!recv) fail("out of memory requesting exclusion rows");
  mdrtMPIExchangeRecords(ids,request.send,request.sendOffset,recv,request.recv,request.recvOffset,sizeof *recv);
  free(ids);
  struct Route reply = beginRoute();
  for (int p = 0; p < peers; ++p)
    for (int j = request.recvOffset[p]; j < request.recvOffset[p]+request.recv[p]; ++j) {
      size_t lo = firstKey(shard,recv[j]<<32), hi = firstKey(shard,(recv[j]+1)<<32);
      if (hi-lo > (size_t)(INT_MAX-reply.send[p])) fail("too many local exclusion keys");
      reply.send[p] += hi-lo;
    }
  finishRoute(&reply);
  uint64_t *send = malloc((reply.sends ? (size_t)reply.sends : 1)*sizeof *send);
  if (!send) fail("out of memory returning exclusion keys");
  for (int p = 0; p < peers; ++p)
    for (int j = request.recvOffset[p]; j < request.recvOffset[p]+request.recv[p]; ++j) {
      size_t lo = firstKey(shard,recv[j]<<32), hi = firstKey(shard,(recv[j]+1)<<32);
      memcpy(send+reply.cursor[p],shard->pairs+lo,(hi-lo)*sizeof *send);
      reply.cursor[p] += hi-lo;
    }
  struct Exclusions *local = newExclusions(reply.receives);
  mdrtMPIExchangeRecords(send,reply.send,reply.sendOffset,local->pairs,reply.recv,reply.recvOffset,sizeof *send);
  /* Each peer returns sorted keys for a disjoint interval of smaller
     endpoints. A key with an absent partner cannot match a candidate. */
  free(send); free(recv); free(request.send); free(reply.send);
  mdrtDistFreeExclusions((int64_t)(intptr_t)h->exclusions);
  h->directoryVersion = shard->version;
  h->exclusions = local;
  ++local->references;
  return (int64_t)(intptr_t)local;
}

static int64_t startPartition(struct Memref2D *x, int64_t gx, int64_t gy, int64_t gz,
                              double lx, double ly, double lz, double *const cuts[3],
                              const struct TupleDonation *donations) {
  if (mdrtMPISize() == 1)
    return _mlir_ciface_mdrtDistMPIPartition(x,gx,gy,gz,lx,ly,lz);
  struct Layout *l = calloc(1,sizeof *l);
  if (!l) fail("out of memory starting spatial partition");
  l->mpi = 1;
  l->references = 1;
  l->domains = gx*gy*gz;
  l->count = x->sizes[0];
  l->grid[0] = gx; l->grid[1] = gy; l->grid[2] = gz;
  l->edges[0] = lx; l->edges[1] = ly; l->edges[2] = lz;
  if (cuts) copyCuts(l,cuts);
  if (donations) {
    size_t bytes = l->domains*sizeof *donations;
    l->donations = malloc(bytes);
    if (!l->donations) fail("out of memory retaining tuple destinations");
    memcpy(l->donations,donations,bytes);
  }
  int64_t first = mdrtMPIIdentityBound(l->count,0), past = mdrtMPIIdentityBound(l->count,1);
  l->initial = calloc(past > first ? past-first : 1,sizeof *l->initial);
  if (!l->initial) fail("out of memory assigning particle owners");
  #pragma omp parallel for schedule(static) if(past-first > 1024)
  for (int64_t j = first; j < past; ++j) {
    int64_t d = 0;
    for (int k = 2; k >= 0; --k) {
      double u = *at2(x,j,k)/l->edges[k];
      if (!isfinite(u)) fail("partition requires finite positions");
      d = d*l->grid[k]+layoutDivision(l,k,u-floor(u));
    }
    l->initial[j-first].id = j;
    l->initial[j-first].domain = d;
  }
  return (int64_t)(intptr_t)l;
}

int64_t _mlir_ciface_mdrtDistStartPartition(struct Memref2D *x, int64_t gx, int64_t gy,
                                           int64_t gz, double lx, double ly, double lz) {
  return startPartition(x,gx,gy,gz,lx,ly,lz,NULL,NULL);
}

/* A solve that only visits disjoint complete groups may keep its
   unconstrained atoms in native identity order. The group center chooses
   the owner of all its members. Spatial force layouts use their own grid. */
void mdrtDistUseIdentityGroups(int64_t layout) {
  struct Layout *l = (struct Layout *)(intptr_t)layout;
  if (!l->initial || l->domains != mdrtMPISize()) return;
  l->identityGroups = 1;
  int64_t n = mdrtMPIIdentityBound(l->count,1)-mdrtMPIIdentityBound(l->count,0);
  int rank = mdrtMPIRank();
  for (int64_t i = 0; i < n; ++i) l->initial[i].domain = rank;
}

/* A native interval has compact storage, but partitioning addresses atoms
   by their original identity. The descriptor is used only at owned indices. */
static struct Memref2D identityView(struct Memref2D *local, int64_t count) {
  struct Memref2D view = *local;
  view.offset -= mdrtMPIIdentityBound(count,0)*view.strides[0];
  view.sizes[0] = count;
  return view;
}

int64_t _mlir_ciface_mdrtDistShardPartition(struct Memref2D *local, int64_t gx, int64_t gy,
    int64_t gz, double lx, double ly, double lz, int64_t count) {
  struct Memref2D view = identityView(local,count);
  return _mlir_ciface_mdrtDistStartPartition(&view,gx,gy,gz,lx,ly,lz);
}

static void assignGroups(struct Layout *l, struct Memref2D *x, struct Members *m, int sharded) {
  int64_t n = m->sizes[0], arity = m->sizes[1];
  if (arity < 2 || arity > 4 || n > l->count) fail("invalid constraint group dimensions");
  int64_t lo = mdrtMPIIdentityBound(n,0), hi = mdrtMPIIdentityBound(n,1), first = mdrtMPIIdentityBound(l->count,0);
  struct Route route = beginRoute();
  for (int64_t t = lo; t < hi; ++t)
    for (int64_t a = 0; a < arity; ++a) {
      int32_t id = m->aligned[m->offset+t*m->strides[0]+a*m->strides[1]];
      if (id < 0 || id >= l->count) fail("invalid atom in constraint group");
      countRoute(&route,mdrtMPIOwner(l->count,id));
    }
  finishRoute(&route);
  struct InitialAtom *out = malloc((route.sends ? (size_t)route.sends : 1)*sizeof *out);
  struct InitialAtom *in = malloc((route.receives ? (size_t)route.receives : 1)*sizeof *in);
  if (!out || !in) fail("out of memory assigning group owners");
  /* Route member requests in the same order as their assignments. The
     reverse exchange returns exactly the sites of this topology interval. */
  double (*sites)[3] = NULL;
  if (sharded && !l->identityGroups) {
    int64_t *request = malloc((route.sends ? route.sends : 1)*sizeof *request);
    int64_t *received = malloc((route.receives ? route.receives : 1)*sizeof *received);
    double (*reply)[3] = malloc((route.receives ? route.receives : 1)*sizeof *reply);
    sites = malloc((route.sends ? route.sends : 1)*sizeof *sites);
    if (!request || !received || !reply || !sites) fail("out of memory requesting constraint sites");
    for (int64_t t = lo; t < hi; ++t)
      for (int a = 0; a < arity; ++a) {
        int32_t id = m->aligned[m->offset+t*m->strides[0]+a*m->strides[1]];
        request[route.cursor[mdrtMPIOwner(l->count,id)]++] = id;
      }
    mdrtMPIExchangeRecords(request,route.send,route.sendOffset,received,route.recv,route.recvOffset,sizeof *request);
    for (int i = 0; i < route.receives; ++i)
      for (int k = 0; k < 3; ++k) reply[i][k] = *at2(x,received[i]-first,k);
    mdrtMPIExchangeRecords(reply,route.recv,route.recvOffset,sites,route.send,route.sendOffset,sizeof *sites);
    memcpy(route.cursor,route.sendOffset,mdrtMPISize()*sizeof *route.cursor);
    free(reply); free(received); free(request);
  }
  for (int64_t t = lo; t < hi; ++t) {
    struct ConstraintGroup g = {.size=arity,.relationCenter=m->aligned[m->offset],.tuple=t};
    double points[4][3];
    int slots[4];
    for (int a = 0; a < arity; ++a) {
      g.atoms[a] = m->aligned[m->offset+t*m->strides[0]+a*m->strides[1]];
      slots[a] = route.cursor[mdrtMPIOwner(l->count,g.atoms[a])]++;
      if (!l->identityGroups)
        for (int k = 0; k < 3; ++k)
          points[a][k] = sites ? sites[slots[a]][k] : *at2(x,g.atoms[a],k);
    }
    int64_t domain = l->identityGroups ? mdrtMPIOwner(l->count,g.atoms[0])
                                       : groupOwner(l,&g,points);
    for (int a = 0; a < arity; ++a) {
      int64_t id = g.atoms[a];
      out[slots[a]] = (struct InitialAtom){id,domain,
          a ? (struct ConstraintGroup){.size=-1,.atoms={g.atoms[0]}} : g};
    }
  }
  free(sites);
  mdrtMPIExchangeRecords(out,route.send,route.sendOffset,in,route.recv,route.recvOffset,sizeof *in);
  for (int i = 0; i < route.receives; ++i) {
    struct InitialAtom *a = l->initial+in[i].id-first;
    if (a->group.size) {
      char message[192];
      snprintf(message,sizeof message,"coupled constraint groups: atom %lld occurs at centers %lld and %lld",
               (long long)a->id+1,(long long)a->group.atoms[0]+1,(long long)in[i].group.atoms[0]+1);
      fail(message);
    }
    *a = in[i];
  }
  l->groupCount += n;
  free(in); free(out); free(route.send);
}

void _mlir_ciface_mdrtDistShardGroupOwners(int64_t handle, struct Memref2D *x, struct Members *members) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  if (!members->sizes[0] || l->groupsFrozen) return;
  if (l->initial) assignGroups(l,x,members,1);
  else _mlir_ciface_mdrtDistGroupOwners(handle,x,members);
}

void mdrtDistEndPartition(int64_t handle) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  if (!l->initial) return;
  int64_t n = mdrtMPIIdentityBound(l->count,1)-mdrtMPIIdentityBound(l->count,0);
  struct Route route = beginRoute();
  for (int64_t i = 0; i < n; ++i) countRoute(&route,mdrtMPIOwner(l->domains,l->initial[i].domain));
  finishRoute(&route);
  struct InitialAtom *out = malloc((n ? n : 1)*sizeof *out);
  struct InitialAtom *in = malloc((route.receives ? (size_t)route.receives : 1)*sizeof *in);
  if (!out || !in) fail("out of memory distributing initial owners");
  for (int64_t i = 0; i < n; ++i) {
    int peer = mdrtMPIOwner(l->domains,l->initial[i].domain);
    out[route.cursor[peer]++] = l->initial[i];
  }
  mdrtMPIExchangeRecords(out,route.send,route.sendOffset,in,route.recv,route.recvOffset,sizeof *in);
  l->rows = calloc(l->domains,sizeof *l->rows);
  if (!l->rows) fail("out of memory counting initial rows");
  for (int i = 0; i < route.receives; ++i) ++l->rows[in[i].domain];
  mdrtMPIShareDomainCounts(l->rows,l->domains);
  l->stride = 1;
  for (int64_t d = 0; d < l->domains; ++d) if (l->rows[d] > l->stride) l->stride = l->rows[d];
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  l->idBase = first;
  size_t slots = past > first ? (past-first)*l->stride : 1;
  l->ids = malloc(slots*sizeof *l->ids);
  if (l->groupCount) l->localGroups = malloc(slots*sizeof *l->localGroups);
  int64_t *cursor = calloc(past > first ? past-first : 1,sizeof *cursor);
  if (!l->ids || !cursor || (l->groupCount && !l->localGroups)) fail("out of memory numbering initial rows");
  for (int i = 0; i < route.receives; ++i) {
    int64_t d = in[i].domain-first, slot = d*l->stride+cursor[d]++;
    l->ids[slot] = in[i].id;
    if (l->localGroups) l->localGroups[slot] = in[i].group;
  }
  free(cursor); free(in); free(out); free(route.send);
  free(l->initial);
  l->initial = NULL;
}

struct Collection {
  struct Route route;
  int64_t *sources, *ids;
};

static void releaseCollection(struct Collection *p) {
  if (!p) return;
  free(p->sources);
  free(p->ids);
  free(p->route.send);
  free(p);
}

/* Pack a rank's owned rows by identity interval. Only the owners of
   those intervals scatter by particle identity. MPI can then concatenate
   the contiguous intervals directly into the native field. */
static struct Collection *collection(struct Layout *l) {
  if (l->collection) return l->collection;
  struct Collection *p = calloc(1,sizeof *p);
  if (!p) fail("out of memory creating a collection schedule");
  p->route = beginRoute();
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->rows[d]; ++r)
      countRoute(&p->route,mdrtMPIOwner(l->count,domainIds(l,d)[r]));
  finishRoute(&p->route);
  size_t sends = p->route.sends ? (size_t)p->route.sends : 1;
  size_t receives = p->route.receives ? (size_t)p->route.receives : 1;
  p->sources = malloc(sends*sizeof *p->sources);
  p->ids = malloc(receives*sizeof *p->ids);
  int64_t *ids = malloc(sends*sizeof *ids);
  if (!p->sources || !p->ids || !ids) fail("out of memory indexing collected fields");
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->rows[d]; ++r) {
      int64_t id = domainIds(l,d)[r];
      int i = p->route.cursor[mdrtMPIOwner(l->count,id)]++;
      p->sources[i] = d*l->stride+r;
      ids[i] = id;
    }
  mdrtMPIExchangeRecords(ids,p->route.send,p->route.sendOffset,p->ids,p->route.recv,
                          p->route.recvOffset,sizeof *ids);
  free(ids);
  l->collection = p;
  return p;
}

void _mlir_ciface_mdrtMPICollectField(int64_t layout, struct Memref3D *source,
    int64_t address, int64_t integer, int64_t destination) {
  struct Layout *l = (struct Layout *)(intptr_t)layout;
  struct Collection *p = collection(l);
  struct Route *r = &p->route;
  int64_t width = source->sizes[2];
  int64_t first = destination == 2 ? mdrtMPIIdentityBound(l->count,0) : 0;
  void *global = (void *)(intptr_t)address;
  if (mdrtMPISize() == 1) {
    #pragma omp parallel for schedule(static) if(r->sends > 1024)
    for (int i = 0; i < r->sends; ++i) {
      int64_t slot = p->sources[i], id = p->ids[i];
      for (int64_t k = 0; k < width; ++k) {
        double value = *at3(source,slot/l->stride,slot%l->stride,k);
        if (integer) ((int32_t *)global)[id*width+k] = (int32_t)value;
        else ((double *)global)[id*width+k] = value;
      }
    }
    return;
  }
  double *out = malloc((r->sends ? (size_t)r->sends : 1)*width*sizeof *out);
  double *in = malloc((r->receives ? (size_t)r->receives : 1)*width*sizeof *in);
  if (!out || !in) fail("out of memory collecting fields");
  #pragma omp parallel for schedule(static) if(r->sends > 1024)
  for (int i = 0; i < r->sends; ++i) {
    int64_t slot = p->sources[i], d = slot/l->stride, row = slot%l->stride;
    for (int64_t k = 0; k < width; ++k) out[i*width+k] = *at3(source,d,row,k);
  }
  mdrtMPIExchangeRecords(out,r->send,r->sendOffset,in,r->recv,r->recvOffset,width*sizeof *in);
  #pragma omp parallel for schedule(static) if(r->receives > 1024)
  for (int i = 0; i < r->receives; ++i)
    for (int64_t k = 0; k < width; ++k) {
      int64_t slot = (p->ids[i]-first)*width+k;
      if (integer) ((int32_t *)global)[slot] = (int32_t)in[i*width+k];
      else ((double *)global)[slot] = in[i*width+k];
    }
  free(out); free(in);
  if (destination != 2) mdrtMPICollectIntervals(global,l->count,width,integer,destination);
}

/* A periodic and an unwrapped consumer use the same owned permutation.
   Wrap a separate buffer so an epoch may adjust it to its retained images. */
void _mlir_ciface_mdrtDistWrapOwned(int64_t layout, struct Memref3D *raw,
                                    struct Memref3D *wrapped) {
  struct Layout *l = (struct Layout *)(intptr_t)layout;
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  #pragma omp parallel for collapse(2) schedule(static) if((past-first)*l->stride > 512)
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->stride; ++r) {
      if (r >= l->rows[d]) continue;
      for (int k = 0; k < 3; ++k) {
        double v = *at3(raw,d,r,k);
        *at3(wrapped,d,r,k) = v - l->edges[k]*floor(v/l->edges[k]);
      }
    }
}

/* The inverse of collection uses its stream in the opposite direction.
   The MPI counts and offsets belong to the immutable spatial layout. */
void _mlir_ciface_mdrtDistDistributeShard(int64_t layout, int64_t base,
    struct Memref3D *dest, int64_t integer, int64_t image) {
  struct Layout *l = (struct Layout *)(intptr_t)layout;
  struct Collection *p = collection(l);
  struct Route *r = &p->route;
  int64_t width = dest->sizes[2], first = mdrtMPIIdentityBound(l->count,0);
  if (mdrtMPISize() == 1) {
    #pragma omp parallel for schedule(static) if(r->sends > 1024)
    for (int i = 0; i < r->sends; ++i) {
      int64_t slot = p->sources[i], id = p->ids[i];
      for (int64_t k = 0; k < width; ++k) {
        int64_t j = id*width+k;
        double v = integer ? ((int32_t *)(intptr_t)base)[j] : ((double *)(intptr_t)base)[j];
        if (image) v -= l->edges[k]*floor(v/l->edges[k]);
        *at3(dest,slot/l->stride,slot%l->stride,k) = v;
      }
    }
    return;
  }
  double *out = malloc((r->receives ? (size_t)r->receives : 1)*width*sizeof *out);
  double *in = malloc((r->sends ? (size_t)r->sends : 1)*width*sizeof *in);
  if (!out || !in) fail("out of memory distributing native rows");
  #pragma omp parallel for schedule(static) if(r->receives > 1024)
  for (int i = 0; i < r->receives; ++i)
    for (int64_t k = 0; k < width; ++k) {
      int64_t j = (p->ids[i]-first)*width+k;
      out[i*width+k] = integer ? ((int32_t *)(intptr_t)base)[j] : ((double *)(intptr_t)base)[j];
    }
  mdrtMPIExchangeRecords(out,r->recv,r->recvOffset,in,r->send,r->sendOffset,width*sizeof *in);
  #pragma omp parallel for schedule(static) if(r->sends > 1024)
  for (int i = 0; i < r->sends; ++i) {
    int64_t slot = p->sources[i];
    for (int64_t k = 0; k < width; ++k) {
      double v = in[i*width+k];
      if (image) v -= l->edges[k]*floor(v/l->edges[k]);
      *at3(dest,slot/l->stride,slot%l->stride,k) = v;
    }
  }
  free(in); free(out);
}

void mdrtMPIExpandField(int64_t source, int64_t count, int64_t dest, int64_t width, int64_t integer) {
  int64_t first = mdrtMPIIdentityBound(count,0), past = mdrtMPIIdentityBound(count,1);
  size_t stride = width*(integer ? sizeof(int32_t) : sizeof(double));
  if (past > first) memcpy((char *)(intptr_t)dest+first*stride,(void *)(intptr_t)source,(past-first)*stride);
  mdrtMPICollectIntervals((void *)(intptr_t)dest,count,width,integer,0);
}

DistWide mdrtMPICombineWide(DistWide value) {
  int peers = mdrtMPISize();
  if (peers == 1) return mdrtDistAddWide(0,value);
  DistWide *terms = malloc((size_t)peers*sizeof *terms);
  if (!terms) fail("out of memory collecting observable words");
  mdrtMPIGatherRankValue(&value,terms,1);
  DistWide total = 0;
  for (int p = 0; p < peers; ++p)
    total = mdrtDistAddWide(total,terms[p]);
  free(terms);
  return total;
}

double mdrtMPICombineDouble(double value) {
  int peers = mdrtMPISize();
  if (peers == 1) return 0.0+value;
  double *terms = malloc((size_t)peers*sizeof *terms);
  if (!terms) fail("out of memory collecting observable values");
  mdrtMPIGatherRankValue(&value,terms,0);
  double total = 0;
  for (int p = 0; p < peers; ++p)
    total += terms[p];
  free(terms);
  return total;
}

static int compareSite(const void *a, const void *b) {
  const struct GhostSite *x = a, *y = b;
  return (x->id > y->id)-(x->id < y->id);
}

/* Each axis may choose its closest periodic image independently for an
   orthorhombic box. The minimum squared distance therefore tests whether
   any image can enter this domain's halo. */
static int reachesBox(const double x[3], const double box[6], struct Halo *h) {
  double sum = 0, limit = h->reach*h->reach;
  for (int k = 0; k < 3; ++k) {
    double edge = h->layout->edges[k], best = INFINITY;
    int64_t span = (int64_t)ceil(h->reach/edge);
    for (int64_t n = -span; n <= span; ++n) {
      double v = outsideBox(x[k]+n*edge,box[k],box[k+3]);
      if (v < best) best = v;
    }
    sum += best*best;
    if (!(sum < limit)) return 0;
  }
  return 1;
}

static void prepareLocalHalo(struct HaloBuild *b, struct Memref3D *x) {
  struct Halo *h = b->halo;
  struct Layout *l = h->layout;
  int peers = mdrtMPISize(), rank = mdrtMPIRank();
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  double (*boxes)[6] = malloc(l->domains*sizeof *boxes);
  if (!boxes) fail("out of memory sharing halo bounds");
  for (int64_t d = first; d < past; ++d) {
    domainBox(l,NULL,d,boxes[d],boxes[d]+3,1);
    if (!b->midpoint)
      for (int64_t r = 0; r < l->rows[d]; ++r)
        for (int k = 0; k < 3; ++k) {
          double v = *at3(x,d,r,k);
          if (v < boxes[d][k]) boxes[d][k] = v;
          if (v > boxes[d][3+k]) boxes[d][3+k] = v;
        }
  }
  if (!mdrtMPIShareDomainPartials(boxes,l->domains,6,0))
    for (int64_t d = 0; d < l->domains; ++d)
      mdrtMPIBroadcast(boxes[d],6,mdrtMPIOwner(l->domains,d));
  for (int64_t d = 0; d < l->domains; ++d) {
    memcpy(b->lists[d].lo,boxes[d],3*sizeof(double));
    memcpy(b->lists[d].hi,boxes[d]+3,3*sizeof(double));
  }
  struct Route route = beginRoute();
  int64_t owned = 0;
  for (int64_t d = first; d < past; ++d) owned += l->rows[d];
  struct GhostSite *local = malloc((owned ? owned : 1)*sizeof *local);
  if (!local) fail("out of memory binning local halo sites");
  int64_t i = 0;
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->rows[d]; ++r)
      local[i++] = (struct GhostSite){domainIds(l,d)[r],d,r,
                                       {*at3(x,d,r,0),*at3(x,d,r,1),*at3(x,d,r,2)}};
  /* Store only routes accepted by the distance test. Each peer sees an
     identity once even if several of its domains import the same site. */
  struct GhostList *exports = calloc(peers,sizeof *exports);
  if (!exports) fail("out of memory planning halo candidates");
  #pragma omp parallel for schedule(static) if(peers > 1 && owned > 256)
  for (int p = 0; p < peers; ++p) {
    if (p == rank) continue;
    struct GhostList *list = &exports[p];
    int64_t q = l->domains/peers, rem = l->domains%peers;
    int64_t lo = p*q+(p<rem ? p : rem), hi = lo+q+(p<rem);
    for (int64_t j = 0; j < owned; ++j) {
      int64_t d = lo;
      while (d < hi && !reachesBox(local[j].x,boxes[d],h)) ++d;
      if (d == hi) continue;
      if (list->size == list->capacity) {
        list->capacity = list->capacity ? 2*list->capacity : 64;
        list->data = realloc(list->data,list->capacity*sizeof *list->data);
        if (!list->data) fail("out of memory selecting halo exports");
      }
      list->data[list->size++].id = j;
    }
    if (list->size > INT_MAX) fail("halo candidate count exceeds MPI range");
    route.send[p] = list->size;
  }
  finishRoute(&route);
  struct GhostSite *send = malloc((route.sends ? (size_t)route.sends : 1)*sizeof *send);
  b->count = owned+route.receives;
  struct GhostSite *sites = malloc((b->count ? b->count : 1)*sizeof *sites);
  if (!send || !sites) fail("out of memory routing halo candidates");
  for (int p = 0; p < peers; ++p) {
    for (size_t k = 0; k < exports[p].size; ++k)
      send[route.sendOffset[p]+k] = local[exports[p].data[k].id];
    free(exports[p].data);
  }
  memcpy(sites,local,owned*sizeof *sites);
  mdrtMPIExchangeRecords(send,route.send,route.sendOffset,sites+owned,route.recv,route.recvOffset,sizeof *send);
  free(send); free(local); free(exports); free(route.send); free(boxes);
  qsort(sites,b->count,sizeof *sites,compareSite);
  b->sites = sites;
  b->points = malloc((b->count ? b->count : 1)*sizeof *b->points);
  if (!b->points) fail("out of memory storing halo coordinates");
  for (int64_t j = 0; j < b->count; ++j) memcpy(b->points[j],sites[j].x,sizeof sites[j].x);
}

static int compareLink(const void *a, const void *b) {
  const struct GhostLink *x = a, *y = b;
  if (x->domain != y->domain) return x->domain < y->domain ? -1 : 1;
  return (x->ghost > y->ghost)-(x->ghost < y->ghost);
}

static void routeGhostRows(struct Halo *h, struct GhostList *lists, struct GhostSite *sites) {
  struct Layout *l = h->layout;
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  mdrtMPIShareDomainCounts(h->rows,l->domains);
  for (int64_t d = 0; d < l->domains; ++d)
    if (h->rows[d] > h->capacity) h->capacity = h->rows[d];
  if (!h->capacity) h->capacity = 1;
  h->idBase = first;
  int64_t cap = h->capacity, slots = past > first ? (past-first)*cap : 1;
  h->ids = malloc(slots*sizeof *h->ids);
  h->image = malloc(slots*3*sizeof *h->image);
  int64_t locals = 0;
  struct Route route = beginRoute();
  int rank = mdrtMPIRank();
  for (int64_t d = first; d < past; ++d) {
    locals += h->rows[d];
    for (int64_t g = 0; g < h->rows[d]; ++g) {
      struct GhostSite site = sites[lists[d].data[g].id];
      int peer = mdrtMPIOwner(l->domains,site.domain);
      if (peer != rank) countRoute(&route,peer);
    }
  }
  finishRoute(&route);
  struct GhostLink *send = malloc((route.sends ? (size_t)route.sends : 1)*sizeof *send);
  int64_t n = locals+route.receives;
  h->links = malloc((n ? n : 1)*sizeof *h->links);
  h->linkOffsets = calloc(l->domains+1,sizeof *h->linkOffsets);
  if (!h->ids || !h->image || !send || !h->links || !h->linkOffsets)
    fail("out of memory recording halo links");
  int64_t *edges = observer ? calloc(l->domains*l->domains,sizeof *edges) : NULL;
  if (observer && !edges) fail("out of memory reporting halo links");
  int64_t k = 0;
  for (int64_t d = first; d < past; ++d) {
    for (int64_t g = 0; g < h->rows[d]; ++g) {
      struct GhostRow *row = &lists[d].data[g];
      struct GhostSite site = sites[row->id];
      int peer = mdrtMPIOwner(l->domains,site.domain);
      h->ids[(d-first)*cap+g] = site.id;
      memcpy(h->image+3*((d-first)*cap+g),row->image,sizeof row->image);
      struct GhostLink link = {d,g,site.domain,site.row};
      h->links[k++] = link;
      if (peer != rank) send[route.cursor[peer]++] = link;
      if (edges) ++edges[d*l->domains+site.domain];
    }
    free(lists[d].data);
  }
  mdrtMPIExchangeRecords(send,route.send,route.sendOffset,h->links+locals,route.recv,route.recvOffset,sizeof *send);
  free(send); free(route.send);
  qsort(h->links,n,sizeof *h->links,compareLink);
  for (int64_t j = 0; j < n; ++j) ++h->linkOffsets[h->links[j].domain+1];
  for (int64_t d = 0; d < l->domains; ++d) h->linkOffsets[d+1] += h->linkOffsets[d];
  if (edges) {
    mdrtMPIShareIndexRows(edges,l->domains,l->domains);
    for (int64_t d = 0; d < l->domains; ++d) {
      observer->domain(observer->context,d,l->rows[d],h->rows[d],lists[d].lo,lists[d].hi);
      for (int64_t src = 0; src < l->domains; ++src) {
        int64_t count = edges[d*l->domains+src];
        if (!count) continue;
        h->edges = realloc(h->edges,(1+h->edgeCount)*sizeof *h->edges);
        if (!h->edges) fail("out of memory reporting halo edges");
        h->edges[h->edgeCount++] = (struct Edge){src,d,count};
      }
    }
    free(edges);
  }
}
#endif
int64_t _mlir_ciface_mdrtDistBeginHalo(int64_t layout, struct Memref3D *x,
                                       double reach, int64_t midpoint, int64_t migrated) {
  struct Halo *h = calloc(1, sizeof *h);
  if (!h)
    fail("out of memory");
  h->references = 1;
  h->layout = (struct Layout *)(intptr_t)layout;
  mdrtDistRetainLayout(layout);
  h->reach = reach;
  struct Layout *l = h->layout;
  int64_t domains = l->domains;
  struct HaloBuild *build = calloc(1,sizeof *build);
  h->rows = calloc(domains,sizeof *h->rows);
  if (!build || !h->rows) fail("out of memory starting halo construction");
  build->halo = h;
  build->midpoint = midpoint & 1;
  build->count = l->count;
  build->past = domains;
  build->lists = calloc(domains,sizeof *build->lists);
  if (!build->lists) fail("out of memory building domain halos");
#ifdef MDIR_ENABLE_MPI
  if (l->mpi && mdrtMPISize() > 1) {
    migrated = 0;
    if (midpoint & 2) {
      build->first = mdrtMPIDomainBound(domains,0);
      build->past = mdrtMPIDomainBound(domains,1);
      prepareLocalHalo(build,x);
    }
  }
#endif
  size_t length = build->count ? build->count : 1;
  if (!build->sites) {
    if (!migrated) shareRows(l,l->rows,x);
    build->points = migrated ? l->migratedPositions : malloc(length*sizeof *build->points);
  }
  build->bins = malloc(((length+GhostBinSize-1)/GhostBinSize)*sizeof *build->bins);
  if (!build->points || !build->bins) fail("out of memory binning halo positions");
  for (int64_t j = 0; j < build->count; ++j) {
    struct GhostBin *bin = build->bins+j/GhostBinSize;
    for (int c = 0; c < 3; ++c) {
      if (!build->sites && !migrated) build->points[j][c] = *at3(x,l->owner[j],l->row[j],c);
      double v = build->points[j][c];
      if (j%GhostBinSize == 0 || v < bin->lo[c]) bin->lo[c] = v;
      if (j%GhostBinSize == 0 || v > bin->hi[c]) bin->hi[c] = v;
    }
  }
  return (int64_t)(intptr_t)build;
}

/* Each domain owns its output rows. A threaded caller visits independent
   domains in parallel while each domain retains its identity/image order. */
void _mlir_ciface_mdrtDistVisitHalo(int64_t handle, int64_t domain, struct Memref3D *x) {
  struct HaloBuild *build = (struct HaloBuild *)(intptr_t)handle;
  struct Halo *h = build->halo;
  double *lo = build->lists[domain].lo, *hi = build->lists[domain].hi;
  if (domain < build->first || domain >= build->past) return;
  if (!build->sites) domainBox(h->layout,build->points,domain,lo,hi,build->midpoint);
  visitGhosts(h,build->points,build->bins,domain,lo,hi,&build->lists[domain],build->sites,build->count);
  h->rows[domain] = build->lists[domain].size;
}

int64_t _mlir_ciface_mdrtDistFinishHalo(int64_t handle, struct Memref3D *x) {
  struct HaloBuild *build = (struct HaloBuild *)(intptr_t)handle;
  struct Halo *h = build->halo;
  struct Layout *l = h->layout;
  struct GhostList *lists = build->lists;
  int64_t domains = l->domains;
#ifdef MDIR_ENABLE_MPI
  if (build->sites) {
    routeGhostRows(h,lists,build->sites);
    free(build->sites);
    free(build->points);
    free(build->bins);
    free(lists);
    free(build);
    return (int64_t)(intptr_t)h;
  }
#endif
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
  if (observer) {
    // Count neighbor multiplicities once per halo, including periodic
    // self-images. Later transfers scale these counts by their field width.
    int64_t *counts = calloc(domains, sizeof *counts);
    if (!counts)
      fail("out of memory counting halo imports");
    for (int64_t d = 0; d < domains; ++d) {
      const double *lo = lists[d].lo, *hi = lists[d].hi;
      observer->domain(observer->context, d, h->layout->rows[d], h->rows[d], lo, hi);
      for (int64_t g = 0; g < h->rows[d]; ++g)
        ++counts[h->sourceDomain[d * cap + g]];
      size_t neighbors = 0;
      for (int64_t e = 0; e < domains; ++e)
        neighbors += counts[e] != 0;
      if (neighbors) {
        h->edges =
            realloc(h->edges, (h->edgeCount + neighbors) * sizeof *h->edges);
        if (!h->edges)
          fail("out of memory counting halo edges");
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

int64_t _mlir_ciface_mdrtDistDriverHalo(int64_t layout, struct Memref3D *x,
                                        double reach, int64_t mode) {
  return buildHalo(layout,x,reach,mode);
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
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  materializeRows(l);
  (void)positions;
  #pragma omp parallel for schedule(static) if(members->sizes[0] > 512)
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

/* Ghosts are emitted in identity order, with all periodic images of an
   atom adjacent. Search the domain's actual rows without a system-sized
   inverse table for each domain. */
static int64_t firstImage(const struct Halo *h, int64_t domain, int64_t id) {
  const int64_t *ids = h->ids + (domain-h->idBase)*h->capacity;
  int64_t lo = 0, hi = h->rows[domain];
  while (lo < hi) {
    int64_t mid = lo + (hi-lo)/2;
    if (ids[mid] < id) lo = mid+1;
    else hi = mid;
  }
  return lo;
}

/* Keep the topology index for parameter lookup. Counting sort is stable
   within each domain, so chunk boundaries are independent of the team. */
int64_t _mlir_ciface_mdrtDistTupleChunks(struct Memref2I *rows, int64_t first,
    int64_t past, int64_t width, struct Memref1I *order, struct Memref1I *starts) {
  int64_t domains = past - first;
  int64_t *offsets = calloc(domains + 1, sizeof *offsets);
  if (!offsets) fail("out of memory sorting local tuple rows");
  for (int64_t t = 0; t < rows->sizes[0]; ++t) {
    int64_t d = rows->aligned[rows->offset + t * rows->strides[0]];
    if (d >= first && d < past) ++offsets[d - first + 1];
  }
  int64_t chunks = 0;
  for (int64_t d = 0; d < domains; ++d) {
    offsets[d+1] += offsets[d];
    for (int64_t t = offsets[d]; t < offsets[d+1]; t += width)
      starts->aligned[chunks++] = t;
  }
  starts->aligned[chunks] = offsets[domains];
  for (int64_t t = 0; t < rows->sizes[0]; ++t) {
    int64_t d = rows->aligned[rows->offset + t * rows->strides[0]];
    if (d >= first && d < past) order->aligned[offsets[d-first]++] = t;
  }
  free(offsets);
  return chunks;
}

/* List tuple-member contributions by destination. Within a destination the
   entries follow topology order, including repeated members of a tuple. */
int64_t _mlir_ciface_mdrtDistTupleIncidence(struct Memref2I *rows,
    struct Memref1I *order, int64_t count, int64_t first, int64_t past,
    int64_t stride, struct Memref1I *entries, struct Memref1I *starts) {
  int64_t domains = past - first, arity = rows->sizes[1] - 1;
  int64_t *ranges = calloc(domains + 1, sizeof *ranges);
  int64_t *used = calloc(domains, sizeof *used);
  if (!ranges || !used) fail("out of memory indexing tuple domains");
  ranges[domains] = count;
  // The chunk builder has already put the topology indices in domain order.
  for (int64_t d = 1; d < domains; ++d) {
    int64_t lo = ranges[d-1], hi = count;
    while (lo < hi) {
      int64_t mid = lo + (hi-lo)/2;
      int64_t tuple = order->aligned[mid];
      int64_t owner = rows->aligned[rows->offset + tuple*rows->strides[0]];
      if (owner < first+d) lo = mid+1;
      else hi = mid;
    }
    ranges[d] = lo;
  }
  #pragma omp parallel for schedule(static) if(domains > 1 && count > 1024)
  for (int64_t d = 0; d < domains; ++d) {
    if (ranges[d] == ranges[d+1]) continue;
    int64_t *cursor = calloc(stride, sizeof *cursor);
    if (!cursor) fail("out of memory indexing tuple forces");
    int64_t base = ranges[d]*arity;
    for (int pass = 0; pass < 2; ++pass) {
      for (int64_t p = ranges[d]; p < ranges[d+1]; ++p) {
        int64_t t = order->aligned[p];
        const int64_t *row = rows->aligned + rows->offset + t*rows->strides[0];
        for (int64_t a = 0; a < arity; ++a) {
          int64_t dest = row[(a+1)*rows->strides[1]];
          if (pass) entries->aligned[cursor[dest]++] = p*arity + a;
          else ++cursor[dest];
        }
      }
      if (!pass) {
        int64_t next = base, active = 0;
        for (int64_t i = 0; i < stride; ++i) {
          int64_t n = cursor[i];
          if (n) starts->aligned[base+active++] = next;
          cursor[i] = next;
          next += n;
        }
        used[d] = active;
      }
    }
    free(cursor);
  }
  int64_t active = 0;
  for (int64_t d = 0; d < domains; ++d) {
    int64_t base = ranges[d]*arity;
    if (active != base && used[d])
      memmove(starts->aligned+active, starts->aligned+base, used[d]*sizeof(int64_t));
    active += used[d];
  }
  starts->aligned[active] = count*arity;
  free(used);
  free(ranges);
  return active;
}

struct TopologySite { int64_t domain, row; double x[3]; };

struct TupleGeometry {
  const int64_t *slots;
  const struct TopologySite *local, *remote;
};

#ifdef MDIR_ENABLE_MPI
/* A geometry snapshot owns one contiguous identity interval per rank.
   Tuple intervals gather only their member sites from this directory. */
static _Atomic uint64_t lastTopologySnapshot;
struct TopologyShard {
  int64_t first, count;
  uint64_t snapshot;
  struct TopologySite sites[];
};
int64_t _mlir_ciface_mdrtDistTopologyShard(int64_t layout, struct Memref3D *x) {
  struct Layout *l = (struct Layout *)(intptr_t)layout;
  int peers = mdrtMPISize(), rank = mdrtMPIRank();
  if (peers == 1) return 0;
  int64_t q = l->count/peers, rem = l->count%peers, n = q+(rank < rem);
  struct TopologyShard *shard = malloc(sizeof *shard+n*sizeof *shard->sites);
  if (!shard) fail("out of memory storing topology sites");
  shard->first = rank*q+(rank < rem ? rank : rem);
  shard->count = n;
  shard->snapshot = atomic_fetch_add_explicit(&lastTopologySnapshot,1,memory_order_relaxed)+1;
  if (!shard->snapshot) fail("topology snapshot version exhausted");
  struct Route route = beginRoute();
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->rows[d]; ++r) {
      int peer = mdrtMPIOwner(l->count,domainIds(l,d)[r]);
      if (peer != rank) countRoute(&route,peer);
    }
  finishRoute(&route);
  struct SiteRecord { int64_t id; struct TopologySite site; };
  struct SiteRecord *send = malloc((route.sends ? (size_t)route.sends : 1)*sizeof *send);
  struct SiteRecord *recv = malloc((route.receives ? (size_t)route.receives : 1)*sizeof *recv);
  if (!send || !recv) fail("out of memory routing topology sites");
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->rows[d]; ++r) {
      int64_t id = domainIds(l,d)[r];
      int peer = mdrtMPIOwner(l->count,id);
      struct TopologySite site = {d,r,{*at3(x,d,r,0),*at3(x,d,r,1),*at3(x,d,r,2)}};
      if (peer == rank) shard->sites[id-shard->first] = site;
      else send[route.cursor[peer]++] = (struct SiteRecord){id,site};
    }
  mdrtMPIExchangeRecords(send,route.send,route.sendOffset,recv,route.recv,route.recvOffset,sizeof *recv);
  for (int i = 0; i < route.receives; ++i) shard->sites[recv[i].id-shard->first] = recv[i].site;
  free(recv); free(send); free(route.send);
  return (int64_t)(intptr_t)shard;
}

void mdrtDistReleaseShard(int64_t handle) { free((void *)(intptr_t)handle); }

static int64_t memberAt(const struct Members *m, int64_t i) {
  return m->aligned[m->offset+i/m->sizes[1]*m->strides[0]+i%m->sizes[1]*m->strides[1]];
}

/* Relations are immutable across iterations of their enclosing step loop.
   Map every member once to a local slot or a unique remote request. */
struct TuplePlan {
  struct TuplePlan *next;
  struct SiteRequests *requests;
  struct Members slice;
  int64_t *slots;
};

struct SiteRequests {
  struct TuplePlan *plans;
  struct Route route;
  int64_t *incoming;
  struct TopologySite *replies;
  uint64_t snapshot;
};

int64_t mdrtDistNewSiteRequests(void) {
  struct SiteRequests *requests = calloc(1,sizeof *requests);
  if (!requests) fail("out of memory starting tuple site requests");
  return (int64_t)(intptr_t)requests;
}

static struct TuplePlan *addTuplePlan(struct SiteRequests *requests) {
  struct TuplePlan *p = calloc(1,sizeof *p);
  if (!p) fail("out of memory starting tuple plan");
  p->requests = requests;
  p->next = requests->plans;
  requests->plans = p;
  return p;
}

int64_t mdrtDistNewTuplePlan(void) {
  return (int64_t)(intptr_t)addTuplePlan((struct SiteRequests *)(intptr_t)mdrtDistNewSiteRequests());
}

int64_t _mlir_ciface_mdrtDistRegisterTuplePlan(int64_t handle, struct Members *members) {
  struct TuplePlan *p = addTuplePlan((struct SiteRequests *)(intptr_t)handle);
  int rank = mdrtMPIRank(), peers = mdrtMPISize();
  int64_t q = members->sizes[0]/peers, rem = members->sizes[0]%peers;
  int64_t first = q*rank+(rank < rem ? rank : rem);
  p->slice = *members;
  p->slice.offset += first*p->slice.strides[0];
  p->slice.sizes[0] = q+(rank < rem);
  return (int64_t)(intptr_t)p;
}

void mdrtDistFreeSiteRequests(int64_t handle) {
  struct SiteRequests *requests = (struct SiteRequests *)(intptr_t)handle;
  struct TuplePlan *p = requests->plans;
  while (p) {
    struct TuplePlan *next = p->next;
    free(p->slots);
    free(p);
    p = next;
  }
  free(requests->route.send);
  free(requests->incoming);
  free(requests->replies);
  free(requests);
}

void mdrtDistFreeTuplePlan(int64_t handle) {
  struct TuplePlan *p = (struct TuplePlan *)(intptr_t)handle;
  mdrtDistFreeSiteRequests((int64_t)(intptr_t)p->requests);
}

static int compareId(const void *x, const void *y) {
  int64_t a = *(const int64_t *)x, b = *(const int64_t *)y;
  return (a > b) - (a < b);
}

static void planSites(struct SiteRequests *requests, const struct TopologyShard *shard,
                      int64_t count) {
  size_t total = 0;
  for (struct TuplePlan *p = requests->plans; p; p = p->next) {
    int64_t rows = p->slice.sizes[0], arity = p->slice.sizes[1];
    if (rows < 0 || arity < 2 || (uint64_t)rows > (SIZE_MAX/sizeof(int64_t)-total)/arity)
      fail("tuple site request extent exceeds address range");
    total += (size_t)rows*arity;
  }
  int64_t *ids = malloc((total ? total : 1)*sizeof *ids);
  if (!ids) fail("out of memory planning tuple sites");
  int64_t remote = 0;
  for (struct TuplePlan *p = requests->plans; p; p = p->next) {
    int64_t n = p->slice.sizes[0]*p->slice.sizes[1];
    p->slots = malloc((n ? n : 1)*sizeof *p->slots);
    if (!p->slots) fail("out of memory planning tuple members");
    for (int64_t i = 0; i < n; ++i) {
      int64_t id = memberAt(&p->slice,i);
      if (id < 0 || id >= count) fail("tuple member outside particle set");
      if (id >= shard->first && id-shard->first < shard->count)
        p->slots[i] = id-shard->first;
      else {
        ids[remote++] = id;
        p->slots[i] = ~id;
      }
    }
  }
  qsort(ids,remote,sizeof *ids,compareId);
  requests->route = beginRoute();
  int64_t unique = 0;
  for (int64_t i = 0; i < remote; ++i)
    if (!i || ids[i] != ids[i-1]) {
      countRoute(&requests->route,mdrtMPIOwner(count,ids[i]));
      ids[unique++] = ids[i];
    }
  finishRoute(&requests->route);
  requests->incoming = malloc((requests->route.receives ? requests->route.receives : 1)*sizeof *requests->incoming);
  if (!requests->incoming) fail("out of memory planning remote tuple members");
  mdrtMPIExchangeRecords(ids,requests->route.send,requests->route.sendOffset,requests->incoming,
      requests->route.recv,requests->route.recvOffset,sizeof *ids);
  for (struct TuplePlan *p = requests->plans; p; p = p->next) {
    int64_t n = p->slice.sizes[0]*p->slice.sizes[1];
    for (int64_t i = 0; i < n; ++i)
      if (p->slots[i] < 0) {
        int64_t id = ~p->slots[i];
        const int64_t *found = bsearch(&id,ids,unique,sizeof *ids,compareId);
        p->slots[i] = ~(int64_t)(found-ids);
      }
  }
  free(ids);
}

static const struct TopologySite *readRemoteSites(struct TuplePlan *p, const struct TopologyShard *shard,
    struct Members *slice, int64_t count, struct TopologySite **temporary) {
  struct SiteRequests *requests = p->requests;
  if (!p->slice.sizes[1]) p->slice = *slice;
  if (!p->slots) planSites(requests,shard,count);
  if (requests->snapshot == shard->snapshot) return requests->replies;
  struct Route *route = &requests->route;
  struct TopologySite *send = malloc((route->receives ? route->receives : 1)*sizeof *send);
  struct TopologySite *recv = requests->replies;
  if (!recv) recv = malloc((route->sends ? route->sends : 1)*sizeof *recv);
  if (!send || !recv) fail("out of memory gathering tuple geometry");
  for (int i = 0; i < route->receives; ++i)
    send[i] = shard->sites[requests->incoming[i]-shard->first];
  mdrtMPIExchangeRecords(send,route->recv,route->recvOffset,recv,route->send,route->sendOffset,sizeof *send);
  free(send);
  if (requests->plans->next) {
    requests->snapshot = shard->snapshot;
    requests->replies = recv;
  } else {
    *temporary = recv;
  }
  return recv;
}

#endif

static int64_t imageRow(struct Halo *h, int64_t d, int64_t id, const int64_t image[3],
                        int64_t tuple) {
  for (int64_t g = firstImage(h,d,id); g < h->rows[d] && h->ids[(d-h->idBase)*h->capacity+g] == id; ++g) {
    const int64_t *n = &h->image[3*((d-h->idBase)*h->capacity+g)];
    if (n[0] == image[0] && n[1] == image[1] && n[2] == image[2]) return h->layout->rows[d]+g;
  }
  char message[192];
  snprintf(message,sizeof message,"missing tuple owner coverage: tuple %lld domain %lld lacks atom %lld",
           (long long)tuple+1,(long long)d,(long long)id+1);
  fail(message);
  return -1;
}

struct TupleMember {
  int64_t id, domain, row, image[3];
  const double *x;
};

#ifdef MDIR_ENABLE_MPI
static int64_t donateTuple(struct Halo *h, int64_t domain, int64_t identity,
                           const struct TupleMember *members, int64_t arity) {
  struct Layout *l = h->layout;
  if (!l->donations || !h->tupleMargin) return domain;
  struct TupleDonation choice = l->donations[domain];
  if (!choice.fraction) return domain;
  /* The topology identity chooses a stable sample across report steps. */
  uint64_t sample = (uint64_t)identity*UINT64_C(0x9e3779b97f4a7c15);
  if ((double)(sample >> 32) >= choice.fraction*0x1p32) return domain;
  int64_t dest = choice.destination;
  double lo[3], hi[3];
  domainBox(l,NULL,dest,lo,hi,1);
  for (int64_t i = 0; i < arity; ++i) {
    const struct TupleMember *m = members+i;
    if (m->domain == dest && !m->image[0] && !m->image[1] && !m->image[2]) continue;
    double distance2 = 0;
    for (int axis = 0; axis < 3; ++axis) {
      double edge = l->edges[axis], span = ceil(h->reach/edge);
      if (m->image[axis] < -span || m->image[axis] > span) return domain;
      double x = m->x[axis]+m->image[axis]*edge;
      double outside = outsideBox(x,lo[axis],hi[axis]);
      distance2 += outside*outside;
    }
    if (!(distance2 < h->tupleMargin*h->tupleMargin)) return domain;
  }
  return dest;
}
#endif

/* Return whether all remote row references fit a neighboring periodic image. */
static int tupleRows(int64_t handle, struct Memref3D *positions,
                      struct Members *members, struct Memref2I *out, int64_t begin, int64_t end,
                      int64_t firstTuple, int64_t pastTuple, const struct TupleGeometry *sites,
                      int64_t identityBase) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  struct Layout *l = h->layout;
  int64_t count = members->sizes[0], arity = members->sizes[1];
  if (arity < 2 || out->sizes[0] != count || out->sizes[1] != 1 + (sites ? 4 : 1)*arity)
    fail("invalid tuple ownership schedule shape");
  if (count == 0) return 1;
  int compact = 1;
  int64_t stride = sites ? 1 : positions->strides[2];
  struct TupleMember *member = malloc(arity*sizeof *member);
  if (!member) fail("out of memory locating tuple members");
  for (int64_t t = firstTuple; t < pastTuple; ++t) {
    int64_t anchor = l->count, pivot = 0;
    for (int64_t a = 0; a < arity; ++a) {
      int64_t id = members->aligned[members->offset + t*members->strides[0] + a*members->strides[1]];
      if (id < 0 || id >= l->count) fail("tuple member outside particle set");
      member[a].id = id;
      if (sites) {
        int64_t slot = sites->slots[t*arity+a];
        const struct TopologySite *site = slot >= 0 ? sites->local+slot : sites->remote+(~slot);
        member[a].domain = site->domain;
        member[a].row = site->row;
        member[a].x = site->x;
      } else {
        member[a].domain = l->owner[id];
        member[a].row = l->row[id];
        member[a].x = at3(positions,l->owner[id],l->row[id],0);
      }
      if (member[a].id < anchor) { anchor = member[a].id; pivot = a; }
    }
    int64_t *record = out->aligned + out->offset + t*out->strides[0];
    int64_t d = 0;
    if (arity == 2) {
      int first = member[0].id == anchor ? 0 : 1, other = 1 - first;
      const double *origin = member[first].x, *point = member[other].x;
      for (int k = 2; k >= 0; --k) {
        int64_t c = k*stride;
        double length = l->edges[k];
        double delta = point[c] - origin[c];
        int64_t shift = -(int64_t)floor(delta / length + 0.5);
        delta += (double)shift * length;
        double u = (origin[c] + 0.5 * delta) / length;
        double cell = floor(u);
        member[first].image[k] = -(int64_t)cell;
        member[other].image[k] = shift - (int64_t)cell;
        d = d*l->grid[k] + layoutDivision(l,k,u-cell);
      }
    } else {
      for (int k = 2; k >= 0; --k) {
        double origin = member[pivot].x[k*stride];
        double lo = 0, hi = 0, length = l->edges[k];
        for (int64_t a = 0; a < arity; ++a) {
          double delta = member[a].x[k*stride] - origin;
          member[a].image[k] = -(int64_t)floor(delta / length + 0.5);
          delta += (double)member[a].image[k] * length;
          lo = fmin(lo, delta);
          hi = fmax(hi, delta);
        }
        double u = (origin + 0.5*(lo + hi)) / length;
        double translation = floor(u);
        d = d*l->grid[k] + layoutDivision(l,k,u-translation);
        for (int64_t a = 0; a < arity; ++a)
          member[a].image[k] -= (int64_t)translation;
      }
    }
#ifdef MDIR_ENABLE_MPI
    if (sites) d = donateTuple(h,d,identityBase+t,member,arity);
#endif
    record[0] = d;
    if (d < begin || d >= end) continue;
    for (int64_t a = 0; a < arity; ++a) {
      int64_t j = member[a].id, row = -1;
      if (member[a].domain == d && !member[a].image[0] && !member[a].image[1] && !member[a].image[2])
        row = member[a].row;
      if (sites) {
        int64_t *ref = record+1+4*a;
        ref[0] = row >= 0 ? row : -1-j;
        memcpy(ref+1,member[a].image,3*sizeof *ref);
        if (row < 0)
          for (int k = 0; k < 3; ++k)
            compact &= member[a].image[k] >= -1 && member[a].image[k] <= 1;
      } else {
        if (row < 0) row = imageRow(h,d,j,member[a].image,t);
        record[(a+1)*out->strides[1]] = row;
      }
    }
  }
  free(member);
  return compact;
}

void _mlir_ciface_mdrtDistTupleRows(int64_t handle, struct Memref3D *positions,
                                    struct Members *members, struct Memref2I *out) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  tupleRows(handle,positions,members,out,0,h->layout->domains,0,members->sizes[0],NULL,0);
}

#ifdef MDIR_ENABLE_MPI
struct TupleSchedule {
  struct TupleSchedule *next;
  int references, relation;
  int64_t count, width;
  int64_t *rows, *indices;
};

/* Group centers carry their immutable relation and parameter index through
   migration. The identity of the first center distinguishes disjoint
   constraint relations, including relations of the same arity. */
struct GroupTuple { int64_t tuple, domain, rows[4]; };

static int compareGroupTuple(const void *a, const void *b) {
  const struct GroupTuple *x = a, *y = b;
  return (x->tuple > y->tuple)-(x->tuple < y->tuple);
}

static int64_t groupSchedule(struct Layout *l, struct Members *m) {
  int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
  int64_t cap = (past-first)*l->stride, n = 0, width = m->sizes[1]+1;
  int relation = m->sizes[0] ? m->aligned[m->offset] : -1;
  for (struct TupleSchedule *s = l->groupSchedules; s; s = s->next)
    if (s->width == width && s->relation == relation) {
      ++s->references;
      return (int64_t)(intptr_t)s;
    }
  struct GroupTuple *tuples = malloc((cap ? cap : 1)*sizeof *tuples);
  struct TupleSchedule *s = calloc(1,sizeof *s);
  if (!tuples || !s) fail("out of memory scheduling owned constraint groups");
  if (m->sizes[0]) {
    for (int64_t d = first; d < past; ++d)
      for (int64_t r = 0; r < l->rows[d]; ++r) {
        struct ConstraintGroup g = rowGroup(l,d,r,domainIds(l,d)[r]);
        if (g.size != width-1 || g.relationCenter != relation) continue;
        struct GroupTuple *t = tuples+n++;
        t->tuple = g.tuple;
        t->domain = d;
        for (int a = 0; a < g.size; ++a) t->rows[a] = findOwnedRow(l,d,g.atoms[a]);
      }
  }
  qsort(tuples,n,sizeof *tuples,compareGroupTuple);
  s->rows = malloc((n ? n : 1)*width*sizeof *s->rows);
  s->indices = malloc((n ? n : 1)*sizeof *s->indices);
  if (!s->rows || !s->indices) fail("out of memory indexing owned constraints");
  s->width = width;
  s->count = n;
  for (int64_t i = 0; i < n; ++i) {
    s->indices[i] = tuples[i].tuple;
    s->rows[i*width] = tuples[i].domain;
    memcpy(s->rows+i*width+1,tuples[i].rows,(width-1)*sizeof *s->rows);
  }
  free(tuples);
  s->relation = relation;
  s->references = 2; /* The layout and the returned schedule. */
  s->next = l->groupSchedules;
  l->groupSchedules = s;
  return (int64_t)(intptr_t)s;
}

/* Assign contiguous input intervals to ranks. A peer's records retain input
   order, so concatenating received intervals preserves topology order even
   when several ranks contribute to the same midpoint domain. */
int64_t _mlir_ciface_mdrtDistScheduleTuples(int64_t handle, int64_t groups, int64_t directory, int64_t plan,
    struct Memref3D *positions, struct Members *members) {
  struct Layout *l = groups ? (struct Layout *)(intptr_t)handle :
                              ((struct Halo *)(intptr_t)handle)->layout;
  if (groups) return groupSchedule(l,members);
  int peers = mdrtMPISize(), rank = mdrtMPIRank();
  int64_t total = members->sizes[0], width = members->sizes[1]+1;
  int64_t q = total/peers, rem = total%peers;
  int64_t first = q*rank+(rank < rem ? rank : rem), count = q+(rank < rem);
  struct Members slice = *members;
  slice.offset += first*slice.strides[0];
  slice.sizes[0] = count;
  if (count > INT_MAX || width < 3 || width >= INT_MAX/(int)sizeof(int64_t))
    fail("tuple schedule exceeds MPI range");
  int64_t recordWidth = directory ? 1+4*(width-1) : width;
  int64_t *rows = malloc((count ? count : 1)*recordWidth*sizeof *rows);
  if (!rows) fail("out of memory locating topology interval");
  struct Memref2I records = {rows,rows,0,{count,recordWidth},{recordWidth,1}};
  struct TupleGeometry geometry;
  struct TopologySite *remote = NULL;
  if (directory) {
    struct TuplePlan *p = (struct TuplePlan *)(intptr_t)plan;
    struct TopologyShard *shard = (struct TopologyShard *)(intptr_t)directory;
    const struct TopologySite *sites = readRemoteSites(p,shard,&slice,l->count,&remote);
    geometry = (struct TupleGeometry){p->slots,shard->sites,sites};
  }
  int compact = directory != 0;
  {
    #pragma omp parallel for schedule(static) reduction(&:compact) if(count > 256)
    for (int64_t start = 0; start < count; start += 256) {
      int64_t end = start+256 < count ? start+256 : count;
      compact &= tupleRows(handle,positions,&slice,&records,0,l->domains,start,end,directory ? &geometry : NULL,first);
    }
  }
  free(remote);
  struct TupleSchedule *schedule = calloc(1,sizeof *schedule);
  if (!schedule) fail("out of memory creating tuple schedule");
  schedule->references = 1;
  schedule->width = width;
  if (peers == 1) {
    schedule->rows = rows;
    schedule->count = count;
    return (int64_t)(intptr_t)schedule;
  }
  int *counts = calloc(5*(size_t)peers,sizeof *counts);
  if (!counts) fail("out of memory counting local tuples");
  int *ns = counts, *nr = ns+peers, *os = nr+peers, *or = os+peers, *cursor = or+peers;
  for (int64_t i = 0; i < count; ++i)
    ++ns[mdrtMPIOwner(l->domains,rows[i*recordWidth])];
  /* Communicate the format decision with the counts. Complement zero
     counts too, so every rank sees a request for full-width images. */
  if (!compact)
    for (int p = 0; p < peers; ++p) ns[p] = ~ns[p];
  mdrtMPIExchangeCounts(ns,nr);
  for (int p = 0; p < peers; ++p) {
    if (ns[p] < 0) ns[p] = ~ns[p];
    if (nr[p] < 0) {
      nr[p] = ~nr[p];
      compact = 0;
    }
  }
  int64_t packetWidth = 1 + (compact ? width : recordWidth);
  int receives = 0;
  for (int p = 0; p < peers; ++p) {
    if (p) os[p] = os[p-1]+ns[p-1];
    or[p] = receives;
    if (nr[p] > INT_MAX-receives) fail("tuple schedule displacement exceeds MPI range");
    receives += nr[p];
  }
  memcpy(cursor,os,peers*sizeof *cursor);
  int64_t *out = malloc((count ? count : 1)*packetWidth*sizeof *out);
  int64_t *in = malloc((receives ? (size_t)receives : 1)*packetWidth*sizeof *in);
  if (!out || !in) fail("out of memory routing tuple rows");
  for (int64_t i = 0; i < count; ++i) {
    int dest = mdrtMPIOwner(l->domains,rows[i*recordWidth]);
    int64_t *record = out+packetWidth*cursor[dest]++;
    record[0] = first+i;
    const int64_t *src = rows+i*recordWidth;
    if (compact) {
      record[1] = src[0];
      for (int64_t a = 0; a < width-1; ++a) {
        const int64_t *ref = src+1+4*a;
        record[2+a] = ref[0] >= 0 ? ref[0] :
            ~((ref[1]+1)*9+(ref[2]+1)*3+ref[3]+1);
      }
    } else {
      memcpy(record+1,src,recordWidth*sizeof *rows);
    }
  }
  mdrtMPIExchangeRecords(out,ns,os,in,nr,or,packetWidth*sizeof *in);
  free(counts);
  free(rows);
  free(out);
  schedule->count = receives;
  schedule->rows = malloc((receives ? (size_t)receives : 1)*width*sizeof *rows);
  schedule->indices = malloc((receives ? (size_t)receives : 1)*sizeof *rows);
  if (!schedule->rows || !schedule->indices) fail("out of memory collecting local tuples");
  #pragma omp parallel for schedule(static) if(receives > 1024)
  for (int i = 0; i < receives; ++i) {
    const int64_t *record = in+packetWidth*i;
    schedule->indices[i] = record[0];
    int64_t *local = schedule->rows+width*i;
    {
      local[0] = record[1];
      for (int64_t a = 0; a < width-1; ++a) {
        if (compact) {
          int64_t code = record[2+a];
          if (code < 0) {
            int imageCode = (int)~code;
            int64_t image[] = {imageCode/9-1, (imageCode/3)%3-1, imageCode%3-1};
            int64_t id = memberAt(members,record[0]*(width-1)+a);
            code = imageRow((struct Halo *)(intptr_t)handle,local[0],id,image,record[0]);
          }
          local[1+a] = code;
        } else {
          const int64_t *ref = record+2+4*a;
          local[1+a] = ref[0] >= 0 ? ref[0] :
              imageRow((struct Halo *)(intptr_t)handle,local[0],-1-ref[0],ref+1,record[0]);
        }
      }
    }
  }
  free(in);
  return (int64_t)(intptr_t)schedule;
}

void mdrtDistReleaseSchedule(int64_t handle) {
  struct TupleSchedule *s = (struct TupleSchedule *)(intptr_t)handle;
  if (--s->references == 0) {
    free(s->rows);
    free(s->indices);
    free(s);
  }
}

static void releaseGroupSchedules(struct TupleSchedule *head) {
  while (head) {
    struct TupleSchedule *next = head->next;
    mdrtDistReleaseSchedule((int64_t)(intptr_t)head);
    head = next;
  }
}

void _mlir_ciface_mdrtDistScheduleView(struct Memref2I *out, int64_t handle) {
  struct TupleSchedule *s = (struct TupleSchedule *)(intptr_t)handle;
  *out = (struct Memref2I){s->rows,s->rows,0,{s->count,s->width},{s->width,1}};
}

void _mlir_ciface_mdrtDistParameterOrder(int64_t handle, struct Memref1I *order,
                                         struct Memref1I *indices) {
  struct TupleSchedule *s = (struct TupleSchedule *)(intptr_t)handle;
  for (int64_t i = 0; i < indices->sizes[0]; ++i) {
    int64_t t = order->aligned[order->offset+i*order->strides[0]];
    indices->aligned[indices->offset+i*indices->strides[0]] = s->indices ? s->indices[t] : t;
  }
}

void _mlir_ciface_mdrtDistTupleRowBlock(int64_t handle, struct Memref3D *positions,
    struct Members *members, struct Memref2I *out, int64_t begin, int64_t end,
    int64_t firstTuple, int64_t pastTuple) {
  tupleRows(handle,positions,members,out,begin,end,firstTuple,pastTuple,NULL,0);
}

void _mlir_ciface_mdrtDistLocalTupleRows(int64_t handle, struct Memref3D *positions,
                                         struct Members *members, struct Memref2I *out) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  int64_t domains = h->layout->domains;
  tupleRows(handle,positions,members,out,mdrtMPIDomainBound(domains,0),
            mdrtMPIDomainBound(domains,1),0,members->sizes[0],NULL,0);
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

union HaloBuffer {
  struct Memref3D real;
  struct Memref3W wide;
};

struct HaloField {
  union HaloBuffer owned, ghosts;
  struct Memref3D result;
  int64_t width, image;
};

/* Partition the existing stream by its owned destination. Counting and
   scattering in stream order preserves every floating-point addition. */
static void gatherHalo(struct Halo *h, int64_t first, int64_t past,
    const struct TransferRow *remotePlan, size_t planSize, const void *recv,
    struct HaloField *fields, int64_t fieldCount, int64_t components, int words, int remote) {
  int64_t stride = h->layout->stride;
  if (!h->gatherOffsets) {
    h->gatherSlots = (past-first)*stride;
    size_t slots = h->gatherSlots;
    h->gatherOffsets = calloc(slots+1,sizeof *h->gatherOffsets);
    if (remotePlan) {
      h->needsPeer = calloc(slots+1,sizeof *h->needsPeer);
      if (!h->needsPeer) fail("out of memory indexing remote halo sums");
    }
    size_t *cursor = calloc(slots+1,sizeof *cursor);
    if (!cursor || !h->gatherOffsets) fail("out of memory indexing halo sums");
    for (int pass = 0; pass < 2; ++pass) {
      // The MPI stream is already restricted to this rank. In process,
      // visit all ghost rows in domain order.
      int64_t domains = remotePlan ? 1 : h->layout->domains;
      for (int64_t d = 0; d < domains; ++d) {
        size_t count = remotePlan ? planSize : (size_t)h->rows[d];
        for (size_t k = 0; k < count; ++k) {
          int64_t slot = remotePlan ? remotePlan[k].domain*h->capacity + remotePlan[k].ghost :
                                     d*h->capacity + (int64_t)k;
          int64_t dest = remotePlan ? (remotePlan[k].source-first)*stride+remotePlan[k].row :
                                      (h->sourceDomain[slot]-first)*stride+h->sourceRow[slot];
          if (pass == 0) {
            ++h->gatherOffsets[dest+1];
            if (remotePlan && remotePlan[k].offset >= 0) h->needsPeer[dest] = 1;
          }
          else h->gatherRows[cursor[dest]++] = (struct GatherRow){slot, remotePlan ? remotePlan[k].offset : -1};
        }
      }
      if (pass == 0) {
        for (size_t i = 0; i < slots; ++i) h->gatherOffsets[i+1] += h->gatherOffsets[i];
        memcpy(cursor,h->gatherOffsets,slots*sizeof *cursor);
        size_t n = h->gatherOffsets[slots];
        h->gatherRows = malloc((n ? n : 1)*sizeof *h->gatherRows);
        if (!h->gatherRows) fail("out of memory retaining halo sums");
      }
    }
    free(cursor);
  }
  #pragma omp parallel for schedule(static) if(h->gatherSlots > 512)
  for (int64_t i = 0; i < h->gatherSlots; ++i) {
    if (remotePlan && h->needsPeer[i] != remote) continue;
    size_t begin = h->gatherOffsets[i], end = h->gatherOffsets[i+1];
    if (begin == end) continue;
    int64_t d = first+i/stride, row = i%stride;
    int64_t base = 0;
    for (int64_t f = 0; f < fieldCount; ++f) {
      struct HaloField *field = fields+f;
      for (int64_t c = 0; c < field->width; ++c) {
        if (words) {
          DistWide sum = *wordAt(&field->owned.wide,d,row,c);
          for (size_t k = begin; k < end; ++k) {
            struct GatherRow t = h->gatherRows[k];
            DistWide v = t.offset < 0 ? *wordAt(&field->ghosts.wide,t.slot/h->capacity,t.slot%h->capacity,c) :
                         ((const DistWide *)recv)[t.offset*components+base+c];
            sum = mdrtDistAddWide(sum,v);
          }
          *wordAt(&field->owned.wide,d,row,c) = sum;
        } else {
          double sum = *at3(&field->owned.real,d,row,c);
          for (size_t k = begin; k < end; ++k) {
            struct GatherRow t = h->gatherRows[k];
            double v = t.offset < 0 ? *at3(&field->ghosts.real,t.slot/h->capacity,t.slot%h->capacity,c) :
                       ((const double *)recv)[t.offset*components+base+c];
            sum += v;
          }
          *at3(&field->owned.real,d,row,c) = sum;
        }
      }
      base += field->width;
    }
  }
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
    size_t ns = 0, nr = 0, nl = 0, near = 0;
    for (int64_t d = 0; d < domains; ++d)
      for (int64_t i = h->links ? h->linkOffsets[d] : 0;
           i < (h->links ? h->linkOffsets[d+1] : h->rows[d]); ++i) {
        int64_t g = h->links ? h->links[i].ghost : i;
        int64_t src = h->links ? h->links[i].source : h->sourceDomain[d*h->capacity+g];
        int64_t row = h->links ? h->links[i].row : h->sourceRow[d*h->capacity+g];
        int owner = owners[src];
        int from = transpose ? owners[d] : owner, to = transpose ? owner : owners[d];
        if (from == rank && to != rank) {
          if (pass) plan->send[ns] = (struct TransferRow){d,g,-1,to,src,row};
          else if (plan->counts[to] == INT_MAX) mdrtMPIAbort("halo count exceeds MPI int");
          else ++plan->counts[to];
          ++ns;
        }
        if (to != rank) continue;
        if (!transpose && from == rank) {
          if (pass) plan->local[nl] = (struct TransferRow){d,g,-1,from,src,row};
          ++nl;
        } else {
          int sameNode = !transpose && mdrtMPISameNode(from);
          size_t slot = transpose ? nr : sameNode ? near : plan->onNode + nr - near;
          if (sameNode) ++near;
          if (pass) plan->receive[slot] = (struct TransferRow){d,g,-1,from,src,row};
          else if (from != rank) {
            if (plan->counts[peers+from] == INT_MAX) mdrtMPIAbort("halo count exceeds MPI int");
            ++plan->counts[peers+from];
          }
          ++nr;
        }
      }
    if (!pass) {
      plan->sends = ns; plan->receives = nr; plan->locals = nl; plan->onNode = near;
      plan->send = malloc(ns * sizeof *plan->send);
      plan->receive = malloc((nr ? nr : 1) * sizeof *plan->receive);
      plan->local = malloc(nl * sizeof *plan->local);
      if ((ns && !plan->send) || !plan->receive || (nl && !plan->local))
        mdrtMPIAbort("out of memory storing halo schedule");
    }
  }
  int64_t *cursor = calloc(2*(size_t)peers,sizeof *cursor);
  if (!cursor) fail("out of memory indexing peer offsets");
  for (int p = 1; p < peers; ++p) {
    cursor[p] = cursor[p-1]+plan->counts[p-1];
    cursor[peers+p] = cursor[peers+p-1]+plan->counts[peers+p-1];
  }
  for (size_t i = 0; i < plan->sends; ++i)
    plan->send[i].offset = cursor[plan->send[i].peer]++;
  for (size_t i = 0; i < plan->receives; ++i)
    if (plan->receive[i].peer != rank)
      plan->receive[i].offset = cursor[peers+plan->receive[i].peer]++;
  free(cursor);
  free(owners);
  return plan;
}

struct HaloBatch {
  int64_t count, capacity;
  struct HaloField fields[];
};

int64_t mdrtDistHaloBatch(int64_t capacity) {
  if (capacity <= 0 || (uint64_t)capacity > (SIZE_MAX-sizeof(struct HaloBatch))/sizeof(struct HaloField))
    fail("invalid number of halo fields");
  struct HaloBatch *batch = malloc(sizeof *batch + capacity*sizeof *batch->fields);
  if (!batch) fail("out of memory grouping halo fields");
  batch->count = 0;
  batch->capacity = capacity;
  return (int64_t)(intptr_t)batch;
}

static void addHaloField(int64_t handle, struct HaloField field) {
  struct HaloBatch *batch = (struct HaloBatch *)(intptr_t)handle;
  if (batch->count == batch->capacity) fail("halo batch already full");
  batch->fields[batch->count++] = field;
}

void _mlir_ciface_mdrtDistHaloField(int64_t handle, struct Memref3D *owned,
                                    struct Memref3D *ghosts, int64_t image) {
  addHaloField(handle,(struct HaloField){.owned.real=*owned,.ghosts.real=*ghosts,
               .width=owned->sizes[2],.image=image});
}

void _mlir_ciface_mdrtDistHaloWordField(int64_t handle, struct Memref3W *owned,
                                        struct Memref3W *ghosts, struct Memref3D *result) {
  addHaloField(handle,(struct HaloField){.owned.wide=*owned,.ghosts.wide=*ghosts,
               .width=owned->sizes[2],.result=*result});
}

static void exchangeHalo(struct Halo *h, struct HaloField *fields, int64_t fieldCount,
                         int transpose, int words) {
  // Use the replicated row bounds so every peer makes the same split.
  // Individually representable fields remain valid if their batch exceeds
  // the MPI count or displacement limit.
  if (fieldCount > 1) {
    int64_t width = 0;
    for (int64_t f = 0; f < fieldCount && width <= INT_MAX; ++f) width += fields[f].width;
    int64_t remaining = width && width <= INT_MAX ? INT_MAX/width : 0;
    for (int64_t d = 0; d < h->layout->domains && remaining >= 0; ++d)
      remaining -= h->rows[d];
    if (width > INT_MAX || remaining < 0) {
      for (int64_t f = 0; f < fieldCount; ++f)
        exchangeHalo(h,fields+f,1,transpose,words);
      return;
    }
  }
  const int peers = mdrtMPISize(), rank = mdrtMPIRank();
  struct TransferPlan *plan = transferPlan(h, transpose);
  int *counts = calloc((size_t)peers * 4, sizeof(int));
  if (!counts) mdrtMPIAbort("out of memory packing halo counts");
  int *ns = counts, *nr = ns + peers, *os = nr + peers, *or = os + peers;
  int64_t components = 0;
  for (int64_t f = 0; f < fieldCount; ++f) {
    if (fields[f].width > INT_MAX-components)
      mdrtMPIAbort("halo field width exceeds MPI count range");
    components += fields[f].width;
  }
  for (int p = 0; p < 2*peers; ++p) {
    if (components && plan->counts[p] > INT_MAX / components)
      mdrtMPIAbort("halo count exceeds MPI int");
    counts[p] = plan->counts[p] * components;
  }
  int sendSize = 0, recvSize = 0;
  for (int p = 0; p < peers; ++p) {
    os[p] =  sendSize;
    or[p] =  recvSize;
    if (ns[p] > INT_MAX - sendSize || nr[p] > INT_MAX - recvSize)
      mdrtMPIAbort("halo displacement exceeds MPI int");
    sendSize += ns[p];
    recvSize += nr[p];
  }
  size_t bytes = words ? sizeof(DistWide) : sizeof(double);
  void *send = malloc((size_t)(sendSize ? sendSize : 1) * bytes);
  void *recv = malloc((size_t)(recvSize ? recvSize : 1) * bytes);
  if (!send || !recv) mdrtMPIAbort("out of memory packing halo rows");
  const char *setting = getenv("MDIR_DIST_HALO_OVERLAP");
  const int overlap = !setting || strcmp(setting, "0") != 0;
  struct HaloRequests *pending = NULL;
  for (int pass = 0; pass < 4; ++pass) {
    if (pass == 1) {
      pending = mdrtMPIStartHalo(send, ns, os, recv, nr, or, words, transpose);
      if (!overlap) {
        mdrtMPIWaitHalo(pending);
        pending = NULL;
      }
      if (transpose) {
        int64_t first = mdrtMPIDomainBound(h->layout->domains,0);
        int64_t past = mdrtMPIDomainBound(h->layout->domains,1);
        gatherHalo(h,first,past,plan->receive,plan->receives,recv,fields,fieldCount,components,words,0);
        if (pending) mdrtMPIWaitHalo(pending);
        gatherHalo(h,first,past,plan->receive,plan->receives,recv,fields,fieldCount,components,words,1);
        break;
      }
    }
    if (pass == 2 && pending) mdrtMPIWaitHalo(pending);
    size_t first = pass == 3 ? plan->onNode : 0;
    size_t length = pass == 0 ? plan->sends : pass == 1 ? plan->locals :
                    pass == 2 ? plan->onNode : plan->receives;
    if (first == length) continue;
    struct TransferRow *rows = pass == 0 ? plan->send : pass == 1 ? plan->local : plan->receive;
    int phase = pass == 0 ? DistPack : pass == 3 ? DistInter : DistIntra;
    if (!transpose) mdrtDistPhase(phase);
    #pragma omp parallel for schedule(static) if(length-first > 256)
    for (size_t k = first; k < length; ++k) {
      int64_t d = rows[k].domain, g = rows[k].ghost, i = d*h->capacity+g;
      int64_t dom = rows[k].source, row = rows[k].row;
      int peer = rows[k].peer;
      int64_t offset = rows[k].offset*components;
      for (int64_t f = 0; f < fieldCount; ++f) {
        struct HaloField *field = fields+f;
        for (int64_t c = 0; c < field->width; ++c, ++offset) {
          if (words) {
            ((DistWide *)send)[offset] = *wordAt(&field->ghosts.wide,d,g,c);
          } else if (transpose) {
            ((double *)send)[offset] = *at3(&field->ghosts.real,d,g,c);
          } else if (!pass) {
            ((double *)send)[offset] = *at3(&field->owned.real,dom,row,c);
          } else {
            double v = peer == rank ? *at3(&field->owned.real,dom,row,c) : ((double *)recv)[offset];
            if (field->image) v += (double)h->image[3*(i-h->idBase*h->capacity)+c] * h->layout->edges[c];
            *at3(&field->ghosts.real,d,g,c) = v;
          }
        }
      }
    }
    if (!transpose) mdrtDistPhase(-phase);
  }
  free(send); free(recv); free(counts);
}
#endif

static void convertHaloWords(struct Halo *h, struct Memref3W *owned, struct Memref3D *out) {
  int64_t components = owned->sizes[2];
  int64_t first = 0, past = h->layout->domains;
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi && mdrtMPISize() > 1) {
    first = mdrtMPIDomainBound(past, 0);
    past = mdrtMPIDomainBound(past, 1);
  }
#endif
  #pragma omp parallel for collapse(2) schedule(static) if(h->layout->count > 1024)
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < h->layout->stride; ++r)
      if (r < h->layout->rows[d]) for (int64_t c = 0; c < components; ++c)
        *at3(out, d, r, c) = (double)mdrtDistNarrowWord(*wordAt(owned, d, r, c), DistPairForce) * 0x1p-40;
}

void _mlir_ciface_mdrtDistHaloSumWords(int64_t handle, struct Memref3W *owned,
    struct Memref3W *ghosts, struct Memref3D *out) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  int64_t components = owned->sizes[2];
  if (observer)
    for (size_t i = 0; i < h->edgeCount; ++i) {
      struct Edge edge = h->edges[i];
      observer->transfer(observer->context, edge.to, edge.from,
                         edge.rows * components * (int64_t)sizeof(DistWide));
    }
  struct HaloField field = {.owned.wide=*owned,.ghosts.wide=*ghosts,.width=components};
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi && mdrtMPISize() > 1)
    exchangeHalo(h,&field,1,1,1);
  else
#endif
    gatherHalo(h,0,h->layout->domains,NULL,0,NULL,&field,1,components,1,0);
  convertHaloWords(h,owned,out);
}

/* G: (G a)_g = a_{s(g)}, with `image` x_{s(g)} + n_g L. */
void _mlir_ciface_mdrtDistHaloCopy(int64_t handle, struct Memref3D *owned,
                                   struct Memref3D *ghosts, int64_t image) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  if (observer)
    for (size_t i = 0; i < h->edgeCount; ++i) {
      struct Edge edge = h->edges[i];
      observer->transfer(observer->context, edge.from, edge.to,
                         edge.rows * owned->sizes[2] * (int64_t)sizeof(double));
    }
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi && mdrtMPISize() > 1) {
    struct HaloField field = {.owned.real=*owned,.ghosts.real=*ghosts,.image=image,.width=owned->sizes[2]};
    exchangeHalo(h,&field,1,0,0);
    return;
  }
#endif
  int64_t k = owned->sizes[2];
  #pragma omp parallel for collapse(2) schedule(static) if(h->layout->count > 1024)
  for (int64_t d = 0; d < h->layout->domains; ++d)
    for (int64_t g = 0; g < h->capacity; ++g) {
      if (g >= h->rows[d]) continue;
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
                         edge.rows * owned->sizes[2] * (int64_t)sizeof(double));
    }
  struct HaloField field = {.owned.real=*owned,.ghosts.real=*ghosts,.width=owned->sizes[2]};
#ifdef MDIR_ENABLE_MPI
  if (h->layout->mpi && mdrtMPISize() > 1) {
    exchangeHalo(h,&field,1,1,0);
    return;
  }
#endif
  int64_t k = owned->sizes[2];
  gatherHalo(h,0,h->layout->domains,NULL,0,NULL,&field,1,k,0,0);
  if (tracing())
    fprintf(stderr,
            "mdrt_dist: halo_sum of %lld components: %lld rows between "
            "domains\n",
            (long long)k, (long long)movedRows(h));
}

#ifdef MDIR_ENABLE_MPI
void mdrtDistFlushHalo(int64_t halo, int64_t handle, int64_t transpose, int64_t words) {
  struct Halo *h = (struct Halo *)(intptr_t)halo;
  struct HaloBatch *batch = (struct HaloBatch *)(intptr_t)handle;
  if (batch->count != batch->capacity) fail("incomplete halo batch");
  if (mdrtMPISize() == 1) {
    for (int64_t f = 0; f < batch->count; ++f) {
      struct HaloField *field = batch->fields+f;
      if (words)
        _mlir_ciface_mdrtDistHaloSumWords(halo,&field->owned.wide,&field->ghosts.wide,&field->result);
      else if (transpose)
        _mlir_ciface_mdrtDistHaloSum(halo,&field->ghosts.real,&field->owned.real);
      else
        _mlir_ciface_mdrtDistHaloCopy(halo,&field->owned.real,&field->ghosts.real,field->image);
    }
  } else {
    if (observer)
      for (size_t i = 0; i < h->edgeCount; ++i) {
        struct Edge edge = h->edges[i];
        for (int64_t f = 0; f < batch->count; ++f)
          observer->transfer(observer->context, transpose ? edge.to : edge.from,
              transpose ? edge.from : edge.to, edge.rows*batch->fields[f].width*
              (words ? (int64_t)sizeof(DistWide) : (int64_t)sizeof(double)));
      }
    exchangeHalo(h,batch->fields,batch->count,transpose,words);
    if (words)
      for (int64_t f = 0; f < batch->count; ++f)
        convertHaloWords(h,&batch->fields[f].owned.wide,&batch->fields[f].result);
  }
  free(batch);
}
#endif

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
    mdrtHostMatrixRelease(m->candidates);
    free(m->candidateCounts);
    free(m->counts);
    free(m->selectedCounts);
    free(m->reference);
    free(m->shifts);
    free(m->bounds);
    free(m->midpointSites);
    free(m->bufferedRows);
    free(m->pruneReference);
    free(m);
  }
  free(h->rows);
  free(h->edges);
#ifdef MDIR_ENABLE_MPI
  mdrtDistFreeExclusions((int64_t)(intptr_t)h->exclusions);
  free(h->links);
  free(h->linkOffsets);
  for (int i = 0; i < 2; ++i) {
    free(h->plans[i].counts);
    free(h->plans[i].send);
    free(h->plans[i].receive);
    free(h->plans[i].local);
  }
#endif
  free(h->needsPeer);
  free(h->gatherRows);
  free(h->gatherOffsets);
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
  struct Halo *halo;
  double *reference, *ownedReference;
#ifdef MDIR_ENABLE_MPI
  int balance;
  uint64_t balanceGeneration;
  double skin;
#endif
};

#ifdef MDIR_ENABLE_MPI
/* Plain and reporting paths may have separate live Verlet caches. All
   paths measure one trajectory and adopt the same boundary generation. */
static _Thread_local struct BalanceState {
  struct BalanceClock clock;
  uint64_t generation;
  double *cuts[3];
  struct TupleDonation *donations;
} balanceState;

static void resetBalance(void) {
  balanceClock = NULL;
  for (int axis = 0; axis < 3; ++axis) free(balanceState.cuts[axis]);
  free(balanceState.donations);
  balanceState = (struct BalanceState){0};
}

static double boundary(const struct BalanceState *e, int axis, int index, int width) {
  return e->cuts[axis] ? e->cuts[axis][index] : (double)index/width;
}

/* Use measured force-kernel cost to pair heavy and light adjacent
   domains [Phillips2020]. A recipient accepts work from at most one
   donor. Use the previous fraction when the same neighbors remain paired. */
static int balanceTuples(struct BalanceState *e, const struct Layout *l,
                          const double (*times)[3], int ranks) {
  struct TupleDonation *nextDonations = calloc(ranks,sizeof *nextDonations);
  unsigned char *paired = calloc(ranks,sizeof *paired);
  if (!nextDonations || !paired) fail("out of memory balancing tuple work");
  double mean = 0;
  for (int r = 0; r < ranks; ++r) mean += times[r][2]/ranks;
  int changed = 0;
  for (int r = 0; r < ranks; ++r) {
    double cost = times[r][2];
    struct TupleDonation old = e->donations ? e->donations[r] : (struct TupleDonation){0};
    if (paired[r] || (cost <= 1.05*mean && !old.fraction)) continue;
    int dest = -1;
    double best = mean;
    int stride = 1;
    for (int axis = 0; axis < 3; ++axis) {
      int cells = l->grid[axis], cell = (r/stride)%cells;
      for (int direction = -1; direction <= 1; direction += 2) {
        if (cell+direction < 0 || cell+direction >= cells) continue;
        int next = r+direction*stride;
        if (times[next][2] >= best || paired[next]) continue;
        best = times[next][2];
        dest = next;
      }
      stride *= cells;
    }
    if (dest < 0 || cost <= best) continue;
    double fraction = 0.25*(cost-best)/cost;
    if (dest == old.destination) fraction += old.fraction;
    nextDonations[r] = (struct TupleDonation){dest,fmin(0.25,fraction)};
    paired[r] = paired[dest] = 1;
    ++changed;
  }
  free(paired);
  int reshuffled = 0;
  for (int r = 0; r < ranks; ++r) {
    struct TupleDonation old = e->donations ? e->donations[r] : (struct TupleDonation){0};
    reshuffled |= old.fraction != nextDonations[r].fraction ||
                  (old.fraction && old.destination != nextDonations[r].destination);
  }
  free(e->donations);
  if (!changed) { free(nextDonations); nextDonations = NULL; }
  e->donations = nextDonations;
  if (mdrtMPIRank() == 0)
    fprintf(stderr,"MDIR: measured bonded tuple balancing: mean %.6g ms/step, "
                   "%d donating domains\n",mean*1000,changed);
  return reshuffled;
}

/* Shift the most imbalanced set of planes at an existing list rebuild
   [Hess2008]. Estimate each slab's force density from the rank clocks.
   The limited displacement keeps neighboring faces ordered and bounds
   changes to halo volume even when the timing sample is noisy. */
static void balanceEpoch(struct BalanceState *e, const struct Layout *l) {
  if (e->clock.steps < 16) return;
  int ranks = mdrtMPISize();
  double (*times)[3] = malloc(ranks*sizeof *times);
  if (!times) fail("out of memory collecting force times");
  double local[] = {fmax(e->clock.force,0)/e->clock.steps, e->clock.wait/e->clock.steps,
                    e->clock.tuples/e->clock.steps};
  mdrtMPIGatherMetrics(local,times,sizeof local);
  e->clock = (struct BalanceClock){0};
  double total = 0, peak = 0, wait = 0;
  for (int r = 0; r < ranks; ++r) {
    total += times[r][0];
    peak = fmax(peak,times[r][0]);
    wait += times[r][1];
  }
  int bestAxis = -1;
  double bestRatio = 1.05;
  double *best = NULL;
  for (int axis = 0; axis < 3; ++axis) {
    int width = l->grid[axis];
    if (width < 2) continue;
    double *slabs = calloc(width,sizeof *slabs);
    if (!slabs) fail("out of memory averaging slab force times");
    int stride = axis == 0 ? 1 : l->grid[0];
    if (axis == 2) stride *= l->grid[1];
    for (int r = 0; r < ranks; ++r) slabs[(r/stride)%width] += times[r][0];
    double high = 0;
    for (int i = 0; i < width; ++i) high = fmax(high,slabs[i]);
    if (high > bestRatio*total/width) {
      bestRatio = high/(total/width);
      bestAxis = axis;
      free(best);
      best = slabs;
    } else {
      free(slabs);
    }
  }
  if (mdrtMPIRank() == 0)
    fprintf(stderr,"MDIR: automatic load balance: mean/peak force %.6g/%.6g ms/step, "
                   "mean arrival wait %.6g ms/step, %s\n",
            total*1000/ranks,peak*1000,wait*1000/ranks,
            best ? "adjusting domain boundaries" : "keeping domain boundaries");
  if (best) {
    int width = l->grid[bestAxis], slab = 0;
    double prefix = 0;
    double *cuts = malloc((width+1)*sizeof *cuts);
    if (!cuts) fail("out of memory shifting domain boundaries");
    cuts[0] = 0;
    cuts[width] = 1;
    for (int i = 1; i < width; ++i) {
      double desired = total*i/width;
      while (slab+1 < width && prefix+best[slab] < desired)
        prefix += best[slab++];
      double lo = boundary(e,bestAxis,slab,width);
      double hi = boundary(e,bestAxis,slab+1,width);
      double target = best[slab] > 0 ? lo+(hi-lo)*(desired-prefix)/best[slab] : lo;
      double old = boundary(e,bestAxis,i,width);
      double shift = fmax(-0.05/width,fmin(0.05/width,0.25*(target-old)));
      cuts[i] = fmax((i-0.20)/width,fmin((i+0.20)/width,old+shift));
    }
    free(e->cuts[bestAxis]);
    e->cuts[bestAxis] = cuts;
  }
  free(best);
  int reshuffled = balanceTuples(e,l,times,ranks);
  if (bestAxis >= 0 || reshuffled) ++e->generation;
  free(times);
}
#endif

int64_t mdrtDistEpochCreate(void) {
  struct TrajectoryEpoch *e = calloc(1, sizeof *e);
  if (!e) fail("out of memory creating a trajectory epoch");
  return (int64_t)(intptr_t)e;
}

int64_t mdrtDistBalancedEpochCreate(void) {
  int64_t handle = mdrtDistEpochCreate();
#ifdef MDIR_ENABLE_MPI
  struct TrajectoryEpoch *e = (struct TrajectoryEpoch *)(intptr_t)handle;
  e->balance = mdrtMPISize() > 1;
#endif
  return handle;
}

void mdrtDistEpochDestroy(int64_t cache) {
  struct TrajectoryEpoch *e = (struct TrajectoryEpoch *)(intptr_t)cache;
  if (e->halo) mdrtDistReleaseHalo((int64_t)(intptr_t)e->halo);
  free(e->ownedReference);
  free(e->reference);
  free(e);
}

int64_t _mlir_ciface_mdrtDistEpochPartition(struct Memref2D *x, int64_t gx,
    int64_t gy, int64_t gz, double lx, double ly, double lz,
    int64_t cache, double skin, int64_t mpi) {
  struct TrajectoryEpoch *e = (struct TrajectoryEpoch *)(intptr_t)cache;
  double box[] = {lx, ly, lz};
  int64_t n = x->sizes[0];
  int64_t first = 0, past = n;
#ifdef MDIR_ENABLE_MPI
  if (mpi) { first = mdrtMPIIdentityBound(n,0); past = mdrtMPIIdentityBound(n,1); }
  struct BalanceState *balance = NULL;
  if (e->balance && observer && gx*gy*gz == mdrtMPISize()) {
    balance = &balanceState;
    balanceClock = &balance->clock;
    e->skin = skin;
  }
#endif
  if (e->halo) {
    struct Layout *l = e->halo->layout;
    int valid = l->count == n && l->edges[0] == lx && l->edges[1] == ly && l->edges[2] == lz;
    if (valid) {
      double maxDisplacement2 = 0;
      #pragma omp parallel for reduction(max:maxDisplacement2) schedule(static) if(n > 1024)
      for (int64_t j = first; j < past; ++j) {
        double r2 = 0;
        for (int c = 0; c < 3; ++c) {
          double dx = *at2(x,j,c) - e->reference[3*(j-first)+c];
          dx -= box[c]*floor(dx/box[c] + 0.5);
          r2 += dx*dx;
        }
        maxDisplacement2 = fmax(maxDisplacement2,isfinite(r2) ? r2 : INFINITY);
      }
      valid = isfinite(maxDisplacement2) && maxDisplacement2 <= 0.25*skin*skin;
    }
#ifdef MDIR_ENABLE_MPI
    if (mpi && mdrtMPISize() > 1) valid = mdrtMPIAllTrue(valid);
#endif
#ifdef MDIR_ENABLE_MPI
    int current = !balance || e->balanceGeneration == balance->generation;
#else
    int current = 1;
#endif
    if (valid && current) {
      mdrtDistRetainLayout((int64_t)(intptr_t)l);
      return (int64_t)(intptr_t)l;
    }
#ifdef MDIR_ENABLE_MPI
    if (balance && current && l->edges[0] == lx && l->edges[1] == ly && l->edges[2] == lz)
      balanceEpoch(balance,l);
#endif
    mdrtDistReleaseHalo((int64_t)(intptr_t)e->halo);
    e->halo = NULL;
  }
  e->reference = realloc(e->reference, (past > first ? past-first : 1)*3*sizeof(double));
  if (!e->reference) fail("out of memory recording trajectory positions");
  #pragma omp parallel for schedule(static) if(n > 1024)
  for (int64_t j = first; j < past; ++j)
    for (int c = 0; c < 3; ++c)
      e->reference[3*(j-first)+c] = *at2(x,j,c);
#ifdef MDIR_ENABLE_MPI
  if (balance) {
    e->balanceGeneration = balance->generation;
    return startPartition(x,gx,gy,gz,lx,ly,lz,balance->cuts,balance->donations);
  }
  if (mpi) return startPartition(x,gx,gy,gz,lx,ly,lz,NULL,NULL);
#endif
  return _mlir_ciface_mdrtDistPartition(x,gx,gy,gz,lx,ly,lz);
}

#ifdef MDIR_ENABLE_MPI
int64_t _mlir_ciface_mdrtDistShardEpochPartition(struct Memref2D *local, int64_t gx,
    int64_t gy, int64_t gz, double lx, double ly, double lz, int64_t cache,
    double skin, int64_t mpi, int64_t count) {
  struct Memref2D view = identityView(local,count);
  return _mlir_ciface_mdrtDistEpochPartition(&view,gx,gy,gz,lx,ly,lz,cache,skin,mpi);
}
#endif

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
  #pragma omp parallel for collapse(2) schedule(static) if((past-first)*l->stride > 512)
  for (int64_t d = first; d < past; ++d)
    for (int64_t r = 0; r < l->stride; ++r) {
      if (r >= l->rows[d]) continue;
      int64_t j = domainIds(l,d)[r];
      for (int c = 0; c < 3; ++c) {
        double len = l->edges[c], ref = e->ownedReference ?
            e->ownedReference[3*((d-first)*l->stride+r)+c] : e->reference[3*j+c];
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
#ifdef MDIR_ENABLE_MPI
    struct Layout *l = (struct Layout *)(intptr_t)layout;
    if (l->mpi && mdrtMPISize() > 1) {
      int64_t first = mdrtMPIDomainBound(l->domains,0), past = mdrtMPIDomainBound(l->domains,1);
      size_t slots = past > first ? (past-first)*l->stride : 1;
      e->ownedReference = realloc(e->ownedReference,3*slots*sizeof(double));
      if (!e->ownedReference) fail("out of memory retaining local reference positions");
      for (int64_t d = first; d < past; ++d)
        for (int64_t r = 0; r < l->rows[d]; ++r)
          for (int k = 0; k < 3; ++k)
            e->ownedReference[3*((d-first)*l->stride+r)+k] = *at3(x,d,r,k);
    }
#endif
    e->halo = (struct Halo *)(intptr_t)buildHalo(layout,x,reach,midpoint);
    e->halo->layout->groupsFrozen = 1;
#ifdef MDIR_ENABLE_MPI
    if ((midpoint & 3) == 3 && l->donations) {
      /* The halo was built at the reference positions. Current unwrapped
         coordinates can drift by half the skin in this epoch. Leave
         another half skin for the coordinate and distance roundoff. */
      double edge = fmax(l->edges[0],fmax(l->edges[1],l->edges[2]));
      e->halo->tupleMargin = fmax(0,reach-e->skin-64*DBL_EPSILON*(edge+reach));
    }
#endif
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
