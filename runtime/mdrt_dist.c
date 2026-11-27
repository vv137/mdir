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
#ifdef MDIR_ENABLE_MPI
#include "mdrt_mpi.h"
#endif
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
  int64_t grid[3], domains, count;
  double edges[3];
  struct ConstraintGroup {
    int32_t atoms[4];
    int size;
  } *groups;
  size_t groupCount;
  int32_t *groupCenters; /* center + 1; zero for a free atom */
  int64_t *owner; /* [count]: the domain that owns each particle */
  int64_t *row;   /* [count]: its owned row there */
  int64_t *rows;  /* [domains]: the number of owned rows */
  int64_t *ids;   /* [domains * count]: the particle of each owned row */
};

struct Halo {
  int64_t references;
  struct Layout *layout;
  double reach;
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

int64_t mdrtDistPairAllowed(int64_t handle, int64_t i, int64_t j) {
  const struct Exclusions *e = (const struct Exclusions *)(intptr_t)handle;
  uint64_t key = ((uint64_t)i << 32) | (uint32_t)j;
  return !bsearch(&key, e->pairs, e->count, sizeof key, compareKeys);
}

void mdrtDistFreeExclusions(int64_t handle) {
  free((void *)(intptr_t)handle);
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
  for (int64_t d = 0; d < h->layout->domains; ++d)
    if (shifts->sizes[1] < h->rows[d])
      fail("missing halo rows for deterministic pair geometry");
  for (int64_t d = 0; d < h->layout->domains; ++d)
    for (int64_t g = 0; g < h->rows[d]; ++g)
      for (int c = 0; c < 3; ++c)
        *at3(shifts, d, g, c) =
            (double)h->image[3 * (d * h->capacity + g) + c] *
            h->layout->edges[c];
}

struct Memref3W {
  DistWide *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};

static DistWide *wordAt(struct Memref3W *m, int64_t d, int64_t r, int64_t c) {
  return m->aligned + m->offset + d * m->strides[0] + r * m->strides[1] +
         c * m->strides[2];
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
  int64_t cap = l->count ? l->count : 1;
  for (int64_t d = 0; d < l->domains; ++d)
    l->rows[d] = 0;
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
  if (!count) return;
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
  l->ids = malloc(sizeof(int64_t) * l->domains * cap);
  if (!l->owner || !l->row || !l->rows || !l->ids)
    fail("out of memory");
  for (int64_t j = 0; j < n; ++j) {
    int64_t cell[3];
    for (int k = 0; k < 3; ++k) {
      double u = *at2(x, j, k) / l->edges[k];
      cell[k] = division(u - floor(u), l->grid[k]);
    }
    int64_t d = cell[0] + gx * (cell[1] + gy * cell[2]);
    l->owner[j] = d;
    l->row[j] = l->rows[d];
    l->ids[d * cap + l->rows[d]] = j;
    ++l->rows[d];
  }
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
  if (!l->mpi)
    return;
  if (field->strides[2] != 1 || field->strides[1] != field->sizes[2])
    mdrtMPIAbort("expected contiguous field rows");
  for (int64_t d = 0; d < l->domains; ++d)
    if (rows[d])
      mdrtMPIBroadcast(at3(field, d, 0, 0), rows[d] * field->sizes[2],
                       mdrtMPIOwner(l->domains, d));
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
  int64_t cap = l->count > 0 ? l->count : 1;
  for (int64_t d = 0; d < l->domains; ++d) {
    rows->aligned[rows->offset + d * rows->strides[0]] = l->rows[d];
    for (int64_t r = 0; r < l->rows[d]; ++r)
      ids->aligned[ids->offset + d * ids->strides[0] + r * ids->strides[1]] =
          l->ids[d * cap + r];
  }
}

void mdrtDistRetainLayout(int64_t handle) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  if (!l || l->references <= 0)
    fail("retain of a released layout");
  ++l->references;
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
  free(l->groups);
  free(l->groupCenters);
  free(l->owner);
  free(l->row);
  free(l->rows);
  free(l->ids);
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
      int64_t j = l->ids[d * cap + r];
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

int64_t _mlir_ciface_mdrtDistMigrate(int64_t oldHandle,
                                     struct Memref3D *positions, double lx,
                                     double ly, double lz) {
  struct Layout *old = (struct Layout *)(intptr_t)oldHandle;
  if (!old || !positions || positions->sizes[0] != old->domains ||
      positions->sizes[1] < old->count || positions->sizes[2] != 3)
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
  next->ids = malloc(sizeof(int64_t) * next->domains * cap);
  if (!next->owner || !next->row || !next->rows || !next->ids)
    fail("out of memory");
  for (int64_t j = 0; j < n; ++j) {
    int64_t cell[3];
    for (int k = 0; k < 3; ++k) {
      double coordinate = *at3(positions, old->owner[j], old->row[j], k);
      if (!isfinite(coordinate))
        fail("migration requires finite positions");
      double u = coordinate / edges[k];
      if (!isfinite(u))
        fail("migration coordinate exceeds cell range");
      cell[k] = division(u - floor(u), next->grid[k]);
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
  if (observer)
    for (int64_t j = 0; j < n; ++j)
      if (old->owner[j] != next->owner[j])
        observer->migration(observer->context, old->owner[j], next->owner[j]);
  checkIdentities(next);
  if (tracing())
    traceMigration(old, next);
  return (int64_t)(intptr_t)next;
}

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
      source->sizes[1] < old->count ||
      destination->sizes[1] < next->count ||
      source->sizes[2] != destination->sizes[2] ||
      (image && source->sizes[2] != 3))
    fail("redistribution requires matching, distinct field buffers");
  shareRows(old, old->rows, source);
  int64_t components = source->sizes[2];
  for (int64_t j = 0; j < old->count; ++j)
    for (int64_t k = 0; k < components; ++k) {
      double value = *at3(source, old->owner[j], old->row[j], k);
      if (image) {
        if (!isfinite(value))
          fail("redistributed positions must be finite");
        value -= next->edges[k] * floor(value / next->edges[k]);
      }
      *at3(destination, next->owner[j], next->row[j], k) = value;
    }
}

/* The box of domain `d`: its region of the grid, widened to hold its owned
   positions, so that every owned particle lies in it whatever the rounding
   of its division (the coverage argument of Section 4.3 needs x_i in it). */
static void domainBox(struct Layout *l, struct Memref3D *x, int64_t d,
                      double lo[3], double hi[3]) {
  int64_t cell[3] = {d % l->grid[0], (d / l->grid[0]) % l->grid[1],
                     d / (l->grid[0] * l->grid[1])};
  for (int k = 0; k < 3; ++k) {
    lo[k] = l->edges[k] * (double)cell[k] / (double)l->grid[k];
    hi[k] = l->edges[k] * (double)(cell[k] + 1) / (double)l->grid[k];
  }
  for (int64_t r = 0; r < l->rows[d]; ++r)
    for (int k = 0; k < 3; ++k) {
      double v = *at3(x, d, r, k);
      if (v < lo[k])
        lo[k] = v;
      if (v > hi[k])
        hi[k] = v;
    }
}

/* Visits the ghosts of domain `d`: each image y = x_j + n L of each particle
   j, in the order of the set and then of n, other than the domain's own
   particles at n = 0, within `reach` of the box. With `out`, stores them;
   returns their number. */
static int64_t visitGhosts(struct Halo *h, struct Memref3D *x, int64_t d,
                           int store) {
  struct Layout *l = h->layout;
  double lo[3], hi[3];
  domainBox(l, x, d, lo, hi);
  int64_t span[3];
  for (int k = 0; k < 3; ++k)
    span[k] = (int64_t)ceil(h->reach / l->edges[k]);
  int64_t count = 0;
  double reach2 = h->reach * h->reach;
  for (int64_t j = 0; j < l->count; ++j) {
    int64_t e = l->owner[j], r = l->row[j];
    for (int64_t na = -span[0]; na <= span[0]; ++na)
      for (int64_t nb = -span[1]; nb <= span[1]; ++nb)
        for (int64_t nc = -span[2]; nc <= span[2]; ++nc) {
          if (e == d && na == 0 && nb == 0 && nc == 0)
            continue;
          int64_t n[3] = {na, nb, nc};
          double dist2 = 0.0;
          for (int k = 0; k < 3; ++k) {
            double y = *at3(x, e, r, k) + (double)n[k] * l->edges[k];
            double out = y < lo[k] ? lo[k] - y : (y > hi[k] ? y - hi[k] : 0.0);
            dist2 += out * out;
          }
          if (!(dist2 < reach2))
            continue;
          if (store) {
            int64_t g = d * h->capacity + count;
            h->sourceDomain[g] = e;
            h->sourceRow[g] = r;
            h->ids[g] = j;
            for (int k = 0; k < 3; ++k)
              h->image[3 * g + k] = n[k];
          }
          ++count;
        }
  }
  return count;
}

int64_t _mlir_ciface_mdrtDistHalo(int64_t layout, struct Memref3D *x,
                                  double reach) {
  struct Halo *h = calloc(1, sizeof *h);
  if (!h)
    fail("out of memory");
  h->references = 1;
  h->layout = (struct Layout *)(intptr_t)layout;
  mdrtDistRetainLayout(layout);
  h->reach = reach;
  shareRows(h->layout, h->layout->rows, x);
  int64_t domains = h->layout->domains;
  h->rows = calloc(domains, sizeof(int64_t));
  if (!h->rows)
    fail("out of memory");
  for (int64_t d = 0; d < domains; ++d) {
    h->rows[d] = visitGhosts(h, x, d, 0);
    if (h->rows[d] > h->capacity)
      h->capacity = h->rows[d];
  }
  int64_t cap = h->capacity > 0 ? h->capacity : 1;
  h->capacity = cap;
  h->sourceDomain = malloc(sizeof(int64_t) * domains * cap);
  h->sourceRow = malloc(sizeof(int64_t) * domains * cap);
  h->ids = malloc(sizeof(int64_t) * domains * cap);
  h->image = malloc(sizeof(int64_t) * 3 * domains * cap);
  if (!h->sourceDomain || !h->sourceRow || !h->ids || !h->image)
    fail("out of memory");
  for (int64_t d = 0; d < domains; ++d)
    visitGhosts(h, x, d, 1);
  if (observer) {
    // Count neighbor multiplicities once per halo, including periodic
    // self-images. Later transfers scale these counts by their field width.
    int64_t *counts = calloc(domains, sizeof *counts);
    if (!counts)
      fail("out of memory counting halo imports");
    for (int64_t d = 0; d < domains; ++d) {
      double lo[3], hi[3];
      domainBox(h->layout, x, d, lo, hi);
      observer->domain(observer->context, d, h->layout->rows[d], h->rows[d], lo,
                       hi);
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
  if (tracing())
    for (int64_t d = 0; d < domains; ++d)
      fprintf(stderr,
              "mdrt_dist: halo reach %g: domain %lld owns %lld rows, "
              "holds %lld ghosts\n",
              reach, (long long)d, (long long)h->layout->rows[d],
              (long long)h->rows[d]);
  return (int64_t)(intptr_t)h;
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
      reference->sizes[1] < l->count || current->sizes[1] < l->count ||
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
  const struct Layout *l = ((struct Halo *)(intptr_t)handle)->layout;
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

/* Index the imported images by identity. An atom may have several images
   on one domain, including periodic images of a locally owned atom. */
void _mlir_ciface_mdrtDistTupleRows(int64_t handle, struct Memref3D *positions,
                                    struct Members *members, struct Memref2I *out) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  struct Layout *l = h->layout;
  int64_t count = members->sizes[0], arity = members->sizes[1];
  if (arity < 2 || out->sizes[0] != count || out->sizes[1] != 1 + arity)
    fail("invalid tuple ownership schedule shape");
  if (count == 0) return;
  shareRows(l, l->rows, positions);
  int64_t *first = malloc(l->domains * l->count * sizeof *first);
  int64_t *next = malloc(l->domains * h->capacity * sizeof *next);
  int64_t *ids = malloc(arity * sizeof *ids);
  int64_t (*image)[3] = calloc(arity, sizeof *image);
  if (!first || !next || !ids || !image) fail("out of memory locating tuples");
  for (int64_t d = 0; d < l->domains; ++d) {
    for (int64_t j = 0; j < l->count; ++j)
      first[d*l->count + j] = -1;
    for (int64_t r = 0; r < h->rows[d]; ++r) {
      int64_t slot = d*h->capacity + r;
      int64_t *head = &first[d*l->count + h->ids[slot]];
      next[slot] = *head;
      *head = r;
    }
  }
  for (int64_t t = 0; t < count; ++t) {
    int64_t anchor = l->count;
    for (int64_t a = 0; a < arity; ++a) {
      ids[a] = members->aligned[members->offset + t*members->strides[0] + a*members->strides[1]];
      if (ids[a] < 0 || ids[a] >= l->count) fail("tuple member outside particle set");
      if (ids[a] < anchor) anchor = ids[a];
    }
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
    int64_t *record = out->aligned + out->offset + t*out->strides[0];
    record[0] = d;
    for (int64_t a = 0; a < arity; ++a) {
      int64_t j = ids[a], row = -1;
      if (l->owner[j] == d && !image[a][0] && !image[a][1] && !image[a][2])
        row = l->row[j];
      for (int64_t g = first[d*l->count+j]; row < 0 && g >= 0; g = next[d*h->capacity+g]) {
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
  free(next);
  free(first);
}

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
static void exchangeHalo(struct Halo *h, struct Memref3D *owned,
                         struct Memref3D *ghosts, int image, int transpose,
                         struct Memref3W *ownedWords, struct Memref3W *ghostWords) {
  const int peers = mdrtMPISize(), rank = mdrtMPIRank();
  int *counts = calloc((size_t)peers * 6, sizeof(int));
  if (!counts) mdrtMPIAbort("out of memory packing halo counts");
  int *ns = counts, *nr = ns + peers, *os = nr + peers, *or = os + peers;
  int *ws = or + peers, *wr = ws + peers;
  const int64_t domains = h->layout->domains, components = ownedWords ? ownedWords->sizes[2] : owned->sizes[2];
  for (int64_t d = 0; d < domains; ++d) {
    int dst = mdrtMPIOwner(domains, d);
    for (int64_t g = 0; g < h->rows[d]; ++g) {
      int src = mdrtMPIOwner(domains, h->sourceDomain[d * h->capacity + g]);
      if (src == dst) continue;
      int sender = transpose ? dst : src, receiver = transpose ? src : dst;
      if (sender == rank) {
        if (ns[receiver] > INT_MAX - components) mdrtMPIAbort("halo count exceeds MPI int");
        ns[receiver] += components;
      }
      if (receiver == rank) {
        if (nr[sender] > INT_MAX - components) mdrtMPIAbort("halo count exceeds MPI int");
        nr[sender] += components;
      }
    }
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
  for (int pass = 0; pass < 2; ++pass) {
    if (pass) mdrtMPIExchange(send, ns, os, recv, nr, or, ownedWords != NULL);
    for (int64_t d = 0; d < domains; ++d) {
      int dst = mdrtMPIOwner(domains, d);
      for (int64_t g = 0; g < h->rows[d]; ++g) {
        int64_t i = d * h->capacity + g;
        int64_t dom = h->sourceDomain[i], row = h->sourceRow[i];
        int src = mdrtMPIOwner(domains, dom);
        int sender = transpose ? dst : src, receiver = transpose ? src : dst;
        if ((!pass && (sender != rank || receiver == rank)) ||
            (pass && receiver != rank)) continue;
        for (int64_t c = 0; c < components; ++c) {
          if (ownedWords) {
            DistWide *from = wordAt(ghostWords, d, g, c);
            if (!pass) ((DistWide *)send)[ws[receiver]++] = *from;
            else {
              DistWide term = sender == rank ? *from : ((DistWide *)recv)[wr[sender]++];
              DistWide *to = wordAt(ownedWords, dom, row, c);
              *to = mdrtDistAddWide(*to, term);
            }
            continue;
          }
          double *a = at3(owned, dom, row, c), *b = at3(ghosts, d, g, c);
          if (!pass) ((double *)send)[ws[receiver]++] = transpose ? *b : *a;
          else {
            double v = sender == rank ? (transpose ? *b : *a) : ((double *)recv)[wr[sender]++];
            if (transpose) *a += v;
            else {
              if (image) v += (double)h->image[3*i+c] * h->layout->edges[c];
              *b = v;
            }
          }
        }
      }
    }
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
                         edge.rows * components * (int64_t)sizeof(DistWide));
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
        *at3(out, d, r, c) = (double)mdrtDistNarrowWord(*wordAt(owned, d, r, c), DistPairForce) * 0x1p-40;
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
                         edge.rows * owned->sizes[2] * (int64_t)sizeof(double));
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
  free(h->rows);
  free(h->edges);
  free(h->sourceDomain);
  free(h->sourceRow);
  free(h->ids);
  free(h->image);
  free(h);
}

/* The precondition under which the relation of md_dist is that of
   md.neighborhood: at most one image of a particle within the cutoff of
   another, a cutoff less than half the least edge. */
void mdrtDistCheckCutoff(double cutoff, double lx, double ly, double lz) {
  double least = lx < ly ? (lx < lz ? lx : lz) : (ly < lz ? ly : lz);
  if (!(2.0 * cutoff < least))
    fail("the cutoff is not less than half the least edge of the cell");
}
