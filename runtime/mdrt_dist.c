/* The in-process transport of md_dist (D[ml2-core], docs/distributed-ml2.md,
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

static int tracing(void) {
  const char *value = getenv("MDRT_DIST_TRACE");
  return value && *value;
}

static void fail(const char *message) {
  fprintf(stderr, "mdrt_dist: %s\n", message);
  fflush(stderr);
  abort();
}

struct Layout {
  int64_t grid[3], domains, count;
  double edges[3];
  int64_t *owner; /* [count]: the domain that owns each particle */
  int64_t *row;   /* [count]: its owned row there */
  int64_t *rows;  /* [domains]: the number of owned rows */
  int64_t *ids;   /* [domains * count]: the particle of each owned row */
};

struct Halo {
  struct Layout *layout;
  double reach;
  int64_t capacity; /* the most ghosts of one domain */
  int64_t *rows;    /* [domains] */
  /* [domains * capacity] each: the source's owner, its owned row, its
     identity, and the image n (3 per ghost). */
  int64_t *sourceDomain, *sourceRow, *ids, *image;
};

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

int64_t _mlir_ciface_mdrtDistPartition(struct Memref2D *x, int64_t gx,
                                       int64_t gy, int64_t gz, double lx,
                                       double ly, double lz) {
  struct Layout *l = calloc(1, sizeof *l);
  if (!l)
    fail("out of memory");
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

void mdrtDistReleaseLayout(int64_t handle) {
  struct Layout *l = (struct Layout *)(intptr_t)handle;
  free(l->owner);
  free(l->row);
  free(l->rows);
  free(l->ids);
  free(l);
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
  h->layout = (struct Layout *)(intptr_t)layout;
  h->reach = reach;
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

/* G: (G a)_g = a_{s(g)}, with `image` x_{s(g)} + n_g L. */
void _mlir_ciface_mdrtDistHaloCopy(int64_t handle, struct Memref3D *owned,
                                   struct Memref3D *ghosts, int64_t image) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
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

void mdrtDistReleaseHalo(int64_t handle) {
  struct Halo *h = (struct Halo *)(intptr_t)handle;
  free(h->rows);
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
