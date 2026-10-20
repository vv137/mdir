/* The FFT of particle mesh Ewald on the host, from pocketfft
   [Reinecke2019] (docs/pme-m1.md).

   Compiled code calls these functions for blocks of the lines of a grid,
   from a parallel loop over the blocks of each of the three passes of a
   transform (D271): the real lines along z, then the complex
   lines along y and along x (backward, in the reverse order). Each line is
   transformed by the same code whatever thread takes it, and nothing is
   summed across lines, so the result does not depend on the number of
   threads. The arithmetic of a line is in f64; a grid in f32 (the mixed
   mode) is converted where its lines are copied for pocketfft, which they
   are anyway. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "pocketfft.h"

/* The plans of the lengths that a run uses, kept for the run. pocketfft
   reads a plan and allocates its own scratch when it transforms, so threads
   may share one; making a plan is not safe in parallel, so
   `mdrtFFTPrepare` makes those of a grid before its parallel loops. */
enum { numPlans = 32 };
static struct {
  size_t length;
  rfft_plan plan;
} realPlans[numPlans];
static struct {
  size_t length;
  cfft_plan plan;
} complexPlans[numPlans];

static rfft_plan getRealPlan(size_t length) {
  for (int i = 0; i != numPlans; ++i) {
    if (realPlans[i].length == length)
      return realPlans[i].plan;
    if (realPlans[i].length == 0) {
      realPlans[i].plan = make_rfft_plan(length);
      if (!realPlans[i].plan)
        break;
      realPlans[i].length = length;
      return realPlans[i].plan;
    }
  }
  fprintf(stderr, "mdrt: cannot make a plan of FFT of length %zu\n", length);
  abort();
}

static cfft_plan getComplexPlan(size_t length) {
  for (int i = 0; i != numPlans; ++i) {
    if (complexPlans[i].length == length)
      return complexPlans[i].plan;
    if (complexPlans[i].length == 0) {
      complexPlans[i].plan = make_cfft_plan(length);
      if (!complexPlans[i].plan)
        break;
      complexPlans[i].length = length;
      return complexPlans[i].plan;
    }
  }
  fprintf(stderr, "mdrt: cannot make a plan of FFT of length %zu\n", length);
  abort();
}

/* Makes the plans of a grid of k1 x k2 x k3 points, before the parallel
   loops of its transforms. */
void mdrtFFTPrepare(int64_t k1, int64_t k2, int64_t k3) {
  getRealPlan((size_t)k3);
  getComplexPlan((size_t)k1);
  getComplexPlan((size_t)k2);
}

static double *allocateLines(size_t count) {
  double *lines = (double *)malloc(count * sizeof(double));
  if (!lines) {
    fprintf(stderr, "mdrt: out of memory in the FFT\n");
    abort();
  }
  return lines;
}

static void check(int status) {
  if (status != 0) {
    fprintf(stderr, "mdrt: pocketfft failed (out of memory)\n");
    abort();
  }
}

/* The complex lines of a pass along y or x are taken in tiles of this many
   neighboring columns of z, which lie next to one another in memory, so
   that the strided lines are read and written by cache lines. */
enum { tileWidth = 8 };

/* The number of blocks of a pass along y (`planes` = k1) or along x
   (`planes` = k2): a block for each tile of the k3 / 2 + 1 columns of z of
   each plane. */
int64_t mdrtFFTColumnBlocks(int64_t planes, int64_t k3) {
  int64_t h = k3 / 2 + 1;
  return planes * ((h + tileWidth - 1) / tileWidth);
}

/* The functions for a grid of the type T, named with SUFFIX: the
   descriptor of a buffer of one dimension as the C interface of MLIR
   passes it, and the transforms of the three passes. */
#define MDRT_FFT_FUNCTIONS(T, SUFFIX)                                          \
  typedef struct {                                                             \
    T *allocated;                                                              \
    T *aligned;                                                                \
    int64_t offset;                                                            \
    int64_t sizes[1];                                                          \
    int64_t strides[1];                                                        \
  } Buffer##SUFFIX;                                                            \
                                                                               \
  void _mlir_ciface_mdrtFFT##SUFFIX##RowsForward(                              \
      Buffer##SUFFIX *real, Buffer##SUFFIX *complex, int64_t k3,               \
      int64_t first, int64_t last) {                                           \
    const T *r = real->aligned + real->offset;                                 \
    T *c = complex->aligned + complex->offset;                                 \
    int64_t h = k3 / 2 + 1;                                                    \
    rfft_plan plan = getRealPlan((size_t)k3);                                  \
    double *line = allocateLines((size_t)k3);                                  \
    for (int64_t row = first; row < last; ++row) {                             \
      for (int64_t z = 0; z != k3; ++z)                                        \
        line[z] = r[row * k3 + z];                                             \
      check(rfft_forward(plan, line, 1.0));                                    \
      /* From r0, r1, i1, r2, i2, ... to pairs. */                             \
      T *out = c + 2 * row * h;                                                \
      out[0] = (T)line[0];                                                     \
      out[1] = 0;                                                              \
      for (int64_t z = 1; z != h; ++z) {                                       \
        out[2 * z] = (T)line[2 * z - 1];                                       \
        out[2 * z + 1] = 2 * z < k3 ? (T)line[2 * z] : 0;                      \
      }                                                                        \
    }                                                                          \
    free(line);                                                                \
  }                                                                            \
                                                                               \
  void _mlir_ciface_mdrtFFT##SUFFIX##RowsBackward(                             \
      Buffer##SUFFIX *complex, Buffer##SUFFIX *real, int64_t k3,               \
      int64_t first, int64_t last) {                                           \
    const T *c = complex->aligned + complex->offset;                           \
    T *r = real->aligned + real->offset;                                       \
    int64_t h = k3 / 2 + 1;                                                    \
    rfft_plan plan = getRealPlan((size_t)k3);                                  \
    double *line = allocateLines((size_t)k3);                                  \
    for (int64_t row = first; row < last; ++row) {                             \
      const T *in = c + 2 * row * h;                                           \
      line[0] = in[0];                                                         \
      for (int64_t z = 1; z != h; ++z) {                                       \
        line[2 * z - 1] = in[2 * z];                                           \
        if (2 * z < k3)                                                        \
          line[2 * z] = in[2 * z + 1];                                         \
      }                                                                        \
      check(rfft_backward(plan, line, 1.0));                                   \
      for (int64_t z = 0; z != k3; ++z)                                        \
        r[row * k3 + z] = (T)line[z];                                          \
    }                                                                          \
    free(line);                                                                \
  }                                                                            \
                                                                               \
  void _mlir_ciface_mdrtFFT##SUFFIX##Columns(                                  \
      Buffer##SUFFIX *complex, int64_t k1, int64_t k2, int64_t k3,             \
      int64_t axis, int64_t forward, int64_t first, int64_t last) {            \
    T *c = complex->aligned + complex->offset;                                 \
    int64_t h = k3 / 2 + 1;                                                    \
    int64_t tiles = (h + tileWidth - 1) / tileWidth;                           \
    int64_t length = axis == 1 ? k2 : k1;                                      \
    /* The strides of a line and of the planes, in complex numbers. */         \
    int64_t along = axis == 1 ? h : k2 * h;                                    \
    int64_t across = axis == 1 ? k2 * h : h;                                   \
    cfft_plan plan = getComplexPlan((size_t)length);                           \
    double *lines = allocateLines((size_t)(2 * tileWidth * length));           \
    for (int64_t u = first; u < last; ++u) {                                   \
      int64_t plane = u / tiles;                                               \
      int64_t z0 = (u % tiles) * tileWidth;                                    \
      int64_t width = h - z0 < tileWidth ? h - z0 : tileWidth;                 \
      T *base = c + 2 * (plane * across + z0);                                 \
      for (int64_t j = 0; j != length; ++j)                                    \
        for (int64_t t = 0; t != width; ++t) {                                 \
          lines[2 * (t * length + j)] = base[2 * (j * along + t)];             \
          lines[2 * (t * length + j) + 1] = base[2 * (j * along + t) + 1];     \
        }                                                                      \
      for (int64_t t = 0; t != width; ++t)                                     \
        check(forward ? cfft_forward(plan, lines + 2 * t * length, 1.0)        \
                      : cfft_backward(plan, lines + 2 * t * length, 1.0));     \
      for (int64_t j = 0; j != length; ++j)                                    \
        for (int64_t t = 0; t != width; ++t) {                                 \
          base[2 * (j * along + t)] = (T)lines[2 * (t * length + j)];          \
          base[2 * (j * along + t) + 1] = (T)lines[2 * (t * length + j) + 1];  \
        }                                                                      \
    }                                                                          \
    free(lines);                                                               \
  }

/* mdrtFFT64RowsForward (and 32, for a grid in f32): the forward
   transforms, exp(−2π i k·m / K), of the real lines along z of rows
   [first, last) of the k1 x k2 rows of `real` into the half-complex grid
   `complex` of k1 x k2 x (k3 / 2 + 1) numbers, interleaved.

   mdrtFFT64RowsBackward: the backward transforms, exp(+2π i k·m / K), not
   normalized, of rows [first, last) of the half-complex grid into the real
   grid.

   mdrtFFT64Columns: the complex transforms of the half-complex grid, in
   place, along y (`axis` 1) or along x (`axis` 0), forward or backward,
   for the blocks [first, last) of mdrtFFTColumnBlocks: block u is tile
   u mod t of the columns of z of plane u / t, a plane of x along y and of
   y along x. */
MDRT_FFT_FUNCTIONS(double, 64)
MDRT_FFT_FUNCTIONS(float, 32)

/* The execution-level linear transform uses the same line transforms as
   the PME lowering. Its contiguous buffers expose the transform as one
   schedulable operation without making spread, influence, or gather part
   of it (D[pme-primitives]; Reinecke2019). */
#define MDRT_FFT_3D(T, SUFFIX)                                                \
  void _mlir_ciface_mdrtFFT##SUFFIX##Forward3D(                               \
      Buffer##SUFFIX *real, Buffer##SUFFIX *complex, int64_t k1,              \
      int64_t k2, int64_t k3) {                                               \
    mdrtFFTPrepare(k1, k2, k3);                                               \
    for (int64_t x = 0; x < k1; ++x)                                           \
      _mlir_ciface_mdrtFFT##SUFFIX##RowsForward(                               \
          real, complex, k3, x * k2, (x + 1) * k2);                           \
    int64_t ny = mdrtFFTColumnBlocks(k1, k3);                                 \
    for (int64_t u = 0; u < ny; ++u)                                           \
      _mlir_ciface_mdrtFFT##SUFFIX##Columns(                                   \
          complex, k1, k2, k3, 1, 1, u, u + 1);                               \
    int64_t nx = mdrtFFTColumnBlocks(k2, k3);                                 \
    for (int64_t u = 0; u < nx; ++u)                                           \
      _mlir_ciface_mdrtFFT##SUFFIX##Columns(                                   \
          complex, k1, k2, k3, 0, 1, u, u + 1);                               \
  }                                                                            \
  void _mlir_ciface_mdrtFFT##SUFFIX##Backward3D(                              \
      Buffer##SUFFIX *complex, Buffer##SUFFIX *real, int64_t k1,              \
      int64_t k2, int64_t k3) {                                               \
    mdrtFFTPrepare(k1, k2, k3);                                               \
    int64_t ny = mdrtFFTColumnBlocks(k1, k3);                                 \
    for (int64_t u = 0; u < ny; ++u)                                           \
      _mlir_ciface_mdrtFFT##SUFFIX##Columns(                                   \
          complex, k1, k2, k3, 1, 0, u, u + 1);                               \
    int64_t nx = mdrtFFTColumnBlocks(k2, k3);                                 \
    for (int64_t u = 0; u < nx; ++u)                                           \
      _mlir_ciface_mdrtFFT##SUFFIX##Columns(                                   \
          complex, k1, k2, k3, 0, 0, u, u + 1);                               \
    for (int64_t x = 0; x < k1; ++x)                                           \
      _mlir_ciface_mdrtFFT##SUFFIX##RowsBackward(                              \
          complex, real, k3, x * k2, (x + 1) * k2);                           \
  }

MDRT_FFT_3D(double, 64)
MDRT_FFT_3D(float, 32)
