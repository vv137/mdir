#include <cufft.h>
#include "pme_callback.h"

// Match the rectangular Gaussian scale kernel, including its zero DC mode.
__device__ cufftComplex mdirGaussianLoad(void *in, unsigned long long offset,
                                        void *caller, void *) {
  const auto &info = *static_cast<const PMECallbackInfo *>(caller);
  unsigned depth = info.nz / 2 + 1;
  unsigned row = static_cast<unsigned>(offset) / depth;
  unsigned z = static_cast<unsigned>(offset) % depth;
  unsigned x = row / info.ny, y = row % info.ny;
  const float *tx = info.tables;
  const float *ty = tx + 3 * info.nx;
  const float *tz = ty + 3 * info.ny;
  float squared = (tx[info.nx + x] + ty[info.ny + y]) + tz[info.nz + z];
  float product = (tx[2 * info.nx + x] * ty[2 * info.ny + y]) *
                  tz[2 * info.nz + z];
  float factor = squared == 0 ? 0 : product * (1.0f / squared);
  cufftComplex v = static_cast<const cufftComplex *>(in)[offset];
  return {v.x * factor, v.y * factor};
}
