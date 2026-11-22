// Load transformations for the Gaussian cubic PME force-only sequence.
// The table kernel writes this evaluation's pointers and cell coefficients
// into device memory before either transform uses them.
#include <cufft.h>
#include <stdint.h>

struct PMEInfo {
  uint64_t reciprocals;
  const int64_t *wide;
  const int32_t *useWide;
  const void *tables;
  uint64_t k1, k2, k3;
  double h00, h10, h11, h20, h21, h22, gauss;
};
static_assert(sizeof(PMEInfo) == 14 * sizeof(uint64_t));

__device__ float pmeLoadF32(void *input, unsigned long long i, void *data,
                            void *) {
  auto &info = *static_cast<const PMEInfo *>(data);
  float word = *info.useWide
                   ? static_cast<float>(info.wide[i])
                   : static_cast<float>(static_cast<const int32_t *>(input)[i]);
  return word * 0x1p-23f;
}

__device__ double pmeLoadF64(void *input, unsigned long long i, void *,
                             void *) {
  return static_cast<double>(static_cast<const int64_t *>(input)[i]) * 0x1p-40;
}

// ceil(2^32/d) gives a quotient at most one too large.
__device__ unsigned quotient(unsigned n, unsigned d, unsigned reciprocal) {
  unsigned q = __umulhi(n, reciprocal);
  return q - (static_cast<uint64_t>(q) * d > n);
}

template <typename Real, bool tilted>
__device__ Real influence(const PMEInfo &p, unsigned long long i) {
  const Real *x = static_cast<const Real *>(p.tables);
  const Real *y = x + 3 * p.k1;
  const Real *z = y + 3 * p.k2;
  unsigned depth = p.k3 / 2 + 1;
  unsigned index = static_cast<unsigned>(i);
  unsigned row = quotient(index, depth, p.reciprocals >> 32);
  unsigned iz = index - row * depth;
  unsigned ix = quotient(row, p.k2, static_cast<unsigned>(p.reciprocals));
  unsigned iy = row - ix * static_cast<unsigned>(p.k2);
  Real squared, attenuation = 1;
  if constexpr (tilted) {
    Real a = Real(p.h00) * x[ix];
    Real b = Real(p.h10) * x[ix] + Real(p.h11) * y[iy];
    Real c = (Real(p.h20) * x[ix] + Real(p.h21) * y[iy]) + Real(p.h22) * z[iz];
    squared = (a * a + b * b) + c * c;
    attenuation = exp(-Real(p.gauss) * squared);
  } else {
    squared = (x[p.k1 + ix] + y[p.k2 + iy]) + z[p.k3 + iz];
  }
  Real product = (x[2 * p.k1 + ix] * y[2 * p.k2 + iy]) * z[2 * p.k3 + iz];
  return squared == 0 ? Real(0) : (product * attenuation) * (Real(1) / squared);
}

template <typename Real, typename Complex, bool tilted>
__device__ Complex scale(void *input, unsigned long long i, void *data) {
  Complex value = static_cast<const Complex *>(input)[i];
  Real factor = influence<Real, tilted>(*static_cast<const PMEInfo *>(data), i);
  return {value.x * factor, value.y * factor};
}

__device__ cufftComplex pmeScaleF32(void *input, unsigned long long i,
                                    void *data, void *) {
  return scale<float, cufftComplex, false>(input, i, data);
}

__device__ cufftComplex pmeScaleF32Tilted(void *input, unsigned long long i,
                                          void *data, void *) {
  return scale<float, cufftComplex, true>(input, i, data);
}

__device__ cufftDoubleComplex pmeScaleF64(void *input, unsigned long long i,
                                          void *data, void *) {
  return scale<double, cufftDoubleComplex, false>(input, i, data);
}

__device__ cufftDoubleComplex pmeScaleF64Tilted(void *input,
                                                unsigned long long i,
                                                void *data, void *) {
  return scale<double, cufftDoubleComplex, true>(input, i, data);
}

// Each half-spectrum mode has a cell-dependent influence, prepared once.
__device__ cufftComplex pmeScaleFlatF32(void *input, unsigned long long i,
                                        void *factors, void *) {
  cufftComplex value = static_cast<const cufftComplex *>(input)[i];
  float w = static_cast<const float *>(factors)[i];
  return {value.x * w, value.y * w};
}

__device__ cufftDoubleComplex pmeScaleFlatF64(void *input, unsigned long long i,
                                              void *factors, void *) {
  cufftDoubleComplex value = static_cast<const cufftDoubleComplex *>(input)[i];
  double w = static_cast<const double *>(factors)[i];
  return {value.x * w, value.y * w};
}
