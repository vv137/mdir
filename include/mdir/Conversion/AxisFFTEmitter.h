// Batched device transforms on a real mesh and its pitched half-spectrum.
#pragma once

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace mdir::kernels {
inline constexpr llvm::StringLiteral kAxisFFTPlans = "mdir.pme_fft_plans";
struct AxisFactors {
  int first, second;
};

inline std::optional<AxisFactors> factorAxis(int64_t n) {
  if (n < 1 || n > 32 * 32)
    return std::nullopt;
  int64_t rest = n;
  for (int radix : {2, 3, 5, 7})
    while (rest % radix == 0)
      rest /= radix;
  if (rest != 1)
    return std::nullopt;
  for (int a = 32; a >= 1; --a)
    if (a * a <= n && n % a == 0 && n / a <= 32)
      return AxisFactors{a, int(n / a)};
  return std::nullopt;
}

inline bool supportsAxisFFT(llvm::ArrayRef<int64_t> grid) {
  return grid.size() == 3 && grid[2] % 2 == 0 && factorAxis(grid[0]) &&
         factorAxis(grid[1]) && factorAxis(grid[2] / 2);
}
inline int64_t axisFFTPitch(int64_t z) { return (z / 2 + 4) / 4 * 4; }

struct FFTCoefficient {
  mlir::Value re, im;
};
using FFTPointwise = llvm::function_ref<FFTCoefficient(
    mlir::OpBuilder &, mlir::Value x, mlir::Value y, mlir::Value z,
    FFTCoefficient, llvm::MutableArrayRef<mlir::Value> sums)>;
using FFTPartial =
    llvm::function_ref<void(mlir::Value block, mlir::ValueRange sums)>;

// Returns the number of X block partials.
int emitAxisConvolution(mlir::OpBuilder &builder, mlir::Location loc,
                        llvm::ArrayRef<int64_t> grid, int64_t pitch,
                        mlir::Value real, mlir::Value spectrum,
                        FFTPointwise pointwise, int sumCount,
                        FFTPartial partial);
} // namespace mdir::kernels
