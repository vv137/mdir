// The ops of functions of the squared distance.
//
// md-exec-simplify-distance takes a function of the distance alone out of a
// pair kernel, and md-exec-expand-radial fits cubics to it in f32 (D94). A
// cubic fits a smooth function only: floor, a comparison, or a minimum
// make the function jump or bend, and stay in the kernel.

#ifndef MDIR_DIALECT_MDEXEC_TRANSFORMS_RADIAL_H
#define MDIR_DIALECT_MDEXEC_TRANSFORMS_RADIAL_H

#include "llvm/ADT/ArrayRef.h"

#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace mdir {
namespace md_exec {

/// Whether a function of the squared distance may hold `op`.
bool isRadialOp(mlir::Operation *op);

/// The value of `op`, one that isRadialOp accepts, from the values of its
/// operands; none for any other op.
std::optional<double> evaluateRadialOp(mlir::Operation *op,
                                       llvm::ArrayRef<double> operands);

/// The tolerance of the tables of md-exec-expand-radial by default.
constexpr double radialTolerance = 3.0e-7;

/// Whether the function `function` (a `func.func` of the squared distance)
/// fits a table up to `cutoff` within `tolerance`: one with a pole at the
/// cutoff, or that vanishes there with all its derivatives, does not, and
/// stays in its kernel (D159).
bool canTabulate(mlir::Operation *function, double cutoff, double tolerance);

} // namespace md_exec
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_TRANSFORMS_RADIAL_H
