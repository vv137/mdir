// Conservative cell deformation for the mobility-group validity tests.
#ifndef MDIR_CONVERSION_TEMPLATEVALIDITYKERNELS_H
#define MDIR_CONVERSION_TEMPLATEVALIDITYKERNELS_H
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include <array>
namespace mdir::kernels {
/// A nonnegative lower bound on the least singular value of the Cartesian
/// affine map from reference cell to current cell. Inputs are valid reduced
/// cells in f64, with the same representation (three or six components).
/// Identical cells return one exactly. Rectangular cells use the least
/// edge ratio; tilted cells bound the inverse map's induced 1/infinity
/// norms. Rounding guards can only make rebuilding more conservative.
mlir::Value emitTemplateCellScaleLower(mlir::OpBuilder &, mlir::Location,
                                      mlir::Value current, mlir::Value reference);
/// Actual WW/SW/SS reaches, with this addition order shared by the build
/// and the validity theorem. `physical` is vector<3xf64>, skin is f64.
mlir::Value emitTemplateFreshReaches(mlir::OpBuilder &, mlir::Location,
                                    mlir::Value physical, mlir::Value skin);
/// Half-margins for water and atoms. Each is the minimum of its incident
/// class margins. Cap a class margin at its fresh-build margin: replacing
/// SW's cell/reach by a fresh epoch then cannot shrink an inactive group's
/// allowance, including during expansion. Negative values mean stale.
std::array<mlir::Value, 2> emitTemplateGroupMotionLimits(
    mlir::OpBuilder &, mlir::Location, mlir::Value currentCell,
    llvm::ArrayRef<mlir::Value> classCells, mlir::Value builtReaches,
    mlir::Value physicalReaches, mlir::Value skin);
/// Compare an actual f64 framed anchor with the fractional reference
/// point in the current cell. Outward error guards cover the projection,
/// subtraction and squared norm. Does not wrap a face or a winding jump.
mlir::Value emitTemplateMotionExceeded(mlir::OpBuilder &, mlir::Location,
    mlir::Value anchor, mlir::Value referenceFraction, mlir::Value currentCell,
    mlir::Value halfMargin);
/// The f32 square of the limit of the narrow test below, for one launch:
/// ((m - delta)^2 (1 - 2^-20)) narrowed, with delta = 2^-18 (|H|_1 + |m|),
/// or -1 where m - delta is negative or NaN, so that every unit fails.
mlir::Value emitTemplateMotionLimitSquared(mlir::OpBuilder &, mlir::Location,
    mlir::Value halfMargin, mlir::Value cell);
/// The motion test in f32: the narrowed framed anchor p, plus the narrowed
/// H t, less H r with both narrowed, squared and summed in f32, against
/// emitTemplateMotionLimitSquared. True when |r|_inf > 2 or the limit is
/// passed; whenever the true |p + H t - H r| exceeds the half margin m,
/// the test is true (docs/decisions/rigid-water-template.md, U5).
mlir::Value emitTemplateMotionExceededNarrow(mlir::OpBuilder &, mlir::Location,
    mlir::Value anchor, mlir::Value referenceFraction, mlir::Value narrowCell,
    mlir::Value narrowOffset, mlir::Value limitSquared);
}
#endif
