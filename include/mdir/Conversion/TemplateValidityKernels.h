// Conservative cell deformation for the two mobility-group validity tests.
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
/// Optional shared norms must be the unchanged ADD reduction of abs(anchor)
/// and abs(currentCell), respectively, in their original f64 precision.
/// Absent operands retain the original independently emitted operations.
mlir::Value emitTemplateMotionExceeded(mlir::OpBuilder &, mlir::Location,
    mlir::Value anchor, mlir::Value referenceFraction, mlir::Value currentCell,
    mlir::Value halfMargin, mlir::Value anchorL1 = {}, mlir::Value cellL1 = {});
}
#endif
