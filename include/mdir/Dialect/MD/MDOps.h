// Ops of the md dialect.

#ifndef MDIR_DIALECT_MD_MDOPS_H
#define MDIR_DIALECT_MD_MDOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDTypes.h"

#define GET_OP_CLASSES
#include "mdir/Dialect/MD/MDOps.h.inc"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mdir {
namespace md {

inline constexpr llvm::StringLiteral kStageRowsAttrName = "md.stage_rows";

/// Static row count retained when a stage body is bufferized.
mlir::IntegerAttr getStageRows(mlir::Operation *body);
void setStageRows(mlir::Operation *body, int64_t rows);
mlir::LogicalResult verifyStageRowsAttribute(mlir::Operation *body,
                                              mlir::Attribute value);

inline constexpr llvm::StringLiteral kStageBodyAttrName = "md.stage_body";

/// Marks an outlined stage body whose tensor form was already verified.
bool hasStageBody(mlir::Operation *body);
void markStageBody(mlir::Operation *body);
mlir::LogicalResult verifyStageBodyAttribute(mlir::Operation *body,
                                              mlir::Attribute value);

/// Σ c_k T_k(y) by Clenshaw's recurrence [Clenshaw1955], in f64.
double evaluateChebyshev(llvm::ArrayRef<double> coefficients, double y);

/// The coefficients of the derivative of Σ c_k T_k(y) in y, one fewer
/// (none for a constant).
llvm::SmallVector<double> getChebyshevDerivative(
    llvm::ArrayRef<double> coefficients);

/// Emits Σ c_k T_k(y) by Clenshaw's recurrence in the type of `y`.
mlir::Value emitChebyshev(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value y,
                          llvm::ArrayRef<double> coefficients);

} // namespace md
} // namespace mdir

#endif // MDIR_DIALECT_MD_MDOPS_H
