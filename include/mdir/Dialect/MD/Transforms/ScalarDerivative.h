// Forward-mode differentiation of scalar computations.

#ifndef MDIR_DIALECT_MD_TRANSFORMS_SCALARDERIVATIVE_H
#define MDIR_DIALECT_MD_TRANSFORMS_SCALARDERIVATIVE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>

namespace mdir {
namespace md {

/// Emits floating-point arithmetic on scalars and fixed-size vectors.
///
/// A null value stands for zero. Every method accepts null operands and
/// returns null when the result is known to be zero, so that no op is
/// emitted for a term that vanishes.
class ScalarEmitter {
public:
  ScalarEmitter(mlir::OpBuilder &builder, mlir::Location loc)
      : builder(builder), loc(loc) {}

  /// A constant of type `type`, which is f64 or a vector of f64.
  mlir::Value constant(double value, mlir::Type type);

  /// A constant of the type of `like`, never null, even for zero.
  mlir::Value constantLike(double value, mlir::Value like);

  mlir::Value add(mlir::Value lhs, mlir::Value rhs);
  mlir::Value sub(mlir::Value lhs, mlir::Value rhs);
  mlir::Value mul(mlir::Value lhs, mlir::Value rhs);
  mlir::Value div(mlir::Value lhs, mlir::Value rhs);
  mlir::Value neg(mlir::Value value);

  /// `factor * value`, with `factor` a compile-time constant.
  mlir::Value scale(double factor, mlir::Value value);

  mlir::OpBuilder &builder;
  mlir::Location loc;
};

/// Computes derivatives with respect to one variable, in forward mode.
///
/// The derivative of a value is emitted at the insertion point of the
/// builder, which must be dominated by every value that is differentiated.
class ScalarDerivative {
public:
  /// Called for a value that no rule covers. Sets `tangent` to the
  /// derivative, or to null for zero. Returns failure if the value cannot be
  /// differentiated.
  using LeafHandler =
      std::function<mlir::LogicalResult(mlir::Value value,
                                        mlir::Value &tangent)>;

  /// The derivative with respect to `variable`; for a vector, along
  /// `seed`, a vector of its type that dominates the derivative (a unit
  /// vector gives the derivative with respect to one component).
  ScalarDerivative(mlir::OpBuilder &builder, mlir::Value variable,
                   LeafHandler leafHandler = nullptr,
                   mlir::Value seed = mlir::Value())
      : builder(builder), variable(variable),
        leafHandler(std::move(leafHandler)), seed(seed) {}

  /// Sets `tangent` to the derivative of `value`, or to null if the
  /// derivative is zero. Emits a diagnostic and returns failure if an op has
  /// no derivative rule.
  mlir::LogicalResult get(mlir::Value value, mlir::Value &tangent);

  mlir::OpBuilder &getBuilder() { return builder; }

private:
  mlir::LogicalResult compute(mlir::Value value, mlir::Value &tangent);

  mlir::OpBuilder &builder;
  mlir::Value variable;
  LeafHandler leafHandler;
  mlir::Value seed;
  llvm::DenseMap<mlir::Value, mlir::Value> tangents;
};

/// Erases the ops of `block` that have no uses and no side effects,
/// including the ops that become unused as a result.
void eraseDeadOps(mlir::Block &block);

} // namespace md
} // namespace mdir

#endif // MDIR_DIALECT_MD_TRANSFORMS_SCALARDERIVATIVE_H
