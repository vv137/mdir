// Forward-mode differentiation of scalar computations.
//
// The derivative rules are those of docs/ops-m0.md, Section 5.4.

#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"

#include "mdir/Dialect/MD/MDOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cmath>

using namespace mlir;
using namespace mdir::md;

//===----------------------------------------------------------------------===//
// ScalarEmitter
//===----------------------------------------------------------------------===//

Value ScalarEmitter::constant(double value, Type type) {
  if (auto vector = dyn_cast<VectorType>(type)) {
    auto element = cast<FloatType>(vector.getElementType());
    APFloat scalar(value);
    bool losesInfo = false;
    scalar.convert(element.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                   &losesInfo);
    return arith::ConstantOp::create(
        builder, loc, type, DenseElementsAttr::get(vector, scalar));
  }
  return arith::ConstantOp::create(builder, loc, type,
                                   builder.getFloatAttr(type, value));
}

Value ScalarEmitter::constantLike(double value, Value like) {
  return constant(value, like.getType());
}

/// Returns true if `value` is a constant whose elements all equal
/// `expected`.
static bool isConstant(Value value, double expected) {
  APFloat constant(0.0);
  if (!value || !matchPattern(value, m_ConstantFloat(&constant)))
    return false;
  bool losesInfo = false;
  constant.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                   &losesInfo);
  return constant.convertToDouble() == expected;
}

Value ScalarEmitter::add(Value lhs, Value rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return arith::AddFOp::create(builder, loc, lhs, rhs);
}

Value ScalarEmitter::sub(Value lhs, Value rhs) {
  if (!rhs)
    return lhs;
  if (!lhs)
    return neg(rhs);
  return arith::SubFOp::create(builder, loc, lhs, rhs);
}

Value ScalarEmitter::mul(Value lhs, Value rhs) {
  if (!lhs || !rhs)
    return Value();
  if (isConstant(lhs, 1.0))
    return rhs;
  if (isConstant(rhs, 1.0))
    return lhs;
  return arith::MulFOp::create(builder, loc, lhs, rhs);
}

Value ScalarEmitter::div(Value lhs, Value rhs) {
  assert(rhs && "division by a value that is known to be zero");
  if (!lhs)
    return Value();
  if (isConstant(rhs, 1.0))
    return lhs;
  return arith::DivFOp::create(builder, loc, lhs, rhs);
}

Value ScalarEmitter::neg(Value value) {
  if (!value)
    return Value();
  return arith::NegFOp::create(builder, loc, value);
}

Value ScalarEmitter::scale(double factor, Value value) {
  if (!value || factor == 0.0)
    return Value();
  if (factor == 1.0)
    return value;
  if (factor == -1.0)
    return neg(value);
  return mul(constantLike(factor, value), value);
}

//===----------------------------------------------------------------------===//
// ScalarDerivative
//===----------------------------------------------------------------------===//

LogicalResult ScalarDerivative::get(Value value, Value &tangent) {
  auto found = tangents.find(value);
  if (found != tangents.end()) {
    tangent = found->second;
    return success();
  }
  if (failed(compute(value, tangent)))
    return failure();
  tangents[value] = tangent;
  return success();
}

/// Returns true if values of `type` can carry a derivative.
static bool isDifferentiableType(Type type) {
  if (isa<FloatType>(type))
    return true;
  auto vector = dyn_cast<VectorType>(type);
  return vector && isa<FloatType>(vector.getElementType());
}

LogicalResult ScalarDerivative::compute(Value value, Value &tangent) {
  tangent = Value();

  if (value == variable) {
    if (seed) {
      tangent = seed;
      return success();
    }
    ScalarEmitter emit(builder, value.getLoc());
    tangent = emit.constantLike(1.0, value);
    return success();
  }

  // Integers, booleans, and other values that are not floating point do not
  // depend on the variable in a differentiable way.
  if (!isDifferentiableType(value.getType()))
    return success();

  Operation *op = value.getDefiningOp();

  // A block argument other than the variable is an independent input.
  if (!op) {
    if (leafHandler)
      return leafHandler(value, tangent);
    return success();
  }

  // A value that a kernel takes from outside, such as one of the cell, does
  // not depend on an argument of the kernel.
  if (auto argument = dyn_cast<BlockArgument>(variable))
    if (!argument.getOwner()->getParent()->isAncestor(op->getParentRegion()))
      return success();

  ScalarEmitter emit(builder, op->getLoc());
  Location loc = op->getLoc();

  // Derivative of operand `index` of `op`.
  auto operandTangent = [&](unsigned index, Value &result) {
    return get(op->getOperand(index), result);
  };

  if (matchPattern(op, m_Constant()))
    return success();

  // A value of a table depends on the types only.
  if (isa<LookupOp>(op))
    return success();

  // The edges of the cell are an input of their own, which the virial
  // takes its derivative with respect to (D154).
  if (op->getName().getStringRef() == "md_exec.cell_edges")
    return success();

  // A number converted from an integer is constant where it is defined.
  if (isa<arith::SIToFPOp, arith::UIToFPOp>(op))
    return success();

  //===--------------------------------------------------------------------===//
  // Arithmetic
  //===--------------------------------------------------------------------===//

  if (isa<arith::AddFOp, arith::SubFOp>(op)) {
    Value lhs, rhs;
    if (failed(operandTangent(0, lhs)) || failed(operandTangent(1, rhs)))
      return failure();
    tangent = isa<arith::AddFOp>(op) ? emit.add(lhs, rhs) : emit.sub(lhs, rhs);
    return success();
  }

  if (isa<arith::MulFOp>(op)) {
    Value lhs, rhs;
    if (failed(operandTangent(0, lhs)) || failed(operandTangent(1, rhs)))
      return failure();
    tangent = emit.add(emit.mul(lhs, op->getOperand(1)),
                       emit.mul(op->getOperand(0), rhs));
    return success();
  }

  if (isa<arith::DivFOp>(op)) {
    // (a / b)' = a' / b − (a / b) · b' / b
    Value lhs, rhs;
    if (failed(operandTangent(0, lhs)) || failed(operandTangent(1, rhs)))
      return failure();
    Value denominator = op->getOperand(1);
    Value first = emit.div(lhs, denominator);
    Value second = emit.div(emit.mul(value, rhs), denominator);
    tangent = emit.sub(first, second);
    return success();
  }

  if (isa<arith::NegFOp>(op)) {
    Value operand;
    if (failed(operandTangent(0, operand)))
      return failure();
    tangent = emit.neg(operand);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Selection
  //===--------------------------------------------------------------------===//

  // The condition is not differentiated.
  if (isa<arith::SelectOp>(op)) {
    Value onTrue, onFalse;
    if (failed(operandTangent(1, onTrue)) || failed(operandTangent(2, onFalse)))
      return failure();
    if (!onTrue && !onFalse)
      return success();
    if (!onTrue)
      onTrue = emit.constantLike(0.0, value);
    if (!onFalse)
      onFalse = emit.constantLike(0.0, value);
    tangent = arith::SelectOp::create(builder, loc, op->getOperand(0), onTrue,
                                      onFalse);
    return success();
  }

  // min(a, b)' = a' if a < b, else b'.
  // max(a, b)' = a' if a > b, else b'.
  if (isa<arith::MinimumFOp, arith::MaximumFOp>(op)) {
    Value lhs, rhs;
    if (failed(operandTangent(0, lhs)) || failed(operandTangent(1, rhs)))
      return failure();
    if (!lhs && !rhs)
      return success();
    if (!lhs)
      lhs = emit.constantLike(0.0, value);
    if (!rhs)
      rhs = emit.constantLike(0.0, value);
    arith::CmpFPredicate predicate = isa<arith::MinimumFOp>(op)
                                         ? arith::CmpFPredicate::OLT
                                         : arith::CmpFPredicate::OGT;
    Value condition = arith::CmpFOp::create(builder, loc, predicate,
                                            op->getOperand(0),
                                            op->getOperand(1));
    tangent = arith::SelectOp::create(builder, loc, condition, lhs, rhs);
    return success();
  }

  // abs(x)' = x' if x >= 0, else −x'.
  if (isa<math::AbsFOp>(op)) {
    Value operand;
    if (failed(operandTangent(0, operand)))
      return failure();
    if (!operand)
      return success();
    Value zero = emit.constantLike(0.0, value);
    Value condition = arith::CmpFOp::create(
        builder, loc, arith::CmpFPredicate::OGE, op->getOperand(0), zero);
    tangent = arith::SelectOp::create(builder, loc, condition, operand,
                                      emit.neg(operand));
    return success();
  }

  // Piecewise constant.
  if (isa<math::FloorOp, math::CeilOp>(op))
    return success();

  //===--------------------------------------------------------------------===//
  // Powers
  //===--------------------------------------------------------------------===//

  if (isa<math::SqrtOp>(op)) {
    // sqrt(a)' = a' / (2 · sqrt(a))
    Value operand;
    if (failed(operandTangent(0, operand)))
      return failure();
    tangent = emit.div(operand, emit.scale(2.0, value));
    return success();
  }

  if (isa<math::FPowIOp>(op)) {
    // (a ^ n)' = n · a ^ (n − 1) · a'
    Value base;
    if (failed(operandTangent(0, base)))
      return failure();
    if (!base)
      return success();

    Value exponent = op->getOperand(1);
    Value factor;
    Value reduced;
    APInt constant;
    if (matchPattern(exponent, m_ConstantInt(&constant))) {
      int64_t n = constant.getSExtValue();
      if (n == 0)
        return success();
      factor = emit.constantLike(static_cast<double>(n), value);
      reduced = arith::ConstantOp::create(
          builder, loc, exponent.getType(),
          builder.getIntegerAttr(exponent.getType(), n - 1));
    } else {
      Value one = arith::ConstantOp::create(
          builder, loc, exponent.getType(),
          builder.getIntegerAttr(exponent.getType(), 1));
      reduced = arith::SubIOp::create(builder, loc, exponent, one);
      factor =
          arith::SIToFPOp::create(builder, loc, value.getType(), exponent);
    }
    Value power =
        math::FPowIOp::create(builder, loc, op->getOperand(0), reduced);
    tangent = emit.mul(emit.mul(factor, power), base);
    return success();
  }

  if (isa<math::PowFOp>(op)) {
    Value base, exponent;
    if (failed(operandTangent(0, base)) || failed(operandTangent(1, exponent)))
      return failure();
    Value a = op->getOperand(0);
    Value b = op->getOperand(1);

    // Through the base: b · a ^ (b − 1) · a'
    Value throughBase;
    if (base) {
      Value reduced = emit.sub(b, emit.constantLike(1.0, b));
      Value power = math::PowFOp::create(builder, loc, a, reduced);
      throughBase = emit.mul(emit.mul(b, power), base);
    }
    // Through the exponent: a ^ b · log(a) · b'
    Value throughExponent;
    if (exponent) {
      Value log = math::LogOp::create(builder, loc, a);
      throughExponent = emit.mul(emit.mul(value, log), exponent);
    }
    tangent = emit.add(throughBase, throughExponent);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Functions of one argument
  //===--------------------------------------------------------------------===//

  if (isa<math::ExpOp, math::LogOp, math::SinOp, math::CosOp, math::TanOp,
          math::AsinOp, math::AcosOp, math::AtanOp, math::SinhOp,
          math::CoshOp, math::TanhOp, math::ErfOp, math::ErfcOp>(op)) {
    Value operand;
    if (failed(operandTangent(0, operand)))
      return failure();
    if (!operand)
      return success();

    Value a = op->getOperand(0);
    Value one = emit.constantLike(1.0, value);
    Value factor;

    if (isa<math::ExpOp>(op)) {
      factor = value;
    } else if (isa<math::LogOp>(op)) {
      factor = emit.div(one, a);
    } else if (isa<math::SinOp>(op)) {
      factor = math::CosOp::create(builder, loc, a);
    } else if (isa<math::CosOp>(op)) {
      factor = emit.neg(math::SinOp::create(builder, loc, a));
    } else if (isa<math::TanOp>(op)) {
      factor = emit.add(one, emit.mul(value, value));
    } else if (isa<math::AsinOp, math::AcosOp>(op)) {
      Value radicand = emit.sub(one, emit.mul(a, a));
      Value root = math::SqrtOp::create(builder, loc, radicand);
      factor = emit.div(one, root);
      if (isa<math::AcosOp>(op))
        factor = emit.neg(factor);
    } else if (isa<math::AtanOp>(op)) {
      factor = emit.div(one, emit.add(one, emit.mul(a, a)));
    } else if (isa<math::SinhOp>(op)) {
      factor = math::CoshOp::create(builder, loc, a);
    } else if (isa<math::CoshOp>(op)) {
      factor = math::SinhOp::create(builder, loc, a);
    } else if (isa<math::TanhOp>(op)) {
      factor = emit.sub(one, emit.mul(value, value));
    } else {
      // erf(a)' = (2 / sqrt(pi)) · exp(−a²); erfc(a)' is its negative.
      Value gaussian =
          math::ExpOp::create(builder, loc, emit.neg(emit.mul(a, a)));
      double coefficient = 2.0 / std::sqrt(M_PI);
      if (isa<math::ErfcOp>(op))
        coefficient = -coefficient;
      factor = emit.scale(coefficient, gaussian);
    }
    tangent = emit.mul(factor, operand);
    return success();
  }

  if (isa<math::Atan2Op>(op)) {
    // atan2(y, x)' = (x · y' − y · x') / (x² + y²)
    Value yTangent, xTangent;
    if (failed(operandTangent(0, yTangent)) ||
        failed(operandTangent(1, xTangent)))
      return failure();
    if (!yTangent && !xTangent)
      return success();
    Value y = op->getOperand(0);
    Value x = op->getOperand(1);
    Value numerator = emit.sub(emit.mul(x, yTangent), emit.mul(y, xTangent));
    Value denominator = emit.add(emit.mul(x, x), emit.mul(y, y));
    tangent = emit.div(numerator, denominator);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Vectors
  //===--------------------------------------------------------------------===//

  if (isa<vector::BroadcastOp>(op)) {
    Value operand;
    if (failed(operandTangent(0, operand)))
      return failure();
    if (operand)
      tangent =
          vector::BroadcastOp::create(builder, loc, value.getType(), operand);
    return success();
  }

  // A component of a vector: the same component of its derivative.
  if (auto extract = dyn_cast<vector::ExtractOp>(op)) {
    if (!extract.getDynamicPosition().empty())
      return op->emitError() << "no derivative rule for '" << op->getName()
                             << "' at a position that is not constant";
    Value operand;
    if (failed(operandTangent(0, operand)))
      return failure();
    if (operand)
      tangent = vector::ExtractOp::create(builder, loc, operand,
                                          extract.getStaticPosition());
    return success();
  }

  if (leafHandler)
    return leafHandler(value, tangent);

  return op->emitError() << "no derivative rule for '" << op->getName()
                         << "'";
}

//===----------------------------------------------------------------------===//
// Dead code
//===----------------------------------------------------------------------===//

void mdir::md::eraseDeadOps(Block &block) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block))) {
      if (op.hasTrait<OpTrait::IsTerminator>() || !isOpTriviallyDead(&op))
        continue;
      op.erase();
      changed = true;
    }
  }
}
