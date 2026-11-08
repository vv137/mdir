// Structured real grid differentiation; complex values use real components.
#include "mdir/Dialect/MD/Transforms/GridKernels.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mdir::md;

Type mdir::md::getGridKernelType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (auto complex = dyn_cast<ComplexType>(type))
    return VectorType::get({2}, complex.getElementType());
  return type;
}

namespace {
Value zero(OpBuilder &b, Location loc, Type type) {
  return arith::ConstantOp::create(b, loc, type,
                                  cast<TypedAttr>(b.getZeroAttr(type)));
}

Value dot(OpBuilder &b, Location loc, Value a, Value c) {
  Value product = arith::MulFOp::create(b, loc, a, c);
  if (isa<VectorType>(product.getType()))
    return vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD,
                                       product, Value());
  return product;
}

Operation *gridOperation(OpBuilder &b, Location loc, bool sum,
                         Type result, ValueRange inputs,
                         DenseI64ArrayAttr grid, StringAttr pairing) {
  OperationState state(loc, sum ? SumGridOp::getOperationName()
                                : MapGridOp::getOperationName());
  state.addOperands(inputs);
  state.addTypes(result);
  state.addAttribute("grid", grid);
  state.addAttribute("pairing", pairing);
  state.addRegion();
  Operation *op = b.create(state);
  Block *block = new Block();
  op->getRegion(0).push_back(block);
  block->addArgument(VectorType::get({3}, b.getIndexType()), loc);
  for (Value input : inputs)
    block->addArgument(getGridKernelType(input.getType()), loc);
  return op;
}
}

Operation *mdir::md::createGridKernel(OpBuilder &b, Location loc, bool sum,
    Type result, ValueRange inputs, DenseI64ArrayAttr grid, StringAttr pairing) {
  return gridOperation(b, loc, sum, result, inputs, grid, pairing);
}

FailureOr<Value> mdir::md::differentiateGridKernel(
    OpBuilder &b, Operation *source, unsigned operand,
    Value seed, bool reverse) {
  Location loc = source->getLoc();
  if (isa<TableType>(source->getOperand(operand).getType())) {
    source->emitOpError("unsupported structured grid derivative for active table operand");
    return failure();
  }
  bool inputTensor = isa<RankedTensorType>(source->getOperand(operand).getType());
  bool sum = reverse ? !inputTensor : isa<SumGridOp>(source);
  Type result = reverse ? source->getOperand(operand).getType()
                        : source->getResult(0).getType();
  SmallVector<Value> inputs(source->getOperands());
  inputs.push_back(seed);
  Operation *op = gridOperation(b, loc, sum, result, inputs,
      source->getAttrOfType<DenseI64ArrayAttr>("grid"),
      source->getAttrOfType<StringAttr>("pairing"));
  Block &body = op->getRegion(0).front();
  OpBuilder kernel = OpBuilder::atBlockEnd(&body);
  Block &old = source->getRegion(0).front();
  IRMapping mapping;
  for (auto [from, to] : llvm::zip(old.getArguments(), body.getArguments()))
    mapping.map(from, to);
  for (Operation &inner : old.without_terminator())
    kernel.clone(inner, mapping);
  Value value = mapping.lookupOrDefault(old.getTerminator()->getOperand(0));
  Value variable = body.getArgument(operand + 1);
  Value direction = body.getArguments().back();
  Value derivative;
  if (!reverse) {
    ScalarDerivative rule(kernel, variable, nullptr, direction);
    if (failed(rule.get(value, derivative))) {
      op->erase();
      return failure();
    }
    if (!derivative)
      derivative = zero(kernel, loc, value.getType());
  } else {
    auto vector = dyn_cast<VectorType>(variable.getType());
    SmallVector<Value> partials;
    int64_t components = vector ? vector.getNumElements() : 1;
    for (int64_t component = 0; component != components; ++component) {
      Value unit;
      if (vector) {
        SmallVector<double> values(components, 0.0);
        values[component] = 1.0;
        unit = arith::ConstantOp::create(kernel, loc, vector,
            DenseElementsAttr::get(vector, ArrayRef<double>(values)));
      }
      ScalarDerivative rule(kernel, variable, nullptr, unit);
      Value partial;
      if (failed(rule.get(value, partial))) {
        op->erase();
        return failure();
      }
      partials.push_back(partial ? dot(kernel, loc, partial, direction)
                                : zero(kernel, loc, kernel.getF64Type()));
    }
    derivative = vector ? Value(vector::FromElementsOp::create(
        kernel, loc, vector, partials)) : partials.front();
  }
  YieldOp::create(kernel, loc, ValueRange{derivative});
  return op->getResult(0);
}

namespace {
template <typename Op>
struct GridRule : DerivativeOpInterface::ExternalModel<GridRule<Op>, Op> {
  DerivativeOperand getDerivativeOperand(Operation *, unsigned) const {
    return DerivativeOperand::Differentiable;
  }
  LogicalResult emitDerivative(Operation *op, Value, ScalarDerivative &,
                               Value &) const {
    // ScalarDerivative does not propagate tensor tangents. Do not interpret
    // one as zero: the explicit grid request carries its complete seed.
    return op->emitOpError(
        "structured grid forward tangents require md-differentiate-grid");
  }
  LogicalResult emitAdjoint(Operation *op, OpBuilder &b, ValueRange cotangents,
                            SmallVectorImpl<Value> &results) const {
    if (cotangents.size() != 1 ||
        cotangents[0].getType() != op->getResult(0).getType())
      return op->emitOpError("expected one cotangent with the grid result type");
    for (unsigned input = 0; input != op->getNumOperands(); ++input) {
      auto value = differentiateGridKernel(b, op, input, cotangents[0], true);
      if (failed(value))
        return failure();
      results.push_back(*value);
    }
    return success();
  }
};
}

void mdir::md::registerGridDerivatives(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, MDDialect *) {
    MapGridOp::attachInterface<GridRule<MapGridOp>>(*context);
    SumGridOp::attachInterface<GridRule<SumGridOp>>(*context);
  });
}

namespace mdir::md {
#define GEN_PASS_DEF_DIFFERENTIATEGRID
#define GEN_PASS_DEF_EXPANDGRIDKERNELS
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"
namespace {
class DifferentiateGrid : public impl::DifferentiateGridBase<DifferentiateGrid> {
  void runOnOperation() final {
    SmallVector<Operation *> requests;
    getOperation()->walk([&](Operation *op) {
      if (isa<GridJVPOp, GridVJPOp>(op))
        requests.push_back(op);
    });
    for (Operation *request : requests) {
      OpBuilder b(request);
      auto result = differentiateGridKernel(b,
          request->getOperand(0).getDefiningOp(),
          request->getAttrOfType<IntegerAttr>("operand").getInt(),
          request->getOperand(1), isa<GridVJPOp>(request));
      if (failed(result)) {
        signalPassFailure();
        return;
      }
      request->getResult(0).replaceAllUsesWith(*result);
      request->erase();
    }
  }
};

Value index(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIndexOp::create(b, loc, n);
}

Value kernelValue(OpBuilder &b, Location loc, Value value,
                  ValueRange xyz) {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor)
    return value;
  Value element = tensor::ExtractOp::create(b, loc, value, xyz);
  if (isa<ComplexType>(tensor.getElementType())) {
    Value real = complex::ReOp::create(b, loc, element);
    Value imag = complex::ImOp::create(b, loc, element);
    return vector::FromElementsOp::create(b, loc,
        getGridKernelType(tensor), ValueRange{real, imag});
  }
  return element;
}

Value evaluate(OpBuilder &b, Operation *op, ValueRange xyz) {
  Location loc = op->getLoc();
  Block &block = op->getRegion(0).front();
  IRMapping mapping;
  mapping.map(block.getArgument(0), vector::FromElementsOp::create(
      b, loc, VectorType::get({3}, b.getIndexType()), xyz));
  for (auto [input, argument] : llvm::zip(op->getOperands(),
                                        block.getArguments().drop_front()))
    mapping.map(argument, kernelValue(b, loc, input, xyz));
  for (Operation &inner : block.without_terminator())
    b.clone(inner, mapping);
  return mapping.lookupOrDefault(block.getTerminator()->getOperand(0));
}

class ExpandGridKernels
    : public impl::ExpandGridKernelsBase<ExpandGridKernels> {
  void runOnOperation() final {
    SmallVector<Operation *> operations;
    getOperation()->walk([&](Operation *op) {
      if (isa<MapGridOp, SumGridOp>(op))
        operations.push_back(op);
    });
    for (Operation *op : operations) {
      OpBuilder b(op);
      Location loc = op->getLoc();
      auto grid = op->getAttrOfType<DenseI64ArrayAttr>("grid").asArrayRef();
      bool half = op->getAttrOfType<StringAttr>("pairing").getValue()
                      == "real_hermitian_half";
      SmallVector<int64_t> shape(grid);
      if (half)
        shape[2] = grid[2] / 2 + 1;
      Value result;
      if (isa<MapGridOp>(op)) {
        auto type = cast<RankedTensorType>(op->getResult(0).getType());
        result = tensor::GenerateOp::create(b, loc, type, ValueRange(),
            [&](OpBuilder &inner, Location where, ValueRange xyz) {
              Value value = evaluate(inner, op, xyz);
              if (isa<ComplexType>(type.getElementType())) {
                Value real = vector::ExtractOp::create(inner, where, value, 0);
                Value imag = vector::ExtractOp::create(inner, where, value, 1);
                value = complex::CreateOp::create(inner, where,
                    type.getElementType(), real, imag);
              }
              tensor::YieldOp::create(inner, where, value);
            });
      } else {
        Value limit = index(b, loc, shape[0] * shape[1] * shape[2]);
        auto loop = scf::ForOp::create(b, loc, index(b, loc, 0), limit,
            index(b, loc, 1), ValueRange{zero(b, loc, op->getResult(0).getType())},
            [&](OpBuilder &inner, Location where, Value i, ValueRange sums) {
              Value plane = index(inner, where, shape[1] * shape[2]);
              Value last = index(inner, where, shape[2]);
              Value x = arith::DivUIOp::create(inner, where, i, plane);
              Value rem = arith::RemUIOp::create(inner, where, i, plane);
              Value y = arith::DivUIOp::create(inner, where, rem, last);
              Value z = arith::RemUIOp::create(inner, where, rem, last);
              Value value = evaluate(inner, op, ValueRange{x, y, z});
              if (half) {
                Value first = arith::CmpIOp::create(inner, where,
                    arith::CmpIPredicate::eq, z, index(inner, where, 0));
                if (grid[2] % 2 == 0) {
                  Value nyquist = arith::CmpIOp::create(inner, where,
                      arith::CmpIPredicate::eq, z, index(inner, where, grid[2] / 2));
                  first = arith::OrIOp::create(inner, where, first, nyquist);
                }
                Value twice = arith::AddFOp::create(inner, where, value, value);
                value = arith::SelectOp::create(inner, where, first, value, twice);
              }
              Value next = arith::AddFOp::create(inner, where, sums[0], value);
              scf::YieldOp::create(inner, where, next);
            });
        result = loop.getResult(0);
      }
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
    }
  }
};
}
}
