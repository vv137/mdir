// Continuous stencil geometry in the row-cell convention (Essmann1995).
#include "mdir/Dialect/MD/Transforms/StencilGeometry.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mdir::md;

StencilCellInverse mdir::md::emitStencilCellInverse(
    OpBuilder &b, Location loc, Value matrix) {
  ScalarEmitter e(b, loc);
  SmallVector<Value> h;
  for (int64_t i = 0; i != 9; ++i)
    h.push_back(vector::ExtractOp::create(b, loc, matrix, i));
  auto minor = [&](int a, int c, int d, int f) {
    return e.sub(e.mul(h[a], h[c]), e.mul(h[d], h[f]));
  };
  SmallVector<Value> adj{
      minor(4,8,5,7), minor(2,7,1,8), minor(1,5,2,4),
      minor(5,6,3,8), minor(0,8,2,6), minor(2,3,0,5),
      minor(3,7,4,6), minor(1,6,0,7), minor(0,4,1,3)};
  Value det = e.add(e.add(e.mul(h[0], adj[0]), e.mul(h[1], adj[3])),
                    e.mul(h[2], adj[6]));
  for (Value &entry : adj)
    entry = e.div(entry, det);
  StencilCellInverse result;
  result.entries.assign(adj.begin(), adj.end());
  result.determinant = det;
  return result;
}

Value mdir::md::emitStencilCellMatrix(OpBuilder &b, Location loc, Value cell) {
  Value numbers = mdir::md_exec::CellNumbersOp::create(b, loc,
      VectorType::get({6}, b.getF64Type()), cell);
  SmallVector<Value> h;
  for (int64_t i = 0; i != 6; ++i)
    h.push_back(vector::ExtractOp::create(b, loc, numbers, i));
  ScalarEmitter e(b, loc);
  Value zero = e.constant(0, b.getF64Type());
  return vector::FromElementsOp::create(b, loc,
      VectorType::get({9}, b.getF64Type()),
      ValueRange{h[0],zero,zero,h[3],h[1],zero,h[4],h[5],h[2]});
}

FailureOr<Value> mdir::md::differentiateStencilGeometry(
    OpBuilder &b, Operation *source, unsigned operand,
    Value seed, bool reverse) {
  Location loc = source->getLoc();
  if (operand != 1 && operand != 2)
    return failure();
  auto stencil = cast<GridStencilType>(source->getOperand(0).getType());
  bool scatter = isa<ScatterStencilOp>(source);
  // Derivatives of the weight belong to the same entries. In reverse,
  // grid cotangents are gathered; in forward, scattering retains its grid.
  SmallVector<Value> particles, grids, uniforms;
  if (scatter) {
    particles.push_back(source->getOperand(3));
    if (reverse)
      grids.push_back(seed);
  } else {
    grids.push_back(source->getOperand(3));
    if (reverse)
      particles.push_back(seed);
  }
  if (!reverse) {
    if (operand == 1)
      particles.push_back(seed);
    else
      uniforms.push_back(seed);
  }
  Type result;
  StringRef destination;
  if (!reverse) {
    result = source->getResult(0).getType();
    destination = scatter ? "grid" : "particles";
  } else if (operand == 1) {
    result = source->getOperand(1).getType();
    destination = "particles";
  } else {
    result = VectorType::get({9}, b.getF64Type());
    destination = "sum";
  }
  auto reduced = ReduceStencilOp::create(b, loc, result,
      source->getOperand(0), source->getOperand(1), source->getOperand(2),
      particles, grids, uniforms, b.getStringAttr(destination));
  Block *block = new Block();
  reduced.getKernel().push_back(block);
  auto v3 = VectorType::get({3}, b.getF64Type());
  auto v9 = VectorType::get({9}, b.getF64Type());
  block->addArgument(v3, loc); // Continuous support distance.
  block->addArgument(v3, loc); // Unwrapped Cartesian position.
  block->addArgument(v9, loc); // Row-cell matrix.
  for (Value field : particles)
    block->addArgument(cast<FieldType>(field.getType()).getKernelValueType(), loc);
  for (Value grid : grids)
    block->addArgument(cast<RankedTensorType>(grid.getType()).getElementType(), loc);
  for (Value uniform : uniforms)
    block->addArgument(uniform.getType(), loc);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  Block &weight = source->getRegion(0).front();
  IRMapping mapping;
  mapping.map(weight.getArgument(0), block->getArgument(0));
  for (Operation &inner : weight.without_terminator())
    kernel.clone(inner, mapping);
  Value primal = mapping.lookupOrDefault(weight.getTerminator()->getOperand(0));
  SmallVector<Value> gradient;
  for (int c = 0; c != 3; ++c) {
    SmallVector<double> unit(3, 0);
    unit[c] = 1;
    Value direction = arith::ConstantOp::create(kernel, loc, v3,
        DenseElementsAttr::get(v3, ArrayRef<double>(unit)));
    ScalarDerivative derivative(kernel, block->getArgument(0), nullptr, direction);
    Value partial;
    if (failed(derivative.get(primal, partial))) {
      reduced.erase();
      return failure();
    }
    ScalarEmitter e(kernel, loc);
    if (!partial)
      partial = e.constant(0, b.getF64Type());
    gradient.push_back(e.scale(stencil.getShape().asArrayRef()[c], partial));
  }
  auto inverse = emitStencilCellInverse(kernel, loc, block->getArgument(2));
  ScalarEmitter e(kernel, loc);
  SmallVector<Value> cartesian, fractional;
  for (int c = 0; c != 3; ++c) {
    Value g, s;
    for (int a = 0; a != 3; ++a) {
      g = e.add(g, e.mul(gradient[a], inverse.entries[3*c+a]));
      Value x = vector::ExtractOp::create(kernel, loc, block->getArgument(1), a);
      s = e.add(s, e.mul(x, inverse.entries[3*a+c]));
    }
    cartesian.push_back(g);
    fractional.push_back(s);
  }
  // The scalar multiplier is q for spread and a grid value for gather.
  // In reverse the other linear operand is the incoming cotangent.
  Value factor;
  unsigned next = 3;
  if (scatter)
    factor = block->getArgument(next++);
  else if (reverse)
    factor = block->getArgument(next++);
  Value positionDirection;
  if (!reverse && operand == 1)
    positionDirection = block->getArgument(next++);
  if (!grids.empty())
    factor = factor ? e.mul(factor, block->getArgument(next++))
                    : block->getArgument(next++);
  SmallVector<Value> cellGradient;
  for (int row = 0; row != 3; ++row)
    for (int column = 0; column != 3; ++column)
      cellGradient.push_back(e.neg(e.mul(fractional[row], cartesian[column])));
  Value contribution;
  if (reverse) {
    SmallVector<Value> values = operand == 1 ? cartesian : cellGradient;
    for (Value &value : values)
      value = e.mul(value, factor);
    contribution = vector::FromElementsOp::create(kernel, loc,
        operand == 1 ? v3 : v9, values);
  } else {
    Value direction = operand == 1 ? positionDirection
                                   : block->getArgument(next);
    auto &values = operand == 1 ? cartesian : cellGradient;
    for (auto [c, partial] : llvm::enumerate(values)) {
      Value component = vector::ExtractOp::create(kernel, loc, direction, c);
      contribution = e.add(contribution, e.mul(partial, component));
    }
    contribution = e.mul(contribution, factor);
  }
  YieldOp::create(kernel, loc, ValueRange{contribution});
  return reduced.getResult();
}
