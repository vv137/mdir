// Reverse composition of the declared grid, FFT and assignment rules (D388).
#include "mdir/Dialect/MD/Transforms/ReciprocalGrid.h"
#include "mdir/Dialect/MD/Transforms/Activity.h"
#include "mdir/Dialect/MD/Transforms/AdjointInterface.h"
#include "mdir/Dialect/MD/Transforms/GridKernels.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/StencilGeometry.h"
#include "mdir/Dialect/MD/Transforms/Stencils.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

using namespace mlir;
using namespace mdir::md;

namespace {
Value matrixCell(Value value) {
  auto elements = value.getDefiningOp<vector::FromElementsOp>();
  if (!elements || elements.getNumOperands() != 9)
    return {};
  int64_t indices[] = {0,-1,-1,3,1,-1,4,5,2};
  Value numbers;
  for (unsigned i = 0; i != 9; ++i) {
    Value element = elements->getOperand(i);
    if (indices[i] < 0) {
      if (!matchPattern(element, m_AnyZeroFloat()))
        return {};
      continue;
    }
    auto extract = element.getDefiningOp<vector::ExtractOp>();
    if (!extract || extract.getStaticPosition().size() != 1 ||
        extract.getStaticPosition()[0] != indices[i])
      return {};
    if (numbers && numbers != extract.getSource())
      return {};
    numbers = extract.getSource();
  }
  auto source = numbers.getDefiningOp<mdir::md_exec::CellNumbersOp>();
  return source ? source.getCell() : Value();
}

FailureOr<bool> depends(Value value, ValueRange variables) {
  ActivityResult unknown;
  bool unproved = false;
  for (Value variable : variables) {
    ActivityAnalysis activity(variable);
    auto result = activity.classify(value);
    if (result.dependence == Activity::Active)
      return true;
    if (result.dependence == Activity::Unknown) {
      unknown = result;
      unproved = true;
    }
  }
  if (unproved) {
    emitError(value.getLoc()) << "cannot prove inactive input in reciprocal grid composition: "
                              << unknown.reason;
    return failure();
  }
  return false;
}

Value add(OpBuilder &b, Location loc, Value a, Value c,
          DenseI64ArrayAttr grid) {
  if (!a) return c;
  if (!c) return a;
  if (auto field = dyn_cast<FieldType>(a.getType())) {
    auto map = MapParticlesOp::create(b, loc, field, ValueRange{a,c});
    Block *body = new Block();
    map.getKernel().push_back(body);
    body->addArgument(field.getKernelValueType(), loc);
    body->addArgument(field.getKernelValueType(), loc);
    OpBuilder at = OpBuilder::atBlockEnd(body);
    Value sum = arith::AddFOp::create(at, loc, body->getArgument(0), body->getArgument(1));
    YieldOp::create(at, loc, ValueRange{sum});
    return map.getResult();
  }
  if (auto tensor = dyn_cast<RankedTensorType>(a.getType())) {
    bool half = tensor.getDimSize(2) != grid.asArrayRef()[2];
    Operation *map = createGridKernel(b, loc, false, tensor, ValueRange{a,c},
        grid, b.getStringAttr(half ? "real_hermitian_half" : "euclidean"));
    Block &body = map->getRegion(0).front();
    OpBuilder at = OpBuilder::atBlockEnd(&body);
    Value sum = arith::AddFOp::create(at, loc, body.getArgument(1), body.getArgument(2));
    YieldOp::create(at, loc, ValueRange{sum});
    return map->getResult(0);
  }
  return arith::AddFOp::create(b, loc, a, c);
}
}

FailureOr<ReciprocalGridGraph> mdir::md::inspectReciprocalGrid(SumGridOp root) {
  if (failed(verifyReciprocalEnergyKernel(root)))
    return failure();
  auto transform = root.getInputs()[0].getDefiningOp<LinearTransformOp>();
  auto product = root.getInputs()[1].getDefiningOp<MapGridOp>();
  if (!transform || !product || transform.getDirection() != "forward" ||
      product.getInputs().size() != 3 ||
      product.getInputs()[0] != transform.getResult() ||
      product.getPairing() != "real_hermitian_half" ||
      product.getGridAttr() != root.getGridAttr()) {
    root.emitOpError("expected a shared forward transform and its explicit Gaussian product");
    return failure();
  }
  if (failed(verifyGaussianReciprocalKernel(product)))
    return failure();
  ReciprocalGridGraph graph;
  llvm::DenseSet<Operation *> seen;
  SmallVector<Value> leaves;
  bool invalid = false;
  std::function<void(Value)> visit = [&](Value value) {
    Operation *op = value.getDefiningOp();
    if (Value cell = matrixCell(value)) {
      if (graph.matrix && graph.matrix != value) {
        root.emitOpError("reciprocal grid composition requires one explicit cell matrix");
        invalid = true;
      }
      graph.matrix = value;
      return;
    }
    if (!op || !isa<MapGridOp, SumGridOp, LinearTransformOp, ScatterStencilOp>(op)) {
      leaves.push_back(value);
      return;
    }
    if (!seen.insert(op).second)
      return;
    if (auto scatter = dyn_cast<ScatterStencilOp>(op)) {
      if (graph.assignment && graph.assignment != scatter) {
        root.emitOpError("reciprocal grid composition currently requires one assignment relation");
        invalid = true;
      }
      graph.assignment = scatter;
      leaves.append({scatter.getPositions(), scatter.getData(), scatter.getCell()});
    } else {
      for (Value input : op->getOperands())
        visit(input);
    }
    graph.order.push_back(op);
  };
  visit(root.getResult());
  if (invalid)
    return failure();
  if (!graph.assignment || !graph.matrix ||
      matrixCell(graph.matrix) != graph.assignment.getCell()) {
    root.emitOpError("expected one assignment and its exact explicit row-cell matrix");
    return failure();
  }
  auto checked = graph.assignment.getCell().getDefiningOp<CheckedRectangularCellOp>();
  if (!checked) {
    root.emitOpError("reciprocal grid composition requires the checked rectangular-cell precondition");
    return failure();
  }
  auto stencil = cast<GridStencilType>(graph.assignment.getStencil().getType());
  if (stencil.getShape() != root.getGridAttr() ||
      transform.getInput() != graph.assignment.getResult()) {
    root.emitOpError("reciprocal grid and transform must use the exact assignment shape and value");
    return failure();
  }
  if (stencil.getOrder() != 4 || stencil.getWindow().getValue() != "bspline") {
    root.emitOpError("reciprocal grid derivatives currently require order-four B-splines");
    return failure();
  }
  Block &weight = graph.assignment.getKernel().front();
  auto window = weight.getTerminator()->getOperand(0).getDefiningOp<BSplineWindowOp>();
  if (!window || window.getDistance() != weight.getArgument(0) || window.getOrder() != 4) {
    root.emitOpError("reciprocal assignment requires the common order-four cardinal window");
    return failure();
  }
  for (Value variable : ValueRange{graph.assignment.getPositions(), checked.getInput()}) {
    ActivityAnalysis activity(variable);
    if (activity.classify(graph.assignment.getData()).dependence != Activity::Inactive) {
      root.emitOpError("position/cell-dependent assignment amplitudes require an upstream derivative rule");
      return failure();
    }
  }
  SmallVector<Value> variables{graph.assignment.getPositions(), graph.assignment.getData(),
                               graph.assignment.getCell(), graph.matrix};
  for (Value leaf : leaves) {
    if (llvm::is_contained(variables, leaf))
      continue;
    auto active = depends(leaf, variables);
    if (failed(active))
      return failure();
    if (*active) {
      root.emitOpError("active input has no declared rule in the reciprocal grid composition");
      return failure();
    }
  }
  return graph;
}

FailureOr<ReciprocalGridDerivative> mdir::md::reverseReciprocalGrid(
    OpBuilder &b, SumGridOp root) {
  auto inspected = inspectReciprocalGrid(root);
  if (failed(inspected))
    return failure();
  auto &graph = *inspected;
  Location loc = root.getLoc();
  ScalarEmitter e(b, loc);
  llvm::DenseMap<Value, Value> adjoints;
  adjoints[root.getResult()] = e.constant(1, b.getF64Type());
  SmallVector<Value> variables{graph.assignment.getPositions(), graph.assignment.getData(),
                               graph.assignment.getCell(), graph.matrix};
  auto accumulate = [&](Value input, Value value) {
    adjoints[input] = add(b, loc, adjoints.lookup(input), value, root.getGridAttr());
  };
  for (Operation *op : llvm::reverse(graph.order)) {
    Value cotangent = adjoints.lookup(op->getResult(0));
    if (!cotangent)
      continue;
    if (isa<MapGridOp, SumGridOp>(op)) {
      for (auto [i, input] : llvm::enumerate(op->getOperands())) {
        auto active = depends(input, variables);
        if (failed(active))
          return failure();
        if (!*active)
          continue;
        auto result = differentiateGridKernel(b, op, i, cotangent, true);
        if (failed(result))
          return failure();
        accumulate(input, *result);
      }
    } else if (auto transform = dyn_cast<LinearTransformOp>(op)) {
      auto rule = cast<AdjointOpInterface>(op);
      Value result = rule.emitTranspose(b, 0, cotangent, {});
      if (!result)
        return failure();
      accumulate(transform.getInput(), result);
    } else {
      auto position = differentiateStencilGeometry(b, op, 1, cotangent, true);
      auto cell = differentiateStencilGeometry(b, op, 2, cotangent, true);
      if (failed(position) || failed(cell))
        return failure();
      accumulate(op->getOperand(1), *position);
      accumulate(op->getOperand(2), *cell);
      accumulate(op->getOperand(3), cloneStencil(b, op, cotangent, true));
    }
  }
  Value x = graph.assignment.getPositions();
  Value gx = adjoints.lookup(x);
  Value gq = adjoints.lookup(graph.assignment.getData());
  if (!gx || !gq) {
    root.emitOpError("reciprocal grid reverse chain did not define its particle cotangents");
    return failure();
  }
  Value gh = add(b, loc, adjoints.lookup(graph.matrix),
                 adjoints.lookup(graph.assignment.getCell()), root.getGridAttr());
  auto v9 = VectorType::get({9}, b.getF64Type());
  auto positional = SumParticlesOp::create(b, loc, v9, ValueRange{x,gx});
  Block *body = new Block();
  positional.getKernel().push_back(body);
  auto v3 = VectorType::get({3}, b.getF64Type());
  body->addArgument(v3, loc);
  body->addArgument(v3, loc);
  OpBuilder kernel = OpBuilder::atBlockEnd(body);
  ScalarEmitter ke(kernel, loc);
  SmallVector<Value> values;
  for (int row = 0; row != 3; ++row)
    for (int col = 0; col != 3; ++col) {
      Value a = vector::ExtractOp::create(kernel, loc, body->getArgument(0), col);
      Value c = vector::ExtractOp::create(kernel, loc, body->getArgument(1), row);
      values.push_back(ke.neg(ke.mul(a,c)));
    }
  YieldOp::create(kernel, loc, ValueRange{vector::FromElementsOp::create(kernel, loc, v9, values)});
  values.clear();
  for (int row = 0; row != 3; ++row)
    for (int col = 0; col != 3; ++col) {
      Value sum;
      for (int axis = 0; axis != 3; ++axis) {
        Value h = vector::ExtractOp::create(b, loc, graph.matrix, 3*axis+col);
        Value g = vector::ExtractOp::create(b, loc, gh, 3*axis+row);
        sum = e.add(sum, e.mul(h,g));
      }
      values.push_back(e.neg(sum));
    }
  Value cellVirial = vector::FromElementsOp::create(b, loc, v9, values);
  Value virial = e.add(positional.getResult(), cellVirial);
  return ReciprocalGridDerivative{graph.assignment, gx, gq, virial};
}
