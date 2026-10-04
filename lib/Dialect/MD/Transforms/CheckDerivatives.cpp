// Central differences of energies at the inputs of an md.evaluate.
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include <cmath>
using namespace mlir;
using namespace mdir::md;
namespace mdir { namespace md {
#define GEN_PASS_DEF_CHECKDERIVATIVES
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"
namespace {

class CheckDerivatives : public impl::CheckDerivativesBase<CheckDerivatives> {
public:
  using impl::CheckDerivativesBase<CheckDerivatives>::CheckDerivativesBase;
  void runOnOperation() final;
private:
  Value energy(OpBuilder &b, EvaluateOp source, ValueRange inputs);
  void compare(OpBuilder &b, Location loc, Value analytic, Value plus,
               Value minus, Value h, StringRef label);
  LogicalResult instrument(EvaluateOp source);
};

Value CheckDerivatives::energy(OpBuilder &b, EvaluateOp source, ValueRange inputs) {
  OperationState state(source.getLoc(), EvaluateOp::getOperationName());
  state.addOperands(inputs);
  state.addTypes(b.getF64Type());
  state.addAttribute("callee", source.getCalleeAttr());
  state.addAttribute("request_kinds", b.getDenseI32ArrayAttr({static_cast<int32_t>(Request::Energy)}));
  state.addAttribute("request_arguments", b.getDenseI64ArrayAttr({-1}));
  return b.create(state)->getResult(0);
}

void CheckDerivatives::compare(OpBuilder &b, Location loc, Value analytic,
                               Value plus, Value minus, Value h, StringRef label) {
  auto constant = [&](double x) { return arith::ConstantFloatOp::create(b, loc, b.getF64Type(), APFloat(x)); };
  Value fd = arith::DivFOp::create(b, loc,
      arith::SubFOp::create(b, loc, plus, minus),
      arith::MulFOp::create(b, loc, constant(2.0), h));
  Value error = math::AbsFOp::create(b, loc, arith::SubFOp::create(b, loc, analytic, fd));
  Value scale = arith::MaximumFOp::create(b, loc, math::AbsFOp::create(b, loc, analytic), math::AbsFOp::create(b, loc, fd));
  Value bound = arith::AddFOp::create(b, loc, constant(atol), arith::MulFOp::create(b, loc, constant(rtol), scale));
  Value agrees = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLE, error, bound);
  // A failure is printed to stderr before aborting; cf.assert's puts can
  // otherwise lose its buffered message when a runner aborts in a pipeline.
  ModuleOp module = getOperation();
  OpBuilder top(module.getBodyRegion());
  auto bytes = MemRefType::get({static_cast<int64_t>(label.size())}, b.getI8Type());
  SmallVector<APInt> data;
  for (unsigned char c : label.bytes()) data.emplace_back(8, c);
  unsigned serial = 0;
  std::string symbol;
  do { symbol = "__mdir_derivative_message_" + std::to_string(serial++); }
  while (SymbolTable::lookupSymbolIn(module, symbol));
  auto global = memref::GlobalOp::create(top, loc, symbol, top.getStringAttr("private"),
      bytes, DenseElementsAttr::get(RankedTensorType::get(bytes.getShape(), b.getI8Type()), data),
      true, IntegerAttr());
  auto dynamic = MemRefType::get({ShapedType::kDynamic}, b.getI8Type());
  if (!module.lookupSymbol<func::FuncOp>("mdrtDerivativeFailure")) {
    auto signature = b.getFunctionType({dynamic, b.getF64Type(), b.getF64Type(),
                                      b.getF64Type(), b.getF64Type()}, {});
    auto function = func::FuncOp::create(top, loc, "mdrtDerivativeFailure", signature);
    function.setPrivate();
  }
  Value failed = arith::XOrIOp::create(b, loc, agrees, arith::ConstantIntOp::create(b, loc, 1, 1));
  auto failure = scf::IfOp::create(b, loc, failed, false);
  OpBuilder at = OpBuilder::atBlockTerminator(failure.thenBlock());
  Value message = memref::GetGlobalOp::create(at, loc, bytes, global.getSymName());
  message = memref::CastOp::create(at, loc, dynamic, message);
  func::CallOp::create(at, loc, "mdrtDerivativeFailure", TypeRange{},
                       ValueRange{message, analytic, fd, error, bound});
}

LogicalResult CheckDerivatives::instrument(EvaluateOp source) {
  Location loc = source.getLoc();
  OpBuilder b(source);
  b.setInsertionPointAfter(source);
  SmallVector<Value> inputs(source.getOperands());
  auto field = dyn_cast<FieldType>(inputs.front().getType());
  if (!field || field.getNumComponents() != 3 || !field.getElementType().isF64())
    return source.emitError("derivative checking requires positions with three f64 components");
  auto bufferType = MemRefType::get({ShapedType::kDynamic, 3}, b.getF64Type());
  auto constant = [&](OpBuilder &at, double x) -> Value { return arith::ConstantFloatOp::create(at, loc, at.getF64Type(), APFloat(x)); };
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value three = arith::ConstantIndexOp::create(b, loc, 3);
  Value positions = mdir::mdrt::ToBufferOp::create(b, loc, bufferType, inputs[0]);
  Value count = memref::DimOp::create(b, loc, positions, zero);
  std::string prefix = "derivative check of '" + source.getCallee().str() + "': ";
  for (auto [resultIndex, kind] : llvm::enumerate(source.getRequestKinds())) {
    Request request = static_cast<Request>(kind);
    if (request == Request::Energy) continue;
    Value analytic = source.getResult(resultIndex);
    if (request == Request::Derivative) {
      int64_t index = source.getRequestArguments()[resultIndex];
      Value value = inputs[index];
      if (!value.getType().isF64()) return source.emitError("derivative checking requires f64 parameters");
      Value h = arith::MulFOp::create(b, loc, constant(b, step),
          arith::MaximumFOp::create(b, loc, constant(b, 1.0), math::AbsFOp::create(b, loc, value)));
      auto shifted = inputs;
      shifted[index] = arith::AddFOp::create(b, loc, value, h);
      Value plus = energy(b, source, shifted);
      shifted[index] = arith::SubFOp::create(b, loc, value, h);
      Value minus = energy(b, source, shifted);
      compare(b, loc, analytic, plus, minus, h, prefix + "parameter " + std::to_string(index));
      continue;
    }
    if (request == Request::Forces) {
      Value forces = mdir::mdrt::ToBufferOp::create(b, loc, bufferType, analytic);
      auto particles = scf::ForOp::create(b, loc, zero, count, one);
      OpBuilder particle = OpBuilder::atBlockTerminator(particles.getBody());
      auto components = scf::ForOp::create(particle, loc, zero, three, one);
      OpBuilder at = OpBuilder::atBlockTerminator(components.getBody());
      SmallVector<Value, 2> indices{particles.getInductionVar(), components.getInductionVar()};
      Value plusBuffer = memref::AllocOp::create(at, loc, bufferType, ValueRange{count});
      Value minusBuffer = memref::AllocOp::create(at, loc, bufferType, ValueRange{count});
      memref::CopyOp::create(at, loc, positions, plusBuffer);
      memref::CopyOp::create(at, loc, positions, minusBuffer);
      Value x = memref::LoadOp::create(at, loc, positions, indices);
      Value h = arith::MulFOp::create(at, loc, constant(at, step),
          arith::MaximumFOp::create(at, loc, constant(at, 1.0), math::AbsFOp::create(at, loc, x)));
      memref::StoreOp::create(at, loc, arith::AddFOp::create(at, loc, x, h), plusBuffer, indices);
      memref::StoreOp::create(at, loc, arith::SubFOp::create(at, loc, x, h), minusBuffer, indices);
      auto shifted = inputs;
      shifted[0] = mdir::mdrt::FromBufferOp::create(at, loc, field, plusBuffer);
      Value plusField = shifted[0];
      Value plus = energy(at, source, shifted);
      shifted[0] = mdir::mdrt::FromBufferOp::create(at, loc, field, minusBuffer);
      Value minusField = shifted[0];
      Value minus = energy(at, source, shifted);
      Value force = memref::LoadOp::create(at, loc, forces, indices);
      compare(at, loc, arith::NegFOp::create(at, loc, force), plus, minus, h, prefix + "force component");
      // Export before freeing: storage must not recycle these imported
      // buffers as field scratch. Each perturbation is immutable after import.
      memref::DeallocOp::create(at, loc, mdir::mdrt::ToBufferOp::create(at, loc, bufferType, plusField));
      memref::DeallocOp::create(at, loc, mdir::mdrt::ToBufferOp::create(at, loc, bufferType, minusField));
      continue;
    }
    // A restricted-triclinic cell supports the six independent strains:
    // three stretches and the three upper-triangular shears. W_ab uses
    // strain x_b += epsilon*x_a. Lower shears give the transpose, tested by
    // symmetry for periodic, rotationally invariant potentials.
    Operation *cell = inputs[1].getDefiningOp();
    if (!cell || !isa<OrthorhombicCellOp, TriclinicCellOp>(cell))
      return source.emitError("virial checking requires an explicit orthorhombic or triclinic cell at the evaluation");
    SmallVector<Value> edges(cell->getOperands());
    while (edges.size() < 6) edges.push_back(constant(b, 0.0));
    // Columns of H: (Lx,0,0), (bx,Ly,0), (cx,cy,Lz).
    SmallVector<Value> matrix{edges[0], edges[3], edges[4], constant(b, 0.0), edges[1], edges[5], constant(b, 0.0), constant(b, 0.0), edges[2]};
    for (int row = 0; row < 3; ++row) for (int col = row; col < 3; ++col) {
      Value h = constant(b, step);
      Value energies[2];
      for (int sign : {1, -1}) {
        Value delta = constant(b, sign * step);
        Value buffer = memref::AllocOp::create(b, loc, bufferType, ValueRange{count});
        auto loop = scf::ForOp::create(b, loc, zero, count, one);
        OpBuilder at = OpBuilder::atBlockTerminator(loop.getBody());
        Value c = arith::ConstantIndexOp::create(at, loc, col);
        // Every component is restored before applying this strain.
        for (int axis = 0; axis < 3; ++axis) {
          Value a = arith::ConstantIndexOp::create(at, loc, axis);
          Value x = memref::LoadOp::create(at, loc, positions, ValueRange{loop.getInductionVar(), a});
          if (axis == row) {
            Value arm = memref::LoadOp::create(at, loc, positions, ValueRange{loop.getInductionVar(), c});
            x = arith::AddFOp::create(at, loc, x, arith::MulFOp::create(at, loc, delta, arm));
          }
          memref::StoreOp::create(at, loc, x, buffer, ValueRange{loop.getInductionVar(), a});
        }
        SmallVector<Value> strained(matrix);
        for (int column = 0; column < 3; ++column)
          strained[3 * row + column] = arith::AddFOp::create(b, loc, matrix[3 * row + column], arith::MulFOp::create(b, loc, delta, matrix[3 * col + column]));
        auto shifted = inputs;
        shifted[0] = mdir::mdrt::FromBufferOp::create(b, loc, field, buffer);
        shifted[1] = TriclinicCellOp::create(b, loc, inputs[1].getType(), strained[0], strained[4], strained[8], strained[1], strained[2], strained[5]);
        energies[sign == 1 ? 0 : 1] = energy(b, source, shifted);
        memref::DeallocOp::create(b, loc, mdir::mdrt::ToBufferOp::create(b, loc, bufferType, shifted[0]));
      }
      Value w = vector::ExtractOp::create(b, loc, analytic, 3 * col + row);
      compare(b, loc, arith::NegFOp::create(b, loc, w), energies[0], energies[1], h, prefix + "virial " + std::to_string(col) + "," + std::to_string(row));
    }
  }
  return success();
}
void CheckDerivatives::runOnOperation() {
  if (!(step > 0 && atol >= 0 && rtol >= 0) ||
      !std::isfinite(step) || !std::isfinite(atol) || !std::isfinite(rtol)) {
    getOperation().emitError("derivative check needs a finite positive step and finite nonnegative tolerances");
    return signalPassFailure();
  }
  getOperation()->setAttr("md.derivative_check", UnitAttr::get(&getContext()));
  SmallVector<EvaluateOp> evaluations;
  getOperation()->walk([&](EvaluateOp op) { evaluations.push_back(op); });
  for (EvaluateOp evaluation : evaluations)
    if (failed(instrument(evaluation))) return signalPassFailure();
}
} // namespace
} } // namespace mdir::md
