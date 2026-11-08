// Cardinal assignment windows and their declared linear structure.
// Cox-de Boor recurrence [deBoor2001], docs/references.md.
#include "mdir/Dialect/MD/Transforms/Stencils.h"
#include "mdir/Dialect/MD/Transforms/AdjointInterface.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mdir::md;

Value mdir::md::cloneStencil(OpBuilder &b, Operation *source, Value data,
                             bool transpose) {
  bool scatter = isa<ScatterStencilOp>(source) != transpose;
  OperationState state(source->getLoc(), scatter
      ? ScatterStencilOp::getOperationName() : GatherStencilOp::getOperationName());
  state.addOperands({source->getOperand(0), source->getOperand(1),
                     source->getOperand(2), data});
  state.addTypes(transpose ? source->getOperand(3).getType()
                           : source->getResult(0).getType());
  state.addRegion();
  Operation *result = b.create(state);
  IRMapping mapping;
  source->getRegion(0).cloneInto(&result->getRegion(0), mapping);
  return result->getResult(0);
}

namespace {
// Evaluate a cardinal window without powers or a second derivative formula.
// At every recursion level the interval predicates are structural; smooth
// first/second derivatives at the order-four joins follow from the polynomial.
Value emitWindow(OpBuilder &b, BSplineWindowOp op) {
  Location loc = op.getLoc();
  ScalarEmitter e(b, loc);
  Value product = e.constant(1, b.getF64Type());
  int64_t order = op.getOrder();
  for (int axis = 0; axis != 3; ++axis) {
    Value t = vector::ExtractOp::create(b, loc, op.getDistance(),
                                       ArrayRef<int64_t>{axis});
    SmallVector<Value> values;
    for (int64_t j = 0; j < order; ++j) {
      Value lo = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGE,
                                      t, e.constant(j, b.getF64Type()));
      Value hi = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                      t, e.constant(j + 1, b.getF64Type()));
      Value inside = arith::AndIOp::create(b, loc, lo, hi);
      values.push_back(arith::SelectOp::create(b, loc, inside,
          e.constant(1, b.getF64Type()), e.constant(0, b.getF64Type())));
    }
    // values[j] is M_(n-1)(t-j), with M_1 the unit interval.
    for (int64_t n = 2; n <= order; ++n) {
      for (int64_t j = 0; j <= order - n; ++j) {
        Value u = e.sub(t, e.constant(j, b.getF64Type()));
        values[j] = e.scale(1.0 / (n - 1),
            e.add(e.mul(u, values[j]),
                  e.mul(e.sub(e.constant(n, b.getF64Type()), u),
                        values[j + 1])));
      }
    }
    product = e.mul(product, values.front());
  }
  return product;
}

struct WindowRule : DerivativeOpInterface::ExternalModel<WindowRule, BSplineWindowOp> {
  DerivativeOperand getDerivativeOperand(Operation *, unsigned) const {
    return DerivativeOperand::Differentiable;
  }
  LogicalResult emitDerivative(Operation *operation, Value,
                               ScalarDerivative &derivative, Value &tangent) const {
    auto window = cast<BSplineWindowOp>(operation);
    if (window.getOrder() != 4)
      return window.emitOpError(
          "window derivatives currently require order-four B-splines");
    Value expanded = emitWindow(derivative.getBuilder(), window);
    return derivative.get(expanded, tangent);
  }
  LogicalResult emitAdjoint(Operation *operation, OpBuilder &b,
                            ValueRange cotangents,
                            SmallVectorImpl<Value> &operandCotangents) const {
    auto window = cast<BSplineWindowOp>(operation);
    if (cotangents.size() != 1 || !cotangents[0].getType().isF64())
      return window.emitOpError("expected one scalar window cotangent");
    SmallVector<Value> components;
    for (int component = 0; component != 3; ++component) {
      SmallVector<double> unit(3, 0.0);
      unit[component] = 1.0;
      auto type = cast<VectorType>(window.getDistance().getType());
      Value direction = arith::ConstantOp::create(b, window.getLoc(), type,
          DenseElementsAttr::get(type, ArrayRef<double>(unit)));
      ScalarDerivative derivative(b, window.getDistance(), nullptr, direction);
      Value partial;
      if (failed(derivative.get(window.getWeight(), partial)))
        return failure();
      ScalarEmitter e(b, window.getLoc());
      components.push_back(partial ? e.mul(partial, cotangents[0])
                                   : e.constant(0, b.getF64Type()));
    }
    operandCotangents.assign(1, vector::FromElementsOp::create(
        b, window.getLoc(), VectorType::get({3}, b.getF64Type()), components));
    return success();
  }
};
}

void mdir::md::registerStencilDerivatives(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, MDDialect *) {
    BSplineWindowOp::attachInterface<WindowRule>(*context);
  });
}

namespace mdir::md {
#define GEN_PASS_DEF_DIFFERENTIATESTENCILS
#define GEN_PASS_DEF_EXPANDSTENCILWINDOWS
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"
namespace {
class DifferentiateStencils
    : public impl::DifferentiateStencilsBase<DifferentiateStencils> {
  void runOnOperation() final {
    SmallVector<Operation *> requests;
    getOperation()->walk([&](Operation *op) {
      if (isa<StencilJVPOp, StencilVJPOp>(op)) requests.push_back(op);
    });
    for (Operation *request : requests) {
      Operation *source = request->getOperand(0).getDefiningOp();
      unsigned operand = request->getAttrOfType<IntegerAttr>("operand").getInt();
      bool reverse = isa<StencilVJPOp>(request);
      OpBuilder b(request);
      Value result, seed = request->getOperand(1);
      if (auto window = dyn_cast<BSplineWindowOp>(source)) {
        // Use the same scalar derivative rules in both directions.
        if (!reverse) {
          ScalarDerivative derivative(b, window.getDistance(), nullptr, seed);
          if (failed(derivative.get(window.getWeight(), result))) {
            signalPassFailure(); return;
          }
        } else {
          SmallVector<Value> cotangents;
          auto rule = cast<DerivativeOpInterface>(source);
          if (failed(rule.emitAdjoint(b, ValueRange{seed}, cotangents))) {
            signalPassFailure(); return;
          }
          result = cotangents.front();
        }
      } else if (operand != 3) {
        request->emitOpError() << "unsupported stencil "
            << (reverse ? "VJP" : "JVP") << " for active "
            << (operand == 1 ? "positions" : operand == 2 ? "cell" : "relation")
            << " operand; structural indices do not prove continuous inactivity";
        signalPassFailure(); return;
      } else if (reverse) {
        auto transpose = cast<AdjointOpInterface>(source);
        result = transpose.emitTranspose(b, operand, seed, {});
      } else {
        result = cloneStencil(b, source, seed, false);
      }
      if (!result) {
        request->emitOpError("derivative produced no value");
        signalPassFailure(); return;
      }
      request->getResult(0).replaceAllUsesWith(result);
      request->erase();
    }
  }
};

class ExpandStencilWindows
    : public impl::ExpandStencilWindowsBase<ExpandStencilWindows> {
  void runOnOperation() final {
    SmallVector<BSplineWindowOp> windows;
    getOperation()->walk([&](BSplineWindowOp op) { windows.push_back(op); });
    for (BSplineWindowOp window : windows) {
      OpBuilder b(window);
      window.replaceAllUsesWith(emitWindow(b, window));
      window.erase();
    }
  }
};
}
}
