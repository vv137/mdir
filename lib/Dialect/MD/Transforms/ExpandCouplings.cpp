// Expansion of the ops of the couplings of dyn.
//
// See docs/dynamics.md, Section 2.2, and docs/decisions/D326.md. Each op
// becomes the operations that lib/Driver/Builder.cpp writes for it, on the
// same operands and in the same order of dependence, so that a run in the
// deterministic mode stays bitwise that of the text (Section 2.2.8).

#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mdir;
using md::FieldType;

namespace mdir {
namespace md {
#define GEN_PASS_DEF_EXPANDCOUPLINGS
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"
} // namespace md
} // namespace mdir

namespace {

/// Builds the arithmetic of the expansions at one place.
struct Emitter {
  OpBuilder &b;
  Location loc;

  Type f64() { return b.getF64Type(); }
  VectorType vector3() { return VectorType::get({3}, b.getF64Type()); }

  Value real(double value) {
    return arith::ConstantOp::create(b, loc, b.getF64FloatAttr(value));
  }
  Value reals(double value) {
    return arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(vector3(), b.getF64FloatAttr(value)));
  }
  Value i32(uint32_t value) {
    return arith::ConstantOp::create(
        b, loc, b.getI32IntegerAttr(static_cast<int32_t>(value)));
  }
  Value i64(int64_t value) {
    return arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(value));
  }
  Value add(Value x, Value y) { return arith::AddFOp::create(b, loc, x, y); }
  Value sub(Value x, Value y) { return arith::SubFOp::create(b, loc, x, y); }
  Value mul(Value x, Value y) { return arith::MulFOp::create(b, loc, x, y); }
  Value div(Value x, Value y) { return arith::DivFOp::create(b, loc, x, y); }
  Value splat(Value x) {
    return vector::BroadcastOp::create(b, loc, vector3(), x);
  }
  Value sum3(Value x) {
    return vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, x);
  }
  Value select(Value condition, Value x, Value y) {
    return arith::SelectOp::create(b, loc, condition, x, y);
  }
  Value compare(arith::CmpFPredicate predicate, Value x, Value y) {
    return arith::CmpFOp::create(b, loc, predicate, x, y);
  }
  Value xori(Value x, Value y) { return arith::XOrIOp::create(b, loc, x, y); }
};

/// The value of one particle of `field` in a kernel.
Type getKernelType(Value field) {
  return cast<FieldType>(field.getType()).getKernelValueType();
}

/// Builds `Loop gather(gathered) {body} : result` with the kernel that
/// `body` writes, given the block arguments, and yields what it returns.
template <typename Loop>
Value buildLoop(OpBuilder &b, Location loc, Type result, ValueRange gathered,
                function_ref<Value(Emitter &, ValueRange)> body) {
  auto loop = Loop::create(b, loc, result, gathered);
  Block *kernel = new Block();
  loop.getKernel().push_back(kernel);
  for (Value field : gathered)
    kernel->addArgument(getKernelType(field), loc);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(kernel);
  Emitter e{b, loc};
  Value yielded = body(e, kernel->getArguments());
  md::YieldOp::create(b, loc, ValueRange{yielded});
  return loop.getResult();
}

class ExpandCouplings
    : public md::impl::ExpandCouplingsBase<ExpandCouplings> {
public:
  void runOnOperation() override;

private:
  /// The function `name` of the host, declared in the module if it is not,
  /// with `llvm.emit_c_interface` if `cInterface`, as the driver declares
  /// it.
  FailureOr<func::FuncOp> getHostFunction(StringRef name, FunctionType type,
                                          bool cInterface);
  /// `func.call @name(arguments)` of the host before `op`.
  FailureOr<func::CallOp> callHost(Operation *op, StringRef name,
                                   ValueRange arguments, TypeRange results,
                                   bool cInterface = false);

  void expandMomentum(dyn::MomentumOp op);
  void expandKinetic(dyn::KineticOp op);
  void expandHalfStepExcess(dyn::HalfStepExcessOp op);
  void expandVelocities(Operation *last);
  void expandNoise(dyn::NoiseOp op);
};

} // namespace

FailureOr<func::FuncOp>
ExpandCouplings::getHostFunction(StringRef name, FunctionType type,
                                 bool cInterface) {
  ModuleOp module = getOperation();
  if (auto function = module.lookupSymbol<func::FuncOp>(name)) {
    if (function.getFunctionType() != type)
      return function.emitOpError()
             << "expected the type " << type << " of the host function, got "
             << function.getFunctionType();
    return function;
  }
  OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
  auto function = func::FuncOp::create(b, module.getLoc(), name, type);
  function.setPrivate();
  if (cInterface)
    function->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  return function;
}

FailureOr<func::CallOp> ExpandCouplings::callHost(Operation *op,
                                                  StringRef name,
                                                  ValueRange arguments,
                                                  TypeRange results,
                                                  bool cInterface) {
  FunctionType type = FunctionType::get(op->getContext(),
                                        TypeRange(arguments), results);
  FailureOr<func::FuncOp> function = getHostFunction(name, type, cInterface);
  if (failed(function))
    return failure();
  OpBuilder b(op);
  return func::CallOp::create(b, op->getLoc(), *function, arguments);
}

/// P = Σ m v, as the removal of the motion of the center of mass sums it.
void ExpandCouplings::expandMomentum(dyn::MomentumOp op) {
  OpBuilder b(op);
  Value sum = buildLoop<md::SumParticlesOp>(
      b, op.getLoc(), op.getType(), {op.getVelocities(), op.getMasses()},
      [](Emitter &e, ValueRange args) {
        return e.mul(e.splat(args[1]), args[0]);
      });
  op.replaceAllUsesWith(sum);
  op.erase();
}

/// K of each axis, Σ (m / 2) (v ⊙ v), or K, Σ (m (v · v)) / 2.
void ExpandCouplings::expandKinetic(dyn::KineticOp op) {
  OpBuilder b(op);
  bool axes = isa<VectorType>(op.getType());
  Value sum = buildLoop<md::SumParticlesOp>(
      b, op.getLoc(), op.getType(), {op.getVelocities(), op.getMasses()},
      [axes](Emitter &e, ValueRange args) {
        Value v = args[0], m = args[1];
        if (axes) {
          Value half = e.real(0.5);
          Value halfMass = e.splat(e.mul(half, m));
          return e.mul(halfMass, e.mul(v, v));
        }
        Value half = e.real(0.5);
        Value squares = e.sum3(e.mul(v, v));
        return e.mul(half, e.mul(m, squares));
      });
  op.replaceAllUsesWith(sum);
  op.erase();
}

/// E = (Δt² / 8) Σ F ⊙ F / m, a particle of mass 0 left out [Jung2018].
void ExpandCouplings::expandHalfStepExcess(dyn::HalfStepExcessOp op) {
  OpBuilder b(op);
  Value sum = buildLoop<md::SumParticlesOp>(
      b, op.getLoc(), op.getType(), {op.getForces(), op.getMasses()},
      [](Emitter &e, ValueRange args) {
        Value f = args[0], m = args[1];
        Value squares = e.mul(f, f);
        Value zero = e.real(0.0), one = e.real(1.0);
        Value massless = e.compare(arith::CmpFPredicate::OEQ, m, zero);
        Value safe = e.select(massless, one, m);
        Value inverse = e.div(one, safe);
        Value weight = e.select(massless, zero, inverse);
        return e.mul(squares, e.splat(weight));
      });
  Emitter e{b, op.getLoc()};
  Value dt2 = e.mul(op.getDt(), op.getDt());
  Value scale = e.mul(dt2, e.real(1.25e-01));
  op.replaceAllUsesWith(e.mul(sum, e.splat(scale)));
  op.erase();
}

/// The velocities of `op`, a dyn.shift or dyn.scale.
static Value getUpdated(Operation *op) { return op->getOperand(0); }

/// Returns true if `op` is a dyn.shift or dyn.scale.
static bool isVelocityUpdate(Operation *op) {
  return op && isa<dyn::ShiftOp, dyn::ScaleOp>(op);
}

/// A chain of dyn.shift and dyn.scale that ends at `last`, whose
/// intermediate values have no other use, as one md.map_particles that
/// applies them in order (docs/dynamics.md, Section 2.2.3): the products
/// keep their order, since (v α) μ⁻¹ and v (α μ⁻¹) need not round alike.
void ExpandCouplings::expandVelocities(Operation *last) {
  SmallVector<Operation *> chain{last};
  Value base = getUpdated(last);
  while (Operation *before = base.getDefiningOp()) {
    if (!isVelocityUpdate(before) || !base.hasOneUse())
      break;
    chain.push_back(before);
    base = getUpdated(before);
  }
  std::reverse(chain.begin(), chain.end());
  OpBuilder b(last);
  Value result = buildLoop<md::MapParticlesOp>(
      b, last->getLoc(), last->getResult(0).getType(), {base},
      [&](Emitter &e, ValueRange args) {
        Value value = args[0];
        for (Operation *op : chain) {
          if (auto shift = dyn_cast<dyn::ShiftOp>(op)) {
            value = e.sub(value, shift.getShift());
            continue;
          }
          Value factor = cast<dyn::ScaleOp>(op).getFactor();
          if (!isa<VectorType>(factor.getType()))
            factor = e.splat(factor);
          value = e.mul(factor, value);
        }
        return value;
      });
  last->getResult(0).replaceAllUsesWith(result);
  for (Operation *op : llvm::reverse(chain))
    op->erase();
}

/// v' = c v + (σ / √m) R, with R three normal numbers of one block of
/// Philox 4x32-10 [Salmon2011] under the key of A13: the seed; the counter
/// of the step, the number of the particle, and stream 2 in the high byte
/// of the last word. The four words give four uniform numbers,
/// (w + 1/2) 2^-32, and the method of Box and Muller two pairs of normal
/// numbers, of which the first three are taken. A particle of mass 0 keeps
/// its velocity.
void ExpandCouplings::expandNoise(dyn::NoiseOp op) {
  OpBuilder b(op);
  Value step = op.getStep();
  auto seed = static_cast<uint64_t>(op.getSeed());
  double c = op.getDecay().convertToDouble();
  double spread = op.getSpread().convertToDouble();
  Value result = buildLoop<md::MapParticlesOp>(
      b, op.getLoc(), op.getType(),
      {op.getVelocities(), op.getMasses(), op.getIds()},
      [&](Emitter &e, ValueRange args) {
        Value v = args[0], m = args[1], id = args[2];
        Type i32 = e.b.getI32Type();
        Value shift = arith::ShRUIOp::create(e.b, e.loc, step, e.i64(32));
        Value c0 = arith::TruncIOp::create(e.b, e.loc, i32, step);
        Value c1 = arith::TruncIOp::create(e.b, e.loc, i32, shift);
        Value c3 = e.i32(2u << 24);
        Value ma = e.i32(0xD2511F53u), mb = e.i32(0xCD9E8D57u);
        Value c2 = id;
        auto k0 = static_cast<uint32_t>(seed);
        auto k1 = static_cast<uint32_t>(seed >> 32);
        for (int round = 0; round != 10; ++round) {
          if (round != 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
          }
          auto p0 = arith::MulUIExtendedOp::create(e.b, e.loc, ma, c0);
          auto p1 = arith::MulUIExtendedOp::create(e.b, e.loc, mb, c2);
          Value key0 = e.i32(k0), key1 = e.i32(k1);
          Value x0 = e.xori(e.xori(p1.getHigh(), c1), key0);
          Value x2 = e.xori(e.xori(p0.getHigh(), c3), key1);
          c0 = x0;
          c1 = p1.getLow();
          c2 = x2;
          c3 = p0.getLow();
        }
        Value half = e.real(5.0e-01);
        Value scale = e.real(0x1.0p-32);
        Value minus2 = e.real(-2.0);
        Value twoPi = e.real(6.283185307179586);
        Value words[] = {c0, c1, c2, c3};
        Value uniform[4];
        for (int w = 0; w != 4; ++w) {
          Value real = arith::UIToFPOp::create(e.b, e.loc, e.f64(), words[w]);
          uniform[w] = e.mul(e.add(real, half), scale);
        }
        Value cosines[2], sine;
        for (int pair = 0; pair != 2; ++pair) {
          Value log = math::LogOp::create(e.b, e.loc, uniform[2 * pair]);
          Value radius = math::SqrtOp::create(e.b, e.loc, e.mul(minus2, log));
          Value angle = e.mul(twoPi, uniform[2 * pair + 1]);
          Value cosine = math::CosOp::create(e.b, e.loc, angle);
          cosines[pair] = e.mul(radius, cosine);
          if (pair == 0)
            sine = e.mul(radius, math::SinOp::create(e.b, e.loc, angle));
        }
        Value noise = vector::FromElementsOp::create(
            e.b, e.loc, e.vector3(), ValueRange{cosines[0], sine, cosines[1]});
        Value zero = e.real(0.0);
        Value massive = e.compare(arith::CmpFPredicate::OGT, m, zero);
        Value one = e.real(1.0);
        Value safe = e.select(massive, m, one);
        Value width = e.real(spread);
        Value root = math::SqrtOp::create(e.b, e.loc, safe);
        Value sigma = e.splat(e.div(width, root));
        Value kick = e.mul(sigma, noise);
        Value damped = e.mul(e.reals(c), v);
        Value moved = e.add(damped, kick);
        return e.select(massive, moved, v);
      });
  op.replaceAllUsesWith(result);
  op.erase();
}

void ExpandCouplings::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<Operation *> ops;
  module.walk([&](Operation *op) {
    if (isa<dyn::MomentumOp, dyn::KineticOp, dyn::HalfStepExcessOp,
            dyn::VelocityRescalingOp, dyn::NoseHooverOp, dyn::NoiseOp,
            dyn::ToBathOp>(op))
      ops.push_back(op);
    // The last update of each chain of the velocities.
    if (isVelocityUpdate(op)) {
      Value result = op->getResult(0);
      bool continued =
          result.hasOneUse() && isVelocityUpdate(*result.user_begin()) &&
          getUpdated(*result.user_begin()) == result;
      if (!continued)
        ops.push_back(op);
    }
  });

  for (Operation *op : ops) {
    Location loc = op->getLoc();
    if (auto momentum = dyn_cast<dyn::MomentumOp>(op)) {
      expandMomentum(momentum);
    } else if (auto kinetic = dyn_cast<dyn::KineticOp>(op)) {
      expandKinetic(kinetic);
    } else if (auto excess = dyn_cast<dyn::HalfStepExcessOp>(op)) {
      expandHalfStepExcess(excess);
    } else if (auto rescaling = dyn_cast<dyn::VelocityRescalingOp>(op)) {
      // The factor of [Bussi2007] from libmdrt, stream 0 of the key of A13.
      OpBuilder b(op);
      Emitter e{b, loc};
      Value seed = e.i64(rescaling.getSeed());
      Value target = e.real(rescaling.getTargetKinetic().convertToDouble());
      Value freedom = e.real(rescaling.getFreedom().convertToDouble());
      Value decay = e.real(rescaling.getDecay().convertToDouble());
      FailureOr<func::CallOp> call = callHost(
          op, "mdrtBussiFactor",
          {seed, rescaling.getStep(), rescaling.getKinetic(), target, freedom,
           decay},
          {e.f64()});
      if (failed(call))
        return signalPassFailure();
      rescaling.replaceAllUsesWith(call->getResult(0));
      op->erase();
    } else if (auto chain = dyn_cast<dyn::NoseHooverOp>(op)) {
      // The chain of the host (D163a), which the driver configures from
      // the control file; step 3a of #541 checks that its attributes agree.
      FailureOr<func::CallOp> call =
          callHost(op, "mdrtNoseHooverFactor",
                   {chain.getStep(), chain.getKinetic()}, {Float64Type::get(
                                                              &getContext())});
      if (failed(call))
        return signalPassFailure();
      chain.replaceAllUsesWith(call->getResult(0));
      op->erase();
    } else if (auto noise = dyn_cast<dyn::NoiseOp>(op)) {
      expandNoise(noise);
    } else if (auto bath = dyn_cast<dyn::ToBathOp>(op)) {
      if (failed(callHost(op, "mdrtAddBath", {bath.getEnergy()}, {},
                          /*cInterface=*/true)))
        return signalPassFailure();
      op->erase();
    } else {
      expandVelocities(op);
    }
  }
}
