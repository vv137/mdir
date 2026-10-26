// Expansion of the ops of the couplings of dyn.
//
// See docs/dynamics.md, Section 2.2, and docs/decisions/D326.md. Each op
// becomes the operations that lib/Driver/Builder*.cpp write for it, on the
// same operands and in the same order of dependence, so that a run in the
// deterministic mode stays bitwise that of the text (Section 2.2.8).

#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ScopeExit.h"
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

  // The barostat (step 2b of #541).
  void expandPressure(dyn::PressureOp op);
  LogicalResult expandCellRescaling(dyn::CellRescalingOp op);
  void expandLoadCell(dyn::LoadCellOp op);
  /// dyn.scale_cell, and the dyn.store_cell of its results that follows it
  /// at once, if `store` is given, interleaved axis by axis as the driver
  /// writes them (emitStrain).
  LogicalResult expandCell(dyn::ScaleCellOp scale, dyn::StoreCellOp store);
  LogicalResult expandStoreCell(dyn::StoreCellOp op);
  void expandGroupKinetic(dyn::GroupKineticOp op);
  void expandScalePositions(dyn::ScalePositionsOp op);
  void expandScalingWork(dyn::ScalingWorkOp op);
  void expandLoadPressureState(dyn::LoadPressureStateOp op);
  LogicalResult expandStorePressureState(dyn::StorePressureStateOp op);
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

/// The places of the edges that each value of the cell is scaled by: the
/// edges by their own, b_x and c_x by that of x, c_y by that of y (D127).
static const int kColumn[] = {0, 1, 2, 0, 0, 1};

/// The volume (L_0 L_1) L_2, as emitStrain computes it; the pipeline's cse
/// takes the first of the ops that compute it.
static Value buildVolume(Emitter &e, Value l0, Value l1, Value l2) {
  return e.mul(e.mul(l0, l1), l2);
}

/// P_aa = c_bar (2 K_a + W_aa + C / (3 V)) / V (emitStrain).
void ExpandCouplings::expandPressure(dyn::PressureOp op) {
  OpBuilder b(op);
  Emitter e{b, op.getLoc()};
  Value volume = buildVolume(e, op.getL0(), op.getL1(), op.getL2());
  Value twice = e.add(op.getKinetic(), op.getKinetic());
  Value virial = e.add(twice, op.getVirial());
  Value constant = e.div(op.getConstant(), volume);
  Value third = e.mul(constant, e.real(1.0 / 3.0));
  Value total = e.add(virial, e.splat(third));
  Value bar = e.div(e.real(16.6053906717), volume);
  op.replaceAllUsesWith(e.mul(total, e.splat(bar)));
  op.erase();
}

/// The strain of [Bernetti2020] from libmdrt, stream 1 of the key of A13
/// (emitStrain).
LogicalResult
ExpandCouplings::expandCellRescaling(dyn::CellRescalingOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Emitter e{b, loc};
  Type f64 = e.f64();
  Value volume = buildVolume(e, op.getL0(), op.getL1(), op.getL2());
  Value p = op.getPressures();
  Value seed = e.i64(op.getSeed());
  Value step = op.getStep();
  auto beta = [&](unsigned k) {
    return e.real(cast<FloatAttr>(op.getCompressibility()[k])
                      .getValueAsDouble());
  };
  Value target = e.real(op.getTarget().convertToDouble());
  Value kT = e.real(op.getKT().convertToDouble());
  Value rate = e.real(op.getRate().convertToDouble());
  auto call = [&](StringRef name, ArrayRef<Value> arguments) -> Value {
    FailureOr<func::CallOp> done = callHost(op, name, arguments, {f64});
    return succeeded(done) ? done->getResult(0) : Value();
  };
  Value strain;
  StringRef coupling = op.getCoupling();
  if (coupling == "anisotropic") {
    // Each axis by the strain of its own pressure and noise, eq. (9b) of
    // [Bernetti2020] on each (D163c).
    Value strains[3];
    for (int k = 0; k != 3; ++k) {
      Value pressure = vector::ExtractOp::create(b, loc, p, k);
      Value axis = e.i64(k);
      strains[k] = call("mdrtBarostatStrainAxis",
                        {seed, step, axis, pressure, target, volume, kT,
                         beta(k), rate});
      if (!strains[k])
        return failure();
    }
    strain = vector::FromElementsOp::create(b, loc, e.vector3(),
                                            ValueRange(strains));
  } else if (coupling == "isotropic") {
    // The mean pressure gives ε, the change of ln V, by a step of λ = √V
    // (eq. S7 of [Bernetti2020]); each axis scales by ε / 3.
    Value third = e.real(1.0 / 3.0);
    Value mean = e.mul(e.sum3(p), third);
    Value change = call("mdrtBarostatStrain", {seed, step, mean, target,
                                               volume, kT, beta(0), rate});
    if (!change)
      return failure();
    strain = e.splat(e.mul(change, third));
  } else {
    // The mean pressure of x and y gives the change of ln A, that of z the
    // change of ln L, by eqs. (9a) and (9b) of [Bernetti2020] (D119).
    Value px = vector::ExtractOp::create(b, loc, p, 0);
    Value py = vector::ExtractOp::create(b, loc, p, 1);
    Value pz = vector::ExtractOp::create(b, loc, p, 2);
    Value half = e.real(0.5);
    Value area = e.mul(e.add(px, py), half);
    Value tension = e.real(op.getTension() ? op.getTension()->convertToDouble()
                                           : 0.0);
    Value xy = call("mdrtBarostatStrainArea",
                    {seed, step, area, target, volume, op.getL2(), kT, beta(0),
                     rate, tension});
    Value z = call("mdrtBarostatStrainHeight",
                   {seed, step, pz, target, volume, kT, beta(1), rate});
    if (!xy || !z)
      return failure();
    Value halfXY = e.mul(xy, half);
    strain = vector::FromElementsOp::create(b, loc, e.vector3(),
                                            ValueRange{halfXY, halfXY, z});
  }
  op.replaceAllUsesWith(strain);
  op.erase();
  return success();
}

void ExpandCouplings::expandLoadCell(dyn::LoadCellOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  SmallVector<Value> values;
  for (unsigned k = 0, n = op.getNumResults(); k != n; ++k) {
    Value index = arith::ConstantIndexOp::create(b, loc, k);
    values.push_back(memref::LoadOp::create(b, loc, op.getMemory(), index));
  }
  op.replaceAllUsesWith(values);
  op.erase();
}

LogicalResult ExpandCouplings::expandCell(dyn::ScaleCellOp scale,
                                          dyn::StoreCellOp store) {
  OpBuilder b(scale);
  Location loc = scale.getLoc();
  ValueRange cell = scale.getCell();
  Value factors[3];
  SmallVector<Value> scaled;
  for (unsigned k = 0, n = cell.size(); k != n; ++k) {
    if (k < 3)
      factors[k] = vector::ExtractOp::create(b, loc, scale.getFactors(), k);
    scaled.push_back(
        arith::MulFOp::create(b, loc, cell[k], factors[kColumn[k]]));
    if (store) {
      Value index = arith::ConstantIndexOp::create(b, loc, k);
      memref::StoreOp::create(b, loc, scaled.back(), store.getMemory(),
                              index);
    }
  }
  scale.replaceAllUsesWith(scaled);
  scale.erase();
  if (!store)
    return success();
  // The record, after the stores: before the dyn.store_cell, which follows
  // them.
  ValueRange values = store.getCell();
  if (failed(callHost(store, "mdrtSetBox", values.take_front(3), {},
                      /*cInterface=*/true)))
    return failure();
  if (values.size() == 6 &&
      failed(callHost(store, "mdrtSetTilt", values.drop_front(3), {},
                      /*cInterface=*/true)))
    return failure();
  store.erase();
  return success();
}

LogicalResult ExpandCouplings::expandStoreCell(dyn::StoreCellOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  ValueRange values = op.getCell();
  for (unsigned k = 0, n = values.size(); k != n; ++k) {
    Value index = arith::ConstantIndexOp::create(b, loc, k);
    memref::StoreOp::create(b, loc, values[k], op.getMemory(), index);
  }
  if (failed(callHost(op, "mdrtSetBox", values.take_front(3), {},
                      /*cInterface=*/true)))
    return failure();
  if (values.size() == 6 &&
      failed(callHost(op, "mdrtSetTilt", values.drop_front(3), {},
                      /*cInterface=*/true)))
    return failure();
  op.erase();
  return success();
}

/// Builds the op `name`, md.sum_tuples or md.gather_tuples, over the tuples
/// of `relation` at `positions` in `cell`, with the displacements of its
/// members 1 to `last` from member 0 and the fields `gathered`; `body`
/// writes the kernel from its arguments (the displacements, then each
/// field for each member) and returns what it yields.
static Operation *
buildTupleOp(OpBuilder &b, Location loc, StringRef name, Type result,
             Value relation, Value positions, Value cell, unsigned last,
             ValueRange gathered,
             function_ref<SmallVector<Value>(Emitter &, ValueRange)> body) {
  auto type = cast<md::RelationType>(relation.getType());
  SmallVector<md::Coordinate> coordinates;
  for (unsigned k = 1; k <= last; ++k)
    coordinates.push_back({md::CoordinateKind::Displacement,
                           {static_cast<int64_t>(k), 0}});
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  md::getCoordinateAttrs(b, coordinates, kinds, members);
  OperationState state(loc, name);
  state.addOperands({relation, positions, cell});
  state.addOperands(gathered);
  state.addAttribute("coordinate_kinds", kinds);
  state.addAttribute("coordinate_members", members);
  state.addAttribute("operandSegmentSizes",
                     b.getDenseI32ArrayAttr(
                         {1, 1, 1, static_cast<int32_t>(gathered.size()), 0}));
  Region *region = state.addRegion();
  state.addTypes(result);
  Block *kernel = new Block();
  region->push_back(kernel);
  for (const md::Coordinate &coordinate : coordinates)
    kernel->addArgument(md::getCoordinateType(coordinate.kind, b.getF64Type()),
                        loc);
  for (Value field : gathered)
    for (unsigned k = 0; k != type.getArity(); ++k)
      kernel->addArgument(getKernelType(field), loc);
  Operation *op;
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(kernel);
    Emitter e{b, loc};
    SmallVector<Value> yielded = body(e, kernel->getArguments());
    md::YieldOp::create(b, loc, yielded);
  }
  op = b.create(state);
  return op;
}

/// `factor` (f64) times `value` (vector<3xf64>), as SiteKernel::scale
/// writes it.
static Value scaleBy(Emitter &e, Value factor, Value value) {
  return e.mul(e.splat(factor), value);
}

/// K_g, E_g, and K_half,g of the centers of the groups (emitGroupKinetic):
/// over the particles first, less the motion within each group
/// [Jung2020, eq. 42].
void ExpandCouplings::expandGroupKinetic(dyn::GroupKineticOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Emitter e{b, loc};
  Value x = op.getPositions(), v = op.getVelocities(), f = op.getForces(),
        m = op.getMasses(), cell = op.getCell();
  VectorType vec3 = e.vector3();
  Value kinetic = buildLoop<md::SumParticlesOp>(
      b, loc, vec3, {v, m}, [](Emitter &e, ValueRange args) {
        Value half = e.real(0.5);
        Value halfMass = e.splat(e.mul(half, args[1]));
        return e.mul(halfMass, e.mul(args[0], args[0]));
      });
  Value force = buildLoop<md::SumParticlesOp>(
      b, loc, vec3, {f, m}, [](Emitter &e, ValueRange args) {
        Value squares = e.mul(args[0], args[0]);
        Value zero = e.real(0.0), one = e.real(1.0);
        Value massless =
            e.compare(arith::CmpFPredicate::OEQ, args[1], zero);
        Value safe = e.select(massless, one, args[1]);
        Value inverse = e.div(one, safe);
        Value weight = e.select(massless, zero, inverse);
        return e.mul(squares, e.splat(weight));
      });
  for (Value group : op.getGroups()) {
    unsigned arity = cast<md::RelationType>(group.getType()).getArity();
    unsigned count = arity - 1;
    // Twice the kinetic energy of the motion about the center, and
    // Σ F_k² / m_k − F_g² / M_g, of each axis.
    for (int part = 0; part != 2; ++part) {
      Operation *sum = buildTupleOp(
          b, loc, md::SumTuplesOp::getOperationName(), vec3, group, x, cell,
          1, {part == 0 ? v : f, m},
          [&](Emitter &e, ValueRange args) -> SmallVector<Value> {
            ValueRange values = args.slice(1, arity);
            ValueRange masses = args.slice(1 + arity, arity);
            Value total = masses[0];
            for (unsigned j = 1; j <= count; ++j)
              total = e.add(total, masses[j]);
            Value result;
            if (part == 0) {
              Value moment = scaleBy(e, masses[0], values[0]);
              for (unsigned j = 1; j <= count; ++j)
                moment = e.add(moment, scaleBy(e, masses[j], values[j]));
              Value center = scaleBy(e, e.div(e.real(1.0), total), moment);
              for (unsigned j = 0; j <= count; ++j) {
                Value relative = e.sub(values[j], center);
                Value term =
                    scaleBy(e, masses[j], e.mul(relative, relative));
                result = j == 0 ? term : e.add(result, term);
              }
            } else {
              Value resultant = values[0];
              for (unsigned j = 1; j <= count; ++j)
                resultant = e.add(resultant, values[j]);
              for (unsigned j = 0; j <= count; ++j) {
                // The order of the arguments of SiteKernel::scale as the
                // compiler of the driver evaluates them.
                Value square = e.mul(values[j], values[j]);
                Value term =
                    scaleBy(e, e.div(e.real(1.0), masses[j]), square);
                result = j == 0 ? term : e.add(result, term);
              }
              Value square = e.mul(resultant, resultant);
              Value whole = scaleBy(e, e.div(e.real(1.0), total), square);
              result = e.sub(result, whole);
            }
            return {result};
          });
      Value within = sum->getResult(0);
      if (part == 0) {
        Value half = e.mul(within, e.reals(0.5));
        kinetic = e.sub(kinetic, half);
      } else {
        force = e.sub(force, within);
      }
    }
  }
  // Without the motion of the center of mass, if the run removes it.
  if (Value total = op.getCenter()) {
    Value momentum = buildLoop<md::SumParticlesOp>(
        b, loc, vec3, {v, m}, [](Emitter &e, ValueRange args) {
          return e.mul(e.splat(args[1]), args[0]);
        });
    Value velocity = e.div(momentum, total);
    Value product = e.mul(momentum, velocity);
    Value center = e.mul(e.reals(0.5), product);
    kinetic = e.sub(kinetic, center);
  }
  Value dt = op.getDt();
  Value dt2 = e.mul(dt, dt);
  Value scale = e.mul(dt2, e.real(1.25e-01));
  Value excess = e.mul(force, e.splat(scale));
  Value half = e.add(kinetic, excess);
  op.getKinetic().replaceAllUsesWith(kinetic);
  op.getExcess().replaceAllUsesWith(excess);
  op.getHalf().replaceAllUsesWith(half);
  op.erase();
}

/// The value 1 − μ, `arith.subf dense<1.0>, mu`, that the block of `op`
/// computes before it, or null.
static Value findOneMinus(Operation *op, Value mu) {
  for (Operation *before = op->getPrevNode(); before;
       before = before->getPrevNode()) {
    auto sub = dyn_cast<arith::SubFOp>(before);
    if (!sub || sub.getRhs() != mu)
      continue;
    DenseElementsAttr one;
    if (matchPattern(sub.getLhs(), m_Constant(&one)) && one.isSplat() &&
        one.getSplatValue<APFloat>().isExactlyValue(1.0))
      return sub.getResult();
  }
  return {};
}

/// μ ⊙ x, and each rigid group back by (μ − 1) times the places of its
/// members about its center (emitGroupScaling, D72).
void ExpandCouplings::expandScalePositions(dyn::ScalePositionsOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Emitter e{b, loc};
  Value x = op.getPositions(), mu = op.getFactors();
  // 1 − μ: that which the block computes before the op, as the driver
  // writes it once for the work of the scaling and the groups, or a new one.
  Value oneMinusMu = findOneMinus(op, mu);
  if (!oneMinusMu)
    oneMinusMu = e.sub(e.reals(1.0), mu);
  Value scaled = buildLoop<md::MapParticlesOp>(
      b, loc, x.getType(), {x},
      [&](Emitter &e, ValueRange args) { return e.mul(mu, args[0]); });
  for (Value group : op.getGroups()) {
    unsigned arity = cast<md::RelationType>(group.getType()).getArity();
    unsigned count = arity - 1;
    Operation *change = buildTupleOp(
        b, loc, md::GatherTuplesOp::getOperationName(), x.getType(), group,
        x, op.getCell(), count, {op.getMasses()},
        [&](Emitter &e, ValueRange args) -> SmallVector<Value> {
          ValueRange places = args.take_front(count);
          ValueRange masses = args.slice(count, arity);
          Value total = masses[0];
          Value moment = e.reals(0.0);
          for (unsigned j = 1; j <= count; ++j) {
            total = e.add(total, masses[j]);
            moment = e.add(moment, scaleBy(e, masses[j], places[j - 1]));
          }
          Value one = e.real(1.0);
          Value center = scaleBy(e, e.div(one, total), moment);
          SmallVector<Value> yielded;
          for (unsigned j = 0; j <= count; ++j) {
            Value place = j == 0 ? Value(arith::NegFOp::create(e.b, e.loc,
                                                               center))
                                 : e.sub(places[j - 1], center);
            yielded.push_back(e.mul(oneMinusMu, place));
          }
          return yielded;
        });
    scaled = buildLoop<md::MapParticlesOp>(
        b, loc, x.getType(), {scaled, change->getResult(0)},
        [](Emitter &e, ValueRange args) { return e.add(args[0], args[1]); });
  }
  op.replaceAllUsesWith(scaled);
  op.erase();
}

/// ΔE_sys of a scaling of the cell, as emitCoupling counts it: of the
/// velocities exactly, Σ (μ⁻² − 1) K, and of the positions by the count
/// (docs/dynamics.md, Section 2.2.7).
void ExpandCouplings::expandScalingWork(dyn::ScalingWorkOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Emitter e{b, loc};
  StringRef count = op.getCount();
  Value constant = op.getConstant();
  ValueRange volumes = op.getVolumes();
  Value third = e.real(1.0 / 3.0);
  // C / (3 V) on each axis, as emitStrain computes it for the pressure.
  auto constantShare = [&](Value volume) {
    Value share = e.div(constant, volume);
    return e.splat(e.mul(share, third));
  };
  // Σ (μ⁻² − 1) K, from μ⁻¹.
  auto velocities = [&](Value inverse) {
    Value squared = e.mul(inverse, inverse);
    Value less = e.sub(squared, e.reals(1.0));
    return e.sum3(e.mul(less, op.getKinetic()));
  };
  Value result;
  if (count == "trotter" || count == "trotter_before") {
    // The scaling was made within the drift of the step (Trotter type,
    // [Bernetti2020], SI Sec. V.C): −Σ ln μ_a W_aa, W the mean of the
    // virials before and after the scaling with those of the constant
    // terms, or that before alone (D92, D116).
    Value logMu = op.getLogStrain();
    Value before = e.add(op.getVirialBefore(), constantShare(volumes[0]));
    Value mu = math::ExpOp::create(b, loc, logMu);
    Value inverse = e.div(e.reals(1.0), mu);
    Value work;
    if (count == "trotter") {
      Value after = e.add(op.getVirialAfter(), constantShare(volumes[1]));
      Value both = e.add(before, after);
      Value mean = e.mul(both, e.reals(0.5));
      work = e.sum3(e.mul(mean, logMu));
    } else {
      work = e.sum3(e.mul(before, logMu));
    }
    result = e.sub(velocities(inverse), work);
  } else {
    Value mu = op.getStrain();
    Value inverse = e.div(e.reals(1.0), mu);
    if (count == "first_order") {
      // −Σ (μ_a − 1) ⅓ (tr W + C / V), isotropic only.
      Value oneMinus = findOneMinus(op, mu);
      if (!oneMinus)
        oneMinus = e.sub(e.reals(1.0), mu);
      Value trace = e.sum3(op.getVirialBefore());
      Value share = e.div(constant, volumes[0]);
      Value total = e.add(trace, share);
      Value split = e.splat(e.mul(total, third));
      Value work = e.sum3(e.mul(oneMinus, split));
      result = e.add(work, velocities(inverse));
    } else {
      // U' − U + C (1/V' − 1/V), counted exactly in the new cell (D77).
      Value kinetic = velocities(inverse);
      ValueRange energies = op.getEnergies();
      Value one = e.real(1.0);
      Value newInverse = e.div(one, volumes[1]);
      Value oldInverse = e.div(one, volumes[0]);
      Value change = e.mul(constant, e.sub(newInverse, oldInverse));
      Value potential = e.add(e.sub(energies[1], energies[0]), change);
      result = e.add(potential, kinetic);
    }
  }
  op.replaceAllUsesWith(result);
  op.erase();
}

/// Replaces each op of pure arithmetic of `block` after `before` (or from
/// its start) up to `after` (or its end) with an equal op that comes
/// before it, in its block or before the ops that hold it, if there is
/// one. The expansion of an op of the
/// barostat computes again what the driver's text computes once and uses
/// in several places, the volume, C / (3 V), 1 - mu, or 1 / mu; a value
/// whose other uses an op has taken over would otherwise be removed as
/// dead before the cse of the pipeline merged the two, and computed at the
/// place of the op instead of its own.
/// Returns true if the ops `a` and `b` compute the same value: the same op
/// with the same attributes on operands that are the same values, as the
/// canonicalization and the cse of the pipeline find them. Two operands are
/// the same if they are one value, or defined by pure ops without regions
/// that compute the same value, up to `depth` ops deep; constants by their
/// values.
static bool isSameValue(Operation *a, Operation *b, unsigned depth = 8) {
  if (a == b)
    return true;
  if (a->getName() != b->getName() || a->getNumRegions() != 0 ||
      b->getNumRegions() != 0 || !isMemoryEffectFree(a) ||
      !isMemoryEffectFree(b))
    return false;
  return OperationEquivalence::isEquivalentTo(
      a, b,
      [&](Value x, Value y) {
        if (x == y)
          return success();
        auto left = dyn_cast<OpResult>(x), right = dyn_cast<OpResult>(y);
        return success(depth != 0 && left && right &&
                       left.getResultNumber() == right.getResultNumber() &&
                       isSameValue(left.getOwner(), right.getOwner(),
                                   depth - 1));
      },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

static void mergeWithBefore(Block *block, Operation *before,
                            Operation *after) {
  Operation *op = before ? before->getNextNode()
                         : (block->empty() ? nullptr : &block->front());
  while (op && op != after) {
    Operation *next = op->getNextNode();
    if (op->getNumRegions() == 0 && op->getNumResults() != 0 &&
        isMemoryEffectFree(op)) {
      // The first equal op among those before it in its block and before
      // the ops that hold the block, up to the function, each of which
      // dominates it: the one that the canonicalization and the cse of the
      // pipeline would keep, constants being equal by their values.
      Operation *first = nullptr;
      for (Operation *at = op; at;) {
        for (Operation *other = at->getPrevNode(); other;
             other = other->getPrevNode())
          if (isSameValue(op, other))
            first = other;
        Operation *parent = at->getParentOp();
        if (!parent || parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
          break;
        at = parent;
      }
      if (first) {
        op->replaceAllUsesWith(first);
        op->erase();
      }
    }
    op = next;
  }
}

/// The state of the pressure of the step before (D92), as emitLevel
/// loads it: the nine values, then the three vectors.
void ExpandCouplings::expandLoadPressureState(dyn::LoadPressureStateOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value loaded[9];
  for (int slot = 0; slot != 9; ++slot) {
    Value index = arith::ConstantIndexOp::create(b, loc, slot);
    loaded[slot] = memref::LoadOp::create(b, loc, op.getMemory(), index);
  }
  Emitter e{b, loc};
  for (int v = 0; v != 3; ++v) {
    Value vector = vector::FromElementsOp::create(
        b, loc, e.vector3(),
        ValueRange{loaded[3 * v], loaded[3 * v + 1], loaded[3 * v + 2]});
    op->getResult(v).replaceAllUsesWith(vector);
  }
  op.erase();
}

/// The state of the pressure for the next step, and for the checkpoint
/// (emitStoreTrotterState).
LogicalResult
ExpandCouplings::expandStorePressureState(dyn::StorePressureStateOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value vectors[] = {op.getVirial(), op.getGroups(), op.getKinetic()};
  SmallVector<Value> values;
  for (int v = 0; v != 3; ++v)
    for (int k = 0; k != 3; ++k) {
      int slot = 3 * v + k;
      values.push_back(vector::ExtractOp::create(b, loc, vectors[v], k));
      Value index = arith::ConstantIndexOp::create(b, loc, slot);
      memref::StoreOp::create(b, loc, values.back(), op.getMemory(), index);
    }
  if (failed(callHost(op, "mdrtSetBarostatState", values, {},
                      /*cInterface=*/true)))
    return failure();
  op.erase();
  return success();
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

  // The barostat (step 2b).
  SmallVector<Operation *> barostat;
  module.walk([&](Operation *op) {
    if (isa<dyn::PressureOp, dyn::CellRescalingOp, dyn::LoadCellOp,
            dyn::ScaleCellOp, dyn::StoreCellOp, dyn::GroupKineticOp,
            dyn::ScalePositionsOp, dyn::ScalingWorkOp,
            dyn::LoadPressureStateOp, dyn::StorePressureStateOp>(op))
      barostat.push_back(op);
  });
  llvm::SmallPtrSet<Operation *, 8> fused;
  for (Operation *op : barostat) {
    if (fused.contains(op))
      continue;
    // The ops that the expansion writes are those between `before` and
    // `after`; a value that the block computes before them is taken
    // again (mergeWithBefore).
    Block *block = op->getBlock();
    Operation *before = op->getPrevNode();
    Operation *last = op;
    if (auto scale = dyn_cast<dyn::ScaleCellOp>(op))
      if (auto store =
              dyn_cast_or_null<dyn::StoreCellOp>(scale->getNextNode()))
        if (store.getCell() == ValueRange(scale.getResults()))
          last = store;
    Operation *after = last->getNextNode();
    auto merge = llvm::make_scope_exit(
        [&] { mergeWithBefore(block, before, after); });
    if (auto pressure = dyn_cast<dyn::PressureOp>(op)) {
      expandPressure(pressure);
    } else if (auto rescaling = dyn_cast<dyn::CellRescalingOp>(op)) {
      if (failed(expandCellRescaling(rescaling)))
        return signalPassFailure();
    } else if (auto load = dyn_cast<dyn::LoadCellOp>(op)) {
      expandLoadCell(load);
    } else if (auto groups = dyn_cast<dyn::GroupKineticOp>(op)) {
      expandGroupKinetic(groups);
    } else if (auto positions = dyn_cast<dyn::ScalePositionsOp>(op)) {
      expandScalePositions(positions);
    } else if (auto work = dyn_cast<dyn::ScalingWorkOp>(op)) {
      expandScalingWork(work);
    } else if (auto load = dyn_cast<dyn::LoadPressureStateOp>(op)) {
      expandLoadPressureState(load);
    } else if (auto store = dyn_cast<dyn::StorePressureStateOp>(op)) {
      if (failed(expandStorePressureState(store)))
        return signalPassFailure();
    } else if (auto scale = dyn_cast<dyn::ScaleCellOp>(op)) {
      auto store = dyn_cast_or_null<dyn::StoreCellOp>(scale->getNextNode());
      if (store && store.getCell() != ValueRange(scale.getResults()))
        store = {};
      if (store)
        fused.insert(store);
      if (failed(expandCell(scale, store)))
        return signalPassFailure();
    } else if (failed(expandStoreCell(cast<dyn::StoreCellOp>(op)))) {
      return signalPassFailure();
    }
  }

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
