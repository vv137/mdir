#include "HostPME.h"
#include "PMEFFTs.h"

using namespace mdir::kernels;

// Lift a per-particle function over all particles or over the D262 list.
// Cell and window arithmetic belong to the selected particle function.
template <HostPME::ParticleStage stage, HostPME::Cell cell, bool listed>
void HostPME::emitParticles() {
  constexpr bool spread = stage == ParticleStage::Spread;
  StringRef stageName = spread                           ? "spread"
                        : stage == ParticleStage::Gather ? "gather"
                                                         : "potential";
  StringRef suffix = cell == Cell::Triclinic ? "_triclinic" : "";
  std::string prefix = ("mdrt.pme_" + stageName).str();
  auto particle =
      module.lookupSymbol<func::FuncOp>(prefix + "_one" + suffix.str());
  SmallVector<Type> inputs(particle.getArgumentTypes());
  // Spread takes the particle index last. Gathering takes the moduli last.
  unsigned particleIndex = inputs.size() - (spread ? 1 : 2);
  inputs.erase(inputs.begin() + particleIndex);
  constexpr unsigned listPosition = 2;
  if constexpr (listed)
    inputs.insert(inputs.begin() + listPosition,
                  MemRefType::get({ShapedType::kDynamic}, i32Type));
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(prefix + (listed ? "_listed" : "") + suffix.str(),
                    b.getFunctionType(inputs, {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  SmallVector<Value> args(entry->getArguments());
  Value list;
  if constexpr (listed) {
    list = args[listPosition];
    args.erase(args.begin() + listPosition);
  }
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value count;
  if constexpr (spread) {
    func::CallOp::create(b, loc, "mdrt.pme_zero", TypeRange{},
                         ValueRange{args[3]});
  } else if constexpr (listed) {
    Value axes = arith::ConstantIndexOp::create(b, loc, 3);
    count = memref::DimOp::create(b, loc, args.front(), zero);
    Value wide =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
    Value none = convert(realType, wide);
    scf::ParallelOp::create(
        b, loc, ValueRange{zero}, ValueRange{count}, ValueRange{one},
        [&](OpBuilder &b, Location, ValueRange iv) {
          scf::ForOp::create(
              b, loc, zero, axes, one, ValueRange{},
              [&](OpBuilder &b, Location, Value axis, ValueRange) {
                store(none, args[8], {iv.front(), axis});
                scf::YieldOp::create(b, loc);
              });
        });
  }
  if constexpr (listed)
    count = func::CallOp::create(b, loc, "mdrt.pme_count_listed",
                                 TypeRange{idxType}, ValueRange{list})
                .getResult(0);
  else
    count = memref::DimOp::create(b, loc, args.front(), zero);
  scf::ParallelOp::create(
      b, loc, ValueRange{zero}, ValueRange{count}, ValueRange{one},
      [&](OpBuilder &b, Location, ValueRange iv) {
        Value index = iv.front();
        if constexpr (listed) {
          Value position = load(list, {index});
          index = arith::IndexCastOp::create(b, loc, idxType, position);
        }
        args.insert(args.begin() + particleIndex, index);
        func::CallOp::create(b, loc, particle, args);
      });
  func::ReturnOp::create(b, loc);
}

void HostPME::emit(bool potential, bool listed) {
  declare("mdrtFFTPrepare", b.getFunctionType({i64Type, i64Type, i64Type}, {}));
  declare("mdrtFFTColumnBlocks",
          b.getFunctionType({i64Type, i64Type}, {i64Type}));
  declare(hostFFTName(realType, "RowsForward"),
          b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, realType),
                             MemRefType::get({ShapedType::kDynamic}, realType),
                             i64Type, i64Type, i64Type},
                            {}))
      ->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  declare(hostFFTName(realType, "RowsBackward"),
          b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, realType),
                             MemRefType::get({ShapedType::kDynamic}, realType),
                             i64Type, i64Type, i64Type},
                            {}))
      ->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  declare(hostFFTName(realType, "Columns"),
          b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, realType),
                             i64Type, i64Type, i64Type, i64Type, i64Type,
                             i64Type, i64Type},
                            {}))
      ->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  emitRaise();
  emitBspline();
  emitWindow();
  emitClenshaw();
  emitSpline();
  emitPlace();
  emitInverseCell();
  emitZero();
  emitSpreadPoint();
  emitSpreadOne();
  emitSpreadOneTriclinic();
  emitParticles<ParticleStage::Spread, Cell::Orthorhombic>();
  emitParticles<ParticleStage::Spread, Cell::Triclinic>();
  emitReal();
  declare("mdrt.pme_fft_forward",
          b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, realType),
                             MemRefType::get({ShapedType::kDynamic}, realType),
                             idxType, idxType, idxType},
                            {}));
  declare("mdrt.pme_fft_backward",
          b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, realType),
                             MemRefType::get({ShapedType::kDynamic}, realType),
                             idxType, idxType, idxType},
                            {}));
  emitWave();
  emitTables();
  emitInfluence();
  emitPointSums();
  emitPointSumsP3m();
  emitConvolveRow();
  emitConvolveRowTriclinic();
  emitConvolve<Cell::Orthorhombic, true>();
  emitConvolve<Cell::Orthorhombic, false>();
  emitConvolve<Cell::Triclinic, true>();
  emitConvolve<Cell::Triclinic, false>();
  emitGatherSums();
  if (potential)
    emitPotentialSum();
  emitGatherOne();
  emitParticles<ParticleStage::Gather, Cell::Orthorhombic>();
  if (potential)
    emitPotentialOne();
  if (potential)
    emitParticles<ParticleStage::Potential, Cell::Orthorhombic>();
  emitSplinesTriclinic();
  emitGatherOneTriclinic();
  emitParticles<ParticleStage::Gather, Cell::Triclinic>();
  if (potential)
    emitPotentialOneTriclinic();
  if (potential)
    emitParticles<ParticleStage::Potential, Cell::Triclinic>();
  if (listed)
    emitCountListed();
  if (listed)
    emitParticles<ParticleStage::Spread, Cell::Orthorhombic, true>();
  if (listed)
    emitParticles<ParticleStage::Gather, Cell::Orthorhombic, true>();
  if (listed)
    emitParticles<ParticleStage::Spread, Cell::Triclinic, true>();
  if (listed)
    emitParticles<ParticleStage::Gather, Cell::Triclinic, true>();
  emitHostPMEFFTs(module);
}
