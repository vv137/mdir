#include "HostPME.h"
#include "PMEFFTs.h"

using namespace mdir::kernels;

void HostPME::emit(bool potential, bool listed) {
  declare("mdrtFFTPrepare", b.getFunctionType({i64Type, i64Type, i64Type}, {}));
  declare("mdrtFFTColumnBlocks",
          b.getFunctionType({i64Type, i64Type}, {i64Type}));
  declare("mdir.template.fft.RowsForward",
          b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, realType),
                             MemRefType::get({ShapedType::kDynamic}, realType),
                             i64Type, i64Type, i64Type},
                            {}))
      ->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  declare("mdir.template.fft.RowsBackward",
          b.getFunctionType({MemRefType::get({ShapedType::kDynamic}, realType),
                             MemRefType::get({ShapedType::kDynamic}, realType),
                             i64Type, i64Type, i64Type},
                            {}))
      ->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  declare("mdir.template.fft.Columns",
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
  emitSpread();
  emitSpreadTriclinic();
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
  emitConvolve();
  emitScale();
  emitConvolveTriclinic();
  emitScaleTriclinic();
  emitGatherSums();
  if (potential)
    emitPotentialSum();
  emitGatherOne();
  emitGather();
  if (potential)
    emitPotentialOne();
  if (potential)
    emitPotential();
  emitSplinesTriclinic();
  emitGatherOneTriclinic();
  emitGatherTriclinic();
  if (potential)
    emitPotentialOneTriclinic();
  if (potential)
    emitPotentialTriclinic();
  if (listed)
    emitCountListed();
  if (listed)
    emitSpreadListed();
  if (listed)
    emitGatherListed();
  if (listed)
    emitSpreadListedTriclinic();
  if (listed)
    emitGatherListedTriclinic();
  emitHostPMEFFTs(module);
}
