#include "mdir/Conversion/TemplateWriterKernels.h"
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/TemplateValidityKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir::kernels;

TemplateWriterFlags mdir::kernels::emitUpdateTemplateFrame(OpBuilder &b,
    Location loc, const TemplateWriterBuffers &v, Value unit,
    ArrayRef<TemplateParticleResult> results) {
  assert((results.size() == 1 || results.size() == 3) && "whole unit required");
  for (const auto &result : results)
    assert(v.positionOutput < result.outputs.size());
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  Value no = arith::ConstantOp::create(b, loc, b.getBoolAttr(false));
  Value real = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, unit,
      arith::ConstantIntOp::create(b, loc, 0, 32));
  bool water = results.size() == 3;
  // Even an unbuilt fresh list must enforce the historical constraint
  // extent. Its final coordinates are available without touching epochs.
  auto active = scf::IfOp::create(b, loc, real, [&](OpBuilder &at, Location) {
    Value extentBad = no;
    SmallVector<Value, 3> positions;
    for (const auto &result : results) positions.push_back(result.outputs[v.positionOutput]);
    if (water) {
      Value zero = createZero(at, loc, positions[0].getType());
      auto whole = emitRetainedTemplateFrame(at, loc, positions, v.cell, v.inverse,
                                               zero, at.getF32Type());
      Value h1 = arith::SubFOp::create(at, loc, whole.whole[1], whole.whole[0]);
      Value h2 = arith::SubFOp::create(at, loc, whole.whole[2], whole.whole[0]);
      Value squared = mdir::md::emitTemplateExtentSquared(at, loc, v.alpha, v.hasSite, h1, h2);
      Value bound = createReal(at, loc, at.getF64Type(), v.extent * v.extent);
      extentBad = arith::AndIOp::create(at, loc, v.constrained,
          arith::CmpFOp::create(at, loc, arith::CmpFPredicate::UGT, squared, bound));
    }
    auto cached = scf::IfOp::create(at, loc, v.built, [&](OpBuilder &in, Location) {
      Value index = arith::IndexCastOp::create(in, loc, in.getIndexType(), unit);
      Value cells = loadElement(in, loc, v.packingCells, index);
      Value framed = arith::AddFOp::create(in, loc, positions[0],
          emitLatticeShift(in, loc, cells, v.cell));
      Value row;
      if (water) {
        row = arith::MulIOp::create(in, loc, index, ix(in, 3));
        auto frame = emitRetainedTemplateFrame(in, loc, positions, v.cell, v.inverse,
                                                cells, in.getF32Type());
        for (int site = 0; site != 3; ++site)
          storeElement(in, loc, frame.packed[site], v.frames,
              arith::AddIOp::create(in, loc, row, ix(in, site)));
      } else {
        row = arith::AddIOp::create(in, loc, index,
            arith::MulIOp::create(in, loc, v.waters, ix(in, 2)));
        storeElement(in, loc, convertReal(in, loc, framed, in.getF32Type()), v.frames, row);
      }
      Value fraction = emitTemplateFractional(in, loc, framed, v.cell, v.inverse);
      storeElement(in, loc, fraction, v.currentPoints, index);
      unsigned group = water ? 0 : 1;
      Value outer = emitTemplateMotionExceeded(in, loc, framed,
          loadElement(in, loc, v.outerReference, index), v.cell, v.outerLimits[group]);
      Value inner = emitTemplateMotionExceeded(in, loc, framed,
          loadElement(in, loc, v.innerReference, index), v.cell, v.innerLimits[group]);
      scf::YieldOp::create(in, loc, ValueRange{outer, inner});
    }, [&](OpBuilder &in, Location) {
      scf::YieldOp::create(in, loc, ValueRange{no, no});
    });
    scf::YieldOp::create(at, loc, ValueRange{cached.getResult(0), cached.getResult(1), extentBad});
  }, [&](OpBuilder &at, Location) {
    scf::YieldOp::create(at, loc, ValueRange{no, no, no});
  });
  return {water ? active.getResult(0) : no, water ? no : active.getResult(0),
          water ? active.getResult(1) : no, water ? no : active.getResult(1),
          active.getResult(2)};
}
