#include "mdir/Conversion/TemplateWriterKernels.h"
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/TemplateValidityKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir::kernels;

void mdir::kernels::prepareTemplateWriterConstants(OpBuilder &b, Location loc,
    TemplateWriterBuffers &v) {
  auto narrow = [&](Value x) -> Value {
    return arith::TruncFOp::create(b, loc,
        VectorType::get(cast<VectorType>(x.getType()).getShape(), b.getF32Type()), x);
  };
  v.narrowCell = narrow(v.cell);
  v.narrowInverse = narrow(v.inverse);
  Value offset = vector::BroadcastOp::create(b, loc,
      VectorType::get({3}, b.getF64Type()), createReal(b, loc, b.getF64Type(), v.pointOffset));
  v.narrowOffset = narrow(emitLatticeShift(b, loc, offset, v.cell));
}

TemplateWriterFlags mdir::kernels::emitUpdateTemplateFrame(OpBuilder &b,
    Location loc, const TemplateWriterBuffers &v, Value unit,
    ArrayRef<TemplateParticleResult> results) {
  assert((results.size() == 1 || results.size() == 3) && "whole unit required");
  for (const auto &result : results)
    assert(v.positionOutput < result.outputs.size());
  assert(v.narrowCell && "prepareTemplateWriterConstants before the launch");
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
    std::array<Value, 2> local;
    if (water) {
      local = emitTemplateLocalGeometry(at, loc, positions, v.cell, v.inverse,
                                        v.narrowCell, v.narrowInverse);
      Value squared = mdir::md::emitTemplateExtentSquared(at, loc, v.alpha, v.hasSite,
                                                          local[0], local[1]);
      // The f32 square has a relative error below 11 * 2^-24 (the narrowed
      // displacements, alpha, the sum and the squares); a bound lowered by
      // 2^-20 keeps every true excess flagged.
      Value bound = createReal(at, loc, at.getF32Type(),
          float(v.extent * v.extent * (1 - 0x1p-20)));
      extentBad = arith::AndIOp::create(at, loc, v.constrained,
          arith::CmpFOp::create(at, loc, arith::CmpFPredicate::UGT, squared, bound));
    }
    auto cached = scf::IfOp::create(at, loc, v.built, [&](OpBuilder &in, Location) {
      Value index = arith::IndexCastOp::create(in, loc, in.getIndexType(), unit);
      Value cells = loadElement(in, loc, v.packingCells, index);
      Value framed = arith::AddFOp::create(in, loc, positions[0],
          emitLatticeShift(in, loc, cells, v.cell));
      Value anchor = convertReal(in, loc, framed, in.getF32Type());
      if (water) {
        Value row = arith::MulIOp::create(in, loc, index, ix(in, 3));
        storeElement(in, loc, anchor, v.frames, row);
        for (int site = 1; site != 3; ++site)
          storeElement(in, loc, arith::AddFOp::create(in, loc, anchor, local[site - 1]),
              v.frames, arith::AddIOp::create(in, loc, row, ix(in, site)));
      } else {
        Value row = arith::AddIOp::create(in, loc, index,
            arith::MulIOp::create(in, loc, v.waters, ix(in, 2)));
        storeElement(in, loc, anchor, v.frames, row);
      }
      unsigned group = water ? 0 : 1;
      auto test = [&](OpBuilder &at) -> std::array<Value, 2> {
        Value outer = emitTemplateMotionExceededNarrow(at, loc, anchor,
            loadElement(at, loc, v.outerReference, index), v.narrowCell,
            v.narrowOffset, v.outerLimits[group]);
        Value inner = emitTemplateMotionExceededNarrow(at, loc, anchor,
            loadElement(at, loc, v.innerReference, index), v.narrowCell,
            v.narrowOffset, v.innerLimits[group]);
        return {outer, inner};
      };
      if (water || !v.atomAnchors) {
        auto flags = test(in);
        scf::YieldOp::create(in, loc, ValueRange{flags[0], flags[1]});
      } else {
        Value atom = arith::SubIOp::create(in, loc, index, v.waters);
        Value owner = memref::LoadOp::create(in, loc, v.atomAnchors, ValueRange{atom});
        Value own = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, owner, unit);
        auto tested = scf::IfOp::create(in, loc, own, [&](OpBuilder &at, Location) {
          auto flags = test(at);
          scf::YieldOp::create(at, loc, ValueRange{flags[0], flags[1]});
        }, [&](OpBuilder &at, Location) {
          scf::YieldOp::create(at, loc, ValueRange{no, no});
        });
        scf::YieldOp::create(in, loc, tested.getResults());
      }
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

void mdir::kernels::emitTemplateCurrentPoint(OpBuilder &b, Location loc,
    const TemplatePointBuffers &v, Value unit) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  Value owner = unit;
  if (v.atomAnchors) {
    Value water = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, unit, v.waters);
    Value atom = arith::SelectOp::create(b, loc, water, ix(b, 0),
        arith::SubIOp::create(b, loc, unit, v.waters));
    Value anchor = arith::IndexCastOp::create(b, loc, b.getIndexType(),
        memref::LoadOp::create(b, loc, v.atomAnchors, ValueRange{atom}));
    owner = arith::SelectOp::create(b, loc, water, unit, anchor);
  }
  // The anchor's row: a water's O, or the solute unit's own site.
  Value ownerWater = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, owner, v.waters);
  Value row = arith::SelectOp::create(b, loc, ownerWater,
      arith::MulIOp::create(b, loc, owner, ix(b, 3)),
      arith::AddIOp::create(b, loc, owner, arith::MulIOp::create(b, loc, v.waters, ix(b, 2))));
  Value particle = row;
  if (v.members)
    particle = arith::IndexCastOp::create(b, loc, b.getIndexType(),
        memref::LoadOp::create(b, loc, v.members, ValueRange{row}));
  Value framed = arith::AddFOp::create(b, loc, loadElement(b, loc, v.positions, particle),
      emitLatticeShift(b, loc, loadElement(b, loc, v.packingCells, owner), v.cell));
  Value fraction = emitTemplateFractional(b, loc, framed, v.cell, v.inverse);
  if (v.pointOffset != 0.0)
    fraction = arith::AddFOp::create(b, loc, fraction,
        vector::BroadcastOp::create(b, loc, fraction.getType(),
            createReal(b, loc, b.getF64Type(), v.pointOffset)));
  storeElement(b, loc, fraction, v.currentPoints, unit);
}
