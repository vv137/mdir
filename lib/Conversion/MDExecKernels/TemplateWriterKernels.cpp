#include "mdir/Conversion/TemplateWriterKernels.h"
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/TemplateValidityKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

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
      unsigned group = water ? 0 : 1;
      auto publish = [&](OpBuilder &at) -> std::array<Value, 2> {
        Value fraction = emitTemplateFractional(at, loc, framed, v.cell, v.inverse);
        Value tested = framed;
        if (v.pointOffset != 0.0) {
          Value offset = vector::BroadcastOp::create(at, loc, fraction.getType(),
              createReal(at, loc, at.getF64Type(), v.pointOffset));
          fraction = arith::AddFOp::create(at, loc, fraction, offset);
          tested = arith::AddFOp::create(at, loc, framed, emitLatticeShift(at, loc, offset, v.cell));
        }
        storeElement(at, loc, fraction, v.currentPoints, index);
        if (!water && v.atomDependents) {
          // A member's point is its anchor's, in the member's own packing
          // frame: the two frames differ by whole lattice vectors.
          Value atom = arith::SubIOp::create(at, loc, index, v.waters);
          for (int64_t d = 0; d != MDRT_TEMPLATE_ANCHOR_DEPENDENTS; ++d) {
            Value slot = arith::AddIOp::create(at, loc,
                arith::MulIOp::create(at, loc, atom, ix(at, MDRT_TEMPLATE_ANCHOR_DEPENDENTS)),
                ix(at, d));
            Value dependent = memref::LoadOp::create(at, loc, v.atomDependents, ValueRange{slot});
            Value present = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::sge, dependent,
                arith::ConstantIntOp::create(at, loc, 0, 32));
            scf::IfOp::create(at, loc, present, [&](OpBuilder &member, Location) {
              Value place = arith::IndexCastOp::create(member, loc, member.getIndexType(), dependent);
              Value offset = arith::SubFOp::create(member, loc,
                  loadElement(member, loc, v.packingCells, place), cells);
              storeElement(member, loc, arith::AddFOp::create(member, loc, fraction, offset),
                           v.currentPoints, place);
              scf::YieldOp::create(member, loc);
            });
          }
        }
        Value outer = emitTemplateMotionExceeded(at, loc, tested,
            loadElement(at, loc, v.outerReference, index), v.cell, v.outerLimits[group]);
        Value inner = emitTemplateMotionExceeded(at, loc, tested,
            loadElement(at, loc, v.innerReference, index), v.cell, v.innerLimits[group]);
        return {outer, inner};
      };
      if (water || !v.atomAnchors) {
        auto flags = publish(in);
        scf::YieldOp::create(in, loc, ValueRange{flags[0], flags[1]});
      } else {
        Value atom = arith::SubIOp::create(in, loc, index, v.waters);
        Value anchor = memref::LoadOp::create(in, loc, v.atomAnchors, ValueRange{atom});
        Value own = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, anchor, unit);
        auto tested = scf::IfOp::create(in, loc, own, [&](OpBuilder &at, Location) {
          auto flags = publish(at);
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
