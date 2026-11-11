#include "mdir/Conversion/TemplateFrameKernels.h"
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir::kernels;

void mdir::kernels::emitSnapshotTemplateFrames(OpBuilder &b, Location loc,
    const TemplateFrameBuffers &v, Value waterActive, Value atomActive) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  Value one = ix(b, 1), block = ix(b, 256);
  auto clear = gpu::LaunchOp::create(b, loc, one, one, one, one, one, one);
  {
    OpBuilder at = OpBuilder::atBlockEnd(&clear.getBody().front());
    Value zero = arith::ConstantIntOp::create(at, loc, 0, 64);
    // The cold owner reads this whole four-word control packet before
    // search. Initialize every copied byte, including its not-yet-used
    // chunk/flag/entry fields; search will reset those fields again.
    for (int64_t slot = 0; slot != 4; ++slot)
      memref::StoreOp::create(at, loc, zero, v.counters, ValueRange{ix(at, slot)});
    gpu::TerminatorOp::create(at, loc);
  }
  Value units = arith::AddIOp::create(b, loc, v.waters, v.atoms);
  Value grid = arith::MaxUIOp::create(b, loc, one,
      arith::CeilDivUIOp::create(b, loc, units, block));
  auto launch = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
  OpBuilder k = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value unit = arith::AddIOp::create(k, loc, launch.getThreadIds().x,
      arith::MulIOp::create(k, loc, launch.getBlockIds().x, block));
  Value valid = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, unit, units);
  Value water = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, unit, v.waters);
  Value active = arith::AndIOp::create(k, loc, valid,
      arith::SelectOp::create(k, loc, water, waterActive, atomActive));
  scf::IfOp::create(k, loc, active, [&](OpBuilder &at, Location) {
    // D107: count particles, not units. In this snapshot phase the flags
    // word is a bad-position count; the host consumes it before search
    // initializes that word for its own flags. No arena ABI changes.
    auto validPositions = [&](OpBuilder &in, ValueRange positions) -> Value {
      Value valid = arith::ConstantIntOp::create(in, loc, 1, 1);
      Value count = arith::ConstantIntOp::create(in, loc, 0, 64);
      Value limit = createReal(in, loc, in.getF64Type(), 1.e100);
      for (Value position : positions) {
        Value magnitude = math::AbsFOp::create(in, loc,
            vector::ExtractOp::create(in, loc, position, 0));
        for (unsigned axis = 1; axis != 3; ++axis)
          magnitude = arith::MaximumFOp::create(in, loc, magnitude,
              math::AbsFOp::create(in, loc,
                  vector::ExtractOp::create(in, loc, position, axis)));
        Value good = arith::CmpFOp::create(in, loc, arith::CmpFPredicate::OLE,
            magnitude, limit);
        valid = arith::AndIOp::create(in, loc, valid, good);
        Value bad = arith::CmpFOp::create(in, loc, arith::CmpFPredicate::UGT,
            magnitude, limit);
        count = arith::AddIOp::create(in, loc, count,
            arith::ExtUIOp::create(in, loc, in.getI64Type(), bad));
      }
      memref::AtomicRMWOp::create(in, loc, arith::AtomicRMWKind::addi,
          count, v.counters, ValueRange{ix(in, TEMPLATE_COUNTER_FLAGS)});
      return valid;
    };
    auto member = [&](OpBuilder &in, Value row) -> Value {
      Value id = memref::LoadOp::create(in, loc, v.members, ValueRange{row});
      return arith::IndexCastOp::create(in, loc, in.getIndexType(), id);
    };
    auto publish = [&](OpBuilder &in, Value anchor, Value cells) {
      Value framed = arith::AddFOp::create(in, loc, anchor,
          emitLatticeShift(in, loc, cells, v.cell));
      Value fractional = emitTemplateFractional(in, loc, framed, v.cell, v.inverse);
      storeElement(in, loc, cells, v.packingCells, unit);
      storeElement(in, loc, fractional, v.outerReference, unit);
      storeElement(in, loc, fractional, v.currentPoints, unit);
    };
    scf::IfOp::create(at, loc, water, [&](OpBuilder &in, Location) {
      Value row = arith::MulIOp::create(in, loc, unit, ix(in, 3));
      SmallVector<Value, 3> positions;
      for (int site = 0; site != 3; ++site)
        positions.push_back(loadElement(in, loc, v.positions,
            member(in, arith::AddIOp::create(in, loc, row, ix(in, site)))));
      scf::IfOp::create(in, loc, validPositions(in, positions), [&](OpBuilder &in, Location) {
        Value cells = emitTemplatePackingCells(in, loc, positions[0], v.cell, v.inverse);
        auto frame = emitRetainedTemplateFrame(in, loc, positions, v.cell, v.inverse,
                                                cells, in.getF32Type());
        for (int site = 0; site != 3; ++site)
          storeElement(in, loc, frame.packed[site], v.frames,
              arith::AddIOp::create(in, loc, row, ix(in, site)));
        publish(in, positions[0], cells);
        Value h1 = arith::SubFOp::create(in, loc, frame.whole[1], frame.whole[0]);
        Value h2 = arith::SubFOp::create(in, loc, frame.whole[2], frame.whole[0]);
        Value extent = mdir::md::emitTemplateExtentSquared(in, loc, v.alpha, v.hasSite, h1, h2);
        Value bits = arith::BitcastOp::create(in, loc, in.getI64Type(), extent);
        memref::AtomicRMWOp::create(in, loc, arith::AtomicRMWKind::maxu, bits, v.counters,
                                    ValueRange{ix(in, TEMPLATE_COUNTER_EXTENT)});
        scf::YieldOp::create(in, loc);
      });
      scf::YieldOp::create(in, loc);
    }, [&](OpBuilder &in, Location) {
      Value row = arith::AddIOp::create(in, loc, unit,
          arith::MulIOp::create(in, loc, v.waters, ix(in, 2)));
      Value position = loadElement(in, loc, v.positions, member(in, row));
      Value valid = validPositions(in, ValueRange{position});
      if (!v.atomAnchors) {
        scf::IfOp::create(in, loc, valid, [&](OpBuilder &in, Location) {
          Value cells = emitTemplatePackingCells(in, loc, position, v.cell, v.inverse);
          Value framed = arith::AddFOp::create(in, loc, position,
              emitLatticeShift(in, loc, cells, v.cell));
          storeElement(in, loc, convertReal(in, loc, framed, in.getF32Type()), v.frames, row);
          publish(in, position, cells);
          scf::YieldOp::create(in, loc);
        });
        scf::YieldOp::create(in, loc);
        return;
      }
      // The anchor's position is final here (no thread of this launch
      // writes positions); its packing cell is the same pure function of
      // it that the anchor's own thread evaluates.
      Value anchorUnit = arith::IndexCastOp::create(in, loc, in.getIndexType(),
          memref::LoadOp::create(in, loc, v.atomAnchors,
              ValueRange{arith::SubIOp::create(in, loc, unit, v.waters)}));
      Value anchorRow = arith::AddIOp::create(in, loc, anchorUnit,
          arith::MulIOp::create(in, loc, v.waters, ix(in, 2)));
      Value anchor = loadElement(in, loc, v.positions, member(in, anchorRow));
      Value huge = createReal(in, loc, in.getF64Type(), 1.e100);
      Value anchorValid = arith::CmpFOp::create(in, loc, arith::CmpFPredicate::OLE,
          vector::ReductionOp::create(in, loc, vector::CombiningKind::MAXNUMF,
              math::AbsFOp::create(in, loc, anchor)), huge);
      scf::IfOp::create(in, loc, arith::AndIOp::create(in, loc, valid, anchorValid),
          [&](OpBuilder &in, Location) {
        Value anchorCells = emitTemplatePackingCells(in, loc, anchor, v.cell, v.inverse);
        Value bond = arith::SubFOp::create(in, loc, position, anchor);
        Value near = emitMinimumImage(in, loc, bond, v.cell, v.inverse);
        // near - bond is a lattice vector: shift the member by it as well.
        Value cells = arith::AddFOp::create(in, loc, anchorCells,
            math::RoundEvenOp::create(in, loc, emitTemplateFractional(in, loc,
                arith::SubFOp::create(in, loc, near, bond), v.cell, v.inverse)));
        Value framed = arith::AddFOp::create(in, loc, position,
            emitLatticeShift(in, loc, cells, v.cell));
        storeElement(in, loc, convertReal(in, loc, framed, in.getF32Type()), v.frames, row);
        publish(in, anchor, cells);
        Value extent = vector::ReductionOp::create(in, loc, vector::CombiningKind::ADD,
            arith::MulFOp::create(in, loc, near, near));
        memref::AtomicRMWOp::create(in, loc, arith::AtomicRMWKind::maxu,
            arith::BitcastOp::create(in, loc, in.getI64Type(), extent), v.counters,
            ValueRange{ix(in, TEMPLATE_COUNTER_ANCHOR_EXTENT)});
        scf::YieldOp::create(in, loc);
      });
      scf::YieldOp::create(in, loc);
    });
    scf::YieldOp::create(at, loc);
  });
  gpu::TerminatorOp::create(k, loc);
}
