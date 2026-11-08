#include "mdir/Conversion/TemplateFrameKernels.h"
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
    memref::StoreOp::create(at, loc, zero, v.counters,
                            ValueRange{ix(at, TEMPLATE_COUNTER_EXTENT)});
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
    }, [&](OpBuilder &in, Location) {
      Value row = arith::AddIOp::create(in, loc, unit,
          arith::MulIOp::create(in, loc, v.waters, ix(in, 2)));
      Value position = loadElement(in, loc, v.positions, member(in, row));
      Value cells = emitTemplatePackingCells(in, loc, position, v.cell, v.inverse);
      Value framed = arith::AddFOp::create(in, loc, position,
          emitLatticeShift(in, loc, cells, v.cell));
      storeElement(in, loc, convertReal(in, loc, framed, in.getF32Type()), v.frames, row);
      publish(in, position, cells);
      scf::YieldOp::create(in, loc);
    });
    scf::YieldOp::create(at, loc);
  });
  gpu::TerminatorOp::create(k, loc);
}
