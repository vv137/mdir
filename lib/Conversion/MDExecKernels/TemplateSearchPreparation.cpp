// Auxiliary Cartesian caches for published hybrid reference groups.
#include "mdir/Conversion/TemplateSearchPreparation.h"
#include "mdir/Conversion/SpatialFrameEmitter.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <limits>
using namespace mlir;
using namespace mdir::kernels;
namespace {
Value asIndex(OpBuilder &b, Location loc, Value n) {
  return arith::IndexCastUIOp::create(b, loc, b.getIndexType(), n);
}
Value word(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIntOp::create(b, loc, n, 32);
}
Value load(OpBuilder &b, Location loc, Value buffer, Value index) {
  return memref::LoadOp::create(b, loc, buffer, ValueRange{index});
}
}
void mdir::kernels::emitPrepareTemplateSearch(OpBuilder &b, Location loc,
                            const TemplateSearchBuffers &v) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  Value one = ix(b, 1), block = ix(b, 256);
  Value count = arith::AddIOp::create(b, loc,
      v.build.groups.waterGroups, v.build.groups.atomGroups);
  Value grid = arith::MaxUIOp::create(b, loc, one,
      arith::DivUIOp::create(b, loc,
          arith::AddIOp::create(b, loc, count, ix(b, 7)), ix(b, 8)));
  auto launch = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
  OpBuilder at = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value thread = arith::AddIOp::create(at, loc,
      arith::MulIOp::create(at, loc, launch.getBlockIds().x, block), launch.getThreadIds().x);
  Value group = arith::DivUIOp::create(at, loc, thread, ix(at, 32));
  Value inside = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, group, count);
  scf::IfOp::create(at, loc, inside, [&](OpBuilder &in, Location) {
    Value lane = arith::RemUIOp::create(in, loc, thread, ix(in, 32));
    Value water = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult,
        group, v.build.groups.waterGroups);
    Value descriptor = arith::SelectOp::create(in, loc, water, group,
        arith::AddIOp::create(in, loc, v.spatial.waters,
            arith::SubIOp::create(in, loc, group, v.build.groups.waterGroups)));
    Value start = asIndex(in, loc, load(in, loc, v.build.groups.starts, descriptor));
    Value length = asIndex(in, loc, load(in, loc, v.build.groups.lengths, descriptor));
    Value local = arith::AndIOp::create(in, loc, lane, ix(in, 15));
    Value place = arith::AddIOp::create(in, loc, start,
        arith::MinUIOp::create(in, loc, local,
            arith::SubIOp::create(in, loc, length, ix(in, 1))));
    Value unit = load(in, loc, v.spatial.order, place);
    Value valid = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult, local, length);
    Value safe = arith::SelectOp::create(in, loc, valid, unit, word(in, loc, 0));
    Value point = loadElement(in, loc, v.spatial.points, asIndex(in, loc, safe));
    auto frame = emitSpatialReferenceFrame(in, loc, point);
    auto fractionalBox = emitSpatialWarpBox(in, loc, frame.wrapped, valid);
    // Keep the original f64 lattice transform before narrowing coordinates.
    Value cartesian = convertReal(in, loc,
        emitLatticeShift(in, loc, frame.wrapped, v.cell), in.getF32Type());
    Value infinity = vector::BroadcastOp::create(in, loc, cartesian.getType(),
        createReal(in, loc, in.getF32Type(), std::numeric_limits<float>::infinity()));
    Value minimum = arith::NegFOp::create(in, loc, emitWarpReduce(in, loc,
        arith::NegFOp::create(in, loc,
            arith::SelectOp::create(in, loc, valid, cartesian, infinity)), Combiner::Max));
    Value maximum = emitWarpReduce(in, loc,
        arith::SelectOp::create(in, loc, valid, cartesian,
            arith::NegFOp::create(in, loc, infinity)), Combiner::Max);
    SpatialBoundingBox box{minimum, maximum};
    // Only valid lower-half lanes write: upper-half lanes duplicate units.
    Value lower = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult,
        lane, ix(in, 16));
    scf::IfOp::create(in, loc, arith::AndIOp::create(in, loc, valid, lower),
        [&](OpBuilder &store, Location) {
      // The selector reads this initialized scalar before the candidate
      // control clear reuses it as the chunk counter. Include retained
      // hybrid references: a large origin must select the general fallback.
      Value maximumFrame = math::AbsFOp::create(store, loc,
          vector::ExtractOp::create(store, loc, frame.lattice, 0));
      for (int64_t axis = 1; axis != 3; ++axis)
        maximumFrame = arith::MaximumFOp::create(store, loc, maximumFrame,
            math::AbsFOp::create(store, loc,
                vector::ExtractOp::create(store, loc, frame.lattice, axis)));
      memref::AtomicRMWOp::create(store, loc, arith::AtomicRMWKind::maxu,
          arith::BitcastOp::create(store, loc, store.getI64Type(), maximumFrame),
          v.counters, ValueRange{ix(store, 0)});
      Value base = arith::MulIOp::create(store, loc, asIndex(store, loc, unit), ix(store, 4));
      for (int64_t axis = 0; axis != 3; ++axis) {
        Value bits = arith::BitcastOp::create(store, loc, store.getI32Type(),
            vector::ExtractOp::create(store, loc, cartesian, axis));
        memref::StoreOp::create(store, loc, bits, v.spatial.sortScratch,
            ValueRange{arith::AddIOp::create(store, loc, base, ix(store, axis))});
      }
      memref::StoreOp::create(store, loc, unit, v.spatial.sortScratch,
          ValueRange{arith::AddIOp::create(store, loc, base, ix(store, 3))});
      scf::YieldOp::create(store, loc);
    });
    Value leader = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, lane, ix(in, 0));
    scf::IfOp::create(in, loc, leader, [&](OpBuilder &store, Location) {
      // Count retained descriptors too: SW reads both hybrid class orders.
      Value column = asIndex(store, loc,
          load(store, loc, v.build.groups.columns, descriptor));
      emitDeviceAtomicAdd(store, loc, v.spatial.counts, column, word(store, loc, 1));
      for (int64_t side = 0; side != 2; ++side) {
        Value row = arith::AddIOp::create(store, loc,
            arith::MulIOp::create(store, loc, descriptor, ix(store, 2)), ix(store, side));
        for (int64_t axis = 0; axis != 3; ++axis) {
          memref::StoreOp::create(store, loc,
              vector::ExtractOp::create(store, loc, side ? box.maximum : box.minimum, axis),
              v.groupBoxes, ValueRange{row, ix(store, axis)});
          memref::StoreOp::create(store, loc,
              vector::ExtractOp::create(store, loc,
                  side ? fractionalBox.maximum : fractionalBox.minimum, axis),
              v.fractionalBoxes, ValueRange{row, ix(store, axis)});
        }
      }
      scf::YieldOp::create(store, loc);
    });
    scf::YieldOp::create(in, loc);
  });
  gpu::TerminatorOp::create(at, loc);
}

