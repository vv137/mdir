#include "mdir/Conversion/TemplatePruneKernels.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir::kernels;

void mdir::kernels::emitPruneTemplateWork(OpBuilder &b, Location loc,
    const TemplatePruneBuffers &v, Value work, Value lane) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto word = [&](OpBuilder &at, int64_t n) -> Value {
    return arith::ConstantIntOp::create(at, loc, n, 32);
  };
  auto address = [&](OpBuilder &at, Value record, int64_t col) -> Value {
    return arith::AddIOp::create(at, loc,
        arith::MulIOp::create(at, loc, record, ix(at, 4)), ix(at, col));
  };
  auto read = [&](OpBuilder &at, Value buffer, Value record, int64_t col) -> Value {
    return memref::LoadOp::create(at, loc, buffer, ValueRange{address(at, record, col)});
  };
  auto index = [&](OpBuilder &at, Value x) -> Value {
    return arith::IndexCastUIOp::create(at, loc, at.getIndexType(), x);
  };
  // An outer record is [class | slice << 2, first, slice begin, row end]:
  // the slice-th chunk of 64 entries of its row (U10).
  Value packed = read(b, v.outerUnits, work, 0);
  Value tag = arith::AndIOp::create(b, loc, packed, word(b, 3));
  Value ww = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, tag, word(b, 0));
  Value ss = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, tag, word(b, 2));
  Value active = arith::SelectOp::create(b, loc, ww, v.waterActive,
      arith::SelectOp::create(b, loc, ss, v.atomActive,
          arith::OrIOp::create(b, loc, v.waterActive, v.atomActive)));
  scf::IfOp::create(b, loc, active, [&](OpBuilder &at, Location) {
    Value firstBegin = read(at, v.outerUnits, work, 1);
    Value firstLength = index(at, memref::LoadOp::create(at, loc, v.groupLengths,
        ValueRange{index(at, firstBegin)}));
    Value beginWord = read(at, v.outerUnits, work, 2);
    Value low = arith::MulIOp::create(at, loc,
        index(at, arith::ShRUIOp::create(at, loc, packed, word(at, 2))), ix(at, 64));
    Value high = arith::AddIOp::create(at, loc, low, ix(at, 64));
    Value begin = arith::SubIOp::create(at, loc, index(at, beginWord), low);
    Value end = index(at, read(at, v.outerUnits, work, 3));
    auto f32 = at.getF32Type();
    auto vec3 = VectorType::get({3}, f32);
    auto real = [&](OpBuilder &in, double x) -> Value {
      return arith::ConstantOp::create(in, loc, FloatAttr::get(f32, x));
    };
    auto narrow = [&](OpBuilder &in, Value x) -> Value {
      return arith::TruncFOp::create(in, loc,
          VectorType::get(cast<VectorType>(x.getType()).getShape(), f32), x);
    };
    auto largest = [&](OpBuilder &in, Value x) -> Value {
      return vector::ReductionOp::create(in, loc, vector::CombiningKind::MAXNUMF,
          math::AbsFOp::create(in, loc, x));
    };
    // Select the class reach rather than index the vector dynamically.
    Value reach = vector::ExtractOp::create(at, loc, v.reaches, 2);
    for (int64_t t = 1; t >= 0; --t)
      reach = arith::SelectOp::create(at, loc,
          arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, tag, word(at, t)),
          vector::ExtractOp::create(at, loc, v.reaches, t), reach);
    Value reach32 = arith::TruncFOp::create(at, loc, f32, reach);
    Value cell = v.narrowCell ? v.narrowCell : narrow(at, v.cell);
    Value cellMagnitude = v.cellMagnitude ? v.cellMagnitude
        : Value(vector::ReductionOp::create(at, loc, vector::CombiningKind::ADD,
              math::AbsFOp::create(at, loc, cell)));
    auto reference = [&](OpBuilder &in, Value unit) -> Value {
      Value water = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult,
          index(in, unit), v.waters);
      Value current = arith::SelectOp::create(in, loc, water, v.waterActive, v.atomActive);
      auto selected = scf::IfOp::create(in, loc, current,
          [&](OpBuilder &take, Location) {
        scf::YieldOp::create(take, loc,
            ValueRange{loadElement(take, loc, v.currentPoints, index(take, unit))});
      }, [&](OpBuilder &take, Location) {
        scf::YieldOp::create(take, loc,
            ValueRange{loadElement(take, loc, v.innerReference, index(take, unit))});
      });
      return selected.getResult(0);
    };
    // The 16 first units are read once per record, by the lane of their
    // place in either half warp, and broadcast to every entry by shuffles.
    Value own = arith::AndIOp::create(at, loc, lane, word(at, 15));
    Value limit = arith::SubIOp::create(at, loc,
        arith::MaxUIOp::create(at, loc, firstLength, ix(at, 1)), ix(at, 1));
    Value place = arith::AddIOp::create(at, loc, index(at, firstBegin),
        arith::MinUIOp::create(at, loc, index(at, own), limit));
    Value ownUnit = memref::LoadOp::create(at, loc, v.order, ValueRange{place});
    Value ownValid = arith::AndIOp::create(at, loc,
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, index(at, own), firstLength),
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::sge, ownUnit, word(at, 0)));
    auto ownPoint = scf::IfOp::create(at, loc, ownValid,
        [&](OpBuilder &take, Location) {
      scf::YieldOp::create(take, loc, ValueRange{narrow(take, reference(take, ownUnit))});
    }, [&](OpBuilder &take, Location) {
      scf::YieldOp::create(take, loc, ValueRange{createZero(take, loc, vec3)});
    });
    Value firstPoint = ownPoint.getResult(0);
    Value firstValid = emitWarpBallot(at, loc, ownValid);
    Value firstMagnitude = emitWarpReduce(at, loc, largest(at, firstPoint), Combiner::Max);
    // The row's surviving entries are compacted from the row's first entry
    // and taken 64 a record: this record's warp tests the row's entries
    // from its first until 64 times (slice + 1) survive, and writes those
    // whose rank falls in its own slice. Each warp writes only its slice,
    // so no record waits for another's prune; a row's first records hold
    // full batches and its last ones may be empty.
    // Batches are warp uniform. A lane past the end tests the first entry
    // of the row and discards it, so that every lane reaches the shuffles.
    auto batches = scf::WhileOp::create(at, loc, TypeRange{at.getIndexType(), at.getIndexType()},
        ValueRange{begin, ix(at, 0)},
        [&](OpBuilder &test, Location, ValueRange state) {
      Value more = arith::AndIOp::create(test, loc,
          arith::CmpIOp::create(test, loc, arith::CmpIPredicate::ult, state[0], end),
          arith::CmpIOp::create(test, loc, arith::CmpIPredicate::ult, state[1], high));
      scf::ConditionOp::create(test, loc, more, state);
    }, [&](OpBuilder &in, Location, ValueRange state) {
      Value batch = state[0];
      ValueRange keptBefore = state.drop_front();
      Value entry = arith::AddIOp::create(in, loc, batch, index(in, lane));
      Value present = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult, entry, end);
      Value safe = arith::SelectOp::create(in, loc, present, entry, begin);
      SmallVector<Value, 4> fields;
      for (int col = 0; col != 4; ++col) fields.push_back(read(in, v.outerEntries, safe, col));
      Value partner = narrow(in, reference(in, fields[0]));
      SmallVector<Value, 3> image;
      for (unsigned axis = 0; axis != 3; ++axis) {
        Value component = arith::AndIOp::create(in, loc,
            arith::ShRUIOp::create(in, loc, fields[1], word(in, MDRT_TEMPLATE_IMAGE_BITS * axis)),
            word(in, MDRT_TEMPLATE_IMAGE_MASK));
        component = arith::SubIOp::create(in, loc, component, word(in, MDRT_TEMPLATE_IMAGE_OFFSET));
        image.push_back(arith::SIToFPOp::create(in, loc, f32, component));
      }
      Value lattice = vector::FromElementsOp::create(in, loc, vec3, image);
      // The test runs in f32 on fractional points of magnitude at most E,
      // E = max|s_i| + |s_j| + |n|. Each Cartesian component then errs by
      // at most 7 u (sum |H|) E, u = 2^-24, and the reach is widened by
      // 32 u ((sum |H|) E + R): every pair the exact test keeps is kept.
      Value magnitude = arith::AddFOp::create(in, loc, firstMagnitude,
          arith::AddFOp::create(in, loc, largest(in, partner), largest(in, lattice)));
      Value widened = arith::AddFOp::create(in, loc, reach32,
          arith::MulFOp::create(in, loc, real(in, 0x1p-19),
              arith::AddFOp::create(in, loc,
                  arith::MulFOp::create(in, loc, cellMagnitude, magnitude), reach32)));
      Value reach2 = arith::MulFOp::create(in, loc, widened, widened);
      Value candidates = arith::AndIOp::create(in, loc, fields[2], firstValid);
      Value mask = word(in, 0);
      for (int bit = 0; bit != 16; ++bit) {
        SmallVector<Value, 3> components;
        for (int64_t axis = 0; axis != 3; ++axis)
          components.push_back(gpu::ShuffleOp::create(in, loc,
              vector::ExtractOp::create(in, loc, firstPoint, axis), word(in, bit),
              word(in, 32), gpu::ShuffleMode::IDX).getResult(0));
        Value point = vector::FromElementsOp::create(in, loc, vec3, components);
        Value delta = arith::SubFOp::create(in, loc,
            arith::SubFOp::create(in, loc, point, partner), lattice);
        Value cartesian = emitLatticeShift(in, loc, delta, cell);
        Value squared = vector::ReductionOp::create(in, loc, vector::CombiningKind::ADD,
            arith::MulFOp::create(in, loc, cartesian, cartesian));
        Value within = arith::CmpFOp::create(in, loc, arith::CmpFPredicate::OLE, squared, reach2);
        mask = arith::OrIOp::create(in, loc, mask,
            arith::SelectOp::create(in, loc, within, word(in, 1 << bit), word(in, 0)));
      }
      mask = arith::AndIOp::create(in, loc, mask, candidates);
      fields[2] = arith::SelectOp::create(in, loc, present, mask, word(in, 0));
      Value keep = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ne,
          fields[2], word(in, 0));
      Value ballot = emitWarpBallot(in, loc, keep);
      Value rank = index(in, emitWarpRank(in, loc, ballot, emitLanesBelow(in, loc, lane)));
      Value count = index(in, emitWarpCount(in, loc, ballot));
      Value ranked = arith::AddIOp::create(in, loc, keptBefore[0], rank);
      Value mine = arith::AndIOp::create(in, loc, keep, arith::AndIOp::create(in, loc,
          arith::CmpIOp::create(in, loc, arith::CmpIPredicate::uge, ranked, low),
          arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult, ranked, high)));
      scf::IfOp::create(in, loc, mine, [&](OpBuilder &write, Location) {
        Value destination = arith::AddIOp::create(write, loc, begin, ranked);
        for (int col = 0; col != 4; ++col)
          memref::StoreOp::create(write, loc, fields[col], v.innerEntries,
              ValueRange{address(write, destination, col)});
        scf::YieldOp::create(write, loc);
      });
      scf::YieldOp::create(in, loc, ValueRange{arith::AddIOp::create(in, loc, batch, ix(in, 32)),
          arith::AddIOp::create(in, loc, keptBefore[0], count)});
    });
    Value leader = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, lane, word(at, 0));
    scf::IfOp::create(at, loc, leader, [&](OpBuilder &write, Location) {
      Value kept = arith::MinUIOp::create(write, loc, high,
          arith::MaxUIOp::create(write, loc, low, batches.getResult(1)));
      Value endWord = arith::IndexCastUIOp::create(write, loc, write.getI32Type(),
          arith::AddIOp::create(write, loc, begin, kept));
      Value fields[] = {tag, firstBegin, beginWord, endWord};
      for (int col = 0; col != 4; ++col)
        memref::StoreOp::create(write, loc, fields[col], v.innerUnits,
            ValueRange{address(write, work, col)});
      scf::YieldOp::create(write, loc);
    });
    NVVM::SyncWarpOp::create(at, loc, word(at, -1));
    scf::YieldOp::create(at, loc);
  });
}
