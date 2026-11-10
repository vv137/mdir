#include "mdir/Conversion/TemplatePruneKernels.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  Value tag = read(b, v.outerUnits, work, 0);
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
    Value begin = index(at, beginWord);
    Value end = index(at, read(at, v.outerUnits, work, 3));
    Value reach = vector::ExtractOp::create(at, loc, v.reaches, index(at, tag));
    Value reach2 = arith::MulFOp::create(at, loc, reach, reach);
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
    // Batches are warp uniform. Inactive lanes never load a tail entry;
    // every lane still participates in the ballot and final barrier.
    auto batches = scf::ForOp::create(at, loc, begin, end, ix(at, 32),
        ValueRange{ix(at, 0)},
        [&](OpBuilder &in, Location, Value batch, ValueRange keptBefore) {
      Value entry = arith::AddIOp::create(in, loc, batch, index(in, lane));
      Value present = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ult, entry, end);
      auto candidate = scf::IfOp::create(in, loc, present,
          [&](OpBuilder &test, Location) {
        SmallVector<Value, 4> fields;
        for (int col = 0; col != 4; ++col) fields.push_back(read(test, v.outerEntries, entry, col));
        Value partner = reference(test, fields[0]);
        SmallVector<Value, 3> image;
        for (unsigned axis = 0; axis != 3; ++axis) {
          Value component = arith::AndIOp::create(test, loc,
              arith::ShRUIOp::create(test, loc, fields[1], word(test, MDRT_TEMPLATE_IMAGE_BITS * axis)),
              word(test, MDRT_TEMPLATE_IMAGE_MASK));
          component = arith::SubIOp::create(test, loc, component, word(test, MDRT_TEMPLATE_IMAGE_OFFSET));
          image.push_back(arith::SIToFPOp::create(test, loc, test.getF64Type(), component));
        }
        Value lattice = vector::FromElementsOp::create(test, loc,
            VectorType::get({3}, test.getF64Type()), image);
        Value mask = word(test, 0);
        for (int bit = 0; bit != 16; ++bit) {
          Value inSpan = arith::CmpIOp::create(test, loc, arith::CmpIPredicate::ult,
              ix(test, bit), firstLength);
          Value limit = arith::SubIOp::create(test, loc,
              arith::MaxUIOp::create(test, loc, firstLength, ix(test, 1)), ix(test, 1));
          Value place = arith::AddIOp::create(test, loc, index(test, firstBegin),
              arith::MinUIOp::create(test, loc, ix(test, bit), limit));
          Value unit = memref::LoadOp::create(test, loc, v.order, ValueRange{place});
          Value valid = arith::AndIOp::create(test, loc, inSpan, arith::AndIOp::create(test, loc,
              arith::CmpIOp::create(test, loc, arith::CmpIPredicate::sge, unit, word(test, 0)),
              arith::CmpIOp::create(test, loc, arith::CmpIPredicate::ne,
                  arith::AndIOp::create(test, loc, fields[2], word(test, 1 << bit)), word(test, 0))));
          auto vote = scf::IfOp::create(test, loc, valid,
              [&](OpBuilder &check, Location) {
            Value delta = arith::SubFOp::create(check, loc,
                arith::SubFOp::create(check, loc, reference(check, unit), partner), lattice);
            Value cartesian = emitLatticeShift(check, loc, delta, v.cell);
            Value squared = vector::ReductionOp::create(check, loc, vector::CombiningKind::ADD,
                arith::MulFOp::create(check, loc, cartesian, cartesian));
            Value within = arith::CmpFOp::create(check, loc, arith::CmpFPredicate::OLE, squared, reach2);
            scf::YieldOp::create(check, loc,
                ValueRange{arith::SelectOp::create(check, loc, within, word(check, 1 << bit), word(check, 0))});
          }, [&](OpBuilder &check, Location) {
            scf::YieldOp::create(check, loc, ValueRange{word(check, 0)});
          });
          mask = arith::OrIOp::create(test, loc, mask, vote.getResult(0));
        }
        fields[2] = mask;
        scf::YieldOp::create(test, loc, fields);
      }, [&](OpBuilder &test, Location) {
        Value zero = word(test, 0);
        scf::YieldOp::create(test, loc, ValueRange{zero, zero, zero, zero});
      });
      Value keep = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ne,
          candidate.getResult(2), word(in, 0));
      Value ballot = emitWarpBallot(in, loc, keep);
      Value rank = index(in, emitWarpRank(in, loc, ballot, emitLanesBelow(in, loc, lane)));
      Value count = index(in, emitWarpCount(in, loc, ballot));
      scf::IfOp::create(in, loc, keep, [&](OpBuilder &write, Location) {
        Value destination = arith::AddIOp::create(write, loc, begin,
            arith::AddIOp::create(write, loc, keptBefore[0], rank));
        for (int col = 0; col != 4; ++col)
          memref::StoreOp::create(write, loc, candidate.getResult(col), v.innerEntries,
              ValueRange{address(write, destination, col)});
        scf::YieldOp::create(write, loc);
      });
      scf::YieldOp::create(in, loc,
          ValueRange{arith::AddIOp::create(in, loc, keptBefore[0], count)});
    });
    Value leader = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, lane, word(at, 0));
    scf::IfOp::create(at, loc, leader, [&](OpBuilder &write, Location) {
      Value endWord = arith::IndexCastUIOp::create(write, loc, write.getI32Type(),
          arith::AddIOp::create(write, loc, begin, batches.getResult(0)));
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
