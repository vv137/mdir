// Single-pass device scan (D[device-single-pass-scan]).
// Merrill and Garland, NVR-2016-002, Sections 4 and 4.4.

#include "mdir/Conversion/CompactionEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mdir::kernels;

static Value at(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantIndexOp::create(b, loc, v);
}

static Value i32c(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                   b.getI32IntegerAttr(v));
}

static Value i64c(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantOp::create(b, loc, b.getI64Type(),
                                   b.getI64IntegerAttr(v));
}

static Value wordPointer(OpBuilder &b, Location loc, Value buffer, Value slot) {
  // The state is a fresh gpu.alloc with zero memref offset; only its base
  // pointer and the word index are needed by the device atomic.
  Type i64 = b.getI64Type();
  Value base = arith::IndexCastOp::create(
      b, loc, i64,
      memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buffer));
  Value byte = arith::MulIOp::create(
      b, loc, arith::IndexCastOp::create(b, loc, i64, slot),
      i64c(b, loc, 8));
  Value address = arith::AddIOp::create(b, loc, base, byte);
  return LLVM::IntToPtrOp::create(
      b, loc, LLVM::LLVMPointerType::get(b.getContext(), 1), address);
}

static Value atomicRead(OpBuilder &b, Location loc, Value state, Value tile) {
  Value slot = arith::AddIOp::create(b, loc, tile, at(b, loc, 1));
  return LLVM::AtomicRMWOp::create(
      b, loc, LLVM::AtomicBinOp::add, wordPointer(b, loc, state, slot),
      i64c(b, loc, 0), LLVM::AtomicOrdering::acquire,
      StringRef("device"));
}

static void atomicPublish(OpBuilder &b, Location loc, Value state, Value tile,
                          int64_t phase, Value sum) {
  Value high = arith::ShLIOp::create(b, loc, i64c(b, loc, phase),
                                     i64c(b, loc, 32));
  Value low = arith::ExtUIOp::create(b, loc, b.getI64Type(), sum);
  Value descriptor = arith::OrIOp::create(b, loc, high, low);
  Value slot = arith::AddIOp::create(b, loc, tile, at(b, loc, 1));
  LLVM::AtomicRMWOp::create(
      b, loc, LLVM::AtomicBinOp::xchg, wordPointer(b, loc, state, slot),
      descriptor, LLVM::AtomicOrdering::release, StringRef("device"));
}

void mdir::kernels::emitSinglePassScan(OpBuilder &b, Location loc,
                                       Value counts, Value start, Value cursor,
                                       Value n, Value state, Value blocks,
                                       Value threads, Value one) {
  Type i32 = b.getI32Type();
  auto shared = MemRefType::get(
      {9}, i32, MemRefLayoutAttrInterface(),
      gpu::AddressSpaceAttr::get(b.getContext(),
                                 gpu::GPUDialect::getWorkgroupAddressSpace()));
  auto prefixShared = MemRefType::get(
      {1}, i32, MemRefLayoutAttrInterface(),
      gpu::AddressSpaceAttr::get(b.getContext(),
                                 gpu::GPUDialect::getWorkgroupAddressSpace()));

  // The first word is a ticket counter; tile t's descriptor is word t + 1.
  Value words = arith::AddIOp::create(b, loc, blocks, one);
  Value grid = arith::DivUIOp::create(
      b, loc,
      arith::AddIOp::create(
          b, loc, words,
          arith::SubIOp::create(b, loc, threads, one)),
      threads);
  auto init = gpu::LaunchOp::create(b, loc, grid, one, one, threads, one, one);
  {
    OpBuilder k = OpBuilder::atBlockEnd(&init.getBody().front());
    Value slot = arith::AddIOp::create(
        k, loc, arith::MulIOp::create(k, loc, init.getBlockIds().x, threads),
        init.getThreadIds().x);
    Value inside = arith::CmpIOp::create(
        k, loc, arith::CmpIPredicate::ult, slot, words);
    scf::IfOp::create(k, loc, inside, [&](OpBuilder &t, Location) {
      memref::StoreOp::create(t, loc, i64c(t, loc, 0), state,
                              ValueRange{slot});
      scf::YieldOp::create(t, loc);
    });
    gpu::TerminatorOp::create(k, loc);
  }

  auto op = gpu::LaunchOp::create(
      b, loc, blocks, one, one, threads, one, one,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/{},
      /*workgroupAttributions=*/TypeRange{shared, prefixShared});
  OpBuilder k = OpBuilder::atBlockEnd(&op.getBody().front());
  Value thread = op.getThreadIds().x;
  Value scratch = op.getWorkgroupAttributionBBArgs().front();
  Value prefixScratch = op.getWorkgroupAttributionBBArgs()[1];
  Value zero = at(k, loc, 0);
  Value leader = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::eq,
                                       thread, zero);
  scf::IfOp::create(k, loc, leader, [&](OpBuilder &t, Location) {
    Value ticket = LLVM::AtomicRMWOp::create(
        t, loc, LLVM::AtomicBinOp::add,
        wordPointer(t, loc, state, at(t, loc, 0)), i64c(t, loc, 1),
        LLVM::AtomicOrdering::monotonic, StringRef("device"));
    memref::StoreOp::create(
        t, loc, arith::TruncIOp::create(t, loc, i32, ticket), scratch,
        ValueRange{at(t, loc, 8)});
    scf::YieldOp::create(t, loc);
  });
  gpu::BarrierOp::create(k, loc);
  Value tile = arith::IndexCastOp::create(
      k, loc, k.getIndexType(),
      memref::LoadOp::create(k, loc, scratch, ValueRange{at(k, loc, 8)}));
  Value first = arith::AddIOp::create(
      k, loc, arith::MulIOp::create(k, loc, tile, at(k, loc, 1024)),
      arith::MulIOp::create(k, loc, thread, at(k, loc, 4)));
  SmallVector<Value, 4> values;
  Value sum = i32c(k, loc, 0);
  for (int64_t j = 0; j != 4; ++j) {
    Value slot = arith::AddIOp::create(k, loc, first, at(k, loc, j));
    Value inside = arith::CmpIOp::create(
        k, loc, arith::CmpIPredicate::ult, slot, n);
    auto read = scf::IfOp::create(k, loc, TypeRange{i32}, inside,
                                  /*withElseRegion=*/true);
    OpBuilder yes = OpBuilder::atBlockEnd(read.thenBlock());
    scf::YieldOp::create(
        yes, loc,
        memref::LoadOp::create(yes, loc, counts, ValueRange{slot})
            .getResult());
    OpBuilder no = OpBuilder::atBlockEnd(read.elseBlock());
    scf::YieldOp::create(no, loc, i32c(no, loc, 0));
    values.push_back(read.getResult(0));
    sum = arith::AddIOp::create(k, loc, sum, values.back());
  }
  BlockScan local = emitBlockScan(k, loc, sum, scratch, thread);

  scf::IfOp::create(k, loc, leader, [&](OpBuilder &t, Location) {
    Value firstTile = arith::CmpIOp::create(
        t, loc, arith::CmpIPredicate::eq, tile, at(t, loc, 0));
    auto prefix = scf::IfOp::create(
        t, loc, TypeRange{i32}, firstTile, /*withElseRegion=*/true);
    {
      OpBuilder head = OpBuilder::atBlockEnd(prefix.thenBlock());
      atomicPublish(head, loc, state, tile, 2, local.total);
      scf::YieldOp::create(head, loc, i32c(head, loc, 0));
      OpBuilder tail = OpBuilder::atBlockEnd(prefix.elseBlock());
      atomicPublish(tail, loc, state, tile, 1, local.total);
      Value prior = arith::SubIOp::create(tail, loc, tile, at(tail, loc, 1));
      Value falseValue = arith::ConstantOp::create(
          tail, loc, tail.getI1Type(), tail.getBoolAttr(false));
      auto look = scf::WhileOp::create(
          tail, loc,
          TypeRange{tail.getIndexType(), i32, tail.getI1Type()},
          ValueRange{prior, i32c(tail, loc, 0), falseValue},
          [&](OpBuilder &w, Location, ValueRange args) {
            Value more = arith::XOrIOp::create(
                w, loc, args[2],
                arith::ConstantOp::create(w, loc, w.getI1Type(),
                                           w.getBoolAttr(true)));
            scf::ConditionOp::create(w, loc, more, args);
          },
          [&](OpBuilder &w, Location, ValueRange args) {
            Value descriptor = atomicRead(w, loc, state, args[0]);
            Value phase = arith::TruncIOp::create(
                w, loc, i32,
                arith::ShRUIOp::create(w, loc, descriptor, i64c(w, loc, 32)));
            Value available = arith::CmpIOp::create(
                w, loc, arith::CmpIPredicate::ne, phase, i32c(w, loc, 0));
            Value value = arith::TruncIOp::create(w, loc, i32, descriptor);
            Value nextSum = arith::AddIOp::create(w, loc, args[1], value);
            Value accumulated = arith::SelectOp::create(
                w, loc, available, nextSum, args[1]);
            Value next = arith::SubIOp::create(w, loc, args[0],
                                               at(w, loc, 1));
            Value position = arith::SelectOp::create(
                w, loc, available, next, args[0]);
            Value complete = arith::CmpIOp::create(
                w, loc, arith::CmpIPredicate::eq, phase, i32c(w, loc, 2));
            scf::YieldOp::create(
                w, loc, ValueRange{position, accumulated, complete});
          });
      Value base = look.getResult(1);
      atomicPublish(tail, loc, state, tile, 2,
                    arith::AddIOp::create(tail, loc, base, local.total));
      scf::YieldOp::create(tail, loc, base);
    }
    // Keep the prefix separate from the block scan's total in scratch[8]:
    // other warps can still load that total after the leader finishes.
    memref::StoreOp::create(t, loc, prefix.getResult(0), prefixScratch,
                            ValueRange{at(t, loc, 0)});
    Value last = arith::CmpIOp::create(
        t, loc, arith::CmpIPredicate::eq, tile,
        arith::SubIOp::create(t, loc, blocks, at(t, loc, 1)));
    scf::IfOp::create(t, loc, last, [&](OpBuilder &end, Location) {
      Value total = arith::AddIOp::create(end, loc, prefix.getResult(0),
                                          local.total);
      memref::StoreOp::create(end, loc, total, start, ValueRange{n});
      scf::YieldOp::create(end, loc);
    });
    scf::YieldOp::create(t, loc);
  });
  gpu::BarrierOp::create(k, loc);
  Value before = arith::AddIOp::create(
      k, loc, memref::LoadOp::create(k, loc, prefixScratch,
                                     ValueRange{at(k, loc, 0)}),
      local.before);
  for (int64_t j = 0; j != 4; ++j) {
    Value slot = arith::AddIOp::create(k, loc, first, at(k, loc, j));
    Value inside = arith::CmpIOp::create(
        k, loc, arith::CmpIPredicate::ult, slot, n);
    scf::IfOp::create(k, loc, inside, [&](OpBuilder &t, Location) {
      memref::StoreOp::create(t, loc, before, start, ValueRange{slot});
      if (cursor)
        memref::StoreOp::create(t, loc, before, cursor, ValueRange{slot});
      scf::YieldOp::create(t, loc);
    });
    before = arith::AddIOp::create(k, loc, before, values[j]);
  }
  gpu::TerminatorOp::create(k, loc);
}
