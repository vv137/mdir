// The emitters of the privatized scatter-accumulate on a device
// (D291); see include/mdir/Conversion/ScatterEmitter.h.

#include "mdir/Conversion/ScatterEmitter.h"

#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

Value kernels::getSharedAddress(OpBuilder &b, Location loc, Value buffer,
                                Value index) {
  Type wide = b.getI64Type();
  auto type = cast<MemRefType>(buffer.getType());
  int64_t bytes = type.getElementType().getIntOrFloatBitWidth() / 8;
  Value base = arith::IndexCastOp::create(
      b, loc, wide,
      memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buffer));
  Value offset = arith::MulIOp::create(
      b, loc, arith::IndexCastOp::create(b, loc, wide, index),
      arith::ConstantOp::create(b, loc, wide, b.getI64IntegerAttr(bytes)));
  return arith::AddIOp::create(b, loc, base, offset);
}

void kernels::emitSharedAdd(OpBuilder &b, Location loc, Value address,
                            Value value) {
  bool isF32 = value.getType().isF32();
  LLVM::InlineAsmOp::create(
      b, loc, TypeRange(), ValueRange{address, value},
      isF32 ? "red.shared.add.f32 [$0], $1;" : "red.shared.add.f64 [$0], $1;",
      isF32 ? "l,f" : "l,d", /*has_side_effects=*/true,
      /*is_align_stack=*/false, LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr(), ArrayAttr());
}

Value kernels::emitSharedCompareSwap(OpBuilder &b, Location loc,
                                     Value address, Value expected,
                                     Value desired) {
  return LLVM::InlineAsmOp::create(
             b, loc, TypeRange{b.getI32Type()},
             ValueRange{address, expected, desired},
             "atom.shared.cas.b32 $0, [$1], $2, $3;", "=r,l,r,r",
             /*has_side_effects=*/true, /*is_align_stack=*/false,
             LLVM::TailCallKind::None, LLVM::AsmDialectAttr(), ArrayAttr())
      .getResult(0);
}

SmallVector<Type> kernels::getHashAttributions(
    MLIRContext *context, int64_t slots,
    ArrayRef<md_exec::scatter::Destination> destinations) {
  auto workgroup = gpu::AddressSpaceAttr::get(
      context, gpu::GPUDialect::getWorkgroupAddressSpace());
  SmallVector<Type> attributions{MemRefType::get(
      {slots}, IntegerType::get(context, 32), MemRefLayoutAttrInterface(),
      workgroup)};
  for (const md_exec::scatter::Destination &destination : destinations)
    attributions.push_back(MemRefType::get({slots * destination.components},
                                           destination.element,
                                           MemRefLayoutAttrInterface(),
                                           workgroup));
  return attributions;
}

BlockHashTable::BlockHashTable(
    OpBuilder &kernel, Location loc, int64_t slots, int64_t probes,
    ValueRange shared, ArrayRef<Value> outs,
    ArrayRef<md_exec::scatter::Destination> destinations, Value thread,
    Value threads)
    : loc(loc), slots(slots), probes(probes),
      shift(32 - llvm::Log2_64(slots)), keys(shared.front()),
      sums(shared.drop_front()), outs(outs), thread(thread),
      threads(threads) {
  for (const md_exec::scatter::Destination &destination : destinations) {
    components.push_back(destination.components);
    elements.push_back(destination.element);
  }
  numSlots = createIndex(kernel, loc, slots);
  empty = arith::ConstantOp::create(kernel, loc, kernel.getI32Type(),
                                    kernel.getI32IntegerAttr(-1));
}

void BlockHashTable::emitClear(OpBuilder &kernel) {
  scf::ForOp::create(
      kernel, loc, thread, numSlots, threads, ValueRange(),
      [&](OpBuilder &b, Location, Value slot, ValueRange) {
        memref::StoreOp::create(b, loc, empty, keys, ValueRange{slot});
        for (auto [k, c] : llvm::enumerate(components)) {
          Value zero = createZero(b, loc, elements[k]);
          for (int64_t j = 0; j != c; ++j) {
            Value place = arith::AddIOp::create(
                b, loc, arith::MulIOp::create(b, loc, slot,
                                              createIndex(b, loc, c)),
                createIndex(b, loc, j));
            memref::StoreOp::create(b, loc, zero, sums[k], ValueRange{place});
          }
        }
        scf::YieldOp::create(b, loc);
      });
  gpu::BarrierOp::create(kernel, loc);
}

// The slot of the index `index`: a multiplicative hash of its number
// [Knuth1998, Section 6.4], then the next slots in turn, until a slot is
// empty, which the thread takes with a compare-and-swap, or holds the
// index; two threads that find one empty slot take it once. The search
// gives up after `probes` slots, and the value goes to the destination with
// an atomic of its own: a block whose values share few indices fills its
// table, where the slots that a search tries grow without bound (measured
// on the tuples, D283: one block of JAC with 494 particles in 512 slots).
std::pair<Value, Value> BlockHashTable::emitFind(OpBuilder &b, Value index) {
  Type i32 = b.getI32Type();
  Value key = arith::IndexCastOp::create(b, loc, i32, index);
  auto constant = [&](OpBuilder &c, int64_t value) -> Value {
    return arith::ConstantOp::create(
        c, loc, i32, c.getI32IntegerAttr(static_cast<int32_t>(value)));
  };
  Value hash = arith::ShRUIOp::create(
      b, loc, arith::MulIOp::create(b, loc, key, constant(b, 0x9E3779B1u)),
      constant(b, shift));
  Type i1 = b.getI1Type();
  auto search = scf::WhileOp::create(
      b, loc, TypeRange{i32, i32, i1}, ValueRange{hash, constant(b, 1)},
      [&](OpBuilder &c, Location, ValueRange arguments) {
        Value slot = arguments[0], tried = arguments[1];
        Value address = getSharedAddress(
            c, loc, keys,
            arith::IndexCastOp::create(c, loc, c.getIndexType(), slot));
        Value old = emitSharedCompareSwap(c, loc, address, empty, key);
        Value taken =
            arith::CmpIOp::create(c, loc, arith::CmpIPredicate::eq, old, empty);
        Value held =
            arith::CmpIOp::create(c, loc, arith::CmpIPredicate::eq, old, key);
        Value hit = arith::OrIOp::create(c, loc, taken, held);
        Value more = arith::CmpIOp::create(c, loc, arith::CmpIPredicate::ult,
                                           tried, constant(c, probes));
        Value next = arith::AndIOp::create(
            c, loc, more,
            arith::XOrIOp::create(
                c, loc, hit,
                arith::ConstantOp::create(c, loc, i1, c.getBoolAttr(true))));
        scf::ConditionOp::create(c, loc, next, ValueRange{slot, tried, hit});
      },
      [&](OpBuilder &c, Location, ValueRange arguments) {
        Value next = arith::AndIOp::create(
            c, loc, arith::AddIOp::create(c, loc, arguments[0], constant(c, 1)),
            constant(c, slots - 1));
        Value tried =
            arith::AddIOp::create(c, loc, arguments[1], constant(c, 1));
        scf::YieldOp::create(c, loc, ValueRange{next, tried});
      });
  return {arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                     search.getResult(0)),
          search.getResult(2)};
}

void BlockHashTable::add(OpBuilder &b, Value out, Value index, Value value) {
  std::pair<Value, Value> &lookup = found[index];
  if (!lookup.first || lookup.first.getParentBlock() != b.getInsertionBlock())
    lookup = emitFind(b, index);
  auto [slot, inTable] = lookup;
  unsigned k = llvm::find(outs, out) - outs.begin();
  int64_t c = components[k];
  scf::IfOp::create(
      b, loc, inTable,
      [&](OpBuilder &then, Location) {
        for (int64_t j = 0; j != c; ++j) {
          Value part = c == 1 && !isa<VectorType>(value.getType())
                           ? value
                           : vector::ExtractOp::create(then, loc, value, j)
                                 .getResult();
          Value place = arith::AddIOp::create(
              then, loc,
              arith::MulIOp::create(then, loc, slot, createIndex(then, loc, c)),
              createIndex(then, loc, j));
          Value address = getSharedAddress(then, loc, sums[k], place);
          emitSharedAdd(then, loc, address, part);
        }
        scf::YieldOp::create(then, loc);
      },
      [&](OpBuilder &otherwise, Location) {
        emitAtomicAdd(otherwise, loc, value, out, index);
        scf::YieldOp::create(otherwise, loc);
      });
}

void BlockHashTable::emitFlush(OpBuilder &kernel) {
  gpu::BarrierOp::create(kernel, loc);
  scf::ForOp::create(
      kernel, loc, thread, numSlots, threads, ValueRange(),
      [&](OpBuilder &b, Location, Value slot, ValueRange) {
        Value key = memref::LoadOp::create(b, loc, keys, ValueRange{slot});
        Value held = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                           key, empty);
        scf::IfOp::create(b, loc, held, [&](OpBuilder &then, Location) {
          Value row =
              arith::IndexCastOp::create(then, loc, then.getIndexType(), key);
          for (auto [k, c] : llvm::enumerate(components)) {
            SmallVector<Value, 3> parts;
            for (int64_t j = 0; j != c; ++j) {
              Value place = arith::AddIOp::create(
                  then, loc,
                  arith::MulIOp::create(then, loc, slot,
                                        createIndex(then, loc, c)),
                  createIndex(then, loc, j));
              parts.push_back(
                  memref::LoadOp::create(then, loc, sums[k], ValueRange{place}));
            }
            Value value =
                cast<MemRefType>(outs[k].getType()).getRank() == 1
                    ? parts.front()
                    : vector::FromElementsOp::create(
                          then, loc, VectorType::get({c}, elements[k]), parts)
                          .getResult();
            emitAtomicAdd(then, loc, value, outs[k], row);
          }
          scf::YieldOp::create(then, loc);
        });
        scf::YieldOp::create(b, loc);
      });
}
