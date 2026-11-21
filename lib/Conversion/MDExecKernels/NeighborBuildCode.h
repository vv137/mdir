// Shared MLIR builders for GPU neighbor searches.
#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mdir::kernels::detail {
using namespace mlir;
using Values = SmallVector<Value>;
using arith::CmpFPredicate;
using arith::CmpIPredicate;

/// The ops of the build, one a call, created where the builder stands.
/// Every op is created by a statement of its own: the order of the
/// arguments of a call in C++ is not fixed, and the order of the ops is
/// that of the code.
class NeighborBuildCode {
public:
  NeighborBuildCode(OpBuilder &b, Location loc)
      : b(b), loc(loc), context(b.getContext()), i1(b.getI1Type()),
        i32(b.getI32Type()), i64(b.getI64Type()), index(b.getIndexType()),
        f32(b.getF32Type()), f64(b.getF64Type()) {
    Attribute global = b.getI64IntegerAttr(1);
    Attribute workgroup = gpu::AddressSpaceAttr::get(
        context, gpu::GPUDialect::getWorkgroupAddressSpace());
    buffer = MemRefType::get({ShapedType::kDynamic}, i32,
                             MemRefLayoutAttrInterface(), global);
    wideBuffer = MemRefType::get({ShapedType::kDynamic}, i64,
                                 MemRefLayoutAttrInterface(), global);
    sizesType = MemRefType::get({5}, i32, MemRefLayoutAttrInterface(), global);
    positions = MemRefType::get({ShapedType::kDynamic, 3}, f64,
                                MemRefLayoutAttrInterface(), global);
    wrappedType = MemRefType::get({ShapedType::kDynamic, 3}, f32,
                                  MemRefLayoutAttrInterface(), global);
    placesType = MemRefType::get({ShapedType::kDynamic, 4}, f32,
                                 MemRefLayoutAttrInterface(), global);
    boxesType = MemRefType::get({ShapedType::kDynamic, 8}, f32,
                                MemRefLayoutAttrInterface(), global);
    ids = MemRefType::get({256}, i32, MemRefLayoutAttrInterface(), workgroup);
    relative = MemRefType::get({64, 4}, f32, MemRefLayoutAttrInterface(),
                               workgroup);
    partnersType = MemRefType::get({1024}, i32, MemRefLayoutAttrInterface(),
                                   workgroup);
    pointer = LLVM::LLVMPointerType::get(context, 1);
  }

  OpBuilder &b;
  Location loc;
  MLIRContext *context;
  Type i1, i32, i64, index, f32, f64;
  MemRefType buffer, wideBuffer, sizesType, positions, wrappedType,
      placesType, boxesType, ids, relative, partnersType;
  Type pointer;

  // Constants.
  Value ci(int64_t v) {
    return arith::ConstantOp::create(b, loc, i32, b.getIntegerAttr(i32, v));
  }
  Value cl(int64_t v) {
    return arith::ConstantOp::create(b, loc, i64, b.getIntegerAttr(i64, v));
  }
  Value cx(int64_t v) { return arith::ConstantIndexOp::create(b, loc, v); }
  Value cb(bool v) {
    return arith::ConstantOp::create(b, loc, i1, b.getIntegerAttr(i1, v));
  }
  Value cf(double v) {
    return arith::ConstantOp::create(b, loc, f32, b.getFloatAttr(f32, v));
  }
  Value cd(double v) {
    return arith::ConstantOp::create(b, loc, f64, b.getFloatAttr(f64, v));
  }

  // Arithmetic.
#define MDIR_BINARY(name, Op)                                                  \
  Value name(Value x, Value y) { return arith::Op::create(b, loc, x, y); }
  MDIR_BINARY(addi, AddIOp)
  MDIR_BINARY(subi, SubIOp)
  MDIR_BINARY(muli, MulIOp)
  MDIR_BINARY(divui, DivUIOp)
  MDIR_BINARY(divsi, DivSIOp)
  MDIR_BINARY(remui, RemUIOp)
  MDIR_BINARY(remsi, RemSIOp)
  MDIR_BINARY(andi, AndIOp)
  MDIR_BINARY(ori, OrIOp)
  MDIR_BINARY(xori, XOrIOp)
  MDIR_BINARY(shli, ShLIOp)
  MDIR_BINARY(shrui, ShRUIOp)
  MDIR_BINARY(shrsi, ShRSIOp)
  MDIR_BINARY(minsi, MinSIOp)
  MDIR_BINARY(maxsi, MaxSIOp)
  MDIR_BINARY(minui, MinUIOp)
  MDIR_BINARY(maxui, MaxUIOp)
  MDIR_BINARY(addf, AddFOp)
  MDIR_BINARY(subf, SubFOp)
  MDIR_BINARY(mulf, MulFOp)
  MDIR_BINARY(divf, DivFOp)
  MDIR_BINARY(maximumf, MaximumFOp)
  MDIR_BINARY(minimumf, MinimumFOp)
  MDIR_BINARY(maxnumf, MaxNumFOp)
  MDIR_BINARY(minnumf, MinNumFOp)
#undef MDIR_BINARY
  /// The gap along an axis between the slab of the cells `row` (signed, in
  /// cells of `cell`) and the box of center `center` and half width
  /// `half`: zero where they overlap.
  Value gapOf(Value row, Value center, Value half, Value cell) {
    Value rowF = sitofp(f32, row);
    Value lo = mulf(rowF, cell);
    Value hi = addf(lo, cell);
    Value top = addf(center, half);
    Value bottom = subf(center, half);
    Value above = subf(lo, top);
    Value below = subf(bottom, hi);
    Value zero = cf(0.0);
    Value most = maximumf(above, below);
    return maximumf(most, zero);
  }
  Value notNarrow(Value narrow) {
    Value yes = cb(true);
    return xori(narrow, yes);
  }
  Value sqrt(Value v) { return math::SqrtOp::create(b, loc, v); }
  /// The floor and the ceiling of a / n, for n > 0, in signed integers.
  Value floorDiv(Value a, Value n) {
    Value q = divsi(a, n);
    Value r = remsi(a, n);
    Value zero = ci(0);
    Value one = ci(1);
    Value negative = cmpi(CmpIPredicate::slt, r, zero);
    Value down = subi(q, one);
    return select(negative, down, q);
  }
  Value ceilDiv(Value a, Value n) {
    Value q = divsi(a, n);
    Value r = remsi(a, n);
    Value zero = ci(0);
    Value one = ci(1);
    Value positive = cmpi(CmpIPredicate::sgt, r, zero);
    Value up = addi(q, one);
    return select(positive, up, q);
  }
  Value cmpi(CmpIPredicate p, Value x, Value y) {
    return arith::CmpIOp::create(b, loc, p, x, y);
  }
  Value cmpf(CmpFPredicate p, Value x, Value y) {
    return arith::CmpFOp::create(b, loc, p, x, y);
  }
  Value select(Value c, Value x, Value y) {
    return arith::SelectOp::create(b, loc, c, x, y);
  }
  Value indexCast(Type type, Value v) {
    return arith::IndexCastOp::create(b, loc, type, v);
  }
  Value sitofp(Type type, Value v) {
    return arith::SIToFPOp::create(b, loc, type, v);
  }
  Value fptosi(Type type, Value v) {
    return arith::FPToSIOp::create(b, loc, type, v);
  }
  Value truncf(Value v) { return arith::TruncFOp::create(b, loc, f32, v); }
  Value trunci(Value v) { return arith::TruncIOp::create(b, loc, i32, v); }
  Value extui(Value v) { return arith::ExtUIOp::create(b, loc, i32, v); }
  Value bitcast(Type type, Value v) {
    return arith::BitcastOp::create(b, loc, type, v);
  }
  Value floor(Value v) { return math::FloorOp::create(b, loc, v); }
  Value roundeven(Value v) { return math::RoundEvenOp::create(b, loc, v); }
  Value absf(Value v) { return math::AbsFOp::create(b, loc, v); }
  Value powf(Value x, Value y) { return math::PowFOp::create(b, loc, x, y); }

  // Memory, with no attributes (the parser of a template sets none).
  Value load(Value memref, ValueRange indices) {
    return memref::LoadOp::create(b, loc, memref, indices,
                                  /*nontemporal=*/BoolAttr(),
                                  /*alignment=*/IntegerAttr(),
                                  /*invariant=*/BoolAttr());
  }
  void store(Value v, Value memref, ValueRange indices) {
    memref::StoreOp::create(b, loc, v, memref, indices,
                            /*nontemporal=*/BoolAttr(),
                            /*alignment=*/IntegerAttr());
  }
  Value dim(Value memref, Value i) {
    return memref::DimOp::create(b, loc, memref, i);
  }
  Value alloc(MemRefType type, ValueRange sizes) {
    return gpu::AllocOp::create(b, loc, type, /*asyncToken=*/Type(),
                                ValueRange{}, sizes, ValueRange{})
        .getMemref();
  }
  /// A buffer of the device, freed: its memory space taken away, as
  /// gpu.dealloc takes it.
  void dealloc(Value memref) {
    auto type = cast<MemRefType>(memref.getType());
    auto plain = MemRefType::get(type.getShape(), type.getElementType());
    Value cast = memref::MemorySpaceCastOp::create(b, loc, plain, memref);
    gpu::DeallocOp::create(b, loc, /*asyncToken=*/Type(), ValueRange{}, cast);
  }
  Value extract(Value vector, int64_t at) {
    return vector::ExtractOp::create(b, loc, vector, ArrayRef<int64_t>{at});
  }

  // The device.
  std::pair<Value, Value> shuffle(gpu::ShuffleMode mode, Value v,
                                  Value offset, Value width) {
    auto op = gpu::ShuffleOp::create(b, loc, v, offset, width, mode);
    return {op.getShuffleResult(), op.getValid()};
  }
  void warpSync(Value mask) { NVVM::SyncWarpOp::create(b, loc, mask); }
  /// The address of `memref` plus `offset` bytes (an i64 already made),
  /// as a pointer of the device: the aligned pointer, cast to i64.
  Value pointerTo(Value base64, Value offset) {
    Value address = addi(base64, offset);
    return LLVM::IntToPtrOp::create(b, loc, pointer, address);
  }
  Value baseAddress(Value memref) {
    Value base = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, memref);
    return indexCast(i64, base);
  }
  /// A relaxed atomic at the scope of the device, whose old value it gives.
  Value atomic(LLVM::AtomicBinOp op, Value ptr, Value v) {
    return LLVM::AtomicRMWOp::create(b, loc, op, ptr, v,
                                     LLVM::AtomicOrdering::monotonic,
                                     "device");
  }

  // Calls.
  ResultRange call(StringRef callee, TypeRange results, ValueRange operands) {
    return func::CallOp::create(b, loc, callee, results, operands)
        .getResults();
  }
  Value call1(StringRef callee, Type result, ValueRange operands) {
    return call(callee, result, operands)[0];
  }

  // Structure. The bodies run with the builder in the region.
  void ifThen(Value cond, function_ref<void()> then) {
    scf::IfOp::create(b, loc, cond, [&](OpBuilder &, Location) {
      then();
      scf::YieldOp::create(b, loc);
    });
  }
  void ifThenElse(Value cond, function_ref<void()> then,
                  function_ref<void()> otherwise) {
    scf::IfOp::create(
        b, loc, cond,
        [&](OpBuilder &, Location) {
          then();
          scf::YieldOp::create(b, loc);
        },
        [&](OpBuilder &, Location) {
          otherwise();
          scf::YieldOp::create(b, loc);
        });
  }
  ResultRange ifElse(TypeRange types, Value cond, function_ref<Values()> then,
                     function_ref<Values()> otherwise) {
    auto op = scf::IfOp::create(b, loc, types, cond, /*addThenBlock=*/true,
                                /*addElseBlock=*/true);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.thenBlock());
    Values v = then();
    scf::YieldOp::create(b, loc, v);
    b.setInsertionPointToStart(op.elseBlock());
    Values w = otherwise();
    scf::YieldOp::create(b, loc, w);
    return op.getResults();
  }
  ResultRange forLoop(Value lower, Value upper, Value step, ValueRange inits,
                      function_ref<Values(Value, ValueRange)> body) {
    return scf::ForOp::create(b, loc, lower, upper, step, inits,
                              [&](OpBuilder &, Location, Value iv,
                                  ValueRange args) {
                                Values v = body(iv, args);
                                scf::YieldOp::create(b, loc, v);
                              })
        .getResults();
  }
  ResultRange whileLoop(TypeRange types, ValueRange inits,
                        function_ref<std::pair<Value, Values>(ValueRange)>
                            before,
                        function_ref<Values(ValueRange)> after) {
    return scf::WhileOp::create(
               b, loc, types, inits,
               [&](OpBuilder &, Location, ValueRange args) {
                 auto [cond, v] = before(args);
                 scf::ConditionOp::create(b, loc, cond, v);
               },
               [&](OpBuilder &, Location, ValueRange args) {
                 Values v = after(args);
                 scf::YieldOp::create(b, loc, v);
               })
        .getResults();
  }

  /// A private function of `type` at the builder, whose body `body` writes
  /// from its arguments; with no body, a declaration.
  func::FuncOp function(StringRef name, FunctionType type,
                        function_ref<void(ValueRange)> body = nullptr) {
    auto op = func::FuncOp::create(b, loc, name, type);
    op.setPrivate();
    if (body) {
      OpBuilder::InsertionGuard guard(b);
      Block *entry = op.addEntryBlock();
      b.setInsertionPointToStart(entry);
      body(entry->getArguments());
    }
    return op;
  }
};

} // namespace mdir::kernels::detail
