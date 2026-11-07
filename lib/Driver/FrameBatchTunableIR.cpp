#include "FrameBatchTunableIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
bool deviceVector(Value value, Type element) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 1 && type.getLayout().isIdentity() &&
      type.getMemorySpaceAsInt() == 1 && type.getElementType() == element;
}
void launchRows(OpBuilder &builder, Value destination,
                llvm::function_ref<void(OpBuilder &, Value)> body) {
  auto loc = builder.getUnknownLoc();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value threads = arith::ConstantIndexOp::create(builder, loc, 128);
  Value count = memref::DimOp::create(builder, loc, destination, zero);
  Value blocks = arith::CeilDivUIOp::create(builder, loc, count, threads);
  // An empty destination launches no kernel (CUDA forbids a zero grid).
  Value nonempty = arith::CmpIOp::create(builder, loc,
      arith::CmpIPredicate::ugt, count, zero);
  scf::IfOp::create(builder, loc, nonempty, [&](OpBuilder &at, Location) {
    auto launch = gpu::LaunchOp::create(at, loc, blocks, one, one,
        threads, one, one);
    OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
    Value base = arith::MulIOp::create(kernel, loc,
        launch.getBlockIds().x, threads);
    Value row = arith::AddIOp::create(kernel, loc,
        base, launch.getThreadIds().x);
    Value valid = arith::CmpIOp::create(kernel, loc,
        arith::CmpIPredicate::ult, row, count);
    scf::IfOp::create(kernel, loc, valid, [&](OpBuilder &inside, Location) {
      body(inside, row);
      scf::YieldOp::create(inside, loc);
    });
    gpu::TerminatorOp::create(kernel, loc);
    scf::YieldOp::create(at, loc);
  });
}
llvm::Error bad(const char *message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}
}
llvm::Expected<Value> mdir::driver::builder::buildFrameBatchParameterMap(
    OpBuilder &builder, llvm::ArrayRef<int64_t> map, int64_t parameters) {
  if (parameters < 0 || llvm::any_of(map, [&](int64_t slot) {
        return slot < -1 || slot >= parameters;
      }))
    return bad("frame parameter map contains an invalid theta index");
  auto entry = builder.getBlock()->getParentOp();
  auto module = entry->getParentOfType<ModuleOp>();
  if (!module)
    return bad("frame parameter map requires an attached entry function");
  auto loc = builder.getUnknownLoc();
  auto hostType = MemRefType::get({static_cast<int64_t>(map.size())}, builder.getI64Type());
  std::string symbol;
  unsigned serial = 0;
  do { symbol = "__mdir_frame_parameter_map_" + std::to_string(serial++); }
  while (module.lookupSymbol(symbol));
  OpBuilder top(module.getBodyRegion());
  auto global = memref::GlobalOp::create(top, loc, symbol,
      top.getStringAttr("private"), hostType,
      DenseIntElementsAttr::get(RankedTensorType::get(hostType.getShape(),
          builder.getI64Type()), map), true, IntegerAttr());
  Value host = memref::GetGlobalOp::create(builder, loc, hostType, global.getSymName());
  auto deviceType = MemRefType::get(hostType.getShape(), builder.getI64Type(),
      MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(1));
  Value device = gpu::AllocOp::create(builder, loc, deviceType, Type(),
      ValueRange(), ValueRange(), ValueRange()).getMemref();
  Type token = gpu::AsyncTokenType::get(builder.getContext());
  Value start = gpu::WaitOp::create(builder, loc, token, ValueRange()).getAsyncToken();
  Value copied = gpu::MemcpyOp::create(builder, loc, token,
      ValueRange{start}, device, host).getAsyncToken();
  gpu::WaitOp::create(builder, loc, Type(), ValueRange{copied});
  auto dynamic = MemRefType::get({ShapedType::kDynamic}, builder.getI64Type(),
      MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(1));
  return Value(memref::CastOp::create(builder, loc, dynamic, device));
}
llvm::Error mdir::driver::builder::buildFrameBatchMappedParameters(
    OpBuilder &builder, Value theta, Value map, Value destination) {
  auto buffer = dyn_cast<MemRefType>(destination.getType());
  if (!buffer)
    return bad("frame parameter destination must be a memref");
  Type real = buffer.getElementType();
  if (!deviceVector(theta, builder.getF64Type()) ||
      !deviceVector(map, builder.getI64Type()) ||
      !(real.isF32() || real.isF64()) || !deviceVector(destination, real))
    return bad("frame parameter update requires device f64 theta, i64 map "
               "and f32/f64 destination vectors");
  auto loc = builder.getUnknownLoc();
  launchRows(builder, destination, [&](OpBuilder &at, Value row) {
    Value slot = memref::LoadOp::create(at, loc, map, row);
    Value zero = arith::ConstantIntOp::create(at, loc, 0, 64);
    Value taken = arith::CmpIOp::create(at, loc,
        arith::CmpIPredicate::sge, slot, zero);
    scf::IfOp::create(at, loc, taken, [&](OpBuilder &write, Location) {
      Value index = arith::IndexCastOp::create(write, loc,
          write.getIndexType(), slot);
      Value value = memref::LoadOp::create(write, loc, theta, index);
      if (real.isF32())
        value = arith::TruncFOp::create(write, loc, real, value);
      memref::StoreOp::create(write, loc, value, destination, row);
      scf::YieldOp::create(write, loc);
    });
  });
  return llvm::Error::success();
}
llvm::Expected<Value> mdir::driver::builder::buildFrameBatchInitialValues(
    OpBuilder &builder, llvm::ArrayRef<double> values) {
  auto entry = builder.getBlock()->getParentOp();
  auto module = entry->getParentOfType<ModuleOp>();
  if (!module)
    return bad("frame initial values require an attached entry function");
  auto loc = builder.getUnknownLoc();
  auto hostType = MemRefType::get({static_cast<int64_t>(values.size())}, builder.getF64Type());
  std::string symbol;
  unsigned serial = 0;
  do { symbol = "__mdir_frame_initial_values_" + std::to_string(serial++); }
  while (module.lookupSymbol(symbol));
  SmallVector<APFloat> data;
  for (double value : values) data.emplace_back(value);
  OpBuilder top(module.getBodyRegion());
  auto global = memref::GlobalOp::create(top, loc, symbol,
      top.getStringAttr("private"), hostType,
      DenseFPElementsAttr::get(RankedTensorType::get(hostType.getShape(),
          builder.getF64Type()), data), true, IntegerAttr());
  Value host = memref::GetGlobalOp::create(builder, loc, hostType, global.getSymName());
  auto deviceType = MemRefType::get(hostType.getShape(), builder.getF64Type(),
      MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(1));
  Value device = gpu::AllocOp::create(builder, loc, deviceType, Type(),
      ValueRange(), ValueRange(), ValueRange()).getMemref();
  Type token = gpu::AsyncTokenType::get(builder.getContext());
  Value start = gpu::WaitOp::create(builder, loc, token, ValueRange()).getAsyncToken();
  Value copied = gpu::MemcpyOp::create(builder, loc, token,
      ValueRange{start}, device, host).getAsyncToken();
  gpu::WaitOp::create(builder, loc, Type(), ValueRange{copied});
  auto dynamic = MemRefType::get({ShapedType::kDynamic}, builder.getF64Type(),
      MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(1));
  return Value(memref::CastOp::create(builder, loc, dynamic, device));
}
llvm::Error mdir::driver::builder::buildFrameBatchOrderedContraction(
    OpBuilder &builder, Value source, Value offsets, Value indices,
    Value weights, Value destination) {
  if (!deviceVector(source, builder.getF64Type()) ||
      !deviceVector(offsets, builder.getI64Type()) ||
      !deviceVector(indices, builder.getI64Type()) ||
      !deviceVector(weights, builder.getF64Type()) ||
      !deviceVector(destination, builder.getF64Type()))
    return bad("frame contraction requires device f64 values, weights and "
               "destination, with i64 CSR offsets and source indices");
  auto loc = builder.getUnknownLoc();
  launchRows(builder, destination, [&](OpBuilder &at, Value row) {
    Value one = arith::ConstantIndexOp::create(at, loc, 1);
    Value next = arith::AddIOp::create(at, loc, row, one);
    Value begin = memref::LoadOp::create(at, loc, offsets, row);
    Value end = memref::LoadOp::create(at, loc, offsets, next);
    begin = arith::IndexCastOp::create(at, loc, at.getIndexType(), begin);
    end = arith::IndexCastOp::create(at, loc, at.getIndexType(), end);
    Value zero = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0));
    auto sum = scf::ForOp::create(at, loc, begin, end, one, ValueRange{zero},
        [&](OpBuilder &term, Location, Value item, ValueRange carried) {
      Value index = memref::LoadOp::create(term, loc, indices, item);
      index = arith::IndexCastOp::create(term, loc, term.getIndexType(), index);
      Value value = memref::LoadOp::create(term, loc, source, index);
      Value weight = memref::LoadOp::create(term, loc, weights, item);
      Value product = arith::MulFOp::create(term, loc, value, weight);
      Value result = arith::AddFOp::create(term, loc, carried[0], product);
      scf::YieldOp::create(term, loc, result);
    });
    memref::StoreOp::create(at, loc, sum.getResult(0), destination, row);
  });
  return llvm::Error::success();
}
