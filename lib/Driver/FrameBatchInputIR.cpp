#include "FrameBatchInputIR.h"
#include "FrameBatchOutputIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
using namespace mdir::driver::builder;
llvm::Expected<func::FuncOp> mdir::driver::builder::buildFrameBatchParameterReader(OpBuilder &builder) {
  auto entry = builder.getBlock()->getParentOp();
  auto module = entry->getParentOfType<ModuleOp>();
  if (!module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "device frame parameter reader requires an attached entry");
  auto loc = builder.getUnknownLoc();
  auto vector = [&](Type element) {
    return MemRefType::get({ShapedType::kDynamic}, element,
        MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(1));
  };
  auto real = vector(builder.getF64Type()), kinds = vector(builder.getI64Type());
  auto status = MemRefType::get({1}, builder.getI64Type(),
      MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(1));
  std::string name;
  unsigned serial = 0;
  do { name = "__mdir_frame_parameters_" + std::to_string(serial++); }
  while (module.lookupSymbol(name));
  OpBuilder top(entry);
  auto function = func::FuncOp::create(top, loc, name,
      top.getFunctionType({real, real, real, kinds, status}, {real}));
  function.setPrivate();
  OpBuilder at = OpBuilder::atBlockEnd(function.addEntryBlock());
  auto args = function.getArguments();
  Value zero = arith::ConstantIndexOp::create(at, loc, 0);
  Value one = arith::ConstantIndexOp::create(at, loc, 1);
  Value count = memref::DimOp::create(at, loc, args[0], zero);
  Value threads = arith::ConstantIndexOp::create(at, loc, 128);
  Value blocks = arith::CeilDivUIOp::create(at, loc, count, threads);
  Value nonempty = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ugt, count, zero);
  scf::IfOp::create(at, loc, nonempty, [&](OpBuilder &body, Location) {
    auto launch = gpu::LaunchOp::create(body, loc, blocks, one, one, threads, one, one);
    launch->setAttr("mdir.frame_host_arithmetic", body.getUnitAttr());
    OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
    Value base = arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, threads);
    Value index = arith::AddIOp::create(kernel, loc, base, launch.getThreadIds().x);
    Value inside = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult, index, count);
    scf::IfOp::create(kernel, loc, inside, [&](OpBuilder &write, Location) {
      Value value = memref::LoadOp::create(write, loc, args[0], index);
      Value kind = memref::LoadOp::create(write, loc, args[3], index);
      Value no = arith::ConstantIntOp::create(write, loc, 0, 1);
      Value finite = buildFrameBatchIsFinite(write, value);
      Value invalid = arith::CmpIOp::create(write, loc, arith::CmpIPredicate::eq, finite, no);
      Value positiveKind = arith::ConstantIntOp::create(write, loc, 1, 64);
      Value chargeKind = arith::ConstantIntOp::create(write, loc, 2, 64);
      Value positive = arith::CmpIOp::create(write, loc, arith::CmpIPredicate::eq, kind, positiveKind);
      Value charge = arith::CmpIOp::create(write, loc, arith::CmpIPredicate::eq, kind, chargeKind);
      Value nil = arith::ConstantOp::create(write, loc, write.getF64FloatAttr(0));
      Value limit = arith::ConstantOp::create(write, loc, write.getF64FloatAttr(100));
      Value negative = arith::CmpFOp::create(write, loc, arith::CmpFPredicate::OLT, value, nil);
      Value magnitude = math::AbsFOp::create(write, loc, value);
      Value tooLarge = arith::CmpFOp::create(write, loc, arith::CmpFPredicate::OGE, magnitude, limit);
      Value badSign = arith::AndIOp::create(write, loc, positive, negative);
      Value badCharge = arith::AndIOp::create(write, loc, charge, tooLarge);
      invalid = arith::OrIOp::create(write, loc, invalid, badSign);
      invalid = arith::OrIOp::create(write, loc, invalid, badCharge);
      buildFrameBatchFailure(write, invalid, zero, args[4], 7);
      Value valid = arith::CmpIOp::create(write, loc, arith::CmpIPredicate::eq, invalid, no);
      Value initial = memref::LoadOp::create(write, loc, args[2], index);
      Value selected = arith::SelectOp::create(write, loc, valid, value, initial);
      memref::StoreOp::create(write, loc, selected, args[1], index);
      scf::YieldOp::create(write, loc);
    });
    gpu::TerminatorOp::create(kernel, loc);
    scf::YieldOp::create(body, loc);
  });
  func::ReturnOp::create(at, loc, args[1]);
  return function;
}
