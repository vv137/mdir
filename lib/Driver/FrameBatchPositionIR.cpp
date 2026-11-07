#include "FrameBatchPositionIR.h"
#include "FrameBatchOutputIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
using namespace mdir::driver::builder;
llvm::Expected<func::FuncOp> mdir::driver::builder::buildFrameBatchPositionReader(
    OpBuilder &builder, MemRefType bytes, Type state, bool acceptsF32, bool hasBaseline) {
  auto entry = builder.getBlock()->getParentOp();
  auto module = entry->getParentOfType<ModuleOp>();
  if (!module || bytes.getRank() != 1 || !bytes.getElementType().isInteger(8) ||
      !bytes.getLayout().isIdentity() || bytes.getMemorySpaceAsInt() != 1 ||
      !(state.isF32() || state.isF64()))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "device frame position reader requires an attached entry and contiguous device bytes");
  auto loc = builder.getUnknownLoc();
  auto rowType = MemRefType::get({ShapedType::kDynamic, 3}, state,
      MemRefLayoutAttrInterface(), bytes.getMemorySpace());
  auto baselineType = MemRefType::get({ShapedType::kDynamic}, builder.getF64Type(),
      MemRefLayoutAttrInterface(), bytes.getMemorySpace());
  auto statusType = MemRefType::get({1}, builder.getI64Type(),
      MemRefLayoutAttrInterface(), bytes.getMemorySpace());
  std::string name;
  unsigned serial = 0;
  do { name = "__mdir_frame_positions_" + std::to_string(serial++); }
  while (module.lookupSymbol(name));
  OpBuilder top(entry);
  auto reader = func::FuncOp::create(top, loc, name,
      top.getFunctionType({bytes, top.getIndexType(), top.getIndexType(),
                          top.getI64Type(), bytes, baselineType, statusType}, {rowType}));
  reader.setPrivate();
  OpBuilder at = OpBuilder::atBlockEnd(reader.addEntryBlock());
  auto args = reader.getArguments();
  Value zero = arith::ConstantIndexOp::create(at, loc, 0);
  Value one = arith::ConstantIndexOp::create(at, loc, 1);
  auto view = [&](OpBuilder &body, Type element, Value source, Value row) -> Value {
    Value width = arith::ConstantIndexOp::create(body, loc, 3 * (element.isF64() ? 8 : 4));
    Value stride = arith::MulIOp::create(body, loc, args[2], width);
    Value offset = arith::MulIOp::create(body, loc, row, stride);
    auto type = MemRefType::get({ShapedType::kDynamic, 3}, element,
        MemRefLayoutAttrInterface(), bytes.getMemorySpace());
    return memref::ViewOp::create(body, loc, type, source, offset, ValueRange{args[2]});
  };
  Value destination = view(at, state, args[4], zero);
  auto copy = [&](OpBuilder &body, Type input) {
    Value source = view(body, input, args[0], args[1]);
    Value threads = arith::ConstantIndexOp::create(body, loc, 128);
    Value blocks = arith::CeilDivUIOp::create(body, loc, args[2], threads);
    Value nonempty = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ugt, args[2], zero);
    scf::IfOp::create(body, loc, nonempty, [&](OpBuilder &work, Location) {
      auto launch = gpu::LaunchOp::create(work, loc, blocks, one, one, threads, one, one);
      launch->setAttr("mdir.frame_host_arithmetic", work.getUnitAttr());
      OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
      Value base = arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, threads);
      Value particle = arith::AddIOp::create(kernel, loc, base, launch.getThreadIds().x);
      Value inside = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult, particle, args[2]);
      scf::IfOp::create(kernel, loc, inside, [&](OpBuilder &write, Location) {
        Value three = arith::ConstantIndexOp::create(write, loc, 3);
        Value first = arith::MulIOp::create(write, loc, particle, three);
        Value no = arith::ConstantIntOp::create(write, loc, 0, 1);
        for (unsigned c = 0; c != 3; ++c) {
          Value column = arith::ConstantIndexOp::create(write, loc, c);
          Value value = memref::LoadOp::create(write, loc, source, ValueRange{particle, column});
          Value valid = buildFrameBatchIsFinite(write, value);
          Value invalid = arith::CmpIOp::create(write, loc, arith::CmpIPredicate::eq, valid, no);
          buildFrameBatchFailure(write, invalid, args[1], args[6], 1);
          if (input.isF32() && state.isF64()) value = arith::ExtFOp::create(write, loc, state, value);
          Value fallback;
          if (hasBaseline) {
            Value index = arith::AddIOp::create(write, loc, first, column);
            fallback = memref::LoadOp::create(write, loc, args[5], index);
            if (state.isF32()) fallback = arith::TruncFOp::create(write, loc, state, fallback);
          } else {
            fallback = arith::ConstantOp::create(write, loc, write.getFloatAttr(state, 0));
          }
          Value safe = arith::SelectOp::create(write, loc, valid, value, fallback);
          memref::StoreOp::create(write, loc, safe, destination, ValueRange{particle, column});
        }
        scf::YieldOp::create(write, loc);
      });
      gpu::TerminatorOp::create(kernel, loc);
      scf::YieldOp::create(work, loc);
    });
  };
  if (acceptsF32) {
    Value four = arith::ConstantIntOp::create(at, loc, 4, 64);
    Value narrow = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, args[3], four);
    scf::IfOp::create(at, loc, narrow, [&](OpBuilder &yes, Location) {
      copy(yes, yes.getF32Type());
      scf::YieldOp::create(yes, loc);
    }, [&](OpBuilder &no, Location) {
      copy(no, state);
      scf::YieldOp::create(no, loc);
    });
  } else {
    copy(at, state);
  }
  func::ReturnOp::create(at, loc, destination);
  return reader;
}
