// Recover the real grid from the fixed-point sum (D70).
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

void DevicePME::emitReal(StringRef symbol) {
  StringRef blockFunction = "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic}, i64Type));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value grid = entry->getArgument(arg++);
  Value realBuffer = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value points = memref::DimOp::create(b, loc, grid, c0);
  Value blocks_points =
      func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{points})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_points, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, c128);
    Value item = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, points);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value inverse = arith::ConstantOp::create(
          b, loc, b.getFloatAttr(realType, 9.094947017729282e-13));
      Value fixed = load(grid, {item});
      Value value = sitofp(realType, fixed);
      Value scaled = mulf(value, inverse);
      store(scaled, realBuffer, {item});
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}
