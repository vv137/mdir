// Number of 128-thread blocks for a flat PME launch.
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

void DevicePME::emitBlocks(StringRef symbol) {
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(idxType);
  auto fn = func::FuncOp::create(b, loc, symbol,
                                 b.getFunctionType(inputs, {idxType}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value count = entry->getArgument(arg++);
  Value c127 = arith::ConstantIndexOp::create(b, loc, 127);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value padded = arith::AddIOp::create(b, loc, count, c127);
  Value result = arith::DivUIOp::create(b, loc, padded, c128);
  func::ReturnOp::create(b, loc, ValueRange{result});
}
