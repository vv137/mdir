// Entry signatures retain the host's buffer and scalar order.
#include "EntryBoundaryIR.h"
#include "mdir/Driver/Builder.h"
#include "mlir/IR/Builders.h"
#include <cassert>
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

llvm::Expected<func::FuncOp> mdir::driver::builder::buildEntryFunction(
    OpBuilder &builder, const Program &program, const EntryBoundaryConfigIR &config,
    llvm::function_ref<llvm::Error(OpBuilder &, const EntryArgumentsIR &)> body) {
  auto loc = builder.getUnknownLoc();
  auto real = [&](Element element) -> Type {
    return element == Element::F64 ? Type(builder.getF64Type()) : Type(builder.getF32Type());
  };
  auto buffer = [&](ArrayRef<int64_t> shape, Type element) -> Type {
    return MemRefType::get(shape, element);
  };
  const int64_t dynamic = ShapedType::kDynamic;
  SmallVector<Type> types;
  SmallVector<bool> readOnly;
  auto append = [&](Type type, bool immutable = false) {
    types.push_back(type);
    readOnly.push_back(immutable);
  };
  append(buffer({dynamic, 3}, real(program.state)));
  append(buffer({dynamic, 3}, real(program.state)));
  if (program.takesForces) append(buffer({dynamic, 3}, real(program.force)));
  append(buffer({dynamic}, real(program.mass)), true);
  for (const auto &field : program.fields)
    append(buffer({dynamic}, field.isInteger ? Type(builder.getI32Type()) : real(program.parameter)), true);
  for (size_t slot = 0; slot != program.tables.size(); ++slot)
    append(buffer({dynamic, dynamic}, builder.getF64Type()), true);
  for (const auto &tuple : program.tupleSets) {
    append(buffer({dynamic, tuple.arity}, builder.getI32Type()), true);
    for (size_t slot = 0; slot != tuple.fields.size(); ++slot)
      append(buffer({dynamic}, builder.getF64Type()), true);
  }
  append(buffer({dynamic}, builder.getI32Type()));
  for (unsigned index = 0; index != 4; ++index) append(builder.getF64Type());
  append(builder.getI64Type());
  if (config.segmentedMinimization) append(builder.getF64Type());
  if (config.branchesStart) append(builder.getI64Type());
  for (size_t slot = 0; slot != program.startValues.size(); ++slot) append(builder.getF64Type());
  auto function = func::FuncOp::create(builder, loc, program.entry,
      builder.getFunctionType(types, {}));
  for (auto [index, immutable] : llvm::enumerate(readOnly))
    if (immutable) function.setArgAttr(index, "mdrt.read_only", builder.getUnitAttr());
  OpBuilder::InsertionGuard guard(builder);
  auto *block = function.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  auto arguments = block->getArguments();
  size_t index = 0;
  EntryArgumentsIR point;
  point.positions = arguments[index++];
  point.velocities = arguments[index++];
  if (program.takesForces) point.forces = arguments[index++];
  point.masses = arguments[index++];
  for (size_t slot = 0; slot != program.fields.size(); ++slot) point.fields.push_back(arguments[index++]);
  for (size_t slot = 0; slot != program.tables.size(); ++slot) point.tables.push_back(arguments[index++]);
  for (const auto &tuple : program.tupleSets) {
    EntryTupleBuffersIR buffers;
    buffers.members = arguments[index++];
    for (size_t slot = 0; slot != tuple.fields.size(); ++slot) buffers.fields.push_back(arguments[index++]);
    point.tuples.push_back(std::move(buffers));
  }
  point.identities = arguments[index++];
  point.lengthX = arguments[index++];
  point.lengthY = arguments[index++];
  point.lengthZ = arguments[index++];
  point.interval = arguments[index++];
  point.start = arguments[index++];
  if (config.segmentedMinimization) point.firstSize = arguments[index++];
  if (config.branchesStart) point.firstCall = arguments[index++];
  for (size_t slot = 0; slot != program.startValues.size(); ++slot) point.startValues.push_back(arguments[index++]);
  assert(index == arguments.size() && "entry metadata and actual arguments must agree");
  if (llvm::Error error = body(builder, point)) {
    function.erase();
    return std::move(error);
  }
  func::ReturnOp::create(builder, loc);
  return function;
}
