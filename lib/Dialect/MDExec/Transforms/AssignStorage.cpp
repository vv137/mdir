// Assignment of buffers to fields.
//
// The pass converts the ops of md_exec from the value form to the storage
// form: it gives every field a buffer and makes the loops update the buffers
// where they are. See docs/ops-m0.md, Section 10.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include <cassert>

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace {

/// The buffers of one region: the body of a function or of a loop.
///
/// A buffer that the region owns may be overwritten once the field it holds
/// is dead. Buffers that hold no live field wait in the pool.
///
/// The body of a loop obtains the buffers it lacks as additional
/// loop-carried values, so that the roles of the buffers can rotate from one
/// iteration to the next.
struct Scope {
  Scope(Scope *parent, MLIRContext *context)
      : parent(parent), builder(context) {}

  bool owns(Value buffer) const { return owned.contains(buffer); }

  bool isDeadAfter(Value value, unsigned position) const {
    auto found = lastUse.find(value);
    return found == lastUse.end() || found->second == position;
  }

  void release(Value buffer) { pool[buffer.getType()].push_back(buffer); }

  /// A buffer of type `type` that holds no live field. `size` is the number
  /// of particles.
  Value request(MemRefType type, Value size, Location loc);

  Scope *parent;
  OpBuilder builder;

  /// The body of the loop, for the scope of a loop.
  Block *body = nullptr;

  llvm::DenseSet<Value> owned;
  llvm::MapVector<Type, SmallVector<Value, 2>> pool;

  /// The additional loop-carried buffers, and the buffers that they are
  /// initialized with.
  SmallVector<Value> extraArguments;
  SmallVector<Value> extraInits;

  /// For every value that is defined in the region, the position of the
  /// last op of the region that uses it.
  llvm::DenseMap<Value, unsigned> lastUse;
};

class Assignment {
public:
  explicit Assignment(ModuleOp module)
      : module(module), context(module.getContext()) {}

  LogicalResult run();

private:
  LogicalResult convertFunction(func::FuncOp function);
  LogicalResult convertBlock(Block &block, Scope &scope);
  LogicalResult convertOp(Operation *op, Scope &scope, unsigned position);

  LogicalResult convertFor(scf::ForOp op, Scope &scope, unsigned position);
  LogicalResult convertYield(scf::YieldOp op, Scope &scope);
  LogicalResult convertBuildNeighbors(BuildNeighborsOp op, Scope &scope);
  LogicalResult convertRefreshNeighbors(RefreshNeighborsOp op, Scope &scope,
                                        unsigned position);
  LogicalResult convertParticleFor(ParticleForOp op, Scope &scope,
                                   unsigned position);
  LogicalResult convertPairFor(PairForOp op, Scope &scope,
                               unsigned position);
  LogicalResult convertGeneric(Operation *op, Scope &scope);

  /// Chooses the buffer that a loop writes the field `destination` to.
  /// `readFields` are the fields that the loop reads; one of them may give
  /// up its buffer if `inPlace` is set. `accumulates` is set if the loop
  /// must add to what the buffer holds.
  LogicalResult chooseDestination(Operation *op, Value destination,
                                  ValueRange readFields, bool inPlace,
                                  Scope &scope, unsigned position,
                                  Value &buffer, bool &accumulates);

  /// Gives `loop`, a loop in the storage form without a kernel, a copy of
  /// the kernel `source`.
  void copyKernel(Operation *loop, Block &source);

  /// The buffer that holds the field `field`.
  LogicalResult getBuffer(Value field, Scope &scope, Value &buffer);

  /// The number of particles of the set that `type`, the type of a field or
  /// of a neighbor structure, belongs to.
  LogicalResult getSize(Operation *op, Type type, Value &size);

  /// Records the buffer of a field and, if it is the first of its particle
  /// set, the number of particles.
  void bind(Value field, Value buffer, Scope &scope);

  /// The storage of the neighbor structure `structure`. An empty structure
  /// has none until it is used; each use gets storage of its own.
  /// `positions` is the type of the field of the positions that the
  /// structure is built at.
  LogicalResult getNeighbors(Operation *op, Value structure, Type positions,
                             Scope &scope, Value &storage);

  ModuleOp module;
  MLIRContext *context;

  /// Values other than fields: the value in the new code.
  IRMapping mapping;
  /// Fields: the buffer that holds the field.
  llvm::DenseMap<Value, Value> buffers;
  /// Neighbor structures: their storage.
  llvm::DenseMap<Value, Value> neighbors;
  /// Particle sets: the number of particles.
  llvm::DenseMap<Attribute, Value> sizes;

  /// The scope of the body of the function that is being converted.
  Scope *root = nullptr;
};

} // namespace

static bool isField(Type type) { return isa<md::FieldType>(type); }

/// Returns true if `type` is a type that belongs to the value form or to the
/// structures of the runtime, other than a neighbor structure.
static bool isUnsupported(Type type) {
  return isa<mdrt::CellsType, mdrt::PermutationType>(type);
}

static FlatSymbolRefAttr getParticleSet(Type type) {
  if (auto field = dyn_cast<md::FieldType>(type))
    return field.getParticleSet();
  return cast<mdrt::NeighborsType>(type).getParticleSet();
}

/// The type of the positions that the neighbor structure `structure` is
/// refreshed or traversed with, from the ops that use it.
static Type findPositionsType(Value structure) {
  for (OpOperand &use : structure.getUses()) {
    Operation *user = use.getOwner();
    if (auto refresh = dyn_cast<RefreshNeighborsOp>(user))
      return refresh.getPositions().getType();
    if (auto loop = dyn_cast<PairForOp>(user))
      return loop.getPositions().getType();
    if (auto loop = dyn_cast<scf::ForOp>(user)) {
      for (auto [init, argument] :
           llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs()))
        if (init == structure)
          if (Type type = findPositionsType(argument))
            return type;
    }
  }
  return Type();
}

//===----------------------------------------------------------------------===//
// Scope
//===----------------------------------------------------------------------===//

Value Scope::request(MemRefType type, Value size, Location loc) {
  auto found = pool.find(type);
  if (found != pool.end() && !found->second.empty())
    return found->second.pop_back_val();

  Value buffer;
  if (parent) {
    extraInits.push_back(parent->request(type, size, loc));
    buffer = body->addArgument(type, loc);
    extraArguments.push_back(buffer);
  } else {
    buffer = memref::AllocOp::create(builder, loc, type, ValueRange{size});
  }
  owned.insert(buffer);
  return buffer;
}

//===----------------------------------------------------------------------===//
// Buffers
//===----------------------------------------------------------------------===//

void Assignment::bind(Value field, Value buffer, Scope &scope) {
  buffers[field] = buffer;
  Attribute set = getParticleSet(field.getType());
  if (sizes.count(set))
    return;
  Value zero =
      arith::ConstantIndexOp::create(scope.builder, field.getLoc(), 0);
  sizes[set] =
      memref::DimOp::create(scope.builder, field.getLoc(), buffer, zero);
}

LogicalResult Assignment::getSize(Operation *op, Type type, Value &size) {
  Attribute set = getParticleSet(type);
  size = sizes.lookup(set);
  if (!size)
    return op->emitOpError()
           << "the number of particles of " << set
           << " is not known here: no field of the set has a buffer yet";
  return success();
}

LogicalResult Assignment::getBuffer(Value field, Scope &scope,
                                    Value &buffer) {
  buffer = buffers.lookup(field);
  if (buffer)
    return success();

  // A destination that no loop has taken.
  Operation *op = field.getDefiningOp();
  if (!op || !isa<EmptyOp, ZerosOp>(op))
    return emitError(field.getLoc()) << "a field has no buffer";

  Value size;
  if (failed(getSize(op, field.getType(), size)))
    return failure();
  auto fieldType = cast<md::FieldType>(field.getType());
  buffer = scope.request(mdrt::getBufferType(fieldType), size, op->getLoc());
  buffers[field] = buffer;

  if (isa<ZerosOp>(op)) {
    OpBuilder &builder = scope.builder;
    Location loc = op->getLoc();
    auto fill = ParticleForOp::create(builder, loc, TypeRange(), ValueRange(),
                                      ValueRange{buffer}, ValueRange());
    Block *block = new Block();
    fill.getKernel().push_back(block);
    OpBuilder kernel = OpBuilder::atBlockEnd(block);
    Type valueType = fieldType.getKernelValueType();
    Value zero = arith::ConstantOp::create(
        kernel, loc, valueType,
        cast<TypedAttr>(kernel.getZeroAttr(valueType)));
    YieldOp::create(kernel, loc, ValueRange{zero});
  }
  return success();
}

LogicalResult Assignment::getNeighbors(Operation *op, Value structure,
                                       Type positions, Scope &scope,
                                       Value &storage) {
  storage = neighbors.lookup(structure);
  if (storage)
    return success();

  auto empty = structure.getDefiningOp<EmptyNeighborsOp>();
  if (!empty)
    return op->emitOpError() << "the neighbor structure has no storage";

  Value size;
  if (failed(getSize(op, structure.getType(), size)))
    return failure();
  if (!positions)
    return op->emitOpError()
           << "cannot tell the type of the positions that the neighbor "
              "structure is built at: nothing refreshes or traverses it";

  // The storage outlives the iterations of any loop around the structure,
  // so it is allocated in the body of the function.
  Type element = cast<md::FieldType>(positions).getElementType();
  storage = EmptyNeighborsOp::create(
      root->builder, empty.getLoc(), structure.getType(), size,
      TypeAttr::get(element), empty.getKindAttr(), empty.getWidthAttr());
  // Inside a loop, every iteration begins with an empty structure.
  if (&scope != root)
    ResetNeighborsOp::create(scope.builder, empty.getLoc(), storage);
  return success();
}

//===----------------------------------------------------------------------===//
// Destinations
//===----------------------------------------------------------------------===//

LogicalResult Assignment::chooseDestination(Operation *op, Value destination,
                                            ValueRange readFields,
                                            bool inPlace, Scope &scope,
                                            unsigned position, Value &buffer,
                                            bool &accumulates) {
  Operation *definition = destination.getDefiningOp();
  bool isEmpty = definition && isa<EmptyOp>(definition);
  bool isZeros = definition && isa<ZerosOp>(definition);
  MemRefType type =
      mdrt::getBufferType(cast<md::FieldType>(destination.getType()));

  // A destination that holds a field: the loop continues in its buffer.
  if (buffers.count(destination) || (!isEmpty && !isZeros)) {
    if (failed(getBuffer(destination, scope, buffer)))
      return failure();
    if (!scope.owns(buffer) || !scope.isDeadAfter(destination, position))
      return op->emitOpError()
             << "needs a buffer of its own: the loop writes to a field that "
                "is used afterward or that belongs to an enclosing region";
    for (Value field : readFields)
      if (buffers.lookup(field) == buffer && !inPlace)
        return op->emitOpError() << "needs a buffer of its own: the loop "
                                    "writes to a field that it reads from "
                                    "other particles";
    accumulates = true;
    return success();
  }

  accumulates = false;

  // A loop over particles reads and writes the same particle, so it may
  // write to the buffer of a field that it reads and that is dead afterward.
  if (inPlace) {
    for (Value field : readFields) {
      Value candidate = buffers.lookup(field);
      if (!candidate || candidate.getType() != type ||
          !scope.owns(candidate) || !scope.isDeadAfter(field, position))
        continue;
      // The buffer now holds the result. Forget that it held the field.
      buffers.erase(field);
      buffer = candidate;
      return success();
    }
  }

  Value size;
  if (failed(getSize(op, destination.getType(), size)))
    return failure();
  buffer = scope.request(type, size, op->getLoc());
  return success();
}

//===----------------------------------------------------------------------===//
// Loops over particles and pairs
//===----------------------------------------------------------------------===//

void Assignment::copyKernel(Operation *loop, Block &source) {
  Block *block = new Block();
  loop->getRegion(0).push_back(block);

  IRMapping local = mapping;
  for (BlockArgument argument : source.getArguments())
    local.map(argument,
              block->addArgument(argument.getType(), argument.getLoc()));
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  for (Operation &nested : source)
    kernel.clone(nested, local);
}

LogicalResult Assignment::convertParticleFor(ParticleForOp op, Scope &scope,
                                             unsigned position) {
  // The buffers of the fields are looked up before a destination takes one
  // of them over.
  SmallVector<Value> ins;
  for (Value field : op.getIns()) {
    Value buffer;
    if (failed(getBuffer(field, scope, buffer)))
      return failure();
    ins.push_back(buffer);
  }

  SmallVector<Value> outs;
  for (Value destination : op.getOuts()) {
    Value buffer;
    bool accumulates;
    if (failed(chooseDestination(op, destination, op.getIns(),
                                 /*inPlace=*/true, scope, position, buffer,
                                 accumulates)))
      return failure();
    outs.push_back(buffer);
  }

  SmallVector<Value> reduce;
  SmallVector<Type> resultTypes;
  for (Value value : op.getReduce()) {
    reduce.push_back(mapping.lookup(value));
    resultTypes.push_back(value.getType());
  }

  auto loop = ParticleForOp::create(scope.builder, op.getLoc(), resultTypes,
                                    ins, outs, reduce);
  copyKernel(loop, op.getKernel().front());

  unsigned numOuts = outs.size();
  for (unsigned i = 0; i != numOuts; ++i) {
    buffers[op.getResult(i)] = outs[i];
    scope.owned.insert(outs[i]);
  }
  for (unsigned i = numOuts, e = op.getNumResults(); i != e; ++i)
    mapping.map(op.getResult(i), loop.getResult(i - numOuts));
  return success();
}

LogicalResult Assignment::convertPairFor(PairForOp op, Scope &scope,
                                         unsigned position) {
  Value storage;
  if (failed(getNeighbors(op, op.getNeighbors(),
                          op.getPositions().getType(), scope, storage)))
    return failure();

  Value positions;
  if (failed(getBuffer(op.getPositions(), scope, positions)))
    return failure();

  SmallVector<Value> ins;
  for (Value field : op.getIns()) {
    Value buffer;
    if (failed(getBuffer(field, scope, buffer)))
      return failure();
    ins.push_back(buffer);
  }

  // The loop reads the positions and the fields in `ins` of other
  // particles, so it cannot write to any of their buffers.
  SmallVector<Value> readFields(op.getIns().begin(), op.getIns().end());
  readFields.push_back(op.getPositions());

  SmallVector<Value> outs;
  SmallVector<bool> overwrite;
  for (Value destination : op.getOuts()) {
    Value buffer;
    bool accumulates;
    if (failed(chooseDestination(op, destination, readFields,
                                 /*inPlace=*/false, scope, position, buffer,
                                 accumulates)))
      return failure();
    outs.push_back(buffer);
    overwrite.push_back(!accumulates);
  }

  SmallVector<Value> reduce;
  SmallVector<Type> resultTypes;
  for (Value value : op.getReduce()) {
    reduce.push_back(mapping.lookup(value));
    resultTypes.push_back(value.getType());
  }

  OpBuilder &builder = scope.builder;
  DenseBoolArrayAttr overwriteAttr;
  if (llvm::is_contained(overwrite, true))
    overwriteAttr = builder.getDenseBoolArrayAttr(overwrite);

  auto loop = PairForOp::create(
      builder, op.getLoc(), resultTypes, storage, positions,
      mapping.lookup(op.getCell()), ins, outs, reduce, op.getCutoffAttr(),
      op.getWeightsAttr(), overwriteAttr, op.getTraversalAttr(),
      op.getConflictAttr());
  copyKernel(loop, op.getKernel().front());

  unsigned numOuts = outs.size();
  for (unsigned i = 0; i != numOuts; ++i) {
    buffers[op.getResult(i)] = outs[i];
    scope.owned.insert(outs[i]);
  }
  for (unsigned i = numOuts, e = op.getNumResults(); i != e; ++i)
    mapping.map(op.getResult(i), loop.getResult(i - numOuts));
  return success();
}

//===----------------------------------------------------------------------===//
// Neighbor structures
//===----------------------------------------------------------------------===//

LogicalResult Assignment::convertBuildNeighbors(BuildNeighborsOp op,
                                                Scope &scope) {
  Location loc = op.getLoc();
  auto cells = op.getCells().getDefiningOp<BuildCellsOp>();
  if (!cells)
    return op.emitOpError() << "expected cells that are the result of "
                               "'md_exec.build_cells'";

  Value positions;
  if (failed(getBuffer(op.getPositions(), scope, positions)))
    return failure();
  Value size;
  if (failed(getSize(op, op.getPositions().getType(), size)))
    return failure();

  // Storage, and a refresh that builds whatever the storage holds. The
  // count of builds is that of a structure that was built once.
  Type element =
      cast<md::FieldType>(op.getPositions().getType()).getElementType();
  Value storage = EmptyNeighborsOp::create(
      root->builder, loc, op.getResult().getType(), size,
      TypeAttr::get(element), op.getKindAttr(), op.getWidthAttr());
  if (&scope != root)
    ResetNeighborsOp::create(scope.builder, loc, storage);
  auto refresh = RefreshNeighborsOp::create(
      scope.builder, loc, storage.getType(), storage, positions,
      mapping.lookup(op.getCell()), op.getCutoffAttr(), op.getSkinAttr(),
      cells.getWidthAttr());
  refresh.setPolicy(RebuildPolicy::Always);
  neighbors[op.getResult()] = refresh.getResult();
  return success();
}

LogicalResult Assignment::convertRefreshNeighbors(RefreshNeighborsOp op,
                                                  Scope &scope,
                                                  unsigned position) {
  // The structure is refreshed where it is, so nothing may use what it was
  // before.
  Value old = op.getNeighbors();
  if (!old.getDefiningOp<EmptyNeighborsOp>() &&
      !scope.isDeadAfter(old, position))
    return op.emitOpError()
           << "needs storage of its own: the neighbor structure is used "
              "after it is refreshed";

  Value storage;
  if (failed(getNeighbors(op, old, op.getPositions().getType(), scope,
                          storage)))
    return failure();
  Value positions;
  if (failed(getBuffer(op.getPositions(), scope, positions)))
    return failure();

  auto refresh = RefreshNeighborsOp::create(
      scope.builder, op.getLoc(), storage.getType(), storage, positions,
      mapping.lookup(op.getCell()), op.getCutoffAttr(), op.getSkinAttr(),
      op.getCellWidthAttr(), op.getPolicyAttr());
  neighbors[op.getResult()] = refresh.getResult();
  return success();
}

//===----------------------------------------------------------------------===//
// Loops that carry fields
//===----------------------------------------------------------------------===//

LogicalResult Assignment::convertFor(scf::ForOp op, Scope &scope,
                                     unsigned position) {
  Location loc = op.getLoc();
  Block &oldBody = *op.getBody();

  Scope inner(&scope, context);
  inner.body = new Block();
  inner.builder.setInsertionPointToEnd(inner.body);

  mapping.map(op.getInductionVar(),
              inner.body->addArgument(op.getInductionVar().getType(), loc));

  // A neighbor structure is refreshed where it is, so the loop need not
  // carry its storage. `carried[i]` is the position of loop-carried value
  // `i` among the values that the new loop carries, or -1.
  SmallVector<int> carried;
  SmallVector<Value> inits;
  SmallVector<Value> structures;
  for (auto [init, argument] :
       llvm::zip(op.getInitArgs(), op.getRegionIterArgs())) {
    if (isa<mdrt::NeighborsType>(init.getType())) {
      Value storage;
      if (failed(getNeighbors(op, init, findPositionsType(argument), scope,
                              storage)))
        return failure();
      neighbors[argument] = storage;
      structures.push_back(storage);
      carried.push_back(-1);
      continue;
    }
    carried.push_back(inits.size());
    if (!isField(init.getType())) {
      if (isUnsupported(init.getType()))
        return op.emitOpError()
               << "cannot carry a value of type " << init.getType();
      inits.push_back(mapping.lookup(init));
      mapping.map(argument,
                  inner.body->addArgument(argument.getType(), loc));
      continue;
    }

    Value buffer;
    if (failed(getBuffer(init, scope, buffer)))
      return failure();
    if (!scope.owns(buffer) || !scope.isDeadAfter(init, position))
      return op.emitOpError()
             << "needs a buffer of its own: the loop updates a field that is "
                "used after the loop or that belongs to an enclosing region";
    scope.owned.erase(buffer);
    inits.push_back(buffer);

    Value inside = inner.body->addArgument(buffer.getType(), loc);
    buffers[argument] = inside;
    inner.owned.insert(inside);
  }

  if (failed(convertBlock(oldBody, inner)))
    return failure();

  OperationState state(loc, scf::ForOp::getOperationName());
  state.addOperands({mapping.lookup(op.getLowerBound()),
                     mapping.lookup(op.getUpperBound()),
                     mapping.lookup(op.getStep())});
  state.addOperands(inits);
  state.addOperands(inner.extraInits);
  for (Value value : inits)
    state.addTypes(value.getType());
  for (Value value : inner.extraInits)
    state.addTypes(value.getType());
  state.addRegion()->push_back(inner.body);
  Operation *loop = scope.builder.create(state);

  unsigned numCarried = inits.size();
  unsigned structure = 0;
  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
    Value oldResult = op.getResult(i);
    if (carried[i] < 0) {
      neighbors[oldResult] = structures[structure++];
      continue;
    }
    Value newResult = loop->getResult(carried[i]);
    if (isField(oldResult.getType())) {
      buffers[oldResult] = newResult;
      scope.owned.insert(newResult);
    } else {
      mapping.map(oldResult, newResult);
    }
  }
  // The buffers that the loop borrowed hold no field when it ends.
  for (unsigned i = numCarried, e = loop->getNumResults(); i != e; ++i) {
    scope.owned.insert(loop->getResult(i));
    scope.release(loop->getResult(i));
  }
  return success();
}

LogicalResult Assignment::convertYield(scf::YieldOp op, Scope &scope) {
  Location loc = op.getLoc();
  SmallVector<Value> values;
  llvm::DenseSet<Value> held;

  for (Value value : op.getOperands()) {
    if (isa<mdrt::NeighborsType>(value.getType())) {
      if (!neighbors.count(value))
        return op.emitOpError() << "a neighbor structure has no storage";
      continue;
    }
    if (!isField(value.getType())) {
      values.push_back(mapping.lookup(value));
      continue;
    }
    Value buffer;
    if (failed(getBuffer(value, scope, buffer)))
      return failure();
    if (!scope.owns(buffer))
      return op.emitOpError()
             << "needs a buffer of its own: the loop yields a field that "
                "belongs to an enclosing region";
    if (!held.insert(buffer).second)
      return op.emitOpError()
             << "needs a buffer of its own: the loop yields one field twice";
    values.push_back(buffer);
  }

  // Hand back as many unused buffers as the loop borrowed.
  for (Value extra : scope.extraArguments) {
    SmallVector<Value, 2> &available = scope.pool[extra.getType()];
    assert(!available.empty() && "the buffers of a loop are conserved");
    values.push_back(available.pop_back_val());
  }

  scf::YieldOp::create(scope.builder, loc, values);
  return success();
}

//===----------------------------------------------------------------------===//
// Other ops
//===----------------------------------------------------------------------===//

LogicalResult Assignment::convertGeneric(Operation *op, Scope &scope) {
  bool touchesFields = false;
  op->walk([&](Operation *nested) {
    for (Type type : nested->getOperandTypes())
      touchesFields |= isField(type) || isUnsupported(type) ||
                       isa<mdrt::NeighborsType>(type);
    for (Type type : nested->getResultTypes())
      touchesFields |= isField(type) || isUnsupported(type) ||
                       isa<mdrt::NeighborsType>(type);
  });
  if (touchesFields)
    return op->emitOpError()
           << "cannot be given storage: the op is not known to the pass and "
              "uses fields or runtime structures";
  scope.builder.clone(*op, mapping);
  return success();
}

/// Verifies that the field type `field` is stored in the buffer type
/// `buffer`. Converting between the two would be a copy that nothing asked
/// for.
static LogicalResult checkStored(Operation *op, Type field, Type buffer) {
  if (mdrt::getBufferType(cast<md::FieldType>(field)) == buffer)
    return success();
  return op->emitOpError()
         << "the field has the type " << field
         << ", which is not the type that the buffer stores; run "
            "'md-exec-assign-precision' first";
}

LogicalResult Assignment::convertOp(Operation *op, Scope &scope,
                                    unsigned position) {
  OpBuilder &builder = scope.builder;

  if (auto from = dyn_cast<mdrt::FromBufferOp>(op)) {
    if (failed(checkStored(op, from.getResult().getType(),
                           from.getBuffer().getType())))
      return failure();
    Value buffer = mapping.lookup(from.getBuffer());
    bind(from.getResult(), buffer, scope);
    scope.owned.insert(buffer);
    return success();
  }
  if (auto to = dyn_cast<mdrt::ToBufferOp>(op)) {
    if (failed(checkStored(op, to.getField().getType(),
                           to.getResult().getType())))
      return failure();
    Value buffer;
    if (failed(getBuffer(to.getField(), scope, buffer)))
      return failure();
    // The buffer is visible outside from here on. Leave it alone.
    scope.owned.erase(buffer);
    mapping.map(to.getResult(), buffer);
    return success();
  }

  // Destinations get their buffer from the loop that writes to them, and an
  // empty neighbor structure its storage from what uses it. Cells are built
  // together with the neighbor structure.
  if (isa<EmptyOp, ZerosOp, BuildCellsOp>(op))
    return success();
  if (auto empty = dyn_cast<EmptyNeighborsOp>(op)) {
    if (empty.isStorageForm())
      return op->emitOpError() << "is in the storage form already";
    return success();
  }

  if (auto build = dyn_cast<BuildNeighborsOp>(op))
    return convertBuildNeighbors(build, scope);
  if (auto refresh = dyn_cast<RefreshNeighborsOp>(op))
    return convertRefreshNeighbors(refresh, scope, position);
  if (auto count = dyn_cast<RebuildCountOp>(op)) {
    Value storage;
    if (failed(getNeighbors(op, count.getNeighbors(),
                            findPositionsType(count.getNeighbors()), scope,
                            storage)))
      return failure();
    mapping.map(count.getResult(),
                RebuildCountOp::create(builder, op->getLoc(),
                                       builder.getI64Type(), storage));
    return success();
  }
  if (auto loop = dyn_cast<ParticleForOp>(op))
    return convertParticleFor(loop, scope, position);
  if (auto loop = dyn_cast<PairForOp>(op))
    return convertPairFor(loop, scope, position);

  if (auto loop = dyn_cast<scf::ForOp>(op))
    return convertFor(loop, scope, position);
  if (auto yield = dyn_cast<scf::YieldOp>(op))
    if (scope.body)
      return convertYield(yield, scope);

  if (auto ret = dyn_cast<func::ReturnOp>(op)) {
    SmallVector<Value> values;
    for (Value value : ret.getOperands()) {
      if (!isField(value.getType())) {
        values.push_back(mapping.lookup(value));
        continue;
      }
      Value buffer;
      if (failed(getBuffer(value, scope, buffer)))
        return failure();
      values.push_back(buffer);
    }
    func::ReturnOp::create(builder, op->getLoc(), values);
    return success();
  }

  bool isCell = isa<md::OrthorhombicCellOp>(op);
  if (!isCell && (isa<md::MDDialect>(op->getDialect()) ||
                  isa<MDExecDialect>(op->getDialect()) ||
                  op->getName().getDialectNamespace() == "dyn"))
    return op->emitOpError()
           << "cannot be given storage; run 'md-differentiate', "
              "'md-inline', and 'convert-md-to-md-exec' first";

  return convertGeneric(op, scope);
}

LogicalResult Assignment::convertBlock(Block &block, Scope &scope) {
  // The last op of the block that uses each value defined in the block,
  // directly or in a nested region.
  unsigned position = 0;
  for (Operation &op : block) {
    op.walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        if (operand.getParentBlock() == &block)
          scope.lastUse[operand] = position;
    });
    ++position;
  }

  position = 0;
  for (Operation &op : block) {
    if (failed(convertOp(&op, scope, position)))
      return failure();

    // The buffers of the fields that die here hold nothing from now on.
    op.walk([&](Operation *nested) {
      for (Value operand : nested->getOperands()) {
        if (!isField(operand.getType()) ||
            operand.getParentBlock() != &block ||
            scope.lastUse.lookup(operand) != position)
          continue;
        auto found = buffers.find(operand);
        if (found == buffers.end())
          continue;
        Value buffer = found->second;
        buffers.erase(found);
        if (scope.owns(buffer))
          scope.release(buffer);
      }
    });
    ++position;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

/// Returns true if `function` has a field in its signature or in its body.
static bool usesFields(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (llvm::any_of(type.getInputs(), isField) ||
      llvm::any_of(type.getResults(), isField))
    return true;
  bool found = false;
  function.walk([&](Operation *op) {
    found |= llvm::any_of(op->getResultTypes(), isField);
    found |= isa<BuildNeighborsOp>(op);
    if (auto empty = dyn_cast<EmptyNeighborsOp>(op))
      found |= !empty.isStorageForm();
  });
  return found;
}

LogicalResult Assignment::convertFunction(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  auto convertType = [](Type type) -> Type {
    if (auto field = dyn_cast<md::FieldType>(type))
      return mdrt::getBufferType(field);
    return type;
  };
  SmallVector<Type> inputs, results;
  for (Type input : type.getInputs()) {
    if (isUnsupported(input) || isa<mdrt::NeighborsType>(input))
      return function.emitOpError()
             << "cannot give storage to an argument of type " << input;
    inputs.push_back(convertType(input));
  }
  for (Type result : type.getResults()) {
    if (isUnsupported(result) || isa<mdrt::NeighborsType>(result))
      return function.emitOpError()
             << "cannot give storage to a result of type " << result;
    results.push_back(convertType(result));
  }

  auto converted =
      func::FuncOp::create(function.getLoc(), function.getName(),
                           FunctionType::get(context, inputs, results));
  converted.setSymVisibilityAttr(function.getSymVisibilityAttr());
  module.getBody()->getOperations().insert(function->getIterator(),
                                           converted);

  if (!function.isExternal()) {
    if (!llvm::hasSingleElement(function.getBody()))
      return function.emitOpError()
             << "cannot give storage to a body with more than one block";

    Scope scope(nullptr, context);
    root = &scope;
    Block *entry = converted.addEntryBlock();
    scope.builder.setInsertionPointToEnd(entry);

    sizes.clear();
    Block &oldEntry = function.getBody().front();
    for (unsigned i = 0, e = oldEntry.getNumArguments(); i != e; ++i) {
      Value oldArgument = oldEntry.getArgument(i);
      Value newArgument = entry->getArgument(i);
      if (isField(oldArgument.getType())) {
        bind(oldArgument, newArgument, scope);
        scope.owned.insert(newArgument);
      } else {
        mapping.map(oldArgument, newArgument);
      }
    }
    if (failed(convertBlock(oldEntry, scope)))
      return failure();
    root = nullptr;
  }

  function.erase();
  return success();
}

LogicalResult Assignment::run() {
  SmallVector<func::FuncOp> functions;
  for (Operation &op : module)
    if (auto function = dyn_cast<func::FuncOp>(&op))
      if (usesFields(function))
        functions.push_back(function);
  for (func::FuncOp function : functions)
    if (failed(convertFunction(function)))
      return failure();
  return success();
}

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_ASSIGNSTORAGE
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class AssignStorage : public impl::AssignStorageBase<AssignStorage> {
public:
  using impl::AssignStorageBase<AssignStorage>::AssignStorageBase;

  void runOnOperation() final {
    if (failed(Assignment(getOperation()).run()))
      signalPassFailure();
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
