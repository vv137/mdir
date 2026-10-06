// Assignment of floating-point types to fields and kernels.
//
// See docs/ops-m0.md, Section 7.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include <cassert>

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;
using mdir::md::FieldType;

namespace {

/// The floating-point type of each role that the pass assigns.
struct Policy {
  Type position;
  Type force;
  Type kernel;
  Type integrator;
  Type accumulator;
};

/// What is known about the fields that are stored together.
struct Storage {
  /// The element type of a buffer that holds the fields, and the op that
  /// states it.
  Type boundary;
  Operation *boundaryOp = nullptr;

  bool isPosition = false;
  bool isForce = false;
  bool isIntegrated = false;

  /// The element type that the fields are stored in.
  Type assigned;
};

} // namespace

static bool isReal(Type type) { return type.isF32() || type.isF64(); }

/// Returns true if `type` is a floating-point type or a vector of one.
static bool isRealLike(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return isReal(vector.getElementType());
  return isReal(type);
}

static bool isRealField(Type type) {
  auto field = dyn_cast<FieldType>(type);
  return field && isReal(field.getElementType());
}

/// `type` with its floating-point type replaced by `real`.
static Type withReal(Type type, Type real) {
  if (auto field = dyn_cast<FieldType>(type))
    return isReal(field.getElementType())
               ? FieldType::get(type.getContext(), field.getParticleSet(),
                                field.getNumComponents(), real)
               : type;
  if (auto vector = dyn_cast<VectorType>(type))
    return isReal(vector.getElementType())
               ? VectorType::get(vector.getShape(), real)
               : type;
  return isReal(type) ? real : type;
}

/// `value` converted to the type `target`, which differs from the type of
/// `value` in the floating-point type at most.
static Value createCast(OpBuilder &builder, Location loc, Value value,
                        Type target) {
  Type source = value.getType();
  if (source == target)
    return value;
  unsigned from = getElementTypeOrSelf(source).getIntOrFloatBitWidth();
  unsigned to = getElementTypeOrSelf(target).getIntOrFloatBitWidth();
  if (from < to)
    return arith::ExtFOp::create(builder, loc, target, value);
  return arith::TruncFOp::create(builder, loc, target, value);
}

/// `attribute`, the value of a constant, converted to the type `type`.
static TypedAttr convertConstant(TypedAttr attribute, Type type) {
  const llvm::fltSemantics &semantics =
      cast<FloatType>(getElementTypeOrSelf(type)).getFloatSemantics();
  auto convert = [&](APFloat value) {
    bool losesInformation;
    value.convert(semantics, APFloat::rmNearestTiesToEven,
                  &losesInformation);
    return value;
  };

  if (auto real = dyn_cast<FloatAttr>(attribute))
    return FloatAttr::get(type, convert(real.getValue()));
  auto elements = cast<DenseFPElementsAttr>(attribute);
  return cast<TypedAttr>(elements.mapValues(
      getElementTypeOrSelf(type), [&](const APFloat &value) {
        return convert(value).bitcastToAPInt();
      }));
}

//===----------------------------------------------------------------------===//
// The fields of a function
//===----------------------------------------------------------------------===//

namespace {

/// Assigns types to the fields and the kernels of one function.
class Assigner {
public:
  Assigner(func::FuncOp function, const Policy &policy,
           bool kernelPositions, bool deterministic)
      : function(function), policy(policy),
        kernelPositions(kernelPositions), deterministic(deterministic) {}

  LogicalResult run();

private:
  /// Fields are stored together if they must have one type: a loop-carried
  /// field and the fields it is initialized and updated with, or the field
  /// that a loop accumulates into and its result.
  unsigned find(Value field);
  void unite(Value first, Value second);

  LogicalResult collect(Operation *op);
  LogicalResult resolve();
  void apply();
  /// Gives the loops that compute in a type narrower than the positions the
  /// positions converted to it, once for each block and positions.
  void convertPositions();
  /// Makes each field that a buffer of the boundary holds reach the other
  /// fields through a loop over particles that copies it (`deterministic`).
  void isolateBoundaries();
  /// A loop over particles, before `before`, that converts `positions` to
  /// the type of the kernels.
  Value createConversion(Operation *before, Value positions);

  /// Gives the kernel of `loop` the type `arithmetic`. The kernel takes
  /// `numGeometry` arguments of its own, then `perField` for each field in
  /// `ins`, then one for each of `parameters`. It yields `perOut` values
  /// for each field in `outs`, then one for each value in `reduce`.
  template <typename OpTy>
  void rewriteLoop(OpTy loop, Type arithmetic, unsigned numGeometry,
                   unsigned perField, ValueRange parameters = {},
                   unsigned perOut = 1);

  void markBoundary(Value field, Type element, Operation *op) {
    boundaries.push_back({field, element, op});
  }

  func::FuncOp function;
  const Policy &policy;
  bool kernelPositions;
  bool deterministic;

  llvm::DenseMap<Value, unsigned> ids;
  SmallVector<Value> fields;
  SmallVector<unsigned> parents;
  llvm::DenseMap<unsigned, Storage> storages;

  struct Boundary {
    Value field;
    Type element;
    Operation *op;
  };
  SmallVector<Boundary> boundaries;
  SmallVector<Value> positions, forces, integrated;

  SmallVector<PairForOp> pairLoops;
  SmallVector<TupleForOp> tupleLoops;
  SmallVector<ParticleForOp> particleLoops;
  func::ReturnOp firstReturn;
};

} // namespace

unsigned Assigner::find(Value field) {
  auto [found, inserted] = ids.try_emplace(field, parents.size());
  if (inserted) {
    parents.push_back(found->second);
    fields.push_back(field);
  }
  unsigned id = found->second;
  while (parents[id] != id) {
    parents[id] = parents[parents[id]];
    id = parents[id];
  }
  return id;
}

void Assigner::unite(Value first, Value second) {
  unsigned a = find(first);
  unsigned b = find(second);
  if (a != b)
    parents[b] = a;
}

/// Returns true if `use` is a use as the destination of a loop.
static bool isDestinationUse(OpOperand &use) {
  unsigned index = use.getOperandNumber();
  auto isOut = [&](OperandRange outs) {
    return !outs.empty() && index >= outs.getBeginOperandIndex() &&
           index < outs.getBeginOperandIndex() + outs.size();
  };
  if (auto loop = dyn_cast<PairForOp>(use.getOwner()))
    return isOut(loop.getOuts());
  if (auto loop = dyn_cast<TupleForOp>(use.getOwner()))
    return isOut(loop.getOuts());
  if (auto loop = dyn_cast<ParticleForOp>(use.getOwner()))
    return isOut(loop.getOuts());
  return false;
}

/// Returns true if `value` is a destination that holds no field yet.
static bool isFreshDestination(Value value) {
  Operation *op = value.getDefiningOp();
  return op && isa<EmptyOp, ZerosOp>(op);
}

LogicalResult Assigner::collect(Operation *op) {
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    Operation *yield = loop.getBody()->getTerminator();
    for (unsigned i = 0, e = loop.getNumResults(); i != e; ++i) {
      if (!isRealField(loop.getResult(i).getType()))
        continue;
      Value argument = loop.getRegionIterArgs()[i];
      unite(argument, loop.getInitArgs()[i]);
      unite(argument, yield->getOperand(i));
      unite(argument, loop.getResult(i));
    }
    return success();
  }
  if (isa<scf::YieldOp>(op) && isa<scf::ForOp>(op->getParentOp()))
    return success();

  if (auto ret = dyn_cast<func::ReturnOp>(op)) {
    for (unsigned i = 0, e = ret.getNumOperands(); i != e; ++i) {
      if (!isRealField(ret.getOperand(i).getType()))
        continue;
      find(ret.getOperand(i));
      if (firstReturn)
        unite(firstReturn.getOperand(i), ret.getOperand(i));
    }
    if (!firstReturn)
      firstReturn = ret;
    return success();
  }

  if (auto from = dyn_cast<mdrt::FromBufferOp>(op)) {
    if (isRealField(from.getResult().getType()))
      markBoundary(
          from.getResult(),
          cast<MemRefType>(from.getBuffer().getType()).getElementType(), op);
    return success();
  }
  if (auto call = dyn_cast<mdrt::HostCallOp>(op)) {
    auto callee = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(
        op, call.getCalleeAttr()));
    if (!callee)
      return op->emitOpError() << "calls a function that is not declared";
    for (auto [operand, type] :
         llvm::zip(call.getOperands(), callee.getArgumentTypes()))
      if (isRealField(operand.getType()))
        markBoundary(operand, cast<MemRefType>(type).getElementType(), op);
    return success();
  }
  if (auto to = dyn_cast<mdrt::ToBufferOp>(op)) {
    if (isRealField(to.getField().getType()))
      markBoundary(
          to.getField(),
          cast<MemRefType>(to.getResult().getType()).getElementType(), op);
    return success();
  }

  // A destination takes the type of the result of the loop that writes to
  // it. It is a field like any other where it is used otherwise.
  if (isa<EmptyOp, ZerosOp>(op)) {
    Value result = op->getResult(0);
    if (isRealField(result.getType()))
      for (OpOperand &use : result.getUses())
        if (!isDestinationUse(use))
          find(result);
    return success();
  }

  auto collectLoop = [&](auto loop, SmallVector<Value> &written) {
    for (Value field : loop.getIns())
      if (isRealField(field.getType()))
        find(field);
    for (unsigned i = 0, e = loop.getOuts().size(); i != e; ++i) {
      Value destination = loop.getOuts()[i];
      Value result = loop.getResult(i);
      if (!isRealField(result.getType()))
        continue;
      written.push_back(result);
      if (!isFreshDestination(destination))
        unite(result, destination);
    }
  };
  auto checkForm = [&](auto loop) -> LogicalResult {
    if (loop.isStorageForm())
      return op->emitOpError()
             << "is in the storage form; precision is assigned before "
                "storage";
    return success();
  };
  if (auto loop = dyn_cast<PairForOp>(op)) {
    if (failed(checkForm(loop)))
      return failure();
    positions.push_back(loop.getPositions());
    collectLoop(loop, forces);
    pairLoops.push_back(loop);
    return success();
  }
  if (auto loop = dyn_cast<TupleForOp>(op)) {
    if (failed(checkForm(loop)))
      return failure();
    positions.push_back(loop.getPositions());
    collectLoop(loop, forces);
    for (Value field : loop.getParameters())
      if (isRealField(field.getType()))
        find(field);
    tupleLoops.push_back(loop);
    return success();
  }
  if (auto loop = dyn_cast<ParticleForOp>(op)) {
    if (failed(checkForm(loop)))
      return failure();
    collectLoop(loop, integrated);
    particleLoops.push_back(loop);
    return success();
  }

  if (auto cells = dyn_cast<BuildCellsOp>(op)) {
    positions.push_back(cells.getPositions());
    return success();
  }
  if (auto build = dyn_cast<BuildNeighborsOp>(op)) {
    positions.push_back(build.getPositions());
    return success();
  }
  if (auto triplets = dyn_cast<BuildTripletsOp>(op)) {
    positions.push_back(triplets.getPositions());
    return success();
  }
  if (auto refresh = dyn_cast<RefreshNeighborsOp>(op)) {
    positions.push_back(refresh.getPositions());
    return success();
  }
  if (auto reference = dyn_cast<ReferencePositionsOp>(op)) {
    // The structure holds the positions as the positions are stored that
    // it is built at.
    Value result = reference.getResult();
    positions.push_back(result);
    for (Operation *user : reference.getNeighbors().getUsers())
      if (auto refresh = dyn_cast<RefreshNeighborsOp>(user))
        unite(result, refresh.getPositions());
    return success();
  }
  if (auto order = dyn_cast<SpatialOrderOp>(op)) {
    if (order.isStorageForm())
      return op->emitOpError() << "is in the storage form; precision is "
                                  "assigned before storage";
    positions.push_back(order.getPositions());
    return success();
  }
  if (auto reciprocal = dyn_cast<ReciprocalOp>(op)) {
    if (reciprocal.isStorageForm())
      return op->emitOpError() << "is in the storage form; precision is "
                                  "assigned before storage";
    // The positions and the charges as they are stored; the forces are
    // forces.
    positions.push_back(reciprocal.getPositions());
    find(reciprocal.getPositions());
    find(reciprocal.getCharges());
    forces.push_back(reciprocal.getForces());
    find(reciprocal.getForces());
    return success();
  }
  if (auto permute = dyn_cast<PermuteOp>(op)) {
    if (permute.isStorageForm())
      return op->emitOpError() << "is in the storage form; precision is "
                                  "assigned before storage";
    if (isRealField(permute.getResult().getType()))
      unite(permute.getResult(), permute.getField());
    return success();
  }

  bool usesFields = llvm::any_of(op->getOperandTypes(), isRealField) ||
                    llvm::any_of(op->getResultTypes(), isRealField);
  if (usesFields)
    return op->emitOpError()
           << "cannot be assigned a precision: the op is not known to the "
              "pass and uses fields";
  return success();
}

LogicalResult Assigner::resolve() {
  for (const Boundary &boundary : boundaries) {
    Storage &storage = storages[find(boundary.field)];
    if (storage.boundary && storage.boundary != boundary.element) {
      InFlightDiagnostic diagnostic =
          boundary.op->emitOpError()
          << "states that a field is stored as " << boundary.element
          << ", but the field is stored as " << storage.boundary
          << " elsewhere";
      diagnostic.attachNote(storage.boundaryOp->getLoc()) << "see here";
      return diagnostic;
    }
    storage.boundary = boundary.element;
    storage.boundaryOp = boundary.op;
  }
  for (Value field : positions)
    storages[find(field)].isPosition = true;
  for (Value field : forces)
    storages[find(field)].isForce = true;
  for (Value field : integrated)
    storages[find(field)].isIntegrated = true;

  bool oneType =
      policy.position == policy.force && policy.force == policy.integrator;
  for (Value field : fields) {
    Storage &storage = storages[find(field)];
    if (storage.assigned)
      continue;
    if (storage.boundary)
      storage.assigned = storage.boundary;
    else if (storage.isPosition)
      storage.assigned = policy.position;
    else if (storage.isForce)
      storage.assigned = policy.force;
    else if (storage.isIntegrated)
      storage.assigned = policy.integrator;
    else if (oneType)
      storage.assigned = policy.position;
    else
      return emitError(field.getLoc())
             << "cannot tell the type that a field is stored in: no buffer "
                "holds the field, no loop writes it, and it is not used as "
                "positions";
  }
  return success();
}

template <typename OpTy>
void Assigner::rewriteLoop(OpTy loop, Type arithmetic, unsigned numGeometry,
                           unsigned perField, ValueRange parameters,
                           unsigned perOut) {
  Location loc = loop.getLoc();
  OpBuilder before(loop);
  unsigned numOuts = loop.getOuts().size();
  unsigned numReduce = loop.getReduce().size();

  // A destination that holds no field yet takes the type of the result.
  for (unsigned i = 0; i != numOuts; ++i) {
    Value destination = loop.getOuts()[i];
    Type type = loop.getResult(i).getType();
    if (destination.getType() == type)
      continue;
    Operation *definition = destination.getDefiningOp();
    assert(definition && (isa<EmptyOp, ZerosOp>(definition)) &&
           "a destination that holds a field is stored with the result");
    Value fresh;
    if (isa<EmptyOp>(definition))
      fresh = EmptyOp::create(before, definition->getLoc(), type);
    else
      fresh = ZerosOp::create(before, definition->getLoc(), type);
    loop->setOperand(loop.getOuts().getBeginOperandIndex() + i, fresh);
    if (definition->use_empty())
      definition->erase();
  }

  // A global sum is accumulated in the accumulator type and converted to
  // the type that the rest of the function computes in.
  for (unsigned i = 0; i != numReduce; ++i) {
    Value init = loop.getReduce()[i];
    Type outside = init.getType();
    Type inside = withReal(outside, policy.accumulator);
    if (inside == outside)
      continue;
    loop->setOperand(loop.getReduce().getBeginOperandIndex() + i,
                     createCast(before, loc, init, inside));
    Value result = loop.getResult(numOuts + i);
    result.setType(inside);
    OpBuilder after(loop.getContext());
    after.setInsertionPointAfter(loop);
    Value converted = createCast(after, loc, result, outside);
    result.replaceAllUsesExcept(converted, converted.getDefiningOp());
  }

  // Every floating-point value of the kernel gets the arithmetic type.
  Region &region = loop.getKernel();
  Block &kernel = region.front();
  for (BlockArgument argument : kernel.getArguments())
    argument.setType(withReal(argument.getType(), arithmetic));

  llvm::DenseMap<Value, Value> captured;
  region.walk([&](Operation *nested) {
    for (OpOperand &operand : nested->getOpOperands()) {
      Value value = operand.get();
      if (region.isAncestor(value.getParentRegion()) ||
          !isRealLike(value.getType()))
        continue;
      Value &converted = captured[value];
      if (!converted) {
        Type type = withReal(value.getType(), arithmetic);
        if (auto constant = value.getDefiningOp<arith::ConstantOp>())
          converted = arith::ConstantOp::create(
              before, constant.getLoc(), type,
              convertConstant(constant.getValue(), type));
        else
          converted = createCast(before, loc, value, type);
      }
      operand.set(converted);
    }

    for (Value result : nested->getResults())
      result.setType(withReal(result.getType(), arithmetic));
    for (Region &inner : nested->getRegions())
      for (Block &block : inner)
        for (BlockArgument argument : block.getArguments())
          argument.setType(withReal(argument.getType(), arithmetic));

    if (auto constant = dyn_cast<arith::ConstantOp>(nested))
      if (isRealLike(constant.getType()))
        constant.setValueAttr(
            convertConstant(constant.getValue(), constant.getType()));
  });

  // The kernel receives the values of a field in the type that the field
  // is stored in.
  OpBuilder entry(&kernel, kernel.begin());
  unsigned index = numGeometry;
  auto takeStored = [&](Value field, unsigned count) {
    Type stored = cast<FieldType>(field.getType()).getKernelValueType();
    for (unsigned i = 0; i != count; ++i, ++index) {
      BlockArgument argument = kernel.getArgument(index);
      Type computed = argument.getType();
      if (stored == computed)
        continue;
      argument.setType(stored);
      Value converted = createCast(entry, loc, argument, computed);
      argument.replaceAllUsesExcept(converted, converted.getDefiningOp());
    }
  };
  for (Value field : loop.getIns())
    takeStored(field, perField);
  for (Value field : parameters)
    takeStored(field, 1);

  // It yields values in the types of what they are added to.
  Operation *yield = kernel.getTerminator();
  OpBuilder exit(yield);
  for (unsigned i = 0, e = yield->getNumOperands(); i != e; ++i) {
    unsigned result = i < numOuts * perOut ? i / perOut
                                           : i - numOuts * (perOut - 1);
    Type expected = loop.getResult(result).getType();
    if (auto field = dyn_cast<FieldType>(expected))
      expected = field.getKernelValueType();
    yield->setOperand(i,
                      createCast(exit, loc, yield->getOperand(i), expected));
  }
}

void Assigner::apply() {
  for (Value field : fields)
    field.setType(
        withReal(field.getType(), storages[find(field)].assigned));

  for (PairForOp loop : pairLoops)
    rewriteLoop(loop, policy.kernel, /*numGeometry=*/2, /*perField=*/2);
  for (TupleForOp loop : tupleLoops)
    rewriteLoop(loop, policy.kernel,
                /*numGeometry=*/loop.getCoordinateKinds().size(),
                /*perField=*/loop.getArity(), loop.getParameters(),
                /*perOut=*/loop.getArity());
  for (ParticleForOp loop : particleLoops)
    rewriteLoop(loop, policy.integrator, /*numGeometry=*/0, /*perField=*/1);

  // The signature follows the arguments and the returned values.
  FunctionType type = function.getFunctionType();
  SmallVector<Type> inputs, results(type.getResults());
  for (BlockArgument argument : function.getArguments())
    inputs.push_back(argument.getType());
  if (firstReturn)
    results.assign(firstReturn.getOperandTypes().begin(),
                   firstReturn.getOperandTypes().end());
  function.setType(FunctionType::get(function.getContext(), inputs, results));
}

/// A loop over particles that copies `field`, placed before `point`; the
/// copy is the result.
static Value createCopy(OpBuilder &builder, Location loc, Value field) {
  auto type = cast<FieldType>(field.getType());
  Value empty = EmptyOp::create(builder, loc, type);
  auto loop = ParticleForOp::create(builder, loc, TypeRange{type},
                                    ValueRange{field}, ValueRange{empty},
                                    /*reduce=*/ValueRange(),
                                    /*scratch=*/ValueRange());
  Block *block = new Block();
  loop.getKernel().push_back(block);
  Value value = block->addArgument(type.getKernelValueType(), loc);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  YieldOp::create(kernel, loc, ValueRange{value});
  return loop.getResult(0);
}

void Assigner::isolateBoundaries() {
  SmallVector<Operation *> boundaries;
  function.walk([&](Operation *op) {
    if (isa<mdrt::FromBufferOp, mdrt::ToBufferOp, mdrt::HostCallOp>(op))
      boundaries.push_back(op);
  });
  for (Operation *op : boundaries) {
    if (auto from = dyn_cast<mdrt::FromBufferOp>(op)) {
      Value result = from.getResult();
      if (!isRealField(result.getType()) || result.use_empty())
        continue;
      OpBuilder builder(op->getContext());
      builder.setInsertionPointAfter(op);
      Value copy = createCopy(builder, op->getLoc(), result);
      result.replaceAllUsesExcept(
          copy, copy.getDefiningOp<ParticleForOp>().getOperation());
      continue;
    }
    OpBuilder builder(op);
    for (OpOperand &operand : op->getOpOperands())
      if (isRealField(operand.get().getType()))
        operand.set(createCopy(builder, op->getLoc(), operand.get()));
  }
}

LogicalResult Assigner::run() {
  // In the deterministic mode the type of a buffer decides that of its
  // copy alone: the fields of the steps are stored as their roles say,
  // whatever buffers the program reaches them from or hands them to.
  if (deterministic)
    isolateBoundaries();
  for (BlockArgument argument : function.getArguments())
    if (isRealField(argument.getType()))
      find(argument);

  LogicalResult status = success();
  function.walk([&](Operation *op) {
    if (succeeded(status) && op != function.getOperation())
      status = collect(op);
  });
  if (failed(status) || failed(resolve()))
    return failure();
  apply();
  if (kernelPositions)
    convertPositions();
  return success();
}

Value Assigner::createConversion(Operation *before, Value positions) {
  OpBuilder builder(before);
  Location loc = before->getLoc();
  auto field = cast<FieldType>(positions.getType());
  auto narrow = cast<FieldType>(withReal(field, policy.kernel));
  Value empty = EmptyOp::create(builder, loc, narrow);
  auto loop = ParticleForOp::create(
      builder, loc, TypeRange{narrow}, ValueRange{positions},
      ValueRange{empty}, /*reduce=*/ValueRange(), /*scratch=*/ValueRange());
  Block *block = new Block();
  loop.getKernel().push_back(block);
  Value position = block->addArgument(field.getKernelValueType(), loc);
  OpBuilder kernel = OpBuilder::atBlockEnd(block);
  Value converted =
      createCast(kernel, loc, position, narrow.getKernelValueType());
  YieldOp::create(kernel, loc, ValueRange{converted});
  return loop.getResult(0);
}

void Assigner::convertPositions() {
  // The subtraction of two positions is where precision is lost; in f32
  // the positions of a cell of 60 Å are 4e-6 Å apart. The terms of the
  // potential take them so, as GROMACS does; the constraints, whose loops
  // are over disjoint tuples, keep the positions they are stored in (D79).
  // A loop over groups takes the positions as they are stored: its
  // lowering moves them to the frames of the groups first, then converts
  // them (D95). The triplets are found at the positions that the loops
  // over them take, so that the test of the cutoff and the kernel see the
  // same differences (D160).
  SmallVector<Operation *> loops;
  function.walk([&](Operation *op) {
    if (isa<BuildTripletsOp>(op))
      loops.push_back(op);
    else if (auto pair = dyn_cast<PairForOp>(op)) {
      if (pair.getTraversal() != Traversal::Unique)
        loops.push_back(op);
    }
    else if (auto tuple = dyn_cast<TupleForOp>(op))
      if (!tuple.getDisjoint())
        loops.push_back(op);
  });
  DenseMap<std::pair<Block *, Value>, Value> converted;
  for (Operation *op : loops) {
    OpOperand &operand =
        isa<PairForOp>(op)    ? cast<PairForOp>(op).getPositionsMutable()
        : isa<TupleForOp>(op) ? cast<TupleForOp>(op).getPositionsMutable()
                              : cast<BuildTripletsOp>(op).getPositionsMutable();
    Value positions = operand.get();
    auto field = dyn_cast<FieldType>(positions.getType());
    if (!field || !isReal(field.getElementType()) ||
        field.getElementType().getIntOrFloatBitWidth() <=
            policy.kernel.getIntOrFloatBitWidth())
      continue;
    Value &slot = converted[{op->getBlock(), positions}];
    if (!slot)
      slot = createConversion(op, positions);
    operand.set(slot);
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_ASSIGNPRECISION
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class AssignPrecision : public impl::AssignPrecisionBase<AssignPrecision> {
public:
  using impl::AssignPrecisionBase<AssignPrecision>::AssignPrecisionBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    Policy policy;
    if (failed(getPolicy(policy)))
      return signalPassFailure();
    if (getOperation()->hasAttr("md.derivative_check") &&
        (!policy.position.isF64() || !policy.force.isF64() ||
         !policy.kernel.isF64() || !policy.integrator.isF64() ||
         !policy.accumulator.isF64())) {
      getOperation().emitError("derivative checking requires double precision for every role");
      return signalPassFailure();
    }

    SmallVector<func::FuncOp> functions;
    for (Operation &op : module)
      if (auto function = dyn_cast<func::FuncOp>(&op))
        if (!function.isExternal())
          functions.push_back(function);
    for (func::FuncOp function : functions)
      if (failed(Assigner(function, policy, kernelPositions, deterministic)
                     .run()))
        return signalPassFailure();
  }

private:
  /// The policy that the options describe: the assignment of the mode, with
  /// the roles that are given on their own replaced.
  LogicalResult getPolicy(Policy &policy) {
    ModuleOp module = getOperation();
    Type single = Float32Type::get(&getContext());
    Type wide = Float64Type::get(&getContext());

    if (mode == "single")
      policy = {single, single, single, single, wide};
    else if (mode == "mixed")
      policy = {wide, single, single, wide, wide};
    else if (mode == "double")
      policy = {wide, wide, wide, wide, wide};
    else
      return module.emitError()
             << "expected the mode 'single', 'mixed', or 'double', got '"
             << mode << "'";

    auto replace = [&](StringRef role, const std::string &name,
                       Type &type) -> LogicalResult {
      if (name.empty())
        return success();
      if (name == "f32")
        type = single;
      else if (name == "f64")
        type = wide;
      else
        return module.emitError()
               << "expected the type 'f32' or 'f64' for the role '" << role
               << "', got '" << name << "'";
      return success();
    };
    if (failed(replace("position", position, policy.position)) ||
        failed(replace("force", force, policy.force)) ||
        failed(replace("kernel", kernel, policy.kernel)) ||
        failed(replace("integrator", integrator, policy.integrator)) ||
        failed(replace("accumulator", accumulator, policy.accumulator)))
      return failure();
    return success();
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
