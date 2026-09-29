// Conversion of md_exec loops to loops over buffers.
//
// The pass assigns a buffer to every field value and emits `scf` loops that
// load from and store to `memref`s. See docs/ops-m0.md, Section 10.

#include "mdir/Conversion/Passes.h"

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mdir;

namespace mdir {
/// The text of the template that builds a neighbor matrix.
extern const char *const neighborsMatrixTemplate;
} // namespace mdir

static const char *const buildNeighborsName = "mdrt.build_neighbors_matrix";
static const char *const reportOverflowName = "mdrtReportNeighborOverflow";

namespace {

/// The storage of a neighbor matrix.
struct Neighbors {
  Value counts;
  Value index;
};

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

class Lowering {
public:
  explicit Lowering(ModuleOp module)
      : module(module), context(module.getContext()) {}

  LogicalResult run();

private:
  LogicalResult lowerFunction(func::FuncOp function);
  LogicalResult lowerBlock(Block &block, Scope &scope);
  LogicalResult lowerOp(Operation *op, Scope &scope, unsigned position);

  LogicalResult lowerFor(scf::ForOp op, Scope &scope, unsigned position);
  LogicalResult lowerYield(scf::YieldOp op, Scope &scope);
  LogicalResult lowerBuildNeighbors(md_exec::BuildNeighborsOp op,
                                    Scope &scope);
  LogicalResult lowerParticleFor(md_exec::ParticleForOp op, Scope &scope,
                                 unsigned position);
  LogicalResult lowerPairFor(md_exec::PairForOp op, Scope &scope,
                             unsigned position);
  LogicalResult lowerGeneric(Operation *op, Scope &scope);

  /// Chooses the buffer that a loop writes the field `destination` to.
  /// `readFields` are the fields that the loop reads; one of them may give
  /// up its buffer if `inPlace` is set. `accumulates` is set if the loop
  /// must add to what the buffer holds.
  LogicalResult chooseDestination(Operation *op, Value destination,
                                  ValueRange readFields, bool inPlace,
                                  Scope &scope, unsigned position,
                                  Value &buffer, bool &accumulates);

  Type convertType(Type type);
  bool needsLowering(Type type);

  /// The buffer that holds the field `field`.
  LogicalResult getBuffer(Value field, Scope &scope, Value &buffer);

  /// The number of particles of the set that `field` belongs to.
  LogicalResult getSize(Operation *op, Type field, Value &size);

  /// Records the buffer of a field and, if it is the first of its particle
  /// set, the number of particles.
  void bind(Value field, Value buffer, Scope &scope);

  Value loadElement(OpBuilder &builder, Location loc, Value buffer,
                    Value particle);
  void storeElement(OpBuilder &builder, Location loc, Value value,
                    Value buffer, Value particle);
  Value createIndex(OpBuilder &builder, Location loc, int64_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value);
  }
  Value createZero(OpBuilder &builder, Location loc, Type type) {
    return arith::ConstantOp::create(
        builder, loc, type, cast<TypedAttr>(builder.getZeroAttr(type)));
  }

  /// Ends the body of a parallel loop with a reduction that adds `values`.
  void createSumReduction(OpBuilder &builder, Location loc,
                          ArrayRef<Value> values);

  LogicalResult addTemplates();
  func::FuncOp getOrDeclare(StringRef name, FunctionType type);

  ModuleOp module;
  MLIRContext *context;

  /// Values other than fields: the value in the new code.
  IRMapping mapping;
  /// Fields: the buffer that holds the field.
  llvm::DenseMap<Value, Value> buffers;
  llvm::DenseMap<Value, Neighbors> neighbors;
  /// Particle sets: the number of particles.
  llvm::DenseMap<Attribute, Value> sizes;

  /// The scope of the body of the function that is being lowered.
  Scope *root = nullptr;
  bool templatesAdded = false;
};

} // namespace

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
// Types and buffers
//===----------------------------------------------------------------------===//

bool Lowering::needsLowering(Type type) {
  return isa<md::MDDialect, mdrt::MDRTDialect>(type.getDialect());
}

Type Lowering::convertType(Type type) {
  if (auto field = dyn_cast<md::FieldType>(type))
    return mdrt::getBufferType(field);
  if (isa<md::CellType>(type))
    return VectorType::get({3}, Float64Type::get(context));
  return type;
}

void Lowering::bind(Value field, Value buffer, Scope &scope) {
  buffers[field] = buffer;
  Attribute set = cast<md::FieldType>(field.getType()).getParticleSet();
  if (sizes.count(set))
    return;
  Value zero = createIndex(scope.builder, field.getLoc(), 0);
  sizes[set] =
      memref::DimOp::create(scope.builder, field.getLoc(), buffer, zero);
}

LogicalResult Lowering::getSize(Operation *op, Type field, Value &size) {
  Attribute set = cast<md::FieldType>(field).getParticleSet();
  size = sizes.lookup(set);
  if (!size)
    return op->emitOpError()
           << "the number of particles of " << set
           << " is not known here: no field of the set has a buffer yet";
  return success();
}

LogicalResult Lowering::getBuffer(Value field, Scope &scope, Value &buffer) {
  buffer = buffers.lookup(field);
  if (buffer)
    return success();

  // A destination that no loop has taken.
  Operation *op = field.getDefiningOp();
  if (!op || !isa<md_exec::EmptyOp, md_exec::ZerosOp>(op))
    return emitError(field.getLoc()) << "a field has no buffer";

  Value size;
  if (failed(getSize(op, field.getType(), size)))
    return failure();
  auto type = cast<MemRefType>(convertType(field.getType()));
  buffer = scope.request(type, size, op->getLoc());
  buffers[field] = buffer;

  if (isa<md_exec::ZerosOp>(op)) {
    OpBuilder &builder = scope.builder;
    Location loc = op->getLoc();
    Type valueType = cast<md::FieldType>(field.getType()).getKernelValueType();
    Value zero = createIndex(builder, loc, 0);
    Value one = createIndex(builder, loc, 1);
    scf::ParallelOp::create(
        builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
        [&](OpBuilder &body, Location, ValueRange ivs) {
          storeElement(body, loc, createZero(body, loc, valueType), buffer,
                       ivs[0]);
        });
  }
  return success();
}

Value Lowering::loadElement(OpBuilder &builder, Location loc, Value buffer,
                            Value particle) {
  auto type = cast<MemRefType>(buffer.getType());
  if (type.getRank() == 1)
    return memref::LoadOp::create(builder, loc, buffer, ValueRange{particle});

  int64_t components = type.getDimSize(1);
  SmallVector<Value, 3> elements;
  for (int64_t c = 0; c < components; ++c)
    elements.push_back(memref::LoadOp::create(
        builder, loc, buffer,
        ValueRange{particle, createIndex(builder, loc, c)}));
  return vector::FromElementsOp::create(
      builder, loc, VectorType::get({components}, type.getElementType()),
      elements);
}

void Lowering::storeElement(OpBuilder &builder, Location loc, Value value,
                            Value buffer, Value particle) {
  auto type = cast<MemRefType>(buffer.getType());
  if (type.getRank() == 1) {
    memref::StoreOp::create(builder, loc, value, buffer,
                            ValueRange{particle});
    return;
  }
  for (int64_t c = 0, e = type.getDimSize(1); c < e; ++c) {
    Value element = vector::ExtractOp::create(builder, loc, value, c);
    memref::StoreOp::create(builder, loc, element, buffer,
                            ValueRange{particle, createIndex(builder, loc, c)});
  }
}

void Lowering::createSumReduction(OpBuilder &builder, Location loc,
                                  ArrayRef<Value> values) {
  auto reduce = scf::ReduceOp::create(builder, loc, values);
  for (Region &region : reduce->getRegions()) {
    Block &block = region.front();
    OpBuilder combiner(context);
    combiner.setInsertionPointToEnd(&block);
    Value sum = arith::AddFOp::create(combiner, loc, block.getArgument(0),
                                      block.getArgument(1));
    scf::ReduceReturnOp::create(combiner, loc, sum);
  }
}

//===----------------------------------------------------------------------===//
// Destinations
//===----------------------------------------------------------------------===//

LogicalResult Lowering::chooseDestination(Operation *op, Value destination,
                                          ValueRange readFields, bool inPlace,
                                          Scope &scope, unsigned position,
                                          Value &buffer, bool &accumulates) {
  Operation *definition = destination.getDefiningOp();
  bool isEmpty = definition && isa<md_exec::EmptyOp>(definition);
  bool isZeros = definition && isa<md_exec::ZerosOp>(definition);
  auto type = cast<MemRefType>(convertType(destination.getType()));

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
// Loops over particles
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lowerParticleFor(md_exec::ParticleForOp op,
                                         Scope &scope, unsigned position) {
  Location loc = op.getLoc();
  OpBuilder &builder = scope.builder;

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

  Type anyField = op.getIns().empty() ? op.getOuts().front().getType()
                                      : op.getIns().front().getType();
  Value size;
  if (failed(getSize(op, anyField, size)))
    return failure();

  SmallVector<Value> inits;
  for (Value value : op.getReduce())
    inits.push_back(mapping.lookup(value));

  Block &kernel = op.getKernel().front();
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  auto loop = scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      inits, [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        Value particle = ivs[0];
        IRMapping local = mapping;
        for (unsigned i = 0, e = ins.size(); i != e; ++i)
          local.map(kernel.getArgument(i),
                    loadElement(body, loc, ins[i], particle));
        for (Operation &nested : kernel.without_terminator())
          body.clone(nested, local);

        Operation *yield = kernel.getTerminator();
        unsigned numOuts = outs.size();
        for (unsigned i = 0; i != numOuts; ++i)
          storeElement(body, loc, local.lookup(yield->getOperand(i)), outs[i],
                       particle);

        SmallVector<Value> contributions;
        for (unsigned i = numOuts, e = yield->getNumOperands(); i != e; ++i)
          contributions.push_back(local.lookup(yield->getOperand(i)));
        if (!contributions.empty())
          createSumReduction(body, loc, contributions);
      });

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
// Loops over pairs
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lowerPairFor(md_exec::PairForOp op, Scope &scope,
                                     unsigned position) {
  Location loc = op.getLoc();
  OpBuilder &builder = scope.builder;

  auto found = neighbors.find(op.getNeighbors());
  if (found == neighbors.end())
    return op.emitOpError() << "the neighbor structure has no storage; only "
                               "the result of 'md_exec.build_neighbors' in "
                               "the same function is supported";
  Neighbors structure = found->second;

  Value positions;
  if (failed(getBuffer(op.getPositions(), scope, positions)))
    return failure();
  Value box = mapping.lookup(op.getCell());

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
  SmallVector<bool> accumulates;
  for (Value destination : op.getOuts()) {
    Value buffer;
    bool accumulate;
    if (failed(chooseDestination(op, destination, readFields,
                                 /*inPlace=*/false, scope, position, buffer,
                                 accumulate)))
      return failure();
    outs.push_back(buffer);
    accumulates.push_back(accumulate);
  }

  Value size;
  if (failed(getSize(op, op.getPositions().getType(), size)))
    return failure();

  SmallVector<Value> inits;
  for (Value value : op.getReduce())
    inits.push_back(mapping.lookup(value));

  double cutoff = op.getCutoff().convertToDouble();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  unsigned numOuts = outs.size();
  unsigned numYields = yield->getNumOperands();

  Type real = builder.getF64Type();
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  Value cutoff2 = arith::ConstantOp::create(
      builder, loc, real, builder.getF64FloatAttr(cutoff * cutoff));

  auto loop = scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      inits, [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        Value central = ivs[0];
        Value centralPosition = loadElement(body, loc, positions, central);
        SmallVector<Value> centralValues;
        for (Value buffer : ins)
          centralValues.push_back(loadElement(body, loc, buffer, central));

        Value count = memref::LoadOp::create(body, loc, structure.counts,
                                             ValueRange{central});
        Value end =
            arith::IndexCastOp::create(body, loc, body.getIndexType(), count);

        // The contributions of the neighbors of one particle are summed in
        // the order of its list.
        SmallVector<Value> sums;
        for (Value value : yield->getOperands())
          sums.push_back(createZero(body, loc, value.getType()));

        auto inner = scf::ForOp::create(
            body, loc, zero, end, one, sums,
            [&](OpBuilder &pair, Location, Value entry, ValueRange partial) {
              Value narrow =
                  memref::LoadOp::create(pair, loc, structure.index,
                                         ValueRange{central, entry});
              Value other = arith::IndexCastOp::create(
                  pair, loc, pair.getIndexType(), narrow);

              // The minimum-image displacement and its squared length.
              Value otherPosition = loadElement(pair, loc, positions, other);
              Value raw =
                  arith::SubFOp::create(pair, loc, centralPosition,
                                        otherPosition);
              Value images = arith::DivFOp::create(pair, loc, raw, box);
              Value nearest = math::RoundEvenOp::create(pair, loc, images);
              Value shift = arith::MulFOp::create(pair, loc, nearest, box);
              Value d = arith::SubFOp::create(pair, loc, raw, shift);
              Value squares = arith::MulFOp::create(pair, loc, d, d);
              Value r2 = vector::ReductionOp::create(
                  pair, loc, vector::CombiningKind::ADD, squares);

              IRMapping local = mapping;
              local.map(kernel.getArgument(0), r2);
              local.map(kernel.getArgument(1), d);
              for (unsigned i = 0, e = ins.size(); i != e; ++i) {
                local.map(kernel.getArgument(2 + 2 * i), centralValues[i]);
                local.map(kernel.getArgument(3 + 2 * i),
                          loadElement(pair, loc, ins[i], other));
              }
              for (Operation &nested : kernel.without_terminator())
                pair.clone(nested, local);

              // A pair beyond the cutoff contributes nothing.
              Value within = arith::CmpFOp::create(
                  pair, loc, arith::CmpFPredicate::OLT, r2, cutoff2);
              SmallVector<Value> updated;
              for (unsigned i = 0; i != numYields; ++i) {
                Value contribution = local.lookup(yield->getOperand(i));
                Value nothing =
                    createZero(pair, loc, contribution.getType());
                Value masked = arith::SelectOp::create(
                    pair, loc, within, contribution, nothing);
                updated.push_back(
                    arith::AddFOp::create(pair, loc, partial[i], masked));
              }
              scf::YieldOp::create(pair, loc, updated);
            });

        for (unsigned i = 0; i != numOuts; ++i) {
          Value total = inner.getResult(i);
          if (accumulates[i])
            total = arith::AddFOp::create(
                body, loc, loadElement(body, loc, outs[i], central), total);
          storeElement(body, loc, total, outs[i], central);
        }

        SmallVector<Value> contributions;
        for (unsigned i = numOuts; i != numYields; ++i) {
          Value total = inner.getResult(i);
          if (auto weights = op.getWeights()) {
            double weight = (*weights)[i - numOuts];
            if (weight != 1.0) {
              Type type = total.getType();
              Value factor;
              if (auto vector = dyn_cast<VectorType>(type))
                factor = arith::ConstantOp::create(
                    body, loc, type,
                    DenseElementsAttr::get(vector, APFloat(weight)));
              else
                factor = arith::ConstantOp::create(
                    body, loc, type, body.getFloatAttr(type, weight));
              total = arith::MulFOp::create(body, loc, factor, total);
            }
          }
          contributions.push_back(total);
        }
        if (!contributions.empty())
          createSumReduction(body, loc, contributions);
      });

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

func::FuncOp Lowering::getOrDeclare(StringRef name, FunctionType type) {
  if (Operation *existing = SymbolTable::lookupSymbolIn(module, name))
    return cast<func::FuncOp>(existing);
  auto function = func::FuncOp::create(module.getLoc(), name, type);
  function.setPrivate();
  module.push_back(function);
  return function;
}

LogicalResult Lowering::addTemplates() {
  if (templatesAdded)
    return success();
  templatesAdded = true;

  ParserConfig config(context);
  OwningOpRef<ModuleOp> templates =
      parseSourceString<ModuleOp>(neighborsMatrixTemplate, config);
  if (!templates)
    return module.emitError() << "cannot parse the neighbor build template";
  for (Operation &op : llvm::make_early_inc_range(*templates)) {
    op.remove();
    module.push_back(&op);
  }
  return success();
}

LogicalResult Lowering::lowerBuildNeighbors(md_exec::BuildNeighborsOp op,
                                            Scope &scope) {
  Location loc = op.getLoc();
  OpBuilder &builder = scope.builder;

  auto cells = op.getCells().getDefiningOp<md_exec::BuildCellsOp>();
  if (!cells)
    return op.emitOpError() << "expected cells that are the result of "
                               "'md_exec.build_cells'";
  if (failed(addTemplates()))
    return failure();

  Value positions;
  if (failed(getBuffer(op.getPositions(), scope, positions)))
    return failure();
  Value box = mapping.lookup(op.getCell());
  Value size;
  if (failed(getSize(op, op.getPositions().getType(), size)))
    return failure();

  // The storage outlives the iterations of any loop around the build, so it
  // is allocated in the body of the function.
  OpBuilder &outer = root->builder;
  Type narrow = builder.getI32Type();
  Value width = createIndex(outer, loc, op.getWidth());
  Neighbors structure;
  structure.counts = memref::AllocOp::create(
      outer, loc, MemRefType::get({ShapedType::kDynamic}, narrow),
      ValueRange{size});
  structure.index = memref::AllocOp::create(
      outer, loc,
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, narrow),
      ValueRange{size, width});
  neighbors[op.getResult()] = structure;

  Type real = builder.getF64Type();
  double cutoff = op.getCutoff().convertToDouble();
  double skin = op.getSkin().convertToDouble();
  Value reach = arith::ConstantOp::create(
      builder, loc, real, builder.getF64FloatAttr(cutoff + skin));
  Value cellWidth = arith::ConstantOp::create(
      builder, loc, real,
      builder.getF64FloatAttr(cells.getWidth().convertToDouble()));

  auto build = cast<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, buildNeighborsName));
  auto call = func::CallOp::create(
      builder, loc, build,
      ValueRange{positions, box, reach, cellWidth, structure.counts,
                 structure.index});
  Value largest = call.getResult(0);

  // A row that is too narrow loses pairs. Stop.
  Type wide = builder.getI64Type();
  auto report = getOrDeclare(
      reportOverflowName, builder.getFunctionType({wide, wide}, {}));
  Value tooMany = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ugt, largest, width);
  scf::IfOp::create(
      builder, loc, tooMany, [&](OpBuilder &then, Location) {
        Value needed = arith::IndexCastOp::create(then, loc, wide, largest);
        Value available = arith::IndexCastOp::create(then, loc, wide, width);
        func::CallOp::create(then, loc, report,
                             ValueRange{needed, available});
        scf::YieldOp::create(then, loc);
      });
  return success();
}

//===----------------------------------------------------------------------===//
// Loops that carry fields
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lowerFor(scf::ForOp op, Scope &scope,
                                 unsigned position) {
  Location loc = op.getLoc();
  Block &oldBody = *op.getBody();

  Scope inner(&scope, context);
  inner.body = new Block();
  inner.builder.setInsertionPointToEnd(inner.body);

  mapping.map(op.getInductionVar(),
              inner.body->addArgument(op.getInductionVar().getType(), loc));

  SmallVector<Value> inits;
  for (auto [init, argument] :
       llvm::zip(op.getInitArgs(), op.getRegionIterArgs())) {
    if (!isa<md::FieldType>(init.getType())) {
      if (needsLowering(init.getType()))
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

    Value carried = inner.body->addArgument(buffer.getType(), loc);
    buffers[argument] = carried;
    inner.owned.insert(carried);
  }

  if (failed(lowerBlock(oldBody, inner)))
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

  unsigned numCarried = op.getNumResults();
  for (unsigned i = 0; i != numCarried; ++i) {
    Value oldResult = op.getResult(i);
    Value newResult = loop->getResult(i);
    if (isa<md::FieldType>(oldResult.getType())) {
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

LogicalResult Lowering::lowerYield(scf::YieldOp op, Scope &scope) {
  Location loc = op.getLoc();
  SmallVector<Value> values;
  llvm::DenseSet<Value> held;

  for (Value value : op.getOperands()) {
    if (!isa<md::FieldType>(value.getType())) {
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

LogicalResult Lowering::lowerGeneric(Operation *op, Scope &scope) {
  bool touchesFields = false;
  op->walk([&](Operation *nested) {
    for (Type type : nested->getOperandTypes())
      touchesFields |= needsLowering(type);
    for (Type type : nested->getResultTypes())
      touchesFields |= needsLowering(type);
  });
  if (touchesFields)
    return op->emitOpError()
           << "cannot be lowered: the op is not known to the pass and uses "
              "fields, cells, or runtime structures";
  scope.builder.clone(*op, mapping);
  return success();
}

LogicalResult Lowering::lowerOp(Operation *op, Scope &scope,
                                unsigned position) {
  Location loc = op->getLoc();
  OpBuilder &builder = scope.builder;

  if (auto from = dyn_cast<mdrt::FromBufferOp>(op)) {
    Value buffer = mapping.lookup(from.getBuffer());
    bind(from.getResult(), buffer, scope);
    scope.owned.insert(buffer);
    return success();
  }
  if (auto to = dyn_cast<mdrt::ToBufferOp>(op)) {
    Value buffer;
    if (failed(getBuffer(to.getField(), scope, buffer)))
      return failure();
    // The buffer is visible outside from here on. Leave it alone.
    scope.owned.erase(buffer);
    mapping.map(to.getResult(), buffer);
    return success();
  }

  if (auto cell = dyn_cast<md::OrthorhombicCellOp>(op)) {
    Type type = convertType(cell.getResult().getType());
    mapping.map(cell.getResult(),
                vector::FromElementsOp::create(
                    builder, loc, type,
                    ValueRange{mapping.lookup(cell.getLx()),
                               mapping.lookup(cell.getLy()),
                               mapping.lookup(cell.getLz())}));
    return success();
  }

  // Destinations get their buffer from the loop that writes to them. Cells
  // are built together with the neighbor structure.
  if (isa<md_exec::EmptyOp, md_exec::ZerosOp, md_exec::BuildCellsOp>(op))
    return success();

  if (auto build = dyn_cast<md_exec::BuildNeighborsOp>(op))
    return lowerBuildNeighbors(build, scope);
  if (auto loop = dyn_cast<md_exec::ParticleForOp>(op))
    return lowerParticleFor(loop, scope, position);
  if (auto loop = dyn_cast<md_exec::PairForOp>(op))
    return lowerPairFor(loop, scope, position);

  if (auto loop = dyn_cast<scf::ForOp>(op))
    return lowerFor(loop, scope, position);
  if (auto yield = dyn_cast<scf::YieldOp>(op))
    if (scope.body)
      return lowerYield(yield, scope);

  if (auto ret = dyn_cast<func::ReturnOp>(op)) {
    SmallVector<Value> values;
    for (Value value : ret.getOperands()) {
      if (!isa<md::FieldType>(value.getType())) {
        values.push_back(mapping.lookup(value));
        continue;
      }
      Value buffer;
      if (failed(getBuffer(value, scope, buffer)))
        return failure();
      values.push_back(buffer);
    }
    func::ReturnOp::create(builder, loc, values);
    return success();
  }

  if (isa<md::MDDialect>(op->getDialect()) ||
      isa<md_exec::MDExecDialect>(op->getDialect()) ||
      op->getName().getDialectNamespace() == "dyn")
    return op->emitOpError()
           << "cannot be lowered; run 'md-differentiate', 'md-inline', and "
              "'convert-md-to-md-exec' first";

  return lowerGeneric(op, scope);
}

LogicalResult Lowering::lowerBlock(Block &block, Scope &scope) {
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
    if (failed(lowerOp(&op, scope, position)))
      return failure();

    // The buffers of the fields that die here hold nothing from now on.
    op.walk([&](Operation *nested) {
      for (Value operand : nested->getOperands()) {
        if (!isa<md::FieldType>(operand.getType()) ||
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

LogicalResult Lowering::lowerFunction(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  SmallVector<Type> inputs, results;
  for (Type input : type.getInputs()) {
    if (needsLowering(input) && !isa<md::FieldType, md::CellType>(input))
      return function.emitOpError()
             << "cannot lower an argument of type " << input;
    inputs.push_back(convertType(input));
  }
  for (Type result : type.getResults()) {
    if (needsLowering(result) && !isa<md::FieldType>(result))
      return function.emitOpError()
             << "cannot lower a result of type " << result;
    results.push_back(convertType(result));
  }

  auto lowered =
      func::FuncOp::create(function.getLoc(), function.getName(),
                           FunctionType::get(context, inputs, results));
  lowered.setSymVisibilityAttr(function.getSymVisibilityAttr());
  module.getBody()->getOperations().insert(function->getIterator(), lowered);

  if (!function.isExternal()) {
    if (!llvm::hasSingleElement(function.getBody()))
      return function.emitOpError()
             << "cannot lower a body with more than one block";

    Scope scope(nullptr, context);
    root = &scope;
    Block *entry = lowered.addEntryBlock();
    scope.builder.setInsertionPointToEnd(entry);

    sizes.clear();
    Block &oldEntry = function.getBody().front();
    for (unsigned i = 0, e = oldEntry.getNumArguments(); i != e; ++i) {
      Value oldArgument = oldEntry.getArgument(i);
      Value newArgument = entry->getArgument(i);
      if (isa<md::FieldType>(oldArgument.getType())) {
        bind(oldArgument, newArgument, scope);
        scope.owned.insert(newArgument);
      } else {
        mapping.map(oldArgument, newArgument);
      }
    }
    if (failed(lowerBlock(oldEntry, scope)))
      return failure();
    root = nullptr;
  }

  function.erase();
  return success();
}

LogicalResult Lowering::run() {
  SmallVector<func::FuncOp> functions;
  for (Operation &op : module)
    if (auto function = dyn_cast<func::FuncOp>(&op))
      functions.push_back(function);
  for (func::FuncOp function : functions)
    if (failed(lowerFunction(function)))
      return failure();

  // What is left of the semantic dialects are definitions that nothing
  // refers to any more.
  for (Operation &op : llvm::make_early_inc_range(module)) {
    if (isa<func::FuncOp>(op))
      continue;
    if (isa<md::ParticleSetOp>(op)) {
      op.erase();
      continue;
    }
    return op.emitOpError()
           << "cannot be lowered; run 'md-inline' first";
  }
  return success();
}

namespace mdir {

#define GEN_PASS_DEF_CONVERTMDEXECTOLOOPS
#include "mdir/Conversion/Passes.h.inc"

namespace {
class ConvertMDExecToLoops
    : public impl::ConvertMDExecToLoopsBase<ConvertMDExecToLoops> {
public:
  using impl::ConvertMDExecToLoopsBase<
      ConvertMDExecToLoops>::ConvertMDExecToLoopsBase;

  void runOnOperation() final {
    Lowering lowering(getOperation());
    if (failed(lowering.run()))
      signalPassFailure();
  }
};
} // namespace

} // namespace mdir
