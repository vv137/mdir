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
  /// The number of neighbors of each particle, and their indices.
  Value counts;
  Value index;
  /// The number of neighbors that a row holds.
  Value width;
  /// The cell that the structure was built in.
  Value box;
  /// Whether the structure has been built, and how often.
  Value valid;
  Value builds;
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
  LogicalResult lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op,
                                      Scope &scope);

  /// Allocates the storage of a neighbor structure that holds `width`
  /// neighbors for each of `size` particles.
  Neighbors allocateNeighbors(Location loc, Value size, int64_t width);

  /// The buffer that holds the configuration that `structure` was built
  /// at. It has the type of `positions`.
  LogicalResult getReference(Operation *op, const Neighbors &structure,
                             Value positions, Value size, Value &reference);

  /// `value`, a floating-point value or a vector of them, converted to the
  /// floating-point type `real`.
  Value convertReal(OpBuilder &builder, Location loc, Value value, Type real);
  Value createReal(OpBuilder &builder, Location loc, Type real,
                   double value) {
    return arith::ConstantOp::create(builder, loc, real,
                                     builder.getFloatAttr(real, value));
  }

  /// Builds `structure` at the configuration `positions`.
  LogicalResult emitBuild(Operation *op, OpBuilder &builder, Location loc,
                          const Neighbors &structure, Value positions,
                          Value box, Value size, double reach,
                          double cellWidth);
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

  /// Ends the body of a parallel loop with a reduction of `values`: their
  /// sum, or their maximum.
  void createReduction(OpBuilder &builder, Location loc,
                       ArrayRef<Value> values, bool isSum);
  void createSumReduction(OpBuilder &builder, Location loc,
                          ArrayRef<Value> values) {
    createReduction(builder, loc, values, /*isSum=*/true);
  }

  /// Adds the templates for positions of the type `real` to the module.
  LogicalResult addTemplates(Type real);
  func::FuncOp getOrDeclare(StringRef name, FunctionType type);

  ModuleOp module;
  MLIRContext *context;

  /// Values other than fields: the value in the new code.
  IRMapping mapping;
  /// Fields: the buffer that holds the field.
  llvm::DenseMap<Value, Value> buffers;
  llvm::DenseMap<Value, Neighbors> neighbors;
  /// Neighbor structures, by the buffer of their flag: the buffer that
  /// holds the configuration that the structure was built at.
  llvm::DenseMap<Value, Value> references;
  /// Particle sets: the number of particles.
  llvm::DenseMap<Attribute, Value> sizes;

  /// The scope of the body of the function that is being lowered.
  Scope *root = nullptr;
  llvm::DenseSet<Type> templatesAdded;
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

void Lowering::createReduction(OpBuilder &builder, Location loc,
                               ArrayRef<Value> values, bool isSum) {
  auto reduce = scf::ReduceOp::create(builder, loc, values);
  for (Region &region : reduce->getRegions()) {
    Block &block = region.front();
    OpBuilder combiner(context);
    combiner.setInsertionPointToEnd(&block);
    Value lhs = block.getArgument(0);
    Value rhs = block.getArgument(1);
    Value result;
    if (isSum) {
      result = arith::AddFOp::create(combiner, loc, lhs, rhs);
    } else {
      // A comparison and a selection: the form of a maximum that the
      // lowering to OpenMP recognizes.
      Value larger = arith::CmpFOp::create(
          combiner, loc, arith::CmpFPredicate::OGT, lhs, rhs);
      result = arith::SelectOp::create(combiner, loc, larger, lhs, rhs);
    }
    scf::ReduceReturnOp::create(combiner, loc, result);
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

  // The displacement is computed in the type of the positions and then
  // converted to the type that the kernel computes in: the subtraction is
  // the step that loses precision.
  Type real =
      cast<md::FieldType>(op.getPositions().getType()).getElementType();
  Type computed = kernel.getArgument(0).getType();
  box = convertReal(builder, loc, box, real);
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  Value cutoff2 = createReal(builder, loc, real, cutoff * cutoff);

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
              local.map(kernel.getArgument(0),
                        convertReal(pair, loc, r2, computed));
              local.map(kernel.getArgument(1),
                        convertReal(pair, loc, d, computed));
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

/// The name of the instance of the template function `name` for positions
/// of the type `real`.
static std::string getInstanceName(StringRef name, Type real) {
  return real.isF64() ? name.str() : (name + "_f32").str();
}

/// The text of the templates for positions of the type `real`. The
/// templates are written for `f64`.
static std::string instantiateTemplates(StringRef text, Type real) {
  if (real.isF64())
    return text.str();

  std::string instance;
  StringRef prefix = "@mdrt.";
  while (!text.empty()) {
    if (text.consume_front("f64")) {
      instance += "f32";
    } else if (text.consume_front(prefix)) {
      StringRef name = text.take_while(
          [](char c) { return llvm::isAlnum(c) || c == '_'; });
      text = text.drop_front(name.size());
      instance += getInstanceName((prefix + name).str(), real);
    } else {
      instance += text.front();
      text = text.drop_front();
    }
  }
  return instance;
}

LogicalResult Lowering::addTemplates(Type real) {
  if (!templatesAdded.insert(real).second)
    return success();

  ParserConfig config(context);
  OwningOpRef<ModuleOp> templates = parseSourceString<ModuleOp>(
      instantiateTemplates(neighborsMatrixTemplate, real), config);
  if (!templates)
    return module.emitError() << "cannot parse the neighbor build template";
  for (Operation &op : llvm::make_early_inc_range(*templates)) {
    op.remove();
    module.push_back(&op);
  }
  return success();
}

Neighbors Lowering::allocateNeighbors(Location loc, Value size,
                                      int64_t width) {
  // The storage outlives the iterations of any loop around the structure,
  // so it is allocated in the body of the function.
  OpBuilder &builder = root->builder;
  Type narrow = builder.getI32Type();
  Type real = builder.getF64Type();

  Neighbors structure;
  structure.width = createIndex(builder, loc, width);
  structure.counts = memref::AllocOp::create(
      builder, loc, MemRefType::get({ShapedType::kDynamic}, narrow),
      ValueRange{size});
  structure.index = memref::AllocOp::create(
      builder, loc,
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, narrow),
      ValueRange{size, structure.width});
  structure.box =
      memref::AllocOp::create(builder, loc, MemRefType::get({3}, real));
  structure.valid = memref::AllocOp::create(
      builder, loc, MemRefType::get({}, builder.getI1Type()));
  structure.builds = memref::AllocOp::create(
      builder, loc, MemRefType::get({}, builder.getI64Type()));

  Value no = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                       builder.getBoolAttr(false));
  Value none = arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                         builder.getI64IntegerAttr(0));
  memref::StoreOp::create(builder, loc, no, structure.valid, ValueRange{});
  memref::StoreOp::create(builder, loc, none, structure.builds, ValueRange{});
  return structure;
}

Value Lowering::convertReal(OpBuilder &builder, Location loc, Value value,
                            Type real) {
  Type source = value.getType();
  Type target = real;
  if (auto vector = dyn_cast<VectorType>(source))
    target = VectorType::get(vector.getShape(), real);
  if (source == target)
    return value;
  if (getElementTypeOrSelf(source).getIntOrFloatBitWidth() <
      real.getIntOrFloatBitWidth())
    return arith::ExtFOp::create(builder, loc, target, value);
  return arith::TruncFOp::create(builder, loc, target, value);
}

LogicalResult Lowering::getReference(Operation *op,
                                     const Neighbors &structure,
                                     Value positions, Value size,
                                     Value &reference) {
  Value &known = references[structure.valid];
  if (!known)
    known = memref::AllocOp::create(root->builder, op->getLoc(),
                                    cast<MemRefType>(positions.getType()),
                                    ValueRange{size});
  if (known.getType() != positions.getType())
    return op->emitOpError()
           << "the neighbor structure was built at positions that are stored "
              "in "
           << known.getType() << ", but these are stored in "
           << positions.getType();
  reference = known;
  return success();
}

LogicalResult Lowering::emitBuild(Operation *op, OpBuilder &builder,
                                  Location loc, const Neighbors &structure,
                                  Value positions, Value box, Value size,
                                  double reach, double cellWidth) {
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  if (failed(addTemplates(real)))
    return failure();
  Value reference;
  if (failed(getReference(op, structure, positions, size, reference)))
    return failure();

  Value reachValue = createReal(builder, loc, real, reach);
  Value widthValue = createReal(builder, loc, real, cellWidth);

  auto build = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module, getInstanceName(buildNeighborsName, real)));
  auto call = func::CallOp::create(
      builder, loc, build,
      ValueRange{positions, convertReal(builder, loc, box, real), reachValue,
                 widthValue, structure.counts, structure.index});
  Value largest = call.getResult(0);

  // A row that is too narrow loses pairs. Stop.
  Type wide = builder.getI64Type();
  auto report = getOrDeclare(
      reportOverflowName, builder.getFunctionType({wide, wide}, {}));
  Value tooMany = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ugt, largest, structure.width);
  scf::IfOp::create(
      builder, loc, tooMany, [&](OpBuilder &then, Location) {
        Value needed = arith::IndexCastOp::create(then, loc, wide, largest);
        Value available =
            arith::IndexCastOp::create(then, loc, wide, structure.width);
        func::CallOp::create(then, loc, report,
                             ValueRange{needed, available});
        scf::YieldOp::create(then, loc);
      });

  // Remember the configuration that the structure was built at.
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      [&](OpBuilder &body, Location, ValueRange ivs) {
        storeElement(body, loc, loadElement(body, loc, positions, ivs[0]),
                     reference, ivs[0]);
      });
  for (int64_t c = 0; c < 3; ++c) {
    Value edge = vector::ExtractOp::create(builder, loc, box, c);
    memref::StoreOp::create(builder, loc, edge, structure.box,
                            ValueRange{createIndex(builder, loc, c)});
  }

  Value yes = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(true));
  memref::StoreOp::create(builder, loc, yes, structure.valid, ValueRange{});

  Value builds =
      memref::LoadOp::create(builder, loc, structure.builds, ValueRange{});
  Value increment = arith::ConstantOp::create(
      builder, loc, wide, builder.getI64IntegerAttr(1));
  Value more = arith::AddIOp::create(builder, loc, builds, increment);
  memref::StoreOp::create(builder, loc, more, structure.builds, ValueRange{});
  return success();
}

LogicalResult Lowering::lowerBuildNeighbors(md_exec::BuildNeighborsOp op,
                                            Scope &scope) {
  Location loc = op.getLoc();
  auto cells = op.getCells().getDefiningOp<md_exec::BuildCellsOp>();
  if (!cells)
    return op.emitOpError() << "expected cells that are the result of "
                               "'md_exec.build_cells'";

  Value positions;
  if (failed(getBuffer(op.getPositions(), scope, positions)))
    return failure();
  Value size;
  if (failed(getSize(op, op.getPositions().getType(), size)))
    return failure();

  Neighbors structure = allocateNeighbors(loc, size, op.getWidth());
  neighbors[op.getResult()] = structure;

  double cutoff = op.getCutoff().convertToDouble();
  double skin = op.getSkin().convertToDouble();
  return emitBuild(op, scope.builder, loc, structure, positions,
                   mapping.lookup(op.getCell()), size, cutoff + skin,
                   cells.getWidth().convertToDouble());
}

LogicalResult
Lowering::lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op,
                                Scope &scope) {
  Location loc = op.getLoc();
  OpBuilder &builder = scope.builder;

  auto found = neighbors.find(op.getNeighbors());
  if (found == neighbors.end())
    return op.emitOpError() << "the neighbor structure has no storage";
  Neighbors structure = found->second;
  // The structure is refreshed where it is.
  neighbors[op.getResult()] = structure;

  Value positions;
  if (failed(getBuffer(op.getPositions(), scope, positions)))
    return failure();
  Value box = mapping.lookup(op.getCell());
  Value size;
  if (failed(getSize(op, op.getPositions().getType(), size)))
    return failure();

  Type real = cast<MemRefType>(positions.getType()).getElementType();
  double cutoff = op.getCutoff().convertToDouble();
  double skin = op.getSkin().convertToDouble();
  Value reference;
  if (failed(getReference(op, structure, positions, size, reference)))
    return failure();

  // The structure is valid if it has been built, in this cell, and no
  // particle has moved more than half the skin since.
  Value valid =
      memref::LoadOp::create(builder, loc, structure.valid, ValueRange{});
  for (int64_t c = 0; c < 3; ++c) {
    Value edge = vector::ExtractOp::create(builder, loc, box, c);
    Value built = memref::LoadOp::create(
        builder, loc, structure.box,
        ValueRange{createIndex(builder, loc, c)});
    Value same = arith::CmpFOp::create(builder, loc,
                                       arith::CmpFPredicate::OEQ, edge, built);
    valid = arith::AndIOp::create(builder, loc, valid, same);
  }

  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  Value none = createZero(builder, loc, real);
  auto farthest = scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      ValueRange{none},
      [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        Value now = loadElement(body, loc, positions, ivs[0]);
        Value then = loadElement(body, loc, reference, ivs[0]);
        Value moved = arith::SubFOp::create(body, loc, now, then);
        Value squares = arith::MulFOp::create(body, loc, moved, moved);
        Value distance2 = vector::ReductionOp::create(
            body, loc, vector::CombiningKind::ADD, squares);
        createReduction(body, loc, {distance2}, /*isSum=*/false);
      });
  Value limit = createReal(builder, loc, real, 0.25 * skin * skin);
  Value near = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OLE, farthest.getResult(0), limit);
  valid = arith::AndIOp::create(builder, loc, valid, near);

  Value yes = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(true));
  Value stale = arith::XOrIOp::create(builder, loc, valid, yes);

  LogicalResult status = success();
  scf::IfOp::create(
      builder, loc, stale, [&](OpBuilder &then, Location) {
        status = emitBuild(op, then, loc, structure, positions, box, size,
                           cutoff + skin,
                           op.getCellWidth().convertToDouble());
        scf::YieldOp::create(then, loc);
      });
  return status;
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

  // A neighbor structure is refreshed where it is, so the loop need not
  // carry its storage. `carried[i]` is the position of loop-carried value
  // `i` among the values that the new loop carries, or -1.
  SmallVector<int> carried;
  SmallVector<Value> inits;
  for (auto [init, argument] :
       llvm::zip(op.getInitArgs(), op.getRegionIterArgs())) {
    if (isa<mdrt::NeighborsType>(init.getType())) {
      auto found = neighbors.find(init);
      if (found == neighbors.end())
        return op.emitOpError() << "a neighbor structure has no storage";
      neighbors[argument] = found->second;
      carried.push_back(-1);
      continue;
    }
    carried.push_back(inits.size());
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

    Value inside = inner.body->addArgument(buffer.getType(), loc);
    buffers[argument] = inside;
    inner.owned.insert(inside);
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

  unsigned numCarried = inits.size();
  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
    Value oldResult = op.getResult(i);
    if (carried[i] < 0) {
      neighbors[oldResult] = neighbors.lookup(op.getInitArgs()[i]);
      continue;
    }
    Value newResult = loop->getResult(carried[i]);
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
    if (isa<mdrt::NeighborsType>(value.getType())) {
      if (!neighbors.count(value))
        return op.emitOpError() << "a neighbor structure has no storage";
      continue;
    }
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
  if (auto refresh = dyn_cast<md_exec::RefreshNeighborsOp>(op))
    return lowerRefreshNeighbors(refresh, scope);
  if (auto empty = dyn_cast<md_exec::EmptyNeighborsOp>(op)) {
    auto type = cast<mdrt::NeighborsType>(empty.getResult().getType());
    Value size = sizes.lookup(type.getParticleSet());
    if (!size)
      return op->emitOpError()
             << "the number of particles of " << type.getParticleSet()
             << " is not known here: no field of the set has a buffer yet";
    neighbors[empty.getResult()] =
        allocateNeighbors(loc, size, empty.getWidth());
    return success();
  }
  if (auto count = dyn_cast<md_exec::RebuildCountOp>(op)) {
    auto found = neighbors.find(count.getNeighbors());
    if (found == neighbors.end())
      return op->emitOpError() << "the neighbor structure has no storage";
    mapping.map(count.getResult(),
                memref::LoadOp::create(builder, loc, found->second.builds,
                                       ValueRange{}));
    return success();
  }
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
