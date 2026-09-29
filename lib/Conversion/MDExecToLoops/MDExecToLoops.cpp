// Conversion of md_exec loops in the storage form to loops over buffers.
//
// Every op of md_exec becomes `scf` loops that load from and store to
// `memref`s, where the op is. See docs/ops-m0.md, Section 10.

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
  /// The number of particles.
  Value size;
  /// The number of neighbors of each particle, and their indices.
  Value counts;
  Value index;
  /// The number of neighbors that a row holds.
  Value width;
  /// The configuration and the cell that the structure was built at.
  Value reference;
  Value box;
  /// Whether the structure has been built, and how often.
  Value valid;
  Value builds;
};

class Lowering {
public:
  explicit Lowering(ModuleOp module)
      : module(module), context(module.getContext()) {}

  LogicalResult run();

private:
  LogicalResult lowerFunction(func::FuncOp function);
  LogicalResult lowerOp(Operation *op);

  void lowerEmptyNeighbors(md_exec::EmptyNeighborsOp op);
  LogicalResult lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op);
  void lowerParticleFor(md_exec::ParticleForOp op);
  LogicalResult lowerPairFor(md_exec::PairForOp op);

  /// The storage of the neighbor structure `structure`.
  LogicalResult getNeighbors(Operation *op, Value structure,
                             Neighbors &storage);

  /// Builds `structure` at the configuration `positions`.
  LogicalResult emitBuild(OpBuilder &builder, Location loc,
                          const Neighbors &structure, Value positions,
                          Value box, double reach, double cellWidth);

  /// `value`, a floating-point value or a vector of them, converted to the
  /// floating-point type `real`.
  Value convertReal(OpBuilder &builder, Location loc, Value value, Type real);
  Value createReal(OpBuilder &builder, Location loc, Type real,
                   double value) {
    return arith::ConstantOp::create(builder, loc, real,
                                     builder.getFloatAttr(real, value));
  }

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
  /// The number of particles that `buffer` holds a field of.
  Value createSize(OpBuilder &builder, Location loc, Value buffer) {
    return memref::DimOp::create(builder, loc, buffer,
                                 createIndex(builder, loc, 0));
  }

  /// Ends the body of a parallel loop with a reduction of `values`: their
  /// sum, or their maximum.
  void createReduction(OpBuilder &builder, Location loc,
                       ArrayRef<Value> values, bool isSum);

  /// Adds the templates for positions of the type `real` to the module.
  LogicalResult addTemplates(Type real);
  func::FuncOp getOrDeclare(StringRef name, FunctionType type);

  ModuleOp module;
  MLIRContext *context;

  llvm::DenseMap<Value, Neighbors> neighbors;
  llvm::DenseSet<Type> templatesAdded;

  /// The ops that have been lowered, in the order of the program.
  SmallVector<Operation *> lowered;
};

} // namespace

//===----------------------------------------------------------------------===//
// Elements
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Loops over particles
//===----------------------------------------------------------------------===//

void Lowering::lowerParticleFor(md_exec::ParticleForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  SmallVector<Value> ins(op.getIns().begin(), op.getIns().end());
  SmallVector<Value> outs(op.getOuts().begin(), op.getOuts().end());
  SmallVector<Value> inits(op.getReduce().begin(), op.getReduce().end());
  Value size = createSize(builder, loc, ins.empty() ? outs.front()
                                                     : ins.front());

  Block &kernel = op.getKernel().front();
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  auto loop = scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      inits, [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        Value particle = ivs[0];
        IRMapping local;
        for (unsigned i = 0, e = ins.size(); i != e; ++i)
          local.map(kernel.getArgument(i),
                    loadElement(body, loc, ins[i], particle));
        for (Operation &nested : kernel.without_terminator())
          body.clone(nested, local);

        Operation *yield = kernel.getTerminator();
        unsigned numOuts = outs.size();
        for (unsigned i = 0; i != numOuts; ++i)
          storeElement(body, loc,
                       local.lookupOrDefault(yield->getOperand(i)), outs[i],
                       particle);

        SmallVector<Value> contributions;
        for (unsigned i = numOuts, e = yield->getNumOperands(); i != e; ++i)
          contributions.push_back(
              local.lookupOrDefault(yield->getOperand(i)));
        if (!contributions.empty())
          createReduction(body, loc, contributions, /*isSum=*/true);
      });

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    op.getResult(i).replaceAllUsesWith(loop.getResult(i));
}

//===----------------------------------------------------------------------===//
// Loops over pairs
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lowerPairFor(md_exec::PairForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  Neighbors structure;
  if (failed(getNeighbors(op, op.getNeighbors(), structure)))
    return failure();

  Value positions = op.getPositions();
  // The cell has become the vector of its edge lengths.
  Value box = op.getCellMutable().get();
  SmallVector<Value> ins(op.getIns().begin(), op.getIns().end());
  SmallVector<Value> outs(op.getOuts().begin(), op.getOuts().end());
  SmallVector<Value> inits(op.getReduce().begin(), op.getReduce().end());
  Value size = createSize(builder, loc, positions);

  double cutoff = op.getCutoff().convertToDouble();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  unsigned numOuts = outs.size();
  unsigned numYields = yield->getNumOperands();

  // The displacement is computed in the type of the positions and then
  // converted to the type that the kernel computes in: the subtraction is
  // the step that loses precision.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
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

              IRMapping local;
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
                Value contribution =
                    local.lookupOrDefault(yield->getOperand(i));
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
          if (!op.overwrites(i))
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
              FloatAttr scalar = body.getFloatAttr(
                  getElementTypeOrSelf(type), weight);
              Value factor;
              if (auto vector = dyn_cast<VectorType>(type))
                factor = arith::ConstantOp::create(
                    body, loc, type,
                    DenseElementsAttr::get(vector, scalar.getValue()));
              else
                factor = arith::ConstantOp::create(body, loc, type, scalar);
              total = arith::MulFOp::create(body, loc, factor, total);
            }
          }
          contributions.push_back(total);
        }
        if (!contributions.empty())
          createReduction(body, loc, contributions, /*isSum=*/true);
      });

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    op.getResult(i).replaceAllUsesWith(loop.getResult(i));
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
  // A module that was lowered in part has the templates already.
  if (SymbolTable::lookupSymbolIn(
          module, getInstanceName(buildNeighborsName, real)))
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

LogicalResult Lowering::getNeighbors(Operation *op, Value structure,
                                     Neighbors &storage) {
  auto found = neighbors.find(structure);
  if (found == neighbors.end())
    return op->emitOpError()
           << "the neighbor structure has no storage; only the result of "
              "'md_exec.empty_neighbors' in the storage form, in the same "
              "function, is supported";
  storage = found->second;
  return success();
}

void Lowering::lowerEmptyNeighbors(md_exec::EmptyNeighborsOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Type narrow = builder.getI32Type();
  Type real = *op.getElement();

  Neighbors structure;
  structure.size = op.getSize();
  structure.width = createIndex(builder, loc, op.getWidth());
  structure.counts = memref::AllocOp::create(
      builder, loc, MemRefType::get({ShapedType::kDynamic}, narrow),
      ValueRange{structure.size});
  structure.index = memref::AllocOp::create(
      builder, loc,
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, narrow),
      ValueRange{structure.size, structure.width});
  structure.reference = memref::AllocOp::create(
      builder, loc, MemRefType::get({ShapedType::kDynamic, 3}, real),
      ValueRange{structure.size});
  structure.box = memref::AllocOp::create(
      builder, loc, MemRefType::get({3}, builder.getF64Type()));
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
  neighbors[op.getResult()] = structure;
}

LogicalResult Lowering::emitBuild(OpBuilder &builder, Location loc,
                                  const Neighbors &structure, Value positions,
                                  Value box, double reach,
                                  double cellWidth) {
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  if (failed(addTemplates(real)))
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
      builder, loc, ValueRange{zero}, ValueRange{structure.size},
      ValueRange{one}, [&](OpBuilder &body, Location, ValueRange ivs) {
        storeElement(body, loc, loadElement(body, loc, positions, ivs[0]),
                     structure.reference, ivs[0]);
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

LogicalResult
Lowering::lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  Neighbors structure;
  if (failed(getNeighbors(op, op.getNeighbors(), structure)))
    return failure();
  // The structure is refreshed where it is.
  neighbors[op.getResult()] = structure;

  Value positions = op.getPositions();
  // The cell has become the vector of its edge lengths.
  Value box = op.getCellMutable().get();
  if (positions.getType() != structure.reference.getType())
    return op.emitOpError()
           << "the storage of the neighbor structure is for positions that "
              "are stored in "
           << structure.reference.getType() << ", but these are stored in "
           << positions.getType();

  Type real = cast<MemRefType>(positions.getType()).getElementType();
  double reach =
      op.getCutoff().convertToDouble() + op.getSkin().convertToDouble();
  double skin = op.getSkin().convertToDouble();
  double cellWidth = op.getCellWidth().convertToDouble();

  if (op.getPolicy() == md_exec::RebuildPolicy::Always)
    return emitBuild(builder, loc, structure, positions, box, reach,
                     cellWidth);

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
      builder, loc, ValueRange{zero}, ValueRange{structure.size},
      ValueRange{one}, ValueRange{none},
      [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        Value now = loadElement(body, loc, positions, ivs[0]);
        Value then = loadElement(body, loc, structure.reference, ivs[0]);
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
        status = emitBuild(then, loc, structure, positions, box, reach,
                           cellWidth);
        scf::YieldOp::create(then, loc);
      });
  return status;
}

//===----------------------------------------------------------------------===//
// Ops and functions
//===----------------------------------------------------------------------===//

/// Returns true if `type` belongs to the value form: a field, or a
/// structure that the storage form does not have.
static bool isValueFormType(Type type) {
  return isa<md::FieldType, md::RelationType, mdrt::CellsType,
             mdrt::PermutationType>(type);
}

LogicalResult Lowering::lowerOp(Operation *op) {
  if (llvm::any_of(op->getOperandTypes(), isValueFormType) ||
      llvm::any_of(op->getResultTypes(), isValueFormType))
    return op->emitOpError()
           << "is not in the storage form; run 'md-exec-assign-storage' "
              "first";

  if (auto empty = dyn_cast<md_exec::EmptyNeighborsOp>(op)) {
    if (!empty.isStorageForm())
      return op->emitOpError()
             << "is not in the storage form; run 'md-exec-assign-storage' "
                "first";
    lowerEmptyNeighbors(empty);
  } else if (auto refresh = dyn_cast<md_exec::RefreshNeighborsOp>(op)) {
    if (failed(lowerRefreshNeighbors(refresh)))
      return failure();
  } else if (auto reset = dyn_cast<md_exec::ResetNeighborsOp>(op)) {
    Neighbors structure;
    if (failed(getNeighbors(op, reset.getNeighbors(), structure)))
      return failure();
    OpBuilder builder(op);
    Location loc = op->getLoc();
    Value no = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                         builder.getBoolAttr(false));
    Value none = arith::ConstantOp::create(
        builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
    memref::StoreOp::create(builder, loc, no, structure.valid, ValueRange{});
    memref::StoreOp::create(builder, loc, none, structure.builds,
                            ValueRange{});
  } else if (auto count = dyn_cast<md_exec::RebuildCountOp>(op)) {
    Neighbors structure;
    if (failed(getNeighbors(op, count.getNeighbors(), structure)))
      return failure();
    OpBuilder builder(op);
    count.getResult().replaceAllUsesWith(memref::LoadOp::create(
        builder, op->getLoc(), structure.builds, ValueRange{}));
  } else if (auto loop = dyn_cast<md_exec::ParticleForOp>(op)) {
    lowerParticleFor(loop);
  } else if (auto loop = dyn_cast<md_exec::PairForOp>(op)) {
    if (failed(lowerPairFor(loop)))
      return failure();
  } else if (auto cell = dyn_cast<md::OrthorhombicCellOp>(op)) {
    OpBuilder builder(op);
    Type real = builder.getF64Type();
    cell.getResult().replaceAllUsesWith(vector::FromElementsOp::create(
        builder, op->getLoc(), VectorType::get({3}, real),
        ValueRange{cell.getLx(), cell.getLy(), cell.getLz()}));
  } else if (isa<md::MDDialect>(op->getDialect()) ||
             isa<md_exec::MDExecDialect>(op->getDialect()) ||
             isa<mdrt::MDRTDialect>(op->getDialect()) ||
             op->getName().getDialectNamespace() == "dyn") {
    return op->emitOpError()
           << "cannot be lowered; run 'md-differentiate', 'md-inline', "
              "'convert-md-to-md-exec', and 'md-exec-assign-storage' first";
  } else {
    // The op stays. It must not use what the lowering removes.
    if (llvm::any_of(op->getOperandTypes(), [](Type type) {
          return isa<mdrt::NeighborsType>(type);
        }))
      return op->emitOpError()
             << "cannot be lowered: the op is not known to the pass and "
                "uses a neighbor structure";
    return success();
  }

  lowered.push_back(op);
  return success();
}

LogicalResult Lowering::lowerFunction(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  Type box = VectorType::get({3}, Float64Type::get(context));
  auto convertType = [&](Type type) -> Type {
    return isa<md::CellType>(type) ? box : type;
  };

  SmallVector<Type> inputs, results;
  for (Type part : llvm::concat<const Type>(type.getInputs(),
                                            type.getResults())) {
    if (isValueFormType(part))
      return function.emitOpError()
             << "has " << part << " in its signature, which is not in the "
             << "storage form; run 'md-exec-assign-storage' first";
    if (isa<mdrt::NeighborsType>(part))
      return function.emitOpError()
             << "cannot lower " << part << " in a signature";
  }
  for (Type input : type.getInputs())
    inputs.push_back(convertType(input));
  for (Type result : type.getResults())
    results.push_back(convertType(result));
  function.setType(FunctionType::get(context, inputs, results));
  if (function.isExternal())
    return success();

  SmallVector<Operation *> ops;
  function.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op != function.getOperation())
      ops.push_back(op);
  });

  // A cell is the vector of its edge lengths.
  for (Block &block : function.getBody())
    for (BlockArgument argument : block.getArguments())
      argument.setType(convertType(argument.getType()));
  for (Operation *op : ops) {
    if (isa<md::OrthorhombicCellOp>(op))
      continue;
    for (Value result : op->getResults())
      result.setType(convertType(result.getType()));
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          argument.setType(convertType(argument.getType()));
  }

  // The kernels are copied into the loops, so the ops inside them are not
  // lowered where they are.
  for (Operation *op : ops) {
    Operation *parent = op->getParentOp();
    if (isa<md_exec::ParticleForOp, md_exec::PairForOp>(parent))
      continue;
    if (failed(lowerOp(op)))
      return failure();
  }

  // Users come after what they use, so erase from the back.
  for (Operation *op : llvm::reverse(lowered))
    op->erase();
  lowered.clear();
  neighbors.clear();
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
