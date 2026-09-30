// Conversion of md_exec loops in the storage form to loops over buffers.
//
// Every op of md_exec becomes `scf` loops that load from and store to
// `memref`s, where the op is. See docs/ops-m0.md, Section 10.

#include "mdir/Conversion/Passes.h"

#include "mdir/Conversion/MDExecKernels.h"
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
using namespace mdir::kernels;

namespace mdir {
/// The text of the template that builds a neighbor matrix.
extern const char *const neighborsMatrixTemplate;
extern const char *const pmeTemplate;
} // namespace mdir

static const char *const spatialOrderName = "mdrt.spatial_order";
static const char *const cellWidthName = "mdrt.cell_width";
static const char *const buildNeighborsName = "mdrt.build_neighbors_matrix";
static const char *const reportOverflowName = "mdrtReportNeighborOverflow";
static const char *const countBuildName = "mdrtCountBuild";

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
  /// The incidence structure of the pairs that the structure leaves out,
  /// or null.
  Value excluded;
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
  LogicalResult lowerSpatialOrder(md_exec::SpatialOrderOp op);
  void lowerPermute(md_exec::PermuteOp op);
  LogicalResult lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op);
  void lowerParticleFor(md_exec::ParticleForOp op);
  LogicalResult lowerPairFor(md_exec::PairForOp op);
  void lowerTupleFor(md_exec::TupleForOp op);

  /// The storage of the neighbor structure `structure`.
  LogicalResult getNeighbors(Operation *op, Value structure,
                             Neighbors &storage);

  /// Builds `structure` at the configuration `positions`.
  LogicalResult emitBuild(OpBuilder &builder, Location loc,
                          const Neighbors &structure, Value positions,
                          Value box, double reach, double cellWidth);

  /// The number of particles that `buffer` holds a field of.
  Value createSize(OpBuilder &builder, Location loc, Value buffer) {
    return memref::DimOp::create(builder, loc, buffer,
                                 createIndex(builder, loc, 0));
  }

  /// Ends the body of a parallel loop with a reduction of `values`: their
  /// sum, or their maximum. Of integers it tells whether any is not zero.
  void createReduction(OpBuilder &builder, Location loc,
                       ArrayRef<Value> values, bool isSum);

  /// Adds the templates for positions of the type `real` to the module.
  LogicalResult addTemplates(Type real);
  /// Adds the templates of particle mesh Ewald for the types of the
  /// positions, the charges, and the forces.
  LogicalResult addPMETemplates(Type position, Type charge, Type force,
                                int64_t order);
  LogicalResult lowerReciprocal(md_exec::ReciprocalOp op);
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
// Reductions
//===----------------------------------------------------------------------===//

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
    if (lhs.getType().isInteger()) {
      result = arith::OrIOp::create(combiner, loc, lhs, rhs);
    } else if (isSum) {
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
// Loops over particles
//===----------------------------------------------------------------------===//

void Lowering::lowerParticleFor(md_exec::ParticleForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // A value of the type i1 is reduced as an integer of 32 bits: threads
  // combine their results with atomic operations, which take no integer
  // of one bit.
  Type wide = builder.getI32Type();
  auto widen = [&](OpBuilder &b, Value value) -> Value {
    if (!value.getType().isInteger(1))
      return value;
    return arith::ExtUIOp::create(b, loc, wide, value);
  };

  SmallVector<Value> inits;
  for (Value value : op.getReduce())
    inits.push_back(widen(builder, value));
  Value size = createSize(builder, loc, op.getIns().empty()
                                            ? op.getOuts().front()
                                            : op.getIns().front());

  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  auto loop = scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      inits, [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        IRMapping local;
        SmallVector<Value> contributions =
            emitParticleKernel(body, op, ivs[0], local);
        for (Value &value : contributions)
          value = widen(body, value);
        if (!contributions.empty())
          createReduction(body, loc, contributions, /*isSum=*/true);
      });

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
    Value result = loop.getResult(i);
    if (op.getResult(i).getType().isInteger(1))
      result = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ne, result,
          arith::ConstantOp::create(builder, loc, wide,
                                    builder.getI32IntegerAttr(0)));
    op.getResult(i).replaceAllUsesWith(result);
  }
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
  SmallVector<Value> inits(op.getReduce().begin(), op.getReduce().end());
  Value size = createSize(builder, loc, positions);

  // The cell has become the vector of its edge lengths.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);

  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  auto loop = scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      inits, [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        IRMapping local;
        SmallVector<Value> contributions =
            emitPairKernel(body, op, structure.counts, structure.index, box,
                           inverse, ivs[0], local);
        if (!contributions.empty())
          createReduction(body, loc, contributions, /*isSum=*/true);
      });

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    op.getResult(i).replaceAllUsesWith(loop.getResult(i));
  return success();
}

//===----------------------------------------------------------------------===//
// Loops over tuples
//===----------------------------------------------------------------------===//

void Lowering::lowerTupleFor(md_exec::TupleForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  Value positions = op.getPositions();
  SmallVector<Value> inits(op.getReduce().begin(), op.getReduce().end());
  Value size = createSize(builder, loc, positions);

  // The cell has become the vector of its edge lengths.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);

  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  auto loop = scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      inits, [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        IRMapping local;
        SmallVector<Value> contributions = emitTupleKernel(
            body, op, op.getIncidence(), box, inverse, ivs[0], local);
        if (!contributions.empty())
          createReduction(body, loc, contributions, /*isSum=*/true);
      });

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    op.getResult(i).replaceAllUsesWith(loop.getResult(i));
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

LogicalResult Lowering::addPMETemplates(Type position, Type charge,
                                        Type force, int64_t order) {
  if (SymbolTable::lookupSymbolIn(
          module, getPMEInstanceName("mdrt.pme_spread", position, charge,
                                     force, order)))
    return success();
  ParserConfig config(context);
  OwningOpRef<ModuleOp> templates = parseSourceString<ModuleOp>(
      instantiatePMETemplates(pmeTemplate, position, charge, force,
                              order), config);
  if (!templates)
    return module.emitError()
           << "cannot parse the template of particle mesh Ewald";
  for (Operation &op : llvm::make_early_inc_range(*templates)) {
    op.remove();
    module.push_back(&op);
  }
  return success();
}

LogicalResult Lowering::lowerReciprocal(md_exec::ReciprocalOp op) {
  if (!op.isStorageForm())
    return op->emitOpError()
           << "is not in the storage form; run 'md-exec-assign-storage' "
              "first";
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value positions = op.getPositions(), charges = op.getCharges();
  Value forces = op.getOut();
  auto elementOf = [](Value buffer) {
    return cast<MemRefType>(buffer.getType()).getElementType();
  };
  Type position = elementOf(positions), charge = elementOf(charges),
       force = elementOf(forces);
  if (failed(addPMETemplates(position, charge, force, op.getOrder())))
    return failure();
  auto instance = [&](StringRef name) {
    return cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
        module, getPMEInstanceName(name, position, charge, force, op.getOrder())));
  };

  ArrayRef<int64_t> grid = op.getGrid();
  Value k1 = createIndex(builder, loc, grid[0]);
  Value k2 = createIndex(builder, loc, grid[1]);
  Value k3 = createIndex(builder, loc, grid[2]);
  Value order = createIndex(builder, loc, op.getOrder());
  Value beta = createReal(builder, loc, builder.getF64Type(),
                          op.getBeta().convertToDouble());
  Value coulomb = createReal(builder, loc, builder.getF64Type(),
                             op.getCoulomb().convertToDouble());
  // The cell is the vector of its edge lengths by now.
  Value box = op.getCellMutable().get();
  Value fixed = op.getScratch()[0], real = op.getScratch()[1],
        complex = op.getScratch()[2];

  func::CallOp::create(builder, loc, instance("mdrt.pme_spread"),
                       ValueRange{positions, charges, box, fixed, k1, k2, k3,
                                  order});
  func::CallOp::create(builder, loc, instance("mdrt.pme_real"),
                       ValueRange{fixed, real});
  // The transforms of the runtime take the numbers of points as i64.
  Type wide = builder.getI64Type();
  SmallVector<Value> sizes;
  for (int64_t points : grid)
    sizes.push_back(arith::ConstantOp::create(builder, loc, wide,
                                              builder.getI64IntegerAttr(points)));
  Type buffer = real.getType();
  FunctionType transform =
      builder.getFunctionType({buffer, buffer, wide, wide, wide}, {});
  func::FuncOp forward = getOrDeclare("mdrtFFTForward3D", transform);
  func::FuncOp backward = getOrDeclare("mdrtFFTBackward3D", transform);
  for (func::FuncOp function : {forward, backward})
    function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func::CallOp::create(builder, loc, forward,
                       ValueRange{real, complex, sizes[0], sizes[1], sizes[2]});
  auto convolve = func::CallOp::create(
      builder, loc, instance("mdrt.pme_convolve"),
      ValueRange{complex, op.getModuli(), box, beta, coulomb, k1, k2, k3});
  func::CallOp::create(builder, loc, backward,
                       ValueRange{complex, real, sizes[0], sizes[1], sizes[2]});
  func::CallOp::create(builder, loc, instance("mdrt.pme_gather"),
                       ValueRange{positions, charges, real, box, k1, k2, k3,
                                  order, forces});
  op.getEnergy().replaceAllUsesWith(convolve.getResult(0));
  op.getVirial().replaceAllUsesWith(convolve.getResult(1));
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
  Type real = cast<MemRefType>(*op.getPositions()).getElementType();

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
  structure.excluded = op.getExcluded();
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
  Value leastValue = createReal(builder, loc, real, cellWidth);
  Value boxValue = convertReal(builder, loc, box, real);

  // The width of the cells follows from the density, which is known when
  // the structure is built.
  auto choose = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module, getInstanceName(cellWidthName, real)));
  Value widthValue =
      func::CallOp::create(builder, loc, choose,
                           ValueRange{structure.size, boxValue, reachValue,
                                      leastValue})
          .getResult(0);

  auto build = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module, getInstanceName(buildNeighborsName, real)));
  auto call = func::CallOp::create(
      builder, loc, build,
      ValueRange{positions, boxValue, reachValue,
                 widthValue, structure.counts, structure.index});
  Value largest = call.getResult(0);

  // The runtime counts the builds, for the log of the run.
  func::CallOp::create(
      builder, loc,
      getOrDeclare(countBuildName, builder.getFunctionType({}, {})),
      ValueRange());

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

  // Leave the excluded pairs out.
  if (structure.excluded)
    scf::ParallelOp::create(
        builder, loc, ValueRange{createIndex(builder, loc, 0)},
        ValueRange{structure.size}, ValueRange{createIndex(builder, loc, 1)},
        [&](OpBuilder &body, Location, ValueRange ivs) {
          emitExclusionFilter(body, loc, structure.counts, structure.index,
                              structure.excluded, ivs[0]);
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
  // A barostat scales the positions with the cell: the reference is
  // compared scaled as the cell was, m = L / L_ref, against half of
  // min(m) R − r_c (D80).
  SmallVector<Value, 3> builtEdges;
  for (int64_t c = 0; c < 3; ++c)
    builtEdges.push_back(memref::LoadOp::create(
        builder, loc, structure.box, ValueRange{createIndex(builder, loc, c)}));
  Value scale = arith::DivFOp::create(
      builder, loc, box,
      vector::FromElementsOp::create(
          builder, loc, VectorType::get({3}, builder.getF64Type()),
          builtEdges));
  Value least = vector::ReductionOp::create(
      builder, loc, vector::CombiningKind::MINNUMF, scale);
  Value margin = arith::SubFOp::create(
      builder, loc,
      arith::MulFOp::create(builder, loc, least,
                            createReal(builder, loc, builder.getF64Type(),
                                       reach)),
      createReal(builder, loc, builder.getF64Type(),
                 op.getCutoff().convertToDouble()));
  Value halfMargin = arith::MulFOp::create(
      builder, loc,
      arith::MaximumFOp::create(
          builder, loc, margin,
          createReal(builder, loc, builder.getF64Type(), 0.0)),
      createReal(builder, loc, builder.getF64Type(), 0.5));
  Value limit2 = arith::MulFOp::create(builder, loc, halfMargin, halfMargin);
  Value scaleReal = scale;
  if (!real.isF64())
    scaleReal = arith::TruncFOp::create(
        builder, loc, VectorType::get({3}, real), scale);
  Value limitReal = limit2;
  if (!real.isF64())
    limitReal = arith::TruncFOp::create(builder, loc, real, limit2);
  (void)skin;

  Value yes = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(true));
  Value near;
  if (Value moved = op.getMoved()) {
    // A loop has made the test.
    near = arith::XOrIOp::create(builder, loc, moved, yes);
  } else {
    Value zero = createIndex(builder, loc, 0);
    Value one = createIndex(builder, loc, 1);
    Value none = createZero(builder, loc, real);
    auto farthest = scf::ParallelOp::create(
        builder, loc, ValueRange{zero}, ValueRange{structure.size},
        ValueRange{one}, ValueRange{none},
        [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
          Value now = loadElement(body, loc, positions, ivs[0]);
          Value then = arith::MulFOp::create(
              body, loc, loadElement(body, loc, structure.reference, ivs[0]),
              scaleReal);
          Value moved = arith::SubFOp::create(body, loc, now, then);
          Value squares = arith::MulFOp::create(body, loc, moved, moved);
          Value distance2 = vector::ReductionOp::create(
              body, loc, vector::CombiningKind::ADD, squares);
          createReduction(body, loc, {distance2}, /*isSum=*/false);
        });
    Value limit = limitReal;
    near = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE,
                                 farthest.getResult(0), limit);
  }
  valid = arith::AndIOp::create(builder, loc, valid, near);

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
// The order of the particles
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lowerSpatialOrder(md_exec::SpatialOrderOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value positions = op.getPositions();
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  if (failed(addTemplates(real)))
    return failure();

  // The cell has become the vector of its edge lengths.
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value width =
      createReal(builder, loc, real, op.getWidth().convertToDouble());
  auto order = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module, getInstanceName(spatialOrderName, real)));
  func::CallOp::create(
      builder, loc, order,
      ValueRange{positions, box, width, op.getIds(), op.getOrder()});
  return success();
}

void Lowering::lowerPermute(md_exec::PermuteOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value field = op.getField();
  Value order = op.getOrder();
  Value out = op.getOut();
  Value size = createSize(builder, loc, field);
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      [&](OpBuilder &body, Location, ValueRange ivs) {
        Value from = memref::LoadOp::create(body, loc, order,
                                            ValueRange{ivs[0]});
        Value particle = arith::IndexCastOp::create(
            body, loc, body.getIndexType(), from);
        storeElement(body, loc, loadElement(body, loc, field, particle),
                     out, ivs[0]);
      });
}

//===----------------------------------------------------------------------===//
// Ops and functions
//===----------------------------------------------------------------------===//

/// Returns true if `type` is the type of a buffer on a device.
static bool isDeviceType(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  return buffer && buffer.getMemorySpace();
}

LogicalResult Lowering::lowerOp(Operation *op) {
  if (llvm::any_of(op->getOperandTypes(), md_exec::isValueFormType) ||
      llvm::any_of(op->getResultTypes(), md_exec::isValueFormType))
    return op->emitOpError()
           << "is not in the storage form; run 'md-exec-assign-storage' "
              "first";

  if (isa<md_exec::MDExecDialect>(op->getDialect())) {
    bool onDevice = llvm::any_of(op->getOperandTypes(), isDeviceType);
    if (auto empty = dyn_cast<md_exec::EmptyNeighborsOp>(op))
      if (auto positions = empty.getPositions())
        onDevice |= isDeviceType(*positions);
    if (onDevice)
      return op->emitOpError()
             << "has its buffers on a device; use "
                "'convert-md-exec-to-gpu'";
  }

  if (auto empty = dyn_cast<md_exec::EmptyNeighborsOp>(op)) {
    if (!empty.isStorageForm())
      return op->emitOpError()
             << "is not in the storage form; run 'md-exec-assign-storage' "
                "first";
    lowerEmptyNeighbors(empty);
  } else if (auto refresh = dyn_cast<md_exec::RefreshNeighborsOp>(op)) {
    if (failed(lowerRefreshNeighbors(refresh)))
      return failure();
  } else if (auto order = dyn_cast<md_exec::SpatialOrderOp>(op)) {
    if (failed(lowerSpatialOrder(order)))
      return failure();
  } else if (auto permute = dyn_cast<md_exec::PermuteOp>(op)) {
    lowerPermute(permute);
  } else if (auto cell = dyn_cast<md_exec::ReferenceCellOp>(op)) {
    // The cell that the structure was built in, as the vector of its
    // edges that cells have become.
    Neighbors structure;
    if (failed(getNeighbors(op, cell.getNeighbors(), structure)))
      return failure();
    OpBuilder builder(op);
    Location loc = op->getLoc();
    SmallVector<Value, 3> edges;
    for (int64_t c = 0; c < 3; ++c)
      edges.push_back(memref::LoadOp::create(
          builder, loc, structure.box,
          ValueRange{createIndex(builder, loc, c)}));
    cell->getResult(0).replaceAllUsesWith(vector::FromElementsOp::create(
        builder, loc, VectorType::get({3}, builder.getF64Type()), edges));
  } else if (auto edges = dyn_cast<md_exec::CellEdgesOp>(op)) {
    // The cell is the vector of its edges by now.
    edges.getResult().replaceAllUsesWith(edges->getOperand(0));
  } else if (auto reference = dyn_cast<md_exec::ReferencePositionsOp>(op)) {
    Neighbors structure;
    if (failed(getNeighbors(op, reference.getNeighbors(), structure)))
      return failure();
    if (reference.getResult().getType() != structure.reference.getType())
      return op->emitOpError()
             << "the structure holds the positions in "
             << structure.reference.getType() << ", not in "
             << reference.getResult().getType();
    reference.getResult().replaceAllUsesWith(structure.reference);
  } else if (auto reset = dyn_cast<md_exec::ResetNeighborsOp>(op)) {
    Neighbors structure;
    if (failed(getNeighbors(op, reset.getNeighbors(), structure)))
      return failure();
    if (Value excluded = reset.getExcluded())
      neighbors[reset.getNeighbors()].excluded = excluded;
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
  } else if (auto loop = dyn_cast<md_exec::TupleForOp>(op)) {
    lowerTupleFor(loop);
  } else if (auto build = dyn_cast<md_exec::BuildIncidenceOp>(op)) {
    if (!build.isStorageForm())
      return op->emitOpError()
             << "is not in the storage form; run 'md-exec-assign-storage' "
                "first";
    if (isDeviceType(build.getResult().getType()))
      return op->emitOpError() << "builds on a device; use "
                                  "'convert-md-exec-to-gpu'";
    OpBuilder builder(op);
    Value incidence = emitBuildIncidence(builder, op->getLoc(),
                                         build.getRelation(), build.getSize());
    build.getResult().replaceAllUsesWith(incidence);
    freeAtEndOfBlock(op, incidence);
  } else if (auto reciprocal = dyn_cast<md_exec::ReciprocalOp>(op)) {
    if (failed(lowerReciprocal(reciprocal)))
      return failure();
  } else if (auto renumber = dyn_cast<md_exec::RenumberOp>(op)) {
    OpBuilder builder(op);
    Value members = emitRenumber(builder, op->getLoc(), renumber.getMembers(),
                                 renumber.getIds());
    renumber.getResult().replaceAllUsesWith(members);
    freeAtEndOfBlock(op, members);
  } else if (auto cell = dyn_cast<md::OrthorhombicCellOp>(op)) {
    OpBuilder builder(op);
    Type real = builder.getF64Type();
    cell->getResult(0).replaceAllUsesWith(vector::FromElementsOp::create(
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
    if (md_exec::isValueFormType(part))
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
    if (isa<md_exec::ParticleForOp, md_exec::PairForOp, md_exec::TupleForOp>(
            parent))
      continue;
    if (failed(lowerOp(op)))
      return failure();
  }

  // Users come after what they use, so erase from the back.
  for (Operation *op : llvm::reverse(lowered))
    op->erase();
  lowered.clear();
  neighbors.clear();

  // The kernels read tables from their buffers.
  lowerLookups(function);
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
    // Ops of other dialects, such as globals, stay.
    if (!isa<md::MDDialect, md_exec::MDExecDialect, mdrt::MDRTDialect>(
            op.getDialect()) &&
        op.getName().getDialectNamespace() != "dyn")
      continue;
    if (isa<md::ParticleSetOp, md::TupleSetOp, md::DisjointUnionOp>(op)) {
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
