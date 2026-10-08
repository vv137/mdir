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
static const char *const createMatrixName = "mdrtHostMatrixCreate";
static const char *const matrixEntriesName = "mdrtHostMatrixEntries";
static const char *const growMatrixName = "mdrtHostMatrixGrow";
static const char *const countBuildName = "mdrtCountBuild";
/// Counts a build at an interval that found the structure no longer valid
/// (D88).
static const char *const countLateBuildName = "mdrtCountLateBuild";

namespace {

/// The storage of a neighbor matrix.
struct Neighbors {
  /// The number of particles.
  Value size;
  /// The number of neighbors of each particle. The runtime holds the rows
  /// of their indices, by `handle`, and makes them wider when a build finds
  /// them too narrow (getMatrixEntries).
  Value counts;
  Value handle;
  /// The number of neighbors that a row holds at first.
  Value width;
  /// The configuration and the cell that the structure was built at.
  Value reference;
  Value box;
  /// Whether the structure has been built, and how often; the refreshes
  /// since the last build (the policy `interval`).
  Value valid;
  Value builds;
  Value age;
  /// The incidence structure of the pairs that the structure leaves out,
  /// or null.
  Value excluded;
};

class Lowering {
public:
  explicit Lowering(ModuleOp module)
      : module(module), context(module.getContext()) {
    // A module with a triclinic cell lowers every cell to the vector of six,
    // a_x, b_y, c_z, b_x, c_x, c_y; one without, to the three edges.
    module.walk([&](md::TriclinicCellOp) { triclinic = true; });
  }

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
  LogicalResult lowerBuildTriplets(md_exec::BuildTripletsOp op);

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
  /// The rows of the neighbor matrix `handle`, as they are where `builder`
  /// is.
  Value getMatrixEntries(OpBuilder &builder, Location loc, Value handle);

  /// Whether the module has a triclinic cell.
  bool triclinic = false;
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
  if (op.getTraversal() != md_exec::Traversal::Directed)
    return op.emitOpError()
           << "takes each pair once, which a loop on the host does not do";

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

  Value entries = getMatrixEntries(builder, loc, structure.handle);
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  auto loop = scf::ParallelOp::create(
      builder, loc, ValueRange{zero}, ValueRange{size}, ValueRange{one},
      inits, [&](OpBuilder &body, Location, ValueRange ivs, ValueRange) {
        IRMapping local;
        SmallVector<Value> contributions =
            emitPairKernel(body, op, structure.counts, entries, box,
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

/// The type that the loops over the triplets `triplets` compute in: f32 if
/// one of them does, so that the test of the cutoff is that of the
/// narrowest (D160), or else that of the positions.
static Type getTripletsReal(md_exec::BuildTripletsOp triplets) {
  Type real = cast<MemRefType>(triplets.getPositions().getType())
                  .getElementType();
  for (Operation *user : triplets.getResult().getUsers()) {
    auto build = dyn_cast<md_exec::BuildIncidenceOp>(user);
    if (!build)
      continue;
    for (Operation *reader : build.getResult().getUsers()) {
      auto loop = dyn_cast<md_exec::TupleForOp>(reader);
      if (!loop || loop.getKernel().front().getNumArguments() == 0)
        continue;
      Type computed = getElementTypeOrSelf(
          loop.getKernel().front().getArgument(0).getType());
      if (computed.getIntOrFloatBitWidth() < real.getIntOrFloatBitWidth())
        real = computed;
    }
  }
  return real;
}

LogicalResult Lowering::lowerBuildTriplets(md_exec::BuildTripletsOp op) {
  if (!op.isStorageForm())
    return op->emitOpError()
           << "is not in the storage form; run 'md-exec-assign-storage' "
              "first";
  Neighbors structure;
  if (failed(getNeighbors(op, op.getNeighbors(), structure)))
    return failure();
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value positions = op.getPositions();
  // The cell has become the vector of its edge lengths.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);
  Value entries = getMatrixEntries(builder, loc, structure.handle);
  Value members = emitBuildTriplets(
      builder, loc, structure.counts, entries, positions, box, inverse,
      getTripletsReal(op), op.getCutoff().convertToDouble());
  op.getResult().replaceAllUsesWith(members);
  freeAtEndOfBlock(op, members);
  return success();
}

//===----------------------------------------------------------------------===//
// Neighbor structures
//===----------------------------------------------------------------------===//

Value Lowering::getMatrixEntries(OpBuilder &builder, Location loc,
                                Value handle) {
  auto type = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                              builder.getI32Type());
  func::FuncOp getter = getOrDeclare(
      matrixEntriesName,
      builder.getFunctionType({builder.getI64Type()}, {type}));
  getter->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  return func::CallOp::create(builder, loc, getter, ValueRange{handle})
      .getResult(0);
}

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
  // A triclinic cell takes the functions of its own (docs/triclinic-m2.md).
  bool tilted = isTriclinic(op.getCellMutable().get());
  auto instance = [&](StringRef name) {
    std::string full = tilted && name != "mdrt.pme_real"
                           ? (name + "_triclinic").str()
                           : name.str();
    return cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
        module, getPMEInstanceName(full, position, charge, force,
                                   op.getOrder())));
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
  // The sum of the dispersion takes the influence function of its own
  // (D162).
  Value dispersion = arith::ConstantOp::create(
      builder, loc, builder.getBoolAttr(op.getDispersion()));
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
      ValueRange{complex, op.getModuli(), box, beta, coulomb,
                 dispersion, k1, k2, k3});
  func::CallOp::create(builder, loc, backward,
                       ValueRange{complex, real, sizes[0], sizes[1], sizes[2]});
  // The forces, or the potential at the particles in their place.
  func::CallOp::create(builder, loc,
                       instance(op.getPotential() ? "mdrt.pme_potential"
                                                  : "mdrt.pme_gather"),
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
  Type wide = builder.getI64Type();
  structure.handle =
      func::CallOp::create(
          builder, loc,
          getOrDeclare(createMatrixName,
                       builder.getFunctionType({wide, wide}, {wide})),
          ValueRange{
              arith::IndexCastOp::create(builder, loc, wide, structure.size),
              arith::IndexCastOp::create(builder, loc, wide,
                                         structure.width)})
          .getResult(0);
  structure.reference = memref::AllocOp::create(
      builder, loc, MemRefType::get({ShapedType::kDynamic, 3}, real),
      ValueRange{structure.size});
  structure.box = memref::AllocOp::create(
      builder, loc, MemRefType::get({triclinic ? 6 : 3}, builder.getF64Type()));
  structure.valid = memref::AllocOp::create(
      builder, loc, MemRefType::get({}, builder.getI1Type()));
  structure.builds = memref::AllocOp::create(
      builder, loc, MemRefType::get({}, builder.getI64Type()));
  structure.age = memref::AllocOp::create(
      builder, loc, MemRefType::get({}, builder.getI64Type()));

  Value no = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                       builder.getBoolAttr(false));
  Value none = arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                         builder.getI64IntegerAttr(0));
  memref::StoreOp::create(builder, loc, no, structure.valid, ValueRange{});
  memref::StoreOp::create(builder, loc, none, structure.builds, ValueRange{});
  memref::StoreOp::create(builder, loc, none, structure.age, ValueRange{});
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
  // A triclinic cell is searched in its fractional coordinates, whose cells
  // have the widths of the cell between its faces: V / |b × c|,
  // b_y c_z / |(c_y, c_z)|, and c_z.
  bool tilted = isTriclinic(box);
  Value widthsValue =
      tilted ? emitFaceWidths(builder, loc, boxValue) : boxValue;

  // The width of the cells follows from the density, which is known when
  // the structure is built.
  auto choose = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module, getInstanceName(cellWidthName, real)));
  Value widthValue =
      func::CallOp::create(builder, loc, choose,
                           ValueRange{structure.size, widthsValue, reachValue,
                                      leastValue})
          .getResult(0);

  auto build = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module, getInstanceName(tilted ? (std::string(buildNeighborsName) +
                                        "_triclinic")
                                     : std::string(buildNeighborsName),
                              real)));
  // Build; if a row was too narrow to hold the neighbors of a particle,
  // the runtime makes the rows wider, and the build is made again.
  Type wide = builder.getI64Type();
  auto again = scf::WhileOp::create(builder, loc, TypeRange(), ValueRange());
  {
    Block *before = builder.createBlock(&again.getBefore());
    OpBuilder at = OpBuilder::atBlockEnd(before);
    Value entries = getMatrixEntries(at, loc, structure.handle);
    Value largest =
        func::CallOp::create(
            at, loc, build,
            tilted ? ValueRange{positions, boxValue, widthsValue, reachValue,
                                widthValue, structure.counts, entries}
                   : ValueRange{positions, boxValue, reachValue, widthValue,
                                structure.counts, entries})
            .getResult(0);
    Value width =
        memref::DimOp::create(at, loc, entries, createIndex(at, loc, 1));
    Value tooMany = arith::CmpIOp::create(
        at, loc, arith::CmpIPredicate::ugt, largest, width);
    scf::IfOp::create(at, loc, tooMany, [&](OpBuilder &then, Location) {
      func::CallOp::create(
          then, loc,
          getOrDeclare(growMatrixName,
                       then.getFunctionType({wide, wide}, {})),
          ValueRange{structure.handle,
                     arith::IndexCastOp::create(then, loc, wide, largest)});
      scf::YieldOp::create(then, loc);
    });
    scf::ConditionOp::create(at, loc, tooMany, ValueRange());
    Block *after = builder.createBlock(&again.getAfter());
    OpBuilder close = OpBuilder::atBlockEnd(after);
    scf::YieldOp::create(close, loc);
  }
  builder.setInsertionPointAfter(again);

  // The runtime counts the builds, for the log of the run.
  func::CallOp::create(
      builder, loc,
      getOrDeclare(countBuildName, builder.getFunctionType({}, {})),
      ValueRange());

  // Leave the excluded pairs out.
  Value filtered = getMatrixEntries(builder, loc, structure.handle);
  if (structure.excluded)
    scf::ParallelOp::create(
        builder, loc, ValueRange{createIndex(builder, loc, 0)},
        ValueRange{structure.size}, ValueRange{createIndex(builder, loc, 1)},
        [&](OpBuilder &body, Location, ValueRange ivs) {
          emitExclusionFilter(body, loc, structure.counts, filtered,
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
  for (int64_t c = 0, e = cast<VectorType>(box.getType()).getNumElements();
       c < e; ++c) {
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
  Value fresh =
      arith::ConstantOp::create(builder, loc, wide, builder.getI64IntegerAttr(0));
  memref::StoreOp::create(builder, loc, fresh, structure.age, ValueRange{});
  return success();
}

LogicalResult
Lowering::lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op) {
  if (op.getPruneSkin())
    return op.emitOpError() << "keeps a dual list (D114), which only a "
                               "device lowers";
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

  // Whether no particle has moved more than half the skin since the
  // structure was built, emitted where `at` is. A barostat scales the
  // positions with the cell: the reference is compared scaled as the cell
  // was, m = L / L_ref, against half of min(m) R − r_c (D80).
  auto emitNear = [&](OpBuilder &at) -> Value {
    SmallVector<Value, 3> builtEdges;
    for (int64_t c = 0; c < 3; ++c)
      builtEdges.push_back(memref::LoadOp::create(
          at, loc, structure.box, ValueRange{createIndex(at, loc, c)}));
    Value scale = arith::DivFOp::create(
        at, loc, getEdges(at, loc, box),
        vector::FromElementsOp::create(
            at, loc, VectorType::get({3}, at.getF64Type()), builtEdges));
    Value least = vector::ReductionOp::create(
        at, loc, vector::CombiningKind::MINNUMF, scale);
    Value margin = arith::SubFOp::create(
        at, loc,
        arith::MulFOp::create(at, loc, least,
                              createReal(at, loc, at.getF64Type(), reach)),
        createReal(at, loc, at.getF64Type(),
                   op.getCutoff().convertToDouble()));
    Value halfMargin = arith::MulFOp::create(
        at, loc,
        arith::MaximumFOp::create(at, loc, margin,
                                  createReal(at, loc, at.getF64Type(), 0.0)),
        createReal(at, loc, at.getF64Type(), 0.5));
    Value limit2 = arith::MulFOp::create(at, loc, halfMargin, halfMargin);
    Value scaleReal = scale;
    if (!real.isF64())
      scaleReal = arith::TruncFOp::create(
          at, loc, VectorType::get({3}, real), scale);
    Value limitReal = limit2;
    if (!real.isF64())
      limitReal = arith::TruncFOp::create(at, loc, real, limit2);

    Value yes = arith::ConstantOp::create(at, loc, at.getI1Type(),
                                          at.getBoolAttr(true));
    // A loop has made the test.
    if (Value moved = op.getMoved())
      return arith::XOrIOp::create(at, loc, moved, yes);
    Value zero = createIndex(at, loc, 0);
    Value one = createIndex(at, loc, 1);
    Value none = createZero(at, loc, real);
    auto farthest = scf::ParallelOp::create(
        at, loc, ValueRange{zero}, ValueRange{structure.size},
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
    return arith::CmpFOp::create(at, loc, arith::CmpFPredicate::OLE,
                                 farthest.getResult(0), limitReal);
  };
  (void)skin;

  Value yes = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(true));
  Value valid =
      memref::LoadOp::create(builder, loc, structure.valid, ValueRange{});
  LogicalResult status = success();

  if (op.getPolicy() == md_exec::RebuildPolicy::Interval) {
    // NOT A DEFAULT (D88): the structure is built every `interval`
    // refreshes whether it is valid or not, and may leave out pairs within
    // the cutoff in between. It is tested at a build only, and a build that
    // finds it no longer valid is counted for the log.
    Type wide = builder.getI64Type();
    Value age = arith::AddIOp::create(
        builder, loc,
        memref::LoadOp::create(builder, loc, structure.age, ValueRange{}),
        arith::ConstantOp::create(builder, loc, wide,
                                  builder.getI64IntegerAttr(1)));
    memref::StoreOp::create(builder, loc, age, structure.age, ValueRange{});
    Value old = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sge, age,
        arith::ConstantOp::create(builder, loc, wide,
                                  builder.getI64IntegerAttr(*op.getInterval())));
    Value due = arith::OrIOp::create(
        builder, loc, arith::XOrIOp::create(builder, loc, valid, yes), old);
    // The branches are filled once they are in the function: the test
    // launches kernels, which look up where they are.
    auto emitIf = [&](OpBuilder &at, Value condition,
                      llvm::function_ref<void(OpBuilder &)> fill) {
      auto branch = scf::IfOp::create(at, loc, condition,
                                      /*withElseRegion=*/false);
      OpBuilder inner(branch.thenBlock()->getTerminator());
      fill(inner);
    };
    emitIf(builder, due, [&](OpBuilder &then) {
      emitIf(then, valid, [&](OpBuilder &test) {
        Value late = arith::XOrIOp::create(
            test, loc, emitNear(test),
            arith::ConstantOp::create(test, loc, test.getI1Type(),
                                      test.getBoolAttr(true)));
        emitIf(test, late, [&](OpBuilder &count) {
          func::CallOp::create(
              count, loc,
              getOrDeclare(countLateBuildName,
                           count.getFunctionType({}, {})),
              ValueRange());
        });
      });
      status = emitBuild(then, loc, structure, positions, box, reach,
                         cellWidth);
    });
    return status;
  }

  // The structure is valid if it has been built, in this cell, and no
  // particle has moved more than half the skin since.
  valid = arith::AndIOp::create(builder, loc, valid, emitNear(builder));
  Value stale = arith::XOrIOp::create(builder, loc, valid, yes);
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

  // The cell has become the vector of its edge lengths; a triclinic one
  // orders the particles by the cells of its diagonal, which only their
  // locality depends on.
  Value box = convertReal(
      builder, loc, getEdges(builder, loc, op.getCellMutable().get()), real);
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
    if (empty.getKind() != md_exec::NeighborKind::Matrix)
      return op->emitOpError()
             << "is a structure of groups, which is built on a device only";
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
    SmallVector<Value, 6> edges;
    int64_t count = cast<MemRefType>(structure.box.getType()).getDimSize(0);
    for (int64_t c = 0; c < count; ++c)
      edges.push_back(memref::LoadOp::create(
          builder, loc, structure.box,
          ValueRange{createIndex(builder, loc, c)}));
    cell->getResult(0).replaceAllUsesWith(vector::FromElementsOp::create(
        builder, loc, VectorType::get({count}, builder.getF64Type()), edges));
  } else if (auto edges = dyn_cast<md_exec::CellEdgesOp>(op)) {
    // The cell is the vector of its edges by now, or of its diagonal and
    // tilts.
    OpBuilder builder(op);
    edges.getResult().replaceAllUsesWith(
        getEdges(builder, op->getLoc(), edges->getOperand(0)));
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
  } else if (auto triplets = dyn_cast<md_exec::BuildTripletsOp>(op)) {
    if (failed(lowerBuildTriplets(triplets)))
      return failure();
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
    SmallVector<Value, 6> values = {cell.getLx(), cell.getLy(), cell.getLz()};
    // In a module with a triclinic cell, an orthorhombic one has no tilts.
    if (triclinic)
      values.append(3, createReal(builder, op->getLoc(), real, 0.0));
    cell->getResult(0).replaceAllUsesWith(vector::FromElementsOp::create(
        builder, op->getLoc(),
        VectorType::get({static_cast<int64_t>(values.size())}, real),
        values));
  } else if (auto cell = dyn_cast<md::TriclinicCellOp>(op)) {
    OpBuilder builder(op);
    Type real = builder.getF64Type();
    cell->getResult(0).replaceAllUsesWith(vector::FromElementsOp::create(
        builder, op->getLoc(), VectorType::get({6}, real),
        ValueRange{cell.getLx(), cell.getLy(), cell.getLz(), cell.getBx(),
                   cell.getCx(), cell.getCy()}));
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
  Type box = VectorType::get({triclinic ? 6 : 3}, Float64Type::get(context));
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
    if (isa<md::OrthorhombicCellOp, md::TriclinicCellOp>(op))
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
