// Conversion of md_exec loops in the storage form to GPU kernels.
//
// Every op of md_exec becomes ops of the upstream gpu dialect, where the op
// is. See docs/ops-m0.md, Section 10.8.

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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

namespace mdir {
/// The text of the template that builds a neighbor matrix on a device.
extern const char *const neighborsMatrixGPUTemplate;
} // namespace mdir

static const char *const buildNeighborsName =
    "mdrt_gpu_build_neighbors_matrix";
static const char *const reportOverflowName = "mdrtReportNeighborOverflow";
static const char *const countBuildName = "mdrtCountBuild";

/// The number of particles whose contributions one thread adds up.
static const int64_t chunkSize = 256;

namespace {

/// The storage of a neighbor matrix. The buffers are on the device, the
/// state of the structure is on the host.
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

/// Where a global sum or maximum arrives: one value on the device, and the
/// buffer of the host that it is copied to.
struct Cell {
  Value device;
  Value host;
};

class Lowering {
public:
  Lowering(ModuleOp module, int64_t blockSize)
      : module(module), context(module.getContext()), blockSize(blockSize) {}

  LogicalResult run();

private:
  LogicalResult lowerFunction(func::FuncOp function);
  void releaseStack(func::FuncOp function);
  LogicalResult lowerOp(Operation *op);

  void lowerEmptyNeighbors(md_exec::EmptyNeighborsOp op);
  LogicalResult lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op);
  LogicalResult lowerParticleFor(md_exec::ParticleForOp op);
  LogicalResult lowerPairFor(md_exec::PairForOp op);

  /// The storage of the neighbor structure `structure`.
  LogicalResult getNeighbors(Operation *op, Value structure,
                             Neighbors &storage);

  /// Builds `structure` at the configuration `positions`.
  LogicalResult emitBuild(OpBuilder &builder, Location loc,
                          const Neighbors &structure, Value positions,
                          Value box, double reach, double cellWidth);

  /// Launches a kernel with one thread for each of `count` items. `body`
  /// emits what the thread of an item does.
  void launchOver(OpBuilder &builder, Location loc, Value count,
                  function_ref<void(OpBuilder &, Value)> body);

  /// Launches a kernel with one thread.
  void launchOne(OpBuilder &builder, Location loc,
                 function_ref<void(OpBuilder &)> body);

  /// Gives the kernel of `launch` the values from outside as it can take
  /// them. A kernel takes numbers and buffers as arguments: a vector enters
  /// as its elements. A constant becomes a constant of the kernel.
  void bringIn(gpu::LaunchOp launch);

  /// Adds up what `contributions` holds for `size` particles, or takes the
  /// maximum, and returns the result on the host. `partial` takes the
  /// results of the chunks.
  Value emitReduction(OpBuilder &builder, Location loc, Value contributions,
                      Value partial, Value size, bool isSum);

  /// Stores the contributions of a particle and returns, for each global
  /// sum of a loop, its result.
  LogicalResult finishSums(Operation *op, OpBuilder &builder,
                           ValueRange reduce, ValueRange scratch, Value size);

  Cell getCell(Type type, Location loc);

  /// Copies what `source` holds to `destination`.
  void createTransfer(OpBuilder &builder, Location loc, Value destination,
                      Value source);

  Value createSize(OpBuilder &builder, Location loc, Value buffer) {
    return memref::DimOp::create(builder, loc, buffer,
                                 createIndex(builder, loc, 0));
  }

  /// The number of groups of `group` items that hold `count` items.
  Value createGroups(OpBuilder &builder, Location loc, Value count,
                     int64_t group) {
    Value padded = arith::AddIOp::create(builder, loc, count,
                                         createIndex(builder, loc, group - 1));
    return arith::DivUIOp::create(builder, loc, padded,
                                  createIndex(builder, loc, group));
  }

  /// Adds the templates for positions of the type `real` to the module.
  LogicalResult addTemplates(Type real);
  func::FuncOp getOrDeclare(StringRef name, FunctionType type);

  ModuleOp module;
  MLIRContext *context;
  int64_t blockSize;

  /// The function that is being lowered.
  func::FuncOp current;

  llvm::DenseMap<Value, Neighbors> neighbors;
  llvm::DenseMap<Type, Cell> cells;
  llvm::DenseSet<Type> templatesAdded;

  /// The ops that have been lowered, in the order of the program.
  SmallVector<Operation *> lowered;
};

} // namespace

/// The type of a buffer on the device.
static MemRefType getDeviceType(ArrayRef<int64_t> shape, Type element) {
  return MemRefType::get(
      shape, element, MemRefLayoutAttrInterface(),
      IntegerAttr::get(IntegerType::get(element.getContext(), 64), 1));
}

static bool isDeviceType(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  return buffer && buffer.getMemorySpace();
}

static Value createDeviceBuffer(OpBuilder &builder, Location loc,
                                MemRefType type, ValueRange sizes) {
  return gpu::AllocOp::create(builder, loc, type, /*asyncToken=*/Type(),
                              /*asyncDependencies=*/ValueRange(), sizes,
                              /*symbolOperands=*/ValueRange())
      .getMemref();
}

//===----------------------------------------------------------------------===//
// Kernels
//===----------------------------------------------------------------------===//

void Lowering::createTransfer(OpBuilder &builder, Location loc,
                              Value destination, Value source) {
  Type token = gpu::AsyncTokenType::get(context);
  Value begin =
      gpu::WaitOp::create(builder, loc, token, ValueRange()).getAsyncToken();
  Value copied = gpu::MemcpyOp::create(builder, loc, token,
                                       ValueRange{begin}, destination, source)
                     .getAsyncToken();
  gpu::WaitOp::create(builder, loc, Type(), ValueRange{copied});
}

void Lowering::bringIn(gpu::LaunchOp launch) {
  Region &region = launch.getBody();
  OpBuilder outside(launch);
  OpBuilder inside(&region.front(), region.front().begin());

  llvm::DenseMap<Value, Value> brought;
  region.walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      if (region.isAncestor(value.getParentRegion()))
        continue;

      // A constant is a constant of the kernel. As an argument it would be
      // a value that the kernel does not know: a power with that exponent
      // would be a loop.
      Operation *definition = value.getDefiningOp();
      bool isConstant = definition && matchPattern(definition, m_Constant());
      auto type = dyn_cast<VectorType>(value.getType());
      if (!isConstant && !type)
        continue;

      Value &replacement = brought[value];
      if (!replacement) {
        Location loc = value.getLoc();
        if (isConstant) {
          replacement = inside.clone(*definition)->getResult(0);
        } else {
          SmallVector<Value, 4> elements;
          for (int64_t i = 0, e = type.getNumElements(); i != e; ++i)
            elements.push_back(
                vector::ExtractOp::create(outside, loc, value, i));
          replacement =
              vector::FromElementsOp::create(inside, loc, type, elements);
        }
      }
      operand.set(replacement);
    }
  });
}

void Lowering::launchOver(OpBuilder &builder, Location loc, Value count,
                          function_ref<void(OpBuilder &, Value)> body) {
  Value one = createIndex(builder, loc, 1);
  Value block = createIndex(builder, loc, blockSize);
  Value grid = createGroups(builder, loc, count, blockSize);
  auto launch =
      gpu::LaunchOp::create(builder, loc, grid, one, one, block, one, one);

  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value threads = createIndex(kernel, loc, blockSize);
  Value base =
      arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, threads);
  Value item =
      arith::AddIOp::create(kernel, loc, base, launch.getThreadIds().x);
  // The threads beyond the last item do nothing.
  Value inside = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                       item, count);
  scf::IfOp::create(kernel, loc, inside, [&](OpBuilder &then, Location) {
    body(then, item);
    scf::YieldOp::create(then, loc);
  });
  gpu::TerminatorOp::create(kernel, loc);
  bringIn(launch);
}

void Lowering::launchOne(OpBuilder &builder, Location loc,
                         function_ref<void(OpBuilder &)> body) {
  Value one = createIndex(builder, loc, 1);
  auto launch =
      gpu::LaunchOp::create(builder, loc, one, one, one, one, one, one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  body(kernel);
  gpu::TerminatorOp::create(kernel, loc);
  bringIn(launch);
}

Cell Lowering::getCell(Type type, Location loc) {
  Cell &cell = cells[type];
  if (cell.device)
    return cell;

  // One value, allocated once, where the function begins.
  Block &entry = current.getBody().front();
  OpBuilder builder(&entry, entry.begin());
  cell.device = createDeviceBuffer(builder, loc, getDeviceType({1}, type),
                                   ValueRange());
  cell.host = memref::AllocaOp::create(builder, loc,
                                       MemRefType::get({1}, type));
  return cell;
}

Value Lowering::emitReduction(OpBuilder &builder, Location loc,
                              Value contributions, Value partial, Value size,
                              bool isSum) {
  Type type = cast<MemRefType>(contributions.getType()).getElementType();
  auto combine = [&](OpBuilder &b, Value lhs, Value rhs) -> Value {
    if (isSum)
      return arith::AddFOp::create(b, loc, lhs, rhs);
    Value larger =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, lhs, rhs);
    return arith::SelectOp::create(b, loc, larger, lhs, rhs);
  };

  // One thread for each chunk of particles.
  Value chunks = createGroups(builder, loc, size, chunkSize);
  launchOver(builder, loc, chunks, [&](OpBuilder &b, Value chunk) {
    Value one = createIndex(b, loc, 1);
    Value length = createIndex(b, loc, chunkSize);
    Value begin = arith::MulIOp::create(b, loc, chunk, length);
    Value full = arith::AddIOp::create(b, loc, begin, length);
    Value isShort =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, size, full);
    Value end = arith::SelectOp::create(b, loc, isShort, size, full);
    auto loop = scf::ForOp::create(
        b, loc, begin, end, one, ValueRange{createZero(b, loc, type)},
        [&](OpBuilder &inner, Location, Value i, ValueRange sums) {
          Value value = memref::LoadOp::create(inner, loc, contributions,
                                               ValueRange{i});
          scf::YieldOp::create(inner, loc, combine(inner, sums[0], value));
        });
    memref::StoreOp::create(b, loc, loop.getResult(0), partial,
                            ValueRange{chunk});
  });

  // One thread for the results of the chunks.
  Cell cell = getCell(type, loc);
  launchOne(builder, loc, [&](OpBuilder &b) {
    Value zero = createIndex(b, loc, 0);
    Value one = createIndex(b, loc, 1);
    auto loop = scf::ForOp::create(
        b, loc, zero, chunks, one, ValueRange{createZero(b, loc, type)},
        [&](OpBuilder &inner, Location, Value i, ValueRange sums) {
          Value value =
              memref::LoadOp::create(inner, loc, partial, ValueRange{i});
          scf::YieldOp::create(inner, loc, combine(inner, sums[0], value));
        });
    memref::StoreOp::create(b, loc, loop.getResult(0), cell.device,
                            ValueRange{zero});
  });

  createTransfer(builder, loc, cell.host, cell.device);
  return memref::LoadOp::create(builder, loc, cell.host,
                                ValueRange{createIndex(builder, loc, 0)});
}

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//

/// Verifies that a loop with global sums has the buffers that it needs.
static LogicalResult checkScratch(Operation *op, unsigned sums,
                                  unsigned scratch) {
  if (scratch == 2 * sums)
    return success();
  return op->emitOpError()
         << "needs 2 buffers in 'scratch' for each global sum; run "
            "'md-exec-assign-storage' with 'memory=device'";
}

LogicalResult Lowering::finishSums(Operation *op, OpBuilder &builder,
                                   ValueRange reduce, ValueRange scratch,
                                   Value size) {
  Location loc = op->getLoc();
  for (unsigned i = 0, e = reduce.size(); i != e; ++i) {
    Value sum = emitReduction(builder, loc, scratch[2 * i],
                              scratch[2 * i + 1], size, /*isSum=*/true);
    Value total = arith::AddFOp::create(builder, loc, reduce[i], sum);
    op->getResult(i).replaceAllUsesWith(total);
  }
  return success();
}

LogicalResult Lowering::lowerParticleFor(md_exec::ParticleForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  if (failed(checkScratch(op, op.getReduce().size(), op.getScratch().size())))
    return failure();

  Value size = createSize(builder, loc, op.getIns().empty()
                                            ? op.getOuts().front()
                                            : op.getIns().front());
  launchOver(builder, loc, size, [&](OpBuilder &body, Value particle) {
    IRMapping local;
    SmallVector<Value> contributions =
        emitParticleKernel(body, op, particle, local);
    for (auto [index, value] : llvm::enumerate(contributions))
      memref::StoreOp::create(body, loc, value, op.getScratch()[2 * index],
                              ValueRange{particle});
  });
  return finishSums(op, builder, op.getReduce(), op.getScratch(), size);
}

LogicalResult Lowering::lowerPairFor(md_exec::PairForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  if (failed(checkScratch(op, op.getReduce().size(), op.getScratch().size())))
    return failure();

  Neighbors structure;
  if (failed(getNeighbors(op, op.getNeighbors(), structure)))
    return failure();

  Value positions = op.getPositions();
  Value size = createSize(builder, loc, positions);

  // The cell has become the vector of its edge lengths.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);

  launchOver(builder, loc, size, [&](OpBuilder &body, Value central) {
    IRMapping local;
    SmallVector<Value> contributions =
        emitPairKernel(body, op, structure.counts, structure.index, box,
                       inverse, central, local);
    for (auto [index, value] : llvm::enumerate(contributions))
      memref::StoreOp::create(body, loc, value, op.getScratch()[2 * index],
                              ValueRange{central});
  });
  return finishSums(op, builder, op.getReduce(), op.getScratch(), size);
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
      instantiateTemplates(neighborsMatrixGPUTemplate, real), config);
  if (!templates)
    return module.emitError()
           << "cannot parse the neighbor build template for devices";
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
  auto positions = cast<MemRefType>(*op.getPositions());

  Neighbors structure;
  structure.size = op.getSize();
  structure.width = createIndex(builder, loc, op.getWidth());
  structure.counts = createDeviceBuffer(
      builder, loc, getDeviceType({ShapedType::kDynamic}, narrow),
      ValueRange{structure.size});
  structure.index = createDeviceBuffer(
      builder, loc,
      getDeviceType({ShapedType::kDynamic, ShapedType::kDynamic}, narrow),
      ValueRange{structure.size, structure.width});
  structure.reference =
      createDeviceBuffer(builder, loc, positions, ValueRange{structure.size});

  // The state of the structure is on the host.
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

  // Remember the configuration that the structure was built at.
  createTransfer(builder, loc, structure.reference, positions);
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

  if (op.getScratch().size() != 2)
    return op.emitOpError()
           << "needs 2 buffers in 'scratch' for the test of validity; run "
              "'md-exec-assign-storage' with 'memory=device'";

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

  // Before the first build the configuration that the test compares with
  // holds nothing. The result of the test then does not count.
  Value moved2 = op.getScratch()[0];
  launchOver(builder, loc, structure.size,
             [&](OpBuilder &body, Value particle) {
               Value now = loadElement(body, loc, positions, particle);
               Value then =
                   loadElement(body, loc, structure.reference, particle);
               Value moved = arith::SubFOp::create(body, loc, now, then);
               Value squares = arith::MulFOp::create(body, loc, moved, moved);
               Value distance2 = vector::ReductionOp::create(
                   body, loc, vector::CombiningKind::ADD, squares);
               memref::StoreOp::create(body, loc, distance2, moved2,
                                       ValueRange{particle});
             });
  Value farthest = emitReduction(builder, loc, moved2, op.getScratch()[1],
                                 structure.size, /*isSum=*/false);
  Value limit = createReal(builder, loc, real, 0.25 * skin * skin);
  Value near = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE,
                                     farthest, limit);
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

  if (isa<md_exec::MDExecDialect>(op->getDialect())) {
    bool onHost = llvm::any_of(op->getOperandTypes(), [](Type type) {
      return isa<MemRefType>(type) && !isDeviceType(type);
    });
    if (auto empty = dyn_cast<md_exec::EmptyNeighborsOp>(op))
      if (auto positions = empty.getPositions())
        onHost |= !isDeviceType(*positions);
    if (onHost)
      return op->emitOpError()
             << "has its buffers on the host; use "
                "'convert-md-exec-to-loops', or run "
                "'md-exec-assign-storage' with 'memory=device'";
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
    if (failed(lowerParticleFor(loop)))
      return failure();
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

  current = function;
  cells.clear();

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

  releaseStack(function);
  return success();
}

/// Makes every loop of the host in `function` that launches kernels or calls
/// functions release, at the end of an iteration, the stack that the
/// iteration has taken. The arguments of a launch and of a call are put on
/// the stack where the launch or the call is. In a loop, the stack would
/// grow with every iteration until the function returns.
void Lowering::releaseStack(func::FuncOp function) {
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) {
    if (loop->getParentOfType<gpu::LaunchOp>())
      return;
    bool takesStack = false;
    loop.getBody()->walk([&](Operation *op) {
      takesStack |=
          isa<gpu::LaunchOp, func::CallOp, mdrt::HostCallOp>(op);
    });
    if (takesStack)
      loops.push_back(loop);
  });

  Type pointer = LLVM::LLVMPointerType::get(context);
  for (scf::ForOp loop : loops) {
    Block &body = *loop.getBody();
    OpBuilder builder(&body, body.begin());
    Value stack = LLVM::StackSaveOp::create(builder, loop.getLoc(), pointer);
    builder.setInsertionPoint(body.getTerminator());
    LLVM::StackRestoreOp::create(builder, loop.getLoc(), stack);
  }
}

LogicalResult Lowering::run() {
  if (blockSize <= 0)
    return module.emitError() << "expected a positive size of a block";

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

#define GEN_PASS_DEF_CONVERTMDEXECTOGPU
#include "mdir/Conversion/Passes.h.inc"

namespace {
class ConvertMDExecToGPU
    : public impl::ConvertMDExecToGPUBase<ConvertMDExecToGPU> {
public:
  using impl::ConvertMDExecToGPUBase<
      ConvertMDExecToGPU>::ConvertMDExecToGPUBase;

  void runOnOperation() final {
    Lowering lowering(getOperation(), blockSize);
    if (failed(lowering.run()))
      signalPassFailure();
  }
};
} // namespace

} // namespace mdir
