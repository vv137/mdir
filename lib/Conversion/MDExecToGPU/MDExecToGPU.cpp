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
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

namespace mdir {
/// The text of the template that builds a neighbor matrix on a device.
extern const char *const neighborsMatrixGPUTemplate;
/// The text of the template of particle mesh Ewald on a device.
extern const char *const pmeGPUTemplate;
} // namespace mdir

static const char *const spatialOrderName = "mdrt_gpu_spatial_order";
static const char *const cellWidthName = "mdrt_gpu_cell_width";
static const char *const buildNeighborsName =
    "mdrt_gpu_build_neighbors_matrix";
static const char *const reportOverflowName = "mdrtReportNeighborOverflow";
static const char *const countBuildName = "mdrtCountBuild";

/// The threads that share the row of one particle in a loop over pairs or
/// tuples. A particle has hundreds of neighbors, and a thread for each
/// particle leaves most of a device idle for a system of thousands; the
/// tuples of a few particles, such as the dihedrals of a protein, would
/// make those threads the longest.

/// The slots of the buffers of the results of global sums, for each type.
static const int64_t resultCapacity = 1024;

namespace {

/// The storage of a neighbor matrix. The buffers are on the device, the
/// state of the structure is on the host.
struct Neighbors {
  /// The number of particles.
  Value size;
  /// The number of neighbors of each particle, and their indices.
  Value counts;
  Value index;
  /// The number of neighbors that a row holds, and the entries of the
  /// matrix, `size` times `width`.
  Value width;
  Value entries;
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

/// Where the global sums and maxima of a loop arrive: numbers on the
/// device, and the buffer of the host that they are copied to.
struct Cell {
  Value device;
  Value host;
  int64_t capacity = 0;
};

/// A flag that the threads of a kernel set: one value on the device, the
/// buffer of the host that it is copied to, and a buffer of the host that
/// holds the value of a flag that is not set.
struct Flag {
  Value device;
  Value host;
  Value clear;
};

class Lowering {
public:
  Lowering(ModuleOp module, int64_t blockSize, int64_t splitLimit,
           int64_t rowLanes, bool fuseRows, bool pmeStream)
      : module(module), context(module.getContext()), blockSize(blockSize),
        splitLimit(splitLimit), rowLanes(rowLanes), fuseRows(fuseRows),
        pmeStream(pmeStream) {}

  LogicalResult run();

private:
  LogicalResult lowerFunction(func::FuncOp function);
  void releaseStack(func::FuncOp function);
  LogicalResult lowerOp(Operation *op);

  void lowerEmptyNeighbors(md_exec::EmptyNeighborsOp op);
  LogicalResult lowerSpatialOrder(md_exec::SpatialOrderOp op);
  void lowerPermute(md_exec::PermuteOp op);
  LogicalResult lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op);
  LogicalResult lowerParticleFor(md_exec::ParticleForOp op);
  LogicalResult lowerPairFor(md_exec::PairForOp op);
  LogicalResult lowerTupleFor(md_exec::TupleForOp op);
  void lowerBuildIncidence(md_exec::BuildIncidenceOp op);
  void lowerRenumber(md_exec::RenumberOp op);
  /// Frees `buffer`, a buffer of the device, where the block of `op` ends.
  void freeDeviceAtEndOfBlock(Operation *op, Value buffer);

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

  /// Launches a kernel with a group of threads for each of `count`
  /// particles, which share its row (RowLanes).
  void launchRows(
      OpBuilder &builder, Location loc, Value count,
      function_ref<void(OpBuilder &, Value, const RowLanes &)> body);


  /// Gives the kernel of `launch` the values from outside as it can take
  /// them. A kernel takes numbers and buffers as arguments: a vector enters
  /// as its elements. A constant becomes a constant of the kernel.
  void bringIn(gpu::LaunchOp launch);

  /// Adds up what each of `contributions` holds for `size` particles, or
  /// takes the maximum, and returns the results on the host. `partials`
  /// take the results of the chunks.
  ///
  /// All of them are reduced by one pair of kernels, and the results
  /// reach the host in one copy for each type of number.
  /// The runs of loops that are lowered together, by their last loop, and
  /// the loops of the runs.
  DenseMap<Operation *, SmallVector<Operation *>> rows;
  DenseSet<Operation *> inRows;

  SmallVector<Value> emitReductions(OpBuilder &builder, Location loc,
                                    ArrayRef<Value> contributions,
                                    ArrayRef<Value> partials, Value size,
                                    bool isSum);

  /// Finds the runs of loops over pairs and tuples in `function` that one
  /// kernel can do (lowerRows), and records them in `rows`.
  void findRows(func::FuncOp function);
  /// Lowers the loops over pairs and tuples of `run` to one kernel, in
  /// which the group of threads of a particle does each loop in turn, and
  /// their global sums to one reduction.
  LogicalResult lowerRows(ArrayRef<Operation *> run);

  /// Stores the contributions of a particle and returns, for each global
  /// sum of a loop, its result.
  LogicalResult finishSums(Operation *op, OpBuilder &builder,
                           ValueRange reduce, ValueRange scratch, Value size);

  /// A place for `count` numbers of the type `element`.
  Cell getCell(Type element, int64_t count, Location loc);

  /// Flag number `number` of the function. It is not set where a kernel
  /// begins that sets it: `readFlag` sees to that.
  Flag getFlag(unsigned number, Location loc);

  /// Returns whether `flag` is set, on the host, and leaves it not set.
  Value readFlag(OpBuilder &builder, Location loc, const Flag &flag);

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
  LogicalResult addPMETemplates(Type position, Type charge, Type force);
  LogicalResult lowerReciprocal(md_exec::ReciprocalOp op);

  ModuleOp module;
  MLIRContext *context;
  int64_t blockSize;
  int64_t splitLimit;
  /// The threads that share the rows of a particle (launchRows).
  int64_t rowLanes;
  /// Whether runs of loops over rows become one kernel (findRows).
  bool fuseRows;
  /// Whether the reciprocal sum runs on a second stream (lowerReciprocal).
  bool pmeStream;

  /// The function that is being lowered.
  func::FuncOp current;
  /// What the kernels that are being made compute, for their names: the op
  /// and the line of the input that it comes from (nameKernel).
  std::string purpose = "kernel";
  int64_t numKernels = 0;
  void setPurpose(Operation *op);
  void nameKernel(gpu::LaunchOp launch);

  llvm::DenseMap<Value, Neighbors> neighbors;
  llvm::DenseMap<Type, SmallVector<Cell, 2>> cells;
  /// Where global sums arrive to be read later (deferReadbacks): a slot of
  /// its own for each sum of the function, in one buffer for each type.
  llvm::DenseMap<Type, Cell> resultCells;
  llvm::DenseMap<Type, int64_t> nextResult;
  int64_t numReadbacks = 0;
  Cell getResults(Type element, Location loc);
  SmallVector<Flag, 2> flags;
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

/// The purpose of the kernels of `op`: its name, and the line of the input
/// where it begins, such as `tuple_for_l2449`: the line of the module that
/// `mdir emit` prints for a run.
void Lowering::setPurpose(Operation *op) {
  std::string name = op->getName().stripDialect().str();
  if (auto location = dyn_cast<FileLineColLoc>(op->getLoc()))
    name += "_l" + std::to_string(location.getLine());
  purpose = name;
}

/// Names the kernel that `launch` becomes, and its module: the function,
/// the purpose, and a number that makes the name unique. A profiler, a
/// sanitizer, and the trace of the runtime report kernels by these names.
void Lowering::nameKernel(gpu::LaunchOp launch) {
  if (launch.getFunctionAttr())
    return;
  std::string name = (current ? current.getName().str() + "_" : "") +
                     purpose + "_" + std::to_string(numKernels++);
  launch.setFunctionAttr(FlatSymbolRefAttr::get(context, name));
  launch.setModuleAttr(FlatSymbolRefAttr::get(context, name));
}

void Lowering::bringIn(gpu::LaunchOp launch) {
  nameKernel(launch);
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

Cell Lowering::getResults(Type element, Location loc) {
  auto found = resultCells.find(element);
  if (found != resultCells.end())
    return found->second;
  Block &entry = current.getBody().front();
  OpBuilder builder(&entry, entry.begin());
  Cell cell;
  cell.capacity = resultCapacity;
  cell.device = createDeviceBuffer(
      builder, loc, getDeviceType({resultCapacity}, element), ValueRange());
  cell.host = memref::AllocaOp::create(
      builder, loc, MemRefType::get({resultCapacity}, element));
  // A copy brings the whole buffer, the slots that no sum has written yet
  // too: they hold zeros.
  Value device = cell.device;
  launchOver(builder, loc, createIndex(builder, loc, resultCapacity),
             [&](OpBuilder &body, Value slot) {
               memref::StoreOp::create(body, loc, createZero(body, loc, element),
                                       device, ValueRange{slot});
             });
  resultCells[element] = cell;
  return cell;
}

Cell Lowering::getCell(Type element, int64_t count, Location loc) {
  SmallVector<Cell, 2> &known = cells[element];
  for (const Cell &cell : known)
    if (cell.capacity >= count)
      return cell;

  // Allocated once, where the function begins.
  Block &entry = current.getBody().front();
  OpBuilder builder(&entry, entry.begin());
  Cell cell;
  cell.capacity = count;
  cell.device = createDeviceBuffer(builder, loc,
                                   getDeviceType({count}, element),
                                   ValueRange());
  cell.host = memref::AllocaOp::create(builder, loc,
                                       MemRefType::get({count}, element));
  known.push_back(cell);
  return cell;
}

Flag Lowering::getFlag(unsigned number, Location loc) {
  Block &entry = current.getBody().front();
  OpBuilder builder(&entry, entry.begin());
  Type narrow = builder.getI32Type();
  while (flags.size() <= number) {
    // Allocated once, where the function begins, and not set.
    Flag flag;
    flag.device = createDeviceBuffer(builder, loc,
                                     getDeviceType({1}, narrow), ValueRange());
    flag.host =
        memref::AllocaOp::create(builder, loc, MemRefType::get({1}, narrow));
    flag.clear =
        memref::AllocaOp::create(builder, loc, MemRefType::get({1}, narrow));
    Value zero = arith::ConstantOp::create(builder, loc, narrow,
                                           builder.getI32IntegerAttr(0));
    memref::StoreOp::create(builder, loc, zero, flag.clear,
                            ValueRange{createIndex(builder, loc, 0)});
    createTransfer(builder, loc, flag.device, flag.clear);
    flags.push_back(flag);
  }
  return flags[number];
}

Value Lowering::readFlag(OpBuilder &builder, Location loc,
                         const Flag &flag) {
  createTransfer(builder, loc, flag.host, flag.device);
  Value value = memref::LoadOp::create(
      builder, loc, flag.host, ValueRange{createIndex(builder, loc, 0)});
  Value zero = arith::ConstantOp::create(builder, loc, value.getType(),
                                         builder.getI32IntegerAttr(0));
  Value isSet = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                      value, zero);
  // Few kernels set the flag, so it is cleared only where it was set.
  scf::IfOp::create(builder, loc, isSet, [&](OpBuilder &then, Location) {
    createTransfer(then, loc, flag.device, flag.clear);
    scf::YieldOp::create(then, loc);
  });
  return isSet;
}

SmallVector<Value>
Lowering::emitReductions(OpBuilder &builder, Location loc,
                         ArrayRef<Value> contributions,
                         ArrayRef<Value> partials, Value size, bool isSum) {
  if (contributions.empty())
    return {};

  // A number, or a vector of numbers.
  SmallVector<Type> types;
  for (Value buffer : contributions)
    types.push_back(md_exec::getKernelValueType(buffer.getType()));
  auto combine = [&](OpBuilder &b, Value lhs, Value rhs) -> Value {
    if (isSum)
      return arith::AddFOp::create(b, loc, lhs, rhs);
    Value larger =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, lhs, rhs);
    return arith::SelectOp::create(b, loc, larger, lhs, rhs);
  };
  auto createZeros = [&](OpBuilder &b) {
    SmallVector<Value> zeros;
    for (Type type : types)
      zeros.push_back(createZero(b, loc, type));
    return zeros;
  };

  // A block of threads for each part of the particles, and one block for
  // the results of the parts. A thread adds up the particles that it takes,
  // every so manyth, and the block adds up its threads by a tree
  // (`gpu.all_reduce`), so that the order of the sum, and the sum, depend
  // only on the number of particles. A block takes at least four
  // particles per thread, and there are no more parts than threads in a
  // block.
  auto reduceBlock = [&](OpBuilder &b, Value value) -> Value {
    auto operation = gpu::AllReduceOperationAttr::get(
        b.getContext(), isSum ? gpu::AllReduceOperation::ADD
                              : gpu::AllReduceOperation::MAXNUMF);
    auto reduceNumber = [&](Value number) -> Value {
      return gpu::AllReduceOp::create(b, loc, number, operation,
                                      b.getUnitAttr());
    };
    if (auto vector = dyn_cast<VectorType>(value.getType())) {
      SmallVector<Value> elements;
      for (int64_t i = 0, e = vector.getNumElements(); i != e; ++i)
        elements.push_back(reduceNumber(
            vector::ExtractOp::create(b, loc, value, i)));
      return vector::FromElementsOp::create(b, loc, vector, elements);
    }
    return reduceNumber(value);
  };
  auto launchBlocks = [&](Value blocks,
                          function_ref<void(OpBuilder &, Value, Value)> body) {
    Value one = createIndex(builder, loc, 1);
    Value threads = createIndex(builder, loc, blockSize);
    auto launch = gpu::LaunchOp::create(builder, loc, blocks, one, one,
                                        threads, one, one);
    OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
    body(kernel, launch.getBlockIds().x, launch.getThreadIds().x);
    gpu::TerminatorOp::create(kernel, loc);
    bringIn(launch);
  };
  Value parts = arith::MinSIOp::create(
      builder, loc, createIndex(builder, loc, blockSize),
      createGroups(builder, loc, size, 4 * blockSize));
  launchBlocks(parts, [&](OpBuilder &b, Value block, Value thread) {
    Value threads = createIndex(b, loc, blockSize);
    Value first = arith::AddIOp::create(
        b, loc, arith::MulIOp::create(b, loc, block, threads), thread);
    Value stride = arith::MulIOp::create(b, loc, parts, threads);
    auto loop = scf::ForOp::create(
        b, loc, first, size, stride, createZeros(b),
        [&](OpBuilder &inner, Location, Value i, ValueRange sums) {
          SmallVector<Value> next;
          for (auto [buffer, sum] : llvm::zip(contributions, sums))
            next.push_back(
                combine(inner, sum, loadElement(inner, loc, buffer, i)));
          scf::YieldOp::create(inner, loc, next);
        });
    SmallVector<Value> totals;
    for (Value value : loop.getResults())
      totals.push_back(reduceBlock(b, value));
    Value leader = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                         thread, createIndex(b, loc, 0));
    scf::IfOp::create(b, loc, leader, [&](OpBuilder &then, Location) {
      for (auto [total, partial] : llvm::zip(totals, partials))
        storeElement(then, loc, total, partial, block);
      scf::YieldOp::create(then, loc);
    });
  });

  // Where the results arrive. Numbers of one type are next to one another,
  // so that one copy brings them to the host.
  struct Place {
    Cell cell;
    int64_t offset;
    int64_t count;
  };
  llvm::MapVector<Type, int64_t> totals;
  SmallVector<Place> places;
  for (Type type : types) {
    Type element = getElementTypeOrSelf(type);
    auto vector = dyn_cast<VectorType>(type);
    int64_t count = vector ? vector.getNumElements() : 1;
    places.push_back({Cell(), totals[element], count});
    totals[element] += count;
  }
  // Each sum has slots of its own in the buffer of results of its type,
  // so that the copy to the host can wait until the host uses them
  // (deferReadbacks); a function with more sums than the buffer holds
  // shares a cell among them and reads at once.
  bool deferred = llvm::all_of(totals, [&](auto &entry) {
    return nextResult[entry.first] + entry.second <= resultCapacity;
  });
  llvm::DenseMap<Type, int64_t> base;
  for (auto &[element, total] : totals) {
    base[element] = deferred ? nextResult[element] : 0;
    if (deferred)
      nextResult[element] += total;
  }
  for (auto [index, type] : llvm::enumerate(types)) {
    Type element = getElementTypeOrSelf(type);
    places[index].cell = deferred ? getResults(element, loc)
                                  : getCell(element, totals[element], loc);
    places[index].offset += base[element];
  }

  // One block for the results of the parts.
  launchBlocks(createIndex(builder, loc, 1), [&](OpBuilder &b, Value,
                                                 Value thread) {
    Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                         thread, parts);
    Value slot = arith::SelectOp::create(b, loc, inside, thread,
                                         createIndex(b, loc, 0));
    SmallVector<Value> totals;
    for (auto [partial, type] : llvm::zip(partials, types)) {
      Value value = loadElement(b, loc, partial, slot);
      Value nothing = createZero(b, loc, type);
      value = arith::SelectOp::create(b, loc, inside, value, nothing);
      totals.push_back(reduceBlock(b, value));
    }
    Value leader = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                         thread, createIndex(b, loc, 0));
    scf::IfOp::create(b, loc, leader, [&](OpBuilder &then, Location) {
      for (auto [index, place] : llvm::enumerate(places)) {
        Value result = totals[index];
        for (int64_t c = 0; c != place.count; ++c) {
          Value number = isa<VectorType>(result.getType())
                             ? Value(vector::ExtractOp::create(then, loc,
                                                               result, c))
                             : result;
          memref::StoreOp::create(
              then, loc, number, place.cell.device,
              ValueRange{createIndex(then, loc, place.offset + c)});
        }
      }
      scf::YieldOp::create(then, loc);
    });
  });

  // The copies and the loads of the results are marked, so that they can
  // be moved to where the host first needs them.
  int64_t readback = numReadbacks++;
  auto mark = [&](Operation *op) {
    if (deferred)
      op->setAttr("mdir.readback", builder.getI64IntegerAttr(readback));
  };
  for (auto &[element, total] : totals) {
    Cell cell = deferred ? getResults(element, loc)
                         : getCell(element, total, loc);
    Type token = gpu::AsyncTokenType::get(context);
    auto begin = gpu::WaitOp::create(builder, loc, token, ValueRange());
    auto copy = gpu::MemcpyOp::create(builder, loc, token,
                                      ValueRange{begin.getAsyncToken()},
                                      cell.host, cell.device);
    auto done = gpu::WaitOp::create(builder, loc, Type(),
                                    ValueRange{copy.getAsyncToken()});
    for (Operation *op : {begin.getOperation(), copy.getOperation(),
                          done.getOperation()})
      mark(op);
  }

  SmallVector<Value> results;
  for (auto [place, type] : llvm::zip(places, types)) {
    SmallVector<Value, 9> numbers;
    for (int64_t c = 0; c != place.count; ++c) {
      auto load = memref::LoadOp::create(
          builder, loc, place.cell.host,
          ValueRange{createIndex(builder, loc, place.offset + c)});
      mark(load);
      mark(load.getIndices()[0].getDefiningOp());
      numbers.push_back(load);
    }
    if (isa<VectorType>(type))
      results.push_back(
          vector::FromElementsOp::create(builder, loc, type, numbers));
    else
      results.push_back(numbers.front());
  }
  return results;
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
  SmallVector<Value> contributions, partials;
  for (unsigned i = 0, e = reduce.size(); i != e; ++i) {
    contributions.push_back(scratch[2 * i]);
    partials.push_back(scratch[2 * i + 1]);
  }
  SmallVector<Value> sums = emitReductions(builder, loc, contributions,
                                           partials, size, /*isSum=*/true);
  for (unsigned i = 0, e = reduce.size(); i != e; ++i) {
    Value total = arith::AddFOp::create(builder, loc, reduce[i], sums[i]);
    op->getResult(i).replaceAllUsesWith(total);
  }
  return success();
}

static void storeContributions(OpBuilder &builder, Location loc,
                               ArrayRef<Value> contributions,
                               ValueRange scratch, Value particle,
                               const RowLanes &sharing);

void Lowering::findRows(func::FuncOp function) {
  auto isRowLoop = [](Operation *op) {
    if (auto tuple = dyn_cast<md_exec::TupleForOp>(op))
      return !tuple.getDisjoint();
    return isa<md_exec::PairForOp>(op);
  };
  auto getPositions = [](Operation *op) -> Value {
    if (auto pair = dyn_cast<md_exec::PairForOp>(op))
      return pair.getPositions();
    return cast<md_exec::TupleForOp>(op).getPositions();
  };
  auto getOuts = [](Operation *op) -> ValueRange {
    if (auto pair = dyn_cast<md_exec::PairForOp>(op))
      return pair.getOuts();
    return cast<md_exec::TupleForOp>(op).getOuts();
  };
  auto finish = [&](SmallVector<Operation *> &run) {
    if (run.size() >= 2 && fuseRows) {
      rows[run.back()] = run;
      inRows.insert(run.begin(), run.end());
    }
    run.clear();
  };
  function.walk([&](Block *block) {
    SmallVector<Operation *> run;
    DenseSet<Value> written, sums;
    for (Operation &op : *block) {
      if (!isRowLoop(&op)) {
        // Constants may lie between the loops of a run; they are not
        // loops, and are lowered where they are.
        if (isa<arith::ConstantOp>(op))
          continue;
        finish(run);
        written.clear();
        sums.clear();
        continue;
      }
      // A loop joins the run if it takes the same positions, reads nothing
      // that a loop of the run writes, so that the order of the loops
      // within a thread is the only order that matters, and has buffers
      // for its global sums of its own: storage may give loops that ran
      // one after the other the same buffers, which one kernel would
      // write at once.
      ValueRange scratch = isa<md_exec::PairForOp>(op)
                               ? cast<md_exec::PairForOp>(op).getScratch()
                               : cast<md_exec::TupleForOp>(op).getScratch();
      bool joins = !run.empty() && getPositions(&op) == getPositions(run[0]);
      if (joins)
        for (Value operand : op.getOperands())
          if ((written.contains(operand) &&
               !llvm::is_contained(getOuts(&op), operand)) ||
              sums.contains(operand))
            joins = false;
      if (!joins) {
        finish(run);
        written.clear();
        sums.clear();
      }
      run.push_back(&op);
      for (Value out : getOuts(&op))
        written.insert(out);
      for (Value buffer : scratch)
        sums.insert(buffer);
    }
    finish(run);
  });
}

LogicalResult Lowering::lowerRows(ArrayRef<Operation *> run) {
  Operation *last = run.back();
  // The kernel of a run is named after its first loop and its length.
  setPurpose(run.front());
  purpose += "_run" + std::to_string(run.size());
  Location loc = last->getLoc();
  OpBuilder builder(last);

  // What each loop needs from the host: its structure of neighbors, and
  // the cell as the vector of its edge lengths.
  struct Loop {
    Operation *op;
    Neighbors structure;
    Value box, inverse;
  };
  SmallVector<Loop> loops;
  Value positions;
  for (Operation *op : run) {
    Loop loop;
    loop.op = op;
    Value cell;
    if (auto pair = dyn_cast<md_exec::PairForOp>(op)) {
      if (failed(checkScratch(op, pair.getReduce().size(),
                              pair.getScratch().size())) ||
          failed(getNeighbors(op, pair.getNeighbors(), loop.structure)))
        return failure();
      positions = pair.getPositions();
      cell = pair.getCellMutable().get();
    } else {
      auto tuple = cast<md_exec::TupleForOp>(op);
      if (failed(checkScratch(op, tuple.getReduce().size(),
                              tuple.getScratch().size())))
        return failure();
      positions = tuple.getPositions();
      cell = tuple.getCellMutable().get();
    }
    Type real = cast<MemRefType>(positions.getType()).getElementType();
    loop.box = convertReal(builder, loc, cell, real);
    loop.inverse = createInverse(builder, loc, loop.box);
    loops.push_back(loop);
  }
  Value size = createSize(builder, loc, positions);

  launchRows(builder, loc, size, [&](OpBuilder &body, Value particle,
                                     const RowLanes &sharing) {
    for (Loop &loop : loops) {
      IRMapping local;
      SmallVector<Value> contributions;
      ValueRange scratch;
      if (auto pair = dyn_cast<md_exec::PairForOp>(loop.op)) {
        contributions = emitPairKernel(
            body, pair, loop.structure.counts, loop.structure.index,
            loop.box, loop.inverse, particle, local, &sharing);
        scratch = pair.getScratch();
      } else {
        auto tuple = cast<md_exec::TupleForOp>(loop.op);
        contributions =
            emitTupleKernel(body, tuple, tuple.getIncidence(), loop.box,
                            loop.inverse, particle, local, &sharing);
        scratch = tuple.getScratch();
      }
      storeContributions(body, loc, contributions, scratch, particle,
                         sharing);
    }
  });

  // One reduction for the global sums of all the loops.
  SmallVector<Value> contributions, partials;
  for (Loop &loop : loops) {
    ValueRange scratch = isa<md_exec::PairForOp>(loop.op)
                             ? cast<md_exec::PairForOp>(loop.op).getScratch()
                             : cast<md_exec::TupleForOp>(loop.op).getScratch();
    for (unsigned i = 0, e = loop.op->getNumResults(); i != e; ++i) {
      contributions.push_back(scratch[2 * i]);
      partials.push_back(scratch[2 * i + 1]);
    }
  }
  SmallVector<Value> sums = emitReductions(builder, loc, contributions,
                                           partials, size, /*isSum=*/true);
  unsigned next = 0;
  for (Loop &loop : loops) {
    ValueRange reduce = isa<md_exec::PairForOp>(loop.op)
                            ? cast<md_exec::PairForOp>(loop.op).getReduce()
                            : cast<md_exec::TupleForOp>(loop.op).getReduce();
    for (unsigned i = 0, e = loop.op->getNumResults(); i != e; ++i) {
      Value total =
          arith::AddFOp::create(builder, loc, reduce[i], sums[next++]);
      loop.op->getResult(i).replaceAllUsesWith(total);
    }
  }
  return success();
}

LogicalResult Lowering::lowerParticleFor(md_exec::ParticleForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // A value in `reduce` is a sum, which takes two buffers, or tells whether
  // the kernel yields true for any particle, which takes a flag.
  SmallVector<int> places;
  unsigned numSums = 0, numFlags = 0;
  for (Value value : op.getReduce())
    places.push_back(value.getType().isInteger(1) ? -int(++numFlags)
                                                  : int(numSums++));
  if (failed(checkScratch(op, numSums, op.getScratch().size())))
    return failure();
  SmallVector<Flag, 2> used;
  for (unsigned i = 0; i != numFlags; ++i)
    used.push_back(getFlag(i, loc));

  Value size = createSize(builder, loc, op.getIns().empty()
                                            ? op.getOuts().front()
                                            : op.getIns().front());
  launchOver(builder, loc, size, [&](OpBuilder &body, Value particle) {
    IRMapping local;
    SmallVector<Value> contributions =
        emitParticleKernel(body, op, particle, local);
    for (auto [index, value] : llvm::enumerate(contributions)) {
      int place = places[index];
      if (place >= 0) {
        storeElement(body, loc, value, op.getScratch()[2 * place], particle);
        continue;
      }
      // Every thread that writes the flag writes the same value, so the
      // threads need not take turns.
      Value device = used[-place - 1].device;
      scf::IfOp::create(body, loc, value, [&](OpBuilder &then, Location) {
        Value set = arith::ConstantOp::create(then, loc, then.getI32Type(),
                                              then.getI32IntegerAttr(1));
        memref::StoreOp::create(then, loc, set, device,
                                ValueRange{createIndex(then, loc, 0)});
        scf::YieldOp::create(then, loc);
      });
    }
  });

  SmallVector<Value> contributions, partials;
  for (unsigned i = 0; i != numSums; ++i) {
    contributions.push_back(op.getScratch()[2 * i]);
    partials.push_back(op.getScratch()[2 * i + 1]);
  }
  SmallVector<Value> sums = emitReductions(builder, loc, contributions,
                                           partials, size, /*isSum=*/true);

  for (auto [index, start] : llvm::enumerate(op.getReduce())) {
    int place = places[index];
    Value total;
    if (place >= 0) {
      total = arith::AddFOp::create(builder, loc, start, sums[place]);
    } else {
      Value isSet = readFlag(builder, loc, used[-place - 1]);
      total = arith::OrIOp::create(builder, loc, start, isSet);
    }
    op.getResult(index).replaceAllUsesWith(total);
  }
  return success();
}

/// `value` from the thread whose number in the warp differs from that of
/// this one by `offset` in its bits. Numbers of 64 bits travel as two of 32,
/// and vectors element by element.
static Value shuffleXor(OpBuilder &builder, Location loc, Value value,
                        int64_t offset) {
  Type type = value.getType();
  if (auto vector = dyn_cast<VectorType>(type)) {
    SmallVector<Value> elements;
    for (int64_t i = 0, e = vector.getNumElements(); i != e; ++i)
      elements.push_back(shuffleXor(
          builder, loc, vector::ExtractOp::create(builder, loc, value, i),
          offset));
    return vector::FromElementsOp::create(builder, loc, vector, elements);
  }
  Type i32 = builder.getI32Type();
  Value distance = arith::ConstantOp::create(
      builder, loc, i32, builder.getI32IntegerAttr(offset));
  Value width =
      arith::ConstantOp::create(builder, loc, i32, builder.getI32IntegerAttr(32));
  auto shuffle = [&](Value word) {
    return gpu::ShuffleOp::create(builder, loc, word, distance, width,
                                  gpu::ShuffleMode::XOR)
        .getShuffleResult();
  };
  if (type.getIntOrFloatBitWidth() == 32)
    return shuffle(value);
  Type i64 = builder.getI64Type();
  Value bits = arith::BitcastOp::create(builder, loc, i64, value);
  Value shift =
      arith::ConstantOp::create(builder, loc, i64, builder.getI64IntegerAttr(32));
  Value low = arith::TruncIOp::create(builder, loc, i32, bits);
  Value high = arith::TruncIOp::create(
      builder, loc, i32, arith::ShRUIOp::create(builder, loc, bits, shift));
  Value lowBack =
      arith::ExtUIOp::create(builder, loc, i64, shuffle(low));
  Value highBack = arith::ShLIOp::create(
      builder, loc, arith::ExtUIOp::create(builder, loc, i64, shuffle(high)),
      shift);
  Value joined = arith::OrIOp::create(builder, loc, lowBack, highBack);
  return arith::BitcastOp::create(builder, loc, type, joined);
}

/// Launches `rowLanes` threads for each of `count` particles, in groups of
/// adjacent threads of one warp, and emits `body` for each with the
/// particle and how its group shares the row. Every thread of a launched
/// block runs the body, so that the warps stay whole for the shuffles;
/// those beyond the last particle take the particle 0, not valid.
void Lowering::launchRows(
    OpBuilder &builder, Location loc, Value count,
    function_ref<void(OpBuilder &, Value, const RowLanes &)> body) {
  Value lanes = createIndex(builder, loc, rowLanes);
  Value threads = arith::MulIOp::create(builder, loc, count, lanes);
  Value one = createIndex(builder, loc, 1);
  Value block = createIndex(builder, loc, blockSize);
  Value grid = createGroups(builder, loc, threads, blockSize);
  auto launch =
      gpu::LaunchOp::create(builder, loc, grid, one, one, block, one, one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value base = arith::MulIOp::create(kernel, loc, launch.getBlockIds().x,
                                     createIndex(kernel, loc, blockSize));
  Value thread =
      arith::AddIOp::create(kernel, loc, base, launch.getThreadIds().x);
  Value item = arith::DivUIOp::create(kernel, loc, thread, lanes);
  RowLanes sharing;
  sharing.lane = arith::RemUIOp::create(kernel, loc, thread, lanes);
  sharing.lanes = rowLanes;
  sharing.valid = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                        item, count);
  sharing.combine = [lanes = rowLanes](OpBuilder &builder, Location loc,
                                       Value value) {
    for (int64_t offset = lanes / 2; offset >= 1; offset /= 2)
      value = arith::AddFOp::create(builder, loc, value,
                                    shuffleXor(builder, loc, value, offset));
    return value;
  };
  Value particle = arith::SelectOp::create(kernel, loc, sharing.valid, item,
                                           createIndex(kernel, loc, 0));
  body(kernel, particle, sharing);
  gpu::TerminatorOp::create(kernel, loc);
  bringIn(launch);
}

/// Stores `contributions` to the global sums of the particle in `scratch`,
/// from the first thread of a group whose particle is valid.
static void storeContributions(OpBuilder &builder, Location loc,
                               ArrayRef<Value> contributions,
                               ValueRange scratch, Value particle,
                               const RowLanes &sharing) {
  if (contributions.empty())
    return;
  Value leader = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       sharing.lane,
                                       createIndex(builder, loc, 0));
  Value writes = arith::AndIOp::create(builder, loc, leader, sharing.valid);
  scf::IfOp::create(builder, loc, writes, [&](OpBuilder &then, Location) {
    for (auto [index, value] : llvm::enumerate(contributions))
      storeElement(then, loc, value, scratch[2 * index], particle);
    scf::YieldOp::create(then, loc);
  });
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

  launchRows(builder, loc, size, [&](OpBuilder &body, Value central,
                                     const RowLanes &sharing) {
    IRMapping local;
    SmallVector<Value> contributions =
        emitPairKernel(body, op, structure.counts, structure.index, box,
                       inverse, central, local, &sharing);
    storeContributions(body, loc, contributions, op.getScratch(), central,
                       sharing);
  });
  return finishSums(op, builder, op.getReduce(), op.getScratch(), size);
}

LogicalResult Lowering::lowerTupleFor(md_exec::TupleForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  if (failed(checkScratch(op, op.getReduce().size(), op.getScratch().size())))
    return failure();

  Value positions = op.getPositions();
  Value size = createSize(builder, loc, positions);

  // The cell has become the vector of its edge lengths.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);

  // A set whose tuples share no particle has each tuple evaluated once, by
  // the thread of its first member; its rows hold one tuple at most, which
  // a group of threads would not share.
  if (op.getDisjoint()) {
    launchOver(builder, loc, size, [&](OpBuilder &body, Value particle) {
      IRMapping local;
      SmallVector<Value> contributions = emitTupleKernel(
          body, op, op.getIncidence(), box, inverse, particle, local);
      for (auto [index, value] : llvm::enumerate(contributions))
        storeElement(body, loc, value, op.getScratch()[2 * index], particle);
    });
    return finishSums(op, builder, op.getReduce(), op.getScratch(), size);
  }
  launchRows(builder, loc, size, [&](OpBuilder &body, Value particle,
                                     const RowLanes &sharing) {
    IRMapping local;
    SmallVector<Value> contributions = emitTupleKernel(
        body, op, op.getIncidence(), box, inverse, particle, local, &sharing);
    storeContributions(body, loc, contributions, op.getScratch(), particle,
                       sharing);
  });
  return finishSums(op, builder, op.getReduce(), op.getScratch(), size);
}

/// The structure is built on the host, where the members are, and copied to
/// the device.
void Lowering::lowerBuildIncidence(md_exec::BuildIncidenceOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value host =
      emitBuildIncidence(builder, loc, op.getRelation(), op.getSize());
  auto type = cast<MemRefType>(op.getResult().getType());
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  Value device = createDeviceBuffer(
      builder, loc, type,
      ValueRange{memref::DimOp::create(builder, loc, host, zero),
                 memref::DimOp::create(builder, loc, host, one)});
  createTransfer(builder, loc, device, host);
  memref::DeallocOp::create(builder, loc, host);
  op.getResult().replaceAllUsesWith(device);
  freeDeviceAtEndOfBlock(op, device);
}

void Lowering::freeDeviceAtEndOfBlock(Operation *op, Value buffer) {
  // The lowering of `gpu.dealloc` takes a buffer without a memory space.
  OpBuilder builder(op->getBlock()->getTerminator());
  auto type = cast<MemRefType>(buffer.getType());
  Value plain = memref::MemorySpaceCastOp::create(
      builder, op->getLoc(),
      MemRefType::get(type.getShape(), type.getElementType()), buffer);
  gpu::DeallocOp::create(builder, op->getLoc(), /*asyncToken=*/Type(),
                         /*asyncDependencies=*/ValueRange(), plain);
}

/// The members are renumbered on the host; the numbers of the particles
/// are copied there from the device.
void Lowering::lowerRenumber(md_exec::RenumberOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value ids = op.getIds();
  auto type = cast<MemRefType>(ids.getType());
  Value host = memref::AllocOp::create(
      builder, loc, MemRefType::get(type.getShape(), type.getElementType()),
      ValueRange{memref::DimOp::create(builder, loc, ids,
                                       createIndex(builder, loc, 0))});
  createTransfer(builder, loc, host, ids);
  Value members = emitRenumber(builder, loc, op.getMembers(), host);
  memref::DeallocOp::create(builder, loc, host);
  op.getResult().replaceAllUsesWith(members);
  freeAtEndOfBlock(op, members);
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

LogicalResult Lowering::addPMETemplates(Type position, Type charge,
                                        Type force) {
  if (SymbolTable::lookupSymbolIn(
          module, getPMEInstanceName("mdrt_gpu_pme_spread", position, charge,
                                     force)))
    return success();
  ParserConfig config(context);
  OwningOpRef<ModuleOp> templates = parseSourceString<ModuleOp>(
      instantiatePMETemplates(pmeGPUTemplate, position, charge, force),
      config);
  if (!templates)
    return module.emitError()
           << "cannot parse the template of particle mesh Ewald for devices";
  for (Operation &op : llvm::make_early_inc_range(*templates)) {
    op.remove();
    module.push_back(&op);
  }
  return success();
}

LogicalResult Lowering::lowerReciprocal(md_exec::ReciprocalOp op) {
  if (!op.isStorageForm() || !isDeviceType(op.getPositions().getType()))
    return op->emitOpError()
           << "expected the storage form with the buffers on the device; "
              "run 'md-exec-assign-storage' with 'memory=device'";
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value positions = op.getPositions(), charges = op.getCharges();
  Value forces = op.getOut();
  auto elementOf = [](Value buffer) {
    return cast<MemRefType>(buffer.getType()).getElementType();
  };
  Type position = elementOf(positions), charge = elementOf(charges),
       force = elementOf(forces);
  if (failed(addPMETemplates(position, charge, force)))
    return failure();
  auto instance = [&](StringRef name) {
    return cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
        module, getPMEInstanceName(name, position, charge, force)));
  };

  ArrayRef<int64_t> grid = op.getGrid();
  Value k1 = createIndex(builder, loc, grid[0]);
  Value k2 = createIndex(builder, loc, grid[1]);
  Value k3 = createIndex(builder, loc, grid[2]);
  Value order = createIndex(builder, loc, op.getOrder());
  Value beta = arith::ConstantOp::create(
      builder, loc, builder.getF64FloatAttr(op.getBeta().convertToDouble()));
  Value coulomb = arith::ConstantOp::create(
      builder, loc,
      builder.getF64FloatAttr(op.getCoulomb().convertToDouble()));
  // The cell is the vector of its edge lengths by now.
  Value box = op.getCellMutable().get();
  Value fixed = op.getScratch()[0], real = op.getScratch()[1],
        complex = op.getScratch()[2], rows = op.getScratch()[3];

  // On a second stream the sum runs beside the loops that follow it, from
  // the positions as they are here, until an op reads its forces (or the
  // buffers it reads are written), where the first stream waits for it.
  auto sideCall = [&](OpBuilder &at, StringRef name) {
    func::FuncOp function =
        getOrDeclare(name, at.getFunctionType({}, {}));
    func::CallOp::create(at, loc, function, ValueRange());
  };
  if (pmeStream)
    sideCall(builder, "mdrtSideBegin");
  func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_spread"),
                       ValueRange{positions, charges, box, fixed, k1, k2, k3,
                                  order});
  func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_real"),
                       ValueRange{fixed, real});
  Type wide = builder.getI64Type();
  SmallVector<Value> sizes;
  for (int64_t points : grid)
    sizes.push_back(arith::ConstantOp::create(
        builder, loc, wide, builder.getI64IntegerAttr(points)));
  Type buffer = real.getType();
  FunctionType transform =
      builder.getFunctionType({buffer, buffer, wide, wide, wide}, {});
  // The transforms of cuFFT in the type of the grid.
  bool narrow = cast<MemRefType>(buffer).getElementType().isF32();
  func::FuncOp forward = getOrDeclare(
      narrow ? "mdrtCudaFFTForward3DF32" : "mdrtCudaFFTForward3D", transform);
  func::FuncOp backward = getOrDeclare(
      narrow ? "mdrtCudaFFTBackward3DF32" : "mdrtCudaFFTBackward3D",
      transform);
  for (func::FuncOp function : {forward, backward})
    function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func::CallOp::create(builder, loc, forward,
                       ValueRange{real, complex, sizes[0], sizes[1], sizes[2]});
  // Without its energy and virial, a step only scales the transform, and
  // the host need not wait for the sums of the rows.
  func::CallOp convolve;
  if (op.getEnergy().use_empty() && op.getVirial().use_empty())
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_scale"),
                         ValueRange{complex, op.getModuli(), box, beta,
                                    coulomb, k1, k2, k3});
  else
    convolve = func::CallOp::create(
        builder, loc, instance("mdrt_gpu_pme_convolve"),
        ValueRange{complex, op.getModuli(), rows, box, beta, coulomb, k1, k2,
                   k3});
  func::CallOp::create(builder, loc, backward,
                       ValueRange{complex, real, sizes[0], sizes[1], sizes[2]});
  func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_gather"),
                       ValueRange{positions, charges, real, box, k1, k2, k3,
                                  order, forces});
  if (pmeStream) {
    sideCall(builder, "mdrtSideEnd");
    // The first op after the sum, in its block, that reads its forces or
    // writes what it reads, or that is not a loop over pairs, tuples, or
    // particles, which could do either by other means.
    Operation *join = nullptr;
    for (Operation *next = op->getNextNode(); next;
         next = next->getNextNode()) {
      // Loops may read what the sum reads at the same time; they may not
      // write it, nor read the forces it writes.
      ValueRange outs;
      if (auto loop = dyn_cast<md_exec::PairForOp>(next))
        outs = loop.getOuts();
      else if (auto loop = dyn_cast<md_exec::TupleForOp>(next))
        outs = loop.getOuts();
      else if (auto loop = dyn_cast<md_exec::ParticleForOp>(next))
        outs = loop.getOuts();
      bool touches =
          llvm::is_contained(next->getOperands(), forces) ||
          llvm::any_of(outs, [&](Value out) {
            return out == positions || out == charges;
          });
      bool isLoop = isa<md_exec::PairForOp, md_exec::TupleForOp,
                        md_exec::ParticleForOp, arith::ConstantOp>(next);
      if (touches || !isLoop) {
        join = next;
        break;
      }
    }
    OpBuilder at(op->getContext());
    if (join)
      at.setInsertionPoint(join);
    else
      at.setInsertionPoint(op->getBlock()->getTerminator());
    sideCall(at, "mdrtSideJoin");
  }
  if (convolve) {
    op.getEnergy().replaceAllUsesWith(convolve.getResult(0));
    op.getVirial().replaceAllUsesWith(convolve.getResult(1));
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
  structure.entries =
      arith::MulIOp::create(builder, loc, structure.size, structure.width);
  structure.counts = createDeviceBuffer(
      builder, loc, getDeviceType({ShapedType::kDynamic}, narrow),
      ValueRange{structure.size});
  structure.index = createDeviceBuffer(
      builder, loc,
      getDeviceType({ShapedType::kDynamic, ShapedType::kDynamic}, narrow),
      ValueRange{structure.size, structure.width});
  structure.reference =
      createDeviceBuffer(builder, loc, positions, ValueRange{structure.size});
  // The test of validity reads the configuration before the first build,
  // and ignores what it finds; it reads zeros, not memory that nothing has
  // written.
  {
    Value reference = structure.reference;
    Type element = positions.getElementType();
    launchOver(builder, loc, structure.size, [&](OpBuilder &body,
                                                 Value particle) {
      Value zero = createZero(body, loc, VectorType::get({3}, element));
      storeElement(body, loc, zero, reference, particle);
    });
  }

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
                 widthValue, createIndex(builder, loc, splitLimit),
                 structure.counts, structure.index});
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

  // Mark the excluded pairs, a thread for each entry of the matrix.
  if (structure.excluded) {
    launchOver(builder, loc, structure.entries, [&](OpBuilder &body,
                                                    Value item) {
      Value width = memref::DimOp::create(body, loc, structure.index,
                                          createIndex(body, loc, 1));
      Value particle = arith::DivUIOp::create(body, loc, item, width);
      Value slot = arith::RemUIOp::create(body, loc, item, width);
      emitExclusionMark(body, loc, structure.counts, structure.index,
                        structure.excluded, particle, slot);
    });
  }

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

  Value moved = op.getMoved();
  if (!moved && op.getScratch().size() != 2)
    return op.emitOpError()
           << "needs 2 buffers in 'scratch' for the test of validity; run "
              "'md-exec-assign-storage' with 'memory=device'";

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

  // Before the first build the configuration that the test compares with
  // holds nothing. The result of the test then does not count.
  Value yes = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(true));
  Value near;
  if (moved) {
    // A loop has made the test.
    near = arith::XOrIOp::create(builder, loc, moved, yes);
  } else {
    Value moved2 = op.getScratch()[0];
    launchOver(builder, loc, structure.size,
               [&](OpBuilder &body, Value particle) {
                 Value now = loadElement(body, loc, positions, particle);
                 Value then = arith::MulFOp::create(
                     body, loc,
                     loadElement(body, loc, structure.reference, particle),
                     scaleReal);
                 Value change = arith::SubFOp::create(body, loc, now, then);
                 Value squares =
                     arith::MulFOp::create(body, loc, change, change);
                 Value distance2 = vector::ReductionOp::create(
                     body, loc, vector::CombiningKind::ADD, squares);
                 memref::StoreOp::create(body, loc, distance2, moved2,
                                         ValueRange{particle});
               });
    Value farthest = emitReductions(builder, loc, {moved2},
                                    {op.getScratch()[1]}, structure.size,
                                    /*isSum=*/false)
                         .front();
    Value limit = limitReal;
    near = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE,
                                 farthest, limit);
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
  launchOver(builder, loc, size, [&](OpBuilder &body, Value place) {
    Value from =
        memref::LoadOp::create(body, loc, order, ValueRange{place});
    Value particle =
        arith::IndexCastOp::create(body, loc, body.getIndexType(), from);
    storeElement(body, loc, loadElement(body, loc, field, particle), out,
                 place);
  });
}

//===----------------------------------------------------------------------===//
// Ops and functions
//===----------------------------------------------------------------------===//

/// Returns true if `type` belongs to the value form: a field, or a
/// structure that the storage form does not have.
static bool isValueFormType(Type type) {
  return isa<md::FieldType, md::RelationType, mdrt::CellsType,
             mdrt::PermutationType, mdrt::IncidenceType, md::TableType>(type);
}

LogicalResult Lowering::lowerOp(Operation *op) {
  setPurpose(op);
  if (llvm::any_of(op->getOperandTypes(), isValueFormType) ||
      llvm::any_of(op->getResultTypes(), isValueFormType))
    return op->emitOpError()
           << "is not in the storage form; run 'md-exec-assign-storage' "
              "first";

  // The members of tuples are on the host, and the incidence structure is
  // built there.
  if (isa<md_exec::MDExecDialect>(op->getDialect()) &&
      !isa<md_exec::BuildIncidenceOp, md_exec::RenumberOp>(op)) {
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
    if (failed(lowerParticleFor(loop)))
      return failure();
  } else if (auto loop = dyn_cast<md_exec::PairForOp>(op)) {
    if (failed(lowerPairFor(loop)))
      return failure();
  } else if (auto loop = dyn_cast<md_exec::TupleForOp>(op)) {
    if (failed(lowerTupleFor(loop)))
      return failure();
  } else if (auto build = dyn_cast<md_exec::BuildIncidenceOp>(op)) {
    if (!build.isStorageForm() ||
        !isDeviceType(build.getResult().getType()))
      return op->emitOpError()
             << "expected the storage form with the structure on the device; "
                "run 'md-exec-assign-storage' with 'memory=device'";
    lowerBuildIncidence(build);
  } else if (auto renumber = dyn_cast<md_exec::RenumberOp>(op)) {
    lowerRenumber(renumber);
  } else if (auto reciprocal = dyn_cast<md_exec::ReciprocalOp>(op)) {
    if (failed(lowerReciprocal(reciprocal)))
      return failure();
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

/// Writes out every power with a constant exponent in the kernels of
/// `function` as products, by squaring: a device would call a function of
/// libdevice for each, and a pair of Lennard-Jones has five of them.
static void expandPowers(func::FuncOp function) {
  SmallVector<math::FPowIOp> powers;
  function.walk([&](math::FPowIOp power) {
    if (power->getParentOfType<gpu::LaunchOp>())
      powers.push_back(power);
  });
  for (math::FPowIOp power : powers) {
    APInt exponent;
    if (!matchPattern(power.getRhs(), m_ConstantInt(&exponent)))
      continue;
    int64_t n = exponent.getSExtValue();
    if (n < -32 || n > 32)
      continue;
    OpBuilder builder(power);
    Location loc = power.getLoc();
    Value base = power.getLhs();
    Type type = base.getType();
    Value one = createReal(builder, loc, getElementTypeOrSelf(type), 1.0);
    if (auto vector = dyn_cast<VectorType>(type))
      one = vector::BroadcastOp::create(builder, loc, vector, one);
    Value result;
    Value square = base;
    for (int64_t m = n < 0 ? -n : n; m != 0; m >>= 1) {
      if (m & 1)
        result = result ? arith::MulFOp::create(builder, loc, result, square)
                        : square;
      if (m >> 1)
        square = arith::MulFOp::create(builder, loc, square, square);
    }
    if (!result)
      result = one;
    if (n < 0)
      result = arith::DivFOp::create(builder, loc, one, result);
    power.getResult().replaceAllUsesWith(result);
    power.erase();
  }
}

/// Moves each group of ops that reads global sums back to the host (marked
/// `mdir.readback` by emitReductions) down its block to just before the
/// first op that needs its values and may have effects, taking the pure ops
/// that use the values along. A copy that then follows another copy of the
/// same buffer with nothing but such ops between is left out: the sums
/// that it would bring were computed before the first copy, which brought
/// the whole buffer. The host then waits once where it used to wait for
/// each sum.
static void deferReadbacks(func::FuncOp function) {
  llvm::MapVector<int64_t, SmallVector<Operation *>> groups;
  function.walk([&](Operation *op) {
    if (auto id = op->getAttrOfType<IntegerAttr>("mdir.readback"))
      groups[id.getInt()].push_back(op);
  });
  auto isPure = [](Operation *op) {
    return op->getNumRegions() == 0 && isMemoryEffectFree(op);
  };
  for (auto &[id, group] : groups) {
    Block *block = group.front()->getBlock();
    if (llvm::any_of(group, [&](Operation *op) {
          return op->getBlock() != block;
        }))
      continue;
    // The ops that move: the group, and the pure ops of the block that use
    // what moves, until the first op that needs it and is not pure.
    llvm::SetVector<Operation *> moving(group.begin(), group.end());
    Operation *target = nullptr;
    for (Operation *op = group.front()->getNextNode(); op;
         op = op->getNextNode()) {
      if (moving.contains(op))
        continue;
      bool uses = false;
      op->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands())
          if (Operation *def = operand.getDefiningOp())
            uses |= moving.contains(def);
      });
      if (!uses)
        continue;
      if (isPure(op) && !op->hasTrait<OpTrait::IsTerminator>()) {
        moving.insert(op);
        continue;
      }
      target = op;
      break;
    }
    if (!target || target->isBeforeInBlock(group.back()))
      continue;
    for (Operation *op : moving)
      op->moveBefore(target);
  }

  // Copies that follow a copy of the same buffer.
  SmallVector<Operation *> redundant;
  function.walk([&](Block *block) {
    llvm::DenseSet<Value> copied;
    for (Operation &op : *block) {
      auto copy = dyn_cast<gpu::MemcpyOp>(op);
      if (copy && op.hasAttr("mdir.readback")) {
        if (!copied.insert(copy.getDst()).second) {
          redundant.push_back(*copy.getAsyncToken().user_begin());
          redundant.push_back(copy);
          redundant.push_back(
              copy.getAsyncDependencies().front().getDefiningOp());
        }
        continue;
      }
      if (op.hasAttr("mdir.readback") || isPure(&op))
        continue;
      copied.clear();
    }
  });
  for (Operation *op : redundant)
    op->erase();
  function.walk([](Operation *op) { op->removeAttr("mdir.readback"); });
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
  resultCells.clear();
  nextResult.clear();
  flags.clear();

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
  // lowered where they are. A run of loops is lowered at its last loop.
  findRows(function);
  for (Operation *op : ops) {
    Operation *parent = op->getParentOp();
    if (isa<md_exec::ParticleForOp, md_exec::PairForOp, md_exec::TupleForOp>(
            parent))
      continue;
    if (inRows.contains(op)) {
      auto run = rows.find(op);
      if (run != rows.end()) {
        if (failed(lowerRows(run->second)))
          return failure();
        for (Operation *loop : run->second)
          lowered.push_back(loop);
      }
      continue;
    }
    if (failed(lowerOp(op)))
      return failure();
  }
  rows.clear();
  inRows.clear();

  // Users come after what they use, so erase from the back.
  for (Operation *op : llvm::reverse(lowered))
    op->erase();
  lowered.clear();
  neighbors.clear();

  // The kernels read tables from their buffers.
  lowerLookups(function);
  expandPowers(function);
  deferReadbacks(function);

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
  if (rowLanes < 1 || rowLanes > 32 || (rowLanes & (rowLanes - 1)) != 0 ||
      blockSize % rowLanes != 0)
    return module.emitError()
           << "expected a power of two up to 32 that divides the block for "
              "'row-lanes'";

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
    if (isa<md::ParticleSetOp, md::TupleSetOp>(op)) {
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
    Lowering lowering(getOperation(), blockSize, splitLimit, rowLanes,
                      fuseRows, pmeStream);
    if (failed(lowering.run()))
      signalPassFailure();
  }
};
} // namespace

} // namespace mdir
