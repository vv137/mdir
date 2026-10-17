// Conversion of md_exec loops in the storage form to GPU kernels.
//
// Every op of md_exec becomes ops of the upstream gpu dialect, where the op
// is. See docs/ops-m0.md, Section 10.8.

#include "mdir/Conversion/Passes.h"

#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/Independence.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

#include <functional>
#include <limits>

namespace mdir {
/// The text of the template that builds a neighbor matrix on a device.
extern const char *const neighborsMatrixGPUTemplate;
/// The text of the template of particle mesh Ewald on a device.
extern const char *const pmeGPUTemplate;
/// The text of its kernels for a triclinic cell.
extern const char *const pmeGPUTriclinicTemplate;
/// The text of its kernels for the prolate kernel (D266).
extern const char *const pmeGPUProlateTemplate;
extern const char *const neighborsGroupsGPUTemplate;
extern const char *const neighborsGroupsGPUTriclinicTemplate;
} // namespace mdir

static const char *const spatialOrderName = "mdrt_gpu_spatial_order";
static const char *const cellWidthName = "mdrt_gpu_cell_width";
static const char *const buildNeighborsName =
    "mdrt_gpu_build_neighbors_matrix";
static const char *const countBuildName = "mdrtCountBuild";
static const char *const flagStartName = "mdrtFlagStart";
static const char *const flagFinishName = "mdrtFlagFinish";
static const char *const flagTakeName = "mdrtFlagTake";
static const char *const flagMemoryName = "mdrtFlagMemory";
/// The renumbered members, the incidence structures, and the copies of the
/// members on the device that the runtime keeps (mdrt_cuda.c,
/// D253).
static const char *const tuplesRenumberName = "mdrtDeviceTuplesRenumber";
static const char *const tuplesIncidenceName = "mdrtDeviceTuplesIncidence";
static const char *const tuplesMembersName = "mdrtDeviceTuplesMembers";
static const char *const tuplesReleaseName = "mdrtDeviceTuplesRelease";
/// Counts a build at an interval that found the structure no longer valid
/// (D88).
static const char *const countLateBuildName = "mdrtCountLateBuild";
static const char *const buildGroupsName = "mdrt_gpu_build_neighbors_groups";
static const char *const noteGroupsName = "mdrtNoteGroups";
static const char *const createGroupsName = "mdrtGroupsCreate";
static const char *const createMatrixName = "mdrtMatrixCreate";
static const char *const matrixEntriesName = "mdrtMatrixEntries";
static const char *const growMatrixName = "mdrtMatrixGrow";
static const char *const groupsBufferName = "mdrtGroupsBuffer";
static const char *const growGroupsName = "mdrtGroupsGrow";
static const char *const reserveBlockOfName = "mdrtGroupsReserveBlockOf";
static const char *const placedGroupsName = "mdrtGroupsPlaced";
static const char *const checkPrunedName = "mdrtCheckPruned";
/// The candidates and the slots of the pairs of images of a learned
/// potential (D277).
static const char *const createEdgesName = "mdrtEdgesCreate";
static const char *const reserveEdgesName = "mdrtEdgesReserve";
static const char *const edgesBufferName = "mdrtEdgesBuffer";

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
  /// The number of neighbors of each place in the order of the cells of
  /// the last build; the particle at each place (D86). The runtime holds
  /// the rows of the places of the neighbors, by `handle`, and makes them
  /// wider when a build finds them too narrow (getMatrixEntries).
  Value counts;
  Value order;
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
  /// What the runtime holds the buffers of the structure by.
  Value handle;
  /// A structure of groups of 16 (D89): the runtime holds its buffers,
  /// which grow when a build finds them too small, by `handle`, and gives
  /// them where they are used (getGroupsBuffers). `placeOf` is the place
  /// of each particle. `sizes`, on the host, holds the places, the longest
  /// list and the blocks of the last build.
  bool groups = false;
  Value placeOf;
  Value sizes;
  /// A dual list (D114): the loops over pairs take the inner list, which
  /// was pruned at the configuration `prunedReference` in the cell
  /// `prunedBox`. A structure of groups has both, whether it is dual or
  /// not; `dual` is set where a refresh keeps it dual.
  bool dual = false;
  Value prunedReference;
  Value prunedBox;
  /// Whether the last refresh of a dual list pruned it, on the host: the
  /// frames of the positions that the pruning left (mdrtGroupsPlaced) are
  /// then those of a loop over pairs that follows it at the same positions
  /// (D268).
  Value prunedNow;
  /// The pairs of images that `md_exec.build_edges` takes from a matrix
  /// (D277): each build counts the candidates of each
  /// particle, gives it the slots at its offset, and writes the candidates
  /// there, which the runtime holds by `edgesHandle` with the slots; the
  /// counts (of candidates at a build, of pairs within the cutoff after a
  /// step) and the offsets are on the device, and their copies to make the
  /// offsets on the host.
  bool edges = false;
  Value edgesHandle;
  Value edgeCounts, edgeOffsets;
  Value hostEdgeCounts, hostEdgeOffsets;
  /// The number that the runtime gave the last build of the slots, on the
  /// host: no two builds in a process have the same.
  Value edgeSerial;
};

/// The inner list of a dual list of groups (D114), as the runtime holds it:
/// each block of the outer list pruned into the same block, its entries at
/// the front, and their number.
struct InnerLists {
  Value entries, masks, counts;
};

/// The buffers of a structure of groups, as the runtime holds them now.
struct GroupBuffers {
  Value order, counts, entries, masks, units, ordinals, shifts;
};

/// What the pruning of a dual list keeps besides the inner list
/// (D268): the block of each group's list by its number in the
/// list, a row of the blocks of the longest list for each group, made at
/// each build; the units of work that hold entries of the inner list (the
/// active units), in the order of the blocks, and their number; and the
/// counts of the active units in runs of 32 and of 1024 blocks.
struct PruneBuffers {
  Value blockOf, activeUnits, activeCount, runSums, wideSums;
};

/// Where the global sums and maxima of a loop arrive: numbers on the
/// device, and the buffer of the host that they are copied to.
struct Cell {
  Value device;
  Value host;
  int64_t capacity = 0;
};

/// A flag that the threads of a kernel set: one value in memory of the host
/// that the runtime maps for the device (mdrtFlagMemory, D118), which the
/// runtime reads after an event and clears (mdrtFlagStart and
/// mdrtFlagFinish, D113).
struct Flag {
  Value device;
};

class Lowering {
public:
  Lowering(ModuleOp module, int64_t blockSize, int64_t rowLanes,
           bool fuseRows, bool fuseIntegration, bool deterministic,
           bool tuplesOnce, bool fuseCenters)
      : module(module), context(module.getContext()), blockSize(blockSize),
        rowLanes(rowLanes), fuseRows(fuseRows),
        fuseIntegration(fuseIntegration), deterministic(deterministic),
        tuplesOnce(tuplesOnce), fuseCenters(fuseCenters) {
    // A module with a triclinic cell lowers every cell to the vector of six,
    // a_x, b_y, c_z, b_x, c_x, c_y; one without, to the three edges.
    module.walk([&](md::TriclinicCellOp) { triclinic = true; });
  }

  LogicalResult run();
  /// Returns true if `op`, an op inside it, or a function that it calls
  /// may allocate or free memory of the device. A function of the runtime without a body
  /// may, unless it was audited not to.
  bool allocates(Operation *op);

  /// Whether the renumbered members of the tuples, their incidence
  /// structures, and the copies of the members on the device are taken from
  /// the runtime, which may keep them from one activation of an embedding
  /// program to the next (`kept-tuples`, D253).
  bool keptTuples = false;
  /// With `keptTuples`: whether the structures are built on the device, by
  /// kernels of the module (`device-tuples`, D257).
  bool deviceTuples = false;
  /// Whether each refresh of a dual list counts the pairs within the cutoff
  /// in its outer and inner lists and hands the counts to the runtime,
  /// which stops the run if they differ (`check-pruning`,
  /// D268).
  bool checkPruning = false;

private:
  /// An entry of the runtime's store for a tuple set built on the device:
  /// its handle, whether the code builds it, and its buffers on the device.
  struct DeviceTuples {
    Value handle, build, file, first, entries, renumbered, ids;
    int64_t arity = 0;
  };
  /// The entries taken where the members are renumbered, by the result of
  /// the renumbering.
  DenseMap<Value, DeviceTuples> deviceTuplesOf;
  /// The identities compared with the store's order, in a block.
  llvm::DenseSet<std::pair<Value, Block *>> comparedOrders;
  Value callRuntime(OpBuilder &builder, Location loc, StringRef name,
                    ValueRange arguments, Type result, bool memrefs);
  void compareOrder(Operation *op, Value ids);
  DeviceTuples takeOnDevice(Operation *op, Value members, Value ids,
                            Value particles);
  void buildOnDevice(md_exec::BuildIncidenceOp op, const DeviceTuples &tuples);
  /// A call of the runtime function `name` with `members`, a buffer of the
  /// host, as a buffer of any arity, and `other` if it is given, which gives
  /// a buffer of the type `type`; the runtime takes the buffer back before
  /// the end of the block of `op`.
  Value takeFromRuntime(Operation *op, StringRef name, Value members,
                        Value other, MemRefType type);
  /// The members that the runtime has renumbered.
  llvm::DenseSet<Value> runtimeMembers;

  LogicalResult lowerFunction(func::FuncOp function);
  void releaseStack(func::FuncOp function);
  LogicalResult lowerOp(Operation *op);

  void lowerEmptyNeighbors(md_exec::EmptyNeighborsOp op);
  LogicalResult lowerSpatialOrder(md_exec::SpatialOrderOp op);
  void lowerPermute(md_exec::PermuteOp op);
  void lowerNonzero(md_exec::NonzeroOp op);
  LogicalResult lowerRefreshNeighbors(md_exec::RefreshNeighborsOp op);
  LogicalResult lowerParticleFor(md_exec::ParticleForOp op);
  LogicalResult lowerPairFor(md_exec::PairForOp op);
  LogicalResult lowerGroupPairFor(md_exec::PairForOp op,
                                  const Neighbors &structure);
  /// The buffers of the structure of groups `handle`, as they are where
  /// `builder` is.
  GroupBuffers getGroupsBuffers(OpBuilder &builder, Location loc,
                                Value handle);
  /// The inner list of the dual list of groups `handle` (D114).
  InnerLists getInnerLists(OpBuilder &builder, Location loc, Value handle);
  /// The buffer `which` of the structure of groups `handle`, in the order
  /// of the runtime (mdrt_cuda.c).
  Value getGroupsBuffer(OpBuilder &builder, Location loc, Value handle,
                        int64_t which);
  /// The buffers of the pruning of the dual list `handle` besides the inner
  /// list (D268).
  PruneBuffers getPruneBuffers(OpBuilder &builder, Location loc,
                               Value handle);
  /// The positions of the places of `handle` in the frames of their groups
  /// at its last pruning, in f32.
  Value getGroupsPlaced(OpBuilder &builder, Location loc, Value handle);
  /// The groups of the last build of `structure`, and the blocks of its
  /// longest list (at least one).
  std::pair<Value, Value> getGroupsShape(OpBuilder &builder, Location loc,
                                         const Neighbors &structure);
  /// Makes the block of each group's list by its number in the list, after
  /// a build of the structure of groups `structure`, for the prunings of a
  /// dual list (D268).
  void emitGroupsBlockIndex(OpBuilder &builder, Location loc,
                            const Neighbors &structure);
  /// Writes the positions `positions` of the `places` places of `buffers`
  /// in the frames of their groups in the cell `box` to `placed`, in f32,
  /// as the loop over pairs takes them; with `reference`, the positions of
  /// the `particles` particles as they are stored there; and zero to the
  /// first `length` elements of each buffer of `cleared`.
  void emitFramePositions(OpBuilder &builder, Location loc,
                          const GroupBuffers &buffers, Value places,
                          Value positions, Value box, Value placed,
                          Value reference, Value particles,
                          ArrayRef<std::pair<Value, Value>> cleared);
  /// Counts the pairs within `cutoff` in the outer list of the dual list
  /// `structure` and in its inner list through its active units, at the
  /// positions `positions`, and hands them to the runtime, which stops the
  /// run if they differ (`check-pruning`, D268).
  void emitPruneCheck(OpBuilder &builder, Location loc,
                      const Neighbors &structure, Value positions, Value box,
                      double cutoff);
  /// Prunes the inner list of the dual list `structure` from its outer one
  /// at the positions `positions` in the cell `box`, with the reach
  /// `reach`, and remembers the configuration and the cell (D114).
  void emitGroupsPrune(OpBuilder &builder, Location loc,
                       const Neighbors &structure, Value positions,
                       Value box, double reach);
  /// The rows of the neighbor matrix `handle`, as they are where `builder`
  /// is.
  Value getMatrixEntries(OpBuilder &builder, Location loc, Value handle);
  /// The candidates (`which` 0) or the slots (1) of the pairs of images of
  /// `handle`, as the last build left them (D277).
  Value getEdgesBuffer(OpBuilder &builder, Location loc, Value handle,
                       int64_t which);
  /// After a build of the matrix `structure` at `positions` in the cell
  /// `box`: the candidates of the pairs of images of each particle within
  /// `reach`, their offsets, and the candidates in their slots.
  void emitEdgeSlots(OpBuilder &builder, Location loc,
                     const Neighbors &structure, Value positions, Value box,
                     double reach);
  LogicalResult lowerBuildEdges(md_exec::BuildEdgesOp op);
  LogicalResult lowerTupleFor(md_exec::TupleForOp op);
  void lowerBuildIncidence(md_exec::BuildIncidenceOp op);
  void lowerRenumber(md_exec::RenumberOp op);
  /// Frees `buffer`, a buffer of the device, where the block of `op` ends.
  void freeDeviceAtEndOfBlock(Operation *op, Value buffer);

  /// The storage of the neighbor structure `structure`.
  LogicalResult getNeighbors(Operation *op, Value structure,
                             Neighbors &storage);

  /// Builds `structure` at the configuration `positions`: the lists of a
  /// structure of groups (emitGroupsBuild) or the rows of a matrix, then
  /// what every build records (finishBuild).
  LogicalResult emitBuild(OpBuilder &builder, Location loc,
                          const Neighbors &structure, Value positions,
                          Value box, double reach, double cellWidth);
  /// Stops the run, through the runtime, if `notNumbers` positions, as a
  /// build counted them, are not numbers (D107).
  void emitStopNotNumbers(OpBuilder &builder, Location loc, Value notNumbers);
  LogicalResult emitGroupsBuild(OpBuilder &builder, Location loc,
                                const Neighbors &structure, Value positions,
                                Value box, double reach);
  LogicalResult finishBuild(OpBuilder &builder, Location loc,
                            const Neighbors &structure, Value positions,
                            Value box);

  /// Launches a kernel with one thread for each of `count` items. `body`
  /// emits what the thread of an item does.
  void launchOver(OpBuilder &builder, Location loc, Value count,
                  function_ref<void(OpBuilder &, Value)> body);

  /// Launches a kernel with a group of threads for each of `count`
  /// particles, which share its row (RowLanes). With `order`, the order of
  /// the cells of a neighbor matrix, an item is a place, and a place that
  /// the build left empty (D107) is not valid.
  void launchRows(
      OpBuilder &builder, Location loc, Value count,
      function_ref<void(OpBuilder &, Value, const RowLanes &)> body,
      Value order = Value());


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
  /// The members on the device of each incidence structure on the device.
  DenseMap<Value, Value> members;
  /// Whether a loop over tuples evaluates each tuple once, adding to its
  /// members with atomics: by default, for a loop that is not over disjoint
  /// tuples and adds to its destinations. Its global sums are summed by
  /// a thread a particle over the tuples it takes (D103).
  bool evaluatesOnce(md_exec::TupleForOp op) {
    if (deterministic || !tuplesOnce || op.getDisjoint())
      return false;
    for (unsigned i = 0, e = op.getOuts().size(); i != e; ++i)
      if (op.overwrites(i))
        return false;
    return true;
  }
  DenseSet<Operation *> inRows;

  /// How a loop provides the contributions of its rows to its global sums
  /// when one kernel can also compute them: `produce` launches the kernel
  /// that stores them in the buffers of contributions, and `row` emits, in
  /// a thread of the kernel of the reduction, the contributions of row `i`.
  struct RowSums {
    function_ref<void(OpBuilder &)> produce;
    function_ref<SmallVector<Value>(OpBuilder &, Value)> row;
  };

  /// Adds up, or takes the maximum of, `values` over the threads of a block
  /// in one pass (D150): each warp by a butterfly of shuffles, then the
  /// warps in their order through `shared`. Every thread gets the totals.
  SmallVector<Value> reduceInBlock(OpBuilder &builder, Location loc,
                                   ValueRange values, Value thread,
                                   Value shared, bool isSum);
  /// Launches `blocks` blocks of threads with shared memory for `numbers`
  /// numbers of each warp; `body` takes the block, the thread, and the
  /// shared memory.
  void launchSharedBlocks(
      OpBuilder &builder, Location loc, Value blocks, int64_t numbers,
      function_ref<void(OpBuilder &, Value, Value, Value)> body);

  /// `rows`, if given, lets a reduction of one part compute the
  /// contributions of the rows itself rather than load them: a set of few
  /// tuples then takes one kernel for its sums instead of two (#20).
  SmallVector<Value> emitReductions(OpBuilder &builder, Location loc,
                                    ArrayRef<Value> contributions,
                                    ArrayRef<Value> partials, Value size,
                                    bool isSum,
                                    std::optional<RowSums> rows = {});

  /// The integration runs (D110), by their loop after, where they are
  /// lowered.
  DenseMap<Operation *, kernels::IntegrationRun> integrations;
  /// Records `run`, a run of loops over disjoint tuples, as an integration
  /// run if a loop over particles before it writes what it reads, a loop
  /// over particles after it reads what they wrote, and nothing else does
  /// between them. Returns whether it did.
  bool recordIntegration(ArrayRef<Operation *> run);
  /// Lowers an integration run to one kernel (emitIntegrationThread) and
  /// the reductions of its loop after.
  LogicalResult lowerIntegration(const kernels::IntegrationRun &given);

  /// Finds the runs of loops over pairs and tuples in `function` that one
  /// kernel can do (lowerRows), and records them in `rows`.
  void findRows(func::FuncOp function);
  /// The runs of loops over tuples that evaluate each tuple once and have no
  /// global sums, by their last loop: one kernel launches the tuples of all
  /// of them (lowerOnceRun), as the bonded terms of a step, or the forces of
  /// a term over the centers of groups beside them (#20).
  DenseMap<Operation *, SmallVector<md_exec::TupleForOp>> onceRuns;
  DenseSet<Operation *> inOnceRuns;
  /// Finds the runs of `onceRuns` in `function`.
  void findOnceRuns(func::FuncOp function);

  /// A loop with global sums over tuples, each evaluated once, whose sums
  /// reach nothing but a later loop over tuples evaluated once, through
  /// arithmetic on the host: the centers of a term over the centers of
  /// groups and its forces (D139). One block of the kernel of the run of
  /// the later loop computes the sums of a reduction of one part, as the
  /// kernel of that reduction would, and then evaluates the later loop,
  /// so that the sums take no kernel of their own (#66).
  struct CenterFusion {
    md_exec::TupleForOp sums, forces;
    /// The arithmetic from the sums to the later loop, in its order.
    SmallVector<Operation *> chain;
    /// The values that stand for the results of `sums` once it is lowered.
    SmallVector<Value> placeholders;
    /// A buffer of the device for the totals of several parts.
    Value totals;
  };
  SmallVector<std::unique_ptr<CenterFusion>> fusions;
  DenseMap<Operation *, CenterFusion *> fusionOfSums, fusionOfForces;
  /// Finds the loops of `fusions` in `function`.
  void findCenterFusions(func::FuncOp function);
  /// Lowers the sums of `fusion` where the loop is: the kernels of a
  /// reduction of several parts, launched only when there are several,
  /// which leave the totals in a buffer of the fusion.
  void lowerFusedSums(CenterFusion &fusion, Value tupleMembers, Value rows,
                      Value tuples, Value box, Value inverse);
  /// Lowers the loops of `run` to one kernel: a thread takes the tuple of
  /// its number in the tuples of the loops, one loop after the other.
  LogicalResult lowerOnceRun(ArrayRef<md_exec::TupleForOp> run);
  /// Lowers the loops over pairs and tuples of `run` to one kernel, in
  /// which the group of threads of a particle does each loop in turn, and
  /// their global sums to one reduction.
  LogicalResult lowerRows(ArrayRef<Operation *> run);

  /// Gathers the positions and the fields of `ins` of a loop over pairs
  /// into the order of the cells of `structure`, in buffers of their own
  /// that are freed after `after` (D86).
  PairLayout gatherInOrder(OpBuilder &builder, Location loc,
                           const Neighbors &structure, Value positions,
                           ValueRange ins, Operation *after);

  /// Stores the contributions of a particle and returns, for each global
  /// sum of a loop, its result.
  /// Lowers a loop over tuples with a flag in `reduce`
  /// (D278).
  LogicalResult lowerTupleForWithFlags(md_exec::TupleForOp op);

  LogicalResult finishSums(Operation *op, OpBuilder &builder,
                           ValueRange reduce, ValueRange scratch, Value size,
                           std::optional<RowSums> rows = {});

  /// A place for `count` numbers of the type `element`.
  Cell getCell(Type element, int64_t count, Location loc);

  /// Flag number `number` of the function. It is not set where a kernel
  /// begins that sets it: `replaceWithFlag` sees to that.
  Flag getFlag(unsigned number, Location loc);

  /// Replaces `result`, the reduction of a flag that `start` begins, with
  /// `start` or whether the flag is set, which it leaves not set: the copy of the flag starts at `builder`, and the
  /// host waits for it only before the first op of the block that uses
  /// `result`, so that it issues the work between before it waits (D113).
  void replaceWithFlag(OpBuilder &builder, Location loc, const Flag &flag,
                       Value start, Value result);
  /// Replaces `results` as `replaceWithFlag` does, for flags that one
  /// kernel sets: one event after the kernel, waited for before the first
  /// op of the block that uses one of them, after which the others are read
  /// as they are (`mdrtFlagTake`, D278).
  void replaceWithFlags(OpBuilder &builder, Location loc, ArrayRef<Flag> flags,
                        ArrayRef<Value> starts, ArrayRef<Value> results);

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
  /// Adds the template that builds groups, or with `tilted` that of a
  /// triclinic cell.
  LogicalResult addGroupsTemplates(bool tilted);
  func::FuncOp getOrDeclare(StringRef name, FunctionType type);
  /// Adds the template of particle mesh Ewald for these types, with
  /// `tilted` the kernels of a triclinic cell as well, and with
  /// `potential` the kernel of the potential at the particles, and with
  /// `listed` the kernels of a list of the particles, and with `tiles` E
  /// the binning and the spreading into tiles of bins of E points
  /// (D[pme-tiles]).
  LogicalResult addPMETemplates(Type position, Type charge, Type force,
                                int64_t order, bool tilted, bool potential,
                                bool listed, int64_t tiles = 0);
  /// Adds the kernels of the prolate kernel (D266) for these
  /// types and points of the window, and with `listed` those of a list of
  /// the particles, and with `tiles` E the spreading into tiles of bins
  /// of E points (D[pme-tiles]).
  LogicalResult addProlateTemplates(Type position, Type charge, Type force,
                                    int64_t order, bool listed,
                                    int64_t tiles = 0);
  LogicalResult lowerReciprocal(md_exec::ReciprocalOp op);

  ModuleOp module;
  MLIRContext *context;
  int64_t blockSize;
  /// The threads that share the rows of a particle (launchRows).
  int64_t rowLanes;
  /// Whether runs of loops over rows become one kernel (findRows).
  bool fuseRows;
  bool fuseIntegration;
  /// Whether sums are added in an order that the threads do not decide
  /// (D84).
  bool deterministic;
  /// Whether the module has a triclinic cell.
  bool triclinic = false;
  /// The number of elements of a cell lowered to a vector.
  int64_t getCellSize() const { return triclinic ? 6 : 3; }
  bool tuplesOnce;
  bool fuseCenters;

  /// The function that is being lowered.
  func::FuncOp current;
  /// What the kernels that are being made compute, for their names: the op
  /// and the line of the input that it comes from (nameKernel).
  std::string purpose = "kernel";
  int64_t numKernels = 0;
  /// The flags of the module so far: each place that a function takes a
  /// flag from has a number of its own, by which the runtime gives it the
  /// same memory at every call (mdrtFlagMemory, D118).
  int64_t numFlags = 0;
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
  /// The loops over pairs of groups that follow a refresh of their dual
  /// list at the same positions and in the same cell, with nothing between
  /// that may write the positions (findSharedFrames, D268).
  llvm::DenseSet<Operation *> sharesPrunedFrames;
  void findSharedFrames(func::FuncOp function, md_exec::BufferAliases &aliases);
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
    Type wide = builder.getI64Type();
    func::FuncOp memory = getOrDeclare(
        flagMemoryName,
        builder.getFunctionType({wide}, {getDeviceType({1}, narrow)}));
    memory->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    Value place = arith::ConstantOp::create(
        builder, loc, wide, builder.getI64IntegerAttr(numFlags++));
    flag.device = func::CallOp::create(builder, loc, memory, ValueRange{place})
                      .getResult(0);
    flags.push_back(flag);
  }
  return flags[number];
}

void Lowering::replaceWithFlag(OpBuilder &builder, Location loc,
                               const Flag &flag, Value start, Value result) {
  replaceWithFlags(builder, loc, {flag}, {start}, {result});
}

void Lowering::replaceWithFlags(OpBuilder &builder, Location loc,
                                ArrayRef<Flag> flags, ArrayRef<Value> starts,
                                ArrayRef<Value> results) {
  Type i64 = builder.getI64Type();
  Type i32 = builder.getI32Type();
  auto pointerType = LLVM::LLVMPointerType::get(context);
  auto getPointer = [&](OpBuilder &b, const Flag &flag) -> Value {
    Value address = arith::IndexCastOp::create(
        b, loc, i64,
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, flag.device));
    return LLVM::IntToPtrOp::create(b, loc, pointerType, address);
  };
  Value slot =
      func::CallOp::create(
          builder, loc,
          getOrDeclare(flagStartName,
                       builder.getFunctionType({pointerType}, {i64})),
          ValueRange{getPointer(builder, flags.front())})
          .getResult(0);
  // The first op of the block that uses a result, directly or within.
  Block *block = builder.getInsertionBlock();
  Operation *first = nullptr;
  for (Value result : results)
    for (Operation *user : result.getUsers()) {
      Operation *inBlock = block->findAncestorOpInBlock(*user);
      if (inBlock && (!first || inBlock->isBeforeInBlock(first)))
        first = inBlock;
    }
  OpBuilder wait(builder.getContext());
  if (first)
    wait.setInsertionPoint(first);
  else
    wait.setInsertionPoint(builder.getInsertionBlock(),
                           builder.getInsertionPoint());
  Value zero =
      arith::ConstantOp::create(wait, loc, i32, wait.getI32IntegerAttr(0));
  for (auto [index, flag] : llvm::enumerate(flags)) {
    // The first flag waits for the event after the kernel; the others,
    // which the same kernel sets (D278), are read
    // after that wait, without an event of their own.
    Value value =
        index == 0
            ? func::CallOp::create(
                  wait, loc,
                  getOrDeclare(flagFinishName,
                               wait.getFunctionType({i64}, {i32})),
                  ValueRange{slot})
                  .getResult(0)
            : func::CallOp::create(
                  wait, loc,
                  getOrDeclare(flagTakeName,
                               wait.getFunctionType({pointerType}, {i32})),
                  ValueRange{getPointer(wait, flag)})
                  .getResult(0);
    Value isSet = arith::CmpIOp::create(wait, loc, arith::CmpIPredicate::ne,
                                        value, zero);
    // The runtime clears the flag as it reads it (D118); a copy of a zero
    // from the host waited for the reciprocal sum issued before it.
    Value result = results[index];
    result.replaceAllUsesWith(
        arith::OrIOp::create(wait, loc, starts[index], isSet));
  }
}

static Value shuffleXor(OpBuilder &builder, Location loc, Value value,
                        int64_t offset);

/// The numbers of `types`: one for a number, and the elements of a vector.
static int64_t countKernelNumbers(TypeRange types) {
  int64_t count = 0;
  for (Type type : types)
    count += isa<VectorType>(type) ? cast<VectorType>(type).getNumElements()
                                   : 1;
  return count;
}

SmallVector<Value> Lowering::reduceInBlock(OpBuilder &b, Location loc,
                                           ValueRange values, Value thread,
                                           Value shared, bool isSum) {
  int64_t warps = blockSize / 32;
  auto combine = [&](OpBuilder &b, Value lhs, Value rhs) -> Value {
    if (isSum)
      return arith::AddFOp::create(b, loc, lhs, rhs);
    Value larger =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, lhs, rhs);
    return arith::SelectOp::create(b, loc, larger, lhs, rhs);
  };
  SmallVector<Value> numbers;
  for (Value value : values) {
    auto vector = dyn_cast<VectorType>(value.getType());
    if (!vector) {
      numbers.push_back(value);
      continue;
    }
    for (int64_t i = 0, e = vector.getNumElements(); i != e; ++i)
      numbers.push_back(vector::ExtractOp::create(b, loc, value, i));
  }
  for (Value &number : numbers)
    for (int64_t offset = 16; offset >= 1; offset /= 2)
      number = combine(b, number, shuffleXor(b, loc, number, offset));
  Value width = createIndex(b, loc, 32);
  Value lane = arith::RemUIOp::create(b, loc, thread, width);
  Value warp = arith::DivUIOp::create(b, loc, thread, width);
  Value count = createIndex(b, loc, numbers.size());
  Value leads = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane,
                                      createIndex(b, loc, 0));
  Type wide = b.getF64Type();
  scf::IfOp::create(b, loc, leads, [&](OpBuilder &then, Location) {
    Value base = arith::MulIOp::create(then, loc, warp, count);
    for (auto [k, number] : llvm::enumerate(numbers)) {
      Value place = arith::AddIOp::create(then, loc, base,
                                          createIndex(then, loc, k));
      Value stored = number;
      if (number.getType() != wide)
        stored = arith::ExtFOp::create(then, loc, wide, number);
      memref::StoreOp::create(then, loc, stored, shared, ValueRange{place});
    }
    scf::YieldOp::create(then, loc);
  });
  gpu::BarrierOp::create(b, loc);
  SmallVector<Value> totals;
  for (auto [k, number] : llvm::enumerate(numbers)) {
    Value total;
    for (int64_t w = 0; w != warps; ++w) {
      Value part = memref::LoadOp::create(
          b, loc, shared,
          ValueRange{createIndex(b, loc, w * numbers.size() + k)});
      if (number.getType() != wide)
        part = arith::TruncFOp::create(b, loc, number.getType(), part);
      total = w == 0 ? part : combine(b, total, part);
    }
    totals.push_back(total);
  }
  SmallVector<Value> results;
  size_t next = 0;
  for (Value value : values) {
    auto vector = dyn_cast<VectorType>(value.getType());
    if (!vector) {
      results.push_back(totals[next++]);
      continue;
    }
    ArrayRef<Value> elements(totals.begin() + next,
                             totals.begin() + next + vector.getNumElements());
    next += vector.getNumElements();
    results.push_back(
        vector::FromElementsOp::create(b, loc, vector, elements));
  }
  return results;
}

void Lowering::launchSharedBlocks(
    OpBuilder &at, Location loc, Value blocks, int64_t numbers,
    function_ref<void(OpBuilder &, Value, Value, Value)> body) {
  int64_t warps = blockSize / 32;
  Value one = createIndex(at, loc, 1);
  Value threads = createIndex(at, loc, blockSize);
  auto workgroup = gpu::AddressSpaceAttr::get(
      context, gpu::GPUDialect::getWorkgroupAddressSpace());
  auto shared = MemRefType::get({warps * numbers}, at.getF64Type(),
                                MemRefLayoutAttrInterface(), workgroup);
  auto launch = gpu::LaunchOp::create(
      at, loc, blocks, one, one, threads, one, one,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/{}, /*workgroupAttributions=*/TypeRange{shared});
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  body(kernel, launch.getBlockIds().x, launch.getThreadIds().x,
       launch.getWorkgroupAttributionBBArgs().front());
  gpu::TerminatorOp::create(kernel, loc);
  bringIn(launch);
}

SmallVector<Value>
Lowering::emitReductions(OpBuilder &builder, Location loc,
                         ArrayRef<Value> contributions,
                         ArrayRef<Value> partials, Value size, bool isSum,
                         std::optional<RowSums> rows) {
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

  // A block of threads for each part of the particles, and one block for
  // the results of the parts. A thread adds up the particles that it takes,
  // every so manyth, and the block adds up its threads for all the sums of
  // the kernel in one pass: each warp by a butterfly of shuffles, its first
  // lane writes the totals of the warp to shared memory, and after one
  // barrier every thread adds the warps in their order (D150). The order
  // of a sum, and the sum, depend only on the number of particles. A block
  // takes at least four particles per thread, and there are no more parts
  // than threads in a block.
  auto countNumbers = [](TypeRange types) { return countKernelNumbers(types); };
  auto reduceTogether = [&](OpBuilder &b, ValueRange values, Value thread,
                            Value shared) -> SmallVector<Value> {
    return reduceInBlock(b, loc, values, thread, shared, isSum);
  };
  auto launchBlocks = [&](OpBuilder &at, Value blocks, int64_t numbers,
                          function_ref<void(OpBuilder &, Value, Value, Value)>
                              body) {
    launchSharedBlocks(at, loc, blocks, numbers, body);
  };
  // The sums go to the kernels in runs whose buffers fit a budget of
  // parameters: a kernel passes each buffer as its descriptor, two
  // pointers, an offset, and a size and a stride for each dimension, and
  // PTX before ISA 8.1 allows 4352 bytes of parameters. A loop with many
  // sums (the centers of a dihedral, its energy and its output together)
  // takes more than one pair of kernels; each sum keeps its tree.
  auto getDescriptorBytes = [](Value buffer) -> int64_t {
    return 8 * (3 + 2 * cast<MemRefType>(buffer.getType()).getRank());
  };
  const int64_t budget = 3072;
  SmallVector<std::pair<size_t, size_t>> runs;
  for (size_t first = 0, e = contributions.size(); first != e;) {
    size_t last = first;
    int64_t bytes = 0;
    while (last != e) {
      int64_t more = getDescriptorBytes(contributions[last]) +
                     getDescriptorBytes(partials[last]);
      if (last != first && bytes + more > budget)
        break;
      bytes += more;
      ++last;
    }
    runs.push_back({first, last});
    first = last;
  }

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

  // Stores the totals of the sums `begin` to `end` where the results
  // arrive.
  auto storeResults = [&](OpBuilder &then, size_t begin, size_t end,
                          ArrayRef<Value> totals) {
    for (size_t index = begin; index != end; ++index) {
      const Place &place = places[index];
      Value result = totals[index - begin];
      for (int64_t c = 0; c != place.count; ++c) {
        Value number =
            isa<VectorType>(result.getType())
                ? Value(vector::ExtractOp::create(then, loc, result, c))
                : result;
        memref::StoreOp::create(
            then, loc, number, place.cell.device,
            ValueRange{createIndex(then, loc, place.offset + c)});
      }
    }
  };

  Value parts = arith::MinSIOp::create(
      builder, loc, createIndex(builder, loc, blockSize),
      createGroups(builder, loc, size, 4 * blockSize));
  // With one part, as for a set of few tuples, its block gives the results
  // and the second kernel is not launched.
  Value onePart = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                        parts, createIndex(builder, loc, 1));
  // A reduction of one part may compute the contributions of the rows in
  // its own block, in the order the buffers would hold them, so that the
  // kernel that stores them is not launched. With several runs of sums each
  // run would evaluate the rows again, and their stores to the
  // destinations of the loop with them, so only one run computes them.
  bool fused = rows && runs.size() == 1;
  if (fused) {
    Value several = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ne, parts,
        createIndex(builder, loc, 1));
    scf::IfOp::create(builder, loc, several, [&](OpBuilder &then, Location) {
      rows->produce(then);
      scf::YieldOp::create(then, loc);
    });
  } else if (rows) {
    rows->produce(builder);
  }
  for (const auto &run : runs) {
    size_t begin = run.first, end = run.second;
    ArrayRef<Value> those = contributions.slice(begin, end - begin);
    ArrayRef<Value> into = partials.slice(begin, end - begin);
    int64_t numbers =
        countNumbers(ArrayRef<Type>(types).slice(begin, end - begin));
    launchBlocks(builder, parts, numbers, [&](OpBuilder &b, Value block,
                                              Value thread, Value shared) {
      Value threads = createIndex(b, loc, blockSize);
      Value first = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, block, threads), thread);
      Value stride = arith::MulIOp::create(b, loc, parts, threads);
      SmallVector<Value> zeros;
      for (Type type : ArrayRef<Type>(types).slice(begin, end - begin))
        zeros.push_back(createZero(b, loc, type));
      auto loop = scf::ForOp::create(
          b, loc, first, size, stride, zeros,
          [&](OpBuilder &inner, Location, Value i, ValueRange sums) {
            SmallVector<Value> values;
            if (fused) {
              SmallVector<Type> wanted;
              for (Value buffer : those)
                wanted.push_back(
                    md_exec::getKernelValueType(buffer.getType()));
              auto choose = scf::IfOp::create(
                  inner, loc, wanted, onePart, /*withElseRegion=*/true);
              OpBuilder computed = choose.getThenBodyBuilder();
              scf::YieldOp::create(computed, loc, rows->row(computed, i));
              OpBuilder loaded = choose.getElseBodyBuilder();
              SmallVector<Value> fromBuffers;
              for (Value buffer : those)
                fromBuffers.push_back(loadElement(loaded, loc, buffer, i));
              scf::YieldOp::create(loaded, loc, fromBuffers);
              values.append(choose.getResults().begin(),
                            choose.getResults().end());
            } else {
              for (Value buffer : those)
                values.push_back(loadElement(inner, loc, buffer, i));
            }
            SmallVector<Value> next;
            for (auto [value, sum] : llvm::zip(values, sums))
              next.push_back(combine(inner, sum, value));
            scf::YieldOp::create(inner, loc, next);
          });
      SmallVector<Value> totals =
          reduceTogether(b, loop.getResults(), thread, shared);
      Value leader = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                           thread, createIndex(b, loc, 0));
      scf::IfOp::create(b, loc, leader, [&](OpBuilder &then, Location) {
        for (auto [total, partial] : llvm::zip(totals, into))
          storeElement(then, loc, total, partial, block);
        scf::IfOp::create(then, loc, onePart, [&](OpBuilder &only, Location) {
          storeResults(only, begin, end, totals);
          scf::YieldOp::create(only, loc);
        });
        scf::YieldOp::create(then, loc);
      });
    });
  }

  // One block for the results of the parts, for each run of sums.
  Value severalParts =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt, parts,
                            createIndex(builder, loc, 1));
  auto several = scf::IfOp::create(builder, loc, severalParts);
  OpBuilder second = several.getThenBodyBuilder();
  for (const auto &run : runs) {
    size_t begin = run.first, end = run.second;
    int64_t numbers =
        countNumbers(ArrayRef<Type>(types).slice(begin, end - begin));
    launchBlocks(second, createIndex(second, loc, 1), numbers,
                 [&](OpBuilder &b, Value, Value thread, Value shared) {
      Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                           thread, parts);
      Value slot = arith::SelectOp::create(b, loc, inside, thread,
                                           createIndex(b, loc, 0));
      SmallVector<Value> values;
      for (size_t index = begin; index != end; ++index) {
        Value value = loadElement(b, loc, partials[index], slot);
        Value nothing = createZero(b, loc, types[index]);
        values.push_back(
            arith::SelectOp::create(b, loc, inside, value, nothing));
      }
      SmallVector<Value> totals = reduceTogether(b, values, thread, shared);
      Value leader = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                           thread, createIndex(b, loc, 0));
      scf::IfOp::create(b, loc, leader, [&](OpBuilder &then, Location) {
        storeResults(then, begin, end, totals);
        scf::YieldOp::create(then, loc);
      });
    });
  }

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

/// Returns true if `loop` tells in a value of `reduce` whether its kernel
/// yields true for some tuple, which takes a flag (D278).
static bool hasFlags(md_exec::TupleForOp loop) {
  return llvm::any_of(loop.getReduce(), [](Value value) {
    return value.getType().isInteger(1);
  });
}

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
                                   Value size, std::optional<RowSums> rows) {
  Location loc = op->getLoc();
  SmallVector<Value> contributions, partials;
  for (unsigned i = 0, e = reduce.size(); i != e; ++i) {
    contributions.push_back(scratch[2 * i]);
    partials.push_back(scratch[2 * i + 1]);
  }
  SmallVector<Value> sums = emitReductions(builder, loc, contributions,
                                           partials, size, /*isSum=*/true,
                                           rows);
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

void Lowering::findCenterFusions(func::FuncOp function) {
  if (!fuseCenters)
    return;
  function.walk([&](md_exec::TupleForOp sums) {
    if (sums.getReduce().empty() || hasFlags(sums) ||
        !sums.getOuts().empty() || !evaluatesOnce(sums) ||
        inRows.contains(sums) ||
        sums->hasAttr(md_exec::kSideAttrName))
      return;
    // The sums are numbers in f64, and the kernels of several parts take
    // all their buffers within the budget of parameters of emitReductions.
    int64_t bytes = 0;
    for (Value buffer : sums.getScratch()) {
      if (!md_exec::getKernelValueType(buffer.getType()).isF64())
        return;
      bytes += 8 * (3 + 2 * cast<MemRefType>(buffer.getType()).getRank());
    }
    if (bytes > 3072)
      return;
    Block *block = sums->getBlock();
    // The arithmetic on the host that the sums reach, and the one loop
    // that it reaches, inside whose kernel it is used.
    llvm::SetVector<Operation *> chain;
    md_exec::TupleForOp forces;
    SmallVector<Value> pending(sums->getResults());
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        Operation *top = block->findAncestorOpInBlock(*user);
        if (!top)
          return;
        if (top != user) {
          auto loop = dyn_cast<md_exec::TupleForOp>(top);
          if (!loop || (forces && forces != loop))
            return;
          forces = loop;
          continue;
        }
        if (user->getNumRegions() != 0 || !isMemoryEffectFree(user) ||
            isa_and_nonnull<md_exec::MDExecDialect, md::MDDialect>(
                user->getDialect()) ||
            !llvm::all_of(user->getResultTypes(),
                          [](Type type) { return type.isIntOrIndexOrFloat(); }))
          return;
        if (chain.insert(user))
          pending.append(user->result_begin(), user->result_end());
      }
    }
    if (!forces || !sums->isBeforeInBlock(forces) ||
        !forces.getReduce().empty() || !evaluatesOnce(forces) ||
        inRows.contains(forces) || fusionOfForces.contains(forces) ||
        forces->hasAttr(md_exec::kSideAttrName) ||
        forces.getPositions() != sums.getPositions() ||
        forces.getCellMutable().get() != sums.getCellMutable().get())
      return;
    for (Operation *op : chain)
      if (!op->isBeforeInBlock(forces))
        return;
    // With one part the sums move to the kernel of the later loop: nothing
    // between them may write what the sums read. Their scratch is used
    // only by the kernels of several parts, where the loop is.
    DenseSet<Value> read;
    for (Value operand : sums->getOperands())
      if (isa<MemRefType>(operand.getType()) &&
          !llvm::is_contained(sums.getScratch(), operand))
        read.insert(operand);
    for (Operation *op = sums->getNextNode(); op != forces;
         op = op->getNextNode()) {
      if (chain.contains(op) || isMemoryEffectFree(op))
        continue;
      SmallVector<Value> writes;
      if (auto pair = dyn_cast<md_exec::PairForOp>(op)) {
        llvm::append_range(writes, pair.getOuts());
        llvm::append_range(writes, pair.getScratch());
      } else if (auto particle = dyn_cast<md_exec::ParticleForOp>(op)) {
        llvm::append_range(writes, particle.getOuts());
        llvm::append_range(writes, particle.getScratch());
      } else if (auto tuple = dyn_cast<md_exec::TupleForOp>(op)) {
        llvm::append_range(writes, tuple.getOuts());
        llvm::append_range(writes, tuple.getScratch());
      } else {
        // Any other op is taken to write every buffer it uses.
        bool touches = false;
        op->walk([&](Operation *inner) {
          for (Value operand : inner->getOperands())
            touches |= read.contains(operand);
        });
        if (touches)
          return;
        continue;
      }
      for (Value buffer : writes)
        if (read.contains(buffer))
          return;
    }
    auto fusion = std::make_unique<CenterFusion>();
    fusion->sums = sums;
    fusion->forces = forces;
    fusion->chain.assign(chain.begin(), chain.end());
    llvm::sort(fusion->chain, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    fusionOfSums[sums] = fusion.get();
    fusionOfForces[forces] = fusion.get();
    fusions.push_back(std::move(fusion));
  });
}

void Lowering::findOnceRuns(func::FuncOp function) {
  auto isOnceLoop = [&](Operation *op) {
    auto tuple = dyn_cast<md_exec::TupleForOp>(op);
    return tuple && !inRows.contains(op) && evaluatesOnce(tuple) &&
           tuple.getReduce().empty() &&
           !op->hasAttr(md_exec::kSideAttrName);
  };
  auto finish = [&](SmallVector<md_exec::TupleForOp> &run) {
    // A loop whose kernel computes the sums of a term over centers is a
    // run of its own as well (#66).
    if (run.size() >= 2 ||
        (run.size() == 1 && fusionOfForces.contains(run.front()))) {
      onceRuns[run.back()] = run;
      for (md_exec::TupleForOp loop : run)
        inOnceRuns.insert(loop);
    }
    run.clear();
  };
  function.walk([&](Block *block) {
    SmallVector<md_exec::TupleForOp> run;
    // What the loops of the run add to, and what they read besides.
    DenseSet<Value> written, read;
    for (Operation &op : *block) {
      if (!isOnceLoop(&op)) {
        // Constants and arithmetic on the host may lie between the loops of a
        // run, as the coordinate of a term over centers that its loop takes:
        // they touch no memory, and the kernel of the run is launched at its
        // last loop, after them.
        if (op.getNumRegions() == 0 && isMemoryEffectFree(&op) &&
            !isa_and_nonnull<md_exec::MDExecDialect, md::MDDialect>(
                op.getDialect()))
          continue;
        finish(run);
        written.clear();
        read.clear();
        continue;
      }
      auto loop = cast<md_exec::TupleForOp>(op);
      // A loop joins if it takes the same positions and cell, reads nothing
      // that another loop of the run adds to, and adds to nothing that one
      // reads: then the loops of the run may run at once. Loops that add to
      // the same destination do so with atomics, in any order, as they do
      // one after the other (D84).
      bool joins = !run.empty() &&
                   loop.getPositions() == run.front().getPositions() &&
                   loop.getCellMutable().get() ==
                       run.front().getCellMutable().get();
      ValueRange outs = loop.getOuts();
      if (joins)
        for (Value operand : op.getOperands())
          joins &= !(written.contains(operand) &&
                     !llvm::is_contained(outs, operand));
      for (Value out : outs)
        joins &= !read.contains(out);
      if (!joins)
        finish(run), written.clear(), read.clear();
      run.push_back(loop);
      for (Value operand : op.getOperands())
        if (!llvm::is_contained(outs, operand))
          read.insert(operand);
      written.insert(outs.begin(), outs.end());
    }
    finish(run);
  });
}

LogicalResult Lowering::lowerOnceRun(ArrayRef<md_exec::TupleForOp> run) {
  md_exec::TupleForOp first = run.front();
  Location loc = first.getLoc();
  setPurpose(run.back());
  OpBuilder builder(run.back());
  Value positions = first.getPositions();
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, first.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);
  // The loops whose block computes the sums of a term over centers (#66),
  // and the others.
  SmallVector<CenterFusion *> fused;
  SmallVector<md_exec::TupleForOp> others;
  for (md_exec::TupleForOp loop : run) {
    if (CenterFusion *fusion = fusionOfForces.lookup(loop))
      fused.push_back(fusion);
    else
      others.push_back(loop);
  }
  auto getMembers = [&](md_exec::TupleForOp loop) -> FailureOr<Value> {
    Value found = members.lookup(loop.getIncidence());
    if (!found)
      return loop->emitOpError()
             << "has no members on the device to evaluate each tuple once";
    return found;
  };
  // The tuples of the other loops, one after the other: loop k takes the
  // numbers from ends[k - 1] to ends[k].
  SmallVector<Value> tupleMembers, ends;
  for (md_exec::TupleForOp loop : others) {
    FailureOr<Value> found = getMembers(loop);
    if (failed(found))
      return failure();
    tupleMembers.push_back(*found);
    Value count = createSize(builder, loc, *found);
    ends.push_back(ends.empty()
                       ? count
                       : arith::AddIOp::create(builder, loc, ends.back(),
                                               count));
  }
  // What the thread of item `item` of the other loops does.
  auto emitItem = [&](OpBuilder &body, Value item) {
    std::function<void(OpBuilder &, size_t)> emitFrom = [&](OpBuilder &at,
                                                           size_t k) {
      auto emitLoop = [&](OpBuilder &b) {
        Value tuple = k == 0 ? item
                             : arith::SubIOp::create(b, loc, item,
                                                     ends[k - 1]);
        IRMapping local;
        emitTupleOnce(b, others[k], tupleMembers[k], tuple, box, inverse,
                      local);
      };
      if (k + 1 == others.size()) {
        emitLoop(at);
        return;
      }
      Value mine = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult,
                                         item, ends[k]);
      scf::IfOp::create(
          at, loc, mine,
          [&](OpBuilder &then, Location) {
            emitLoop(then);
            scf::YieldOp::create(then, loc);
          },
          [&](OpBuilder &otherwise, Location) {
            emitFrom(otherwise, k + 1);
            scf::YieldOp::create(otherwise, loc);
          });
    };
    emitFrom(body, 0);
  };
  if (fused.empty()) {
    launchOver(builder, loc, ends.back(), emitItem);
    return success();
  }

  // Block k computes the sums of fusion k as a reduction of one part
  // computes them (emitReductions), the same rows in the same order, or
  // takes the totals of several parts from the buffer of the fusion
  // (lowerFusedSums); then the arithmetic after them, and the
  // tuples of its loop of forces, every so manyth. The blocks after them
  // take the items of the other loops.
  struct Fused {
    Value sumsMembers, forcesMembers, rows, tuples, forcesTuples, onePart;
    SmallVector<Type> types;
  };
  SmallVector<Fused> setup;
  int64_t numbers = 1;
  Value size = createSize(builder, loc, positions);
  for (CenterFusion *fusion : fused) {
    Fused entry;
    FailureOr<Value> sumsMembers = getMembers(fusion->sums);
    FailureOr<Value> forcesMembers = getMembers(fusion->forces);
    if (failed(sumsMembers) || failed(forcesMembers))
      return failure();
    entry.sumsMembers = *sumsMembers;
    entry.forcesMembers = *forcesMembers;
    entry.tuples = createSize(builder, loc, entry.sumsMembers);
    entry.forcesTuples = createSize(builder, loc, entry.forcesMembers);
    entry.rows = arith::MinUIOp::create(builder, loc, size, entry.tuples);
    Value parts = arith::MinSIOp::create(
        builder, loc, createIndex(builder, loc, blockSize),
        createGroups(builder, loc, entry.rows, 4 * blockSize));
    entry.onePart =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, parts,
                              createIndex(builder, loc, 1));
    for (unsigned i = 0, e = fusion->sums.getReduce().size(); i != e; ++i)
      entry.types.push_back(md_exec::getKernelValueType(
          fusion->sums.getScratch()[2 * i].getType()));
    numbers = std::max(numbers, countKernelNumbers(entry.types));
    setup.push_back(entry);
  }
  Value numFused = createIndex(builder, loc, fused.size());
  Value blocks = numFused;
  if (!others.empty())
    blocks = arith::AddIOp::create(
        builder, loc, blocks,
        createGroups(builder, loc, ends.back(), blockSize));

  auto emitFused = [&](OpBuilder &b, size_t k, Value thread, Value shared) {
    CenterFusion &fusion = *fused[k];
    const Fused &entry = setup[k];
    md_exec::TupleForOp sums = fusion.sums;
    Value threads = createIndex(b, loc, blockSize);
    auto choose = scf::IfOp::create(b, loc, entry.types, entry.onePart,
                                    /*withElseRegion=*/true);
    {
      OpBuilder one = choose.getThenBodyBuilder();
      SmallVector<Value> zeros;
      for (Type type : entry.types)
        zeros.push_back(createZero(one, loc, type));
      auto loop = scf::ForOp::create(
          one, loc, thread, entry.rows, threads, zeros,
          [&](OpBuilder &inner, Location, Value i, ValueRange partial) {
            SmallVector<Value> values = emitTuplesOnceWithSums(
                inner, sums, entry.sumsMembers, i, entry.rows, entry.tuples,
                box, inverse);
            SmallVector<Value> next;
            for (auto [sum, value] : llvm::zip(partial, values))
              next.push_back(arith::AddFOp::create(inner, loc, sum, value));
            scf::YieldOp::create(inner, loc, next);
          });
      scf::YieldOp::create(one, loc,
                           reduceInBlock(one, loc, loop.getResults(), thread,
                                         shared, /*isSum=*/true));
    }
    {
      OpBuilder several = choose.getElseBodyBuilder();
      SmallVector<Value> totals;
      for (unsigned i = 0, e = entry.types.size(); i != e; ++i)
        totals.push_back(memref::LoadOp::create(
            several, loc, fusion.totals,
            ValueRange{createIndex(several, loc, i)}));
      scf::YieldOp::create(several, loc, totals);
    }
    // As finishSums adds them to the values the loop starts from.
    IRMapping map;
    for (auto [i, placeholder] : llvm::enumerate(fusion.placeholders))
      map.map(placeholder,
              arith::AddFOp::create(b, loc, sums.getReduce()[i],
                                    choose.getResult(i)));
    for (Operation *op : fusion.chain)
      b.clone(*op, map);
    IRMapping local;
    for (Operation *op : fusion.chain)
      for (Value result : op->getResults())
        local.map(result, map.lookup(result));
    scf::ForOp::create(b, loc, thread, entry.forcesTuples, threads,
                       ValueRange(),
                       [&](OpBuilder &inner, Location, Value tuple,
                           ValueRange) {
                         IRMapping inside = local;
                         emitTupleOnce(inner, fusion.forces,
                                       entry.forcesMembers, tuple, box,
                                       inverse, inside);
                         scf::YieldOp::create(inner, loc);
                       });
  };
  launchSharedBlocks(
      builder, loc, blocks, numbers,
      [&](OpBuilder &body, Value block, Value thread, Value shared) {
        std::function<void(OpBuilder &, size_t)> dispatch =
            [&](OpBuilder &at, size_t k) {
              if (k == fused.size()) {
                if (others.empty())
                  return;
                Value item = arith::AddIOp::create(
                    at, loc,
                    arith::MulIOp::create(
                        at, loc,
                        arith::SubIOp::create(at, loc, block, numFused),
                        createIndex(at, loc, blockSize)),
                    thread);
                Value inside = arith::CmpIOp::create(
                    at, loc, arith::CmpIPredicate::ult, item, ends.back());
                scf::IfOp::create(at, loc, inside,
                                  [&](OpBuilder &then, Location) {
                                    emitItem(then, item);
                                    scf::YieldOp::create(then, loc);
                                  });
                return;
              }
              Value mine =
                  arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq,
                                        block, createIndex(at, loc, k));
              scf::IfOp::create(
                  at, loc, mine,
                  [&](OpBuilder &then, Location) {
                    emitFused(then, k, thread, shared);
                    scf::YieldOp::create(then, loc);
                  },
                  [&](OpBuilder &otherwise, Location) {
                    dispatch(otherwise, k + 1);
                    scf::YieldOp::create(otherwise, loc);
                  });
            };
        dispatch(body, 0);
      });
  return success();
}

void Lowering::findSharedFrames(func::FuncOp function,
                                md_exec::BufferAliases &aliases) {
  function.walk([&](md_exec::PairForOp pair) {
    if (pair.getTraversal() != md_exec::Traversal::Unique)
      return;
    auto refresh =
        pair.getNeighbors().getDefiningOp<md_exec::RefreshNeighborsOp>();
    if (!refresh || !refresh.getPruneSkin() ||
        refresh->getBlock() != pair->getBlock() ||
        refresh.getPositions() != pair.getPositions() ||
        refresh.getCellMutable().get() != pair.getCellMutable().get())
      return;
    Value positions = pair.getPositions();
    for (Operation *crossed = refresh->getNextNode(); crossed != pair;
         crossed = crossed->getNextNode()) {
      // An op whose effects are not all declared, or that may write or
      // free the positions, ends the sharing.
      std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
          getEffectsRecursively(crossed);
      if (!effects)
        return;
      for (const MemoryEffects::EffectInstance &effect : *effects) {
        if (!isa<MemoryEffects::Write, MemoryEffects::Free>(
                effect.getEffect()))
          continue;
        Value value = effect.getValue();
        if (!value || aliases.mayAlias(value, positions))
          return;
      }
    }
    sharesPrunedFrames.insert(pair.getOperation());
  });
}

void Lowering::findRows(func::FuncOp function) {
  // A loop over tuples with a flag is lowered alone
  // (D278).
  auto isRowLoop = [&](Operation *op) {
    if (auto tuple = dyn_cast<md_exec::TupleForOp>(op))
      return !tuple.getDisjoint() && !evaluatesOnce(tuple) &&
             !hasFlags(tuple);
    // A loop that takes each pair once runs over groups, not rows (D89).
    auto pair = dyn_cast<md_exec::PairForOp>(op);
    return pair && pair.getTraversal() == md_exec::Traversal::Directed;
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
    DenseSet<Value> written, read, sums;
    for (Operation &op : *block) {
      if (!isRowLoop(&op)) {
        // Constants may lie between the loops of a run; they are not
        // loops, and are lowered where they are.
        if (isa<arith::ConstantOp>(op))
          continue;
        finish(run);
        written.clear();
        read.clear();
        sums.clear();
        continue;
      }
      // A loop joins the run if it takes the same positions, reads nothing
      // that a loop of the run writes and writes nothing that one reads,
      // so that the order of the loops within a thread is the only order
      // that matters, and has buffers for its global sums of its own:
      // storage may give loops that ran one after the other the same
      // buffers, which one kernel would write at once.
      ValueRange scratch = isa<md_exec::PairForOp>(op)
                               ? cast<md_exec::PairForOp>(op).getScratch()
                               : cast<md_exec::TupleForOp>(op).getScratch();
      bool joins = !run.empty() && getPositions(&op) == getPositions(run[0]);
      // The loops over pairs of a run share the order of one structure.
      if (auto pair = dyn_cast<md_exec::PairForOp>(op))
        for (Operation *member : run)
          if (auto other = dyn_cast<md_exec::PairForOp>(member))
            joins &= other.getNeighbors() == pair.getNeighbors();
      if (joins)
        for (Value operand : op.getOperands())
          if ((written.contains(operand) &&
               !llvm::is_contained(getOuts(&op), operand)) ||
              sums.contains(operand))
            joins = false;
      for (Value out : llvm::concat<Value>(getOuts(&op), scratch))
        joins &= !read.contains(out);
      if (!joins) {
        finish(run);
        written.clear();
        read.clear();
        sums.clear();
      }
      run.push_back(&op);
      for (Value out : getOuts(&op))
        written.insert(out);
      for (Value operand : op.getOperands())
        if (!llvm::is_contained(getOuts(&op), operand))
          read.insert(operand);
      for (Value buffer : scratch)
        sums.insert(buffer);
    }
    finish(run);
  });

  // Runs of loops over disjoint tuples: a thread evaluates the tuple of
  // each loop whose member at place 0 it is. A loop joins the run if it
  // reads nothing that a loop of the run writes and writes nothing that
  // one reads, in its destinations or its scratch (a loop that
  // overwrites writes every particle), and has
  // buffers for its global sums of its own. The sets of a disjoint union
  // give such runs (md-bypass-updates, D83). A run between a loop over
  // particles that it reads and one that reads it is an integration run,
  // one kernel with them (D110).
  auto finishDisjoint = [&](SmallVector<Operation *> &run) {
    if (fuseRows && fuseIntegration && !run.empty() &&
        recordIntegration(run))
      run.clear();
    else
      finish(run);
  };
  function.walk([&](Block *block) {
    SmallVector<Operation *> run;
    DenseSet<Value> written, read, sums;
    for (Operation &op : *block) {
      auto tuple = dyn_cast<md_exec::TupleForOp>(op);
      if (!tuple || !tuple.getDisjoint()) {
        if (isa<arith::ConstantOp>(op))
          continue;
        finishDisjoint(run);
        written.clear();
        read.clear();
        sums.clear();
        continue;
      }
      // The loops with flags, the tests of the constraints, are a run of
      // their own: one kernel, whose flags one event covers
      // (D278); no other loop joins them.
      bool joins = !run.empty() &&
                   hasFlags(tuple) ==
                       hasFlags(cast<md_exec::TupleForOp>(run.front()));
      for (Value operand : op.getOperands())
        joins &= !written.contains(operand) && !sums.contains(operand);
      for (Value out :
           llvm::concat<Value>(tuple.getOuts(), tuple.getScratch()))
        joins &= !read.contains(out);
      if (!joins) {
        finishDisjoint(run);
        written.clear();
        read.clear();
        sums.clear();
      }
      run.push_back(&op);
      for (Value out : tuple.getOuts())
        written.insert(out);
      for (Value operand : op.getOperands())
        if (!llvm::is_contained(tuple.getOuts(), operand))
          read.insert(operand);
      for (Value buffer : tuple.getScratch())
        sums.insert(buffer);
    }
    finishDisjoint(run);
  });
}

/// Returns true if `op` writes every element of `buffer` and reads none:
/// a loop that overwrites it, or the forces of the reciprocal sum.
static bool overwritesAll(Operation *op, Value buffer) {
  auto readsOnlyAs = [&](ValueRange outs) {
    unsigned uses = 0;
    for (Value operand : op->getOperands())
      uses += operand == buffer;
    return uses == unsigned(llvm::count(outs, buffer));
  };
  if (auto loop = dyn_cast<md_exec::ParticleForOp>(op))
    return llvm::is_contained(loop.getOuts(), buffer) &&
           readsOnlyAs(loop.getOuts());
  auto overwrites = [&](auto loop) {
    for (auto [index, out] : llvm::enumerate(loop.getOuts()))
      if (out == buffer && !loop.overwrites(index))
        return false;
    return llvm::is_contained(loop.getOuts(), buffer) &&
           readsOnlyAs(loop.getOuts());
  };
  if (auto loop = dyn_cast<md_exec::TupleForOp>(op))
    return overwrites(loop);
  if (auto loop = dyn_cast<md_exec::PairForOp>(op))
    return overwrites(loop);
  if (auto reciprocal = dyn_cast<md_exec::ReciprocalOp>(op))
    return reciprocal.getOut() == buffer &&
           readsOnlyAs(ValueRange(reciprocal.getOut()));
  return false;
}

/// Returns true if nothing reads `buffer` after `op` before an op writes
/// all of it: the ops after `op` in its block, and, in the body of an
/// scf.for that passes `buffer` on to the next iteration unchanged, the
/// ops before `op` too and the ops after the loop for its result.
static bool isDeadAfter(Value buffer, Operation *op) {
  Block *block = op->getBlock();
  auto touches = [&](Operation *other) {
    return llvm::any_of(buffer.getUsers(), [&](Operation *user) {
      return other->isAncestor(user);
    });
  };
  bool wrapped = false;
  for (Operation *next = op->getNextNode(); next && next != op;) {
    if (next->hasTrait<OpTrait::IsTerminator>()) {
      auto loop = dyn_cast<scf::ForOp>(block->getParentOp());
      auto argument = dyn_cast<BlockArgument>(buffer);
      if (wrapped || !loop || !argument || argument.getOwner() != block ||
          argument.getArgNumber() == 0)
        return false;
      unsigned index = argument.getArgNumber() - 1;
      if (next->getOperand(index) != buffer ||
          !isDeadAfter(loop.getResult(index), loop))
        return false;
      wrapped = true;
      next = &block->front();
      continue;
    }
    if (touches(next))
      return overwritesAll(next, buffer);
    next = next->getNextNode();
  }
  return false;
}

bool Lowering::recordIntegration(ArrayRef<Operation *> run) {
  // The loop before: the op before the run but constants, a loop over
  // particles with no reductions whose destinations the run reads.
  Operation *first = run.front();
  Operation *previous = first->getPrevNode();
  while (previous && isa<arith::ConstantOp>(previous))
    previous = previous->getPrevNode();
  // A loop of an other run (the loop after of the run before, in a
  // minimization) stays there.
  auto before = dyn_cast_or_null<md_exec::ParticleForOp>(previous);
  if (!before || inRows.contains(before) || !before.getReduce().empty() ||
      before.getOuts().empty())
    return false;
  DenseSet<Value> beforeOuts(before.getOuts().begin(),
                             before.getOuts().end());
  DenseSet<Value> loopOuts;
  SmallVector<md_exec::TupleForOp, 4> loops;
  Value positions;
  bool readsBefore = false;
  for (Operation *op : run) {
    auto loop = cast<md_exec::TupleForOp>(op);
    // The loops take the particles of one set, have no reductions but a
    // flag each at most, the test of the constraints that the member at
    // place 0 sets (D278), and write none of what the
    // loop before wrote.
    if (loop.getOuts().empty() || loop.getReduce().size() > 1 ||
        (loop.getReduce().size() == 1 &&
         !loop.getReduce().front().getType().isInteger(1)))
      return false;
    if (!positions)
      positions = loop.getPositions();
    else if (loop.getPositions() != positions)
      return false;
    for (Value out : loop.getOuts())
      if (beforeOuts.contains(out))
        return false;
    for (Value operand : loop.getOperands())
      readsBefore |= beforeOuts.contains(operand);
    loops.push_back(loop);
    loopOuts.insert(loop.getOuts().begin(), loop.getOuts().end());
  }
  if (!readsBefore)
    return false;
  // The loop after: the first op after the run that touches what the loop
  // before or the run wrote, a loop over particles that reads it; the ops
  // between touch none of it.
  Operation *next = run.back()->getNextNode();
  md_exec::ParticleForOp after;
  while (next) {
    bool touches = llvm::any_of(next->getOperands(), [&](Value operand) {
      return beforeOuts.contains(operand) || loopOuts.contains(operand);
    });
    if (touches) {
      after = dyn_cast<md_exec::ParticleForOp>(next);
      break;
    }
    if (next->hasTrait<OpTrait::IsTerminator>() || next->getNumRegions() != 0)
      return false;
    next = next->getNextNode();
  }
  if (!after || inRows.contains(after))
    return false;
  // The loop after reads what the loop before wrote and what the loops
  // over tuples wrote, and takes the particles of the loop before.
  bool readsLoops = false;
  for (Value in : after.getIns())
    readsLoops |= loopOuts.contains(in);
  if (!readsLoops || after.getIns().empty() || before.getIns().empty())
    return false;
  kernels::IntegrationRun integration;
  // In the deterministic mode the joined kernel computes the displacements
  // of the tuples as the kernels of the loops alone do (#102).
  integration.storedPositions = deterministic;
  integration.before = before;
  integration.loops = loops;
  integration.after = after;
  for (md_exec::TupleForOp loop : loops) {
    SmallVector<bool, 2> keep;
    for (Value out : loop.getOuts())
      keep.push_back(!isDeadAfter(out, after));
    integration.keepOuts.push_back(keep);
  }
  integrations[after] = integration;
  inRows.insert(before);
  inRows.insert(run.begin(), run.end());
  inRows.insert(after);
  return true;
}

LogicalResult Lowering::lowerIntegration(const kernels::IntegrationRun &given) {
  kernels::IntegrationRun run = given;
  md_exec::ParticleForOp before = run.before, after = run.after;
  Location loc = after.getLoc();
  OpBuilder builder(after);
  setPurpose(before);
  purpose += "_integration" + std::to_string(run.loops.size());

  // The flag of each loop that tests its tuples (D278),
  // one of its own (D118).
  SmallVector<Flag> loopFlags;
  for (md_exec::TupleForOp loop : run.loops) {
    if (loop.getReduce().empty()) {
      run.flags.push_back(Value());
      continue;
    }
    loopFlags.push_back(getFlag(flags.size(), loc));
    run.flags.push_back(loopFlags.back().device);
  }

  // The cells of the loops over tuples, in the type of their kernels.
  SmallVector<Value> boxes, inverses;
  for (md_exec::TupleForOp loop : run.loops) {
    Type real = cast<MemRefType>(loop.getPositions().getType()).getElementType();
    Type computed =
        cast<VectorType>(loop.getKernel().front().getArgument(0).getType())
            .getElementType();
    Value box = convertReal(builder, loc, loop.getCellMutable().get(), real);
    boxes.push_back(convertReal(builder, loc, box, computed));
    inverses.push_back(
        convertReal(builder, loc, createInverse(builder, loc, box), computed));
  }

  // The reductions of the loop after, as lowerParticleFor has them.
  SmallVector<int> places;
  unsigned numSums = 0, numFlags = 0;
  for (Value value : after.getReduce())
    places.push_back(value.getType().isInteger(1) ? -int(++numFlags)
                                                  : int(numSums++));
  if (failed(checkScratch(after, numSums, after.getScratch().size())))
    return failure();
  SmallVector<Flag, 2> used;
  for (unsigned i = 0; i != numFlags; ++i)
    // A flag of its own for each loop: the runtime reads a flag after the
    // kernels issued before the read, not at its place in the stream, so a
    // later loop that shared it could set it before the read (D118).
    used.push_back(getFlag(flags.size(), loc));

  Value size = createSize(builder, loc, before.getIns().front());
  auto storeAfter = [&](OpBuilder &b, Value member,
                        ArrayRef<Value> contributions) {
          for (auto [index, value] : llvm::enumerate(contributions)) {
            int place = places[index];
            if (place >= 0) {
              storeElement(b, loc, value, after.getScratch()[2 * place],
                           member);
              continue;
            }
            Value device = used[-place - 1].device;
            scf::IfOp::create(b, loc, value, [&](OpBuilder &then, Location) {
              Value set = arith::ConstantOp::create(
                  then, loc, then.getI32Type(), then.getI32IntegerAttr(1));
              memref::StoreOp::create(then, loc, set, device,
                                      ValueRange{createIndex(then, loc, 0)});
              scf::YieldOp::create(then, loc);
            });
          }
  };
  // The particles in no tuple and the tuples within a warp, which list
  // the tuples across warps; then those. The count of the list is
  // allocated once, where the function begins, and the second kernel
  // leaves it zero.
  Type narrow = builder.getI32Type();
  Value acrossList = createDeviceBuffer(
      builder, loc, getDeviceType({ShapedType::kDynamic}, narrow),
      ValueRange{size});
  Value acrossCount;
  {
    Block &entry = current.getBody().front();
    OpBuilder atEntry(&entry, entry.begin());
    acrossCount = createDeviceBuffer(atEntry, loc, getDeviceType({1}, narrow),
                                     ValueRange());
    launchOver(atEntry, loc, createIndex(atEntry, loc, 1),
               [&](OpBuilder &body, Value) {
                 memref::StoreOp::create(
                     body, loc,
                     arith::ConstantOp::create(body, loc, narrow,
                                               body.getI32IntegerAttr(0)),
                     acrossCount, ValueRange{createIndex(body, loc, 0)});
               });
  }
  int64_t stride = kernels::getIntegrationStride(run);
  Value warps = arith::CeilDivUIOp::create(builder, loc, size,
                                           createIndex(builder, loc, stride));
  Value slots = arith::MulIOp::create(builder, loc, warps,
                                      createIndex(builder, loc, 32));
  launchOver(builder, loc, slots, [&](OpBuilder &body, Value thread) {
    kernels::emitIntegrationThread(body, run, boxes, inverses, thread, size,
                                   acrossList, acrossCount, /*across=*/false,
                                   storeAfter);
  });
  Value threads = createIndex(builder, loc, blockSize);
  launchOver(builder, loc, threads, [&](OpBuilder &body, Value thread) {
    kernels::emitIntegrationThread(body, run, boxes, inverses, thread,
                                   threads, acrossList, acrossCount,
                                   /*across=*/true, storeAfter);
  });
  {
    auto type = cast<MemRefType>(acrossList.getType());
    Value plain = memref::MemorySpaceCastOp::create(
        builder, loc, MemRefType::get(type.getShape(), type.getElementType()),
        acrossList);
    gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange(), plain);
  }

  SmallVector<Value> contributions, partials;
  for (unsigned i = 0; i != numSums; ++i) {
    contributions.push_back(after.getScratch()[2 * i]);
    partials.push_back(after.getScratch()[2 * i + 1]);
  }
  SmallVector<Value> sums = emitReductions(builder, loc, contributions,
                                           partials, size, /*isSum=*/true);
  for (auto [index, start] : llvm::enumerate(after.getReduce())) {
    int place = places[index];
    if (place < 0) {
      replaceWithFlag(builder, loc, used[-place - 1], start,
                      after.getResult(index));
      continue;
    }
    Value total = arith::AddFOp::create(builder, loc, start, sums[place]);
    after.getResult(index).replaceAllUsesWith(total);
  }
  // The flags of the loops over tuples, which both kernels set: one event
  // after them for all.
  if (!loopFlags.empty()) {
    SmallVector<Value> starts, results;
    for (md_exec::TupleForOp loop : run.loops)
      if (!loop.getReduce().empty()) {
        starts.push_back(loop.getReduce().front());
        results.push_back(loop.getResult(0));
      }
    replaceWithFlags(builder, loc, loopFlags, starts, results);
  }
  return success();
}

/// The particle at `place` of `order`, the order of the cells of a
/// neighbor matrix, or particle 0 at a place that the build left empty
/// because the positions that it found there are not numbers or are beyond
/// any cell (D107): every index stays inside the buffers of the particles.
static Value emitPlacedParticle(OpBuilder &builder, Location loc, Value order,
                                Value place) {
  Value at = memref::LoadOp::create(builder, loc, order, ValueRange{place});
  Value zero = arith::ConstantOp::create(
      builder, loc, at.getType(), builder.getIntegerAttr(at.getType(), 0));
  Value placed = arith::CmpIOp::create(builder, loc,
                                       arith::CmpIPredicate::sge, at, zero);
  Value safe = arith::SelectOp::create(builder, loc, placed, at, zero);
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    safe);
}

PairLayout Lowering::gatherInOrder(OpBuilder &builder, Location loc,
                                   const Neighbors &structure,
                                   Value positions, ValueRange ins,
                                   Operation *after) {
  PairLayout layout;
  layout.order = structure.order;
  SmallVector<Value> sources = {positions};
  sources.append(ins.begin(), ins.end());
  SmallVector<Value> targets;
  for (Value source : sources) {
    auto type = cast<MemRefType>(source.getType());
    SmallVector<Value> sizes = {structure.size};
    Value target = createDeviceBuffer(builder, loc, type, sizes);
    targets.push_back(target);
  }
  launchOver(builder, loc, structure.size, [&](OpBuilder &body, Value place) {
    // A place left empty by the build (D107) takes the values of particle
    // 0, which no loop reads there (launchRows).
    Value particle = emitPlacedParticle(body, loc, structure.order, place);
    for (auto [source, target] : llvm::zip(sources, targets))
      storeElement(body, loc, loadElement(body, loc, source, particle),
                   target, place);
  });
  // The buffers are freed once the loop has read them.
  OpBuilder release(after->getContext());
  release.setInsertionPointAfter(after);
  for (Value target : targets) {
    auto type = cast<MemRefType>(target.getType());
    Value plain = memref::MemorySpaceCastOp::create(
        release, loc, MemRefType::get(type.getShape(), type.getElementType()),
        target);
    gpu::DeallocOp::create(release, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange(), plain);
  }
  layout.positions = targets.front();
  layout.ins.assign(targets.begin() + 1, targets.end());
  return layout;
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
    /// The rows of the neighbor matrix of a loop over pairs.
    Value entries;
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
      loop.entries = getMatrixEntries(builder, loc, loop.structure.handle);
    } else {
      auto tuple = cast<md_exec::TupleForOp>(op);
      if (!hasFlags(tuple) &&
          failed(checkScratch(op, tuple.getReduce().size(),
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

  // The loops over pairs run in the order of the cells of their structure,
  // which is one for the run (findRows), and the loops over tuples take the
  // particle at the place of the thread (D86).
  DenseMap<Operation *, PairLayout> layouts;
  Value order;
  for (Loop &loop : loops)
    if (auto pair = dyn_cast<md_exec::PairForOp>(loop.op)) {
      layouts[loop.op] = gatherInOrder(builder, loc, loop.structure,
                                       pair.getPositions(), pair.getIns(),
                                       last);
      order = loop.structure.order;
    }

  // Loops over disjoint tuples: one thread for each particle, which
  // evaluates the tuples of which it is the member at place 0. The loops
  // may be over particles of different sets, and a thread past those of
  // a loop does nothing in it.
  auto isDisjoint = [](Operation *op) {
    auto tuple = dyn_cast<md_exec::TupleForOp>(op);
    return tuple && tuple.getDisjoint();
  };
  if (isDisjoint(run.front())) {
    SmallVector<Value> sizes;
    for (Loop &loop : loops) {
      sizes.push_back(createSize(
          builder, loc, cast<md_exec::TupleForOp>(loop.op).getPositions()));
      size = arith::MaxUIOp::create(builder, loc, size, sizes.back());
    }
    // A run of loops with flags (findRows): each loop a flag of its own,
    // set by a thread whose tuple yields true (D278).
    bool flagged = hasFlags(cast<md_exec::TupleForOp>(run.front()));
    SmallVector<SmallVector<Flag, 1>> loopFlags(loops.size());
    if (flagged)
      for (auto [loop, used] : llvm::zip(loops, loopFlags)) {
        auto tuple = cast<md_exec::TupleForOp>(loop.op);
        if (!llvm::all_of(tuple.getReduce(), [](Value value) {
              return value.getType().isInteger(1);
            }))
          return tuple->emitOpError()
                 << "joins a run of loops with flags with a global sum";
        for (unsigned i = 0, e = tuple.getReduce().size(); i != e; ++i)
          used.push_back(getFlag(flags.size(), loc));
      }
    launchOver(builder, loc, size, [&](OpBuilder &body, Value particle) {
      for (auto [loop, own, used] : llvm::zip(loops, sizes, loopFlags)) {
        auto tuple = cast<md_exec::TupleForOp>(loop.op);
        Value inside = arith::CmpIOp::create(
            body, loc, arith::CmpIPredicate::ult, particle, own);
        auto branch = scf::IfOp::create(body, loc, inside);
        OpBuilder then = branch.getThenBodyBuilder();
        IRMapping local;
        SmallVector<Value> contributions =
            emitTupleKernel(then, tuple, tuple.getIncidence(), loop.box,
                            loop.inverse, particle, local);
        for (auto [index, value] : llvm::enumerate(contributions)) {
          if (!flagged) {
            storeElement(then, loc, value, tuple.getScratch()[2 * index],
                         particle);
            continue;
          }
          Value device = used[index].device;
          scf::IfOp::create(then, loc, value, [&](OpBuilder &set, Location) {
            Value one = arith::ConstantOp::create(set, loc, set.getI32Type(),
                                                  set.getI32IntegerAttr(1));
            memref::StoreOp::create(set, loc, one, device,
                                    ValueRange{createIndex(set, loc, 0)});
            scf::YieldOp::create(set, loc);
          });
        }
      }
    });
    if (flagged) {
      // One event after the kernel for all the flags of the run.
      SmallVector<Flag> all;
      SmallVector<Value> starts, results;
      for (auto [loop, used] : llvm::zip(loops, loopFlags)) {
        auto tuple = cast<md_exec::TupleForOp>(loop.op);
        for (auto [index, start] : llvm::enumerate(tuple.getReduce())) {
          all.push_back(used[index]);
          starts.push_back(start);
          results.push_back(tuple->getResult(index));
        }
      }
      replaceWithFlags(builder, loc, all, starts, results);
      return success();
    }
    // The global sums of each loop, over its own particles.
    for (auto [loop, own] : llvm::zip(loops, sizes)) {
      auto tuple = cast<md_exec::TupleForOp>(loop.op);
      if (failed(finishSums(tuple, builder, tuple.getReduce(),
                            tuple.getScratch(), own)))
        return failure();
    }
    return success();
  }

  launchRows(builder, loc, size, [&](OpBuilder &body, Value particle,
                                     const RowLanes &sharing) {
    // What the loops add to each destination, in their order, stored once
    // at the end: loops of a run may add to one buffer, as the terms of a
    // potential add to the forces (md-exec-accumulate-destinations). The
    // additions are those that the loops would make one after the other.
    struct Destination {
      bool loads = false;
      SmallVector<Value, 4> additions;
    };
    llvm::MapVector<Value, Destination> destinations;
    Value place = particle;
    if (order)
      particle = emitPlacedParticle(body, loc, order, place);
    for (Loop &loop : loops) {
      IRMapping local;
      SmallVector<Value> contributions, totals;
      ValueRange scratch, outs;
      auto overwrites = [&](unsigned i) {
        if (auto pair = dyn_cast<md_exec::PairForOp>(loop.op))
          return pair.overwrites(i);
        return cast<md_exec::TupleForOp>(loop.op).overwrites(i);
      };
      if (auto pair = dyn_cast<md_exec::PairForOp>(loop.op)) {
        contributions = emitPairKernel(
            body, pair, loop.structure.counts, loop.entries,
            loop.box, loop.inverse, place, local, &sharing, &totals,
            &layouts[loop.op]);
        scratch = pair.getScratch();
        outs = pair.getOuts();
      } else {
        auto tuple = cast<md_exec::TupleForOp>(loop.op);
        contributions = emitTupleKernel(body, tuple, tuple.getIncidence(),
                                        loop.box, loop.inverse, particle,
                                        local, &sharing, &totals);
        scratch = tuple.getScratch();
        outs = tuple.getOuts();
      }
      for (auto [i, out] : llvm::enumerate(outs)) {
        Destination &destination = destinations[out];
        if (overwrites(i)) {
          destination.loads = false;
          destination.additions.clear();
        } else if (destination.additions.empty()) {
          destination.loads = true;
        }
        destination.additions.push_back(totals[i]);
      }
      storeContributions(body, loc, contributions, scratch, particle,
                         sharing);
    }
    if (destinations.empty())
      return;
    Value leader = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq,
                                         sharing.lane,
                                         createIndex(body, loc, 0));
    Value writes = arith::AndIOp::create(body, loc, leader, sharing.valid);
    scf::IfOp::create(body, loc, writes, [&](OpBuilder &then, Location) {
      for (auto &[buffer, destination] : destinations) {
        Value total;
        ArrayRef<Value> additions = destination.additions;
        if (destination.loads) {
          total = loadElement(then, loc, buffer, particle);
        } else {
          total = additions.front();
          additions = additions.drop_front();
        }
        for (Value addition : additions)
          total = arith::AddFOp::create(then, loc, total, addition);
        storeElement(then, loc, total, buffer, particle);
      }
      scf::YieldOp::create(then, loc);
    });
  }, order);

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
    // A flag of its own for each loop: the runtime reads a flag after the
    // kernels issued before the read, not at its place in the stream, so a
    // later loop that shared it could set it before the read (D118).
    used.push_back(getFlag(flags.size(), loc));

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
    if (place < 0) {
      replaceWithFlag(builder, loc, used[-place - 1], start,
                      op.getResult(index));
      continue;
    }
    Value total = arith::AddFOp::create(builder, loc, start, sums[place]);
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
    function_ref<void(OpBuilder &, Value, const RowLanes &)> body,
    Value order) {
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
  if (order) {
    // A place left empty by a build that found positions that are not
    // numbers or are beyond any cell holds no particle (D107): its group
    // does nothing, so that no access goes before the buffers of the
    // particles while the part that has failed runs to its end (#168).
    Value at = memref::LoadOp::create(kernel, loc, order, ValueRange{particle});
    Value placed = arith::CmpIOp::create(
        kernel, loc, arith::CmpIPredicate::sge, at,
        arith::ConstantOp::create(kernel, loc, at.getType(),
                                  kernel.getIntegerAttr(at.getType(), 0)));
    sharing.valid = arith::AndIOp::create(kernel, loc, sharing.valid, placed);
  }
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
  bool unique = op.getTraversal() == md_exec::Traversal::Unique;
  if (unique != structure.groups)
    return op.emitOpError()
           << (unique ? "takes each pair once, which only a structure of "
                        "groups gives"
                      : "takes each pair in both orders, which a structure "
                        "of groups does not give");
  if (unique)
    return lowerGroupPairFor(op, structure);

  Value positions = op.getPositions();
  Value size = createSize(builder, loc, positions);

  // The cell has become the vector of its edge lengths.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);

  // The loop runs in the order of the cells of the last build (D86).
  PairLayout layout = gatherInOrder(builder, loc, structure, positions,
                                    op.getIns(), op);
  Value entries = getMatrixEntries(builder, loc, structure.handle);
  launchRows(builder, loc, size, [&](OpBuilder &body, Value central,
                                     const RowLanes &sharing) {
    IRMapping local;
    SmallVector<Value> contributions = emitPairKernel(
        body, op, structure.counts, entries, box, inverse, central,
        local, &sharing, /*outTotals=*/nullptr, &layout);
    storeContributions(body, loc, contributions, op.getScratch(), central,
                       sharing);
  }, structure.order);
  return finishSums(op, builder, op.getReduce(), op.getScratch(), size);
}

LogicalResult Lowering::lowerGroupPairFor(md_exec::PairForOp op,
                                          const Neighbors &structure) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  // The atomic additions make the sums depend on the order of the threads;
  // the deterministic mode needs them in fixed point (D89), which is to
  // come.
  if (deterministic)
    return op.emitOpError()
           << "takes each pair once with atomic additions in floating "
              "point, which the deterministic mode does not allow";
  if (blockSize % 32 != 0)
    return op.emitOpError() << "runs a warp for each unit of work, which "
                               "needs blocks of whole warps, not "
                            << blockSize << " threads";

  Value positions = op.getPositions();
  Value size = createSize(builder, loc, positions);
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);

  // The loop adds to its destinations: one that it would overwrite is
  // cleared first.
  for (auto [i, out] : llvm::enumerate(op.getOuts())) {
    if (!op.overwrites(i))
      continue;
    Value destination = out;
    launchOver(builder, loc, size, [&](OpBuilder &body, Value particle) {
      Type element = getElementTypeOrSelf(destination.getType());
      auto type = cast<MemRefType>(destination.getType());
      Type value = type.getRank() == 1
                       ? element
                       : VectorType::get({type.getDimSize(1)}, element);
      storeElement(body, loc, createZero(body, loc, value), destination,
                   particle);
    });
  }

  // The buffers of the structure as the runtime holds them now.
  GroupBuffers buffers = getGroupsBuffers(builder, loc, structure.handle);

  // The positions and the fields that the kernel reads, in the order of
  // the places of the last build; an empty place takes those of the
  // particle 0, which no pair reads. A position is moved by whole cells to
  // the frame of its group (the shift of its place, ten bits an axis from
  // -512), so that the kernel takes no minimum image (D95). The positions
  // come as they are stored, unwrapped and in f64 in mixed precision: the
  // shift is added there, and the sum converted to the type of the kernel,
  // so that a position in the frame is as exact however far the particle
  // has gone.
  Type computed = op.getKernel().front().getArgument(0).getType();
  Value places = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      memref::LoadOp::create(builder, loc, structure.sizes,
                             ValueRange{createIndex(builder, loc, 0)}));
  // Where the loop follows a refresh of a dual list at the same positions
  // and in the same cell (sharesPrunedFrames), and computes in f32, a
  // pruning of that refresh has left the positions in the frames as the
  // loop takes them, to the bit: the loop takes them as they are then
  // (D268).
  Value sharedFrames, pruned;
  {
    auto stored = cast<MemRefType>(positions.getType());
    MemRefType framed =
        MemRefType::get(stored.getShape(), computed, stored.getLayout(),
                        stored.getMemorySpace());
    if (structure.dual && structure.prunedNow && computed.isF32() &&
        sharesPrunedFrames.contains(op.getOperation())) {
      Value frames = getGroupsPlaced(builder, loc, structure.handle);
      if (frames.getType() == framed) {
        sharedFrames = frames;
        pruned = memref::LoadOp::create(builder, loc, structure.prunedNow,
                                        ValueRange{});
      }
    }
  }
  // The kernel below takes it as a number.
  Value pruned32 =
      pruned ? Value(arith::ExtUIOp::create(builder, loc, builder.getI32Type(),
                                            pruned))
             : Value();
  SmallVector<Value> sources = {positions};
  llvm::append_range(sources, op.getIns());
  SmallVector<Value> targets;
  for (auto [index, source] : llvm::enumerate(sources)) {
    auto type = cast<MemRefType>(source.getType());
    if (index == 0)
      type = MemRefType::get(type.getShape(), computed, type.getLayout(),
                             type.getMemorySpace());
    targets.push_back(
        createDeviceBuffer(builder, loc, type, ValueRange{places}));
  }
  launchOver(builder, loc, places, [&](OpBuilder &body, Value place) {
    Value at = memref::LoadOp::create(body, loc, buffers.order,
                                      ValueRange{place});
    Value empty = arith::CmpIOp::create(
        body, loc, arith::CmpIPredicate::slt, at,
        arith::ConstantOp::create(body, loc, body.getI32Type(),
                                  body.getI32IntegerAttr(0)));
    Value particle = arith::SelectOp::create(
        body, loc, empty, createIndex(body, loc, 0),
        arith::IndexCastOp::create(body, loc, body.getIndexType(), at));
    Value packed = memref::LoadOp::create(body, loc, buffers.shifts,
                                          ValueRange{place});
    auto positionType = cast<MemRefType>(sources.front().getType());
    Type element = positionType.getElementType();
    SmallVector<Value> cells;
    for (int64_t k = 0; k != 3; ++k) {
      Value bits = arith::AndIOp::create(
          body, loc,
          arith::ShRUIOp::create(
              body, loc, packed,
              arith::ConstantOp::create(body, loc, body.getI32Type(),
                                        body.getI32IntegerAttr(10 * k))),
          arith::ConstantOp::create(body, loc, body.getI32Type(),
                                    body.getI32IntegerAttr(1023)));
      Value count = arith::SubIOp::create(
          body, loc, bits,
          arith::ConstantOp::create(body, loc, body.getI32Type(),
                                    body.getI32IntegerAttr(512)));
      cells.push_back(arith::SIToFPOp::create(body, loc, element, count));
    }
    Value shift = emitLatticeShift(
        body, loc,
        vector::FromElementsOp::create(body, loc,
                                       VectorType::get({3}, element), cells),
        convertReal(body, loc, box, element));
    for (auto [index, pair] : llvm::enumerate(llvm::zip(sources, targets))) {
      auto [source, target] = pair;
      auto store = [&, source = source, target = target](OpBuilder &at) {
        Value value = loadElement(at, loc, source, particle);
        if (index == 0)
          value = convertReal(
              at, loc, arith::AddFOp::create(at, loc, value, shift),
              computed);
        storeElement(at, loc, value, target, place);
      };
      if (index != 0 || !pruned32) {
        store(body);
        continue;
      }
      scf::IfOp::create(
          body, loc,
          arith::CmpIOp::create(
              body, loc, arith::CmpIPredicate::eq, pruned32,
              arith::ConstantOp::create(body, loc, body.getI32Type(),
                                        body.getI32IntegerAttr(0))),
          [&](OpBuilder &then, Location) {
            store(then);
            scf::YieldOp::create(then, loc);
          });
    }
  });
  PairLayout layout;
  layout.order = buffers.order;
  layout.positions = targets.front();
  if (sharedFrames) {
    Type type = targets.front().getType();
    auto choice = scf::IfOp::create(builder, loc, TypeRange{type}, pruned,
                                    /*withElseRegion=*/true);
    OpBuilder then = OpBuilder::atBlockEnd(choice.thenBlock());
    scf::YieldOp::create(then, loc, sharedFrames);
    OpBuilder otherwise = OpBuilder::atBlockEnd(choice.elseBlock());
    scf::YieldOp::create(otherwise, loc, targets.front());
    layout.positions = choice.getResult(0);
  }
  layout.ins.assign(targets.begin() + 1, targets.end());
  kernels::GroupLists lists{buffers.entries, buffers.masks, buffers.counts,
                            buffers.units, buffers.ordinals, buffers.order};
  // A dual list: the inner list, in the blocks of the outer one (D114),
  // through the units that hold entries of it (D268).
  Value activeUnits, activeCount;
  if (structure.dual) {
    InnerLists inner = getInnerLists(builder, loc, structure.handle);
    lists.entries = inner.entries;
    lists.masks = inner.masks;
    lists.blockCounts = inner.counts;
    PruneBuffers prune = getPruneBuffers(builder, loc, structure.handle);
    activeUnits = prune.activeUnits;
    activeCount = prune.activeCount;
  }

  // A warp for each unit of work, as many warps as particles at most: the
  // sums of the warps go to the scratch of the particles. A warp takes the
  // units w, w + warps, ...
  Value units = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      memref::LoadOp::create(builder, loc, structure.sizes,
                             ValueRange{createIndex(builder, loc, 2)}));
  Value one = createIndex(builder, loc, 1);
  Value warps = arith::MaxUIOp::create(
      builder, loc, arith::MinUIOp::create(builder, loc, units, size), one);
  Value warp32 = createIndex(builder, loc, 32);
  Value threads = arith::MulIOp::create(builder, loc, warps, warp32);
  Value block = createIndex(builder, loc, blockSize);
  Value grid = createGroups(builder, loc, threads, blockSize);
  auto launch =
      gpu::LaunchOp::create(builder, loc, grid, one, one, block, one, one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value thread = arith::AddIOp::create(
      kernel, loc,
      arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, block),
      launch.getThreadIds().x);
  Value warp = arith::DivUIOp::create(kernel, loc, thread, warp32);
  Value lane = arith::RemUIOp::create(kernel, loc, thread, warp32);
  Value valid = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                      warp, warps);
  // Of a dual list, the warps take the active units, as many as the last
  // pruning left on the device; the grid is that of all units.
  Value end = units;
  if (activeCount)
    end = arith::IndexCastOp::create(
        kernel, loc, kernel.getIndexType(),
        memref::LoadOp::create(kernel, loc, activeCount,
                               ValueRange{createIndex(kernel, loc, 0)}));
  Value first = arith::SelectOp::create(kernel, loc, valid, warp, end);

  Operation *yield = op.getKernel().front().getTerminator();
  unsigned numOuts = op.getOuts().size();
  SmallVector<Value> initial;
  for (unsigned i = numOuts, e = yield->getNumOperands(); i != e; ++i)
    initial.push_back(
        createZero(kernel, loc, yield->getOperand(i).getType()));
  auto loop = scf::ForOp::create(
      kernel, loc, first, end, warps, initial,
      [&](OpBuilder &body, Location, Value unit, ValueRange sums) {
        if (activeUnits)
          unit = arith::IndexCastOp::create(
              body, loc, body.getIndexType(),
              memref::LoadOp::create(body, loc, activeUnits,
                                     ValueRange{unit}));
        IRMapping local;
        SmallVector<Value> contributions = kernels::emitGroupPairKernel(
            body, op, lists, layout, box, inverse, unit, lane, local);
        SmallVector<Value> next;
        for (auto [sum, contribution] : llvm::zip(sums, contributions))
          next.push_back(arith::AddFOp::create(body, loc, sum, contribution));
        scf::YieldOp::create(body, loc, next);
      });
  SmallVector<Value> totals(loop.getResults());
  for (Value &total : totals)
    for (int64_t offset = 16; offset >= 1; offset /= 2)
      total = arith::AddFOp::create(kernel, loc, total,
                                    shuffleXor(kernel, loc, total, offset));
  if (!totals.empty()) {
    Value leader = arith::AndIOp::create(
        kernel, loc, valid,
        arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::eq, lane,
                              createIndex(kernel, loc, 0)));
    scf::IfOp::create(kernel, loc, leader, [&](OpBuilder &then, Location) {
      for (auto [index, total] : llvm::enumerate(totals))
        storeElement(then, loc, total, op.getScratch()[2 * index], warp);
      scf::YieldOp::create(then, loc);
    });
  }
  gpu::TerminatorOp::create(kernel, loc);
  bringIn(launch);

  // The buffers are freed once the loop has read them.
  for (Value target : targets) {
    auto type = cast<MemRefType>(target.getType());
    Value plain = memref::MemorySpaceCastOp::create(
        builder, loc, MemRefType::get(type.getShape(), type.getElementType()),
        target);
    gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange(), plain);
  }
  return finishSums(op, builder, op.getReduce(), op.getScratch(), warps);
}

void Lowering::lowerFusedSums(CenterFusion &fusion, Value tupleMembers,
                              Value rows, Value tuples, Value box,
                              Value inverse) {
  md_exec::TupleForOp op = fusion.sums;
  Location loc = op.getLoc();
  OpBuilder builder(op);
  SmallVector<Value> contributions, partials;
  SmallVector<Type> types;
  for (unsigned i = 0, e = op.getReduce().size(); i != e; ++i) {
    contributions.push_back(op.getScratch()[2 * i]);
    partials.push_back(op.getScratch()[2 * i + 1]);
    types.push_back(md_exec::getKernelValueType(contributions[i].getType()));
  }
  int64_t numbers = countKernelNumbers(types);
  {
    Block &entry = current.getBody().front();
    OpBuilder at(&entry, entry.begin());
    fusion.totals = createDeviceBuffer(
        at, loc, getDeviceType({numbers}, builder.getF64Type()), ValueRange());
  }
  // The parts as emitReductions takes them (D150). Several parts are
  // reduced by its kernels, in its order, into the buffer of the totals,
  // for the block of the forces.
  Value parts = arith::MinSIOp::create(
      builder, loc, createIndex(builder, loc, blockSize),
      createGroups(builder, loc, rows, 4 * blockSize));
  Value several = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                        parts, createIndex(builder, loc, 1));
  scf::IfOp::create(builder, loc, several, [&](OpBuilder &then, Location) {
    launchOver(then, loc, rows, [&](OpBuilder &body, Value row) {
      SmallVector<Value> values = emitTuplesOnceWithSums(
          body, op, tupleMembers, row, rows, tuples, box, inverse);
      for (auto [index, value] : llvm::enumerate(values))
        storeElement(body, loc, value, contributions[index], row);
    });
    launchSharedBlocks(
        then, loc, parts, numbers,
        [&](OpBuilder &b, Value block, Value thread, Value shared) {
          Value threads = createIndex(b, loc, blockSize);
          Value first = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, block, threads), thread);
          Value stride = arith::MulIOp::create(b, loc, parts, threads);
          SmallVector<Value> zeros;
          for (Type type : types)
            zeros.push_back(createZero(b, loc, type));
          auto loop = scf::ForOp::create(
              b, loc, first, rows, stride, zeros,
              [&](OpBuilder &inner, Location, Value i, ValueRange sums) {
                SmallVector<Value> next;
                for (auto [buffer, sum] : llvm::zip(contributions, sums))
                  next.push_back(arith::AddFOp::create(
                      inner, loc, sum, loadElement(inner, loc, buffer, i)));
                scf::YieldOp::create(inner, loc, next);
              });
          SmallVector<Value> totals = reduceInBlock(
              b, loc, loop.getResults(), thread, shared, /*isSum=*/true);
          Value leader = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::eq, thread, createIndex(b, loc, 0));
          scf::IfOp::create(b, loc, leader, [&](OpBuilder &lead, Location) {
            for (auto [total, partial] : llvm::zip(totals, partials))
              storeElement(lead, loc, total, partial, block);
            scf::YieldOp::create(lead, loc);
          });
        });
    launchSharedBlocks(
        then, loc, createIndex(then, loc, 1), numbers,
        [&](OpBuilder &b, Value, Value thread, Value shared) {
          Value inside = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::ult, thread, parts);
          Value slot = arith::SelectOp::create(b, loc, inside, thread,
                                               createIndex(b, loc, 0));
          SmallVector<Value> values;
          for (auto [partial, type] : llvm::zip(partials, types))
            values.push_back(arith::SelectOp::create(
                b, loc, inside, loadElement(b, loc, partial, slot),
                createZero(b, loc, type)));
          SmallVector<Value> totals =
              reduceInBlock(b, loc, values, thread, shared, /*isSum=*/true);
          Value leader = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::eq, thread, createIndex(b, loc, 0));
          scf::IfOp::create(b, loc, leader, [&](OpBuilder &lead, Location) {
            for (auto [index, total] : llvm::enumerate(totals))
              memref::StoreOp::create(lead, loc, total, fusion.totals,
                                      ValueRange{createIndex(lead, loc, index)});
            scf::YieldOp::create(lead, loc);
          });
        });
    scf::YieldOp::create(then, loc);
  });
  // The arithmetic after the sums is computed in the kernel of the forces;
  // on the host it takes these and is erased with the loop of the forces.
  for (Value result : op->getResults()) {
    Value zero = createZero(builder, loc, result.getType());
    result.replaceAllUsesWith(zero);
    fusion.placeholders.push_back(zero);
  }
}

LogicalResult Lowering::lowerTupleForWithFlags(md_exec::TupleForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // A value in `reduce` is a sum, which takes two buffers, or tells whether
  // the kernel yields true for some tuple, which takes a flag of the loop's
  // own (D118), as in a loop over particles (D278).
  SmallVector<int> places;
  unsigned numSums = 0, numFlags = 0;
  for (Value value : op.getReduce())
    places.push_back(value.getType().isInteger(1) ? -int(++numFlags)
                                                  : int(numSums++));
  if (failed(checkScratch(op, numSums, op.getScratch().size())))
    return failure();
  if (!op.getDisjoint() && !evaluatesOnce(op))
    return op->emitOpError()
           << "takes a flag in 'reduce' over tuples that may share "
              "particles, which a device evaluates once per tuple only "
              "outside the deterministic mode";
  SmallVector<Flag, 2> used;
  for (unsigned i = 0; i != numFlags; ++i)
    used.push_back(getFlag(flags.size(), loc));

  Value positions = op.getPositions();
  Value size = createSize(builder, loc, positions);
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);

  // The contributions of a row: a sum to its buffer, and a flag set by a
  // thread whose tuples yield true. Every thread that writes the flag
  // writes the same value, so the threads need not take turns.
  auto store = [&](OpBuilder &body, ArrayRef<Value> contributions,
                   Value row) {
    for (auto [index, value] : llvm::enumerate(contributions)) {
      int place = places[index];
      if (place >= 0) {
        storeElement(body, loc, value, op.getScratch()[2 * place], row);
        continue;
      }
      Value device = used[-place - 1].device;
      scf::IfOp::create(body, loc, value, [&](OpBuilder &then, Location) {
        Value set = arith::ConstantOp::create(then, loc, then.getI32Type(),
                                              then.getI32IntegerAttr(1));
        memref::StoreOp::create(then, loc, set, device,
                                ValueRange{createIndex(then, loc, 0)});
        scf::YieldOp::create(then, loc);
      });
    }
  };

  Value rows = size;
  if (op.getDisjoint()) {
    // Each tuple once, by the thread of its member at place 0.
    launchOver(builder, loc, size, [&](OpBuilder &body, Value particle) {
      IRMapping local;
      store(body,
            emitTupleKernel(body, op, op.getIncidence(), box, inverse,
                            particle, local),
            particle);
    });
  } else {
    // Each tuple once, by one of m rows, which takes the tuples r, r + m,
    // ..., as a loop with global sums does (D103).
    Value tupleMembers = members.lookup(op.getIncidence());
    if (!tupleMembers)
      return op->emitOpError()
             << "has no members on the device to evaluate each tuple once";
    Value tuples = createSize(builder, loc, tupleMembers);
    rows = arith::MinUIOp::create(builder, loc, size, tuples);
    launchOver(builder, loc, rows, [&](OpBuilder &body, Value row) {
      store(body,
            emitTuplesOnceWithSums(body, op, tupleMembers, row, rows, tuples,
                                   box, inverse),
            row);
    });
  }

  SmallVector<Value> contributions, partials;
  for (unsigned i = 0; i != numSums; ++i) {
    contributions.push_back(op.getScratch()[2 * i]);
    partials.push_back(op.getScratch()[2 * i + 1]);
  }
  SmallVector<Value> sums = emitReductions(builder, loc, contributions,
                                           partials, rows, /*isSum=*/true);
  for (auto [index, start] : llvm::enumerate(op.getReduce())) {
    int place = places[index];
    if (place < 0) {
      replaceWithFlag(builder, loc, used[-place - 1], start,
                      op.getResult(index));
      continue;
    }
    Value total = arith::AddFOp::create(builder, loc, start, sums[place]);
    op.getResult(index).replaceAllUsesWith(total);
  }
  return success();
}

LogicalResult Lowering::lowerTupleFor(md_exec::TupleForOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  if (hasFlags(op))
    return lowerTupleForWithFlags(op);
  if (failed(checkScratch(op, op.getReduce().size(), op.getScratch().size())))
    return failure();

  Value positions = op.getPositions();
  Value size = createSize(builder, loc, positions);

  // The cell has become the vector of its edge lengths.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCellMutable().get(), real);
  Value inverse = createInverse(builder, loc, box);

  // By default a set whose tuples may share particles has each tuple
  // evaluated once, by a thread of its own, which adds to the members with
  // atomics (D84).
  if (evaluatesOnce(op)) {
    Value tupleMembers = members.lookup(op.getIncidence());
    if (!tupleMembers)
      return op->emitOpError()
             << "has no members on the device to evaluate each tuple once";
    Value tuples = createSize(builder, loc, tupleMembers);
    if (op.getReduce().empty()) {
      launchOver(builder, loc, tuples, [&](OpBuilder &body, Value tuple) {
        IRMapping local;
        emitTupleOnce(body, op, tupleMembers, tuple, box, inverse, local);
      });
      return success();
    }
    // With global sums, a thread for each of m rows, the tuples or the
    // particles if they are fewer, takes the tuples r, r + m, ..., and
    // writes the sums of their contributions to its row: the reduction is
    // over the rows, in an order that does not depend on the threads. A
    // set of few tuples, as the pairs of the centers of groups (D139),
    // launches and reduces as many rows as it has tuples.
    // Few rows, as those of such a set, are reduced in one part, whose
    // block computes them itself (#20).
    Value rows = arith::MinUIOp::create(builder, loc, size, tuples);
    // The sums of a term over centers, which the kernel of its forces
    // computes with one part (#66).
    if (CenterFusion *fusion = fusionOfSums.lookup(op)) {
      lowerFusedSums(*fusion, tupleMembers, rows, tuples, box, inverse);
      return success();
    }
    auto produce = [&](OpBuilder &at) {
      launchOver(at, loc, rows, [&](OpBuilder &body, Value row) {
        SmallVector<Value> contributions = emitTuplesOnceWithSums(
            body, op, tupleMembers, row, rows, tuples, box, inverse);
        for (auto [index, value] : llvm::enumerate(contributions))
          storeElement(body, loc, value, op.getScratch()[2 * index], row);
      });
    };
    auto row = [&](OpBuilder &body, Value i) {
      return emitTuplesOnceWithSums(body, op, tupleMembers, i, rows, tuples,
                                    box, inverse);
    };
    return finishSums(op, builder, op.getReduce(), op.getScratch(), rows,
                      RowSums{produce, row});
  }
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
  // The structure of the members of a tuple set, in the order of the files
  // or renumbered, is the runtime's to keep, with the copy of the members
  // for the loops that evaluate each tuple once; one of tuples that the
  // program finds (the triplets) is built here.
  if (deviceTuples) {
    auto found = deviceTuplesOf.find(op.getRelation());
    if (found != deviceTuplesOf.end()) {
      buildOnDevice(op, found->second);
      return;
    }
    if (isGivenToFunction(op.getRelation())) {
      buildOnDevice(op, takeOnDevice(op, op.getRelation(), Value(),
                                     op.getSize()));
      return;
    }
  }
  if (keptTuples && (runtimeMembers.count(op.getRelation()) ||
                     isGivenToFunction(op.getRelation()))) {
    Value relation = op.getRelation();
    Value device =
        takeFromRuntime(op, tuplesIncidenceName, relation, op.getSize(),
                        cast<MemRefType>(op.getResult().getType()));
    op.getResult().replaceAllUsesWith(device);
    auto hostType = cast<MemRefType>(relation.getType());
    if (deterministic || !tuplesOnce || hostType.getRank() != 2)
      return;
    members[device] = takeFromRuntime(
        op, tuplesMembersName, relation, Value(),
        getDeviceType(hostType.getShape(), hostType.getElementType()));
    return;
  }
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

  // The members too, for the loops that evaluate each tuple once.
  if (deterministic || !tuplesOnce)
    return;
  Value relation = op.getRelation();
  auto hostType = dyn_cast<MemRefType>(relation.getType());
  if (!hostType || hostType.getRank() != 2)
    return;
  Value rows = memref::DimOp::create(builder, loc, relation, zero);
  Value copy = createDeviceBuffer(
      builder, loc,
      getDeviceType(hostType.getShape(), hostType.getElementType()),
      hostType.isDynamicDim(1)
          ? ValueRange{rows, memref::DimOp::create(builder, loc, relation, one)}
          : ValueRange{rows});
  createTransfer(builder, loc, copy, relation);
  members[device] = copy;
  freeDeviceAtEndOfBlock(op, copy);
}

Value Lowering::takeFromRuntime(Operation *op, StringRef name, Value members,
                                Value other, MemRefType type) {
  Location loc = op->getLoc();
  OpBuilder builder(op);
  Type narrow = builder.getI32Type();
  MemRefType wide = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, narrow);
  MemRefType given =
      type.getMemorySpace()
          ? getDeviceType({ShapedType::kDynamic, ShapedType::kDynamic}, narrow)
          : wide;
  if (members.getType() != wide)
    members = memref::CastOp::create(builder, loc, wide, members);
  SmallVector<Value, 2> arguments = {members};
  SmallVector<Type, 2> types = {wide};
  if (other) {
    arguments.push_back(other);
    types.push_back(other.getType());
  }
  func::FuncOp callee =
      getOrDeclare(name, builder.getFunctionType(types, {given}));
  callee->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  // The runtime copies the members it is given and compares them, and
  // never writes them (`mdrtTuplesMembersOf`, D258).
  callee.setArgAttr(0, mdrt::getReadOnlyAttrName(), builder.getUnitAttr());
  Value taken =
      func::CallOp::create(builder, loc, callee, arguments).getResult(0);
  // The runtime frees it, or keeps it for the next activation.
  OpBuilder atEnd(op->getBlock()->getTerminator());
  Value plain = taken;
  if (given != wide)
    plain = memref::MemorySpaceCastOp::create(atEnd, loc, wide, taken);
  func::FuncOp release = getOrDeclare(
      tuplesReleaseName, atEnd.getFunctionType({wide}, {}));
  release->setAttr("llvm.emit_c_interface", atEnd.getUnitAttr());
  func::CallOp::create(atEnd, loc, release, ValueRange{plain});
  if (type == given)
    return taken;
  return memref::CastOp::create(builder, loc, type, taken);
}

Value Lowering::callRuntime(OpBuilder &builder, Location loc, StringRef name,
                            ValueRange arguments, Type result, bool memrefs) {
  SmallVector<Type> types;
  for (Value argument : arguments)
    types.push_back(argument.getType());
  SmallVector<Type, 1> results;
  if (result)
    results.push_back(result);
  func::FuncOp callee =
      getOrDeclare(name, builder.getFunctionType(types, results));
  // A function that takes or gives a buffer has the C interface.
  if (memrefs)
    callee->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto call = func::CallOp::create(builder, loc, callee, arguments);
  return result ? call.getResult(0) : Value();
}

/// The identities `ids` against the store's copy of the last order: a
/// kernel sets a flag where one differs (every such thread writes the same
/// value), and the runtime reads the flag, once in a block for each value
/// of the identities (D257).
void Lowering::compareOrder(Operation *op, Value ids) {
  if (!comparedOrders.insert({ids, op->getBlock()}).second)
    return;
  Location loc = op->getLoc();
  OpBuilder builder(op);
  Type narrow = builder.getI32Type();
  Value kept = callRuntime(builder, loc, "mdrtDeviceTuplesOrderKept", {ids},
                           ids.getType(), true);
  Value flag = callRuntime(builder, loc, "mdrtDeviceTuplesOrderFlag", {},
                           getDeviceType({1}, narrow), true);
  Value count =
      memref::DimOp::create(builder, loc, ids, createIndex(builder, loc, 0));
  launchOver(builder, loc, count, [&](OpBuilder &body, Value p) {
    Value now = memref::LoadOp::create(body, loc, ids, ValueRange{p});
    Value before = memref::LoadOp::create(body, loc, kept, ValueRange{p});
    Value differs = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ne,
                                          now, before);
    scf::IfOp::create(body, loc, differs, [&](OpBuilder &then, Location) {
      memref::StoreOp::create(
          then, loc,
          arith::ConstantOp::create(then, loc, narrow,
                                    then.getI32IntegerAttr(1)),
          flag, ValueRange{createIndex(then, loc, 0)});
      scf::YieldOp::create(then, loc);
    });
  });
  callRuntime(builder, loc, "mdrtDeviceTuplesOrderEnd", {ids}, Type(), true);
}

/// Takes the entry of the store for `members` (on the host), in the order
/// `ids` or, without them, in the order of the files; where the runtime
/// says to build it, launches the inverse of the order and the renumbered
/// members. The entry is given back at the end of the block.
Lowering::DeviceTuples Lowering::takeOnDevice(Operation *op, Value members,
                                              Value ids, Value particles) {
  Location loc = op->getLoc();
  if (ids)
    compareOrder(op, ids);
  OpBuilder builder(op);
  Type narrow = builder.getI32Type();
  Type wide64 = builder.getI64Type();
  MemRefType wide = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, narrow);
  MemRefType device2 =
      getDeviceType({ShapedType::kDynamic, ShapedType::kDynamic}, narrow);
  MemRefType device1 = getDeviceType({ShapedType::kDynamic}, narrow);
  DeviceTuples tuples;
  tuples.arity = cast<MemRefType>(members.getType()).getDimSize(1);
  tuples.ids = ids;
  Value given = members;
  if (given.getType() != wide)
    given = memref::CastOp::create(builder, loc, wide, given);
  Value count = arith::IndexCastOp::create(builder, loc, wide64, particles);
  Value ordered = arith::ConstantOp::create(
      builder, loc, narrow, builder.getI32IntegerAttr(ids ? 1 : 0));
  tuples.handle = callRuntime(builder, loc, "mdrtDeviceTuplesTake",
                              {given, count, ordered}, wide64, true);
  // The runtime copies the members and compares them, and never writes
  // them (`mdrtTuplesMembersOf`, D258).
  cast<func::FuncOp>(SymbolTable::lookupSymbolIn(module, "mdrtDeviceTuplesTake"))
      .setArgAttr(0, mdrt::getReadOnlyAttrName(), builder.getUnitAttr());
  Value build = callRuntime(builder, loc, "mdrtDeviceTuplesBuild",
                            {tuples.handle}, narrow, false);
  tuples.build = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, build,
      arith::ConstantOp::create(builder, loc, narrow,
                                builder.getI32IntegerAttr(0)));
  tuples.file = callRuntime(builder, loc, "mdrtDeviceTuplesFile",
                            {tuples.handle}, device2, true);
  tuples.first = callRuntime(builder, loc, "mdrtDeviceTuplesFirst",
                             {tuples.handle}, device1, true);
  tuples.entries = callRuntime(builder, loc, "mdrtDeviceTuplesEntries",
                               {tuples.handle}, device1, true);
  tuples.renumbered = callRuntime(builder, loc, "mdrtDeviceTuplesRenumbered",
                                  {tuples.handle}, device2, true);
  if (ids) {
    Value place = callRuntime(builder, loc, "mdrtDeviceTuplesPlace",
                              {tuples.handle}, device1, true);
    Value file = tuples.file, renumbered = tuples.renumbered;
    int64_t arity = tuples.arity;
    Value tupleCount = memref::DimOp::create(builder, loc, file,
                                             createIndex(builder, loc, 0));
    Value elements = arith::MulIOp::create(builder, loc, tupleCount,
                                           createIndex(builder, loc, arity));
    scf::IfOp::create(builder, loc, tuples.build, [&](OpBuilder &then,
                                                       Location) {
      // The inverse of the order: place[ids[p]] = p, each element once.
      launchOver(then, loc, particles, [&](OpBuilder &body, Value p) {
        Value id = memref::LoadOp::create(body, loc, ids, ValueRange{p});
        memref::StoreOp::create(
            body, loc, arith::IndexCastOp::create(body, loc, narrow, p), place,
            ValueRange{arith::IndexCastOp::create(body, loc,
                                                  body.getIndexType(), id)});
      });
      // The members at their places.
      Value hasTuples = arith::CmpIOp::create(
          then, loc, arith::CmpIPredicate::ugt, elements,
          createIndex(then, loc, 0));
      scf::IfOp::create(then, loc, hasTuples, [&](OpBuilder &some, Location) {
        launchOver(some, loc, elements, [&](OpBuilder &body, Value i) {
          Value a = createIndex(body, loc, arity);
          Value t = arith::DivUIOp::create(body, loc, i, a);
          Value s = arith::RemUIOp::create(body, loc, i, a);
          Value member =
              memref::LoadOp::create(body, loc, file, ValueRange{t, s});
          Value now = memref::LoadOp::create(
              body, loc, place,
              ValueRange{arith::IndexCastOp::create(
                  body, loc, body.getIndexType(), member)});
          memref::StoreOp::create(body, loc, now, renumbered,
                                  ValueRange{t, s});
        });
        scf::YieldOp::create(some, loc);
      });
      scf::YieldOp::create(then, loc);
    });
  }
  OpBuilder atEnd(op->getBlock()->getTerminator());
  callRuntime(atEnd, loc, "mdrtDeviceTuplesDone", {tuples.handle}, Type(),
              false);
  return tuples;
}

/// The rows of the incidence structure of an entry taken by
/// `takeOnDevice`, where it is to be built: one thread for each slot k of
/// each place p, which copies the k-th tuple of the particle at p in the
/// order of the files, its members relabeled.
void Lowering::buildOnDevice(md_exec::BuildIncidenceOp op,
                             const DeviceTuples &tuples) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Type narrow = builder.getI32Type();
  MemRefType device2 =
      getDeviceType({ShapedType::kDynamic, ShapedType::kDynamic}, narrow);
  Value structure = callRuntime(builder, loc, "mdrtDeviceTuplesStructure",
                                {tuples.handle}, device2, true);
  int64_t arity = tuples.arity, entry = arity + 2;
  Value particles = memref::DimOp::create(builder, loc, structure,
                                          createIndex(builder, loc, 0));
  Value width = memref::DimOp::create(builder, loc, structure,
                                      createIndex(builder, loc, 1));
  Value slots = arith::DivUIOp::create(
      builder, loc,
      arith::SubIOp::create(builder, loc, width, createIndex(builder, loc, 1)),
      createIndex(builder, loc, entry));
  // A row has at least its count.
  Value lanes = arith::MaxUIOp::create(builder, loc, slots,
                                       createIndex(builder, loc, 1));
  Value items = arith::MulIOp::create(builder, loc, particles, lanes);
  Value some = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt,
                                     items, createIndex(builder, loc, 0));
  Value build = arith::AndIOp::create(builder, loc, tuples.build, some);
  Value ids = tuples.ids, first = tuples.first, entries = tuples.entries,
        renumbered = tuples.renumbered;
  scf::IfOp::create(builder, loc, build, [&](OpBuilder &then, Location) {
    launchOver(then, loc, items, [&](OpBuilder &body, Value i) {
      Type index = body.getIndexType();
      auto toIndex = [&](Value value) -> Value {
        return arith::IndexCastOp::create(body, loc, index, value);
      };
      Value p = arith::DivUIOp::create(body, loc, i, lanes);
      Value k = arith::RemUIOp::create(body, loc, i, lanes);
      // The particle at p in the order of the files.
      Value q = ids ? toIndex(memref::LoadOp::create(body, loc, ids,
                                                     ValueRange{p}))
                    : p;
      Value from = toIndex(memref::LoadOp::create(body, loc, first,
                                                  ValueRange{q}));
      Value next = arith::AddIOp::create(body, loc, q,
                                         createIndex(body, loc, 1));
      Value to = toIndex(memref::LoadOp::create(body, loc, first,
                                                ValueRange{next}));
      Value count = arith::SubIOp::create(body, loc, to, from);
      Value zero = createIndex(body, loc, 0);
      Value head = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq,
                                         k, zero);
      scf::IfOp::create(body, loc, head, [&](OpBuilder &b, Location) {
        memref::StoreOp::create(
            b, loc, arith::IndexCastOp::create(b, loc, narrow, count),
            structure, ValueRange{p, zero});
        scf::YieldOp::create(b, loc);
      });
      Value inside = arith::CmpIOp::create(body, loc,
                                           arith::CmpIPredicate::ult, k, count);
      // A slot past the count of the particle holds zeros: the kernels
      // of an integration run read the first tuple of every particle and
      // keep it only where the count is not zero, and the memory of the
      // structure is not cleared when it is allocated (#347).
      auto empty = [&](OpBuilder &b, Location) {
        Value zero32 = arith::ConstantOp::create(b, loc, narrow,
                                                 b.getI32IntegerAttr(0));
        Value base = arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(b, loc, k, createIndex(b, loc, entry)),
            createIndex(b, loc, 1));
        for (int64_t c = 0; c != entry; ++c)
          memref::StoreOp::create(
              b, loc, zero32, structure,
              ValueRange{p, arith::AddIOp::create(b, loc, base,
                                                  createIndex(b, loc, c))});
        scf::YieldOp::create(b, loc);
      };
      Value fits = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ult,
                                         k, slots);
      scf::IfOp::create(
          body, loc,
          arith::AndIOp::create(
              body, loc, fits,
              arith::XOrIOp::create(
                  body, loc, inside,
                  arith::ConstantOp::create(body, loc, body.getI1Type(),
                                            body.getBoolAttr(true)))),
          empty);
      scf::IfOp::create(body, loc, inside, [&](OpBuilder &b, Location) {
        Value at = arith::MulIOp::create(
            b, loc, arith::AddIOp::create(b, loc, from, k),
            createIndex(b, loc, 2));
        Value t = memref::LoadOp::create(b, loc, entries, ValueRange{at});
        Value s = memref::LoadOp::create(
            b, loc, entries,
            ValueRange{arith::AddIOp::create(b, loc, at,
                                             createIndex(b, loc, 1))});
        Value base = arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(b, loc, k, createIndex(b, loc, entry)),
            createIndex(b, loc, 1));
        auto column = [&](int64_t c) -> Value {
          return arith::AddIOp::create(b, loc, base, createIndex(b, loc, c));
        };
        memref::StoreOp::create(b, loc, t, structure, ValueRange{p, column(0)});
        memref::StoreOp::create(b, loc, s, structure, ValueRange{p, column(1)});
        Value tuple = arith::IndexCastOp::create(b, loc, index, t);
        for (int64_t j = 0; j != arity; ++j) {
          Value member = memref::LoadOp::create(
              b, loc, renumbered,
              ValueRange{tuple, createIndex(b, loc, j)});
          memref::StoreOp::create(b, loc, member, structure,
                                  ValueRange{p, column(2 + j)});
        }
        scf::YieldOp::create(b, loc);
      });
    });
    scf::YieldOp::create(then, loc);
  });
  callRuntime(builder, loc, "mdrtDeviceTuplesBuilt", {tuples.handle}, Type(),
              false);
  auto type = cast<MemRefType>(op.getResult().getType());
  Value result = structure;
  if (type != device2)
    result = memref::CastOp::create(builder, loc, type, structure);
  op.getResult().replaceAllUsesWith(result);

  // The members for the loops that evaluate each tuple once: the renumbered
  // members on the device, written before on the same stream.
  auto hostType = cast<MemRefType>(op.getRelation().getType());
  if (deterministic || !tuplesOnce || hostType.getRank() != 2)
    return;
  MemRefType membersType =
      getDeviceType(hostType.getShape(), hostType.getElementType());
  Value copy = renumbered;
  if (copy.getType() != membersType)
    copy = memref::CastOp::create(builder, loc, membersType, copy);
  members[result] = copy;
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
  if (deviceTuples) {
    // The renumbered members are on the device; the result of the op on
    // the host is read by nothing but the build of its structure, which
    // takes the entry from here.
    OpBuilder builder(op);
    Value ids = op.getIds();
    Value particles =
        memref::DimOp::create(builder, loc, ids, createIndex(builder, loc, 0));
    DeviceTuples tuples = takeOnDevice(op, op.getMembers(), ids, particles);
    // A value of its own, so that two renumberings of one set (the anchors
    // in the order before and in the new one) keep their entries apart.
    Value result = memref::CastOp::create(
        builder, loc, op.getResult().getType(), op.getMembers());
    op.getResult().replaceAllUsesWith(result);
    deviceTuplesOf[result] = tuples;
    return;
  }
  if (keptTuples) {
    Value members =
        takeFromRuntime(op, tuplesRenumberName, op.getMembers(), op.getIds(),
                        cast<MemRefType>(op.getResult().getType()));
    runtimeMembers.insert(members);
    op.getResult().replaceAllUsesWith(members);
    return;
  }
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

LogicalResult Lowering::addGroupsTemplates(bool tilted) {
  if (SymbolTable::lookupSymbolIn(
          module, tilted ? std::string(buildGroupsName) + "_triclinic"
                         : std::string(buildGroupsName)))
    return success();
  ParserConfig config(context);
  OwningOpRef<ModuleOp> templates = parseSourceString<ModuleOp>(
      tilted ? neighborsGroupsGPUTriclinicTemplate : neighborsGroupsGPUTemplate,
      config);
  if (!templates)
    return module.emitError()
           << "cannot parse the template that builds groups of neighbors";
  for (Operation &op : llvm::make_early_inc_range(*templates)) {
    op.remove();
    module.push_back(&op);
  }
  return success();
}

/// Unrolls, in the kernels of `root`, every loop whose trip count is a
/// constant of at most `limit`, innermost first, folding constants between
/// the rounds so that a loop whose bounds were the induction variables of
/// the loops around it unrolls too. The B-splines of PME are held in small
/// arrays indexed by the loops over the order: unrolled, the indices are
/// constants and the arrays become registers; otherwise they stay in local
/// memory (the gather of Cellulose took 303 us so).
static void unrollSmallLoops(Operation *root, int64_t limit) {
  MLIRContext *context = root->getContext();
  RewritePatternSet patterns(context);
  for (Dialect *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName name : context->getRegisteredOperations())
    name.getCanonicalizationPatterns(patterns, context);
  FrozenRewritePatternSet frozen(std::move(patterns));
  for (;;) {
    (void)applyPatternsGreedily(root, frozen);
    SmallVector<scf::ForOp> loops;
    root->walk([&](scf::ForOp loop) {
      if (!loop->getParentOfType<gpu::LaunchOp>())
        return;
      // The innermost of the loops that can unroll: a loop inside whose
      // bounds are not constants yet may become one once this unrolls.
      bool innermost = true;
      loop.getBody()->walk([&](scf::ForOp inner) {
        std::optional<APInt> count = inner.getStaticTripCount();
        innermost &= !(count && count->getSExtValue() <= limit);
      });
      std::optional<APInt> count = loop.getStaticTripCount();
      if (innermost && count && count->getSExtValue() <= limit)
        loops.push_back(loop);
    });
    if (loops.empty())
      return;
    for (scf::ForOp loop : loops)
      (void)loopUnrollFull(loop);
  }
}

LogicalResult Lowering::addPMETemplates(Type position, Type charge,
                                        Type force, int64_t order,
                                        bool tilted, bool potential,
                                        bool listed, int64_t tiles) {
  // A triclinic cell takes the kernels of its own as well, which only it
  // parses and specializes (docs/triclinic-m2.md); the spreading into
  // bricks, of its list too.
  if (tilted && failed(addPMETemplates(position, charge, force, order,
                                       /*tilted=*/false, /*potential=*/false,
                                       listed, tiles)))
    return failure();
  auto has = [&](StringRef name) {
    std::string full = tilted ? (name + "_triclinic").str() : name.str();
    return SymbolTable::lookupSymbolIn(
               module, getPMEInstanceName(full, position, charge, force,
                                          order)) != nullptr;
  };
  // The functions of the tiles of an edge (D[pme-tiles]), which do not
  // depend on the cell: in PMEGPU.mlir alone.
  std::string edge = std::to_string(tiles);
  auto hasTiles = [&]() {
    if (!tiles)
      return true;
    auto named = [&](StringRef name) {
      return SymbolTable::lookupSymbolIn(
                 module, getPMEInstanceName(name, position, charge, force,
                                            order)) != nullptr;
    };
    return named("mdrt_gpu_pme_spread_tiles" + edge) &&
           named("mdrt_gpu_pme_bin" + edge + (listed ? "_listed" : ""));
  };
  if (has("mdrt_gpu_pme_spread") &&
      (!potential || has("mdrt_gpu_pme_potential")) &&
      (!listed || has("mdrt_gpu_pme_spread_listed")) && hasTiles())
    return success();
  ParserConfig config(context);
  OwningOpRef<ModuleOp> templates = parseSourceString<ModuleOp>(
      instantiatePMETemplates(tilted ? pmeGPUTriclinicTemplate
                                     : pmeGPUTemplate,
                              position, charge, force, order),
      config);
  if (!templates)
    return module.emitError()
           << "cannot parse the template of particle mesh Ewald for devices";
  unrollSmallLoops(templates->getOperation(), /*limit=*/8);
  // The arrays of the splines, indexed by constants now, become values.
  PassManager promote(context);
  promote.addPass(createSROA());
  promote.addPass(createMem2Reg());
  promote.addPass(createCanonicalizerPass());
  if (failed(promote.run(*templates)))
    return module.emitError()
           << "cannot promote the splines of particle mesh Ewald to values";
  // The canonicalization hoists the constants of the kernels to their
  // functions, where the outlining would make them arguments: a division
  // by 32 became one by an argument, of 64 bits, at twice the time of the
  // spreading (2026-10-01). They go back into the kernels.
  WalkResult sunk = templates->walk([](gpu::LaunchOp launch) {
    return failed(sinkOperationsIntoLaunchOp(
               launch, [](Operation *op) {
                 return isa<arith::ConstantOp>(op);
               }))
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  if (sunk.wasInterrupted())
    return module.emitError()
           << "cannot sink the constants of particle mesh Ewald into its "
              "kernels";
  // The functions of the potential at the particles (the fourth result of
  // md.reciprocal), and those of a list of the particles
  // (D262), only where a sum asks for them, so
  // that a module without one is what it was; they may come after the
  // others, for a sum of the same types that was lowered before.
  for (Operation &op : llvm::make_early_inc_range(*templates)) {
    auto function = dyn_cast<func::FuncOp>(&op);
    bool isPotential =
        function && function.getSymName().contains("pme_potential");
    bool isListed = function && function.getSymName().contains("_listed");
    // The binning and the tiles of the edge that the sum takes, alone.
    bool isTiles = function &&
                   (function.getSymName().starts_with("mdrt_gpu_pme_bin") ||
                    function.getSymName().contains("_spread_tiles"));
    bool ownEdge = false;
    if (isTiles && tiles) {
      StringRef name = function.getSymName();
      ownEdge = name.contains("mdrt_gpu_pme_bin" + edge + "_") ||
                name.contains("mdrt_gpu_pme_spread_tiles" + edge + "_");
    }
    if ((isPotential && !potential) || (isListed && !listed) ||
        (isTiles && !ownEdge) ||
        (function && SymbolTable::lookupSymbolIn(module, function.getSymName()))) {
      op.erase();
      continue;
    }
    op.remove();
    module.push_back(&op);
  }
  return success();
}

LogicalResult Lowering::addProlateTemplates(Type position, Type charge,
                                            Type force, int64_t order,
                                            bool listed, int64_t tiles) {
  auto has = [&](StringRef name) {
    return SymbolTable::lookupSymbolIn(
               module,
               getPMEInstanceName(name, position, charge, force, order)) !=
           nullptr;
  };
  std::string edge = std::to_string(tiles);
  if (has("mdrt_gpu_pme_prolate_gather") &&
      (!listed || has("mdrt_gpu_pme_prolate_gather_listed")) &&
      (!tiles || has("mdrt_gpu_pme_prolate_spread_tiles" + edge)))
    return success();
  ParserConfig config(context);
  OwningOpRef<ModuleOp> templates = parseSourceString<ModuleOp>(
      instantiatePMETemplates(pmeGPUProlateTemplate, position, charge, force,
                              order),
      config);
  if (!templates)
    return module.emitError()
           << "cannot parse the template of the prolate kernel for devices";
  // As addPMETemplates: the loops over the points unroll, and the
  // constants go back into the kernels.
  unrollSmallLoops(templates->getOperation(), /*limit=*/8);
  PassManager promote(context);
  promote.addPass(createCanonicalizerPass());
  if (failed(promote.run(*templates)))
    return module.emitError()
           << "cannot canonicalize the kernels of the prolate kernel";
  WalkResult sunk = templates->walk([](gpu::LaunchOp launch) {
    return failed(sinkOperationsIntoLaunchOp(
               launch, [](Operation *op) {
                 return isa<arith::ConstantOp>(op);
               }))
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  if (sunk.wasInterrupted())
    return module.emitError()
           << "cannot sink the constants of the prolate kernel into its "
              "kernels";
  // The kernels of a list of the particles (D262) only where a sum asks
  // for them, as addPMETemplates.
  for (Operation &op : llvm::make_early_inc_range(*templates)) {
    auto function = dyn_cast<func::FuncOp>(&op);
    bool isListed = function && function.getSymName().contains("_listed");
    // The tiles of the edge that the sum takes, alone.
    bool isTiles =
        function && function.getSymName().contains("_spread_tiles");
    bool ownEdge = isTiles && tiles &&
                   function.getSymName().contains("_spread_tiles" + edge + "_");
    if ((isListed && !listed) || (isTiles && !ownEdge) ||
        (function &&
         SymbolTable::lookupSymbolIn(module, function.getSymName()))) {
      op.erase();
      continue;
    }
    op.remove();
    module.push_back(&op);
  }
  return success();
}

/// The bricks of the spreading (D108), the scratch after the factors of
/// the influence function where the grid is in f32; the bins of the tiles
/// (D[pme-tiles]), in i32, come after them where the sum has `tiles`.
static Value getBricks(md_exec::ReciprocalOp op) {
  if (op.getScratch().size() <= 6)
    return Value();
  Value bricks = op.getScratch()[6];
  if (cast<MemRefType>(bricks.getType()).getElementType().isInteger(32))
    return Value();
  return bricks;
}

/// The bins of the spreading into tiles (D[pme-tiles]): the last scratch
/// of a sum with `tiles`.
static Value getBins(md_exec::ReciprocalOp op) {
  if (!op.getTiles() || op.getScratch().empty())
    return Value();
  Value bins = op.getScratch().back();
  if (!cast<MemRefType>(bins.getType()).getElementType().isInteger(32))
    return Value();
  return bins;
}

/// The reciprocal sum of the prolate kernel on a device (D266):
/// the coefficients of the table into the type of the grid, the places, the
/// spreading (into bricks, or in fixed point in the deterministic mode and
/// in f64), the transforms, the factors of the edges and the product with
/// the influence function, and the gathering or the potential. With
/// `listed`, the spreading takes the listed particles alone, and so does
/// the gathering unless `gather_all` (D262); the places are those of every
/// particle.
static LogicalResult lowerProlate(
    md_exec::ReciprocalOp op, Value listed, OpBuilder &builder,
    llvm::function_ref<func::FuncOp(StringRef)> instance,
    llvm::function_ref<func::FuncOp(StringRef, FunctionType)> declare,
    bool deterministic) {
  Location loc = op.getLoc();
  ArrayRef<int64_t> grid = op.getGrid();
  auto index = [&](int64_t v) {
    return arith::ConstantIndexOp::create(builder, loc, v).getResult();
  };
  Value k1 = index(grid[0]), k2 = index(grid[1]), k3 = index(grid[2]);
  Value beta = arith::ConstantOp::create(builder, loc,
                                         builder.getF64FloatAttr(0.0));
  Value coulomb = arith::ConstantOp::create(
      builder, loc,
      builder.getF64FloatAttr(op.getCoulomb().convertToDouble()));
  Value box = op.getCellMutable().get();
  Value positions = op.getPositions(), charges = op.getCharges();
  Value fixed = op.getScratch()[0], real = op.getScratch()[1],
        complex = op.getScratch()[2], rows = op.getScratch()[3];
  if (op.getScratch().size() <= 5)
    return op->emitOpError()
           << "expected buffers for the places and the tables";
  Value places = op.getScratch()[4], tables = op.getScratch()[5];
  Value bricks = getBricks(op);
  func::CallOp::create(builder, loc,
                       instance("mdrt_gpu_pme_prolate_coefficients"),
                       ValueRange{op.getModuli(), tables, k1, k2, k3});
  func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_prolate_places"),
                       ValueRange{positions, box, places, k1, k2, k3});
  // The list follows the charges in the kernels that take it.
  auto withList = [&](ValueRange values) {
    SmallVector<Value> operands(values);
    if (listed)
      operands.insert(operands.begin() + 1, listed);
    return operands;
  };
  std::string spreadTag = listed ? "_listed" : "";
  // Into tiles (D[pme-tiles]): the bins of PMEGPU.mlir from the places,
  // then the window into the tiles.
  if (int64_t tiles = op.getTiles().value_or(0)) {
    std::string edge = std::to_string(tiles);
    Value bins = getBins(op);
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_bin" + edge + spreadTag),
                         withList({charges, places, bins, real, k1, k2, k3}));
    func::CallOp::create(
        builder, loc, instance("mdrt_gpu_pme_prolate_spread_tiles" + edge),
        ValueRange{charges, places, tables, bins, real, k1, k2, k3});
  } else if (deterministic || !bricks) {
    func::CallOp::create(
        builder, loc,
        instance("mdrt_gpu_pme_prolate_spread_fixed" + spreadTag),
        withList({charges, places, tables, fixed, k1, k2, k3}));
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_real"),
                         ValueRange{fixed, real});
  } else {
    func::CallOp::create(
        builder, loc,
        instance("mdrt_gpu_pme_prolate_spread_bricks" + spreadTag),
        withList({charges, places, tables, bricks, real, k1, k2, k3}));
  }
  Type wide = builder.getI64Type();
  SmallVector<Value> sizes;
  for (int64_t points : grid)
    sizes.push_back(arith::ConstantOp::create(
        builder, loc, wide, builder.getI64IntegerAttr(points)));
  Type buffer = real.getType();
  FunctionType transform =
      builder.getFunctionType({buffer, buffer, wide, wide, wide}, {});
  bool narrow = cast<MemRefType>(buffer).getElementType().isF32();
  func::FuncOp forward = declare(
      narrow ? "mdrtCudaFFTForward3DF32" : "mdrtCudaFFTForward3D", transform);
  func::FuncOp backward = declare(
      narrow ? "mdrtCudaFFTBackward3DF32" : "mdrtCudaFFTBackward3D",
      transform);
  for (func::FuncOp function : {forward, backward})
    function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func::CallOp::create(builder, loc, forward,
                       ValueRange{real, complex, sizes[0], sizes[1], sizes[2]});
  func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_prolate_tables"),
                       ValueRange{op.getModuli(), tables, box, beta, coulomb,
                                  k1, k2, k3});
  func::CallOp convolve;
  if (op.getEnergy().use_empty() && op.getVirial().use_empty())
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_prolate_scale"),
                         ValueRange{complex, tables, box, beta, coulomb, k1,
                                    k2, k3});
  else
    convolve = func::CallOp::create(
        builder, loc, instance("mdrt_gpu_pme_prolate_convolve"),
        ValueRange{complex, tables, rows, box, beta, coulomb, k1, k2, k3});
  func::CallOp::create(builder, loc, backward,
                       ValueRange{complex, real, sizes[0], sizes[1], sizes[2]});
  // With `gather_all`, every particle is gathered, as without a list
  // (D262).
  if (op.getGatherAll())
    listed = Value();
  func::CallOp::create(
      builder, loc,
      instance(op.getPotential() ? "mdrt_gpu_pme_prolate_potential"
               : listed          ? "mdrt_gpu_pme_prolate_gather_listed"
                                 : "mdrt_gpu_pme_prolate_gather"),
      withList({charges, places, tables, real, box, k1, k2, k3, op.getOut()}));
  if (convolve) {
    op.getEnergy().replaceAllUsesWith(convolve.getResult(0));
    op.getVirial().replaceAllUsesWith(convolve.getResult(1));
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
  // A triclinic cell takes the kernels of its own where they depend on the
  // cell (docs/triclinic-m2.md).
  bool tilted = isTriclinic(op.getCellMutable().get());
  // The listed particles alone (D262); the
  // potential at the particles takes every particle.
  Value listed = op.getPotential() ? Value() : op.getListed();
  // The spreading into tiles (D[pme-tiles]): never in the deterministic
  // mode, whose grid is in fixed point (D70, D84), which
  // md-exec-choose-spreading does not tile; checked here as well.
  int64_t tiles = op.getTiles().value_or(0);
  Value bins = getBins(op);
  if (tiles) {
    if (deterministic)
      return op->emitOpError()
             << "spreads into tiles, which the deterministic mode does not "
                "allow: its grid is in fixed point";
    if (!bins)
      return op->emitOpError() << "has tiles but no buffer of the bins";
  }
  if (failed(addPMETemplates(position, charge, force, op.getOrder(), tilted,
                             op.getPotential(), static_cast<bool>(listed),
                             tiles)))
    return failure();
  // The kernels that do not depend on the cell, which a triclinic cell
  // shares: the conversion from fixed point, the bricks, the binning and
  // the tiles.
  auto instance = [&](StringRef name) {
    bool own = tilted && name != "mdrt_gpu_pme_real" &&
               name != "mdrt_gpu_pme_spread_bricks" &&
               name != "mdrt_gpu_pme_spread_bricks_listed" &&
               !name.starts_with("mdrt_gpu_pme_bin") &&
               !name.starts_with("mdrt_gpu_pme_spread_tiles");
    std::string full = own ? (name + "_triclinic").str() : name.str();
    return cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
        module,
        getPMEInstanceName(full, position, charge, force, op.getOrder())));
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
  // Splines of order 4 in f32 are spread from the weights of the
  // particles into bricks of their own, which each spreading leaves zero
  // (D85, D108).
  Value weights = op.getScratch().size() > 4 ? op.getScratch()[4] : Value();
  Value bricks = getBricks(op);
  // The potential at the particles (the fourth result of md.reciprocal)
  // has the kernel of its own, which takes no weights.
  bool usesWeights = weights && bricks && !deterministic &&
                     op.getOrder() == 4 && force.isF32() &&
                     !op.getPotential();

  // Marked, the sum runs on a second stream, beside the ops up to its join
  // (md_exec.join, D87). All the work that it issues goes there, and none
  // of it allocates or frees memory: the runtime hands a block that is
  // freed to the next allocation without waiting, as work that runs in
  // the order in which it was issued may.
  bool side = op->hasAttr(md_exec::kSideAttrName);
  func::CallOp begin;
  if (side)
    begin = func::CallOp::create(
        builder, loc,
        getOrDeclare("mdrtSideBegin", builder.getFunctionType({}, {})),
        ValueRange());
  // The prolate kernel has kernels of its own (D266).
  if (op.getProlate()) {
    if (failed(addProlateTemplates(position, charge, force, op.getOrder(),
                                   static_cast<bool>(listed), tiles)))
      return failure();
    // A triclinic cell takes its own places, gathering, and influence;
    // the coefficients and the spreading do not depend on the cell.
    auto prolateInstance = [&](StringRef name) {
      bool own = tilted && name != "mdrt_gpu_pme_real" &&
                 name != "mdrt_gpu_pme_prolate_coefficients" &&
                 !name.starts_with("mdrt_gpu_pme_prolate_spread") &&
                 !name.starts_with("mdrt_gpu_pme_bin");
      std::string full = own ? (name + "_triclinic").str() : name.str();
      return cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
          module,
          getPMEInstanceName(full, position, charge, force, op.getOrder())));
    };
    if (failed(lowerProlate(
            op, listed, builder, prolateInstance,
            [&](StringRef name, FunctionType type) {
              return getOrDeclare(name, type);
            },
            deterministic)))
      return failure();
    if (side) {
      auto end = func::CallOp::create(
          builder, loc,
          getOrDeclare("mdrtSideEnd", builder.getFunctionType({}, {})),
          ValueRange());
      for (Operation *issued = begin->getNextNode(); issued != end;
           issued = issued->getNextNode())
        if (allocates(issued))
          return op->emitOpError()
                 << "allocates or frees memory on a second stream";
    }
    return success();
  }
  // The charges in fixed point in the deterministic mode, then converted;
  // by default with floating-point atomics, straight to the grid (D84).
  // With a list, the gathering stores the forces of the listed particles
  // and a force of zero at each of the others.
  // Into tiles: the places of every particle (the weights kernel, for the
  // gathering of order 4 as well), the binning of the particles or of the
  // entries of the list, and the tiles.
  std::string edge = std::to_string(tiles);
  if (tiles) {
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_weights"),
                         ValueRange{positions, box, weights, k1, k2, k3});
    SmallVector<Value> binned = {charges};
    if (listed)
      binned.push_back(listed);
    binned.append({weights, bins, real, k1, k2, k3});
    func::CallOp::create(
        builder, loc,
        instance("mdrt_gpu_pme_bin" + edge + (listed ? "_listed" : "")),
        binned);
    func::CallOp::create(builder, loc,
                         instance("mdrt_gpu_pme_spread_tiles" + edge),
                         ValueRange{charges, weights, bins, real, k1, k2, k3});
  } else if (listed && deterministic) {
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_spread_listed"),
                         ValueRange{positions, charges, listed, box, fixed, k1,
                                    k2, k3, order});
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_real"),
                         ValueRange{fixed, real});
  } else if (listed && usesWeights) {
    // The weights of every particle, for a gathering of every particle as
    // well as of the listed ones.
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_weights"),
                         ValueRange{positions, box, weights, k1, k2, k3});
    func::CallOp::create(builder, loc,
                         instance("mdrt_gpu_pme_spread_bricks_listed"),
                         ValueRange{positions, charges, listed, weights,
                                    bricks, real, k1, k2, k3});
  } else if (listed) {
    func::CallOp::create(builder, loc,
                         instance("mdrt_gpu_pme_spread_float_listed"),
                         ValueRange{positions, charges, listed, box, real, k1,
                                    k2, k3, order});
  } else if (deterministic) {
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_spread"),
                         ValueRange{positions, charges, box, fixed, k1, k2,
                                    k3, order});
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_real"),
                         ValueRange{fixed, real});
  } else if (usesWeights) {
    // The weights of the particles once, then the charges into bricks of
    // the grid and the bricks into the grid (weights_kernels in
    // scripts/generate-pme-gpu-template.py).
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_weights"),
                         ValueRange{positions, box, weights, k1, k2, k3});
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_spread_bricks"),
                         ValueRange{positions, charges, weights, bricks, real,
                                    k1, k2, k3});
  } else {
    func::CallOp::create(builder, loc,
                         instance("mdrt_gpu_pme_spread_float"),
                         ValueRange{positions, charges, box, real, k1, k2, k3,
                                    order});
  }
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
  // The factors of the influence function along the edges first (D104).
  if (op.getScratch().size() <= 5)
    return op->emitOpError()
           << "expected a buffer for the factors of the influence function";
  Value tables = op.getScratch()[5];
  // The sum of the dispersion multiplies by an influence function of its
  // own (D162); it spreads and gathers as the Coulomb sum does.
  std::string kind = op.getDispersion() ? "_dispersion" : "";
  func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_tables" + kind),
                       ValueRange{op.getModuli(), tables, box, beta, coulomb,
                                  k1, k2, k3});
  func::CallOp convolve;
  if (op.getEnergy().use_empty() && op.getVirial().use_empty())
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_scale" + kind),
                         ValueRange{complex, tables, box, beta, coulomb, k1,
                                    k2, k3});
  else
    convolve = func::CallOp::create(
        builder, loc, instance("mdrt_gpu_pme_convolve" + kind),
        ValueRange{complex, tables, rows, box, beta, coulomb, k1, k2, k3});
  func::CallOp::create(builder, loc, backward,
                       ValueRange{complex, real, sizes[0], sizes[1], sizes[2]});
  // The gathering takes the places of the particles from the weights
  // where it has them (D102). With a list, it takes the listed particles,
  // or with `gather_all` every particle, as without a list: there most
  // particles are listed, and the others share the points of the grid of
  // their neighbors in a warp (D262).
  if (op.getGatherAll())
    listed = Value();
  if (listed && usesWeights)
    func::CallOp::create(builder, loc,
                         instance("mdrt_gpu_pme_gather_weights_listed"),
                         ValueRange{positions, charges, listed, real, box,
                                    weights, k1, k2, k3, forces});
  else if (listed)
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_gather_listed"),
                         ValueRange{positions, charges, listed, real, box, k1,
                                    k2, k3, order, forces});
  else if (usesWeights)
    func::CallOp::create(builder, loc, instance("mdrt_gpu_pme_gather_weights"),
                         ValueRange{positions, charges, real, box, weights,
                                    k1, k2, k3, forces});
  else
    func::CallOp::create(builder, loc,
                         instance(op.getPotential() ? "mdrt_gpu_pme_potential"
                                                    : "mdrt_gpu_pme_gather"),
                         ValueRange{positions, charges, real, box, k1, k2, k3,
                                    order, forces});
  if (side) {
    auto end = func::CallOp::create(
        builder, loc,
        getOrDeclare("mdrtSideEnd", builder.getFunctionType({}, {})),
        ValueRange());
    for (Operation *issued = begin->getNextNode(); issued != end;
         issued = issued->getNextNode())
      if (allocates(issued))
        return op->emitOpError()
               << "allocates or frees memory on a second stream";
  }
  if (convolve) {
    op.getEnergy().replaceAllUsesWith(convolve.getResult(0));
    op.getVirial().replaceAllUsesWith(convolve.getResult(1));
  }
  return success();
}

/// The functions of the runtime without a body here that allocate and free
/// nothing with the allocator of the runtime: the transforms of cuFFT,
/// whose plans hold their work areas (runtime/mdrt_cuda.c).
static bool isAuditedNotToAllocate(StringRef name) {
  return name == "mdrtCudaFFTForward3D" || name == "mdrtCudaFFTBackward3D" ||
         name == "mdrtCudaFFTForward3DF32" ||
         name == "mdrtCudaFFTBackward3DF32";
}

bool Lowering::allocates(Operation *op) {
  DenseSet<Operation *> seen;
  SmallVector<Operation *> pending = {op};
  while (!pending.empty()) {
    Operation *next = pending.pop_back_val();
    if (!seen.insert(next).second)
      continue;
    WalkResult result = next->walk([&](Operation *nested) {
      // Memory of the host comes from malloc, which orders nothing on the
      // device.
      if (isa<gpu::AllocOp, gpu::DeallocOp>(nested))
        return WalkResult::interrupt();
      if (auto alloc = dyn_cast<memref::AllocOp>(nested))
        if (alloc.getType().getMemorySpace())
          return WalkResult::interrupt();
      if (auto dealloc = dyn_cast<memref::DeallocOp>(nested))
        if (dealloc.getMemref().getType().getMemorySpace())
          return WalkResult::interrupt();
      auto call = dyn_cast<func::CallOp>(nested);
      if (!call)
        return WalkResult::advance();
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          module, call.getCalleeAttr());
      if (!callee)
        return WalkResult::interrupt();
      if (callee.isExternal())
        return isAuditedNotToAllocate(callee.getName())
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      pending.push_back(callee);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return true;
  }
  return false;
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

/// Whether the structure `value` reaches `md_exec.build_edges`, through
/// refreshes and the arguments and results of loops.
static bool feedsEdges(Value value) {
  SmallVector<Value> pending = {value};
  llvm::DenseSet<Value> seen;
  while (!pending.empty()) {
    Value next = pending.pop_back_val();
    if (!seen.insert(next).second)
      continue;
    for (OpOperand &use : next.getUses()) {
      Operation *user = use.getOwner();
      if (isa<md_exec::BuildEdgesOp>(user))
        return true;
      if (auto refresh = dyn_cast<md_exec::RefreshNeighborsOp>(user))
        pending.push_back(refresh.getResult());
      if (auto loop = dyn_cast<scf::ForOp>(user)) {
        if (use.getOperandNumber() >= loop.getNumControlOperands()) {
          unsigned index = use.getOperandNumber() - loop.getNumControlOperands();
          pending.push_back(loop.getRegionIterArgs()[index]);
          pending.push_back(loop.getResult(index));
        }
      }
      if (auto yield = dyn_cast<scf::YieldOp>(user))
        if (auto loop = dyn_cast<scf::ForOp>(yield->getParentOp())) {
          pending.push_back(loop.getResult(use.getOperandNumber()));
          pending.push_back(
              loop.getRegionIterArgs()[use.getOperandNumber()]);
        }
    }
  }
  return false;
}

void Lowering::lowerEmptyNeighbors(md_exec::EmptyNeighborsOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Type narrow = builder.getI32Type();
  auto positions = cast<MemRefType>(*op.getPositions());

  Neighbors structure;
  structure.size = op.getSize();
  structure.width = createIndex(builder, loc, op.getWidth());
  if (op.getKind() == md_exec::NeighborKind::Groups) {
    // Groups of 16 (D89). The runtime holds the buffers, sized from an
    // estimate, and makes them larger when a build finds them too small
    // (emitGroupsBuild). The compact order leaves at most 63 places of a
    // chunk empty, one chunk for each column at most. A list takes blocks
    // of 64 entries as it fills them; it holds the neighbors at later
    // places of the 16 particles of its group, on average half of their
    // union, about as many as a row of the matrix.
    structure.groups = true;
    Type wide = builder.getI64Type();
    // Places for a quarter more than the particles and a chunk; blocks for
    // as many entries as a row for each group and one block more.
    Value places = arith::AddIOp::create(
        builder, loc,
        arith::AddIOp::create(
            builder, loc, structure.size,
            arith::DivUIOp::create(builder, loc, structure.size,
                                   createIndex(builder, loc, 4))),
        createIndex(builder, loc, 64));
    Value filled = arith::AddIOp::create(
        builder, loc,
        arith::DivUIOp::create(builder, loc, structure.size,
                               createIndex(builder, loc, 16)),
        createIndex(builder, loc, 1));
    Value blocks = arith::MulIOp::create(
        builder, loc, filled,
        createIndex(builder, loc, (op.getWidth() + 63) / 64 + 1));
    structure.handle =
        func::CallOp::create(
            builder, loc,
            getOrDeclare(createGroupsName,
                         builder.getFunctionType({wide, wide}, {wide})),
            ValueRange{
                arith::IndexCastOp::create(builder, loc, wide, places),
                arith::IndexCastOp::create(builder, loc, wide, blocks)})
            .getResult(0);
    structure.placeOf = createDeviceBuffer(
        builder, loc, getDeviceType({ShapedType::kDynamic}, narrow),
        ValueRange{structure.size});
    // The places, the longest list, the blocks, the groups whose excluded
    // partners did not fit the memory of a warp (D106), and the positions
    // that are not numbers (D107).
    structure.sizes =
        memref::AllocOp::create(builder, loc, MemRefType::get({5}, narrow));
  } else {
    structure.counts = createDeviceBuffer(
        builder, loc, getDeviceType({ShapedType::kDynamic}, narrow),
        ValueRange{structure.size});
    // The runtime holds the rows, which grow wider when a build finds them
    // too narrow (emitBuild).
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
    structure.order = createDeviceBuffer(
        builder, loc, getDeviceType({ShapedType::kDynamic}, narrow),
        ValueRange{structure.size});
  }
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
      builder, loc, MemRefType::get({getCellSize()}, builder.getF64Type()));
  // The configuration and the cell of the last pruning of a dual list
  // (D114). The cell is not a number until then, so that the test of the
  // inner list fails before the first pruning.
  if (structure.groups) {
    structure.prunedReference = createDeviceBuffer(
        builder, loc, positions, ValueRange{structure.size});
    structure.prunedBox = memref::AllocOp::create(
        builder, loc, MemRefType::get({getCellSize()}, builder.getF64Type()));
    Value notNumber = arith::ConstantOp::create(
        builder, loc, builder.getF64Type(),
        builder.getF64FloatAttr(std::numeric_limits<double>::quiet_NaN()));
    for (int64_t c = 0; c < getCellSize(); ++c)
      memref::StoreOp::create(builder, loc, notNumber, structure.prunedBox,
                              ValueRange{createIndex(builder, loc, c)});
    Value pruned = structure.prunedReference;
    Type element = positions.getElementType();
    launchOver(builder, loc, structure.size, [&](OpBuilder &body,
                                                 Value particle) {
      Value zero = createZero(body, loc, VectorType::get({3}, element));
      storeElement(body, loc, zero, pruned, particle);
    });
    structure.prunedNow = memref::AllocOp::create(
        builder, loc, MemRefType::get({}, builder.getI1Type()));
    memref::StoreOp::create(
        builder, loc,
        arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                  builder.getBoolAttr(false)),
        structure.prunedNow, ValueRange{});
  }
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
  // The slots of the pairs of images, where a matrix gives them
  // (D277).
  if (!structure.groups && feedsEdges(op.getResult())) {
    structure.edges = true;
    structure.edgesHandle =
        func::CallOp::create(
            builder, loc,
            getOrDeclare(createEdgesName,
                         builder.getFunctionType({}, {builder.getI64Type()})),
            ValueRange())
            .getResult(0);
    Value more = arith::AddIOp::create(builder, loc, structure.size,
                                       createIndex(builder, loc, 1));
    structure.edgeCounts = createDeviceBuffer(
        builder, loc, getDeviceType({ShapedType::kDynamic}, narrow),
        ValueRange{structure.size});
    structure.edgeOffsets = createDeviceBuffer(
        builder, loc, getDeviceType({ShapedType::kDynamic}, narrow),
        ValueRange{more});
    structure.hostEdgeCounts = memref::AllocOp::create(
        builder, loc, MemRefType::get({ShapedType::kDynamic}, narrow),
        ValueRange{structure.size});
    structure.hostEdgeOffsets = memref::AllocOp::create(
        builder, loc, MemRefType::get({ShapedType::kDynamic}, narrow),
        ValueRange{more});
    // The copies on the host are freed where the function ends; the
    // buffers of the device go back to the pool with the activation.
    structure.edgeSerial = memref::AllocOp::create(
        builder, loc, MemRefType::get({}, builder.getI64Type()));
    memref::StoreOp::create(
        builder, loc,
        arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                  builder.getI64IntegerAttr(-1)),
        structure.edgeSerial, ValueRange{});
    freeAtEndOfBlock(op, structure.hostEdgeCounts);
    freeAtEndOfBlock(op, structure.hostEdgeOffsets);
    freeAtEndOfBlock(op, structure.edgeSerial);
  }
  structure.excluded = op.getExcluded();
  neighbors[op.getResult()] = structure;
}

void Lowering::emitStopNotNumbers(OpBuilder &builder, Location loc,
                                  Value notNumbers) {
  // A position that is not a number would send every particle into one
  // bin of the build, which sorts a bin in one thread: the builds give such
  // a position no place and count it, and the run, which has failed by
  // then, stops with a word (D107).
  Type wide = builder.getI64Type();
  Value failed = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, notNumbers,
      arith::ConstantOp::create(builder, loc, wide,
                                builder.getI64IntegerAttr(0)));
  scf::IfOp::create(builder, loc, failed, [&](OpBuilder &then, Location) {
    func::CallOp::create(
        then, loc,
        getOrDeclare("mdrtStopNotNumbers", then.getFunctionType({wide}, {})),
        ValueRange{notNumbers});
    scf::YieldOp::create(then, loc);
  });
}

LogicalResult Lowering::emitBuild(OpBuilder &builder, Location loc,
                                  const Neighbors &structure, Value positions,
                                  Value box, double reach,
                                  double cellWidth) {
  if (structure.groups) {
    if (failed(emitGroupsBuild(builder, loc, structure, positions, box,
                               reach)))
      return failure();
    // The block of each group's list by its number, for the prunings of a
    // dual list until the next build (D268); a build by a
    // refresh that does not prune can precede the first pruning.
    emitGroupsBlockIndex(builder, loc, structure);
    return finishBuild(builder, loc, structure, positions, box);
  }
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  if (failed(addTemplates(real)))
    return failure();

  Value reachValue = createReal(builder, loc, real, reach);
  Value leastValue = createReal(builder, loc, real, cellWidth);
  Value boxValue = convertReal(builder, loc, box, real);
  // A triclinic cell is searched in its fractional coordinates, whose cells
  // have the widths of the cell between its faces.
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

  // The search enters the excluded pairs as the particle itself; without
  // them it takes a buffer with no rows.
  Value excluded = structure.excluded;
  Value noExcluded;
  if (!excluded) {
    MemRefType type = getDeviceType(
        {ShapedType::kDynamic, ShapedType::kDynamic}, builder.getI32Type());
    noExcluded = createDeviceBuffer(
        builder, loc, type,
        ValueRange{createIndex(builder, loc, 0), createIndex(builder, loc, 1)});
    excluded = noExcluded;
  }
  auto build = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module,
      getInstanceName(tilted ? std::string(buildNeighborsName) + "_triclinic"
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
    SmallVector<Value, 9> operands = {positions, boxValue};
    if (tilted)
      operands.push_back(widthsValue);
    operands.append({reachValue, widthValue, excluded, structure.counts,
                     entries, structure.order});
    auto built = func::CallOp::create(at, loc, build, operands);
    Value largest = built.getResult(0);
    emitStopNotNumbers(
        at, loc, arith::IndexCastOp::create(at, loc, wide, built.getResult(1)));
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

  if (noExcluded) {
    Value plain = memref::MemorySpaceCastOp::create(
        builder, loc,
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                        builder.getI32Type()),
        noExcluded);
    gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange(), plain);
  }

  if (failed(finishBuild(builder, loc, structure, positions, box)))
    return failure();
  if (structure.edges)
    emitEdgeSlots(builder, loc, structure, positions, box, reach);
  return success();
}

Value Lowering::getMatrixEntries(OpBuilder &builder, Location loc,
                                Value handle) {
  MemRefType type = getDeviceType(
      {ShapedType::kDynamic, ShapedType::kDynamic}, builder.getI32Type());
  func::FuncOp getter = getOrDeclare(
      matrixEntriesName,
      builder.getFunctionType({builder.getI64Type()}, {type}));
  getter->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  return func::CallOp::create(builder, loc, getter, ValueRange{handle})
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// The pairs of images of a learned potential (D277)
//===----------------------------------------------------------------------===//

Value Lowering::getEdgesBuffer(OpBuilder &builder, Location loc, Value handle,
                               int64_t which) {
  Type wide = builder.getI64Type();
  Type narrow = builder.getI32Type();
  MemRefType any =
      getDeviceType({ShapedType::kDynamic, ShapedType::kDynamic}, narrow);
  func::FuncOp getter = getOrDeclare(
      edgesBufferName, builder.getFunctionType({wide, wide}, {any}));
  getter->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  Value buffer =
      func::CallOp::create(
          builder, loc, getter,
          ValueRange{handle, arith::ConstantOp::create(
                                 builder, loc, wide,
                                 builder.getI64IntegerAttr(which))})
          .getResult(0);
  return memref::CastOp::create(
      builder, loc,
      getDeviceType({ShapedType::kDynamic, which == 0 ? 4 : 5}, narrow),
      buffer);
}

/// Visits the images `s` of the displacement `d` (three f64) in the cell
/// `box` (a vector of three edges, or of six for a triclinic cell) whose
/// displacement d + s H has each component within `reach`: the s_c of
/// [(-d_z - reach) / c_z, (-d_z + reach) / c_z], for each the s_b of the
/// same range of the y component over b_y, and for each the s_a of that of
/// the x component over a_x, as the lower-triangular cell orders them
/// (docs/triclinic-m2.md, Section 2). `visit` takes the image (three i32),
/// the squared length of d + s H, and the count carried, and returns the
/// count carried on. A range is at most 1024 a side, so that a cell or a
/// displacement that is not a number gives loops that end (#168).
static Value forImages(
    OpBuilder &builder, Location loc, Value d, Value box, double reach,
    Value carried,
    function_ref<Value(OpBuilder &, Value, Value, Value, Value, Value)>
        visit) {
  Type real = builder.getF64Type();
  Type narrow = builder.getI32Type();
  auto constant = [&](OpBuilder &b, double value) -> Value {
    return arith::ConstantOp::create(b, loc, real, b.getF64FloatAttr(value));
  };
  auto element = [&](OpBuilder &b, Value vector, int64_t k) -> Value {
    return vector::ExtractOp::create(b, loc, vector, k);
  };
  auto mul = [&](OpBuilder &b, Value x, Value y) -> Value {
    return arith::MulFOp::create(b, loc, x, y);
  };
  auto add = [&](OpBuilder &b, Value x, Value y) -> Value {
    return arith::AddFOp::create(b, loc, x, y);
  };
  bool tilted = isTriclinic(box);
  Value ax = element(builder, box, 0), by = element(builder, box, 1),
        cz = element(builder, box, 2);
  Value bx = tilted ? element(builder, box, 3) : constant(builder, 0.0);
  Value cx = tilted ? element(builder, box, 4) : constant(builder, 0.0);
  Value cy = tilted ? element(builder, box, 5) : constant(builder, 0.0);
  Value dx = element(builder, d, 0), dy = element(builder, d, 1),
        dz = element(builder, d, 2);
  Value one = arith::ConstantOp::create(builder, loc, narrow,
                                        builder.getI32IntegerAttr(1));
  // The whole numbers k with |u + k L| <= reach, as the range [low, high).
  auto range = [&](OpBuilder &b, Value u, Value length) {
    Value r = constant(b, reach);
    Value most = constant(b, 1024.0), least = constant(b, -1024.0);
    Value minus = arith::NegFOp::create(b, loc, u);
    Value low = math::CeilOp::create(
        b, loc,
        arith::DivFOp::create(b, loc, arith::SubFOp::create(b, loc, minus, r),
                              length));
    Value high = math::FloorOp::create(
        b, loc,
        arith::DivFOp::create(b, loc, arith::AddFOp::create(b, loc, minus, r),
                              length));
    low = arith::MinNumFOp::create(
        b, loc, arith::MaxNumFOp::create(b, loc, low, least), most);
    high = arith::MinNumFOp::create(
        b, loc, arith::MaxNumFOp::create(b, loc, high, least), most);
    Value lowWhole = arith::FPToSIOp::create(b, loc, narrow, low);
    Value highWhole = arith::AddIOp::create(
        b, loc, arith::FPToSIOp::create(b, loc, narrow, high), one);
    return std::make_pair(lowWhole, highWhole);
  };
  auto real64 = [&](OpBuilder &b, Value whole) -> Value {
    return arith::SIToFPOp::create(b, loc, real, whole);
  };
  auto [lowC, highC] = range(builder, dz, cz);
  auto overC = scf::ForOp::create(
      builder, loc, lowC, highC, one, ValueRange{carried},
      [&](OpBuilder &bc, Location, Value sc, ValueRange carriedC) {
        Value fc = real64(bc, sc);
        Value z = add(bc, dz, mul(bc, fc, cz));
        Value y1 = add(bc, dy, mul(bc, fc, cy));
        Value x1 = add(bc, dx, mul(bc, fc, cx));
        Value zz = mul(bc, z, z);
        auto [lowB, highB] = range(bc, y1, by);
        auto overB = scf::ForOp::create(
            bc, loc, lowB, highB, one, carriedC,
            [&](OpBuilder &bb, Location, Value sb, ValueRange carriedB) {
              Value fb = real64(bb, sb);
              Value y = add(bb, y1, mul(bb, fb, by));
              Value x2 = add(bb, x1, mul(bb, fb, bx));
              Value zy = add(bb, zz, mul(bb, y, y));
              auto [lowA, highA] = range(bb, x2, ax);
              auto overA = scf::ForOp::create(
                  bb, loc, lowA, highA, one, carriedB,
                  [&](OpBuilder &ba, Location, Value sa,
                      ValueRange carriedA) {
                    Value x = add(ba, x2, mul(ba, real64(ba, sa), ax));
                    Value r2 = add(ba, zy, mul(ba, x, x));
                    scf::YieldOp::create(
                        ba, loc,
                        visit(ba, sa, sb, sc, r2, carriedA.front()));
                  });
              scf::YieldOp::create(bb, loc, overA.getResults());
            });
        scf::YieldOp::create(bc, loc, overB.getResults());
      });
  return overC.getResult(0);
}

/// For the particle at `place` of the last build of a matrix, its
/// candidates: every image `s` of each particle `j` of its row, and of the
/// particle `i` itself but the image 0, with |x_j - x_i + s H| < `member`
/// at `positions` (f64) in the cell `box`, in the order of the row, then of
/// the images (forImages). `taken` takes `i`, `j` (i32), the image, and the
/// count of the candidates before, and is emitted where the candidate is
/// taken. Returns `i` (index) and the count of the candidates, which are
/// meaningful only where `placed` (also returned) holds: a place that the
/// build left empty (D107) has none.
struct CandidateWalk {
  Value placed, particle, count;
};
static CandidateWalk forCandidates(
    OpBuilder &builder, Location loc, Value order, Value counts,
    Value entries, Value positions, Value box, double member, Value place,
    function_ref<void(OpBuilder &, Value, Value, Value, Value, Value, Value)>
        taken) {
  Type narrow = builder.getI32Type();
  Type index = builder.getIndexType();
  auto whole = [&](OpBuilder &b, int64_t value) -> Value {
    return arith::ConstantOp::create(b, loc, narrow,
                                     b.getI32IntegerAttr(value));
  };
  Value at = memref::LoadOp::create(builder, loc, order, ValueRange{place});
  Value zero32 = whole(builder, 0);
  CandidateWalk walk;
  walk.placed = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                      at, zero32);
  Value safeAt = arith::SelectOp::create(builder, loc, walk.placed, at, zero32);
  Value i = arith::IndexCastOp::create(builder, loc, index, safeAt);
  walk.particle = i;
  Value xi = loadElement(builder, loc, positions, i);
  // An empty place visits nothing.
  Value row = arith::IndexCastOp::create(
      builder, loc, index,
      memref::LoadOp::create(builder, loc, counts, ValueRange{place}));
  Value total = arith::SelectOp::create(
      builder, loc, walk.placed,
      arith::AddIOp::create(builder, loc, row, createIndex(builder, loc, 1)),
      createIndex(builder, loc, 0));
  Value self = arith::IndexCastOp::create(builder, loc, narrow, place);
  Value limit2 = createReal(builder, loc, builder.getF64Type(),
                            member * member);
  auto loop = scf::ForOp::create(
      builder, loc, createIndex(builder, loc, 0), total,
      createIndex(builder, loc, 1), ValueRange{zero32},
      [&](OpBuilder &b, Location, Value k, ValueRange found) {
        // The entry k of the row, or the place itself after the row.
        Value inRow =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, k, row);
        Value last = arith::SubIOp::create(
            b, loc, arith::MaxUIOp::create(b, loc, row, createIndex(b, loc, 1)),
            createIndex(b, loc, 1));
        Value safe = arith::MinUIOp::create(b, loc, k, last);
        Value entry =
            memref::LoadOp::create(b, loc, entries, ValueRange{place, safe});
        Value q = arith::SelectOp::create(b, loc, inRow, entry, self);
        // A row enters an excluded pair as the place itself; the
        // neighborhood of the edges excludes none, and the particle's own
        // images come after the row, once.
        Value repeated = arith::AndIOp::create(
            b, loc, inRow,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, q, self));
        auto skip = scf::IfOp::create(b, loc, TypeRange{narrow}, repeated,
                                      /*withElseRegion=*/true);
        {
          OpBuilder same = skip.getThenBodyBuilder();
          scf::YieldOp::create(same, loc, found.front());
        }
        {
          OpBuilder other = skip.getElseBodyBuilder();
          Value jValue = memref::LoadOp::create(
              other, loc, order,
              ValueRange{arith::IndexCastOp::create(other, loc, index, q)});
          Value j = arith::IndexCastOp::create(other, loc, index, jValue);
          Value d = arith::SubFOp::create(
              other, loc, loadElement(other, loc, positions, j), xi);
          Value isSelf = arith::CmpIOp::create(
              other, loc, arith::CmpIPredicate::eq, j, i);
          Value counted = forImages(
              other, loc, d, box, member, found.front(),
              [&](OpBuilder &v, Value sa, Value sb, Value sc, Value r2,
                  Value carried) -> Value {
                auto isZero = [&](Value s) -> Value {
                  return arith::CmpIOp::create(v, loc,
                                               arith::CmpIPredicate::eq, s,
                                               whole(v, 0));
                };
                Value zeroShift = arith::AndIOp::create(
                    v, loc, arith::AndIOp::create(v, loc, isZero(sa),
                                                  isZero(sb)),
                    isZero(sc));
                Value itself =
                    arith::AndIOp::create(v, loc, isSelf, zeroShift);
                Value within = arith::CmpFOp::create(
                    v, loc, arith::CmpFPredicate::OLT, r2, limit2);
                Value notItself = arith::XOrIOp::create(
                    v, loc, itself,
                    arith::ConstantOp::create(v, loc, v.getI1Type(),
                                              v.getBoolAttr(true)));
                Value take = arith::AndIOp::create(v, loc, within, notItself);
                auto when = scf::IfOp::create(v, loc, TypeRange{narrow}, take,
                                              /*withElseRegion=*/true);
                {
                  OpBuilder then = when.getThenBodyBuilder();
                  taken(then, i, jValue, sa, sb, sc, carried);
                  scf::YieldOp::create(
                      then, loc,
                      arith::AddIOp::create(then, loc, carried,
                                            whole(then, 1))
                          .getResult());
                }
                {
                  OpBuilder otherwise = when.getElseBodyBuilder();
                  scf::YieldOp::create(otherwise, loc, carried);
                }
                return when.getResult(0);
              });
          scf::YieldOp::create(other, loc, counted);
        }
        scf::YieldOp::create(b, loc, skip.getResult(0));
      });
  walk.count = loop.getResult(0);
  return walk;
}

void Lowering::emitEdgeSlots(OpBuilder &builder, Location loc,
                             const Neighbors &structure, Value positions,
                             Value box, double reach) {
  Type narrow = builder.getI32Type();
  Type wide = builder.getI64Type();
  // A candidate is an image within the reach at the build, with a margin
  // for the rounding of the test of the refresh: a pair within the cutoff
  // at a configuration for which the structure is valid was within the
  // reach here (`md_exec.refresh_neighbors`, D80).
  double member = reach * (1.0 + 1e-6);
  Value order = structure.order, counts = structure.counts;
  Value entries = getMatrixEntries(builder, loc, structure.handle);
  Value edgeCounts = structure.edgeCounts;
  Value edgeOffsets = structure.edgeOffsets;
  Value hostCounts = structure.hostEdgeCounts;
  Value hostOffsets = structure.hostEdgeOffsets;
  auto none = [](OpBuilder &, Value, Value, Value, Value, Value, Value) {};

  // The candidates of each particle; a particle that the build gave no
  // place (D107), which stops the run, has none.
  launchOver(builder, loc, structure.size, [&](OpBuilder &body, Value i) {
    memref::StoreOp::create(
        body, loc,
        arith::ConstantOp::create(body, loc, narrow, body.getI32IntegerAttr(0)),
        edgeCounts, ValueRange{i});
  });
  launchOver(builder, loc, structure.size, [&](OpBuilder &body, Value place) {
    CandidateWalk walk = forCandidates(body, loc, order, counts, entries,
                                       positions, box, member, place, none);
    scf::IfOp::create(body, loc, walk.placed, [&](OpBuilder &then, Location) {
      memref::StoreOp::create(then, loc, walk.count, edgeCounts,
                              ValueRange{walk.particle});
      scf::YieldOp::create(then, loc);
    });
  });

  // Their offsets, on the host: the build has waited for the device, and
  // waits for this copy. A count beyond int32 stops the run.
  createTransfer(builder, loc, hostCounts, edgeCounts);
  Value zero64 =
      arith::ConstantOp::create(builder, loc, wide, builder.getI64IntegerAttr(0));
  memref::StoreOp::create(
      builder, loc, arith::TruncIOp::create(builder, loc, narrow, zero64),
      hostOffsets, ValueRange{createIndex(builder, loc, 0)});
  auto sums = scf::ForOp::create(
      builder, loc, createIndex(builder, loc, 0), structure.size,
      createIndex(builder, loc, 1), ValueRange{zero64},
      [&](OpBuilder &b, Location, Value i, ValueRange sum) {
        Value count = arith::ExtSIOp::create(
            b, loc, wide, memref::LoadOp::create(b, loc, hostCounts, i));
        Value next = arith::AddIOp::create(b, loc, sum.front(), count);
        Value after = arith::AddIOp::create(b, loc, i, createIndex(b, loc, 1));
        memref::StoreOp::create(
            b, loc, arith::TruncIOp::create(b, loc, narrow, next), hostOffsets,
            ValueRange{after});
        scf::YieldOp::create(b, loc, next);
      });
  Value length = sums.getResult(0);
  Value tooMany = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sgt, length,
      arith::ConstantOp::create(builder, loc, wide,
                                builder.getI64IntegerAttr(INT32_MAX)));
  // The runtime stops at a negative length.
  Value reserved = arith::SelectOp::create(
      builder, loc, tooMany,
      arith::ConstantOp::create(builder, loc, wide,
                                builder.getI64IntegerAttr(-1)),
      length);
  Value serial =
      func::CallOp::create(
          builder, loc,
          getOrDeclare(reserveEdgesName,
                       builder.getFunctionType({wide, wide}, {wide})),
          ValueRange{structure.edgesHandle, reserved})
          .getResult(0);
  memref::StoreOp::create(builder, loc, serial, structure.edgeSerial,
                          ValueRange{});
  createTransfer(builder, loc, edgeOffsets, hostOffsets);

  // The candidates in their slots, in the same order.
  Value candidates = getEdgesBuffer(builder, loc, structure.edgesHandle, 0);
  launchOver(builder, loc, structure.size, [&](OpBuilder &body, Value place) {
    Value at = memref::LoadOp::create(body, loc, order, ValueRange{place});
    Value placed = arith::CmpIOp::create(
        body, loc, arith::CmpIPredicate::sge, at,
        arith::ConstantOp::create(body, loc, narrow,
                                  body.getI32IntegerAttr(0)));
    Value safeAt = arith::SelectOp::create(
        body, loc, placed, at,
        arith::ConstantOp::create(body, loc, narrow,
                                  body.getI32IntegerAttr(0)));
    Value i = arith::IndexCastOp::create(body, loc, body.getIndexType(),
                                         safeAt);
    Value begin = memref::LoadOp::create(body, loc, edgeOffsets, ValueRange{i});
    Value end = memref::LoadOp::create(
        body, loc, edgeOffsets,
        ValueRange{arith::AddIOp::create(body, loc, i,
                                         createIndex(body, loc, 1))});
    Value room = arith::SubIOp::create(body, loc, end, begin);
    forCandidates(
        body, loc, order, counts, entries, positions, box, member, place,
        [&](OpBuilder &b, Value, Value j, Value sa, Value sb, Value sc,
            Value before) {
          // The same candidates as counted; a slot beyond the particle's
          // would be another's, and is not written.
          Value fits = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                             before, room);
          scf::IfOp::create(b, loc, fits, [&](OpBuilder &then, Location) {
            Value slot = arith::IndexCastOp::create(
                then, loc, then.getIndexType(),
                arith::AddIOp::create(then, loc, begin, before));
            Value values[4] = {j, sa, sb, sc};
            for (int64_t c = 0; c != 4; ++c)
              memref::StoreOp::create(then, loc, values[c], candidates,
                                      ValueRange{slot, createIndex(then, loc,
                                                                   c)});
            scf::YieldOp::create(then, loc);
          });
        });
  });
}

LogicalResult Lowering::lowerBuildEdges(md_exec::BuildEdgesOp op) {
  if (!op.isStorageForm())
    return op.emitOpError() << "is not in the storage form; run "
                               "'md-exec-assign-storage' first";
  Neighbors structure;
  if (failed(getNeighbors(op, op.getNeighbors(), structure)))
    return failure();
  if (!structure.edges)
    return op.emitOpError() << "the neighbor structure is not a matrix that "
                               "keeps the slots of the pairs of images";
  Location loc = op.getLoc();
  OpBuilder builder(op);
  setPurpose(op);
  Type narrow = builder.getI32Type();
  Value positions = op.getPositions();
  Value box = op.getCellMutable().get();
  Value candidates = getEdgesBuffer(builder, loc, structure.edgesHandle, 0);
  Value slots = getEdgesBuffer(builder, loc, structure.edgesHandle, 1);
  Value offsets = structure.edgeOffsets, counts = structure.edgeCounts;
  double cutoff = op.getCutoff().convertToDouble();

  // Each particle's candidates, those within the cutoff now first, in the
  // order of the candidates, then the others from the end of its slots.
  launchOver(builder, loc, structure.size, [&](OpBuilder &body, Value i) {
    Value i32 = arith::IndexCastOp::create(body, loc, narrow, i);
    Value begin = arith::IndexCastOp::create(
        body, loc, body.getIndexType(),
        memref::LoadOp::create(body, loc, offsets, ValueRange{i}));
    Value end = arith::IndexCastOp::create(
        body, loc, body.getIndexType(),
        memref::LoadOp::create(
            body, loc, offsets,
            ValueRange{arith::AddIOp::create(body, loc, i,
                                             createIndex(body, loc, 1))}));
    Value xi = loadElement(body, loc, positions, i);
    Value limit2 =
        createReal(body, loc, body.getF64Type(), cutoff * cutoff);
    auto loop = scf::ForOp::create(
        body, loc, begin, end, createIndex(body, loc, 1),
        ValueRange{begin, end},
        [&](OpBuilder &b, Location, Value k, ValueRange ends) {
          Value row[4];
          for (int64_t c = 0; c != 4; ++c)
            row[c] = memref::LoadOp::create(
                b, loc, candidates, ValueRange{k, createIndex(b, loc, c)});
          Value j = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                               row[0]);
          SmallVector<Value, 3> cells;
          for (int64_t c = 1; c != 4; ++c)
            cells.push_back(
                arith::SIToFPOp::create(b, loc, b.getF64Type(), row[c]));
          Value shift = emitLatticeShift(
              b, loc,
              vector::FromElementsOp::create(
                  b, loc, VectorType::get({3}, b.getF64Type()), cells),
              box);
          Value d = arith::AddFOp::create(
              b, loc,
              arith::SubFOp::create(b, loc, loadElement(b, loc, positions, j),
                                    xi),
              shift);
          Value d2 = arith::MulFOp::create(b, loc, d, d);
          Value r2 = vector::ReductionOp::create(
              b, loc, vector::CombiningKind::ADD, d2);
          Value within = arith::CmpFOp::create(
              b, loc, arith::CmpFPredicate::OLT, r2, limit2);
          Value back =
              arith::SubIOp::create(b, loc, ends[1], createIndex(b, loc, 1));
          Value slot = arith::SelectOp::create(b, loc, within, ends[0], back);
          Value values[5] = {i32, row[0], row[1], row[2], row[3]};
          for (int64_t c = 0; c != 5; ++c)
            memref::StoreOp::create(b, loc, values[c], slots,
                                    ValueRange{slot, createIndex(b, loc, c)});
          Value front = arith::SelectOp::create(
              b, loc, within,
              arith::AddIOp::create(b, loc, ends[0], createIndex(b, loc, 1)),
              ends[0]);
          Value rear = arith::SelectOp::create(b, loc, within, ends[1], back);
          scf::YieldOp::create(b, loc, ValueRange{front, rear});
        });
    Value strict = arith::SubIOp::create(body, loc, loop.getResult(0), begin);
    memref::StoreOp::create(
        body, loc, arith::IndexCastOp::create(body, loc, narrow, strict),
        counts, ValueRange{i});
  });

  op.getResult(0).replaceAllUsesWith(slots);
  op.getResult(1).replaceAllUsesWith(counts);
  op.getResult(2).replaceAllUsesWith(offsets);
  op.getResult(3).replaceAllUsesWith(
      memref::LoadOp::create(builder, loc, structure.edgeSerial, ValueRange{}));
  return success();
}

GroupBuffers Lowering::getGroupsBuffers(OpBuilder &builder, Location loc,
                                       Value handle) {
  Type wide = builder.getI64Type();
  MemRefType type =
      getDeviceType({ShapedType::kDynamic}, builder.getI32Type());
  func::FuncOp getter = getOrDeclare(
      groupsBufferName, builder.getFunctionType({wide, wide}, {type}));
  getter->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto get = [&](int64_t which) -> Value {
    return func::CallOp::create(
               builder, loc, getter,
               ValueRange{handle, arith::ConstantOp::create(
                                      builder, loc, wide,
                                      builder.getI64IntegerAttr(which))})
        .getResult(0);
  };
  // In the order of the runtime (mdrt_cuda.c).
  GroupBuffers buffers;
  buffers.order = get(0);
  buffers.counts = get(1);
  buffers.entries = get(2);
  buffers.masks = get(3);
  buffers.units = get(4);
  buffers.ordinals = get(5);
  buffers.shifts = get(6);
  return buffers;
}

Value Lowering::getGroupsBuffer(OpBuilder &builder, Location loc,
                                Value handle, int64_t which) {
  Type wide = builder.getI64Type();
  MemRefType type =
      getDeviceType({ShapedType::kDynamic}, builder.getI32Type());
  func::FuncOp getter = getOrDeclare(
      groupsBufferName, builder.getFunctionType({wide, wide}, {type}));
  getter->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  return func::CallOp::create(
             builder, loc, getter,
             ValueRange{handle, arith::ConstantOp::create(
                                    builder, loc, wide,
                                    builder.getI64IntegerAttr(which))})
      .getResult(0);
}

InnerLists Lowering::getInnerLists(OpBuilder &builder, Location loc,
                                   Value handle) {
  // In the order of the runtime (mdrt_cuda.c).
  return {getGroupsBuffer(builder, loc, handle, 7),
          getGroupsBuffer(builder, loc, handle, 8),
          getGroupsBuffer(builder, loc, handle, 9)};
}

PruneBuffers Lowering::getPruneBuffers(OpBuilder &builder, Location loc,
                                       Value handle) {
  // In the order of the runtime (mdrt_cuda.c).
  PruneBuffers buffers;
  buffers.blockOf = getGroupsBuffer(builder, loc, handle, 10);
  buffers.activeUnits = getGroupsBuffer(builder, loc, handle, 11);
  buffers.activeCount = getGroupsBuffer(builder, loc, handle, 12);
  buffers.runSums = getGroupsBuffer(builder, loc, handle, 13);
  buffers.wideSums = getGroupsBuffer(builder, loc, handle, 14);
  return buffers;
}

Value Lowering::getGroupsPlaced(OpBuilder &builder, Location loc,
                                Value handle) {
  Type wide = builder.getI64Type();
  MemRefType type =
      getDeviceType({ShapedType::kDynamic, 3}, builder.getF32Type());
  func::FuncOp getter = getOrDeclare(
      placedGroupsName, builder.getFunctionType({wide}, {type}));
  getter->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  return func::CallOp::create(builder, loc, getter, ValueRange{handle})
      .getResult(0);
}

std::pair<Value, Value> Lowering::getGroupsShape(OpBuilder &builder,
                                                 Location loc,
                                                 const Neighbors &structure) {
  auto size = [&](int64_t which) -> Value {
    return arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(),
        memref::LoadOp::create(builder, loc, structure.sizes,
                               ValueRange{createIndex(builder, loc, which)}));
  };
  Value groups = arith::DivUIOp::create(builder, loc, size(0),
                                        createIndex(builder, loc, 16));
  // The blocks of a group at most: of its longest list.
  Value stride = arith::CeilDivUIOp::create(builder, loc, size(1),
                                            createIndex(builder, loc, 64));
  stride = arith::MaxUIOp::create(builder, loc, stride,
                                  createIndex(builder, loc, 1));
  return {groups, stride};
}

void Lowering::emitGroupsBlockIndex(OpBuilder &builder, Location loc,
                                    const Neighbors &structure) {
  Type i32 = builder.getI32Type();
  Type wide = builder.getI64Type();
  auto [groups, stride] = getGroupsShape(builder, loc, structure);
  Value blocks = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      memref::LoadOp::create(builder, loc, structure.sizes,
                             ValueRange{createIndex(builder, loc, 2)}));
  // A row of `stride` blocks for each group; the runtime makes room for
  // them before the buffer is taken.
  func::CallOp::create(
      builder, loc,
      getOrDeclare(reserveBlockOfName,
                   builder.getFunctionType({wide, wide}, {})),
      ValueRange{structure.handle,
                 arith::IndexCastOp::create(
                     builder, loc, wide,
                     arith::MulIOp::create(builder, loc, groups, stride))});
  GroupBuffers buffers = getGroupsBuffers(builder, loc, structure.handle);
  Value blockOf = getGroupsBuffer(builder, loc, structure.handle, 10);
  launchOver(builder, loc, blocks, [&](OpBuilder &b, Value block) {
    Value group = arith::IndexCastOp::create(
        b, loc, b.getIndexType(),
        memref::LoadOp::create(b, loc, buffers.units, ValueRange{block}));
    Value ordinal = arith::IndexCastOp::create(
        b, loc, b.getIndexType(),
        memref::LoadOp::create(b, loc, buffers.ordinals, ValueRange{block}));
    memref::StoreOp::create(
        b, loc, arith::IndexCastOp::create(b, loc, i32, block), blockOf,
        ValueRange{arith::AddIOp::create(
            b, loc, arith::MulIOp::create(b, loc, group, stride), ordinal)});
  });
}

void Lowering::emitFramePositions(
    OpBuilder &builder, Location loc, const GroupBuffers &buffers,
    Value places, Value positions, Value box, Value placed, Value reference,
    Value particles, ArrayRef<std::pair<Value, Value>> cleared) {
  Type i32 = builder.getI32Type();
  Type f64 = builder.getF64Type();
  auto vector3 = VectorType::get({3}, builder.getF32Type());
  auto constant32 = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v));
  };
  auto when = [&](OpBuilder &b, Value condition,
                  function_ref<void(OpBuilder &)> fill) {
    scf::IfOp::create(b, loc, condition, [&](OpBuilder &then, Location) {
      fill(then);
      scf::YieldOp::create(then, loc);
    });
  };
  Value count = places;
  if (reference)
    count = arith::MaxUIOp::create(builder, loc, count, particles);
  for (auto [buffer, length] : cleared)
    count = arith::MaxUIOp::create(builder, loc, count, length);
  launchOver(builder, loc, count, [&](OpBuilder &b, Value item) {
    // The positions as they are stored, particle by particle, in their
    // order: the configuration of a pruning (D268).
    if (reference)
      when(b,
           arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item,
                                 particles),
           [&](OpBuilder &c) {
             storeElement(c, loc, loadElement(c, loc, positions, item),
                          reference, item);
           });
    when(b,
         arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item,
                               places),
         [&](OpBuilder &c) {
           Value place = item;
           Value at = memref::LoadOp::create(c, loc, buffers.order,
                                             ValueRange{place});
           // An empty place takes the frame of the particle 0, as the loop
           // over pairs gives it, which no pair reads.
           Value empty = arith::CmpIOp::create(
               c, loc, arith::CmpIPredicate::slt, at, constant32(c, 0));
           Value particle = arith::SelectOp::create(
               c, loc, empty, createIndex(c, loc, 0),
               arith::IndexCastOp::create(c, loc, c.getIndexType(), at));
           Value packed = memref::LoadOp::create(c, loc, buffers.shifts,
                                                 ValueRange{place});
           SmallVector<Value> cells;
           for (int64_t k = 0; k != 3; ++k) {
             Value bits = arith::AndIOp::create(
                 c, loc,
                 arith::ShRUIOp::create(c, loc, packed, constant32(c, 10 * k)),
                 constant32(c, 1023));
             cells.push_back(arith::SIToFPOp::create(
                 c, loc, f64,
                 arith::SubIOp::create(c, loc, bits, constant32(c, 512))));
           }
           Value shift = emitLatticeShift(
               c, loc,
               vector::FromElementsOp::create(c, loc,
                                              VectorType::get({3}, f64), cells),
               box);
           storeElement(
               c, loc,
               arith::TruncFOp::create(
                   c, loc, vector3,
                   arith::AddFOp::create(
                       c, loc, loadElement(c, loc, positions, particle),
                       shift)),
               placed, place);
         });
    for (auto [buffer, length] : cleared)
      when(b,
           arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item,
                                 length),
           [&, buffer = buffer](OpBuilder &c) {
             memref::StoreOp::create(c, loc, constant32(c, 0), buffer,
                                     ValueRange{item});
           });
  });
}

/// Adds `value`, an i32, to element `index` of the buffer of the device
/// `buffer` of i32, atomically.
static void emitAtomicAddI32(OpBuilder &b, Location loc, Value buffer,
                             Value index, Value value) {
  Type wide = b.getI64Type();
  Value base = arith::IndexCastOp::create(
      b, loc, wide,
      memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buffer));
  Value address = arith::AddIOp::create(
      b, loc, base,
      arith::MulIOp::create(
          b, loc, arith::IndexCastOp::create(b, loc, wide, index),
          arith::ConstantOp::create(b, loc, wide, b.getI64IntegerAttr(4))));
  Value pointer = LLVM::IntToPtrOp::create(
      b, loc, LLVM::LLVMPointerType::get(b.getContext(), 1), address);
  LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::add, pointer, value,
                            LLVM::AtomicOrdering::monotonic,
                            StringRef("device"));
}

void Lowering::emitGroupsPrune(OpBuilder &builder, Location loc,
                               const Neighbors &structure, Value positions,
                               Value box, double reach) {
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  Type f64 = builder.getF64Type();
  Type index = builder.getIndexType();
  auto vector3 = VectorType::get({3}, f32);
  // Remember the cell of the pruning; the kernel of the frames below
  // remembers the configuration.
  for (int64_t c = 0; c < getCellSize(); ++c)
    memref::StoreOp::create(builder, loc,
                            vector::ExtractOp::create(builder, loc, box, c),
                            structure.prunedBox,
                            ValueRange{createIndex(builder, loc, c)});
  func::CallOp::create(
      builder, loc,
      getOrDeclare("mdrtCountPrune", builder.getFunctionType({}, {})),
      ValueRange());

  GroupBuffers buffers = getGroupsBuffers(builder, loc, structure.handle);
  InnerLists inner = getInnerLists(builder, loc, structure.handle);
  PruneBuffers prune = getPruneBuffers(builder, loc, structure.handle);
  Value placed = getGroupsPlaced(builder, loc, structure.handle);
  Value places = arith::IndexCastOp::create(
      builder, loc, index,
      memref::LoadOp::create(builder, loc, structure.sizes,
                             ValueRange{createIndex(builder, loc, 0)}));
  auto [groups, stride] = getGroupsShape(builder, loc, structure);
  // The reach widened as the build widens it: by 3e-6 of the sum of the
  // edges, far more than the rounding of the positions to f32 can move a
  // distance, so that every pair within the reach in f64 is kept.
  // For a triclinic cell, the diagonal and the magnitudes of the tilts.
  Value sum = vector::ReductionOp::create(
      builder, loc, vector::CombiningKind::ADD,
      isTriclinic(box) ? Value(math::AbsFOp::create(builder, loc, box)) : box);
  Value widened = arith::AddFOp::create(
      builder, loc, createReal(builder, loc, f64, reach),
      arith::MulFOp::create(builder, loc, sum,
                            createReal(builder, loc, f64, 3.0e-6)));
  Value reach2 = arith::TruncFOp::create(
      builder, loc, f32, arith::MulFOp::create(builder, loc, widened, widened));
  Value box32 = arith::TruncFOp::create(
      builder, loc,
      VectorType::get({cast<VectorType>(box.getType()).getNumElements()}, f32),
      box);
  auto constant32 = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v));
  };
  auto toIndex = [&](OpBuilder &b, Value v) -> Value {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
  };
  Value warp32 = createIndex(builder, loc, 32);
  // The blocks of the outer list, in runs of 32 and of 1024, for the active
  // units.
  Value blocks = arith::IndexCastOp::create(
      builder, loc, index,
      memref::LoadOp::create(builder, loc, structure.sizes,
                             ValueRange{createIndex(builder, loc, 2)}));
  Value runs = arith::CeilDivUIOp::create(builder, loc, blocks, warp32);
  Value wideRuns = arith::CeilDivUIOp::create(
      builder, loc, blocks, createIndex(builder, loc, 1024));

  // The positions of the places in the frames of their groups, in f32, as
  // the loop over pairs takes them (D95, D101), which the loop that follows
  // at the same positions takes as they are (lowerGroupPairFor); the
  // configuration of the pruning, in place of a copy; and the counts of the
  // active units of the runs, and of all, from zero: a build that placed no
  // particle (D107) leaves no block, and no kernel below counts them.
  emitFramePositions(builder, loc, buffers, places, positions, box, placed,
                     structure.prunedReference, structure.size,
                     {{prune.runSums, runs},
                      {prune.wideSums, wideRuns},
                      {prune.activeCount, createIndex(builder, loc, 1)}});

  // The blocks of each group in the order of its list, made at the build
  // (emitGroupsBlockIndex).
  Value blockOf = prune.blockOf;

  // A warp for each group: it takes the entries of the outer list in their
  // order, 32 at a time, and writes those with a pair within the reach to
  // the front of the blocks of the group, in their order (a ballot and a
  // count of the bits below each lane), so that the inner list fills its
  // blocks and adds no order of its own to that of the outer list.
  launchOver(
      builder, loc, arith::MulIOp::create(builder, loc, groups, warp32),
      [&](OpBuilder &b, Value thread) {
        Value group = arith::DivUIOp::create(b, loc, thread, warp32);
        Value lane = arith::IndexCastOp::create(
            b, loc, i32, arith::RemUIOp::create(b, loc, thread, warp32));
        Value count = memref::LoadOp::create(b, loc, buffers.counts,
                                             ValueRange{group});
        Value base = arith::MulIOp::create(b, loc, group, stride);
        // The particles of the group, in every lane.
        Value u = arith::AndIOp::create(b, loc, lane, constant32(b, 15));
        Value member = arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(b, loc, group, createIndex(b, loc, 16)),
            toIndex(b, u));
        Value mine = loadElement(b, loc, placed, member);
        Value width = constant32(b, 32);
        SmallVector<Value> members;
        for (int64_t p = 0; p != 16; ++p) {
          SmallVector<Value> parts;
          for (int64_t c = 0; c != 3; ++c)
            parts.push_back(
                gpu::ShuffleOp::create(
                    b, loc, vector::ExtractOp::create(b, loc, mine, c),
                    constant32(b, p), width, gpu::ShuffleMode::IDX)
                    .getShuffleResult());
          members.push_back(
              vector::FromElementsOp::create(b, loc, vector3, parts));
        }
        Value below = arith::SubIOp::create(
            b, loc, arith::ShLIOp::create(b, loc, constant32(b, 1), lane),
            constant32(b, 1));
        // The slot of entry `e` of the list of the group.
        auto slotOf = [&](OpBuilder &c, Value e) -> Value {
          Value block = memref::LoadOp::create(
              c, loc, blockOf,
              ValueRange{arith::AddIOp::create(
                  c, loc, base,
                  toIndex(c, arith::DivUIOp::create(c, loc, e,
                                                    constant32(c, 64))))});
          return arith::AddIOp::create(
              c, loc,
              arith::MulIOp::create(c, loc, toIndex(c, block),
                                    createIndex(c, loc, 64)),
              toIndex(c, arith::RemUIOp::create(c, loc, e,
                                                constant32(c, 64))));
        };
        auto rounds = scf::ForOp::create(
            b, loc, constant32(b, 0), count, constant32(b, 32),
            ValueRange{constant32(b, 0)},
            [&](OpBuilder &c, Location, Value first, ValueRange carried) {
              Value kept = carried[0];
              Value e = arith::AddIOp::create(c, loc, first, lane);
              Value has = arith::CmpIOp::create(
                  c, loc, arith::CmpIPredicate::slt, e, count);
              Value at = slotOf(
                  c, arith::SelectOp::create(c, loc, has, e, first));
              Value entry = memref::LoadOp::create(c, loc, buffers.entries,
                                                   ValueRange{at});
              Value mask = arith::SelectOp::create(
                  c, loc, has,
                  memref::LoadOp::create(c, loc, buffers.masks,
                                         ValueRange{at}),
                  constant32(c, 0));
              // The entry in the frame of the group: moved by the whole
              // cells, or lattice vectors, of the shift in its mask (D95,
              // D115).
              Value position = loadElement(
                  c, loc, placed,
                  toIndex(c, arith::SelectOp::create(c, loc, has, entry,
                                                     constant32(c, 0))));
              int64_t width = getEntryShiftBits(isTriclinic(box));
              int64_t offset = getEntryShiftOffset(isTriclinic(box));
              SmallVector<Value> cells;
              for (int64_t axis = 0; axis != 3; ++axis) {
                Value bits = arith::AndIOp::create(
                    c, loc,
                    arith::ShRUIOp::create(c, loc, mask,
                                           constant32(c, 16 + width * axis)),
                    constant32(c, (1 << width) - 1));
                cells.push_back(arith::SIToFPOp::create(
                    c, loc, f32,
                    arith::SubIOp::create(c, loc, bits,
                                          constant32(c, offset))));
              }
              position = arith::AddFOp::create(
                  c, loc, position,
                  emitLatticeShift(
                      c, loc,
                      vector::FromElementsOp::create(c, loc, vector3, cells),
                      box32));
              // The bits of the pairs still within the reach.
              Value kept16 = constant32(c, 0);
              for (int64_t p = 0; p != 16; ++p) {
                Value d = arith::SubFOp::create(c, loc, members[p], position);
                Value r2 = vector::ReductionOp::create(
                    c, loc, vector::CombiningKind::ADD,
                    arith::MulFOp::create(c, loc, d, d));
                Value within = arith::CmpFOp::create(
                    c, loc, arith::CmpFPredicate::OLE, r2, reach2);
                Value bit = constant32(c, 1 << p);
                Value had = arith::CmpIOp::create(
                    c, loc, arith::CmpIPredicate::ne,
                    arith::AndIOp::create(c, loc, mask, bit),
                    constant32(c, 0));
                kept16 = arith::SelectOp::create(
                    c, loc, arith::AndIOp::create(c, loc, had, within),
                    arith::OrIOp::create(c, loc, kept16, bit), kept16);
              }
              Value keep = arith::CmpIOp::create(
                  c, loc, arith::CmpIPredicate::ne, kept16, constant32(c, 0));
              Value ballot = gpu::BallotOp::create(c, loc, i32, keep);
              Value before = math::CtPopOp::create(
                  c, loc, arith::AndIOp::create(c, loc, ballot, below));
              scf::IfOp::create(c, loc, keep, [&](OpBuilder &d, Location) {
                Value to = slotOf(d, arith::AddIOp::create(d, loc, kept,
                                                           before));
                memref::StoreOp::create(d, loc, entry, inner.entries,
                                        ValueRange{to});
                memref::StoreOp::create(
                    d, loc,
                    arith::OrIOp::create(
                        d, loc,
                        arith::AndIOp::create(d, loc, mask,
                                              constant32(d, ~0xFFFF)),
                        kept16),
                    inner.masks, ValueRange{to});
                scf::YieldOp::create(d, loc);
              });
              scf::YieldOp::create(
                  c, loc,
                  ValueRange{arith::AddIOp::create(
                      c, loc, kept, math::CtPopOp::create(c, loc, ballot))});
            });
        // The entries that each block of the group keeps: its blocks fill
        // in order.
        Value kept = rounds.getResult(0);
        Value blocksOfGroup = arith::CeilDivUIOp::create(b, loc, count,
                                                         constant32(b, 64));
        scf::ForOp::create(
            b, loc, lane, blocksOfGroup, constant32(b, 32), ValueRange(),
            [&](OpBuilder &c, Location, Value k, ValueRange) {
              Value block = memref::LoadOp::create(
                  c, loc, blockOf,
                  ValueRange{arith::AddIOp::create(c, loc, base,
                                                   toIndex(c, k))});
              Value left = arith::SubIOp::create(
                  c, loc, kept,
                  arith::MulIOp::create(c, loc, k, constant32(c, 64)));
              Value held = arith::MaxSIOp::create(
                  c, loc,
                  arith::MinSIOp::create(c, loc, left, constant32(c, 64)),
                  constant32(c, 0));
              memref::StoreOp::create(c, loc, held, inner.counts,
                                      ValueRange{toIndex(c, block)});
              // A block that holds entries is an active unit: it counts in
              // its runs of 32 and 1024 blocks (D268).
              scf::IfOp::create(
                  c, loc,
                  arith::CmpIOp::create(c, loc, arith::CmpIPredicate::sgt,
                                        held, constant32(c, 0)),
                  [&](OpBuilder &d, Location) {
                    Value at = toIndex(d, block);
                    emitAtomicAddI32(
                        d, loc, prune.runSums,
                        arith::DivUIOp::create(d, loc, at,
                                               createIndex(d, loc, 32)),
                        constant32(d, 1));
                    emitAtomicAddI32(
                        d, loc, prune.wideSums,
                        arith::DivUIOp::create(d, loc, at,
                                               createIndex(d, loc, 1024)),
                        constant32(d, 1));
                    scf::YieldOp::create(d, loc);
                  });
              scf::YieldOp::create(c, loc);
            });
      });

  // The active units (D268): the blocks that hold entries of
  // the inner list, in the order of the blocks, as the loop over pairs took
  // all blocks before, so that it launches no work for a block that the
  // pruning left empty. A warp for each run of 32 blocks: an active block
  // follows those of the runs of 1024 blocks before its own (their counts),
  // of the runs of 32 before its own in that run of 1024 (their counts), and
  // of the blocks before it in its run (a ballot).
  launchOver(
      builder, loc, arith::MulIOp::create(builder, loc, runs, warp32),
      [&](OpBuilder &b, Value block) {
        Value run = arith::DivUIOp::create(b, loc, block, warp32);
        Value laneIndex = arith::RemUIOp::create(b, loc, block, warp32);
        Value lane = arith::IndexCastOp::create(b, loc, i32, laneIndex);
        Value wide = arith::DivUIOp::create(b, loc, run, warp32);
        Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                             block, blocks);
        Value held = memref::LoadOp::create(
            b, loc, inner.counts,
            ValueRange{arith::SelectOp::create(b, loc, inside, block,
                                               createIndex(b, loc, 0))});
        Value active = arith::AndIOp::create(
            b, loc, inside,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, held,
                                  constant32(b, 0)));
        Value ballot = gpu::BallotOp::create(b, loc, i32, active);
        Value below = arith::SubIOp::create(
            b, loc, arith::ShLIOp::create(b, loc, constant32(b, 1), lane),
            constant32(b, 1));
        Value rank = math::CtPopOp::create(
            b, loc, arith::AndIOp::create(b, loc, ballot, below));
        // The runs of 1024 blocks before this one, a lane each in turn.
        Value earlier =
            scf::ForOp::create(
                b, loc, laneIndex, wide, warp32, ValueRange{constant32(b, 0)},
                [&](OpBuilder &c, Location, Value j, ValueRange carried) {
                  scf::YieldOp::create(
                      c, loc,
                      ValueRange{arith::AddIOp::create(
                          c, loc, carried[0],
                          memref::LoadOp::create(c, loc, prune.wideSums,
                                                 ValueRange{j}))});
                })
                .getResult(0);
        // The runs of 32 before this one in its run of 1024, a lane each.
        Value near = arith::AddIOp::create(
            b, loc, arith::MulIOp::create(b, loc, wide, warp32), laneIndex);
        Value before = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                             near, run);
        earlier = arith::AddIOp::create(
            b, loc, earlier,
            arith::SelectOp::create(
                b, loc, before,
                memref::LoadOp::create(
                    b, loc, prune.runSums,
                    ValueRange{arith::SelectOp::create(
                        b, loc, before, near, createIndex(b, loc, 0))}),
                constant32(b, 0)));
        for (int64_t offset = 16; offset >= 1; offset /= 2)
          earlier = arith::AddIOp::create(b, loc, earlier,
                                          shuffleXor(b, loc, earlier, offset));
        scf::IfOp::create(b, loc, active, [&](OpBuilder &c, Location) {
          memref::StoreOp::create(
              c, loc, arith::IndexCastOp::create(c, loc, i32, block),
              prune.activeUnits,
              ValueRange{toIndex(c, arith::AddIOp::create(c, loc, earlier,
                                                          rank))});
          scf::YieldOp::create(c, loc);
        });
        // The first lane of the last run writes the number of all.
        Value last = arith::AndIOp::create(
            b, loc,
            arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::eq, run,
                arith::SubIOp::create(b, loc, runs, createIndex(b, loc, 1))),
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane,
                                  constant32(b, 0)));
        scf::IfOp::create(b, loc, last, [&](OpBuilder &c, Location) {
          memref::StoreOp::create(
              c, loc,
              arith::AddIOp::create(c, loc, earlier,
                                    math::CtPopOp::create(c, loc, ballot)),
              prune.activeCount, ValueRange{createIndex(c, loc, 0)});
          scf::YieldOp::create(c, loc);
        });
      });

  // The loops over pairs at these positions may take the frames as they are
  // (lowerGroupPairFor).
  memref::StoreOp::create(
      builder, loc,
      arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                builder.getBoolAttr(true)),
      structure.prunedNow, ValueRange{});
}

void Lowering::emitPruneCheck(OpBuilder &builder, Location loc,
                              const Neighbors &structure, Value positions,
                              Value box, double cutoff) {
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  Type wide = builder.getI64Type();
  auto vector3 = VectorType::get({3}, f32);
  auto constant32 = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v));
  };
  auto toIndex = [&](OpBuilder &b, Value v) -> Value {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
  };
  GroupBuffers buffers = getGroupsBuffers(builder, loc, structure.handle);
  InnerLists inner = getInnerLists(builder, loc, structure.handle);
  PruneBuffers prune = getPruneBuffers(builder, loc, structure.handle);
  auto size = [&](int64_t which) -> Value {
    return toIndex(builder, memref::LoadOp::create(
                                builder, loc, structure.sizes,
                                ValueRange{createIndex(builder, loc, which)}));
  };
  Value places = size(0), blocks = size(2);
  // The frames of this step's positions, apart from those of the pruning.
  Value placed = createDeviceBuffer(
      builder, loc, getDeviceType({ShapedType::kDynamic, 3}, f32),
      ValueRange{places});
  emitFramePositions(builder, loc, buffers, places, positions, box, placed,
                     Value(), Value(), {});
  // The two counts, from zero.
  Value counts = createDeviceBuffer(
      builder, loc, getDeviceType({2}, i32), ValueRange());
  launchOver(builder, loc, createIndex(builder, loc, 2),
             [&](OpBuilder &b, Value slot) {
               memref::StoreOp::create(b, loc, constant32(b, 0), counts,
                                       ValueRange{slot});
             });
  Value box32 = arith::TruncFOp::create(
      builder, loc,
      VectorType::get({cast<VectorType>(box.getType()).getNumElements()}, f32),
      box);
  float cutoff2 = static_cast<float>(cutoff * cutoff);
  // The pairs within the cutoff of entry `slot` of `entries` and `masks`,
  // of the group `group`, if `valid`.
  auto countPairs = [&](OpBuilder &o, Value group, Value slot, Value entries,
                        Value masks, Value valid) -> Value {
    Value entry = arith::SelectOp::create(
        o, loc, valid,
        memref::LoadOp::create(o, loc, entries, ValueRange{slot}),
        constant32(o, 0));
    Value mask = arith::SelectOp::create(
        o, loc, valid,
        memref::LoadOp::create(o, loc, masks, ValueRange{slot}),
        constant32(o, 0));
    Value position = loadElement(o, loc, placed, toIndex(o, entry));
    int64_t width = getEntryShiftBits(isTriclinic(box));
    int64_t offset = getEntryShiftOffset(isTriclinic(box));
    SmallVector<Value> cells;
    for (int64_t axis = 0; axis != 3; ++axis)
      cells.push_back(arith::SIToFPOp::create(
          o, loc, f32,
          arith::SubIOp::create(
              o, loc,
              arith::AndIOp::create(
                  o, loc,
                  arith::ShRUIOp::create(o, loc, mask,
                                         constant32(o, 16 + width * axis)),
                  constant32(o, (1 << width) - 1)),
              constant32(o, offset))));
    position = arith::AddFOp::create(
        o, loc, position,
        emitLatticeShift(
            o, loc, vector::FromElementsOp::create(o, loc, vector3, cells),
            box32));
    Value pairs = constant32(o, 0);
    for (int64_t q = 0; q != 16; ++q) {
      Value member = loadElement(
          o, loc, placed,
          arith::AddIOp::create(
              o, loc,
              arith::MulIOp::create(o, loc, group, createIndex(o, loc, 16)),
              createIndex(o, loc, q)));
      Value d = arith::SubFOp::create(o, loc, member, position);
      Value r2 = vector::ReductionOp::create(
          o, loc, vector::CombiningKind::ADD,
          arith::MulFOp::create(o, loc, d, d));
      Value within = arith::AndIOp::create(
          o, loc,
          arith::CmpFOp::create(
              o, loc, arith::CmpFPredicate::OLE, r2,
              arith::ConstantOp::create(o, loc, f32,
                                        o.getF32FloatAttr(cutoff2))),
          arith::CmpIOp::create(
              o, loc, arith::CmpIPredicate::ne,
              arith::AndIOp::create(o, loc, mask, constant32(o, 1 << q)),
              constant32(o, 0)));
      pairs = arith::AddIOp::create(
          o, loc, pairs, arith::ExtUIOp::create(o, loc, i32, within));
    }
    return pairs;
  };
  Value c64 = createIndex(builder, loc, 64);
  Value slots = arith::MulIOp::create(builder, loc, blocks, c64);
  // The outer list: every block of every group.
  launchOver(builder, loc, slots, [&](OpBuilder &b, Value t) {
    Value block = arith::DivUIOp::create(b, loc, t, c64);
    Value slot = arith::RemUIOp::create(b, loc, t, c64);
    Value group = toIndex(
        b, memref::LoadOp::create(b, loc, buffers.units, ValueRange{block}));
    Value ordinal = toIndex(
        b, memref::LoadOp::create(b, loc, buffers.ordinals, ValueRange{block}));
    Value count = toIndex(
        b, memref::LoadOp::create(b, loc, buffers.counts, ValueRange{group}));
    Value valid = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult,
        arith::AddIOp::create(
            b, loc, arith::MulIOp::create(b, loc, ordinal, c64), slot),
        count);
    emitAtomicAddI32(b, loc, counts, createIndex(b, loc, 0),
                     countPairs(b, group, t, buffers.entries, buffers.masks,
                                valid));
  });
  // The inner list: the blocks of the active units only, so that a unit
  // that they leave out loses its pairs.
  launchOver(builder, loc, slots, [&](OpBuilder &b, Value t) {
    Value i = arith::DivUIOp::create(b, loc, t, c64);
    Value slot = arith::RemUIOp::create(b, loc, t, c64);
    Value active = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, i,
        toIndex(b, memref::LoadOp::create(b, loc, prune.activeCount,
                                          ValueRange{createIndex(b, loc, 0)})));
    scf::IfOp::create(b, loc, active, [&](OpBuilder &c, Location) {
      Value block = toIndex(c, memref::LoadOp::create(
                                   c, loc, prune.activeUnits, ValueRange{i}));
      Value group = toIndex(
          c, memref::LoadOp::create(c, loc, buffers.units, ValueRange{block}));
      Value valid = arith::CmpIOp::create(
          c, loc, arith::CmpIPredicate::ult, slot,
          toIndex(c, memref::LoadOp::create(c, loc, inner.counts,
                                            ValueRange{block})));
      Value at = arith::AddIOp::create(
          c, loc, arith::MulIOp::create(c, loc, block, c64), slot);
      emitAtomicAddI32(c, loc, counts, createIndex(c, loc, 1),
                       countPairs(c, group, at, inner.entries, inner.masks,
                                  valid));
      scf::YieldOp::create(c, loc);
    });
  });
  Value host = memref::AllocOp::create(builder, loc, MemRefType::get({2}, i32));
  createTransfer(builder, loc, host, counts);
  auto read = [&](int64_t which) -> Value {
    return arith::ExtUIOp::create(
        builder, loc, wide,
        memref::LoadOp::create(builder, loc, host,
                               ValueRange{createIndex(builder, loc, which)}));
  };
  func::CallOp::create(
      builder, loc,
      getOrDeclare(checkPrunedName, builder.getFunctionType({wide, wide}, {})),
      ValueRange{read(0), read(1)});
  memref::DeallocOp::create(builder, loc, host);
  for (Value buffer : {placed, counts}) {
    auto type = cast<MemRefType>(buffer.getType());
    Value plain = memref::MemorySpaceCastOp::create(
        builder, loc,
        MemRefType::get(type.getShape(), type.getElementType()), buffer);
    gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange(), plain);
  }
}

LogicalResult Lowering::emitGroupsBuild(OpBuilder &builder, Location loc,
                                        const Neighbors &structure,
                                        Value positions, Value box,
                                        double reach) {
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  if (!real.isF64())
    return module.emitError()
           << "structures of groups of neighbors are built from positions "
              "in f64, not "
           << real;
  // A triclinic cell takes the build of its own (docs/triclinic-m2.md).
  bool tilted = isTriclinic(box);
  if (failed(addGroupsTemplates(tilted)))
    return failure();

  // Without excluded pairs the build takes a buffer with no rows.
  Value excluded = structure.excluded;
  Value noExcluded;
  if (!excluded) {
    MemRefType type = getDeviceType(
        {ShapedType::kDynamic, ShapedType::kDynamic}, builder.getI32Type());
    noExcluded = createDeviceBuffer(
        builder, loc, type,
        ValueRange{createIndex(builder, loc, 0), createIndex(builder, loc, 1)});
    excluded = noExcluded;
  }
  auto build = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
      module, tilted ? std::string(buildGroupsName) + "_triclinic"
                     : std::string(buildGroupsName)));
  Type wide = builder.getI64Type();
  Value reachValue = createReal(builder, loc, builder.getF64Type(), reach);

  // Build; if a buffer was too small, the runtime makes more room, and the
  // build is made again.
  auto again = scf::WhileOp::create(builder, loc, TypeRange(), ValueRange());
  {
    Block *before = builder.createBlock(&again.getBefore());
    OpBuilder at = OpBuilder::atBlockEnd(before);
    GroupBuffers buffers = getGroupsBuffers(at, loc, structure.handle);
    func::CallOp::create(
        at, loc, build,
        ValueRange{positions, box, reachValue, excluded, buffers.order,
                   structure.placeOf, buffers.entries, buffers.masks,
                   buffers.counts, buffers.units, buffers.ordinals,
                   buffers.shifts, structure.sizes});
    auto load = [&](int64_t which) -> Value {
      return arith::ExtUIOp::create(
          at, loc, wide,
          memref::LoadOp::create(at, loc, structure.sizes,
                                 ValueRange{createIndex(at, loc, which)}));
    };
    auto capacity = [&](Value buffer) -> Value {
      return arith::IndexCastOp::create(
          at, loc, wide,
          memref::DimOp::create(at, loc, buffer, createIndex(at, loc, 0)));
    };
    Value places = load(0), blocks = load(2);
    Value more = arith::OrIOp::create(
        at, loc,
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ugt, places,
                              capacity(buffers.order)),
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ugt, blocks,
                              capacity(buffers.units)));
    scf::IfOp::create(at, loc, more, [&](OpBuilder &then, Location) {
      func::CallOp::create(
          then, loc,
          getOrDeclare(growGroupsName,
                       then.getFunctionType({wide, wide, wide}, {})),
          ValueRange{structure.handle, places, blocks});
      scf::YieldOp::create(then, loc);
    });
    // The runtime keeps the largest use of the blocks and counts the
    // groups with too many excluded partners, for the log.
    func::CallOp::create(
        at, loc,
        getOrDeclare(noteGroupsName,
                     at.getFunctionType({wide, wide, wide, wide}, {})),
        ValueRange{blocks, capacity(buffers.units), load(1), load(3)});
    scf::ConditionOp::create(at, loc, more, ValueRange());
    Block *after = builder.createBlock(&again.getAfter());
    OpBuilder close = OpBuilder::atBlockEnd(after);
    scf::YieldOp::create(close, loc);
  }
  builder.setInsertionPointAfter(again);
  func::CallOp::create(
      builder, loc,
      getOrDeclare(countBuildName, builder.getFunctionType({}, {})),
      ValueRange());
  emitStopNotNumbers(
      builder, loc,
      arith::ExtUIOp::create(
          builder, loc, wide,
          memref::LoadOp::create(builder, loc, structure.sizes,
                                 ValueRange{createIndex(builder, loc, 4)})));

  if (noExcluded) {
    Value plain = memref::MemorySpaceCastOp::create(
        builder, loc,
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                        builder.getI32Type()),
        noExcluded);
    gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange(), plain);
  }
  return success();
}

LogicalResult Lowering::finishBuild(OpBuilder &builder, Location loc,
                                    const Neighbors &structure,
                                    Value positions, Value box) {
  Type wide = builder.getI64Type();
  // Remember the configuration that the structure was built at.
  createTransfer(builder, loc, structure.reference, positions);
  for (int64_t c = 0; c < getCellSize(); ++c) {
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
    if (moved)
      return arith::XOrIOp::create(at, loc, moved, yes);
    Value moved2 = op.getScratch()[0];
    launchOver(at, loc, structure.size, [&](OpBuilder &body, Value particle) {
      Value now = loadElement(body, loc, positions, particle);
      Value then = arith::MulFOp::create(
          body, loc, loadElement(body, loc, structure.reference, particle),
          scaleReal);
      Value change = arith::SubFOp::create(body, loc, now, then);
      Value squares = arith::MulFOp::create(body, loc, change, change);
      Value distance2 = vector::ReductionOp::create(
          body, loc, vector::CombiningKind::ADD, squares);
      memref::StoreOp::create(body, loc, distance2, moved2,
                              ValueRange{particle});
    });
    Value farthest =
        emitReductions(at, loc, {moved2}, {op.getScratch()[1]},
                       structure.size, /*isSum=*/false)
            .front();
    return arith::CmpFOp::create(at, loc, arith::CmpFPredicate::OLE,
                                 farthest, limitReal);
  };
  (void)skin;

  Value yes = arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(true));
  // Before the first build the configuration that the test compares with
  // holds nothing, and the structure is not valid whatever the test says.
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

  // A dual list (D114): where the outer list is not valid it is built and
  // the inner one pruned from it; where only the inner one is not, the
  // inner one is pruned again.
  if (std::optional<APFloat> prune = op.getPruneSkin()) {
    if (!structure.groups)
      return op.emitOpError()
             << "keeps a dual list (D114), which needs a structure of groups";
    if (!op.getStale())
      return op.emitOpError() << "keeps a dual list without the test of "
                                 "its inner list ('stale')";
    double inner =
        op.getCutoff().convertToDouble() + prune->convertToDouble();
    // A pruning below says that it pruned (D268).
    memref::StoreOp::create(
        builder, loc,
        arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                  builder.getBoolAttr(false)),
        structure.prunedNow, ValueRange{});
    // The branches are filled once they are in the function: the pruning
    // launches kernels, which look up where they are.
    auto rebuild = scf::IfOp::create(builder, loc, stale,
                                     /*withElseRegion=*/true);
    {
      OpBuilder then(rebuild.thenBlock()->getTerminator());
      status = emitBuild(then, loc, structure, positions, box, reach,
                         cellWidth);
      emitGroupsPrune(then, loc, structure, positions, box, inner);
      OpBuilder otherwise(rebuild.elseBlock()->getTerminator());
      auto again = scf::IfOp::create(otherwise, loc, op.getStale(),
                                     /*withElseRegion=*/false);
      OpBuilder prune(again.thenBlock()->getTerminator());
      emitGroupsPrune(prune, loc, structure, positions, box, inner);
    }
    if (checkPruning)
      emitPruneCheck(builder, loc, structure, positions, box,
                     op.getCutoff().convertToDouble());
    structure.dual = true;
    neighbors[op.getResult()] = structure;
    return status;
  }

  // The branch is filled once it is in the function: a build of groups
  // launches kernels, which look up where they are.
  auto rebuild = scf::IfOp::create(builder, loc, stale,
                                   /*withElseRegion=*/false);
  OpBuilder then(rebuild.thenBlock()->getTerminator());
  return emitBuild(then, loc, structure, positions, box, reach, cellWidth);
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
  launchOver(builder, loc, size, [&](OpBuilder &body, Value place) {
    Value from =
        memref::LoadOp::create(body, loc, order, ValueRange{place});
    Value particle =
        arith::IndexCastOp::create(body, loc, body.getIndexType(), from);
    storeElement(body, loc, loadElement(body, loc, field, particle), out,
                 place);
  });
}

/// The places of the particles whose value is not zero, in ascending
/// order, then −1 − the place of each of the others, in ascending order
/// (D262), in integers: a thread counts the
/// listed particles of each chunk of 256 places, one thread sums the counts
/// of the chunks before each and of all, and a thread writes the entries
/// of each chunk from there. It runs where the particles take a new order,
/// not at every step.
void Lowering::lowerNonzero(md_exec::NonzeroOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value field = op.getField(), out = op.getOut();
  Value size = createSize(builder, loc, field);
  Value one = createIndex(builder, loc, 1);
  const int64_t width = 256;
  Value chunk = createIndex(builder, loc, width);
  Value chunks = createGroups(builder, loc, size, width);
  Type i32 = builder.getI32Type();
  Type element = cast<MemRefType>(field.getType()).getElementType();
  // The counts of the chunks, and after them that of all.
  Value counts = createDeviceBuffer(
      builder, loc, getDeviceType({ShapedType::kDynamic}, i32),
      ValueRange{arith::AddIOp::create(builder, loc, chunks, one)});

  // The places of chunk `c`, and whether particle `i` is listed: a value
  // that is not a number is.
  auto bounds = [&](OpBuilder &body, Value c) {
    Value from = arith::MulIOp::create(body, loc, c, chunk);
    Value to = arith::MinUIOp::create(
        body, loc, arith::AddIOp::create(body, loc, from, chunk), size);
    return std::make_pair(from, to);
  };
  auto isListed = [&](OpBuilder &body, Value i) -> Value {
    Value value = memref::LoadOp::create(body, loc, field, ValueRange{i});
    Value none = arith::ConstantOp::create(body, loc,
                                           body.getFloatAttr(element, 0.0));
    return arith::CmpFOp::create(body, loc, arith::CmpFPredicate::UNE, value,
                                 none);
  };
  auto constant = [&](OpBuilder &body, int32_t value) -> Value {
    return arith::ConstantOp::create(body, loc, body.getI32IntegerAttr(value));
  };
  launchOver(builder, loc, chunks, [&](OpBuilder &body, Value c) {
    auto [from, to] = bounds(body, c);
    Value add = constant(body, 1);
    auto loop = scf::ForOp::create(
        body, loc, from, to, createIndex(body, loc, 1),
        ValueRange{constant(body, 0)},
        [&](OpBuilder &inner, Location, Value i, ValueRange args) {
          Value more = arith::AddIOp::create(inner, loc, args[0], add);
          Value next = arith::SelectOp::create(inner, loc, isListed(inner, i),
                                               more, args[0]);
          scf::YieldOp::create(inner, loc, ValueRange{next});
        });
    memref::StoreOp::create(body, loc, loop.getResult(0), counts,
                            ValueRange{c});
  });
  launchOver(builder, loc, one, [&](OpBuilder &body, Value) {
    auto loop = scf::ForOp::create(
        body, loc, createIndex(body, loc, 0), chunks, createIndex(body, loc, 1),
        ValueRange{constant(body, 0)},
        [&](OpBuilder &inner, Location, Value c, ValueRange args) {
          Value count =
              memref::LoadOp::create(inner, loc, counts, ValueRange{c});
          memref::StoreOp::create(inner, loc, args[0], counts, ValueRange{c});
          Value next = arith::AddIOp::create(inner, loc, args[0], count);
          scf::YieldOp::create(inner, loc, ValueRange{next});
        });
    memref::StoreOp::create(body, loc, loop.getResult(0), counts,
                            ValueRange{chunks});
  });
  launchOver(builder, loc, chunks, [&](OpBuilder &body, Value c) {
    auto [from, to] = bounds(body, c);
    // The listed particles of the chunk from the count of those before it,
    // the others from the count of all and of the others before it.
    Value listedFirst =
        memref::LoadOp::create(body, loc, counts, ValueRange{c});
    Value all = memref::LoadOp::create(body, loc, counts, ValueRange{chunks});
    Value from32 = arith::IndexCastOp::create(body, loc, i32, from);
    Value othersFirst = arith::AddIOp::create(
        body, loc, all, arith::SubIOp::create(body, loc, from32, listedFirst));
    Value add = constant(body, 1);
    Value minusOne = constant(body, -1);
    scf::ForOp::create(
        body, loc, from, to, createIndex(body, loc, 1),
        ValueRange{listedFirst, othersFirst},
        [&](OpBuilder &inner, Location, Value i, ValueRange args) {
          Value listed = isListed(inner, i);
          Value place = arith::IndexCastOp::create(inner, loc, i32, i);
          Value other = arith::SubIOp::create(inner, loc, minusOne, place);
          Value at32 =
              arith::SelectOp::create(inner, loc, listed, args[0], args[1]);
          Value entry =
              arith::SelectOp::create(inner, loc, listed, place, other);
          Value at = arith::IndexCastOp::create(inner, loc,
                                                inner.getIndexType(), at32);
          memref::StoreOp::create(inner, loc, entry, out, ValueRange{at});
          Value nextListed = arith::SelectOp::create(
              inner, loc, listed,
              arith::AddIOp::create(inner, loc, args[0], add), args[0]);
          Value nextOthers = arith::SelectOp::create(
              inner, loc, listed, args[1],
              arith::AddIOp::create(inner, loc, args[1], add));
          scf::YieldOp::create(inner, loc,
                               ValueRange{nextListed, nextOthers});
        });
  });
  freeDeviceAtEndOfBlock(op, counts);
}

//===----------------------------------------------------------------------===//
// Ops and functions
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lowerOp(Operation *op) {
  setPurpose(op);
  if (llvm::any_of(op->getOperandTypes(), md_exec::isValueFormType) ||
      llvm::any_of(op->getResultTypes(), md_exec::isValueFormType))
    return op->emitOpError()
           << "is not in the storage form; run 'md-exec-assign-storage' "
              "first";

  // The members of tuples are on the host, and the incidence structure is
  // built there.
  if (isa_and_nonnull<md_exec::MDExecDialect>(op->getDialect()) &&
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
  } else if (auto nonzero = dyn_cast<md_exec::NonzeroOp>(op)) {
    lowerNonzero(nonzero);
  } else if (auto cell = dyn_cast<md_exec::ReferenceCellOp>(op)) {
    // The cell that the structure was built in, or its inner list pruned
    // in (D114), as the vector of its edges that cells have become.
    Neighbors structure;
    if (failed(getNeighbors(op, cell.getNeighbors(), structure)))
      return failure();
    if (cell.getPruned() && !structure.prunedBox)
      return op->emitOpError()
             << "takes the cell of the pruning of a dual list, which only "
                "a structure of groups keeps";
    OpBuilder builder(op);
    Location loc = op->getLoc();
    SmallVector<Value, 6> edges;
    for (int64_t c = 0; c < getCellSize(); ++c)
      edges.push_back(memref::LoadOp::create(
          builder, loc, cell.getPruned() ? structure.prunedBox : structure.box,
          ValueRange{createIndex(builder, loc, c)}));
    cell->getResult(0).replaceAllUsesWith(vector::FromElementsOp::create(
        builder, loc, VectorType::get({getCellSize()}, builder.getF64Type()),
        edges));
  } else if (auto numbers = dyn_cast<md_exec::CellNumbersOp>(op)) {
    // The cell is the vector of its edges by now, or of its diagonal and
    // tilts; an orthorhombic one takes tilts of zero.
    OpBuilder builder(op);
    Location loc = op->getLoc();
    Value cell = numbers->getOperand(0);
    Value result = cell;
    if (cast<VectorType>(cell.getType()).getDimSize(0) != 6) {
      Value zero = arith::ConstantOp::create(builder, loc,
                                             builder.getF64FloatAttr(0.0));
      SmallVector<Value, 6> values;
      for (int64_t k = 0; k != 3; ++k)
        values.push_back(vector::ExtractOp::create(builder, loc, cell, k));
      values.append(3, zero);
      result = vector::FromElementsOp::create(
          builder, loc, VectorType::get({6}, builder.getF64Type()), values);
    }
    numbers.getResult().replaceAllUsesWith(result);
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
    if (reference.getPruned() && !structure.prunedReference)
      return op->emitOpError()
             << "takes the configuration of the pruning of a dual list, "
                "which only a structure of groups keeps";
    if (reference.getResult().getType() != structure.reference.getType())
      return op->emitOpError()
             << "the structure holds the positions in "
             << structure.reference.getType() << ", not in "
             << reference.getResult().getType();
    reference.getResult().replaceAllUsesWith(
        reference.getPruned() ? structure.prunedReference
                              : structure.reference);
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
  } else if (auto edges = dyn_cast<md_exec::BuildEdgesOp>(op)) {
    if (failed(lowerBuildEdges(edges)))
      return failure();
  } else if (auto count = dyn_cast<md_exec::RebuildCountOp>(op)) {
    Neighbors structure;
    if (failed(getNeighbors(op, count.getNeighbors(), structure)))
      return failure();
    OpBuilder builder(op);
    count.getResult().replaceAllUsesWith(memref::LoadOp::create(
        builder, op->getLoc(), structure.builds, ValueRange{}));
  } else if (isa<md_exec::JoinOp>(op)) {
    // The first stream waits for the work of the second.
    OpBuilder builder(op);
    func::CallOp::create(
        builder, op->getLoc(),
        getOrDeclare("mdrtSideJoin", builder.getFunctionType({}, {})),
        ValueRange());
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
  } else if (isa_and_nonnull<md::MDDialect>(op->getDialect()) ||
             isa_and_nonnull<md_exec::MDExecDialect>(op->getDialect()) ||
             isa_and_nonnull<mdrt::MDRTDialect>(op->getDialect()) ||
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

/// Kernels that take a value the host computes from global sums compute it
/// themselves from the copy of the sums on the device, so that the host
/// need not wait for the sums to launch them (D156). The value is a chain
/// of pure ops over loads of the results that emitReductions marked
/// `mdir.readback`; the chain is cloned into the kernel with each load
/// reading the device's buffer, which the kernel of the reduction wrote
/// before, in the order of the stream. A group of copies whose loads the
/// host then no longer needs is removed. The work of a second stream (D87)
/// is left as it is, since its order relative to the first is the host's.
static void keepReadbacksOnDevice(func::FuncOp function) {
  llvm::DenseMap<Value, Value> deviceOf;
  function.walk([&](gpu::MemcpyOp copy) {
    if (copy->hasAttr("mdir.readback"))
      deviceOf[copy.getDst()] = copy.getSrc();
  });
  if (deviceOf.empty())
    return;
  // The ops of the work of a second stream.
  llvm::DenseSet<Operation *> side;
  function.walk([&](Block *block) {
    bool inside = false;
    for (Operation &op : *block) {
      if (auto call = dyn_cast<func::CallOp>(op)) {
        if (call.getCallee() == "mdrtSideBegin")
          inside = true;
        else if (call.getCallee() == "mdrtSideEnd")
          inside = false;
      }
      if (inside)
        op.walk([&](Operation *nested) { side.insert(nested); });
    }
  });
  auto isPure = [](Operation *op) {
    return op->getNumRegions() == 0 && isMemoryEffectFree(op);
  };
  auto isSum = [&](Operation *op) {
    auto load = dyn_cast_or_null<memref::LoadOp>(op);
    return load && load->hasAttr("mdir.readback") &&
           deviceOf.count(load.getMemRef()) && !side.contains(op);
  };
  // Whether `value` is computed by pure ops from at least one load of the
  // sums.
  llvm::DenseMap<Value, bool> fromSums;
  std::function<bool(Value)> readsSums = [&](Value value) -> bool {
    auto found = fromSums.find(value);
    if (found != fromSums.end())
      return found->second;
    fromSums[value] = false;
    Operation *def = value.getDefiningOp();
    bool result = false;
    if (isSum(def)) {
      result = true;
    } else if (def && isPure(def) && !side.contains(def)) {
      for (Value operand : def->getOperands())
        result |= readsSums(operand);
    }
    fromSums[value] = result;
    return result;
  };

  SmallVector<gpu::LaunchOp> launches;
  function.walk([&](gpu::LaunchOp launch) {
    if (!side.contains(launch))
      launches.push_back(launch);
  });
  for (gpu::LaunchOp launch : launches) {
    Region &body = launch.getBody();
    llvm::SetVector<Value> captured;
    body.walk([&](Operation *op) {
      for (Value operand : op->getOperands())
        if (!body.isAncestor(operand.getParentRegion()))
          captured.insert(operand);
    });
    OpBuilder builder = OpBuilder::atBlockBegin(&body.front());
    IRMapping mapping;
    std::function<Value(Value)> clone = [&](Value value) -> Value {
      if (Value done = mapping.lookupOrNull(value))
        return done;
      if (!readsSums(value))
        return value;
      Operation *def = value.getDefiningOp();
      Value result;
      if (auto load = dyn_cast<memref::LoadOp>(def)) {
        SmallVector<Value> indices;
        for (Value index : load.getIndices())
          indices.push_back(builder.clone(*index.getDefiningOp())->getResult(0));
        result = memref::LoadOp::create(builder, load.getLoc(),
                                        deviceOf.lookup(load.getMemRef()),
                                        indices);
      } else {
        for (Value operand : def->getOperands())
          mapping.map(operand, clone(operand));
        Operation *copy = builder.clone(*def, mapping);
        result = copy->getResult(cast<OpResult>(value).getResultNumber());
      }
      mapping.map(value, result);
      return result;
    };
    for (Value value : captured) {
      if (!readsSums(value))
        continue;
      Value inside = clone(value);
      value.replaceUsesWithIf(inside, [&](OpOperand &use) {
        return body.isAncestor(use.getOwner()->getParentRegion());
      });
    }
  }

  // The host's chains of pure ops and loads that nothing uses any more,
  // and then the copies whose loads are all gone.
  bool erased = true;
  while (erased) {
    erased = false;
    SmallVector<Operation *> dead;
    function.walk([&](Operation *op) {
      if (op->use_empty() && op->getNumResults() > 0 &&
          (isPure(op) || isSum(op)))
        dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
    erased = !dead.empty();
  }
  llvm::MapVector<int64_t, SmallVector<Operation *>> groups;
  function.walk([&](Operation *op) {
    if (auto id = op->getAttrOfType<IntegerAttr>("mdir.readback"))
      groups[id.getInt()].push_back(op);
  });
  for (auto &[id, group] : groups) {
    if (llvm::any_of(group, [](Operation *op) {
          return isa<memref::LoadOp>(op);
        }))
      continue;
    SmallVector<Operation *> trio;
    for (Operation *op : group)
      if (isa<gpu::WaitOp, gpu::MemcpyOp>(op))
        trio.push_back(op);
    // Users first: the last wait, the copy, the first wait of each copy.
    for (Operation *op : llvm::reverse(trio))
      if (op->use_empty() ||
          llvm::all_of(op->getUsers(), [](Operation *) { return false; }))
        op->erase();
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
      // A group stays out of the work of a second stream, which begins
      // with a call to mdrtSideBegin: issued there, a copy would go to
      // that stream (D87).
      if (auto call = dyn_cast<func::CallOp>(op))
        if (call.getCallee().starts_with("mdrtSide")) {
          target = op;
          break;
        }
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
  Type box = VectorType::get({getCellSize()}, Float64Type::get(context));
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

  // The ops that run on a second stream are independent of the ops before
  // their joins, which the lowering issues in the same order (D87).
  md_exec::BufferAliases aliases;
  WalkResult sides = function.walk([&](Operation *op) {
    if (!op->hasAttr(md_exec::kSideAttrName))
      return WalkResult::advance();
    if (!isa<md_exec::ReciprocalOp>(op)) {
      op->emitOpError() << "cannot run on a second stream; only the "
                           "reciprocal sums of PME can";
      return WalkResult::interrupt();
    }
    return failed(md_exec::verifySide(op, aliases)) ? WalkResult::interrupt()
                                           : WalkResult::advance();
  });
  if (sides.wasInterrupted())
    return failure();

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

  // A reciprocal sum between the test of a neighbor structure and the
  // first loop that takes the structure goes before the test: it takes
  // the positions only, and the device computes it while the host waits
  // for the test (D113). Its operands, its scratch included, must already
  // be there: a run of one step with no energies allocates the scratch
  // after the test. Buffer dominance does not imply that its contents are
  // ready: do not cross a producer of charges or another memory dependency
  // (D179, issue #26).
  DominanceInfo dominance(function);
  function.walk([&](md_exec::RefreshNeighborsOp refresh) {
    SmallVector<md_exec::ReciprocalOp> sums;
    for (Operation *next = refresh->getNextNode(); next;
         next = next->getNextNode()) {
      bool uses = llvm::any_of(refresh->getResults(), [&](Value result) {
        return llvm::any_of(result.getUsers(), [&](Operation *user) {
          return next->isAncestor(user);
        });
      });
      if (uses || next->hasTrait<OpTrait::IsTerminator>())
        break;
      // A sum on the second stream has its window from
      // md-exec-assign-streams, which this leaves as it is (D87).
      auto reciprocal = dyn_cast<md_exec::ReciprocalOp>(next);
      if (!reciprocal || reciprocal->hasAttr(md_exec::kSideAttrName))
        continue;
      bool independent = true;
      for (Value written : llvm::concat<Value>(
               reciprocal.getOut() ? ValueRange(reciprocal.getOut())
                                   : ValueRange(),
               reciprocal.getScratch()))
        independent &= !llvm::is_contained(refresh->getOperands(), written);
      for (Value operand : reciprocal->getOperands())
        independent &= !llvm::is_contained(refresh->getResults(), operand) &&
                       dominance.properlyDominates(operand, refresh);
      for (Operation *crossed = refresh->getNextNode();
           independent && crossed != next; crossed = crossed->getNextNode())
        independent = md_exec::areIndependent(crossed, reciprocal, aliases);
      if (independent)
        sums.push_back(reciprocal);
    }
    for (md_exec::ReciprocalOp reciprocal : sums)
      reciprocal->moveBefore(refresh);
  });

  findSharedFrames(function, aliases);

  // The kernels are copied into the loops, so the ops inside them are not
  // lowered where they are. A run of loops is lowered at its last loop.
  findRows(function);
  findCenterFusions(function);
  findOnceRuns(function);
  for (Operation *op : ops) {
    Operation *parent = op->getParentOp();
    if (isa<md_exec::ParticleForOp, md_exec::PairForOp, md_exec::TupleForOp>(
            parent))
      continue;
    if (inOnceRuns.contains(op)) {
      auto run = onceRuns.find(op);
      if (run != onceRuns.end()) {
        if (failed(lowerOnceRun(run->second)))
          return failure();
        // The arithmetic of a fused term over centers goes after its loop.
        for (md_exec::TupleForOp loop : run->second)
          if (CenterFusion *fusion = fusionOfForces.lookup(loop))
            lowered.append(fusion->chain.begin(), fusion->chain.end());
        for (md_exec::TupleForOp loop : run->second)
          lowered.push_back(loop);
      }
      continue;
    }
    if (inRows.contains(op)) {
      auto run = rows.find(op);
      if (run != rows.end()) {
        if (failed(lowerRows(run->second)))
          return failure();
        for (Operation *loop : run->second)
          lowered.push_back(loop);
      }
      auto integration = integrations.find(op);
      if (integration != integrations.end()) {
        const kernels::IntegrationRun &fused = integration->second;
        if (failed(lowerIntegration(fused)))
          return failure();
        lowered.push_back(fused.before);
        for (md_exec::TupleForOp loop : fused.loops)
          lowered.push_back(loop);
        lowered.push_back(fused.after);
      }
      continue;
    }
    if (failed(lowerOp(op)))
      return failure();
  }
  rows.clear();
  sharesPrunedFrames.clear();
  integrations.clear();
  inRows.clear();
  onceRuns.clear();
  inOnceRuns.clear();
  fusionOfSums.clear();
  fusionOfForces.clear();
  fusions.clear();

  // Users come after what they use, so erase from the back.
  for (Operation *op : llvm::reverse(lowered))
    op->erase();
  lowered.clear();
  neighbors.clear();

  // The kernels read tables from their buffers.
  lowerLookups(function);
  expandPowers(function);
  keepReadbacksOnDevice(function);
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
    if (!isa_and_nonnull<md::MDDialect, md_exec::MDExecDialect,
                         mdrt::MDRTDialect>(op.getDialect()) &&
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

#define GEN_PASS_DEF_CONVERTMDEXECTOGPU
#include "mdir/Conversion/Passes.h.inc"

namespace {
class ConvertMDExecToGPU
    : public impl::ConvertMDExecToGPUBase<ConvertMDExecToGPU> {
public:
  using impl::ConvertMDExecToGPUBase<
      ConvertMDExecToGPU>::ConvertMDExecToGPUBase;

  void runOnOperation() final {
    if (getOperation()->hasAttr("md.derivative_check")) {
      getOperation().emitError("derivative checking requires the CPU lowering");
      return signalPassFailure();
    }
    Lowering lowering(getOperation(), blockSize, rowLanes, fuseRows,
                      fuseIntegration, deterministic, tuplesOnce,
                      fuseCenters);
    lowering.keptTuples = keptTuples;
    lowering.deviceTuples = keptTuples && deviceTuples;
    lowering.checkPruning = checkPruning;
    if (failed(lowering.run()))
      return signalPassFailure();
    // Products and sums in the kernels may become fused multiply-adds, as
    // nvcc makes them by default: the backend fuses only what is marked.
    if (contract)
      getOperation()->walk([](gpu::LaunchOp launch) {
        launch.getBody().walk([](arith::ArithFastMathInterface op) {
          arith::FastMathFlags flags = arith::FastMathFlags::none;
          if (arith::FastMathFlagsAttr given = op.getFastMathFlagsAttr())
            flags = given.getValue();
          op->setAttr(op.getFastMathAttrName(),
                      arith::FastMathFlagsAttr::get(
                          op->getContext(),
                          flags | arith::FastMathFlags::contract));
        });
      });
    // Explicit fused multiply-adds (#102): every sum or difference with a
    // product as an operand becomes one, by a rule of the formula alone.
    // LLVM contracts only a product with one use, so the further uses
    // that the energies and the virial of a step give a product would
    // decide its rounding; here the product stays for its other uses. Of
    // two products, the first operand is fused.
    if (explicitFma)
      getOperation()->walk([](gpu::LaunchOp launch) {
        SmallVector<Operation *> sums;
        launch.getBody().walk([&](Operation *op) {
          if (isa<arith::AddFOp, arith::SubFOp>(op))
            sums.push_back(op);
        });
        for (Operation *op : sums) {
          Value lhs = op->getOperand(0), rhs = op->getOperand(1);
          auto lhsProduct = lhs.getDefiningOp<arith::MulFOp>();
          auto rhsProduct = rhs.getDefiningOp<arith::MulFOp>();
          if (!lhsProduct && !rhsProduct)
            continue;
          OpBuilder builder(op);
          Location loc = op->getLoc();
          Value result;
          bool subtract = isa<arith::SubFOp>(op);
          if (lhsProduct) {
            // a * b + c, or a * b - c = fma(a, b, -c).
            Value addend = subtract
                               ? arith::NegFOp::create(builder, loc, rhs)
                                     .getResult()
                               : rhs;
            result = math::FmaOp::create(builder, loc, lhsProduct.getLhs(),
                                         lhsProduct.getRhs(), addend);
          } else {
            // c + a * b, or c - a * b = fma(-a, b, c).
            Value factor = rhsProduct.getLhs();
            if (subtract)
              factor = arith::NegFOp::create(builder, loc, factor);
            result = math::FmaOp::create(builder, loc, factor,
                                         rhsProduct.getRhs(), lhs);
          }
          op->getResult(0).replaceAllUsesWith(result);
          op->erase();
        }
      });
    // An approximate division in f32 is a product with an approximate
    // reciprocal that flushes subnormal numbers (D98).
    getOperation()->walk([](gpu::LaunchOp launch) {
      SmallVector<arith::DivFOp> divisions;
      launch.getBody().walk([&](arith::DivFOp op) {
        if (op.getType().isF32() &&
            arith::bitEnumContainsAll(op.getFastmath(),
                                      arith::FastMathFlags::afn))
          divisions.push_back(op);
      });
      for (arith::DivFOp op : divisions) {
        OpBuilder builder(op);
        Value reciprocal = NVVM::RcpApproxFtzF32Op::create(
            builder, op.getLoc(), builder.getF32Type(), op.getRhs());
        Value result = reciprocal;
        // Round the approximate quotient before subsequent arithmetic.
        // Contracting this product with a cutoff shift changes cancellation
        // in parameter derivatives even when the reciprocal is unchanged.
        if (!matchPattern(op.getLhs(), m_OneFloat()))
          result = arith::MulFOp::create(
              builder, op.getLoc(), op.getLhs(), reciprocal,
              op.getFastmath() & ~arith::FastMathFlags::contract);
        op.replaceAllUsesWith(result);
        op.erase();
      }
    });
  }
};
} // namespace

} // namespace mdir
