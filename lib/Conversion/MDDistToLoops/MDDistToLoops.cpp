// The executors of md_dist: every domain in one process.
//
// See docs/distributed-ml2.md, Sections 6 and 7 (D275, D290). Every
// op runs for all domains in turn, in the order of the domains; collective
// ops are the points where all domains' values are complete, so running
// op by op is a bulk-synchronous schedule of the global view. Layouts and
// halos are handles of the in-process transport (runtime/mdrt_dist.c), and
// a distributed field is a buffer memref<D x rows x k x f64>.
//
// The reference executor (--convert-md-dist-to-loops) runs the loops of
// each domain itself, over all pairs of its rows. The second executor
// (--convert-md-dist-to-md-exec) gives the loops of each domain to md_exec,
// over its local rows, and keeps the transfers.

#include "mdir/Conversion/Passes.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mdir/Dialect/MDExec/Algebra.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDExec/NeighborContract.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"

// The codes of the phases and quantities that the runtime also reads.
#include "../../../runtime/mdrt_dist_stats.h"

using namespace mlir;
namespace md = mdir::md;
namespace mdrt = mdir::mdrt;
namespace md_exec = mdir::md_exec;
using namespace mdir::md_dist;

namespace mdir {
#define GEN_PASS_DEF_CONVERTMDDISTTOLOOPS
#define GEN_PASS_DEF_CONVERTMDDISTTOMDEXEC
#define GEN_PASS_DEF_LOWERMDRTHALOEVENTS
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

namespace {

/// What a value of md_dist is in the lowered program.
struct Storage {
  /// A layout or a halo: the handle of the transport, the number of rows
  /// of each domain (memref<?xi64>), and the particle of each row
  /// (memref<?x?xi64>).
  Value handle, rows, ids;
  /// The number of rows that a buffer of the owned or the ghost rows has.
  Value capacity;
  /// A layout: the number of particles of the set.
  Value count;
  /// A field (memref<?x?x?xf64>) or a partial sum (memref<?x?xf64>).
  Value buffer;
  /// A state (D399): the layout handle, the halo
  /// handle, the halo's reference positions, and one buffer per field.
  SmallVector<Value> state;
};

/// What the second executor keeps of a relation, for each domain: the
/// positions of its local rows, owned then ghost, as a field of md_exec,
/// its neighbor structure, and its number of owned rows, the centers.
struct DomainRelation {
  Value positions, neighbors, owned, cell, centers;
};

class Lowering {
public:
  /// With `throughExec`, the loops of each domain are those of md_exec
  /// (D290): a structure holds `width` neighbors of a row at first,
  /// in cells of 1/`cellsPerReach` of its reach.
  Lowering(ModuleOp module, bool throughExec = false, int64_t width = 64,
           int64_t cellsPerReach = 3, bool haloEvents = false, bool mpi = false, bool parallelDomains = false)
      : module(module), loc(UnknownLoc::get(module.getContext())),
        throughExec(throughExec), width(width), cellsPerReach(cellsPerReach),
        haloEvents(haloEvents), mpi(mpi), parallelDomains(parallelDomains) {}

  bool meshMixed = false;
  int64_t meshX = 1, meshY = 1;
  bool fixedPoint = false;
  bool matrixPairs = false;
  double matrixSkin = 0;
  LogicalResult run(func::FuncOp function);

private:
  LogicalResult lower(Operation *op);

  //===--------------------------------------------------------------------===//
  // The second executor (D290)
  //===--------------------------------------------------------------------===//

  LogicalResult lowerNeighborhoodExec(NeighborhoodOp op);
  DomainRelation buildLocalRelation(OpBuilder &b, NeighborhoodOp op,
                                    Value d, FlatSymbolRefAttr set);
  void forPairDomains(OpBuilder &b, Value rel,
      function_ref<void(OpBuilder &, Value, FlatSymbolRefAttr, DomainRelation)> emit);
  LogicalResult lowerPairGroup(Operation *anchor, DominanceInfo &dominance);
  scf::ForOp domainLoop;
  DomainRelation domainRelation;
  md_exec::PairForOp lastPair;
  llvm::DenseSet<Operation *> grouped;
  llvm::DenseMap<std::pair<Value, Value>, Value> domainFields;
  LogicalResult lowerSumPairsExec(SumPairsOp op);
  LogicalResult lowerGatherPairsExec(GatherPairsOp op);
  LogicalResult lowerSumParticlesExec(SumParticlesOp op);
  LogicalResult lowerMapParticlesExec(MapParticlesOp op);
  /// Declares the particle sets of the owned and the local rows of each
  /// domain, once per module.
  void declareSets();
  /// The particle set of the owned or the local rows of domain `d`.
  FlatSymbolRefAttr getSet(int64_t d, bool local, Value epoch);
  /// The number of rows of domain `d` that the rows of `rows`, a layout or
  /// a halo, give it, as an index.
  Value countOf(OpBuilder &b, Storage &rows, Value d);
  /// A field of md_exec on `set` of `k` components of `element`, of the
  /// rows of domain `d` of each buffer of `parts`, one after another, with
  /// the number of rows of each in `counts`: a buffer of all domains
  /// (memref<D x rows x k>) or of the identities (memref<D x rows>).
  Value makeField(OpBuilder &b, FlatSymbolRefAttr set, int64_t k,
                  Type element, Value d, ArrayRef<Value> parts,
                  ArrayRef<Value> counts, Value ghostShift = {});
  /// Copies `count` rows of `field`, a field of md_exec, from row `first`
  /// on, to the rows of domain `d` of `buffer`, a buffer of all domains.
  void copyBack(OpBuilder &b, Value field, Value first, Value count,
                Value buffer, Value d);
  /// Gives `loop`, a loop over pairs of md_exec, the kernel of the pair op
  /// `op` of md_dist: the distance from its square, the kernel inlined, and
  /// its value, for a unique relation, where the identity of the center is
  /// the lesser, and multiplied by `scale`.
  void addPairKernel(md_exec::PairForOp loop, Operation *op, bool unique,
                     double scale);
  /// The loop over the pairs of domain `d` of the pair op `op` (a
  /// `sum_pairs` or a `gather_pairs`), with the destinations `outs` and the
  /// sums `reduce`, and the policy `(directed, atomic)` for `atomic`.
  md_exec::PairForOp createPairLoop(OpBuilder &b, Operation *op, Value d, FlatSymbolRefAttr set, DomainRelation relation,
                                    TypeRange results, ValueRange outs,
                                    ValueRange reduce, DenseF64ArrayAttr weights,
                                    bool atomic, double scale);
  llvm::DenseMap<Value, Value> trajectoryCaches;
  // A native reporting block follows a completed plain-step loop.
  struct InnerEpoch {
    scf::ForOp loop;
    PartitionOp partition;
    Type haloType;
    int64_t mode;
    Value cache;
  };
  SmallVector<InnerEpoch> innerEpochs;
  llvm::DenseMap<std::pair<Operation *, Value>, Value> scopedExclusions;
  int64_t haloMode(HaloOp op);
  llvm::DenseMap<std::pair<Block *,Value>,Value> topologyShards;
  llvm::DenseMap<Value, Value> tupleSchedules;
  // A schedule used solely by tuple kernels is read-only. Group row
  // numbers depend on the group layout alone.
  using TupleRowKey = std::tuple<Block *, Value, Value, Value, Type>;
  llvm::DenseMap<TupleRowKey, Value> sharedTupleRows;
  llvm::DenseSet<std::pair<Block *, Value>> sharedTuplePositions;
  LogicalResult lowerPartition(PartitionOp op);
  LogicalResult lowerMigrate(MigrateOp op);
  LogicalResult lowerDistribute(DistributeOp op);
  LogicalResult lowerMesh(MeshOp op);
  LogicalResult lowerCollect(CollectOp op);
  LogicalResult lowerRedistribute(RedistributeOp op);
  LogicalResult lowerHalo(HaloOp op);
  LogicalResult lowerCovered(CoveredOp op);
  LogicalResult lowerHaloCopy(HaloCopyOp op);
  struct HaloGroup {
    Value halo, handle;
    SmallVector<Operation *> transfers;
  };
  SmallVector<HaloGroup> haloGroups;
  llvm::DenseMap<Operation *, unsigned> haloGroupOf;
  void planHaloGroups(ArrayRef<Operation *> ops);
  bool batchHalo(OpBuilder &b, Operation *op, Value halo, Value owned, Value ghosts,
                 Value imageOrOutput, bool transpose, bool words);
  LogicalResult lowerHaloSum(HaloSumOp op);
  LogicalResult lowerZeros(ZerosOp op);
  LogicalResult lowerTupleKernel(TupleKernelOp op);
  LogicalResult lowerTupleRows(TupleRowsOp op);
  LogicalResult lowerNeighborhood(NeighborhoodOp op);
  LogicalResult lowerPartitionCenters(PartitionCentersOp op);
  template <typename OpTy>
  void emitPairs(
      OpTy op, Location loc, Value first, Value past, bool parallel,
      llvm::DenseMap<Value, Storage> &storage, Type accumulator, Value shifts,
      function_ref<Value(OpBuilder &, Value, Value)> add,
      function_ref<Value(OpBuilder &, Type)> zero,
      function_ref<Value(OpBuilder &, Value, Value, Value, Type)> load,
      function_ref<Value(OpBuilder &, Block &, ValueRange)> kernel,
      function_ref<Value(OpBuilder &, Value value, Value acc, Value d, Value k,
                         Value isOwned, Value ghostRow)>
          visit,
      function_ref<void(OpBuilder &, Value acc, Value d, Value i)> finishCenter,
      function_ref<void(OpBuilder &, Value acc, Value d)> finishDomain);
  LogicalResult lowerSumPairs(SumPairsOp op);
  LogicalResult lowerGatherPairs(GatherPairsOp op);
  LogicalResult lowerSumParticles(SumParticlesOp op);
  DenseMap<Operation *,Value> shardExtent;
  void shardDriverLoop(scf::ForOp loop);
  Value bufferAddress(OpBuilder &b, Value buffer) {
    return arith::IndexCastOp::create(b,loc,b.getI64Type(),
        memref::ExtractAlignedPointerAsIndexOp::create(b,loc,buffer));
  }
  void lowerCollectedSum(md::SumParticlesOp op);
  LogicalResult lowerMapParticles(MapParticlesOp op);
  LogicalResult lowerAllReduce(AllReduceOp op);
  LogicalResult lowerPack(PackOp op);
  LogicalResult lowerUnpack(UnpackOp op);
  /// The tuple a state value lowers to.
  SmallVector<Value> stateOf(Value state);
  /// A copy of `buffer`, a buffer of all domains, that the state owns: not
  /// freed at the end of the block.
  Value ownedCapacity(OpBuilder &b, Value handle);
  std::pair<Value,Value> rowViews(OpBuilder &b, Value handle, bool ghosts);
  llvm::SmallPtrSet<Operation *, 8> constraintKernels;
  Value copyForState(OpBuilder &b, Value buffer);

  /// The decomposition of a type of md_dist, and its number of domains.
  DecompositionOp getDecomposition(Operation *op, FlatSymbolRefAttr name);
  /// The edge lengths of the cell `cell`, which must come from
  /// md.orthorhombic_cell.
  LogicalResult getEdges(Operation *op, Value cell, Value edges[3]);
  /// The cell a halo was built in (its md_dist.halo or md_dist.unpack).
  LogicalResult getHaloEdges(Operation *op, Value halo, Value edges[3]);

  func::FuncOp declare(StringRef name, FunctionType type, bool cInterface);
  Value index(OpBuilder &b, int64_t value) {
    return arith::ConstantIndexOp::create(b, loc, value);
  }
  Value real(OpBuilder &b, double value) {
    return arith::ConstantOp::create(b, loc, b.getF64FloatAttr(value));
  }
  Value toIndex(OpBuilder &b, Value value) {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), value);
  }
  /// A zero of `type`, f64 or a vector of f64.
  Value zero(OpBuilder &b, Type type);
  /// The fixed-point format of the words of one quantity in the
  /// deterministic mode: its fractional bits, and the quantity that a range
  /// error names (runtime/mdrt_dist_stats.h).
  struct Format {
    int bits;
    DistQuantity quantity;
  };
  static constexpr Format pairEnergy{32, DistPairEnergy};
  static constexpr Format pairForce{40, DistPairForce};
  static constexpr Format particleSum{32, DistParticleSum};
  /// The format of a sum of particles that the driver collects, from the
  /// quantity that the builder names on it (`md_dist.quantity`).
  static Format getCollectedFormat(Operation *op);
  Type sumType(Type type);
  Value quantize(OpBuilder &b, Value value, Format format);
  Value finish(OpBuilder &b, Value sum, Format format);
  Value add(OpBuilder &b, Value lhs, Value rhs);
  llvm::DenseMap<Value, Value> imageShifts, fieldWords, excludedPairs;
  llvm::DenseMap<Value, std::pair<Value, Format>> observableWords;
  Value decodeObservable(OpBuilder &b, Value sum, Format format);
  bool needsWords(Value field) {
    return fixedPoint && llvm::any_of(field.getUsers(), [](Operation *user) {
             return user->hasAttr("md_dist.add_terms");
           });
  }
  /// A new buffer of `domains x rows x k` of f64, freed at the end.
  Value allocate(OpBuilder &b, Value rows, int64_t k);
  Value inputSnapshot(OpBuilder &b, md::FieldType type, Value field, Value indices = {});
  /// A new buffer of the given type and sizes, freed at the end.
  Value allocate(OpBuilder &b, MemRefType type, ValueRange sizes);
  void initializeBuffer(OpBuilder &b, Value buffer, bool domainRows, Value source = {});
  /// Loads the value of row `row` of domain `domain` of `buffer`: an f64,
  /// or a vector<3xf64> for three components.
  Value loadValue(OpBuilder &b, Value buffer, Value domain, Value row,
                  Type type);
  void storeValue(OpBuilder &b, Value value, Value buffer, Value domain,
                  Value row);
  void addValue(OpBuilder &b, Value value, Value buffer, Value domain,
                Value row);
  /// Inlines `kernel` with `arguments` and returns the value it yields.
  void cloneKernelOp(OpBuilder &b, Operation &op, IRMapping &mapping);
  Value inlineKernel(OpBuilder &b, Block &kernel, ValueRange arguments);

  /// The layout or halo storage of the rows of a field.
  Storage &rowsOf(Value field);

  ModuleOp module;
  Location loc;
  llvm::DenseMap<Value, Storage> storage;
  /// Where the buffers are freed and the handles released: before the
  /// terminator of the block of the op being lowered. A buffer of a step of a
  /// loop is freed at the end of the step; only a state outlives its block.
  Operation *end = nullptr;
  int64_t domains = 0;
  DecompositionOp decomposition;

  bool throughExec;
  int64_t width, cellsPerReach;
  bool haloEvents, mpi, parallelDomains;
  Value domainBound(OpBuilder &b, bool end) {
    if (!mpi)
      return index(b, end ? domains : 0);
    auto i64 = b.getI64Type();
    auto bound = declare("mdrtMPIDomainBound", b.getFunctionType({i64, i64}, {i64}), false);
    Value n = arith::ConstantIntOp::create(b, loc, domains, 64);
    Value side = arith::ConstantIntOp::create(b, loc, end, 64);
    return toIndex(b, func::CallOp::create(b, loc, bound, ValueRange{n, side}).getResult(0));
  }
  llvm::DenseMap<Value, SmallVector<DomainRelation>> relations;
  /// The epochs of the sets of md_exec: layouts and halos, numbered.
  llvm::DenseMap<Value, int64_t> ownedEpochs, localEpochs;
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

DecompositionOp Lowering::getDecomposition(Operation *op,
                                           FlatSymbolRefAttr name) {
  auto found = SymbolTable::lookupNearestSymbolFrom<DecompositionOp>(op, name);
  if (!found)
    op->emitOpError() << "'" << name.getValue()
                      << "' does not name a decomposition";
  return found;
}

LogicalResult Lowering::getEdges(Operation *op, Value cell, Value edges[3]) {
  auto box = cell.getDefiningOp<md::OrthorhombicCellOp>();
  if (!box)
    return op->emitOpError()
           << "the reference executor runs an orthorhombic cell, from "
              "md.orthorhombic_cell, only";
  edges[0] = box.getLx();
  edges[1] = box.getLy();
  edges[2] = box.getLz();
  return success();
}

LogicalResult Lowering::getHaloEdges(Operation *op, Value halo,
                                     Value edges[3]) {
  Value cell = cellOf(halo);
  if (!cell)
    return op->emitOpError() << "cannot find the cell of the halo";
  return getEdges(op, cell, edges);
}

func::FuncOp Lowering::declare(StringRef name, FunctionType type,
                               bool cInterface) {
  if (Operation *existing = SymbolTable::lookupSymbolIn(module, name))
    return cast<func::FuncOp>(existing);
  auto function = func::FuncOp::create(module.getLoc(), name, type);
  function.setPrivate();
  if (cInterface)
    function->setAttr("llvm.emit_c_interface", UnitAttr::get(type.getContext()));
  module.push_back(function);
  return function;
}

// All terms are rounded before entering any center or domain subtotal.
Type Lowering::sumType(Type type) {
  if (!fixedPoint)
    return type;
  Type word = IntegerType::get(module.getContext(), 128);
  if (auto vector = dyn_cast<VectorType>(type))
    return VectorType::get(vector.getShape(), word);
  return word;
}

Lowering::Format Lowering::getCollectedFormat(Operation *op) {
  auto name = op->getAttrOfType<StringAttr>("md_dist.quantity");
  if (!name)
    return particleSum;
  // Each format is fitted to its quantity. A kinetic energy keeps Q32, the
  // format of the potential energy, so that the total energy is a sum of
  // words. Sum F^2/m of the kinetic correction is a square of forces of Q40
  // range (below 2^23): Q16 leaves it 2^47 in (kJ/mol/nm)^2/amu, and a
  // quantum that dt^2/8 scales below 10^-10 kJ/mol per particle.
  return llvm::StringSwitch<Format>(name.getValue())
      .Case("kinetic energy", Format{32, DistKinetic})
      .Case("momentum", Format{32, DistMomentum})
      .Case("force square", Format{16, DistForceSquare})
      .Default(particleSum);
}

Value Lowering::quantize(OpBuilder &b, Value value, Format format) {
  if (!fixedPoint)
    return value;
  // The runtime checks the rounded value before casting. There is no
  // floating-to-integer instruction with an out-of-range operand.
  auto convert = declare(
      "mdrtDistWord",
      b.getFunctionType({b.getF64Type(), b.getF64Type(), b.getI64Type()},
                        {b.getI64Type()}),
      false);
  Value scale = real(b, double(uint64_t{1} << format.bits));
  Value quantity = arith::ConstantIntOp::create(b, loc, format.quantity, 64);
  auto scalar = [&](Value x) {
    Value word =
        func::CallOp::create(b, loc, convert, ValueRange{x, scale, quantity})
            .getResult(0);
    return arith::ExtSIOp::create(b, loc, b.getIntegerType(128), word)
        .getResult();
  };
  if (auto vector = dyn_cast<VectorType>(value.getType())) {
    SmallVector<Value> parts;
    for (int64_t c = 0; c < vector.getNumElements(); ++c)
      parts.push_back(scalar(vector::ExtractOp::create(b, loc, value, c)));
    return vector::FromElementsOp::create(
        b, loc, cast<VectorType>(sumType(vector)), parts);
  }
  return scalar(value);
}

// The conversion of a word of 64 bits to binary64 is exact up to 2^53 and
// rounds to nearest above: a function of the word alone, so that the value
// is the same on every grid either way.
Value Lowering::finish(OpBuilder &b, Value sum, Format format) {
  if (!fixedPoint)
    return sum;
  Type type = b.getF64Type();
  auto narrow =
      declare("mdrtDistNarrowWord",
              b.getFunctionType({b.getIntegerType(128), b.getI64Type()},
                                {b.getI64Type()}),
              false);
  Value quantity = arith::ConstantIntOp::create(b, loc, format.quantity, 64);
  auto scalar = [&](Value v) {
    return func::CallOp::create(b, loc, narrow, ValueRange{v, quantity})
        .getResult(0);
  };
  if (auto vector = dyn_cast<VectorType>(sum.getType())) {
    SmallVector<Value> parts;
    for (int64_t c = 0; c < vector.getNumElements(); ++c)
      parts.push_back(scalar(vector::ExtractOp::create(b, loc, sum, c)));
    sum = vector::FromElementsOp::create(
        b, loc, VectorType::get(vector.getShape(), b.getI64Type()), parts);
  } else
    sum = scalar(sum);
  Value scale = real(b, 1.0 / double(uint64_t{1} << format.bits));
  if (auto vector = dyn_cast<VectorType>(sum.getType())) {
    type = VectorType::get(vector.getShape(), type);
    scale = vector::BroadcastOp::create(b, loc, type, scale);
  }
  return arith::MulFOp::create(
      b, loc, arith::SIToFPOp::create(b, loc, type, sum), scale);
}

Value Lowering::decodeObservable(OpBuilder &b, Value sum, Format format) {
  Value value = finish(b, sum, format);
  if (fixedPoint)
    observableWords[value] = {sum, format};
  return value;
}

Value Lowering::add(OpBuilder &b, Value lhs, Value rhs) {
  if (!fixedPoint)
    return arith::AddFOp::create(b, loc, lhs, rhs);
  // The count certificate bounds every pair subtotal by N(N-1) * 2^63.
  // Check the wider sums across different terms as well.
  auto wide = b.getIntegerType(128);
  auto checked = declare("mdrtDistAddWide",
                         b.getFunctionType({wide, wide}, {wide}), false);
  auto scalar = [&](Value a, Value v) {
    return func::CallOp::create(b, loc, checked, ValueRange{a, v}).getResult(0);
  };
  if (auto vector = dyn_cast<VectorType>(lhs.getType())) {
    SmallVector<Value> parts;
    for (int64_t c = 0; c < vector.getNumElements(); ++c)
      parts.push_back(scalar(vector::ExtractOp::create(b, loc, lhs, c),
                             vector::ExtractOp::create(b, loc, rhs, c)));
    return vector::FromElementsOp::create(b, loc, vector, parts);
  }
  return scalar(lhs, rhs);
}

Value Lowering::zero(OpBuilder &b, Type type) {
  if (isa<IntegerType>(getElementTypeOrSelf(type)))
    return arith::ConstantOp::create(b, loc, b.getZeroAttr(type));
  if (auto vector = dyn_cast<VectorType>(type))
    return arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(vector, b.getF64FloatAttr(0.0)));
  return real(b, 0.0);
}

Value Lowering::allocate(OpBuilder &b, MemRefType type, ValueRange sizes) {
  Value buffer = memref::AllocOp::create(b, loc, type, sizes);
  OpBuilder atEnd(end);
  memref::DeallocOp::create(atEnd, loc, buffer);
  return buffer;
}

Value Lowering::allocate(OpBuilder &b, Value rows, int64_t k) {
  auto type = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic,
                               ShapedType::kDynamic},
                              b.getF64Type());
  return allocate(b, type, {index(b, domains), rows, index(b, k)});
}

// Field descriptors retain global domain indices for transport. Touch only
// the local slabs; collective consumers fill remote rows on demand.
void Lowering::initializeBuffer(OpBuilder &b, Value buffer, bool domainRows, Value source) {
  auto type = cast<MemRefType>(buffer.getType());
  Value z = index(b, 0), one = index(b, 1);
  SmallVector<Value> lower, upper, steps;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    lower.push_back(domainRows && i == 0 ? domainBound(b, false) : z);
    upper.push_back(domainRows && i == 0 ? domainBound(b, true) :
                    Value(memref::DimOp::create(b, loc, buffer, i)));
    steps.push_back(one);
  }
  Value init = source ? Value() : zero(b, type.getElementType());
  scf::ParallelOp::create(b, loc, lower, upper, steps,
      [&](OpBuilder &at, Location, ValueRange indices) {
        Value v = source ? Value(memref::LoadOp::create(at, loc, source, indices)) : init;
        memref::StoreOp::create(at, loc, v, buffer, indices);
      });
}

Value Lowering::loadValue(OpBuilder &b, Value buffer, Value domain, Value row,
                          Type type) {
  auto vector = dyn_cast<VectorType>(type);
  int64_t k = vector ? vector.getNumElements() : 1;
  SmallVector<Value> parts;
  for (int64_t c = 0; c < k; ++c)
    parts.push_back(memref::LoadOp::create(
        b, loc, buffer, ValueRange{domain, row, index(b, c)}));
  if (!vector) {
    if (type.isInteger(32) && parts[0].getType().isF64())
      return arith::FPToSIOp::create(b, loc, type, parts[0]);
    return parts[0];
  }
  return vector::FromElementsOp::create(b, loc, vector, parts);
}

void Lowering::storeValue(OpBuilder &b, Value value, Value buffer,
                          Value domain, Value row) {
  auto vector = dyn_cast<VectorType>(value.getType());
  int64_t k = vector ? vector.getNumElements() : 1;
  for (int64_t c = 0; c < k; ++c) {
    Value part =
        vector ? Value(vector::ExtractOp::create(b, loc, value, c)) : value;
    if (part.getType().isInteger(32) && cast<MemRefType>(buffer.getType()).getElementType().isF64())
      part = arith::SIToFPOp::create(b, loc, b.getF64Type(), part);
    memref::StoreOp::create(b, loc, part, buffer,
                            ValueRange{domain, row, index(b, c)});
  }
}

void Lowering::addValue(OpBuilder &b, Value value, Value buffer, Value domain,
                        Value row) {
  Value old = loadValue(b, buffer, domain, row, value.getType());
  storeValue(b, add(b, old, value), buffer, domain, row);
}

void Lowering::cloneKernelOp(OpBuilder &b, Operation &op, IRMapping &mapping) {
  auto lookup = dyn_cast<md::LookupOp>(op);
  if (!lookup) {
    b.clone(op, mapping);
    return;
  }
  Value table = lookup.getTable();
  Value buffer;
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfterValue(table);
    buffer = mdrt::ToBufferOp::create(b, loc, mdrt::getTableBufferType(cast<md::TableType>(table.getType())), table);
  }
  SmallVector<Value> indices;
  for (Value i : lookup.getIndices()) {
    Value value = mapping.lookupOrDefault(i);
    indices.push_back(value.getType().isIndex() ? value : toIndex(b, value));
  }
  mapping.map(lookup.getResult(), memref::LoadOp::create(b, loc, buffer, indices));
}

Value Lowering::inlineKernel(OpBuilder &b, Block &kernel,
                             ValueRange arguments) {
  IRMapping mapping;
  mapping.map(kernel.getArguments(), arguments);
  for (Operation &op : kernel.without_terminator())
    cloneKernelOp(b, op, mapping);
  auto yield = cast<YieldOp>(kernel.getTerminator());
  return mapping.lookupOrDefault(yield.getValues()[0]);
}

Storage &Lowering::rowsOf(Value field) {
  auto type = cast<FieldType>(field.getType());
  Value rows = type.isOwned() ? layoutOf(field) : haloOf(field);
  return storage[rows];
}

//===----------------------------------------------------------------------===//
// Ownership and halos
//===----------------------------------------------------------------------===//

// Partition and distribution read the input while the ordinary step still
// owns it. A snapshot gives those reads block lifetime, even when a later
// kick or drift reuses the field's storage.
Value Lowering::inputSnapshot(OpBuilder &b, md::FieldType type, Value field, Value indices) {
  if (!module->hasAttr("md_dist.driver"))
    return mdrt::ToBufferOp::create(b, loc, mdrt::getBufferType(type), field);
  auto snapshot = mdrt::SnapshotOp::create(b, loc, mdrt::getBufferType(type), field, indices);
  OpBuilder atEnd(end);
  memref::DeallocOp::create(atEnd, loc, snapshot.getResult());
  return snapshot.getResult();
}

std::pair<Value,Value> Lowering::rowViews(OpBuilder &b, Value handle, bool ghosts) {
  Type i64 = b.getI64Type();
  auto view = [&](StringRef name, ArrayRef<int64_t> shape) {
    constexpr int64_t dyn = ShapedType::kDynamic;
    Type type = shape.size() == 1 ? MemRefType::get(shape,i64) :
        MemRefType::get(shape,i64,StridedLayoutAttr::get(b.getContext(),dyn,{dyn,1}));
    auto fn = declare(name, b.getFunctionType({i64,i64}, {type}), true);
    return func::CallOp::create(b, loc, fn, ValueRange{
        handle, arith::ConstantIntOp::create(b, loc, ghosts, 64)}).getResult(0);
  };
  constexpr int64_t dyn = ShapedType::kDynamic;
  return {view("mdrtDistRowView", {dyn}), view("mdrtDistIdView", {dyn,dyn})};
}

Value Lowering::ownedCapacity(OpBuilder &b, Value handle) {
  Type i64 = b.getI64Type();
  auto capacity = declare("mdrtDistOwnedCapacity", b.getFunctionType({i64}, {i64}), false);
  return toIndex(b, func::CallOp::create(b, loc, capacity, ValueRange{handle}).getResult(0));
}

int64_t Lowering::haloMode(HaloOp op) {
  bool midpoint = matrixPairs && module->hasAttr("md_dist.driver") &&
      llvm::any_of(op.getResult().getUsers(), [](Operation *user) {
        auto n = dyn_cast<NeighborhoodOp>(user);
        return n && n.getResult().getType().getTraversal() == Traversal::Midpoint;
      }) && llvm::all_of(op.getResult().getUsers(), [](Operation *user) {
        auto n = dyn_cast<NeighborhoodOp>(user);
        return !n || n.getResult().getType().getTraversal() == Traversal::Midpoint;
      });
  bool local = mpi && module->hasAttr("md_dist.driver") &&
      llvm::all_of(op.getResult().getUsers(), [](Operation *user) {
        auto rows = dyn_cast<TupleRowsOp>(user);
        return !rows || llvm::all_of(rows.getRows().getUsers(),
            [](Operation *use) { return isa<TupleKernelOp>(use); });
      });
  return midpoint + (local ? 2 : 0);
}

LogicalResult Lowering::lowerPartition(PartitionOp op) {
  OpBuilder b(op);
  Value edges[3];
  if (failed(getEdges(op, op.getCell(), edges)))
    return failure();
  auto fieldType = cast<md::FieldType>(op.getPositions().getType());
  if (!fieldType.getElementType().isF64())
    return op.emitOpError() << "the reference executor runs f64 only";
  Value positions;
  if (mpi && module->hasAttr("md_dist.driver")) {
    // Partition and group registration finish their reads in this block.
    auto view = mdrt::ToBufferOp::create(b,loc,mdrt::getBufferType(fieldType),op.getPositions());
    view->setAttr("mdrt.partition_view",b.getUnitAttr());
    positions = view;
  } else {
    positions = inputSnapshot(b,fieldType,op.getPositions());
  }
  if (fixedPoint) {
    Value count = shardExtent.lookup(op);
    if (!count) count = memref::DimOp::create(b, loc, positions, 0);
    auto check = declare("mdrtDistCheckWordCount", b.getFunctionType({b.getI64Type()}, {}), false);
    func::CallOp::create(b, loc, check, ValueRange{
        arith::IndexCastOp::create(b, loc, b.getI64Type(), count)});
  }
  auto wide = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                              b.getF64Type());
  Value cast = memref::CastOp::create(b, loc, wide, positions);
  ArrayRef<int64_t> grid = decomposition.getGrid();
  Type i64 = b.getI64Type();
  // The native constrained integrator materializes global coordinates. Keep
  // its pair epoch across iterations with the same particle numbering.
  // Relation operands after a spatial permutation bound the cache lifetime.
  Value cache;
  if (matrixPairs && matrixSkin > 0 && module->hasAttr("md_dist.driver")) {
    HaloOp pairHalo;
    for (Operation *user : op.getLayout().getUsers())
      if (auto h = dyn_cast<HaloOp>(user))
        if (llvm::any_of(h.getResult().getUsers(), [](Operation *use) {
              return isa<NeighborhoodOp>(use);
            }))
          pairHalo = h;
    if (pairHalo) {
      scf::ForOp scope;
      for (auto loop = op->getParentOfType<scf::ForOp>(); loop;
           loop = loop->getParentOfType<scf::ForOp>()) {
        if (!loop.isDefinedOutsideOfLoop(op.getCell()) ||
            !llvm::all_of(op.getGroups(), [&](Value g) { return loop.isDefinedOutsideOfLoop(g); }))
          break;
        scope = loop;
      }
      if (scope) {
        auto inner = op->getParentOfType<scf::ForOp>();
        Type haloType = pairHalo.getResult().getType();
        int64_t mode = haloMode(pairHalo);
        if (op->getParentOp() == scope) {
          for (auto it = innerEpochs.begin(); it != innerEpochs.end(); ++it) {
            PartitionOp prior = it->partition;
            if (it->loop->getParentOp() == scope && it->loop->isBeforeInBlock(op) &&
                prior.getPositions().getType() == fieldType &&
                prior.getCell() == op.getCell() && prior.getGroups() == op.getGroups() &&
                shardExtent.lookup(prior) == shardExtent.lookup(op) &&
                it->haloType == haloType && it->mode == mode) {
              cache = it->cache;
              innerEpochs.erase(it);
              break;
            }
          }
        }
        if (!cache) {
          OpBuilder init(scope);
          cache = func::CallOp::create(init, loc,
              declare("mdrtDistEpochCreate", b.getFunctionType({}, {i64}), false)).getResult(0);
          OpBuilder after(scope->getBlock(), std::next(scope->getIterator()));
          func::CallOp::create(after, loc,
              declare("mdrtDistEpochDestroy", b.getFunctionType({i64}, {}), false), ValueRange{cache});
          if (op->getParentOp() == inner && inner->getParentOp() == scope)
            innerEpochs.push_back({inner, op, haloType, mode, cache});
        }
        trajectoryCaches[pairHalo.getResult()] = cache;
        trajectoryCaches[pairHalo.getPositions()] = cache;
      }
    }
  }
  SmallVector<Value> args{cast};
  for (int64_t n : grid)
    args.push_back(arith::ConstantIntOp::create(b, loc, n, 64));
  llvm::append_range(args, edges);
  StringRef name = mpi ? "mdrtDistStartPartition" : "mdrtDistPartition";
  if (cache) {
    name = "mdrtDistEpochPartition";
    args.append({cache, real(b, matrixSkin), arith::ConstantIntOp::create(b, loc, mpi, 64)});
  }
  Value globalCount = shardExtent.lookup(op);
  if (globalCount) {
    name = cache ? "mdrtDistShardEpochPartition" : "mdrtDistShardPartition";
    args.push_back(arith::IndexCastOp::create(b,loc,i64,globalCount));
  }
  Value handle = func::CallOp::create(b, loc,
      declare(name, b.getFunctionType(ValueRange(args).getTypes(), {i64}), true), args).getResult(0);
  for (Value group : op.getGroups()) {
    auto type = mdrt::getMembersType(mlir::cast<md::RelationType>(group.getType()));
    Value members = mdrt::ToBufferOp::create(b, loc, type, group);
    auto dynamic = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, b.getI32Type());
    Value view = memref::CastOp::create(b, loc, dynamic, members);
    auto apply = declare(globalCount ? "mdrtDistShardGroupOwners" : "mdrtDistGroupOwners", b.getFunctionType({i64, wide, dynamic}, {}), true);
    func::CallOp::create(b, loc, apply, ValueRange{handle, cast, view});
  }
  if (mpi)
    func::CallOp::create(b,loc,declare("mdrtDistEndPartition",
        b.getFunctionType({i64}, {}),false),handle);
  Value count = globalCount ? globalCount : Value(memref::DimOp::create(b, loc, positions, 0));
  Value capacity = ownedCapacity(b, handle);
  auto [rows, ids] = rowViews(b, handle, false);
  OpBuilder atEnd(end);
  func::CallOp::create(
      atEnd, loc,
      declare("mdrtDistReleaseLayout", b.getFunctionType({i64}, {}), false),
      ValueRange{handle});
  storage[op.getLayout()] = {handle, rows, ids, capacity, count, Value()};
  return success();
}

static bool isMigrationPosition(RedistributeOp field) {
  auto move = field.getNewLayout().getDefiningOp<MigrateOp>();
  return field.getImage() && move && move.getPositions() == field.getField();
}

LogicalResult Lowering::lowerMigrate(MigrateOp op) {
  OpBuilder b(op);
  if (!op.getPositions().getType().getElementType().isF64())
    return op.emitOpError() << "the in-process executor runs f64 fields only";
  Value edges[3];
  if (failed(getEdges(op, op.getCell(), edges)))
    return failure();
  Storage &old = storage[op.getOldLayout()];
  Value positions = storage[op.getPositions()].buffer;
  Type i64 = b.getI64Type(), f64 = b.getF64Type();
  bool keepPositions = mpi && llvm::any_of(op.getLayout().getUsers(), [&](Operation *use) {
    auto field = dyn_cast<RedistributeOp>(use);
    return field && field.getNewLayout() == op.getLayout() && isMigrationPosition(field);
  });
  func::FuncOp migrate = declare(
      keepPositions ? "mdrtDistMigratePositions" : "mdrtDistMigrate",
      b.getFunctionType({i64, positions.getType(), f64, f64, f64}, {i64}),
      true);
  Value handle = func::CallOp::create(
                     b, loc, migrate,
                     ValueRange{old.handle, positions, edges[0], edges[1],
                                edges[2]})
                     .getResult(0);
  Value capacity = ownedCapacity(b, handle);
  auto [rows, ids] = rowViews(b, handle, false);
  OpBuilder atEnd(end);
  func::CallOp::create(
      atEnd, loc,
      declare("mdrtDistReleaseLayout", b.getFunctionType({i64}, {}), false),
      ValueRange{handle});
  storage[op.getLayout()] = {handle, rows, ids, capacity, old.count,
                             Value()};
  return success();
}

LogicalResult Lowering::lowerDistribute(DistributeOp op) {
  OpBuilder b(op);
  auto fieldType = cast<md::FieldType>(op.getField().getType());
  if (!fieldType.getElementType().isF64() && !fieldType.getElementType().isInteger(32))
    return op.emitOpError() << "the CPU executor transports f64 fields and i32 particle types";
  Value edges[3];
  if (op.getImage() && failed(getEdges(op, op.getImage(), edges)))
    return failure();
  Storage &layout = storage[op.getLayout()];
  int64_t k = fieldType.getNumComponents();
  Value first = domainBound(b, false), past = domainBound(b, true);
  Value z = index(b, 0), one = index(b, 1), offsets, ids;
  if (shardExtent.count(op)) {
    auto view = mdrt::ToBufferOp::create(b,loc,mdrt::getBufferType(fieldType),op.getField());
    view->setAttr("mdrt.partition_view",b.getUnitAttr());
    Value owned = allocate(b,layout.capacity,k);
    auto i64 = b.getI64Type();
    auto fn = declare("mdrtDistDistributeShard",b.getFunctionType({i64,i64,owned.getType(),i64,i64},{}),true);
    func::CallOp::create(b,loc,fn,ValueRange{layout.handle,bufferAddress(b,view),owned,
        arith::ConstantIntOp::create(b,loc,fieldType.getElementType().isInteger(32),64),
        arith::ConstantIntOp::create(b,loc,bool(op.getImage()),64)});
    if (Value cache = trajectoryCaches.lookup(op.getResult())) {
      SmallVector<Value> args{cache, layout.handle, owned};
      func::CallOp::create(b,loc,declare("mdrtDistEpochImages",
          b.getFunctionType(ValueRange(args).getTypes(),{}),true),args);
    }
    storage[op.getResult()].buffer = owned;
    return success();
  }
  if (mpi && module->hasAttr("md_dist.driver")) {
    auto indexRows = MemRefType::get({ShapedType::kDynamic}, b.getIndexType());
    offsets = allocate(b, indexRows, {index(b, domains)});
    Value total = scf::ForOp::create(b, loc, first, past, one, ValueRange{z},
        [&](OpBuilder &at, Location, Value d, ValueRange acc) {
          memref::StoreOp::create(at, loc, acc[0], offsets, d);
          Value n = toIndex(at, memref::LoadOp::create(at, loc, layout.rows, d));
          scf::YieldOp::create(at, loc, arith::AddIOp::create(at, loc, acc[0], n).getResult());
        }).getResult(0);
    ids = allocate(b, indexRows, {total});
    scf::ForOp::create(b, loc, first, past, one, ValueRange(),
        [&](OpBuilder &at, Location, Value d, ValueRange) {
          Value base = memref::LoadOp::create(at, loc, offsets, d);
          Value n = toIndex(at, memref::LoadOp::create(at, loc, layout.rows, d));
          scf::ParallelOp::create(at, loc, ValueRange{z}, ValueRange{n}, ValueRange{one},
              [&](OpBuilder &body, Location, ValueRange r) {
                Value id = toIndex(body, memref::LoadOp::create(body, loc, layout.ids, ValueRange{d,r[0]}));
                memref::StoreOp::create(body, loc, id, ids, arith::AddIOp::create(body, loc, base, r[0]).getResult());
              });
          scf::YieldOp::create(at, loc);
        });
  }
  Value global = inputSnapshot(b, fieldType, op.getField(), ids);
  Value owned = allocate(b, layout.capacity, k);
  // owned[d][r] = field[ids[d][r]], wrapped into the cell with `image`.
  scf::ForOp::create(
      b, loc, first, past, index(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value count = toIndex(
            bd, memref::LoadOp::create(bd, loc, layout.rows, ValueRange{d}));
        Value base = ids ? Value(memref::LoadOp::create(bd, loc, offsets, d)) : Value();
        scf::ParallelOp::create(bd, loc, ValueRange{z}, ValueRange{count}, ValueRange{one},
            [&](OpBuilder &br, Location, ValueRange iv) {
              Value r = iv[0];
              Value j = ids ? Value(arith::AddIOp::create(br, loc, base, r)) :
                  toIndex(br, memref::LoadOp::create(br, loc, layout.ids, ValueRange{d, r}));
              for (int64_t c = 0; c < k; ++c) {
                Value v =
                    k == 1 ? Value(memref::LoadOp::create(br, loc, global,
                                                          ValueRange{j}))
                           : Value(memref::LoadOp::create(
                                 br, loc, global, ValueRange{j, index(br, c)}));
                if (v.getType().isInteger(32))
                  v = arith::SIToFPOp::create(br, loc, br.getF64Type(), v);
                if (op.getImage()) {
                  // v − L floor(v / L)
                  Value u = arith::DivFOp::create(br, loc, v, edges[c]);
                  Value whole = math::FloorOp::create(br, loc, u);
                  v = arith::SubFOp::create(
                      br, loc, v,
                      arith::MulFOp::create(br, loc, whole, edges[c]));
                }
                memref::StoreOp::create(br, loc, v, owned,
                                        ValueRange{d, r, index(br, c)});
              }
            });
        scf::YieldOp::create(bd, loc);
      });
  if (Value cache = trajectoryCaches.lookup(op.getResult())) {
    SmallVector<Value> args{cache, layout.handle, owned};
    func::CallOp::create(b, loc, declare("mdrtDistEpochImages",
        b.getFunctionType(ValueRange(args).getTypes(), {}), true), args);
  }
  storage[op.getResult()].buffer = owned;
  return success();
}

LogicalResult Lowering::lowerMesh(MeshOp op) {
  OpBuilder b(op);
  Value edges[3];
  if (failed(getEdges(op, op.getCell(), edges))) return failure();
  Storage &layout = storage[op.getLayout()];
  Value forces = allocate(b, layout.capacity, 3);
  auto sumsType = MemRefType::get({10}, b.getF64Type());
  Value sums = allocate(b, sumsType, {});
  Value table = mdrt::ToBufferOp::create(b, loc,
      mdrt::getTableBufferType(cast<md::TableType>(op.getModuli().getType())), op.getModuli());
  SmallVector<Value> args{layout.rows, domainBound(b, false), domainBound(b, true),
      storage[op.getPositions()].buffer, storage[op.getCharges()].buffer, table, forces, sums};
  llvm::append_range(args, edges);
  args.append({real(b,op.getBeta().convertToDouble()), real(b,op.getCoulomb().convertToDouble())});
  for (int64_t n : op.getGrid()) args.push_back(index(b,n));
  for (int64_t n : {int64_t(op.getOrder()), meshX, meshY, int64_t(mpi), int64_t(fixedPoint), int64_t(meshMixed)})
    args.push_back(index(b,n));
  args.push_back(index(b, parallelDomains));
  auto solve = declare("mdrtDistPME",
      b.getFunctionType(ValueRange(args).getTypes(), {}), true);
  for (unsigned i : {0u, 3u, 4u, 5u})
    solve.setArgAttr(i, "mdrt.read_only", b.getUnitAttr());
  func::CallOp::create(b, loc, solve, args);
  storage[op.getForces()].buffer = forces;
  op.getEnergy().replaceAllUsesWith(memref::LoadOp::create(b,loc,sums,ValueRange{index(b,0)}));
  SmallVector<Value> virial;
  for (int i=1; i<=9; ++i)
    virial.push_back(memref::LoadOp::create(b,loc,sums,ValueRange{index(b,i)}));
  op.getVirial().replaceAllUsesWith(vector::FromElementsOp::create(b,loc,
      op.getVirial().getType(), virial));
  return success();
}

LogicalResult Lowering::lowerCollect(CollectOp op) {
  OpBuilder b(op);
  auto fieldType = cast<md::FieldType>(op.getResult().getType());
  Storage &layout = storage[op.getLayout()];
  Value owned = storage[op.getField()].buffer;
  // Other consumers can feed the next replicated segment. Only a private
  // checkpoint snapshot can skip sending the gathered field to those peers.
  bool checkpointOnly = llvm::all_of(op.getResult().getUsers(), [](Operation *user) {
    auto call = dyn_cast<mdrt::HostCallOp>(user);
    return call && call.getCallee() == "mdrtWriteCheckpoint";
  });
  Value limit = index(b, domains);
  Value ids = layout.ids;
  int64_t k = fieldType.getNumComponents();
  MemRefType type = mdrt::getBufferType(fieldType);
  // The buffer is handed to the program as a field, which owns it from
  // then on; it is not freed here.
  Value global =
      memref::AllocOp::create(b, loc, type, ValueRange{shardExtent.count(op) ? shardExtent.lookup(op) : layout.count});
  if (mpi) {
    Type i64 = b.getI64Type();
    Value base = arith::IndexCastOp::create(b,loc,i64,
        memref::ExtractAlignedPointerAsIndexOp::create(b,loc,global));
    auto collect = declare("mdrtMPICollectField",
        b.getFunctionType({i64,owned.getType(),i64,i64,i64}, {}), true);
    func::CallOp::create(b,loc,collect,ValueRange{layout.handle,owned,base,
        arith::ConstantIntOp::create(b,loc,fieldType.getElementType().isInteger(32),64),
        arith::ConstantIntOp::create(b,loc,shardExtent.count(op) ? 2 : checkpointOnly,64)});
  } else scf::ForOp::create(
      b, loc, index(b, 0), limit, index(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value rows = toIndex(
            bd, memref::LoadOp::create(bd, loc, layout.rows, ValueRange{d}));
        scf::ForOp::create(
            bd, loc, index(bd, 0), rows, index(bd, 1), ValueRange(),
            [&](OpBuilder &br, Location, Value r, ValueRange) {
              Value j = toIndex(br, memref::LoadOp::create(
                                        br, loc, ids, ValueRange{d, r}));
              for (int64_t c = 0; c < k; ++c) {
                Value v = memref::LoadOp::create(
                    br, loc, owned, ValueRange{d, r, index(br, c)});
                if (fieldType.getElementType().isInteger(32) && v.getType().isF64())
                  v = arith::FPToSIOp::create(br, loc, br.getI32Type(), v);
                if (k == 1)
                  memref::StoreOp::create(br, loc, v, global, ValueRange{j});
                else
                  memref::StoreOp::create(br, loc, v, global,
                                          ValueRange{j, index(br, c)});
              }
              scf::YieldOp::create(br, loc);
            });
        scf::YieldOp::create(bd, loc);
      });
  auto field = mdrt::FromBufferOp::create(b, loc, fieldType, global);
  if (module->hasAttr("md_dist.driver"))
    field->setAttr("mdrt.collect_buffer", b.getUnitAttr());
  op.getResult().replaceAllUsesWith(field);
  return success();
}

LogicalResult Lowering::lowerRedistribute(RedistributeOp op) {
  OpBuilder b(op);
  if (!op.getField().getType().getElementType().isF64() &&
      !op.getField().getType().getElementType().isInteger(32))
    return op.emitOpError() << "the in-process executor runs f64 fields only";
  Storage &old = storage[op.getOldLayout()];
  Storage &next = storage[op.getNewLayout()];
  Value source = storage[op.getField()].buffer;
  int64_t components = op.getField().getType().getNumComponents();
  Value destination = allocate(b, next.capacity, components);
  Type i64 = b.getI64Type();
  if (mpi && isMigrationPosition(op)) {
    auto fn = declare("mdrtDistMigratedPositions",
        b.getFunctionType({i64,destination.getType()}, {}), true);
    func::CallOp::create(b, loc, fn, ValueRange{next.handle,destination});
    storage[op.getResult()].buffer = destination;
    return success();
  }
  Value image = arith::ConstantOp::create(
      b, loc, b.getI64IntegerAttr(op.getImage() ? 1 : 0));
  func::CallOp::create(
      b, loc,
      declare("mdrtDistRedistribute",
              b.getFunctionType({i64, i64, source.getType(),
                                 destination.getType(), i64}, {}), true),
      ValueRange{old.handle, next.handle, source, destination, image});
  storage[op.getResult()].buffer = destination;
  return success();
}

LogicalResult Lowering::lowerHalo(HaloOp op) {
  OpBuilder b(op);
  // All constraint members are owned. Such a solve only needs the layout
  // for row lookup and never reads the halo or an imported field.
  if (llvm::all_of(op.getResult().getUsers(), [&](Operation *use) {
        if (auto rows = dyn_cast<TupleRowsOp>(use)) return rows.getGroup();
        if (auto copy = dyn_cast<HaloCopyOp>(use))
          return !copy.getResult().use_empty() &&
                 llvm::all_of(copy.getResult().getUsers(), [&](Operation *u) {
                   return constraintKernels.contains(u);
                 });
        auto kernel = dyn_cast<TupleKernelOp>(use);
        return kernel && constraintKernels.contains(use) &&
               llvm::none_of(kernel->getResultTypes(), [](Type t) {
                 auto field = dyn_cast<FieldType>(t);
                 return field && field.getPartial();
               });
      }))
    return success();
  Storage &layout = storage[op.getLayout()];
  Value positions = storage[op.getPositions()].buffer;
  Type i64 = b.getI64Type();
  double reach = op.getResult().getType().getReach().convertToDouble();
  SmallVector<Value> args{layout.handle, positions, real(b, reach)};
  Value cache = trajectoryCaches.lookup(op.getResult());
  if (cache) args.push_back(cache);
  int64_t mode = haloMode(op);
  bool midpoint = mode & 1, local = mode & 2;
  StringRef name = midpoint ? "mdrtDistMidpointHalo" : "mdrtDistHalo";
  if (cache) {
    name = "mdrtDistEpochHalo";
    args.push_back(arith::ConstantIntOp::create(b, loc, mode, 64));
  }
  Value handle;
  if (parallelDomains && !cache && domains > 1) {
    args.push_back(arith::ConstantIntOp::create(b, loc, mode, 64));
    auto field = op.getPositions().getDefiningOp<RedistributeOp>();
    bool migrated = mpi && field && isMigrationPosition(field);
    args.push_back(arith::ConstantIntOp::create(b, loc, migrated, 64));
    Value build = func::CallOp::create(b, loc, declare("mdrtDistBeginHalo",
        b.getFunctionType(ValueRange(args).getTypes(), {i64}), true), args).getResult(0);
    auto visit = declare("mdrtDistVisitHalo", b.getFunctionType({i64,i64,positions.getType()}, {}), true);
    scf::ParallelOp::create(b, loc, ValueRange{index(b,0)}, ValueRange{index(b,domains)},
        ValueRange{index(b,1)}, [&](OpBuilder &body, Location, ValueRange iv) {
          Value d = arith::IndexCastOp::create(body, loc, i64, iv[0]);
          func::CallOp::create(body, loc, visit, ValueRange{build,d,positions});
        });
    handle = func::CallOp::create(b, loc, declare("mdrtDistFinishHalo",
        b.getFunctionType({i64,positions.getType()}, {i64}), true), ValueRange{build,positions}).getResult(0);
  } else {
    if (!cache && local) {
      name = "mdrtDistDriverHalo";
      args.push_back(arith::ConstantIntOp::create(b,loc,mode,64));
    }
    handle = func::CallOp::create(b, loc,
        declare(name, b.getFunctionType(ValueRange(args).getTypes(), {i64}), true), args).getResult(0);
  }
  Value capacity = toIndex(
      b, func::CallOp::create(
             b, loc,
             declare("mdrtDistHaloCapacity", b.getFunctionType({i64}, {i64}),
                     false),
             ValueRange{handle})
             .getResult(0));
  auto [rows, ids] = rowViews(b, handle, true);
  OpBuilder atEnd(end);
  func::CallOp::create(
      atEnd, loc,
      declare("mdrtDistReleaseHalo", b.getFunctionType({i64}, {}), false),
      ValueRange{handle});
  storage[op.getResult()] = {handle, rows, ids, capacity, Value(), Value()};
  return success();
}

LogicalResult Lowering::lowerCovered(CoveredOp op) {
  OpBuilder b(op);
  Value reference = storage[op.getReference()].buffer;
  Value current = storage[op.getCurrent()].buffer;
  Type i64 = b.getI64Type();
  Value answer = func::CallOp::create(
                     b, loc,
                     declare("mdrtDistCovered",
                             b.getFunctionType({i64, reference.getType(),
                                                current.getType(), b.getF64Type()},
                                               {i64}),
                             true),
                     ValueRange{storage[op.getHalo()].handle, reference,
                                current,
                                real(b, op.getCutoff().convertToDouble())})
                     .getResult(0);
  Value zero = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(0));
  Value covered = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                       answer, zero);
  op.getResult().replaceAllUsesWith(covered);
  return success();
}

// Group fields in one block and epoch. A reverse packet may span force
// producers as long as none consumes an outstanding destination.
void Lowering::planHaloGroups(ArrayRef<Operation *> ops) {
  haloGroups.clear();
  haloGroupOf.clear();
  if (!mpi) return;
  for (bool transpose : {false,true}) {
    HaloGroup *group = nullptr;
    llvm::DenseSet<Operation *> consumers;
    for (Operation *op : ops) {
      if (group && (op->getBlock() != group->transfers.front()->getBlock() ||
                    consumers.contains(op)))
        group = nullptr;
      if (isa<DistributeOp, RedistributeOp>(op) ||
          (transpose && isa<TupleRowsOp, TupleKernelOp, ZerosOp, AllReduceOp>(op)))
        continue;
      Value halo;
      if (auto copy = dyn_cast<HaloCopyOp>(op); copy && !transpose) {
        if (!copy->use_empty() && llvm::all_of(copy->getUsers(),
            [&](Operation *u) { return constraintKernels.contains(u); }))
          continue;
        halo = copy.getHalo();
      } else if (auto sum = dyn_cast<HaloSumOp>(op); sum && transpose) {
        // These producers stage both ends in wide words in fixed-point
        // mode. Other field representations keep their individual sums.
        if (fixedPoint && !isa_and_nonnull<TupleKernelOp,GatherPairsOp>(sum.getOwned().getDefiningOp())) {
          group = nullptr;
          continue;
        }
        halo = sum.getHalo();
      } else {
        group = nullptr;
        continue;
      }
      if (!group || group->halo != halo) {
        haloGroups.emplace_back();
        group = &haloGroups.back();
        group->halo = halo;
        consumers.clear();
      }
      group->transfers.push_back(op);
      haloGroupOf[op] = haloGroups.size()-1;
      for (auto *user : op->getUsers())
        consumers.insert(op->getBlock()->findAncestorOpInBlock(*user));
    }
  }
}

bool Lowering::batchHalo(OpBuilder &b, Operation *op, Value halo, Value owned,
                         Value ghosts, Value imageOrOutput, bool transpose, bool words) {
  auto it = haloGroupOf.find(op);
  if (it == haloGroupOf.end()) return false;
  auto &group = haloGroups[it->second];
  if (group.transfers.size() == 1) return false;
  Type i64 = b.getI64Type();
  if (op == group.transfers.front()) {
    Value size = arith::ConstantIntOp::create(b, loc, group.transfers.size(), 64);
    group.handle = func::CallOp::create(b, loc, declare("mdrtDistHaloBatch",
        b.getFunctionType({i64}, {i64}), false), ValueRange{size}).getResult(0);
  }
  SmallVector<Value> fields{group.handle,owned,ghosts,imageOrOutput};
  func::CallOp::create(b, loc, declare(words ? "mdrtDistHaloWordField" : "mdrtDistHaloField",
      b.getFunctionType(ValueRange(fields).getTypes(), {}), true), fields);
  if (op == group.transfers.back()) {
    Value direction = arith::ConstantIntOp::create(b, loc, transpose, 64);
    Value fixed = arith::ConstantIntOp::create(b, loc, words, 64);
    func::CallOp::create(b, loc, declare("mdrtDistFlushHalo",
        b.getFunctionType({i64,i64,i64,i64}, {}), false),
        ValueRange{halo,group.handle,direction,fixed});
  }
  return true;
}

LogicalResult Lowering::lowerHaloCopy(HaloCopyOp op) {
  OpBuilder b(op);
  Storage &halo = storage[op.getHalo()];
  Value owned = storage[op.getField()].buffer;
  int64_t k = op.getResult().getType().getNumComponents();
  // Group schedules validate that every member resides on the common
  // owner. A copy used only by such kernels has no destination reads.
  if (!op.getResult().use_empty() && llvm::all_of(op.getResult().getUsers(), [&](Operation *use) {
        return constraintKernels.contains(use);
      }))
    return success();
  Value ghosts = allocate(b, halo.capacity, k);
  Type i64 = b.getI64Type();
  bool separateImage = fixedPoint || llvm::any_of(op.getResult().getUsers(),
      [](Operation *user) {
        if (isa<TupleKernelOp>(user)) return true;
        auto relation = dyn_cast<NeighborhoodOp>(user);
        return relation && relation.getResult().getType().getTraversal() == Traversal::Midpoint;
      });
  Value image = arith::ConstantOp::create(
      b, loc, b.getI64IntegerAttr(op.getImage() && !separateImage ? 1 : 0));
  if (op.getImage() && separateImage) {
    Value shifts = allocate(b, halo.capacity, 3);
    auto fill = declare("mdrtDistImageShifts",
                        b.getFunctionType({i64, shifts.getType()}, {}), true);
    func::CallOp::create(b, loc, fill, ValueRange{halo.handle, shifts});
    imageShifts[op.getResult()] = shifts;
  }
  if (batchHalo(b,op,halo.handle,owned,ghosts,image,false,false)) {
    storage[op.getResult()].buffer = ghosts;
    return success();
  }
  if (haloEvents) {
    auto start = mdrt::HaloCopyStartOp::create(
        b, loc, mdrt::EventType::get(module.getContext()), halo.handle, owned,
        ghosts, image);
    auto wait = mdrt::HaloWaitOp::create(b, loc, ghosts.getType(),
                                         start.getEvent(), halo.handle, owned,
                                         ghosts);
    storage[op.getResult()].buffer = wait.getResult();
  } else {
    func::FuncOp copy = declare(
        "mdrtDistHaloCopy",
        b.getFunctionType({i64, owned.getType(), ghosts.getType(), i64}, {}),
        true);
    func::CallOp::create(b, loc, copy,
                         ValueRange{halo.handle, owned, ghosts, image});
    storage[op.getResult()].buffer = ghosts;
  }
  return success();
}

/// The lowering of a sum across domains adds the domains, then the rows, in
/// their order: the same bits on every run for the same decomposition and
/// build, which meets every algebra (`fixed_order` as md_dist defines it,
/// #409). The sum must state one (D301).
static LogicalResult checkAlgebra(Operation *op) {
  if (md_exec::algebra::getAlgebras(op))
    return success();
  return op->emitOpError() << md_exec::algebra::getMissingMessage();
}

LogicalResult Lowering::lowerHaloSum(HaloSumOp op) {
  if (failed(checkAlgebra(op)))
    return failure();
  OpBuilder b(op);
  Storage &halo = storage[op.getHalo()];
  Storage &layout = storage[layoutOf(op.getHalo())];
  Value owned = storage[op.getOwned()].buffer;
  Value ghosts = storage[op.getGhosts()].buffer;
  int64_t k = op.getResult().getType().getNumComponents();
  Value result = allocate(b, layout.capacity, k);
  if (fixedPoint &&
      cast<MemRefType>(owned.getType()).getElementType().isInteger(128)) {
    auto sum = declare("mdrtDistHaloSumWords",
                       b.getFunctionType({b.getI64Type(), owned.getType(),
                                          ghosts.getType(), result.getType()},
                                         {}),
                       true);
    auto type = cast<MemRefType>(owned.getType());
    Value words = allocate(b, type, {index(b, domains), layout.capacity, index(b, k)});
    initializeBuffer(b, words, true, owned);
    if (!batchHalo(b,op,halo.handle,words,ghosts,result,true,true))
      func::CallOp::create(b, loc, sum, ValueRange{halo.handle, words, ghosts, result});
    fieldWords[op.getResult()] = words;
    storage[op.getResult()].buffer = result;
    return success();
  }
  initializeBuffer(b, result, true, owned);
  if (mpi && batchHalo(b,op,halo.handle,result,ghosts,
                       arith::ConstantIntOp::create(b,loc,0,64),true,false)) {
    storage[op.getResult()].buffer = result;
    return success();
  }
  if (haloEvents) {
    auto start = mdrt::HaloSumStartOp::create(
        b, loc, mdrt::EventType::get(module.getContext()), halo.handle, ghosts,
        result);
    auto wait = mdrt::HaloWaitOp::create(b, loc, result.getType(),
                                         start.getEvent(), halo.handle, ghosts,
                                         result);
    storage[op.getResult()].buffer = wait.getResult();
  } else {
    func::FuncOp sum = declare(
        "mdrtDistHaloSum",
        b.getFunctionType({b.getI64Type(), ghosts.getType(), result.getType()},
                          {}),
        true);
    func::CallOp::create(b, loc, sum, ValueRange{halo.handle, ghosts, result});
    storage[op.getResult()].buffer = result;
  }
  return success();
}

LogicalResult Lowering::lowerZeros(ZerosOp op) {
  OpBuilder b(op);
  Storage &rows = storage[op.getRows()];
  Value buffer =
      allocate(b, rows.capacity, op.getResult().getType().getNumComponents());
  initializeBuffer(b, buffer, true);
  storage[op.getResult()].buffer = buffer;
  return success();
}

LogicalResult Lowering::lowerTupleKernel(TupleKernelOp op) {
  OpBuilder b(op);
  Value edge[3];
  if (failed(getHaloEdges(op, op.getHalo(), edge)))
    return failure();
  auto vec = VectorType::get({3}, b.getF64Type());
  Value edges = vector::FromElementsOp::create(b, loc, vec, ValueRange(edge));
  auto coordinates = md::getCoordinates(op.getCoordinateKinds(), op.getCoordinateMembers());
  int64_t arity = op.getRows().getType().getDimSize(1) - 1;
  bool reduce = isa<PartialType>(op.getResult(0).getType());
  bool solve = !reduce && !cast<FieldType>(op.getResult(0).getType()).getPartial();
  bool hasFlag = solve && op.getNumResults() == 2;
  Type resultType = reduce ? cast<PartialType>(op.getResult(0).getType()).getValueType() : vec;
  auto vector = dyn_cast<VectorType>(resultType);
  int64_t width = vector ? vector.getNumElements() : 1;
  Type scalar = sumType(b.getF64Type());
  Storage halo = storage[op.getHalo()];
  Storage layout = storage[layoutOf(op.getHalo())];
  auto buffer = [&](ArrayRef<int64_t> shape, ValueRange sizes, bool domainRows = true) {
    Value value = allocate(b, MemRefType::get(shape, scalar), sizes);
    initializeBuffer(b, value, domainRows);
    return value;
  };
  constexpr int64_t dyn = ShapedType::kDynamic;
  Value partial, ownedOut, ghostOut;
  if (reduce) {
    partial = buffer({dyn, dyn}, {index(b, domains), index(b, width)});
    storage[op.getResult(0)].buffer = partial;
  } else if (solve) {
    ownedOut = allocate(b, layout.capacity, 3);
    initializeBuffer(b, ownedOut, true);
    storage[op.getResult(0)].buffer = ownedOut;
  } else {
    ownedOut = buffer({dyn, dyn, dyn}, {index(b, domains), layout.capacity, index(b, 3)});
    ghostOut = buffer({dyn, dyn, dyn}, {index(b, domains), halo.capacity, index(b, 3)});
    storage[op.getResult(0)].buffer = ownedOut;
    storage[op.getResult(1)].buffer = ghostOut;
  }
  Value count = memref::DimOp::create(b, loc, op.getRows(), 0);
  Value first = domainBound(b, false), past = domainBound(b, true);
  auto indices = [&](Value size) {
    return allocate(b, MemRefType::get({dyn}, b.getIndexType()), {size});
  };
  auto callIndex = [&](StringRef name, ValueRange args) -> Value {
    auto fn = declare(name, b.getFunctionType(args.getTypes(), {b.getIndexType()}), true);
    return func::CallOp::create(b, loc, fn, args).getResult(0);
  };
  Value z = index(b, 0), one = index(b, 1);
  Value order = indices(count);
  Value starts = indices(arith::AddIOp::create(b, loc, count, one));
  Value records = memref::CastOp::create(b, loc, MemRefType::get({dyn, dyn}, b.getI64Type()), op.getRows());
  Value chunks = callIndex("mdrtDistTupleChunks", {records, first, past,
      index(b, 128), order, starts});
  Value localCount = memref::LoadOp::create(b, loc, starts, chunks);
  bool indexed = module->hasAttr("md_dist.driver");
  SmallVector<Value> parameters;
  if (!op.getParameters().empty()) {
    Value ids;
    if (Value schedule = tupleSchedules.lookup(op.getRows())) {
      ids = indices(localCount);
      auto remap = declare("mdrtDistParameterOrder",
          b.getFunctionType({b.getI64Type(),order.getType(),ids.getType()}, {}), true);
      func::CallOp::create(b, loc, remap, ValueRange{schedule,order,ids});
    } else if (indexed)
      ids = memref::ReinterpretCastOp::create(b, loc, cast<MemRefType>(order.getType()),
          order, b.getIndexAttr(0), ArrayRef<OpFoldResult>{localCount},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
    for (Value field : op.getParameters())
      parameters.push_back(inputSnapshot(b, cast<md::FieldType>(field.getType()), field, ids));
  }
  Value energies, flags, contributions;
  if (reduce)
    energies = buffer({dyn, dyn}, {chunks, index(b, width)}, false);
  if (hasFlag) {
    flags = allocate(b, MemRefType::get({dyn}, b.getI32Type()), {chunks});
    initializeBuffer(b, flags, false);
  }
  if (parallelDomains && !reduce && !solve)
    contributions = allocate(b, MemRefType::get({dyn, dyn, dyn}, scalar),
                             {localCount, index(b, arity), index(b, 3)});
  auto tupleColumn = [&](OpBuilder &at, Value tuple, Value col) {
    return toIndex(at, memref::LoadOp::create(at, loc, op.getRows(), ValueRange{tuple, col}));
  };
  auto visitChunk = [&](OpBuilder &at, Value chunk) {
        Value begin = memref::LoadOp::create(at, loc, starts, chunk);
        Value end = memref::LoadOp::create(at, loc, starts, arith::AddIOp::create(at, loc, chunk, one).getResult());
        scf::ForOp::create(at, loc, begin, end, one, ValueRange(),
            [&](OpBuilder &k, Location, Value pos, ValueRange) {
          Value t = memref::LoadOp::create(k, loc, order, pos);
          auto row = [&](OpBuilder &at, int64_t column) {
            return tupleColumn(at, t, index(at, column));
          };
          Value d = row(k, 0);
          Value n = toIndex(k, memref::LoadOp::create(k, loc, layout.rows, d));
          SmallVector<Value> localRows, isOwned, ghostRows, positions, args;
          for (int64_t a = 0; a < arity; ++a) {
            Value r = row(k, a + 1);
            localRows.push_back(r);
            isOwned.push_back(arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, r, n));
            ghostRows.push_back(arith::SubIOp::create(k, loc, r, n));
          }
          auto read = [&](int64_t a, Value owned, Value ghosts) -> Value {
            Type type = cast<FieldType>(owned.getType()).getKernelValueType();
            if (constraintKernels.contains(op))
              return loadValue(k, storage[owned].buffer, d, localRows[a], type);
            auto value = scf::IfOp::create(k, loc, isOwned[a], [&](OpBuilder &yes, Location) {
              scf::YieldOp::create(yes, loc, loadValue(yes, storage[owned].buffer, d, localRows[a], type));
            }, [&](OpBuilder &no, Location) {
              scf::YieldOp::create(no, loc, loadValue(no, storage[ghosts].buffer, d, ghostRows[a], type));
            });
            return value.getResult(0);
          };
          for (int64_t a = 0; a < arity; ++a)
            positions.push_back(read(a, op.getPositions(), op.getGhosts()));
          Value half = arith::ConstantOp::create(k, loc, DenseElementsAttr::get(vec, k.getF64FloatAttr(0.5)));
          for (auto coordinate : coordinates) {
            SmallVector<Value, 3> values;
            for (auto displacement : md::getDisplacements(coordinate)) {
              auto m = displacement.members;
              Value raw = arith::SubFOp::create(k, loc, positions[m[0]], positions[m[1]]);
              Value image = math::FloorOp::create(k, loc, arith::AddFOp::create(k, loc, half,
                                                         arith::DivFOp::create(k, loc, raw, edges)));
              values.push_back(arith::SubFOp::create(k, loc, raw, arith::MulFOp::create(k, loc, image, edges)));
            }
            args.push_back(md::emitCoordinate(k, loc, coordinate.kind, values));
          }
          for (size_t f = 0; f < op.getGathered().size(); f += 2)
            for (int64_t a = 0; a < arity; ++a)
              args.push_back(read(a, op.getGathered()[f], op.getGathered()[f+1]));
          for (auto [field, values] : llvm::zip(op.getParameters(), parameters)) {
            auto type = cast<md::FieldType>(field.getType());
            if (type.getNumComponents() == 1)
              args.push_back(memref::LoadOp::create(k, loc, values, indexed ? pos : t));
            else {
              SmallVector<Value> components;
              for (unsigned c = 0; c < type.getNumComponents(); ++c)
                components.push_back(memref::LoadOp::create(k, loc, values, ValueRange{indexed ? pos : t, index(k, c)}));
              args.push_back(vector::FromElementsOp::create(k, loc,
                  cast<VectorType>(type.getKernelValueType()), components));
            }
          }
          Block &body = op.getKernel().front();
          IRMapping map;
          map.map(body.getArguments(), args);
          for (Operation &nested : body.without_terminator())
            cloneKernelOp(k, nested, map);
          auto yielded = body.getTerminator()->getOperands();
          if (reduce) {
            Value word = quantize(k, map.lookupOrDefault(yielded.front()), pairEnergy);
            for (int64_t c = 0; c < width; ++c) {
              Value component = vector ? Value(vector::ExtractOp::create(k, loc, word, c)) : word;
              SmallVector<Value, 2> place{chunk, index(k, c)};
              Value old = memref::LoadOp::create(k, loc, energies, place);
              memref::StoreOp::create(k, loc, add(k, old, component), energies, place);
            }
          } else if (solve) {
            for (int64_t a = 0; a < arity; ++a)
              storeValue(k, map.lookupOrDefault(yielded[a]), ownedOut, d, localRows[a]);
            if (hasFlag) {
              Value old = memref::LoadOp::create(k, loc, flags, chunk);
              Value flagged = arith::ExtUIOp::create(k, loc, k.getI32Type(), map.lookupOrDefault(yielded.back()));
              memref::StoreOp::create(k, loc, arith::OrIOp::create(k, loc, old, flagged), flags, chunk);
            }
          } else {
            for (int64_t a = 0; a < arity; ++a) {
              Value word = quantize(k, map.lookupOrDefault(yielded[a]), pairForce);
              if (contributions) {
                storeValue(k, word, contributions, pos, index(k, a));
              } else {
                scf::IfOp::create(k, loc, isOwned[a], [&](OpBuilder &yes, Location) {
                  addValue(yes, word, ownedOut, d, localRows[a]);
                  scf::YieldOp::create(yes, loc);
                }, [&](OpBuilder &no, Location) {
                  addValue(no, word, ghostOut, d, ghostRows[a]);
                  scf::YieldOp::create(no, loc);
                });
              }
            }
          }
          scf::YieldOp::create(k, loc);
        });
  };
  if (parallelDomains) {
    scf::ParallelOp::create(b, loc, ValueRange{z}, ValueRange{chunks}, ValueRange{one},
        [&](OpBuilder &at, Location, ValueRange iv) { visitChunk(at, iv.front()); });
  } else {
    scf::ForOp::create(b, loc, z, chunks, one, ValueRange{},
        [&](OpBuilder &at, Location, Value chunk, ValueRange) {
          visitChunk(at, chunk);
          scf::YieldOp::create(at, loc);
        });
  }
  if (reduce)
    scf::ForOp::create(b, loc, z, chunks, one, ValueRange(),
        [&](OpBuilder &at, Location, Value chunk, ValueRange) {
          Value pos = memref::LoadOp::create(at, loc, starts, chunk);
          Value t = memref::LoadOp::create(at, loc, order, pos);
          Value d = tupleColumn(at, t, z);
          for (int64_t c = 0; c < width; ++c) {
            Value col = index(at, c);
            Value old = memref::LoadOp::create(at, loc, partial, ValueRange{d, col});
            Value term = memref::LoadOp::create(at, loc, energies, ValueRange{chunk, col});
            memref::StoreOp::create(at, loc, add(at, old, term), partial, ValueRange{d, col});
          }
          scf::YieldOp::create(at, loc);
        });
  if (contributions) {
    Value members = index(b, arity);
    Value size = arith::MulIOp::create(b, loc, localCount, members);
    Value entries = indices(size);
    Value offsets = indices(arith::AddIOp::create(b, loc, size, one));
    Value stride = arith::AddIOp::create(b, loc, layout.capacity, halo.capacity);
    Value active = callIndex("mdrtDistTupleIncidence", {records, order, localCount,
        first, past, stride, entries, offsets});
    auto sumVec = VectorType::get({3}, scalar);
    scf::ParallelOp::create(b, loc, ValueRange{z}, ValueRange{active}, ValueRange{one},
        [&](OpBuilder &at, Location, ValueRange iv) {
          Value begin = memref::LoadOp::create(at, loc, offsets, iv[0]);
          Value end = memref::LoadOp::create(at, loc, offsets, arith::AddIOp::create(at, loc, iv[0], one).getResult());
          auto member = [&](OpBuilder &at, Value i) {
            Value e = memref::LoadOp::create(at, loc, entries, i);
            return std::pair<Value,Value>{arith::DivUIOp::create(at, loc, e, members),
                                          arith::RemUIOp::create(at, loc, e, members)};
          };
          auto [p, a] = member(at, begin);
          Value t = memref::LoadOp::create(at, loc, order, p);
          Value d = tupleColumn(at, t, z);
          Value r = tupleColumn(at, t, arith::AddIOp::create(at, loc, a, one));
          Value n = toIndex(at, memref::LoadOp::create(at, loc, layout.rows, d));
          auto sum = scf::ForOp::create(at, loc, begin, end, one, ValueRange{zero(at, sumVec)},
              [&](OpBuilder &body, Location, Value i, ValueRange acc) {
                auto [p, a] = member(body, i);
                Value value = loadValue(body, contributions, p, a, sumVec);
                scf::YieldOp::create(body, loc, add(body, acc[0], value));
              }).getResult(0);
          scf::IfOp::create(at, loc, arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, r, n),
              [&](OpBuilder &yes, Location) {
                storeValue(yes, sum, ownedOut, d, r);
                scf::YieldOp::create(yes, loc);
              }, [&](OpBuilder &no, Location) {
                storeValue(no, sum, ghostOut, d, arith::SubIOp::create(no, loc, r, n));
                scf::YieldOp::create(no, loc);
              });
        });
  }
  if (hasFlag) {
    Value init = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value flag = scf::ForOp::create(b, loc, z, chunks, one, ValueRange{init},
        [&](OpBuilder &at, Location, Value chunk, ValueRange acc) {
          Value term = memref::LoadOp::create(at, loc, flags, chunk);
          scf::YieldOp::create(at, loc, arith::OrIOp::create(at, loc, acc[0], term).getResult());
        }).getResult(0);
    if (mpi) {
      auto maximum = declare("mdrtMPIMax", b.getFunctionType({b.getI32Type()}, {b.getI32Type()}), false);
      flag = func::CallOp::create(b, loc, maximum, flag).getResult(0);
    }
    op.getResult(1).replaceAllUsesWith(arith::TruncIOp::create(b, loc, b.getI1Type(), flag));
  }
  return success();
}

LogicalResult Lowering::lowerTupleRows(TupleRowsOp op) {
  OpBuilder b(op);
  auto relation = cast<md::RelationType>(op.getTuples().getType());
  Value members = mdrt::ToBufferOp::create(b, loc, mdrt::getMembersType(relation), op.getTuples());
  auto dynamic = [&](Type type) {
    return MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, type);
  };
  Value count = memref::DimOp::create(b, loc, members, 0);
  Value source = memref::CastOp::create(b, loc, dynamic(b.getI32Type()), members);
  Value positions = storage[op.getPositions()].buffer;
  StringRef name = op.getGroup() ? "mdrtDistConstraintRows" : "mdrtDistTupleRows";
  if (mpi && module->hasAttr("md_dist.driver") &&
      llvm::all_of(op.getRows().getUsers(), [](Operation *u) { return isa<TupleKernelOp>(u); })) {
    Type i64 = b.getI64Type();
    Value handle = op.getGroup() ? storage[layoutOf(op.getHalo())].handle : storage[op.getHalo()].handle;
    TupleRowKey rowKey{op->getBlock(), op.getTuples(), handle,
                       op.getGroup() ? Value() : op.getPositions(), op.getRows().getType()};
    if (Value existing = sharedTupleRows.lookup(rowKey)) {
      op.getRows().replaceAllUsesWith(existing);
      return success();
    }
    auto key = std::make_pair(op->getBlock(),op.getPositions());
    Value &shard = topologyShards[key];
    if (!op.getGroup() && !shard) {
      auto gather = declare("mdrtDistTopologyShard",
          b.getFunctionType({i64,positions.getType()}, {i64}), true);
      shard = func::CallOp::create(b, loc, gather,
          ValueRange{storage[layoutOf(op.getHalo())].handle,positions}).getResult(0);
      OpBuilder cleanup(end);
      func::CallOp::create(cleanup, loc, declare("mdrtDistReleaseShard",
          b.getFunctionType({i64}, {}), false), shard);
    }
    Value flag = arith::ConstantIntOp::create(b,loc,op.getGroup() ? 1 : 0,64);
    Value none = arith::ConstantIntOp::create(b,loc,0,64);
    Value plan = none;
    if (!op.getGroup()) {
      Operation *scope = op;
      while (auto loop = dyn_cast<scf::ForOp>(scope->getParentOp())) {
        if (!loop.isDefinedOutsideOfLoop(op.getTuples())) break;
        scope = loop;
      }
      OpBuilder init(scope);
      plan = func::CallOp::create(init,loc,declare("mdrtDistNewTuplePlan",
          b.getFunctionType({}, {i64}),false)).getResult(0);
      OpBuilder cleanup(end);
      if (scope != op) cleanup.setInsertionPointAfter(scope);
      func::CallOp::create(cleanup,loc,declare("mdrtDistFreeTuplePlan",
          b.getFunctionType({i64},{}),false),plan);
    }
    auto build = declare("mdrtDistScheduleTuples",
        b.getFunctionType({i64,i64,i64,i64,positions.getType(),source.getType()}, {i64}), true);
    Value schedule = func::CallOp::create(b, loc, build,
        ValueRange{handle,flag,op.getGroup() ? none : shard,plan,positions,source}).getResult(0);
    auto view = declare("mdrtDistScheduleView",
        b.getFunctionType({i64}, {dynamic(i64)}), true);
    Value table = func::CallOp::create(b, loc, view, schedule).getResult(0);
    Value rows = memref::CastOp::create(b, loc, op.getRows().getType(), table);
    tupleSchedules[rows] = schedule;
    sharedTupleRows[rowKey] = rows;
    op.getRows().replaceAllUsesWith(rows);
    OpBuilder cleanup(end);
    func::CallOp::create(cleanup, loc, declare("mdrtDistReleaseSchedule",
        b.getFunctionType({i64}, {}), false), schedule);
    return success();
  }
  if (mpi && !op.getGroup()) {
    // The field is immutable within its SSA lifetime. All tuple sets in
    // this block can read the same replicated coordinates.
    if (sharedTuplePositions.insert({op->getBlock(), op.getPositions()}).second) {
      Value layout = storage[layoutOf(op.getHalo())].handle;
      auto share = declare("mdrtMPICollectRows",
          b.getFunctionType({b.getI64Type(),positions.getType()}, {}), true);
      func::CallOp::create(b, loc, share, ValueRange{layout,positions});
    }
  }
  Value rows = allocate(b, op.getRows().getType(), {count});
  Value result = memref::CastOp::create(b, loc, dynamic(b.getI64Type()), rows);
  auto fill = declare(name,
      b.getFunctionType({b.getI64Type(), positions.getType(), source.getType(), result.getType()}, {}), true);
  func::CallOp::create(b, loc, fill, ValueRange{op.getGroup() ? storage[layoutOf(op.getHalo())].handle
                                 : storage[op.getHalo()].handle, positions, source, result});
  op.getRows().replaceAllUsesWith(rows);
  return success();
}

LogicalResult Lowering::lowerNeighborhood(NeighborhoodOp op) {
  // The relation is found where it is traversed; here only the precondition
  // under which it is the relation of md.neighborhood is checked.
  Value edges[3];
  if (failed(getHaloEdges(op, op.getHalo(), edges)))
    return failure();
  OpBuilder b(op);
  Type f64 = b.getF64Type();
  if (Value excluded = op.getExcluded()) {
    Type i64 = b.getI64Type();
    auto type = mdrt::getMembersType(cast<md::RelationType>(excluded.getType()));
    // A spatial permutation produces a new relation. Keep its sorted
    // keys across exactly the serial loops that carry the same topology.
    Operation *scope = op;
    while (auto loop = dyn_cast<scf::ForOp>(scope->getParentOp())) {
      if (!loop.isDefinedOutsideOfLoop(excluded)) break;
      scope = loop;
    }
    bool local = mpi && module->hasAttr("md_dist.driver");
    auto key = std::make_pair(scope, excluded);
    Value handle = scopedExclusions.lookup(key);
    bool created = !handle;
    if (created) {
      OpBuilder init(scope);
      Value members = mdrt::ToBufferOp::create(init, loc, type, excluded);
      auto build = declare(local ? "mdrtDistExclusionShard" : "mdrtDistExclusions",
                           b.getFunctionType({type}, {i64}), true);
      handle = func::CallOp::create(init, loc, build, members).getResult(0);
      scopedExclusions[key] = handle;
    }
    Value selected = handle;
    if (local) {
      auto project = declare("mdrtDistHaloExclusions",b.getFunctionType({i64,i64},{i64}),false);
      selected = func::CallOp::create(b,loc,project,ValueRange{storage[op.getHalo()].handle,handle}).getResult(0);
    }
    excludedPairs[op.getResult()] = selected;
    OpBuilder cleanup(end);
    auto release = declare("mdrtDistFreeExclusions", b.getFunctionType({i64}, {}), false);
    if (local) func::CallOp::create(cleanup,loc,release,selected);
    if (created) {
      if (scope != op) cleanup.setInsertionPointAfter(scope);
      func::CallOp::create(cleanup, loc, release, handle);
    }
  }
  func::CallOp::create(
      b, loc,
      declare("mdrtDistCheckCutoff",
              b.getFunctionType({f64, f64, f64, f64}, {}), false),
      ValueRange{real(b, op.getCutoff().convertToDouble()), edges[0],
                 edges[1], edges[2]});
  return success();
}

LogicalResult Lowering::lowerPartitionCenters(PartitionCentersOp op) {
  if (failed(proveCenterPartition(op)))
    return failure();
  // Materialize the relation's ghost membership after the position copy.
  // The reference relation is otherwise traversed lazily by emitPairs. Its
  // ghost pair predicate below is identical to that traversal's predicate;
  // the two stored bits are complements, giving exact disjoint coverage.
  auto relation = op.getRelation().getDefiningOp<NeighborhoodOp>();
  OpBuilder b(op);
  Storage &layout = storage[op.getLayout()];
  Storage &halo = storage[op.getHalo()];
  auto maskType = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic},
      b.getI32Type());
  Value interior = allocate(b, maskType,
                            {index(b, domains), layout.capacity, index(b, 1)});
  Value boundary = allocate(b, maskType,
                            {index(b, domains), layout.capacity, index(b, 1)});
  Value xo = storage[relation.getPositions()].buffer;
  Value xg = storage[relation.getGhosts()].buffer;
  bool unique = relation.getResult().getType().getTraversal() ==
                Traversal::Unique;
  double cutoff = relation.getCutoff().convertToDouble();
  auto c = [&](OpBuilder &at, int64_t v) -> Value {
    return arith::ConstantIndexOp::create(at, loc, v);
  };
  scf::ForOp::create(
      b, loc, domainBound(b, false), domainBound(b, true), c(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value nOwned = toIndex(
            bd, memref::LoadOp::create(bd, loc, layout.rows, ValueRange{d}));
        Value nGhost = toIndex(
            bd, memref::LoadOp::create(bd, loc, halo.rows, ValueRange{d}));
        scf::ForOp::create(
            bd, loc, c(bd, 0), nOwned, c(bd, 1), ValueRange(),
            [&](OpBuilder &bi, Location, Value i, ValueRange) {
              Value xi = loadValue(bi, xo, d, i,
                                   VectorType::get({3}, bi.getF64Type()));
              Value idi = unique
                              ? Value(memref::LoadOp::create(
                                    bi, loc, layout.ids, ValueRange{d, i}))
                              : Value();
              Value no = arith::ConstantIntOp::create(bi, loc, 0, 1);
              auto scan = scf::ForOp::create(
                  bi, loc, c(bi, 0), nGhost, c(bi, 1), ValueRange{no},
                  [&](OpBuilder &bg, Location, Value k, ValueRange found) {
                    Value xk = loadValue(bg, xg, d, k,
                                         VectorType::get({3}, bg.getF64Type()));
                    Value delta = arith::SubFOp::create(bg, loc, xi, xk);
                    if (Value shifts = imageShifts.lookup(relation.getGhosts()))
                      delta = arith::SubFOp::create(
                          bg, loc, delta,
                          loadValue(bg, shifts, d, k, delta.getType()));
                    Value r2 = vector::ReductionOp::create(
                        bg, loc, vector::CombiningKind::ADD,
                        arith::MulFOp::create(bg, loc, delta, delta));
                    Value inside = arith::CmpFOp::create(
                        bg, loc, arith::CmpFPredicate::OLT, r2,
                        real(bg, cutoff * cutoff));
                    if (unique) {
                      Value idk = memref::LoadOp::create(
                          bg, loc, halo.ids, ValueRange{d, k});
                      inside = arith::AndIOp::create(
                          bg, loc, inside,
                          arith::CmpIOp::create(
                              bg, loc, arith::CmpIPredicate::slt, idi, idk));
                    }
                    Value any = arith::OrIOp::create(bg, loc, found[0], inside);
                    scf::YieldOp::create(bg, loc, any);
                  });
              Value one = arith::ConstantIntOp::create(bi, loc, 1, 32);
              Value zero = arith::ConstantIntOp::create(bi, loc, 0, 32);
              Value interiorBit = arith::SelectOp::create(
                  bi, loc, scan.getResult(0), zero, one);
              Value boundaryBit = arith::SelectOp::create(
                  bi, loc, scan.getResult(0), one, zero);
              memref::StoreOp::create(bi, loc, interiorBit, interior,
                                      ValueRange{d, i, c(bi, 0)});
              memref::StoreOp::create(bi, loc, boundaryBit, boundary,
                                      ValueRange{d, i, c(bi, 0)});
              scf::YieldOp::create(bi, loc);
            });
        scf::YieldOp::create(bd, loc);
      });
  storage[op.getInterior()].buffer = interior;
  storage[op.getBoundary()].buffer = boundary;
  return success();
}

//===----------------------------------------------------------------------===//
// Pairs
//===----------------------------------------------------------------------===//

namespace {
/// The values of a local row k of a domain: owned (k < n) or ghost.
struct PairSide {
  Value position;
  SmallVector<Value> gathered;
  Value id;
};
} // namespace

/// Emits, for every domain, every owned center `i`, and every local row
/// `k` of a pair of the relation of `op` within its cutoff, `visit` with
/// the builder inside the test, the kernel's value, and the indices; the
/// visit returns the new accumulator of `i`. `finishCenter` receives the
/// accumulator of each center, `finishDomain` the sum over its centers.
template <typename OpTy>
void Lowering::emitPairs(
    OpTy op, Location loc, Value first, Value past, bool parallel,
    llvm::DenseMap<Value, Storage> &storage, Type accumulator, Value shifts,
    function_ref<Value(OpBuilder &, Value, Value)> add,
    function_ref<Value(OpBuilder &, Type)> zero,
    function_ref<Value(OpBuilder &, Value, Value, Value, Type)> load,
    function_ref<Value(OpBuilder &, Block &, ValueRange)> kernel,
    function_ref<Value(OpBuilder &, Value value, Value acc, Value d, Value k,
                       Value isOwned, Value ghostRow)>
        visit,
    function_ref<void(OpBuilder &, Value acc, Value d, Value i)> finishCenter,
    function_ref<void(OpBuilder &, Value acc, Value d)> finishDomain) {
  auto neighborhood = op.getRelation().template getDefiningOp<NeighborhoodOp>();
  auto traversal = op.getRelation().getType().getTraversal();
  bool midpoint = traversal == Traversal::Midpoint;
  bool unique = traversal != Traversal::Directed;
  double cutoff = neighborhood.getCutoff().convertToDouble();
  Value haloValue = neighborhood.getHalo();
  Storage &halo = storage[haloValue];
  Storage &layout = storage[layoutOf(haloValue)];
  Value xo = storage[op.getPositions()].buffer;
  Value xg = storage[op.getGhosts()].buffer;
  Value selected = op.getCenterSet();
  Value centerMask = selected ? storage[selected].buffer : Value();
  bool ghostFree = selected &&
                   selected ==
                       selected.template getDefiningOp<PartitionCentersOp>()
                           .getInterior();
  SmallVector<Value> ownedFields, ghostFields;
  SmallVector<Type> types;
  for (unsigned g = 0; g < op.getGathered().size(); g += 2) {
    ownedFields.push_back(storage[op.getGathered()[g]].buffer);
    ghostFields.push_back(storage[op.getGathered()[g + 1]].buffer);
    types.push_back(
        cast<FieldType>(op.getGathered()[g].getType()).getKernelValueType());
  }
  OpBuilder b(op);
  auto c = [&](OpBuilder &at, int64_t v) -> Value {
    return arith::ConstantIndexOp::create(at, loc, v);
  };
  Type f64 = b.getF64Type();
  auto vec3 = VectorType::get({3}, f64);
  Type i64 = b.getI64Type();
  auto indices = MemRefType::get({ShapedType::kDynamic}, i64);
  auto buildCells = declare(midpoint ? "mdrtDistMidpointCells" : "mdrtDistPairCells",
      b.getFunctionType({b.getIndexType(), b.getIndexType(), b.getIndexType(),
                         f64, xo.getType(), xg.getType(), xg.getType(), i64,
                         indices, indices}, {}), true);
  Value exclusions = excludedPairs.lookup(neighborhood.getResult());
  func::FuncOp allowed;
  if (exclusions)
    allowed = declare("mdrtDistPairAllowed", b.getFunctionType({i64, i64, i64}, {i64}), false);
  Value images, edges[3];
  if (midpoint) {
    // The neighborhood checked these cell operands when it was lowered.
    (void)getHaloEdges(op, haloValue, edges);
    constexpr int64_t dyn = ShapedType::kDynamic;
    auto imageType = MemRefType::get({dyn,dyn,3},i64,
        StridedLayoutAttr::get(b.getContext(),dyn,{dyn,3,1}));
    auto view = declare("mdrtDistHaloImageWords", b.getFunctionType({i64}, {imageType}), true);
    images = func::CallOp::create(b, loc, view, halo.handle).getResult(0);
  }
  auto visitDomain = [&](OpBuilder &bd, Value d) {
        Value owned = arith::IndexCastOp::create(
            bd, loc, bd.getIndexType(),
            memref::LoadOp::create(bd, loc, layout.rows, ValueRange{d}));
        Value ghosts = arith::IndexCastOp::create(
            bd, loc, bd.getIndexType(),
            memref::LoadOp::create(bd, loc, halo.rows, ValueRange{d}));
        Value local = arith::AddIOp::create(bd, loc, owned, ghosts);
        Value count = ghostFree ? owned : local;
        Value order = memref::AllocOp::create(bd, loc, indices, ValueRange{count});
        Value centerCount = midpoint ? count : owned;
        Value slots = arith::MulIOp::create(bd, loc, centerCount, c(bd, 18));
        Value spans = memref::AllocOp::create(bd, loc, indices, ValueRange{slots});
        func::CallOp::create(bd, loc, buildCells,
            ValueRange{d, owned, count, real(bd, cutoff), xo, xg,
                       shifts ? shifts : xg,
                       arith::ConstantIntOp::create(bd, loc, shifts ? 1 : 0, 64),
                       order, spans});
        auto readSide = [&](OpBuilder &bk, Value k, Value isOwned, Value ghostRow) {
          // The values of row k, from the owned or the ghost rows.
          SmallVector<Type> sideTypes = {vec3};
          sideTypes.append(types.begin(), types.end());
          if (shifts) sideTypes.push_back(vec3);
          if (unique)
            sideTypes.push_back(i64);
          auto side = scf::IfOp::create(bk, loc, sideTypes, isOwned,
                                        /*withElseRegion=*/true);
          {
            OpBuilder t = OpBuilder::atBlockEnd(side.thenBlock());
            SmallVector<Value> values = {load(t, xo, d, k, vec3)};
            for (unsigned g = 0; g < ownedFields.size(); ++g)
              values.push_back(
                  load(t, ownedFields[g], d, k, types[g]));
            if (shifts) values.push_back(zero(t, vec3));
            if (unique)
              values.push_back(memref::LoadOp::create(
                  t, loc, layout.ids, ValueRange{d, k}));
            scf::YieldOp::create(t, loc, values);
          }
          {
            OpBuilder e = OpBuilder::atBlockEnd(side.elseBlock());
            SmallVector<Value> values = {
                load(e, xg, d, ghostRow, vec3)};
            for (unsigned g = 0; g < ghostFields.size(); ++g)
              values.push_back(
                  load(e, ghostFields[g], d, ghostRow, types[g]));
            if (shifts) values.push_back(load(e, shifts, d, ghostRow, vec3));
            if (unique)
              values.push_back(memref::LoadOp::create(
                  e, loc, halo.ids, ValueRange{d, ghostRow}));
            scf::YieldOp::create(e, loc, values);
          }
          return side;
        };
        auto centers = scf::ForOp::create(
            bd, loc, c(bd, 0), centerCount, c(bd, 1),
            ValueRange{zero(bd, accumulator)},
            [&](OpBuilder &bi, Location, Value i, ValueRange domainSum) {
              Value spanCount = c(bi, 9);
              if (centerMask) {
                Value bit = memref::LoadOp::create(
                    bi, loc, centerMask, ValueRange{d, i, c(bi, 0)});
                Value active = arith::CmpIOp::create(
                    bi, loc, arith::CmpIPredicate::ne, bit,
                    arith::ConstantIntOp::create(bi, loc, 0, 32));
                spanCount = arith::SelectOp::create(bi, loc, active, spanCount,
                                                    c(bi, 0));
              }
              Value xi, idi, centerShift, centerImage[3];
              SmallVector<Value> gi;
              if (midpoint) {
                Value localCenter = arith::CmpIOp::create(
                    bi, loc, arith::CmpIPredicate::ult, i, owned);
                Value ghost = arith::SubIOp::create(bi, loc, i, owned);
                for (int axis = 0; axis < 3; ++axis) {
                  centerImage[axis] = scf::IfOp::create(bi, loc, localCenter,
                      [&](OpBuilder &yes, Location) {
                        scf::YieldOp::create(yes, loc, real(yes, 0));
                      }, [&](OpBuilder &no, Location) {
                        Value v = memref::LoadOp::create(no, loc, images, ValueRange{d,ghost,c(no,axis)});
                        Value f = arith::SIToFPOp::create(no, loc, f64, v);
                        scf::YieldOp::create(no, loc, arith::NegFOp::create(no, loc, f).getResult());
                      }).getResult(0);
                }
                auto side = readSide(bi, i, localCenter, ghost);
                xi = side.getResult(0);
                llvm::append_range(gi, side.getResults().slice(1, types.size()));
                centerShift = side.getResult(1 + types.size());
                idi = side.getResults().back();
              } else {
                xi = load(bi, xo, d, i, vec3);
                for (unsigned g = 0; g < ownedFields.size(); ++g)
                  gi.push_back(load(bi, ownedFields[g], d, i, types[g]));
                idi = unique ? Value(memref::LoadOp::create(
                                         bi, loc, layout.ids, ValueRange{d, i}))
                                   : Value();
              }
              auto slabs = scf::ForOp::create(
                  bi, loc, c(bi, 0), spanCount, c(bi, 1),
                  ValueRange{zero(bi, accumulator)},
                  [&](OpBuilder &bs, Location, Value slab,
                      ValueRange subtotal) {
                    Value base = arith::AddIOp::create(
                        bs, loc, arith::MulIOp::create(bs, loc, i, c(bs, 18)),
                        arith::MulIOp::create(bs, loc, slab, c(bs, 2)));
                    auto bound = [&](Value index) {
                      return toIndex(
                          bs, memref::LoadOp::create(bs, loc, spans, index));
                    };
                    Value begin = bound(base);
                    Value end = bound(arith::AddIOp::create(bs, loc, base, c(bs, 1)));
                    auto rows = scf::ForOp::create(bs, loc, begin, end, c(bs, 1),
                        subtotal, [&](OpBuilder &bk, Location, Value slot, ValueRange acc) {
                    Value k = toIndex(bk, memref::LoadOp::create(bk, loc, order, slot));
                    Value isOwned = arith::CmpIOp::create(
                        bk, loc, arith::CmpIPredicate::ult, k, owned);
                    Value shifted = arith::SubIOp::create(bk, loc, k, owned);
                    Value ghostRow =
                        arith::SelectOp::create(bk, loc, isOwned, c(bk, 0),
                                                shifted);
                    auto side = readSide(bk, k, isOwned, ghostRow);
                    Value xk = side.getResult(0);
                    // d = x_i − x_k: ghosts carry their images, so no
                    // minimum image is taken.
                    Value dvec = arith::SubFOp::create(bk, loc, xi, xk);
                    if (shifts) {
                      Value shift = side.getResult(1 + types.size());
                      if (midpoint)
                        shift = arith::SubFOp::create(bk, loc, shift, centerShift);
                      dvec = arith::SubFOp::create(bk, loc, dvec, shift);
                    }
                    Value r2 = vector::ReductionOp::create(
                        bk, loc, vector::CombiningKind::ADD,
                        arith::MulFOp::create(bk, loc, dvec, dvec));
                    Value inside = arith::CmpFOp::create(
                        bk, loc, arith::CmpFPredicate::OLT, r2,
                        arith::ConstantOp::create(
                            bk, loc, bk.getF64FloatAttr(cutoff * cutoff)));
                    Value pair;
                    if (unique)
                      pair = arith::CmpIOp::create(
                          bk, loc, arith::CmpIPredicate::slt, idi,
                          side.getResults().back());
                    else
                      pair = arith::CmpIOp::create(
                          bk, loc, arith::CmpIPredicate::ne, k, i);
                    Value take = arith::AndIOp::create(bk, loc, inside, pair);
                    if (midpoint) {
                      auto owner = scf::IfOp::create(bk, loc, take,
                          [&](OpBuilder &t, Location) {
                            Value owner = c(t,0), matches = take;
                            auto grid = decomposition.getGrid();
                            for (int axis = 2; axis >= 0; --axis) {
                              Value x = vector::ExtractOp::create(t, loc, xi, axis);
                              Value dx = vector::ExtractOp::create(t, loc, dvec, axis);
                              Value mid = arith::SubFOp::create(t, loc, x,
                                  arith::MulFOp::create(t, loc, real(t,0.5), dx));
                              Value u = arith::DivFOp::create(t, loc, mid, edges[axis]);
                              Value shift = math::FloorOp::create(t, loc, u);
                              matches = arith::AndIOp::create(t, loc, matches, arith::CmpFOp::create(t,
                                  loc, arith::CmpFPredicate::OEQ, shift, centerImage[axis]));
                              Value frac = arith::SubFOp::create(t, loc, u, shift);
                              Value cell = toIndex(t, arith::FPToSIOp::create(t, loc, i64,
                                  arith::MulFOp::create(t, loc, frac, real(t,grid[axis]))));
                              cell = arith::MinUIOp::create(t, loc, cell, c(t,grid[axis]-1));
                              owner = arith::AddIOp::create(t, loc, cell,
                                  arith::MulIOp::create(t, loc, owner, c(t,grid[axis])));
                            }
                            Value same = arith::CmpIOp::create(t, loc, arith::CmpIPredicate::eq, owner, d);
                            scf::YieldOp::create(t, loc, arith::AndIOp::create(t, loc, same, matches).getResult());
                          }, [&](OpBuilder &e, Location) { scf::YieldOp::create(e, loc, take); });
                      take = owner.getResult(0);
                    }
                    if (exclusions) {
                      auto eligible = func::CallOp::create(bk, loc, allowed,
                          ValueRange{exclusions, idi, side.getResults().back()});
                      Value ok = arith::TruncIOp::create(bk, loc, bk.getI1Type(), eligible.getResult(0));
                      take = arith::AndIOp::create(bk, loc, take, ok);
                    }
                    auto branch = scf::IfOp::create(bk, loc, {accumulator},
                                                    take,
                                                    /*withElseRegion=*/true);
                    {
                      OpBuilder t = OpBuilder::atBlockEnd(branch.thenBlock());
                      Value r = math::SqrtOp::create(t, loc, r2);
                      SmallVector<Value> arguments = {r, dvec};
                      for (unsigned g = 0; g < ownedFields.size(); ++g) {
                        arguments.push_back(gi[g]);
                        arguments.push_back(side.getResult(1 + g));
                      }
                      Value value =
                          kernel(t, op.getKernel().front(), arguments);
                      scf::YieldOp::create(
                          t, loc,
                          ValueRange{visit(t, value, acc[0], d, k, isOwned,
                                           ghostRow)});
                    }
                    {
                      OpBuilder e = OpBuilder::atBlockEnd(branch.elseBlock());
                      scf::YieldOp::create(e, loc, ValueRange{acc[0]});
                    }
                    scf::YieldOp::create(bk, loc, branch.getResults());
                  });
                    scf::YieldOp::create(bs, loc, rows.getResults());
                  });
              Value sum = slabs.getResult(0);
              finishCenter(bi, sum, d, i);
              Value next = isa<SumPairsOp>(op) ? add(bi, domainSum[0], sum)
                                               : domainSum[0];
              scf::YieldOp::create(bi, loc, ValueRange{next});
            });
        memref::DeallocOp::create(bd, loc, spans);
        memref::DeallocOp::create(bd, loc, order);
        finishDomain(bd, centers.getResult(0), d);
  };
  Value one = c(b, 1);
  if (parallel) {
    auto loop = scf::ParallelOp::create(b, loc, ValueRange{first}, ValueRange{past}, ValueRange{one},
        [&](OpBuilder &bd, Location, ValueRange indices) {
          visitDomain(bd, indices.front());
          scf::ReduceOp::create(bd, loc);
        });
    // Give each domain's cell construction and loops a function scope.
    // SCF-to-OpenMP otherwise retains an alloca_scope around the call to
    // the cell builder, which cannot contain the lowered loop backedges.
    IRRewriter rewriter(b.getContext());
    unsigned suffix = 0;
    auto name = SymbolTable::generateSymbolName<32>("mdir_dist_pairs",
        [&](StringRef s) { return SymbolTable::lookupSymbolIn(module, s); }, suffix);
    auto domain = outlineSingleBlockRegion(rewriter, loc, loop.getRegion(), name);
    assert(succeeded(domain) && "the parallel domain region has one block");
    (*domain).setPrivate();
  } else {
    scf::ForOp::create(b, loc, first, past, one, ValueRange(),
        [&](OpBuilder &bd, Location, Value d, ValueRange) {
          visitDomain(bd, d);
          scf::YieldOp::create(bd, loc);
        });
  }
}

LogicalResult Lowering::lowerSumPairs(SumPairsOp op) {
  OpBuilder b(op);
  Type type = op.getResult().getType().getValueType();
  auto vector = dyn_cast<VectorType>(type);
  int64_t length = vector ? vector.getNumElements() : 1;
  auto partialType = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, sumType(b.getF64Type()));
  Value partial =
      allocate(b, partialType, {index(b, domains), index(b, length)});
  bool directed =
      op.getRelation().getType().getTraversal() == Traversal::Directed;
  // Over a directed relation every pair is visited at both of its owners.
  double weight = directed ? 0.5 : 1.0;
  emitPairs(
      op, loc, domainBound(b, false), domainBound(b, true), parallelDomains, storage, sumType(type), imageShifts.lookup(op.getGhosts()),
      [&](OpBuilder &at, Value a, Value v) { return add(at, a, v); },
      [&](OpBuilder &at, Type t) { return zero(at, t); },
      [&](OpBuilder &at, Value buffer, Value d, Value row, Type t) {
        return loadValue(at, buffer, d, row, t);
      },
      [&](OpBuilder &at, Block &kernel, ValueRange arguments) {
        return inlineKernel(at, kernel, arguments);
      },
      [&](OpBuilder &at, Value value, Value acc, Value, Value, Value, Value) {
        return add(at, acc, quantize(at, value, pairEnergy));
      },
      [&](OpBuilder &, Value, Value, Value) {},
      [&](OpBuilder &at, Value sum, Value d) {
        Value scaled = sum;
        if (!fixedPoint) {
          Value w = real(at, weight);
          if (vector)
            w = vector::BroadcastOp::create(at, loc, vector, w);
          scaled = arith::MulFOp::create(at, loc, sum, w);
        }
        for (int64_t c = 0; c < length; ++c) {
          Value part = vector ? Value(vector::ExtractOp::create(at, loc,
                                                                scaled, c))
                              : scaled;
          memref::StoreOp::create(at, loc, part, partial,
                                  ValueRange{d, index(at, c)});
        }
      });
  storage[op.getResult()].buffer = partial;
  return success();
}

LogicalResult Lowering::lowerGatherPairs(GatherPairsOp op) {
  OpBuilder b(op);
  auto fieldType = cast<FieldType>(op.getResults()[0].getType());
  Type type = fieldType.getKernelValueType();
  int64_t k = fieldType.getNumComponents();
  auto neighborhood = op.getRelation().getDefiningOp<NeighborhoodOp>();
  Storage &halo = storage[neighborhood.getHalo()];
  Storage &layout = storage[layoutOf(neighborhood.getHalo())];
  // With two results the contributions of each pair go to both of its
  // particles; over a directed relation, where every pair is visited at
  // both of its owners, each with the weight one half (the transpose of a
  // directed md_dist.sum_pairs, D276).
  bool twoEnded = op.getResults().size() == 2;
  bool directed =
      op.getRelation().getType().getTraversal() == Traversal::Directed;
  double weight = directed && twoEnded ? 0.5 : 1.0;
  auto wordBuffer = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic},
      sumType(b.getF64Type()));
  auto allocateSums = [&](Value rows) {
    return allocate(b, wordBuffer, {index(b, domains), rows, index(b, k)});
  };
  Value ownedOut = twoEnded ? allocateSums(layout.capacity)
                            : allocate(b, layout.capacity, k);
  Value raw;
  if (!twoEnded && needsWords(op.getResults()[0])) {
    raw = allocateSums(layout.capacity);
    fieldWords[op.getResults()[0]] = raw;
  }
  Value ghostOut;
  if (twoEnded) {
    initializeBuffer(b, ownedOut, true);
    ghostOut = allocateSums(halo.capacity);
    initializeBuffer(b, ghostOut, true);
  }
  // The value at the second particle of a pair computed once: the
  // kernel's value under exchange, which the verifier proved.
  double sign = op.getExchange() == md::Exchange::Antisymmetric ? -1.0 : 1.0;
  auto scale = [&](OpBuilder &at, Value value, double factor) -> Value {
    if (factor == 1.0)
      return value;
    Value s = real(at, factor);
    if (auto vector = dyn_cast<VectorType>(type))
      s = vector::BroadcastOp::create(at, loc, vector, s);
    return arith::MulFOp::create(at, loc, value, s);
  };
  emitPairs(
      op, loc, domainBound(b, false), domainBound(b, true), parallelDomains, storage, sumType(type), imageShifts.lookup(op.getGhosts()),
      [&](OpBuilder &at, Value a, Value v) { return add(at, a, v); },
      [&](OpBuilder &at, Type t) { return zero(at, t); },
      [&](OpBuilder &at, Value buffer, Value d, Value row, Type t) {
        return loadValue(at, buffer, d, row, t);
      },
      [&](OpBuilder &at, Block &kernel, ValueRange arguments) {
        return inlineKernel(at, kernel, arguments);
      },
      [&](OpBuilder &at, Value value, Value acc, Value d, Value kRow,
          Value isOwned, Value ghostRow) {
        if (twoEnded) {
          Value other =
              quantize(at, scale(at, value, sign * weight), pairForce);
          scf::IfOp::create(
              at, loc, isOwned,
              [&](OpBuilder &t, Location) {
                addValue(t, other, ownedOut, d, kRow);
                scf::YieldOp::create(t, loc);
              },
              [&](OpBuilder &e, Location) {
                addValue(e, other, ghostOut, d, ghostRow);
                scf::YieldOp::create(e, loc);
              });
        }
        return add(at, acc,
                   quantize(at, fixedPoint ? scale(at, value, weight) : value,
                            pairForce));
      },
      [&](OpBuilder &at, Value sum, Value d, Value i) {
        if (op.getRelation().getType().getTraversal() == Traversal::Midpoint) {
          Value n = toIndex(at, memref::LoadOp::create(at, loc, layout.rows, d));
          Value local = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, i, n);
          scf::IfOp::create(at, loc, local, [&](OpBuilder &t, Location) {
            addValue(t, sum, ownedOut, d, i);
            scf::YieldOp::create(t, loc);
          }, [&](OpBuilder &e, Location) {
            Value r = arith::SubIOp::create(e, loc, i, n);
            addValue(e, sum, ghostOut, d, r);
            scf::YieldOp::create(e, loc);
          });
          return;
        }
        if (raw) storeValue(at, sum, raw, d, i);
        if (twoEnded)
          addValue(at, fixedPoint ? sum : scale(at, sum, weight), ownedOut, d,
                   i);
        else
          storeValue(at, finish(at, sum, pairForce), ownedOut, d, i);
      },
      [&](OpBuilder &, Value, Value) {});
  storage[op.getResults()[0]].buffer = ownedOut;
  if (twoEnded)
    storage[op.getResults()[1]].buffer = ghostOut;
  return success();
}

//===----------------------------------------------------------------------===//
// Particles and reductions
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lowerSumParticles(SumParticlesOp op) {
  OpBuilder b(op);
  Type type = op.getResult().getType().getValueType();
  auto vector = dyn_cast<VectorType>(type);
  int64_t length = vector ? vector.getNumElements() : 1;
  auto partialType = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, sumType(b.getF64Type()));
  Value partial =
      allocate(b, partialType, {index(b, domains), index(b, length)});
  Storage &rows = rowsOf(op.getGathered()[0]);
  scf::ForOp::create(
      b, loc, domainBound(b, false), domainBound(b, true), index(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value count = toIndex(
            bd, memref::LoadOp::create(bd, loc, rows.rows, ValueRange{d}));
        auto loop = scf::ForOp::create(
            bd, loc, index(bd, 0), count, index(bd, 1),
            ValueRange{zero(bd, sumType(type))},
            [&](OpBuilder &br, Location, Value r, ValueRange acc) {
              SmallVector<Value> arguments;
              for (Value field : op.getGathered())
                arguments.push_back(loadValue(
                    br, storage[field].buffer, d, r,
                    cast<FieldType>(field.getType()).getKernelValueType()));
              Value value =
                  inlineKernel(br, op.getKernel().front(), arguments);
              scf::YieldOp::create(
                  br, loc,
                  ValueRange{
                      add(br, acc[0], quantize(br, value, particleSum))});
            });
        Value sum = loop.getResult(0);
        for (int64_t c = 0; c < length; ++c) {
          Value part =
              vector ? Value(vector::ExtractOp::create(bd, loc, sum, c)) : sum;
          memref::StoreOp::create(bd, loc, part, partial,
                                  ValueRange{d, index(bd, c)});
        }
        scf::YieldOp::create(bd, loc);
      });
  storage[op.getResult()].buffer = partial;
  return success();
}

// Reporting also reads complete fields in particle-set order, including the
// initial kinetic energy outside the domain step loop. Give these reductions
// the same integer arithmetic as a sum of owned rows.
void Lowering::lowerCollectedSum(md::SumParticlesOp op) {
  OpBuilder b(op);
  loc = op.getLoc();
  end = op->getBlock()->getTerminator();
  SmallVector<Value> buffers;
  for (Value field : op.getGathered()) {
    auto type = cast<md::FieldType>(field.getType());
    buffers.push_back(inputSnapshot(b, type, field));
  }
  Value count = memref::DimOp::create(b, loc, buffers.front(), 0);
  Block &kernel = op.getKernel().front();
  Format format = getCollectedFormat(op);
  auto loop = scf::ForOp::create(
      b, loc, index(b, 0), count, index(b, 1),
      ValueRange{zero(b, sumType(op.getResult().getType()))},
      [&](OpBuilder &body, Location, Value row, ValueRange acc) {
        IRMapping mapping;
        for (auto [argument, buffer] :
             llvm::zip(kernel.getArguments(), buffers)) {
          Value value;
          if (auto vector = dyn_cast<VectorType>(argument.getType())) {
            SmallVector<Value> parts;
            for (int64_t c = 0; c < vector.getNumElements(); ++c)
              parts.push_back(memref::LoadOp::create(
                  body, loc, buffer, ValueRange{row, index(body, c)}));
            value = vector::FromElementsOp::create(body, loc, vector, parts);
          } else {
            value = memref::LoadOp::create(body, loc, buffer, ValueRange{row});
          }
          mapping.map(argument, value);
        }
        for (Operation &nested : kernel.without_terminator())
          body.clone(nested, mapping);
        Value term =
            mapping.lookupOrDefault(kernel.getTerminator()->getOperand(0));
        scf::YieldOp::create(body, loc,
                             add(body, acc[0], quantize(body, term, format)));
      });
  Value total = loop.getResult(0);
  if (shardExtent.count(op)) {
    auto scalar = [&](Value v) {
      auto fn = declare(fixedPoint ? "mdrtMPICombineWide" : "mdrtMPICombineDouble",
          b.getFunctionType({v.getType()},{v.getType()}),false);
      return func::CallOp::create(b,loc,fn,v).getResult(0);
    };
    if (auto vec = dyn_cast<VectorType>(total.getType())) {
      SmallVector<Value> parts;
      for (int i=0; i<vec.getNumElements(); ++i)
        parts.push_back(scalar(vector::ExtractOp::create(b,loc,total,i)));
      total = vector::FromElementsOp::create(b,loc,vec,parts);
    } else total = scalar(total);
  }
  op.replaceAllUsesWith(decodeObservable(b, total, format));
  op.erase();
}

LogicalResult Lowering::lowerMapParticles(MapParticlesOp op) {
  OpBuilder b(op);
  FieldType resultType = op.getResult().getType();
  Storage &rows = rowsOf(op.getGathered()[0]);
  Value result = allocate(b, rows.capacity, resultType.getNumComponents());
  bool combine = fixedPoint && op->hasAttr("md_dist.add_terms");
  Value raw;
  if (combine && needsWords(op.getResult())) {
    raw =
        allocate(b,
                 MemRefType::get(cast<MemRefType>(result.getType()).getShape(),
                                 b.getIntegerType(128)),
                 {index(b, domains), rows.capacity,
                  index(b, resultType.getNumComponents())});
    fieldWords[op.getResult()] = raw;
  }
  auto mapRow = [&](OpBuilder &br, Value d, Value r) {
    SmallVector<Value> arguments;
    Value sum;
    for (Value field : op.getGathered()) {
      Type type = cast<FieldType>(field.getType()).getKernelValueType();
      Value words = fieldWords.lookup(field);
      if (combine && words) {
        Value term = loadValue(br, words, d, r, sumType(type));
        sum = sum ? add(br, sum, term) : term;
      } else {
        Value term = loadValue(br, storage[field].buffer, d, r, type);
        if (combine) {
          term = quantize(br, term, pairForce);
          sum = sum ? add(br, sum, term) : term;
        } else
          arguments.push_back(term);
      }
    }
    if (raw) storeValue(br, sum, raw, d, r);
    Value value = combine ? finish(br, sum, pairForce)
                          : inlineKernel(br, op.getKernel().front(), arguments);
    storeValue(br, value, result, d, r);
  };
  Value first = domainBound(b, false), past = domainBound(b, true);
  Value z = index(b, 0), one = index(b, 1);
  if (parallelDomains) {
    scf::ParallelOp::create(b, loc, ValueRange{first, z}, ValueRange{past, rows.capacity},
        ValueRange{one, one}, [&](OpBuilder &at, Location, ValueRange iv) {
          Value count = toIndex(at, memref::LoadOp::create(at, loc, rows.rows, iv[0]));
          Value live = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, iv[1], count);
          scf::IfOp::create(at, loc, live, [&](OpBuilder &br, Location) {
            mapRow(br, iv[0], iv[1]);
            scf::YieldOp::create(br, loc);
          });
        });
  } else {
    scf::ForOp::create(b, loc, first, past, one, ValueRange{},
        [&](OpBuilder &at, Location, Value d, ValueRange) {
          Value count = toIndex(at, memref::LoadOp::create(at, loc, rows.rows, d));
          scf::ForOp::create(at, loc, z, count, one, ValueRange{},
              [&](OpBuilder &br, Location, Value r, ValueRange) {
                mapRow(br, d, r);
                scf::YieldOp::create(br, loc);
              });
          scf::YieldOp::create(at, loc);
        });
  }
  storage[op.getResult()].buffer = result;
  return success();
}

LogicalResult Lowering::lowerAllReduce(AllReduceOp op) {
  if (failed(checkAlgebra(op)))
    return failure();
  OpBuilder b(op);
  Type type = op.getResult().getType();
  auto vector = dyn_cast<VectorType>(type);
  int64_t length = vector ? vector.getNumElements() : 1;
  Value partial = storage[op.getPartial()].buffer;
  if (mpi)
    func::CallOp::create(b, loc,
        declare(fixedPoint ? "mdrtMPIShareWords" : "mdrtMPISharePartials", b.getFunctionType({partial.getType()}, {}), true),
        ValueRange{partial});
  // The partial sums are added in the order of the domains.
  auto loop = scf::ForOp::create(
      b, loc, index(b, 0), index(b, domains), index(b, 1),
      ValueRange{zero(b, sumType(type))},
      [&](OpBuilder &bd, Location, Value d, ValueRange acc) {
        SmallVector<Value> parts;
        for (int64_t c = 0; c < length; ++c)
          parts.push_back(memref::LoadOp::create(
              bd, loc, partial, ValueRange{d, index(bd, c)}));
        Value value =
            vector ? Value(vector::FromElementsOp::create(
                         bd, loc, cast<VectorType>(sumType(type)), parts))
                   : parts[0];
        scf::YieldOp::create(bd, loc, ValueRange{add(bd, acc[0], value)});
      });
  // A directed relation visits each pair twice: its words hold twice the
  // sum, which one more fractional bit halves exactly.
  Format format = particleSum;
  if (auto pairs = op.getPartial().getDefiningOp<SumPairsOp>()) {
    format = pairEnergy;
    if (pairs.getRelation().getType().getTraversal() == Traversal::Directed)
      ++format.bits;
  }
  op.getResult().replaceAllUsesWith(
      decodeObservable(b, loop.getResult(0), format));
  return success();
}

//===----------------------------------------------------------------------===//
// The state of a step loop (D399)
//===----------------------------------------------------------------------===//

SmallVector<Value> Lowering::stateOf(Value state) {
  auto found = storage.find(state);
  if (found != storage.end() && !found->second.state.empty())
    return found->second.state;
  // A block argument or a result of control flow, after the structural
  // conversion: the cast from its tuple.
  if (auto cast = state.getDefiningOp<UnrealizedConversionCastOp>())
    return SmallVector<Value>(cast.getInputs());
  return {};
}

Value Lowering::copyForState(OpBuilder &b, Value buffer) {
  auto type = cast<MemRefType>(buffer.getType());
  SmallVector<Value> sizes;
  for (int64_t i = 0; i < type.getRank(); ++i)
    sizes.push_back(memref::DimOp::create(b, loc, buffer, i));
  Value copy = memref::AllocOp::create(b, loc, type, sizes);
  initializeBuffer(b, copy, true, buffer);
  return copy;
}

LogicalResult Lowering::lowerPack(PackOp op) {
  OpBuilder b(op);
  Type i64 = b.getI64Type();
  Storage &layout = storage[op.getLayout()];
  Storage &halo = storage[op.getHalo()];
  Value reference = storage[referenceOf(op.getHalo())].buffer;
  if (!layout.handle || !halo.handle || !reference)
    return op.emitOpError() << "cannot find the storage of the state";
  // The state holds a reference to the layout and to the halo, and copies
  // of the buffers, so that every buffer of the block is still freed at its
  // end and the halo events of D310 see allocations of their block.
  func::CallOp::create(
      b, loc,
      declare("mdrtDistRetainLayout", b.getFunctionType({i64}, {}), false),
      ValueRange{layout.handle});
  func::CallOp::create(
      b, loc,
      declare("mdrtDistRetainHalo", b.getFunctionType({i64}, {}), false),
      ValueRange{halo.handle});
  SmallVector<Value> tuple = {layout.handle, halo.handle,
                              copyForState(b, reference)};
  for (Value field : op.getFields()) {
    Value buffer = storage[field].buffer;
    if (!buffer)
      return op.emitOpError() << "cannot find the buffer of a field";
    tuple.push_back(copyForState(b, buffer));
  }
  storage[op.getResult()].state = tuple;
  return success();
}

LogicalResult Lowering::lowerUnpack(UnpackOp op) {
  OpBuilder b(op);
  SmallVector<Value> tuple = stateOf(op.getState());
  if (tuple.size() != op.getFields().size() + 3)
    return op.emitOpError() << "cannot find the tuple of the state";
  Type i64 = b.getI64Type(), f64 = b.getF64Type();
  Value edges[3];
  if (failed(getEdges(op, op.getCell(), edges)))
    return failure();
  Value layoutHandle = tuple[0], haloHandle = tuple[1];
  // The cell the state is used with must be the halo's: its images and the
  // regions of ownership were made in it.
  func::CallOp::create(
      b, loc,
      declare("mdrtDistCheckHaloCell",
              b.getFunctionType({i64, f64, f64, f64}, {}), false),
      ValueRange{haloHandle, edges[0], edges[1], edges[2]});
  // The layout: the unpack takes over the state's reference to it, and
  // releases it at the end of its block.
  Value count = toIndex(
      b, func::CallOp::create(
             b, loc,
             declare("mdrtDistLayoutCount", b.getFunctionType({i64}, {i64}),
                     false),
             ValueRange{layoutHandle})
             .getResult(0));
  Value capacity = ownedCapacity(b, layoutHandle);
  auto [rows, ids] = rowViews(b, layoutHandle, false);
  // The halo, likewise.
  Value haloCapacity = toIndex(
      b, func::CallOp::create(
             b, loc,
             declare("mdrtDistHaloCapacity", b.getFunctionType({i64}, {i64}),
                     false),
             ValueRange{haloHandle})
             .getResult(0));
  auto [haloRows, haloIds] = rowViews(b, haloHandle, true);
  OpBuilder atEnd(end);
  func::CallOp::create(
      atEnd, loc,
      declare("mdrtDistReleaseHalo", b.getFunctionType({i64}, {}), false),
      ValueRange{haloHandle});
  func::CallOp::create(
      atEnd, loc,
      declare("mdrtDistReleaseLayout", b.getFunctionType({i64}, {}), false),
      ValueRange{layoutHandle});
  Storage &layout = storage[op.getLayout()];
  layout.handle = layoutHandle;
  layout.rows = rows;
  layout.ids = ids;
  layout.capacity = capacity;
  layout.count = count;
  Storage &halo = storage[op.getHalo()];
  halo.handle = haloHandle;
  halo.rows = haloRows;
  halo.ids = haloIds;
  halo.capacity = haloCapacity;
  // Pack allocated each tuple field independently. Consuming the state
  // transfers those allocations to this block until its terminator.
  SmallVector<Value> results = {op.getReference()};
  results.append(op.getFields().begin(), op.getFields().end());
  for (auto [result, buffer] :
       llvm::zip(results, ArrayRef<Value>(tuple).drop_front(2))) {
    // A transfer event requires a direct allocation for its alias proof.
    if (haloEvents && llvm::any_of(result.getUsers(), [](Operation *use) {
          return isa<HaloCopyOp, HaloSumOp>(use);
        })) {
      Value copy = copyForState(b, buffer);
      storage[result].buffer = copy;
      memref::DeallocOp::create(b, loc, buffer);
      memref::DeallocOp::create(atEnd, loc, copy);
    } else {
      storage[result].buffer = buffer;
      memref::DeallocOp::create(atEnd, loc, buffer);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// The second executor: each domain through md_exec (D290)
//===----------------------------------------------------------------------===//

FlatSymbolRefAttr Lowering::getSet(int64_t d, bool local, Value epoch) {
  // A set of md_exec has one size in a function, and the rows of a domain
  // change with its epoch: a layout for the owned rows, a halo for the local
  // ones (D399). The first epoch keeps the plain name.
  auto &epochs = local ? localEpochs : ownedEpochs;
  int64_t number = epochs.try_emplace(epoch, epochs.size()).first->second;
  std::string name =
      (decomposition.getSymName() +
       (number ? ".e" + Twine(number) : Twine()) + ".d" + Twine(d) +
       (local ? ".local" : ".owned"))
          .str();
  if (!SymbolTable::lookupSymbolIn(module, StringAttr::get(module.getContext(), name))) {
    OpBuilder b(decomposition);
    md::ParticleSetOp::create(b, decomposition.getLoc(),
                              StringAttr::get(module.getContext(), name));
  }
  return FlatSymbolRefAttr::get(module.getContext(), name);
}

void Lowering::declareSets() {
  OpBuilder b(decomposition);
  for (int64_t d = 0; d < domains; ++d)
    for (bool local : {false, true}) {
      std::string name = (decomposition.getSymName() + ".d" + Twine(d) +
                          (local ? ".local" : ".owned"))
                             .str();
      if (!SymbolTable::lookupSymbolIn(module, StringAttr::get(module.getContext(), name)))
        md::ParticleSetOp::create(b, decomposition.getLoc(), name);
    }
}

Value Lowering::countOf(OpBuilder &b, Storage &rows, Value d) {
  return toIndex(b, memref::LoadOp::create(b, loc, rows.rows, d));
}

Value Lowering::makeField(OpBuilder &b, FlatSymbolRefAttr set, int64_t k,
                          Type element, Value d, ArrayRef<Value> parts,
                          ArrayRef<Value> counts, Value ghostShift) {
  auto fieldType = md::FieldType::get(module.getContext(), set, k, element);
  MemRefType type = mdrt::getBufferType(fieldType);
  Value total = index(b, 0);
  for (Value count : counts)
    total = arith::AddIOp::create(b, loc, total, count);
  // The buffer is handed to md_exec, which owns it from then on.
  Value buffer = memref::AllocOp::create(b, loc, type, ValueRange{total});
  Value offset = index(b, 0);
  for (auto [part, count] : llvm::zip(parts, counts)) {
    auto partType = cast<MemRefType>(part.getType());
    scf::ForOp::create(
        b, loc, index(b, 0), count, index(b, 1), ValueRange(),
        [&](OpBuilder &br, Location, Value r, ValueRange) {
          Value row = arith::AddIOp::create(br, loc, offset, r);
          for (int64_t c = 0; c < k; ++c) {
            SmallVector<Value, 3> from = {d, r};
            if (partType.getRank() == 3)
              from.push_back(index(br, c));
            Value v = memref::LoadOp::create(br, loc, part, from);
            if (ghostShift && part == parts.back()) {
              Value shift = memref::LoadOp::create(br, loc, ghostShift, from);
              v = arith::AddFOp::create(br, loc, v, shift);
            }
            if (element.isInteger(32) && v.getType().isF64())
              v = arith::FPToSIOp::create(br, loc, element, v);
            SmallVector<Value, 2> to = {row};
            if (type.getRank() == 2)
              to.push_back(index(br, c));
            memref::StoreOp::create(br, loc, v, buffer, to);
          }
          scf::YieldOp::create(br, loc);
        });
    offset = arith::AddIOp::create(b, loc, offset, count);
  }
  return mdrt::FromBufferOp::create(b, loc, fieldType, buffer);
}

void Lowering::copyBack(OpBuilder &b, Value field, Value first, Value count,
                        Value buffer, Value d) {
  auto fieldType = cast<md::FieldType>(field.getType());
  MemRefType type = mdrt::getBufferType(fieldType);
  int64_t k = fieldType.getNumComponents();
  Value flat = mdrt::ToBufferOp::create(b, loc, type, field);
  scf::ForOp::create(
      b, loc, index(b, 0), count, index(b, 1), ValueRange(),
      [&](OpBuilder &br, Location, Value r, ValueRange) {
        Value row = arith::AddIOp::create(br, loc, first, r);
        for (int64_t c = 0; c < k; ++c) {
          SmallVector<Value, 2> from = {row};
          if (type.getRank() == 2)
            from.push_back(index(br, c));
          Value v = memref::LoadOp::create(br, loc, flat, from);
          memref::StoreOp::create(br, loc, v, buffer,
                                  ValueRange{d, r, index(br, c)});
        }
        scf::YieldOp::create(br, loc);
      });
}

LogicalResult Lowering::lowerNeighborhoodExec(NeighborhoodOp op) {
  if (!matrixPairs && op.getResult().getType().getTraversal() == Traversal::Midpoint)
    return op.emitOpError("midpoint requires the distributed CPU pair lowering");
  // The precondition under which the relation is that of md.neighborhood,
  // as in the reference executor.
  if (failed(lowerNeighborhood(op)))
    return failure();
  auto &local = relations[op.getResult()];
  if (!matrixPairs) {
    OpBuilder b(op);
    for (int64_t d = 0; d < domains; ++d)
      local.push_back(buildLocalRelation(b, op, index(b, d),
                          getSet(d, true, op.getHalo())));
  }
  return success();
}

DomainRelation Lowering::buildLocalRelation(OpBuilder &b, NeighborhoodOp op,
                                             Value d, FlatSymbolRefAttr set) {
  Storage &halo = storage[op.getHalo()];
  Storage &layout = storage[layoutOf(op.getHalo())];
  Value owned = storage[op.getPositions()].buffer;
  Value ghosts = storage[op.getGhosts()].buffer;
  double cutoff = op.getCutoff().convertToDouble();
  // The structure is built at every evaluation, with the halo: no skin.
  double skin = 0.0;
  MLIRContext *context = module.getContext();
  DomainRelation relation;
  relation.cell = cellOf(op.getHalo());
  Value nOwned = countOf(b, layout, d);
  Value nGhosts = countOf(b, halo, d);
  // The local rows, owned then ghost; the ghosts carry their images.
  Value positions = makeField(b, set, 3, b.getF64Type(), d,
                              {owned, ghosts}, {nOwned, nGhosts}, imageShifts.lookup(op.getGhosts()));
  if (matrixPairs) {
    auto traversal = op.getResult().getType().getTraversal();
    bool midpoint = traversal == Traversal::Midpoint;
    Value centers = midpoint ? Value(arith::AddIOp::create(b, loc, nOwned, nGhosts)) : nOwned;
    Value excluded = excludedPairs.lookup(op.getResult());
    if (!excluded)
      excluded = arith::ConstantIntOp::create(b, loc, 0, 64);
    Value packed = positions.getDefiningOp<mdrt::FromBufferOp>().getBuffer();
    Value structure = mdrt::DomainMatrixOp::create(b, loc,
        md_exec::getBuildType(set, md_exec::NeighborKind::Matrix, md_exec::Images::Explicit),
        halo.handle, d, centers, packed, excluded, owned, ghosts,
        APFloat(cutoff), APFloat(matrixSkin), width,
        midpoint ? 2 : traversal == Traversal::Unique ? 1 : 0);
    return {positions, structure, nOwned, relation.cell, centers};
  }
  Value cells = md_exec::BuildCellsOp::create(
      b, loc, mdrt::CellsType::get(context, set), positions,
      relation.cell,
      APFloat((cutoff + skin) / static_cast<double>(cellsPerReach)));
  Value structure = md_exec::BuildNeighborsOp::create(
      b, loc,
      md_exec::getBuildType(set, md_exec::NeighborKind::Matrix,
                            md_exec::Images::Explicit),
      cells, positions, relation.cell, /*excluded=*/Value(), nOwned,
      APFloat(cutoff),
      APFloat(skin), md_exec::NeighborKind::Matrix,
      static_cast<uint64_t>(width), md_exec::Images::Explicit);
  return {positions, structure, nOwned, relation.cell, nOwned};
}

void Lowering::addPairKernel(md_exec::PairForOp loop, Operation *op,
                             bool unique, double scale) {
  Block &source = op->getRegion(0).front();
  Block *block = new Block();
  loop.getKernel().push_back(block);
  Type f64 = Float64Type::get(module.getContext());
  block->addArgument(f64, loc);
  block->addArgument(VectorType::get({3}, f64), loc);
  for (Value field : loop.getIns()) {
    Type type = cast<md::FieldType>(field.getType()).getKernelValueType();
    block->addArgument(type, loc);
    block->addArgument(type, loc);
  }
  OpBuilder b = OpBuilder::atBlockEnd(block);
  IRMapping mapping;
  // The kernel of md_dist takes the distance.
  mapping.map(source.getArgument(0),
              math::SqrtOp::create(b, loc, block->getArgument(0)));
  mapping.map(source.getArgument(1), block->getArgument(1));
  for (unsigned i = 2, e = source.getNumArguments(); i != e; ++i)
    mapping.map(source.getArgument(i), block->getArgument(i));
  for (Operation &nested : source.without_terminator())
    b.clone(nested, mapping);
  Value value = mapping.lookupOrDefault(
      cast<YieldOp>(source.getTerminator()).getValues()[0]);
  Type type = value.getType();
  if (unique) {
    // The identities of the two rows are the last field of `ins`: each pair
    // is taken where the identity of its center is the lesser
    // (Section 4.6).
    unsigned last = block->getNumArguments();
    Value lesser = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                         block->getArgument(last - 2),
                                         block->getArgument(last - 1));
    value = arith::SelectOp::create(b, loc, lesser, value, zero(b, type));
  }
  if (scale != 1.0) {
    Value s = real(b, scale);
    if (auto vector = dyn_cast<VectorType>(type))
      s = vector::BroadcastOp::create(b, loc, vector, s);
    value = arith::MulFOp::create(b, loc, value, s);
  }
  md_exec::YieldOp::create(b, loc, ValueRange{value});
}

md_exec::PairForOp
Lowering::createPairLoop(OpBuilder &b, Operation *op, Value d, FlatSymbolRefAttr set, DomainRelation relation,
                         TypeRange results, ValueRange outs, ValueRange reduce,
                         DenseF64ArrayAttr weights, bool atomic,
                         double scale) {
  Value relationValue = op->getOperand(0);
  auto relationType = cast<RelationType>(relationValue.getType());
  bool unique = !matrixPairs && relationType.getTraversal() == Traversal::Unique;
  auto neighborhood = relationValue.getDefiningOp<NeighborhoodOp>();
  Storage &halo = storage[neighborhood.getHalo()];
  Storage &layout = storage[layoutOf(neighborhood.getHalo())];
  Value nOwned = relation.owned;
  Value nGhosts = countOf(b, halo, d);
  Value selected = isa<SumPairsOp>(op)
                       ? cast<SumPairsOp>(op).getCenterSet()
                       : cast<GatherPairsOp>(op).getCenterSet();
  Value centerSet;
  bool ghostFree = false;
  if (selected) {
    centerSet = makeField(b, set, 1, b.getI32Type(), d,
                          {storage[selected].buffer}, {nOwned});
    ghostFree = selected ==
                selected.getDefiningOp<PartitionCentersOp>().getInterior();
  }

  // Each gathered field as one field of the local rows: its owned rows and
  // its copy at the ghosts.
  ValueRange gathered = isa<SumPairsOp>(op)
                            ? cast<SumPairsOp>(op).getGathered()
                            : cast<GatherPairsOp>(op).getGathered();
  SmallVector<Value> ins;
  for (unsigned g = 0; g < gathered.size(); g += 2) {
    auto type = cast<FieldType>(gathered[g].getType());
    auto key = std::make_pair(gathered[g], gathered[g + 1]);
    Value packed = matrixPairs ? domainFields.lookup(key) : Value();
    if (!packed) {
      packed = makeField(b, set, type.getNumComponents(), type.getElementType(), d,
                         {storage[key.first].buffer, storage[key.second].buffer},
                         {nOwned, nGhosts});
      if (matrixPairs)
        domainFields[key] = packed;
    }
    ins.push_back(packed);
  }
  if (unique)
    ins.push_back(makeField(b, set, 1, b.getI64Type(), d,
                            {layout.ids, halo.ids}, {nOwned, nGhosts}));

  md::Exchange exchange = isa<SumPairsOp>(op)
                              ? cast<SumPairsOp>(op).getExchange()
                              : cast<GatherPairsOp>(op).getExchange();
  MLIRContext *context = module.getContext();
  ArrayAttr exchangeAttr =
      ArrayAttr::get(context, {md::ExchangeAttr::get(context, exchange)});
  double cutoff = neighborhood.getCutoff().convertToDouble();
  auto loop = md_exec::PairForOp::create(
      b, loc, results, relation.neighbors, relation.positions,
      relation.cell, ins, outs, reduce, /*scratch=*/ValueRange(),
      b.getF64FloatAttr(cutoff), weights, /*overwrite=*/DenseBoolArrayAttr(),
      md_exec::TraversalAttr::get(context, md_exec::Traversal::Directed),
      md_exec::ConflictAttr::get(context, atomic
                                              ? md_exec::Conflict::Atomic
                                              : md_exec::Conflict::OwnerOnly),
      exchangeAttr, relation.centers, centerSet, b.getBoolAttr(ghostFree),
      md_exec::ImagesAttr::get(context, md_exec::Images::Explicit),
      /*scatter=*/md_exec::ScatterAttr(),
      /*hostLanes=*/IntegerAttr());
  addPairKernel(loop, op, unique, scale);
  if (matrixPairs) lastPair = loop;
  return loop;
}

void Lowering::forPairDomains(OpBuilder &b, Value rel,
    function_ref<void(OpBuilder &, Value, FlatSymbolRefAttr, DomainRelation)> emit) {
  auto n = rel.getDefiningOp<NeighborhoodOp>();
  if (!matrixPairs) {
    for (int64_t i = 0; i < domains; ++i)
      emit(b, index(b, i), getSet(i, true, n.getHalo()), relations[rel][i]);
    return;
  }
  Value d = domainLoop.getInductionVar();
  OpBuilder body(domainLoop.getBody()->getTerminator());
  if (lastPair)
    body.setInsertionPointAfter(lastPair);
  auto set = getSet(0, true, n.getHalo());
  emit(body, d, set, domainRelation);
}

LogicalResult Lowering::lowerPairGroup(Operation *anchor, DominanceInfo &dominance) {
  Value rel = anchor->getOperand(0);
  auto n = rel.getDefiningOp<NeighborhoodOp>();
  SmallVector<Operation *> group{anchor};
  // A later many-body stage can read the result of an earlier scatter.
  // Only bring a kernel forward when all its inputs already dominate the
  // first evaluation, including tables and scalars captured by its body.
  for (Operation *user : rel.getUsers()) {
    if (!isa<SumPairsOp, GatherPairsOp>(user) || user->getBlock() != anchor->getBlock() ||
        !anchor->isBeforeInBlock(user) || grouped.contains(user))
      continue;
    llvm::SetVector<Value> inputs;
    inputs.insert(user->getOperands().begin(), user->getOperands().end());
    getUsedValuesDefinedAbove(user->getRegion(0), inputs);
    if (llvm::all_of(inputs, [&](Value v) { return dominance.dominates(v, anchor); }))
      group.push_back(user);
  }
  llvm::sort(group, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  OpBuilder b(anchor);
  domainLoop = scf::ForOp::create(b, loc, domainBound(b, false), domainBound(b, true), index(b, 1));
  domainLoop->setAttr("md_dist.domain_loop", b.getUnitAttr());
  OpBuilder body(domainLoop.getBody()->getTerminator());
  domainRelation = buildLocalRelation(body, n, domainLoop.getInductionVar(), getSet(0, true, n.getHalo()));
  for (Operation *op : group) {
    grouped.insert(op);
    if (auto sum = dyn_cast<SumPairsOp>(op)) {
      if (failed(lowerSumPairsExec(sum))) return failure();
    } else if (failed(lowerGatherPairsExec(cast<GatherPairsOp>(op)))) {
      return failure();
    }
  }
  domainFields.clear();
  lastPair = {};
  domainLoop = {};
  return success();
}

LogicalResult Lowering::lowerSumPairsExec(SumPairsOp op) {
  OpBuilder b(matrixPairs ? domainLoop.getOperation() : op.getOperation());
  Type type = op.getResult().getType().getValueType();
  auto vector = dyn_cast<VectorType>(type);
  int64_t length = vector ? vector.getNumElements() : 1;
  auto partialType = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, b.getF64Type());
  Value partial =
      allocate(b, partialType, {index(b, domains), index(b, length)});
  bool directed =
      op.getRelation().getType().getTraversal() == Traversal::Directed;
  // Over a directed relation every pair is visited at both of its owners.
  DenseF64ArrayAttr weights;
  if (directed)
    weights = b.getDenseF64ArrayAttr({0.5});
  forPairDomains(b, op.getRelation(), [&](OpBuilder &b, Value d, FlatSymbolRefAttr set, DomainRelation relation) {
    Value init = zero(b, type);
    auto loop = createPairLoop(b, op, d, set, relation, TypeRange{type}, ValueRange(),
                               ValueRange{init}, weights, /*atomic=*/matrixPairs && !directed,
                               /*scale=*/1.0);
    Value sum = loop.getResult(0);
    for (int64_t c = 0; c < length; ++c) {
      Value part =
          vector ? Value(vector::ExtractOp::create(b, loc, sum, c)) : sum;
      memref::StoreOp::create(b, loc, part, partial,
                              ValueRange{d, index(b, c)});
    }
  });
  storage[op.getResult()].buffer = partial;
  return success();
}

LogicalResult Lowering::lowerGatherPairsExec(GatherPairsOp op) {
  OpBuilder b(matrixPairs ? domainLoop.getOperation() : op.getOperation());
  auto fieldType = cast<FieldType>(op.getResults()[0].getType());
  int64_t k = fieldType.getNumComponents();
  auto neighborhood = op.getRelation().getDefiningOp<NeighborhoodOp>();
  Storage &halo = storage[neighborhood.getHalo()];
  Storage &layout = storage[layoutOf(neighborhood.getHalo())];
  // With two results, the contributions of each pair go to both of its
  // particles, with the weight one half over a directed relation (D276).
  bool twoEnded = op.getResults().size() == 2;
  bool directed =
      op.getRelation().getType().getTraversal() == Traversal::Directed;
  double scale = directed && twoEnded ? 0.5 : 1.0;
  Value ownedOut = allocate(b, layout.capacity, k);
  Value ghostOut;
  if (twoEnded)
    ghostOut = allocate(b, halo.capacity, k);
  forPairDomains(b, op.getRelation(), [&](OpBuilder &b, Value d, FlatSymbolRefAttr set, DomainRelation relation) {
    auto localType = md::FieldType::get(module.getContext(),
                                        set,
                                        k,
                                        b.getF64Type());
    Value destination = md_exec::ZerosOp::create(b, loc, localType);
    auto loop = createPairLoop(b, op, d, set, relation, TypeRange{localType},
                               ValueRange{destination}, ValueRange(),
                               DenseF64ArrayAttr(), /*atomic=*/twoEnded,
                               scale);
    Value nOwned = relation.owned;
    copyBack(b, loop.getResult(0), index(b, 0), nOwned, ownedOut, d);
    if (twoEnded)
      copyBack(b, loop.getResult(0), nOwned, countOf(b, halo, d), ghostOut,
               d);
  });
  storage[op.getResults()[0]].buffer = ownedOut;
  if (twoEnded)
    storage[op.getResults()[1]].buffer = ghostOut;
  return success();
}

/// The fields of a loop over particles of the second executor: owned rows.
static LogicalResult checkOwnedRows(Operation *op, ValueRange gathered) {
  for (Value field : gathered)
    if (!cast<FieldType>(field.getType()).isOwned())
      return op->emitOpError()
             << "runs over ghost rows, which the executor through md_exec "
                "does not run; the reference executor does";
  return success();
}

LogicalResult Lowering::lowerSumParticlesExec(SumParticlesOp op) {
  if (failed(checkOwnedRows(op, op.getGathered())))
    return failure();
  OpBuilder b(op);
  Type type = op.getResult().getType().getValueType();
  auto vector = dyn_cast<VectorType>(type);
  int64_t length = vector ? vector.getNumElements() : 1;
  auto partialType = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, b.getF64Type());
  Value partial =
      allocate(b, partialType, {index(b, domains), index(b, length)});
  Storage &layout = rowsOf(op.getGathered()[0]);
  for (int64_t d = 0; d < domains; ++d) {
    FlatSymbolRefAttr set =
        getSet(d, /*local=*/false, layoutOf(op.getGathered()[0]));
    Value count = countOf(b, layout, index(b, d));
    SmallVector<Value> ins;
    for (Value field : op.getGathered())
      ins.push_back(makeField(
          b, set, cast<FieldType>(field.getType()).getNumComponents(),
          b.getF64Type(), index(b, d), {storage[field].buffer}, {count}));
    auto loop = md_exec::ParticleForOp::create(
        b, loc, TypeRange{type}, ins, ValueRange(),
        ValueRange{zero(b, type)}, /*scratch=*/ValueRange());
    Block *block = new Block();
    loop.getKernel().push_back(block);
    Block &source = op.getKernel().front();
    IRMapping mapping;
    for (BlockArgument argument : source.getArguments())
      mapping.map(argument, block->addArgument(argument.getType(), loc));
    OpBuilder kb = OpBuilder::atBlockEnd(block);
    for (Operation &nested : source.without_terminator())
      kb.clone(nested, mapping);
    md_exec::YieldOp::create(
        kb, loc,
        ValueRange{mapping.lookupOrDefault(
            cast<YieldOp>(source.getTerminator()).getValues()[0])});
    Value sum = loop.getResult(0);
    for (int64_t c = 0; c < length; ++c) {
      Value part =
          vector ? Value(vector::ExtractOp::create(b, loc, sum, c)) : sum;
      memref::StoreOp::create(b, loc, part, partial,
                              ValueRange{index(b, d), index(b, c)});
    }
  }
  storage[op.getResult()].buffer = partial;
  return success();
}

LogicalResult Lowering::lowerMapParticlesExec(MapParticlesOp op) {
  if (failed(checkOwnedRows(op, op.getGathered())))
    return failure();
  OpBuilder b(op);
  FieldType resultType = op.getResult().getType();
  int64_t k = resultType.getNumComponents();
  Storage &layout = rowsOf(op.getGathered()[0]);
  Value result = allocate(b, layout.capacity, k);
  for (int64_t d = 0; d < domains; ++d) {
    FlatSymbolRefAttr set =
        getSet(d, /*local=*/false, layoutOf(op.getGathered()[0]));
    Value count = countOf(b, layout, index(b, d));
    SmallVector<Value> ins;
    for (Value field : op.getGathered())
      ins.push_back(makeField(
          b, set, cast<FieldType>(field.getType()).getNumComponents(),
          b.getF64Type(), index(b, d), {storage[field].buffer}, {count}));
    auto outType =
        md::FieldType::get(module.getContext(), set, k, b.getF64Type());
    Value destination = md_exec::EmptyOp::create(b, loc, outType);
    auto loop = md_exec::ParticleForOp::create(
        b, loc, TypeRange{outType}, ins, ValueRange{destination},
        ValueRange(), /*scratch=*/ValueRange());
    Block *block = new Block();
    loop.getKernel().push_back(block);
    Block &source = op.getKernel().front();
    IRMapping mapping;
    for (BlockArgument argument : source.getArguments())
      mapping.map(argument, block->addArgument(argument.getType(), loc));
    OpBuilder kb = OpBuilder::atBlockEnd(block);
    for (Operation &nested : source.without_terminator())
      kb.clone(nested, mapping);
    md_exec::YieldOp::create(
        kb, loc,
        ValueRange{mapping.lookupOrDefault(
            cast<YieldOp>(source.getTerminator()).getValues()[0])});
    copyBack(b, loop.getResult(0), index(b, 0), count, result, index(b, d));
  }
  storage[op.getResult()].buffer = result;
  return success();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lower(Operation *op) {
  loc = op->getLoc();
  end = op->getBlock()->getTerminator();
  if (matrixPairs) {
    if (auto n = dyn_cast<NeighborhoodOp>(op)) {
      return lowerNeighborhoodExec(n);
    }

  }
  if (throughExec) {
    if (auto o = dyn_cast<NeighborhoodOp>(op))
      return lowerNeighborhoodExec(o);
    if (auto o = dyn_cast<SumPairsOp>(op))
      return lowerSumPairsExec(o);
    if (auto o = dyn_cast<GatherPairsOp>(op))
      return lowerGatherPairsExec(o);
    if (auto o = dyn_cast<SumParticlesOp>(op))
      return lowerSumParticlesExec(o);
    if (auto o = dyn_cast<MapParticlesOp>(op))
      return lowerMapParticlesExec(o);
  }
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](PartitionOp o) { return lowerPartition(o); })
      .Case([&](MigrateOp o) { return lowerMigrate(o); })
      .Case([&](DistributeOp o) { return lowerDistribute(o); })
      .Case([&](MeshOp o) { return lowerMesh(o); })
      .Case([&](CollectOp o) { return lowerCollect(o); })
      .Case([&](RedistributeOp o) { return lowerRedistribute(o); })
      .Case([&](HaloOp o) { return lowerHalo(o); })
      .Case([&](CoveredOp o) { return lowerCovered(o); })
      .Case([&](HaloCopyOp o) { return lowerHaloCopy(o); })
      .Case([&](HaloSumOp o) { return lowerHaloSum(o); })
      .Case([&](ZerosOp o) { return lowerZeros(o); })
      .Case([&](NeighborhoodOp o) { return lowerNeighborhood(o); })
      .Case([&](TupleKernelOp o) { return lowerTupleKernel(o); })
      .Case([&](TupleRowsOp o) { return lowerTupleRows(o); })
      .Case([&](PartitionCentersOp o) { return lowerPartitionCenters(o); })
      .Case([&](SumPairsOp o) { return lowerSumPairs(o); })
      .Case([&](GatherPairsOp o) { return lowerGatherPairs(o); })
      .Case([&](SumParticlesOp o) { return lowerSumParticles(o); })
      .Case([&](MapParticlesOp o) { return lowerMapParticles(o); })
      .Case([&](AllReduceOp o) { return lowerAllReduce(o); })
      .Case([&](PackOp o) { return lowerPack(o); })
      .Case([&](UnpackOp o) { return lowerUnpack(o); })
      .Default([&](Operation *other) {
        return other->emitOpError() << "is not run by the reference executor";
      });
}

/// Whether `type` is a type of md_dist, which the lowered program does not
/// have.
static bool isDistributedType(Type type) {
  return isa<LayoutType, HaloType, FieldType, PartialType, RelationType,
             StateType>(type);
}

/// The mark of a loop that carries a state, which the second executor
/// outlines.
static constexpr llvm::StringLiteral kStepLoop = "md_dist.step_loop";

/// Gives `scf.for`, `scf.if`, and `scf.while` that carry a md_dist.state its
/// tuple instead (D399): the layout and halo handles,
/// the reference positions, and one buffer per field. The ops of md_dist
/// keep the state, through unrealized casts from and to the tuple, which
/// the lowering of pack and unpack resolves.
static LogicalResult convertStateFlow(func::FuncOp function) {
  bool carries = false;
  function.walk([&](Operation *op) {
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op))
      for (Type type : op->getResultTypes())
        if (isa<StateType>(type)) {
          carries = true;
          if (isa<scf::ForOp>(op))
            op->setAttr(kStepLoop, UnitAttr::get(op->getContext()));
        }
  });
  if (!carries)
    return success();
  MLIRContext *context = function.getContext();
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  converter.addConversion(
      [context](StateType state, SmallVectorImpl<Type> &results) {
        Type i64 = IntegerType::get(context, 64);
        auto buffer = MemRefType::get({ShapedType::kDynamic,
                                       ShapedType::kDynamic,
                                       ShapedType::kDynamic},
                                      Float64Type::get(context));
        results.append({i64, i64, buffer});
        results.append(state.getFields().size(), buffer);
        return success();
      });
  converter.addSourceMaterialization(
      [](OpBuilder &b, Type type, ValueRange inputs, Location loc) -> Value {
        return UnrealizedConversionCastOp::create(b, loc, type, inputs)
            .getResult(0);
      });
  converter.addTargetMaterialization(
      [](OpBuilder &b, TypeRange types, ValueRange inputs, Location loc,
         Type) -> SmallVector<Value> {
        return SmallVector<Value>(
            UnrealizedConversionCastOp::create(b, loc, types, inputs)
                .getResults());
      });
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                       target);
  return applyPartialConversion(function, target, std::move(patterns));
}

/// Moves the body of each loop that carried a state into a function of its
/// own, called by the loop (D399). The second executor
/// needs it: a particle set of md_exec has one size in a function, and its
/// storage is assigned for the whole function, while the rows of a domain
/// change with every migration and rebuild of the halo. In the function of
/// a step, the sets of md_exec have the sizes of that step. A cell and the
/// constants the body reads are made again in the function, so that the
/// passes that need the op of a cell find it.
static LogicalResult outlineSteps(ModuleOp module, func::FuncOp function, StringRef marker) {
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(marker))
      loops.push_back(loop);
  });
  unsigned number = 0;
  for (scf::ForOp loop : loops) {
    loop->removeAttr(marker);
    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(loop.getRegion(), captured);
    // The values made again in the function, and those passed to it.
    SmallVector<Operation *> remade;
    SmallVector<Value> passed;
    llvm::DenseSet<Operation *> seen;
    std::function<void(Value)> visit = [&](Value value) {
      Operation *def = value.getDefiningOp();
      bool remake = def && (def->hasTrait<OpTrait::ConstantLike>() ||
                            isa<md::OrthorhombicCellOp>(def));
      if (!remake) {
        if (!llvm::is_contained(passed, value))
          passed.push_back(value);
        return;
      }
      if (!seen.insert(def).second)
        return;
      for (Value operand : def->getOperands())
        visit(operand);
      remade.push_back(def);
    };
    for (Value value : captured)
      visit(value);
    Block *body = loop.getBody();
    SmallVector<Type> inputs(body->getArgumentTypes());
    for (Value value : passed)
      inputs.push_back(value.getType());
    SmallVector<Type> results(loop.getResultTypes());
    OpBuilder b(function);
    std::string name =
        (function.getSymName() + "_step" + Twine(number++)).str();
    while (SymbolTable::lookupSymbolIn(module, name))
      name += "_";
    auto step = func::FuncOp::create(b, loop.getLoc(), name,
                                     b.getFunctionType(inputs, results));
    step.setPrivate();
    // A procedure of a domain is called at every step: the buffers it
    // allocates are freed at its end (md-dist-release-domain-buffers).
    if (marker == "md_dist.domain_loop")
      step->setAttr("md_dist.domain_procedure", b.getUnitAttr());
    Block *entry = step.addEntryBlock();
    IRMapping mapping;
    for (auto [argument, value] :
         llvm::zip(body->getArguments(),
                   entry->getArguments().take_front(body->getNumArguments())))
      mapping.map(argument, value);
    for (auto [value, argument] :
         llvm::zip(passed,
                   entry->getArguments().drop_front(body->getNumArguments())))
      mapping.map(value, argument);
    OpBuilder inside = OpBuilder::atBlockBegin(entry);
    for (Operation *op : remade)
      inside.clone(*op, mapping);
    for (Operation &op : body->without_terminator())
      inside.clone(op, mapping);
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    SmallVector<Value> returned;
    for (Value value : yield.getOperands())
      returned.push_back(mapping.lookupOrDefault(value));
    func::ReturnOp::create(inside, loop.getLoc(), returned);
    // The loop calls the step.
    SmallVector<Value> operands(body->getArguments());
    operands.append(passed.begin(), passed.end());
    SmallVector<Operation *> moved;
    for (Operation &op : body->without_terminator())
      moved.push_back(&op);
    for (Operation *op : llvm::reverse(moved)) {
      op->dropAllUses();
      op->erase();
    }
    OpBuilder call(yield);
    auto result = func::CallOp::create(call, loop.getLoc(), step, operands);
    yield->setOperands(result.getResults());
  }
  return success();
}

// Group tuples never read ghosts. Their members stay together on the
// initial spatial layout throughout the interval of unchanged topology.
// Coordinates still take their current primary-cell image at each solve.
static void retainConstraintLayouts(scf::ForOp root) {
  SmallVector<PartitionOp> partitions;
  root.walk([&](PartitionOp p) {
    if (p->hasAttr("md_dist.constraint_layout") &&
        root.isDefinedOutsideOfLoop(p.getCell()) &&
        llvm::all_of(p.getGroups(),[&](Value g) { return root.isDefinedOutsideOfLoop(g); }))
      partitions.push_back(p);
  });
  for (auto p : partitions) {
    SmallVector<MigrateOp> moves;
    for (Operation *u : p.getLayout().getUsers())
      if (auto move = dyn_cast<MigrateOp>(u)) moves.push_back(move);
    if (moves.empty()) continue;
    OpBuilder before(root);
    auto base = PartitionOp::create(before,p.getLoc(),p.getLayout().getType(),
        root.getInitArgs().front(),p.getCell(),p.getGroups());
    for (auto move : moves) {
      auto input = move.getPositions().getDefiningOp<DistributeOp>();
      if (!input || input.getLayout() != p.getLayout()) continue;
      SmallVector<RedistributeOp> fields;
      for (Operation *u : move.getLayout().getUsers())
        if (auto f = dyn_cast<RedistributeOp>(u)) fields.push_back(f);
      if (fields.empty() || !llvm::all_of(fields,[&](RedistributeOp f) {
            return f.getField() == input.getResult() && f.getOldLayout() == p.getLayout();
          })) continue;
      for (auto f : fields) {
        OpBuilder at(f);
        auto positions = DistributeOp::create(at,f.getLoc(),f.getResult().getType(),
            input.getField(),base.getLayout(),f.getImage());
        if (move->hasAttr("md_dist.step")) positions->setAttr("md_dist.step",at.getUnitAttr());
        f.replaceAllUsesWith(positions.getResult());
        f.erase();
      }
      move.replaceAllUsesWith(base.getLayout());
      move.erase();
      if (input.use_empty()) input.erase();
    }
    if (p.use_empty()) p.erase();
    if (base.use_empty()) base.erase();
  }
}

// The constrained native step carries ordinary particle fields in scf.for.
// Give its maps, kicks and drifts the contiguous identity interval on this
// rank. The separate particle-set symbol also gives storage assignment the
// local extent for loop-carried destinations and their copies.
void Lowering::shardDriverLoop(scf::ForOp root) {
  OpBuilder entry(root);
  loc = root.getLoc();
  Value initial = root.getInitArgs().front();
  auto original = cast<md::FieldType>(initial.getType());
  auto sourceSet = original.getParticleSet();
  std::string name = (decomposition.getSymName() + ".native").str();
  OpBuilder symbols(decomposition);
  auto symbol = module.lookupSymbol<md::ParticleSetOp>(name);
  if (!symbol) symbol = md::ParticleSetOp::create(symbols,loc,name);
  auto shardSet = FlatSymbolRefAttr::get(module.getContext(),name);
  auto retype = [&](Value field, FlatSymbolRefAttr set) {
    auto t = cast<md::FieldType>(field.getType());
    return md::FieldType::get(module.getContext(),set,t.getNumComponents(),t.getElementType());
  };
  auto inSet = [](Value field, FlatSymbolRefAttr set) {
    auto t = dyn_cast<md::FieldType>(field.getType());
    return t && t.getParticleSet() == set;
  };
  auto view = [&](OpBuilder &b, Value field) -> Value {
    auto op = mdrt::ToBufferOp::create(b,loc,mdrt::getBufferType(cast<md::FieldType>(field.getType())),field);
    op->setAttr("mdrt.partition_view",b.getUnitAttr());
    return op;
  };
  auto i64 = entry.getI64Type();
  Value extent = memref::DimOp::create(entry,loc,view(entry,initial),0);
  Value extent64 = arith::IndexCastOp::create(entry,loc,i64,extent);
  auto bound = [&](int side) {
    auto fn = declare("mdrtMPIIdentityBound",entry.getFunctionType({i64,i64},{i64}),false);
    Value side64 = arith::ConstantIntOp::create(entry,loc,side,64);
    return toIndex(entry,func::CallOp::create(entry,loc,fn,ValueRange{extent64,side64}).getResult(0));
  };
  Value first = bound(0);
  Value length = arith::SubIOp::create(entry,loc,bound(1),first);
  Value ids = memref::AllocOp::create(entry,loc,MemRefType::get({ShapedType::kDynamic},entry.getIndexType()),ValueRange{length});
  scf::ParallelOp::create(entry,loc,ValueRange{index(entry,0)},ValueRange{length},ValueRange{index(entry,1)},
      [&](OpBuilder &b, Location, ValueRange iv) {
        memref::StoreOp::create(b,loc,arith::AddIOp::create(b,loc,first,iv[0]),ids,iv);
      });
  DenseMap<Value,Value> immutable;
  auto compact = [&](OpBuilder &use, Value field) -> Value {
    if (!inSet(field,sourceSet)) return field;
    bool external = root.isDefinedOutsideOfLoop(field);
    if (external && immutable.count(field)) return immutable.lookup(field);
    OpBuilder b(external ? entry : use);
    auto copy = mdrt::SnapshotOp::create(b,loc,mdrt::getBufferType(cast<md::FieldType>(field.getType())),field,ids);
    Value result = mdrt::FromBufferOp::create(b,loc,retype(field,shardSet),copy);
    if (external) immutable[field] = result;
    return result;
  };
  auto expand = [&](OpBuilder &b, Value field) -> Value {
    auto t = retype(field,sourceSet);
    Value dst = memref::AllocOp::create(b,loc,mdrt::getBufferType(t),ValueRange{extent});
    auto fn = declare("mdrtMPIExpandField",b.getFunctionType({i64,i64,i64,i64,i64},{}),false);
    func::CallOp::create(b,loc,fn,ValueRange{bufferAddress(b,view(b,field)),extent64,bufferAddress(b,dst),
        arith::ConstantIntOp::create(b,loc,t.getNumComponents(),64),
        arith::ConstantIntOp::create(b,loc,t.getElementType().isInteger(32),64)});
    auto result = mdrt::FromBufferOp::create(b,loc,t,dst);
    result->setAttr("mdrt.collect_buffer",b.getUnitAttr());
    return result;
  };
  std::function<void(scf::ForOp)> visit = [&](scf::ForOp loop) {
    OpBuilder pre(loop);
    for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
      if (!inSet(arg,sourceSet)) continue;
      Value seed = compact(pre,loop.getInitArgs()[i]);
      loop.getInitArgsMutable()[i].set(seed);
      arg.setType(seed.getType());
      loop.getResult(i).setType(seed.getType());
    }
    DenseMap<Value,Value> expanded;
    for (Operation &op : llvm::make_early_inc_range(*loop.getBody())) {
      OpBuilder b(&op);
      if (auto nested = dyn_cast<scf::ForOp>(op)) { visit(nested); continue; }
      if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        for (auto [i, operand] : llvm::enumerate(yield->getOpOperands()))
          if (inSet(loop.getResult(i),shardSet)) operand.set(compact(b,operand.get()));
        continue;
      }
      if (auto collect = dyn_cast<CollectOp>(op)) {
        collect.getResult().setType(retype(collect.getResult(),shardSet));
        shardExtent[&op] = length;
        continue;
      }
      if (auto part = dyn_cast<PartitionOp>(op)) {
        if (inSet(part.getPositions(),shardSet)) shardExtent[&op] = extent;
        continue;
      }
      if (auto distribute = dyn_cast<DistributeOp>(op)) {
        if (inSet(distribute.getField(),shardSet)) shardExtent[&op] = length;
        continue;
      }
      bool map = isa<md::MapParticlesOp,mdir::dyn::KickOp,mdir::dyn::DriftOp>(op);
      if ((map || isa<md::SumParticlesOp>(op)) && llvm::any_of(op.getOperands(),
          [&](Value v) { return inSet(v,sourceSet) || inSet(v,shardSet); })) {
        for (OpOperand &operand : op.getOpOperands())
          operand.set(compact(b,operand.get()));
        if (map) op.getResult(0).setType(retype(op.getResult(0),shardSet));
        else shardExtent[&op] = length;
        continue;
      }
      // A host observation, permutation or other complete-field consumer
      // sees the original particle set at its actual point of use.
      for (OpOperand &operand : op.getOpOperands()) {
        Value shard = operand.get();
        if (!inSet(shard,shardSet)) continue;
        Value &full = expanded[shard];
        if (!full) full = expand(b,shard);
        operand.set(full);
      }
    }
  };
  visit(root);
  OpBuilder after(root->getBlock(),std::next(root->getIterator()));
  memref::DeallocOp::create(after,loc,ids);
  for (Value result : root.getResults()) {
    if (!inSet(result,shardSet) || result.use_empty()) continue;
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : result.getUses()) uses.push_back(&use);
    Value full = expand(after,result);
    for (OpOperand *use : uses) use->set(full);
  }
}

LogicalResult Lowering::run(func::FuncOp function) {
  // The words of the deterministic mode are those of the reference
  // executor; the executor through md_exec sums in binary64 only.
  if (fixedPoint && throughExec)
    return function.emitError(
        "deterministic fixed-point sums require the reference executor");
  SmallVector<Operation *> ops;
  auto collect = [&] {
    ops.clear();
    function.walk([&](Operation *op) {
      if (isa_and_nonnull<MDDistDialect>(op->getDialect()) &&
          !isa<YieldOp>(op) && !op->getParentOfType<SumPairsOp>() &&
          !op->getParentOfType<GatherPairsOp>() &&
          !op->getParentOfType<SumParticlesOp>() &&
          !op->getParentOfType<MapParticlesOp>())
        ops.push_back(op);
    });
  };
  collect();
  shardExtent.clear();
  auto lowerNativeSums = [&] {
    if (!module->hasAttr("md_dist.driver")) return;
    SmallVector<md::SumParticlesOp> sums;
    function.walk([&](md::SumParticlesOp sum) {
      if (fixedPoint || shardExtent.count(sum)) sums.push_back(sum);
    });
    for (auto sum : sums) lowerCollectedSum(sum);
  };
  if (ops.empty()) {
    lowerNativeSums();
    return success();
  }
  for (BlockArgument argument : function.getArguments())
    if (isDistributedType(argument.getType()))
      return function.emitOpError()
             << "the reference executor runs the values of md_dist within "
                "a function; an argument of md_dist cannot be run";
  // Control flow carries only states (D399); every other
  // value of md_dist stays in the region of its op and is used by ops of
  // md_dist.
  if (failed(verifyStateFlow(function)))
    return failure();
  for (Operation *op : ops)
    for (Value result : op->getResults())
      if (isDistributedType(result.getType()))
        for (Operation *user : result.getUsers()) {
          if (isa<StateType>(result.getType()) &&
              isa<scf::YieldOp, scf::ForOp>(user))
            continue;
          if (!isa_and_nonnull<MDDistDialect>(user->getDialect()))
            return user->emitOpError()
                   << "uses a value of md_dist, which the reference "
                      "executor cannot hand to an op of another dialect";
        }
  if (failed(convertStateFlow(function)))
    return failure();
  collect();
  // One decomposition per function.
  for (Operation *op : ops) {
    for (Type type : op->getResultTypes()) {
      FlatSymbolRefAttr name;
      if (auto layout = dyn_cast<LayoutType>(type))
        name = layout.getDecomposition();
      if (!name)
        continue;
      DecompositionOp found = getDecomposition(op, name);
      if (!found)
        return failure();
      if (decomposition && decomposition != found)
        return op->emitOpError() << "the reference executor runs one "
                                    "decomposition per function";
      decomposition = found;
    }
  }
  if (!decomposition)
    return ops.front()->emitOpError()
           << "has no layout from md_dist.partition to run";
  domains = decomposition.getNumDomains();
  if (mpi && module->hasAttr("md_dist.driver")) {
    SmallVector<scf::ForOp> roots;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
      if (!loop.getNumRegionIterArgs() || !isa<md::FieldType>(loop.getInitArgs().front().getType()) ||
          !loop.walk([](CollectOp) { return WalkResult::interrupt(); }).wasInterrupted())
        return WalkResult::advance();
      roots.push_back(loop);
      return WalkResult::skip();
    });
    for (auto root : roots) {
      if (root.getNumRegionIterArgs() == 3) retainConstraintLayouts(root);
      shardDriverLoop(root);
    }
    collect();
  }
  lowerNativeSums();
  if (throughExec)
    declareSets();
  bool statistics = module->hasAttr("md_dist.driver");
  auto mark = [&](Operation *before, int code) {
    OpBuilder b(before);
    auto type = b.getFunctionType({b.getI64Type()}, {});
    auto value = arith::ConstantIntOp::create(b, before->getLoc(), code, 64);
    func::CallOp::create(b, before->getLoc(),
                         declare("mdrtDistPhase", type, false),
                         ValueRange{value});
  };
  if (statistics) {
    SmallVector<std::pair<Operation *, int>> measured;
    function.walk([&](Operation *op) {
      if (auto call = dyn_cast<func::CallOp>(op)) {
        if (call.getCallee().starts_with("mdrtWrite"))
          measured.emplace_back(op, DistOutput);
      } else if (op->getName().getStringRef() == "dyn.kick" ||
                 op->getName().getStringRef() == "dyn.drift") {
        measured.emplace_back(op, DistIntegrate);
      }
    });
    for (auto [op, phase] : measured) {
      mark(op, phase);
      mark(op->getNextNode(), -phase);
    }
  }
  constraintKernels.clear();
  for (Operation *op : ops)
    if (auto rows = dyn_cast<TupleRowsOp>(op))
      if (rows.getGroup())
        for (Operation *use : rows->getUsers())
          if (isa<TupleKernelOp>(use)) constraintKernels.insert(use);
  planHaloGroups(ops);
  DominanceInfo dominance(function);
  for (Operation *op : ops) {
    if (grouped.contains(op)) continue;
    int phase = 0;
    if (statistics) {
      if (isa<SumPairsOp, GatherPairsOp>(op)) phase = DistPairs;
      if (isa<SumParticlesOp, MapParticlesOp, TupleRowsOp, TupleKernelOp>(op)) phase = DistOther;
      if (op->hasAttr("md_dist.integration")) phase = DistIntegrate;
      if (auto rows = dyn_cast<TupleRowsOp>(op))
        if (rows.getGroup()) phase = DistIntegrate;
      if (constraintKernels.contains(op)) phase = DistIntegrate;
      if (isa<PackOp, UnpackOp>(op)) phase = DistState;
      if (isa<HaloOp>(op)) phase = DistHaloBuild;
      if (isa<HaloCopyOp>(op)) phase = DistCopy;
      if (isa<HaloSumOp>(op)) phase = DistSum;
      if (isa<AllReduceOp>(op)) phase = DistReduce;
      if (isa<MigrateOp, RedistributeOp>(op)) phase = DistMigration;
      if (isa<CoveredOp>(op) || op->hasAttr("md_dist.step"))
        if (auto step = op->getParentOfType<scf::ForOp>())
          mark(&step.getBody()->front(), DistStep);
      if (phase) mark(op, phase);
    }
    if (matrixPairs && isa<SumPairsOp, GatherPairsOp>(op)) {
      loc = op->getLoc();
      end = op->getBlock()->getTerminator();
      if (failed(lowerPairGroup(op, dominance))) return failure();
    } else if (failed(lower(op))) {
      return failure();
    }
    if (phase) mark(op, -phase);
  }
  // The casts of the structural conversion: a cast of a state to its tuple
  // takes the tuple of the state; a cast of a tuple to a state goes with the
  // ops of the state.
  SmallVector<UnrealizedConversionCastOp> toTuples, toStates;
  function.walk([&](UnrealizedConversionCastOp cast) {
    if (cast.getNumOperands() == 1 &&
        isa<StateType>(cast.getOperand(0).getType()))
      toTuples.push_back(cast);
    else if (cast.getNumResults() == 1 &&
             isa<StateType>(cast.getResult(0).getType()))
      toStates.push_back(cast);
  });
  for (UnrealizedConversionCastOp cast : toTuples) {
    SmallVector<Value> tuple = stateOf(cast.getOperand(0));
    if (tuple.size() != cast.getNumResults())
      return cast.emitOpError() << "cannot find the tuple of the state";
    cast.replaceAllUsesWith(tuple);
    cast.erase();
  }
  for (Operation *op : llvm::reverse(ops)) {
    op->dropAllUses();
    op->erase();
  }
  for (UnrealizedConversionCastOp cast : toStates) {
    if (!cast.use_empty())
      return cast.emitOpError() << "a state is left after its lowering";
    cast.erase();
  }
  if (throughExec || matrixPairs) {
    if (failed(outlineSteps(module, function, matrixPairs ? "md_dist.domain_loop" : kStepLoop)))
      return failure();
  }
  if (!throughExec) {
    function.walk([](scf::ForOp loop) { loop->removeAttr(kStepLoop); });
  }
  // Distinct potential terms can produce separate reductions of the same
  // observable. Add their words before the final conversion to binary64.
  if (fixedPoint) {
    function.walk([&](arith::AddFOp op) {
      auto lhs = observableWords.find(op.getLhs());
      auto rhs = observableWords.find(op.getRhs());
      if (lhs == observableWords.end() || rhs == observableWords.end())
        return;
      auto [a, aFormat] = lhs->second;
      auto [b, bFormat] = rhs->second;
      OpBuilder at(op);
      // Bring the word of fewer fractional bits to the other's by checked
      // doublings, which are exact.
      for (int bits = aFormat.bits; bits < bFormat.bits; ++bits)
        a = add(at, a, a);
      for (int bits = bFormat.bits; bits < aFormat.bits; ++bits)
        b = add(at, b, b);
      Format format = aFormat.bits >= bFormat.bits ? aFormat : bFormat;
      op.replaceAllUsesWith(decodeObservable(at, add(at, a, b), format));
    });
  }
  observableWords.clear();
  fieldWords.clear();
  excludedPairs.clear();
  imageShifts.clear();
  storage.clear();
  grouped.clear();
  relations.clear();
  trajectoryCaches.clear();
  innerEpochs.clear();
  scopedExclusions.clear();
  sharedTuplePositions.clear();
  tupleSchedules.clear();
  sharedTupleRows.clear();
  topologyShards.clear();
  ownedEpochs.clear();
  localEpochs.clear();
  decomposition = nullptr;
  return success();
}

namespace {
class ConvertMDDistToLoops
    : public mdir::impl::ConvertMDDistToLoopsBase<ConvertMDDistToLoops> {
public:
  using ConvertMDDistToLoopsBase::ConvertMDDistToLoopsBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    Lowering lowering(module, false, matrixWidth, 3, haloEvents || mpi, mpi, parallelDomains);
    if (!(matrixSkin >= 0) || matrixWidth <= 0) {
      module.emitError("matrix-skin must be nonnegative and matrix-width positive");
      return signalPassFailure();
    }
    lowering.matrixSkin = matrixSkin;
    lowering.meshMixed = meshMixed;
    lowering.meshX = meshX;
    lowering.meshY = meshY;
    lowering.fixedPoint = fixedPoint;
    lowering.matrixPairs = matrixPairs && !fixedPoint;
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions)
      if (failed(lowering.run(function)))
        return signalPassFailure();
    // The declarations of md_dist are not needed once nothing refers to
    // them.
    for (auto op : llvm::make_early_inc_range(module.getOps<DecompositionOp>()))
      op.erase();
  }
};

class ConvertMDDistToMDExec
    : public mdir::impl::ConvertMDDistToMDExecBase<ConvertMDDistToMDExec> {
public:
  using ConvertMDDistToMDExecBase::ConvertMDDistToMDExecBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (width <= 0 || cells <= 0) {
      module.emitError() << "expected a positive width and number of cells";
      return signalPassFailure();
    }
    Lowering lowering(module, /*throughExec=*/true, width, cells, haloEvents);
    SmallVector<func::FuncOp> all(module.getOps<func::FuncOp>());
    bool left = false;
    for (func::FuncOp function : all) {
      if (!functions.empty() &&
          !llvm::is_contained(functions, function.getSymName())) {
        left |= function
                    .walk([](Operation *op) {
                      return isa_and_nonnull<MDDistDialect>(op->getDialect())
                                 ? WalkResult::interrupt()
                                 : WalkResult::advance();
                    })
                    .wasInterrupted();
        continue;
      }
      if (failed(lowering.run(function)))
        return signalPassFailure();
    }
    // The reference executor runs what is left of md_dist and needs the
    // decompositions.
    if (!left)
      for (auto op :
           llvm::make_early_inc_range(module.getOps<DecompositionOp>()))
        op.erase();
  }
};

/// The reference strategy completes transport at start. Keep the event
/// schedule visible until this separate pass, so a later transport can
/// replace the implementation without changing the dependency contract.
class LowerMDRTHaloEvents
    : public mdir::impl::LowerMDRTHaloEventsBase<LowerMDRTHaloEvents> {
public:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<Operation *> starts;
    module.walk([&](Operation *op) {
      if (isa<mdrt::HaloCopyStartOp, mdrt::HaloSumStartOp>(op))
        starts.push_back(op);
    });
    for (Operation *start : starts) {
      // A malformed event must never be translated into a transport call.
      if (failed(mlir::verify(start))) {
        signalPassFailure();
        return;
      }
      Value event, halo, source, destination;
      StringRef name;
      SmallVector<Value> operands;
      if (auto copy = dyn_cast<mdrt::HaloCopyStartOp>(start)) {
        event = copy.getEvent();
        halo = copy.getHalo();
        source = copy.getSource();
        destination = copy.getDestination();
        name = "mdrtDistHaloCopy";
        operands.append({halo, source, destination, copy.getImage()});
      } else {
        auto sum = cast<mdrt::HaloSumStartOp>(start);
        event = sum.getEvent();
        halo = sum.getHalo();
        source = sum.getSource();
        destination = sum.getDestination();
        name = "mdrtDistHaloSum";
        operands.append({halo, source, destination});
      }
      auto wait = cast<mdrt::HaloWaitOp>(*event.getUsers().begin());
      OpBuilder builder(start);
      SmallVector<Type> types;
      for (Value operand : operands)
        types.push_back(operand.getType());
      auto type = builder.getFunctionType(types, {});
      auto function = module.lookupSymbol<func::FuncOp>(name);
      if (!function) {
        function = func::FuncOp::create(start->getLoc(), name, type);
        function.setPrivate();
        function->setAttr("llvm.emit_c_interface",
                          UnitAttr::get(module.getContext()));
        module.push_back(function);
      } else if (function.getFunctionType() != type) {
        start->emitOpError() << "transport declaration has a different type";
        signalPassFailure();
        return;
      }
      func::CallOp::create(builder, start->getLoc(), function, operands);
      wait.getResult().replaceAllUsesWith(destination);
      wait.erase();
      start->erase();
    }
  }
};
} // namespace
