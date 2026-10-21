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
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mdir/Dialect/MDExec/Algebra.h"
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
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"

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
};

/// What the second executor keeps of a relation, for each domain: the
/// positions of its local rows, owned then ghost, as a field of md_exec,
/// its neighbor structure, and its number of owned rows, the centers.
struct DomainRelation {
  SmallVector<Value> positions, neighbors, centers;
  Value cell;
};

class Lowering {
public:
  /// With `throughExec`, the loops of each domain are those of md_exec
  /// (D290): a structure holds `width` neighbors of a row at first,
  /// in cells of 1/`cellsPerReach` of its reach.
  Lowering(ModuleOp module, bool throughExec = false, int64_t width = 64,
           int64_t cellsPerReach = 3, bool haloEvents = false)
      : module(module), loc(UnknownLoc::get(module.getContext())),
        throughExec(throughExec), width(width), cellsPerReach(cellsPerReach),
        haloEvents(haloEvents) {}

  LogicalResult run(func::FuncOp function);

private:
  LogicalResult lower(Operation *op);

  //===--------------------------------------------------------------------===//
  // The second executor (D290)
  //===--------------------------------------------------------------------===//

  LogicalResult lowerNeighborhoodExec(NeighborhoodOp op);
  LogicalResult lowerSumPairsExec(SumPairsOp op);
  LogicalResult lowerGatherPairsExec(GatherPairsOp op);
  LogicalResult lowerSumParticlesExec(SumParticlesOp op);
  LogicalResult lowerMapParticlesExec(MapParticlesOp op);
  /// Declares the particle sets of the owned and the local rows of each
  /// domain, once per module.
  void declareSets();
  /// The particle set of the owned or the local rows of domain `d`.
  FlatSymbolRefAttr getSet(int64_t d, bool local);
  /// The number of rows of domain `d` that the rows of `rows`, a layout or
  /// a halo, give it, as an index.
  Value countOf(OpBuilder &b, Storage &rows, int64_t d);
  /// A field of md_exec on `set` of `k` components of `element`, of the
  /// rows of domain `d` of each buffer of `parts`, one after another, with
  /// the number of rows of each in `counts`: a buffer of all domains
  /// (memref<D x rows x k>) or of the identities (memref<D x rows>).
  Value makeField(OpBuilder &b, FlatSymbolRefAttr set, int64_t k,
                  Type element, int64_t d, ArrayRef<Value> parts,
                  ArrayRef<Value> counts);
  /// Copies `count` rows of `field`, a field of md_exec, from row `first`
  /// on, to the rows of domain `d` of `buffer`, a buffer of all domains.
  void copyBack(OpBuilder &b, Value field, Value first, Value count,
                Value buffer, int64_t d);
  /// Gives `loop`, a loop over pairs of md_exec, the kernel of the pair op
  /// `op` of md_dist: the distance from its square, the kernel inlined, and
  /// its value, for a unique relation, where the identity of the center is
  /// the lesser, and multiplied by `scale`.
  void addPairKernel(md_exec::PairForOp loop, Operation *op, bool unique,
                     double scale);
  /// The loop over the pairs of domain `d` of the pair op `op` (a
  /// `sum_pairs` or a `gather_pairs`), with the destinations `outs` and the
  /// sums `reduce`, and the policy `(directed, atomic)` for `atomic`.
  md_exec::PairForOp createPairLoop(OpBuilder &b, Operation *op, int64_t d,
                                    TypeRange results, ValueRange outs,
                                    ValueRange reduce, DenseF64ArrayAttr weights,
                                    bool atomic, double scale);
  LogicalResult lowerPartition(PartitionOp op);
  LogicalResult lowerMigrate(MigrateOp op);
  LogicalResult lowerDistribute(DistributeOp op);
  LogicalResult lowerCollect(CollectOp op);
  LogicalResult lowerRedistribute(RedistributeOp op);
  LogicalResult lowerHalo(HaloOp op);
  LogicalResult lowerHaloCopy(HaloCopyOp op);
  LogicalResult lowerHaloSum(HaloSumOp op);
  LogicalResult lowerZeros(ZerosOp op);
  LogicalResult lowerNeighborhood(NeighborhoodOp op);
  LogicalResult lowerPartitionCenters(PartitionCentersOp op);
  LogicalResult lowerSumPairs(SumPairsOp op);
  LogicalResult lowerGatherPairs(GatherPairsOp op);
  LogicalResult lowerSumParticles(SumParticlesOp op);
  LogicalResult lowerMapParticles(MapParticlesOp op);
  LogicalResult lowerAllReduce(AllReduceOp op);

  /// The decomposition of a type of md_dist, and its number of domains.
  DecompositionOp getDecomposition(Operation *op, FlatSymbolRefAttr name);
  /// The edge lengths of the cell `cell`, which must come from
  /// md.orthorhombic_cell.
  LogicalResult getEdges(Operation *op, Value cell, Value edges[3]);

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
  /// A new buffer of `domains x rows x k` of f64, freed at the end.
  Value allocate(OpBuilder &b, Value rows, int64_t k);
  /// A new buffer of the given type and sizes, freed at the end.
  Value allocate(OpBuilder &b, MemRefType type, ValueRange sizes);
  void fillZero(OpBuilder &b, Value buffer);
  /// Loads the value of row `row` of domain `domain` of `buffer`: an f64,
  /// or a vector<3xf64> for three components.
  Value loadValue(OpBuilder &b, Value buffer, Value domain, Value row,
                  Type type);
  void storeValue(OpBuilder &b, Value value, Value buffer, Value domain,
                  Value row);
  void addValue(OpBuilder &b, Value value, Value buffer, Value domain,
                Value row);
  /// Inlines `kernel` with `arguments` and returns the value it yields.
  Value inlineKernel(OpBuilder &b, Block &kernel, ValueRange arguments);

  /// The layout or halo storage of the rows of a field.
  Storage &rowsOf(Value field);

  ModuleOp module;
  Location loc;
  llvm::DenseMap<Value, Storage> storage;
  /// Where the buffers are freed and the handles released: before the
  /// terminator of the block of the ops.
  Operation *end = nullptr;
  int64_t domains = 0;
  DecompositionOp decomposition;

  bool throughExec;
  int64_t width, cellsPerReach;
  bool haloEvents;
  llvm::DenseMap<Value, DomainRelation> relations;
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

Value Lowering::zero(OpBuilder &b, Type type) {
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

void Lowering::fillZero(OpBuilder &b, Value buffer) {
  auto type = cast<MemRefType>(buffer.getType());
  SmallVector<Value> sizes;
  for (int64_t i = 0; i < type.getRank(); ++i)
    sizes.push_back(memref::DimOp::create(b, loc, buffer, i));
  Value zeroValue = real(b, 0.0);
  std::function<void(OpBuilder &, SmallVector<Value> &)> nest =
      [&](OpBuilder &at, SmallVector<Value> &indices) {
        if (indices.size() == sizes.size()) {
          memref::StoreOp::create(at, loc, zeroValue, buffer, indices);
          return;
        }
        scf::ForOp::create(at, loc, index(at, 0), sizes[indices.size()],
                           index(at, 1), ValueRange(),
                           [&](OpBuilder &body, Location, Value i,
                               ValueRange) {
                             indices.push_back(i);
                             nest(body, indices);
                             indices.pop_back();
                             scf::YieldOp::create(body, loc);
                           });
      };
  SmallVector<Value> indices;
  nest(b, indices);
}

Value Lowering::loadValue(OpBuilder &b, Value buffer, Value domain, Value row,
                          Type type) {
  auto vector = dyn_cast<VectorType>(type);
  int64_t k = vector ? vector.getNumElements() : 1;
  SmallVector<Value> parts;
  for (int64_t c = 0; c < k; ++c)
    parts.push_back(memref::LoadOp::create(
        b, loc, buffer, ValueRange{domain, row, index(b, c)}));
  if (!vector)
    return parts[0];
  return vector::FromElementsOp::create(b, loc, vector, parts);
}

void Lowering::storeValue(OpBuilder &b, Value value, Value buffer,
                          Value domain, Value row) {
  auto vector = dyn_cast<VectorType>(value.getType());
  int64_t k = vector ? vector.getNumElements() : 1;
  for (int64_t c = 0; c < k; ++c) {
    Value part =
        vector ? Value(vector::ExtractOp::create(b, loc, value, c)) : value;
    memref::StoreOp::create(b, loc, part, buffer,
                            ValueRange{domain, row, index(b, c)});
  }
}

void Lowering::addValue(OpBuilder &b, Value value, Value buffer, Value domain,
                        Value row) {
  Value old = loadValue(b, buffer, domain, row, value.getType());
  storeValue(b, arith::AddFOp::create(b, loc, old, value), buffer, domain,
             row);
}

Value Lowering::inlineKernel(OpBuilder &b, Block &kernel,
                             ValueRange arguments) {
  IRMapping mapping;
  mapping.map(kernel.getArguments(), arguments);
  for (Operation &op : kernel.without_terminator())
    b.clone(op, mapping);
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

LogicalResult Lowering::lowerPartition(PartitionOp op) {
  OpBuilder b(op);
  Value edges[3];
  if (failed(getEdges(op, op.getCell(), edges)))
    return failure();
  auto fieldType = cast<md::FieldType>(op.getPositions().getType());
  if (!fieldType.getElementType().isF64())
    return op.emitOpError() << "the reference executor runs f64 only";
  Value positions = mdrt::ToBufferOp::create(
      b, loc, mdrt::getBufferType(fieldType), op.getPositions());
  auto wide = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                              b.getF64Type());
  Value cast = memref::CastOp::create(b, loc, wide, positions);
  ArrayRef<int64_t> grid = decomposition.getGrid();
  Type i64 = b.getI64Type(), f64 = b.getF64Type();
  func::FuncOp partition = declare(
      "mdrtDistPartition",
      b.getFunctionType({wide, i64, i64, i64, f64, f64, f64}, {i64}), true);
  Value handle =
      func::CallOp::create(
          b, loc, partition,
          ValueRange{cast,
                     arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(grid[0])),
                     arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(grid[1])),
                     arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(grid[2])),
                     edges[0], edges[1], edges[2]})
          .getResult(0);
  Value count = memref::DimOp::create(b, loc, positions, 0);
  Value capacity = arith::MaxUIOp::create(b, loc, count, index(b, 1));
  auto rowsType = MemRefType::get({ShapedType::kDynamic}, i64);
  auto idsType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, i64);
  Value rows = allocate(b, rowsType, {index(b, domains)});
  Value ids = allocate(b, idsType, {index(b, domains), capacity});
  func::FuncOp fill = declare(
      "mdrtDistLayoutRows", b.getFunctionType({i64, rowsType, idsType}, {}),
      true);
  func::CallOp::create(b, loc, fill, ValueRange{handle, rows, ids});
  OpBuilder atEnd(end);
  func::CallOp::create(
      atEnd, loc,
      declare("mdrtDistReleaseLayout", b.getFunctionType({i64}, {}), false),
      ValueRange{handle});
  storage[op.getLayout()] = {handle, rows, ids, capacity, count, Value()};
  return success();
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
  func::FuncOp migrate = declare(
      "mdrtDistMigrate",
      b.getFunctionType({i64, positions.getType(), f64, f64, f64}, {i64}),
      true);
  Value handle = func::CallOp::create(
                     b, loc, migrate,
                     ValueRange{old.handle, positions, edges[0], edges[1],
                                edges[2]})
                     .getResult(0);
  auto rowsType = MemRefType::get({ShapedType::kDynamic}, i64);
  auto idsType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, i64);
  Value rows = allocate(b, rowsType, {index(b, domains)});
  Value ids = allocate(b, idsType, {index(b, domains), old.capacity});
  func::CallOp::create(
      b, loc,
      declare("mdrtDistLayoutRows", b.getFunctionType({i64, rowsType, idsType},
                                                       {}), true),
      ValueRange{handle, rows, ids});
  OpBuilder atEnd(end);
  func::CallOp::create(
      atEnd, loc,
      declare("mdrtDistReleaseLayout", b.getFunctionType({i64}, {}), false),
      ValueRange{handle});
  storage[op.getLayout()] = {handle, rows, ids, old.capacity, old.count,
                             Value()};
  return success();
}

LogicalResult Lowering::lowerDistribute(DistributeOp op) {
  OpBuilder b(op);
  auto fieldType = cast<md::FieldType>(op.getField().getType());
  if (!fieldType.getElementType().isF64())
    return op.emitOpError() << "the reference executor runs f64 only";
  Value edges[3];
  if (op.getImage() && failed(getEdges(op, op.getImage(), edges)))
    return failure();
  Storage &layout = storage[op.getLayout()];
  int64_t k = fieldType.getNumComponents();
  Value global = mdrt::ToBufferOp::create(
      b, loc, mdrt::getBufferType(fieldType), op.getField());
  Value owned = allocate(b, layout.capacity, k);
  // owned[d][r] = field[ids[d][r]], wrapped into the cell with `image`.
  scf::ForOp::create(
      b, loc, index(b, 0), index(b, domains), index(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value count = toIndex(
            bd, memref::LoadOp::create(bd, loc, layout.rows, ValueRange{d}));
        scf::ForOp::create(
            bd, loc, index(bd, 0), count, index(bd, 1), ValueRange(),
            [&](OpBuilder &br, Location, Value r, ValueRange) {
              Value j = toIndex(br, memref::LoadOp::create(
                                        br, loc, layout.ids, ValueRange{d, r}));
              for (int64_t c = 0; c < k; ++c) {
                Value v =
                    k == 1 ? Value(memref::LoadOp::create(br, loc, global,
                                                          ValueRange{j}))
                           : Value(memref::LoadOp::create(
                                 br, loc, global, ValueRange{j, index(br, c)}));
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
              scf::YieldOp::create(br, loc);
            });
        scf::YieldOp::create(bd, loc);
      });
  storage[op.getResult()].buffer = owned;
  return success();
}

LogicalResult Lowering::lowerCollect(CollectOp op) {
  OpBuilder b(op);
  auto fieldType = cast<md::FieldType>(op.getResult().getType());
  Storage &layout = storage[op.getLayout()];
  Value owned = storage[op.getField()].buffer;
  int64_t k = fieldType.getNumComponents();
  MemRefType type = mdrt::getBufferType(fieldType);
  // The buffer is handed to the program as a field, which owns it from
  // then on; it is not freed here.
  Value global =
      memref::AllocOp::create(b, loc, type, ValueRange{layout.count});
  scf::ForOp::create(
      b, loc, index(b, 0), index(b, domains), index(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value rows = toIndex(
            bd, memref::LoadOp::create(bd, loc, layout.rows, ValueRange{d}));
        scf::ForOp::create(
            bd, loc, index(bd, 0), rows, index(bd, 1), ValueRange(),
            [&](OpBuilder &br, Location, Value r, ValueRange) {
              Value j = toIndex(br, memref::LoadOp::create(
                                        br, loc, layout.ids, ValueRange{d, r}));
              for (int64_t c = 0; c < k; ++c) {
                Value v = memref::LoadOp::create(
                    br, loc, owned, ValueRange{d, r, index(br, c)});
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
  Value field = mdrt::FromBufferOp::create(b, loc, fieldType, global);
  op.getResult().replaceAllUsesWith(field);
  return success();
}

LogicalResult Lowering::lowerRedistribute(RedistributeOp op) {
  OpBuilder b(op);
  if (!op.getField().getType().getElementType().isF64())
    return op.emitOpError() << "the in-process executor runs f64 fields only";
  Storage &old = storage[op.getOldLayout()];
  Storage &next = storage[op.getNewLayout()];
  Value source = storage[op.getField()].buffer;
  int64_t components = op.getField().getType().getNumComponents();
  Value destination = allocate(b, next.capacity, components);
  Type i64 = b.getI64Type();
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
  Storage &layout = storage[op.getLayout()];
  Value positions = storage[op.getPositions()].buffer;
  Type i64 = b.getI64Type(), f64 = b.getF64Type();
  auto bufferType = cast<MemRefType>(positions.getType());
  func::FuncOp build = declare(
      "mdrtDistHalo", b.getFunctionType({i64, bufferType, f64}, {i64}), true);
  double reach = op.getResult().getType().getReach().convertToDouble();
  Value handle = func::CallOp::create(
                     b, loc, build,
                     ValueRange{layout.handle, positions, real(b, reach)})
                     .getResult(0);
  Value capacity = toIndex(
      b, func::CallOp::create(
             b, loc,
             declare("mdrtDistHaloCapacity", b.getFunctionType({i64}, {i64}),
                     false),
             ValueRange{handle})
             .getResult(0));
  auto rowsType = MemRefType::get({ShapedType::kDynamic}, i64);
  auto idsType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, i64);
  Value rows = allocate(b, rowsType, {index(b, domains)});
  Value ids = allocate(b, idsType, {index(b, domains), capacity});
  func::CallOp::create(
      b, loc,
      declare("mdrtDistHaloRows",
              b.getFunctionType({i64, rowsType, idsType}, {}), true),
      ValueRange{handle, rows, ids});
  OpBuilder atEnd(end);
  func::CallOp::create(
      atEnd, loc,
      declare("mdrtDistReleaseHalo", b.getFunctionType({i64}, {}), false),
      ValueRange{handle});
  storage[op.getResult()] = {handle, rows, ids, capacity, Value(), Value()};
  return success();
}

LogicalResult Lowering::lowerHaloCopy(HaloCopyOp op) {
  OpBuilder b(op);
  Storage &halo = storage[op.getHalo()];
  Value owned = storage[op.getField()].buffer;
  int64_t k = op.getResult().getType().getNumComponents();
  Value ghosts = allocate(b, halo.capacity, k);
  Type i64 = b.getI64Type();
  Value image = arith::ConstantOp::create(
      b, loc, b.getI64IntegerAttr(op.getImage() ? 1 : 0));
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
  memref::CopyOp::create(b, loc, owned, result);
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
  fillZero(b, buffer);
  storage[op.getResult()].buffer = buffer;
  return success();
}

LogicalResult Lowering::lowerNeighborhood(NeighborhoodOp op) {
  // The relation is found where it is traversed; here only the precondition
  // under which it is the relation of md.neighborhood is checked.
  auto halo = op.getHalo().getDefiningOp<HaloOp>();
  if (!halo)
    return op.emitOpError() << "cannot find the cell of the halo";
  Value edges[3];
  if (failed(getEdges(op, halo.getCell(), edges)))
    return failure();
  OpBuilder b(op);
  Type f64 = b.getF64Type();
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
      b, loc, c(b, 0), c(b, domains), c(b, 1), ValueRange(),
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
static void emitPairs(
    OpTy op, Location loc, int64_t domains,
    llvm::DenseMap<Value, Storage> &storage, Type accumulator,
    function_ref<Value(OpBuilder &, Type)> zero,
    function_ref<Value(OpBuilder &, Value, Value, Value, Type)> load,
    function_ref<Value(OpBuilder &, Block &, ValueRange)> kernel,
    function_ref<Value(OpBuilder &, Value value, Value acc, Value d, Value k,
                       Value isOwned, Value ghostRow)>
        visit,
    function_ref<void(OpBuilder &, Value acc, Value d, Value i)> finishCenter,
    function_ref<void(OpBuilder &, Value acc, Value d)> finishDomain) {
  auto neighborhood = op.getRelation().template getDefiningOp<NeighborhoodOp>();
  bool unique =
      op.getRelation().getType().getTraversal() == Traversal::Unique;
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
  scf::ForOp::create(
      b, loc, c(b, 0), c(b, domains), c(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value owned = arith::IndexCastOp::create(
            bd, loc, bd.getIndexType(),
            memref::LoadOp::create(bd, loc, layout.rows, ValueRange{d}));
        Value ghosts = arith::IndexCastOp::create(
            bd, loc, bd.getIndexType(),
            memref::LoadOp::create(bd, loc, halo.rows, ValueRange{d}));
        Value local = arith::AddIOp::create(bd, loc, owned, ghosts);
        auto centers = scf::ForOp::create(
            bd, loc, c(bd, 0), owned, c(bd, 1),
            ValueRange{zero(bd, accumulator)},
            [&](OpBuilder &bi, Location, Value i, ValueRange domainSum) {
              Value rowEnd = ghostFree ? owned : local;
              if (centerMask) {
                Value bit = memref::LoadOp::create(
                    bi, loc, centerMask, ValueRange{d, i, c(bi, 0)});
                Value active = arith::CmpIOp::create(
                    bi, loc, arith::CmpIPredicate::ne, bit,
                    arith::ConstantIntOp::create(bi, loc, 0, 32));
                rowEnd = arith::SelectOp::create(bi, loc, active, rowEnd,
                                                c(bi, 0));
              }
              Value xi = load(bi, xo, d, i, vec3);
              SmallVector<Value> gi;
              for (unsigned g = 0; g < ownedFields.size(); ++g)
                gi.push_back(load(bi, ownedFields[g], d, i, types[g]));
              Value idi = unique ? Value(memref::LoadOp::create(
                                       bi, loc, layout.ids, ValueRange{d, i}))
                                 : Value();
              auto rows = scf::ForOp::create(
                  bi, loc, c(bi, 0), rowEnd, c(bi, 1),
                  ValueRange{zero(bi, accumulator)},
                  [&](OpBuilder &bk, Location, Value k, ValueRange acc) {
                    Value isOwned = arith::CmpIOp::create(
                        bk, loc, arith::CmpIPredicate::ult, k, owned);
                    Value shifted = arith::SubIOp::create(bk, loc, k, owned);
                    Value ghostRow =
                        arith::SelectOp::create(bk, loc, isOwned, c(bk, 0),
                                                shifted);
                    // The values of row k, from the owned or the ghost rows.
                    SmallVector<Type> sideTypes = {vec3};
                    sideTypes.append(types.begin(), types.end());
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
                      if (unique)
                        values.push_back(memref::LoadOp::create(
                            e, loc, halo.ids, ValueRange{d, ghostRow}));
                      scf::YieldOp::create(e, loc, values);
                    }
                    Value xk = side.getResult(0);
                    // d = x_i − x_k: ghosts carry their images, so no
                    // minimum image is taken.
                    Value dvec = arith::SubFOp::create(bk, loc, xi, xk);
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
              Value sum = rows.getResult(0);
              finishCenter(bi, sum, d, i);
              Value next =
                  accumulator.isF64() || isa<VectorType>(accumulator)
                      ? Value(arith::AddFOp::create(bi, loc, domainSum[0], sum))
                      : domainSum[0];
              scf::YieldOp::create(bi, loc, ValueRange{next});
            });
        finishDomain(bd, centers.getResult(0), d);
        scf::YieldOp::create(bd, loc);
      });
}

LogicalResult Lowering::lowerSumPairs(SumPairsOp op) {
  OpBuilder b(op);
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
  double weight = directed ? 0.5 : 1.0;
  emitPairs(
      op, loc, domains, storage, type,
      [&](OpBuilder &at, Type t) { return zero(at, t); },
      [&](OpBuilder &at, Value buffer, Value d, Value row, Type t) {
        return loadValue(at, buffer, d, row, t);
      },
      [&](OpBuilder &at, Block &kernel, ValueRange arguments) {
        return inlineKernel(at, kernel, arguments);
      },
      [&](OpBuilder &at, Value value, Value acc, Value, Value, Value, Value) {
        return Value(arith::AddFOp::create(at, loc, acc, value));
      },
      [&](OpBuilder &, Value, Value, Value) {},
      [&](OpBuilder &at, Value sum, Value d) {
        Value w = real(at, weight);
        if (vector)
          w = vector::BroadcastOp::create(at, loc, vector, w);
        Value scaled = arith::MulFOp::create(at, loc, sum, w);
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
  Value ownedOut = allocate(b, layout.capacity, k);
  Value ghostOut;
  if (twoEnded) {
    fillZero(b, ownedOut);
    ghostOut = allocate(b, halo.capacity, k);
    fillZero(b, ghostOut);
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
      op, loc, domains, storage, type,
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
          Value other = scale(at, value, sign * weight);
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
        return Value(arith::AddFOp::create(at, loc, acc, value));
      },
      [&](OpBuilder &at, Value sum, Value d, Value i) {
        if (twoEnded)
          addValue(at, scale(at, sum, weight), ownedOut, d, i);
        else
          storeValue(at, sum, ownedOut, d, i);
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
      {ShapedType::kDynamic, ShapedType::kDynamic}, b.getF64Type());
  Value partial =
      allocate(b, partialType, {index(b, domains), index(b, length)});
  Storage &rows = rowsOf(op.getGathered()[0]);
  scf::ForOp::create(
      b, loc, index(b, 0), index(b, domains), index(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value count = toIndex(
            bd, memref::LoadOp::create(bd, loc, rows.rows, ValueRange{d}));
        auto loop = scf::ForOp::create(
            bd, loc, index(bd, 0), count, index(bd, 1),
            ValueRange{zero(bd, type)},
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
                  ValueRange{arith::AddFOp::create(br, loc, acc[0], value)});
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

LogicalResult Lowering::lowerMapParticles(MapParticlesOp op) {
  OpBuilder b(op);
  FieldType resultType = op.getResult().getType();
  Storage &rows = rowsOf(op.getGathered()[0]);
  Value result = allocate(b, rows.capacity, resultType.getNumComponents());
  scf::ForOp::create(
      b, loc, index(b, 0), index(b, domains), index(b, 1), ValueRange(),
      [&](OpBuilder &bd, Location, Value d, ValueRange) {
        Value count = toIndex(
            bd, memref::LoadOp::create(bd, loc, rows.rows, ValueRange{d}));
        scf::ForOp::create(
            bd, loc, index(bd, 0), count, index(bd, 1), ValueRange(),
            [&](OpBuilder &br, Location, Value r, ValueRange) {
              SmallVector<Value> arguments;
              for (Value field : op.getGathered())
                arguments.push_back(loadValue(
                    br, storage[field].buffer, d, r,
                    cast<FieldType>(field.getType()).getKernelValueType()));
              Value value =
                  inlineKernel(br, op.getKernel().front(), arguments);
              storeValue(br, value, result, d, r);
              scf::YieldOp::create(br, loc);
            });
        scf::YieldOp::create(bd, loc);
      });
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
  // The partial sums are added in the order of the domains.
  auto loop = scf::ForOp::create(
      b, loc, index(b, 0), index(b, domains), index(b, 1),
      ValueRange{zero(b, type)},
      [&](OpBuilder &bd, Location, Value d, ValueRange acc) {
        SmallVector<Value> parts;
        for (int64_t c = 0; c < length; ++c)
          parts.push_back(memref::LoadOp::create(
              bd, loc, partial, ValueRange{d, index(bd, c)}));
        Value value =
            vector ? Value(vector::FromElementsOp::create(bd, loc, vector,
                                                          parts))
                   : parts[0];
        scf::YieldOp::create(
            bd, loc, ValueRange{arith::AddFOp::create(bd, loc, acc[0], value)});
      });
  op.getResult().replaceAllUsesWith(loop.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// The second executor: each domain through md_exec (D290)
//===----------------------------------------------------------------------===//

FlatSymbolRefAttr Lowering::getSet(int64_t d, bool local) {
  std::string name = (decomposition.getSymName() + ".d" + Twine(d) +
                      (local ? ".local" : ".owned"))
                         .str();
  return FlatSymbolRefAttr::get(module.getContext(), name);
}

void Lowering::declareSets() {
  OpBuilder b(decomposition);
  for (int64_t d = 0; d < domains; ++d)
    for (bool local : {false, true}) {
      FlatSymbolRefAttr set = getSet(d, local);
      if (!SymbolTable::lookupSymbolIn(module, set.getAttr()))
        md::ParticleSetOp::create(b, decomposition.getLoc(), set.getAttr());
    }
}

Value Lowering::countOf(OpBuilder &b, Storage &rows, int64_t d) {
  return toIndex(
      b, memref::LoadOp::create(b, loc, rows.rows, ValueRange{index(b, d)}));
}

Value Lowering::makeField(OpBuilder &b, FlatSymbolRefAttr set, int64_t k,
                          Type element, int64_t d, ArrayRef<Value> parts,
                          ArrayRef<Value> counts) {
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
            SmallVector<Value, 3> from = {index(br, d), r};
            if (partType.getRank() == 3)
              from.push_back(index(br, c));
            Value v = memref::LoadOp::create(br, loc, part, from);
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
                        Value buffer, int64_t d) {
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
                                  ValueRange{index(br, d), r, index(br, c)});
        }
        scf::YieldOp::create(br, loc);
      });
}

LogicalResult Lowering::lowerNeighborhoodExec(NeighborhoodOp op) {
  // The precondition under which the relation is that of md.neighborhood,
  // as in the reference executor.
  if (failed(lowerNeighborhood(op)))
    return failure();
  auto haloOp = op.getHalo().getDefiningOp<HaloOp>();
  OpBuilder b(op);
  Storage &halo = storage[op.getHalo()];
  Storage &layout = storage[layoutOf(op.getHalo())];
  Value owned = storage[op.getPositions()].buffer;
  Value ghosts = storage[op.getGhosts()].buffer;
  double cutoff = op.getCutoff().convertToDouble();
  // The structure is built at every evaluation, with the halo: no skin.
  double skin = 0.0;
  MLIRContext *context = module.getContext();
  DomainRelation relation;
  relation.cell = haloOp.getCell();
  for (int64_t d = 0; d < domains; ++d) {
    FlatSymbolRefAttr set = getSet(d, /*local=*/true);
    Value nOwned = countOf(b, layout, d);
    Value nGhosts = countOf(b, halo, d);
    // The local rows, owned then ghost; the ghosts carry their images.
    Value positions = makeField(b, set, 3, b.getF64Type(), d,
                                {owned, ghosts}, {nOwned, nGhosts});
    Value cells = md_exec::BuildCellsOp::create(
        b, loc, mdrt::CellsType::get(context, set), positions,
        relation.cell,
        APFloat((cutoff + skin) / static_cast<double>(cellsPerReach)));
    Value structure = md_exec::BuildNeighborsOp::create(
        b, loc, mdrt::NeighborsType::get(context, set), cells, positions,
        relation.cell, /*excluded=*/Value(), nOwned, APFloat(cutoff),
        APFloat(skin), md_exec::NeighborKind::Matrix,
        static_cast<uint64_t>(width), md_exec::Images::Explicit);
    relation.positions.push_back(positions);
    relation.neighbors.push_back(structure);
    relation.centers.push_back(nOwned);
  }
  relations[op.getResult()] = relation;
  return success();
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
Lowering::createPairLoop(OpBuilder &b, Operation *op, int64_t d,
                         TypeRange results, ValueRange outs, ValueRange reduce,
                         DenseF64ArrayAttr weights, bool atomic,
                         double scale) {
  Value relationValue = op->getOperand(0);
  DomainRelation &relation = relations[relationValue];
  auto relationType = cast<RelationType>(relationValue.getType());
  bool unique = relationType.getTraversal() == Traversal::Unique;
  auto neighborhood = relationValue.getDefiningOp<NeighborhoodOp>();
  Storage &halo = storage[neighborhood.getHalo()];
  Storage &layout = storage[layoutOf(neighborhood.getHalo())];
  FlatSymbolRefAttr set = getSet(d, /*local=*/true);
  Value nOwned = relation.centers[d];
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
    ins.push_back(makeField(b, set, type.getNumComponents(), b.getF64Type(),
                            d,
                            {storage[gathered[g]].buffer,
                             storage[gathered[g + 1]].buffer},
                            {nOwned, nGhosts}));
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
      b, loc, results, relation.neighbors[d], relation.positions[d],
      relation.cell, ins, outs, reduce, /*scratch=*/ValueRange(),
      b.getF64FloatAttr(cutoff), weights, /*overwrite=*/DenseBoolArrayAttr(),
      md_exec::TraversalAttr::get(context, md_exec::Traversal::Directed),
      md_exec::ConflictAttr::get(context, atomic
                                              ? md_exec::Conflict::Atomic
                                              : md_exec::Conflict::OwnerOnly),
      exchangeAttr, nOwned, centerSet, b.getBoolAttr(ghostFree),
      md_exec::ImagesAttr::get(context, md_exec::Images::Explicit),
      /*scatter=*/md_exec::ScatterAttr());
  addPairKernel(loop, op, unique, scale);
  return loop;
}

LogicalResult Lowering::lowerSumPairsExec(SumPairsOp op) {
  OpBuilder b(op);
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
  for (int64_t d = 0; d < domains; ++d) {
    Value init = zero(b, type);
    auto loop = createPairLoop(b, op, d, TypeRange{type}, ValueRange(),
                               ValueRange{init}, weights, /*atomic=*/false,
                               /*scale=*/1.0);
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

LogicalResult Lowering::lowerGatherPairsExec(GatherPairsOp op) {
  OpBuilder b(op);
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
  for (int64_t d = 0; d < domains; ++d) {
    auto localType = md::FieldType::get(module.getContext(),
                                        getSet(d, /*local=*/true), k,
                                        b.getF64Type());
    Value destination = md_exec::ZerosOp::create(b, loc, localType);
    auto loop = createPairLoop(b, op, d, TypeRange{localType},
                               ValueRange{destination}, ValueRange(),
                               DenseF64ArrayAttr(), /*atomic=*/twoEnded,
                               scale);
    Value nOwned = relations[op.getRelation()].centers[d];
    copyBack(b, loop.getResult(0), index(b, 0), nOwned, ownedOut, d);
    if (twoEnded)
      copyBack(b, loop.getResult(0), nOwned, countOf(b, halo, d), ghostOut,
               d);
  }
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
    FlatSymbolRefAttr set = getSet(d, /*local=*/false);
    Value count = countOf(b, layout, d);
    SmallVector<Value> ins;
    for (Value field : op.getGathered())
      ins.push_back(makeField(
          b, set, cast<FieldType>(field.getType()).getNumComponents(),
          b.getF64Type(), d, {storage[field].buffer}, {count}));
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
    FlatSymbolRefAttr set = getSet(d, /*local=*/false);
    Value count = countOf(b, layout, d);
    SmallVector<Value> ins;
    for (Value field : op.getGathered())
      ins.push_back(makeField(
          b, set, cast<FieldType>(field.getType()).getNumComponents(),
          b.getF64Type(), d, {storage[field].buffer}, {count}));
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
    copyBack(b, loop.getResult(0), index(b, 0), count, result, d);
  }
  storage[op.getResult()].buffer = result;
  return success();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lower(Operation *op) {
  loc = op->getLoc();
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
      .Case([&](CollectOp o) { return lowerCollect(o); })
      .Case([&](RedistributeOp o) { return lowerRedistribute(o); })
      .Case([&](HaloOp o) { return lowerHalo(o); })
      .Case([&](HaloCopyOp o) { return lowerHaloCopy(o); })
      .Case([&](HaloSumOp o) { return lowerHaloSum(o); })
      .Case([&](ZerosOp o) { return lowerZeros(o); })
      .Case([&](NeighborhoodOp o) { return lowerNeighborhood(o); })
      .Case([&](PartitionCentersOp o) { return lowerPartitionCenters(o); })
      .Case([&](SumPairsOp o) { return lowerSumPairs(o); })
      .Case([&](GatherPairsOp o) { return lowerGatherPairs(o); })
      .Case([&](SumParticlesOp o) { return lowerSumParticles(o); })
      .Case([&](MapParticlesOp o) { return lowerMapParticles(o); })
      .Case([&](AllReduceOp o) { return lowerAllReduce(o); })
      .Default([&](Operation *other) {
        return other->emitOpError() << "is not run by the reference executor";
      });
}

/// Whether `type` is a type of md_dist, which the lowered program does not
/// have.
static bool isDistributedType(Type type) {
  return isa<LayoutType, HaloType, FieldType, PartialType, RelationType>(
      type);
}

LogicalResult Lowering::run(func::FuncOp function) {
  SmallVector<Operation *> ops;
  function.walk([&](Operation *op) {
    if (isa_and_nonnull<MDDistDialect>(op->getDialect()) && !isa<YieldOp>(op) &&
        !op->getParentOfType<SumPairsOp>() &&
        !op->getParentOfType<GatherPairsOp>() &&
        !op->getParentOfType<SumParticlesOp>() &&
        !op->getParentOfType<MapParticlesOp>())
      ops.push_back(op);
  });
  if (ops.empty())
    return success();
  Block *block = ops.front()->getBlock();
  for (BlockArgument argument : function.getArguments())
    if (isDistributedType(argument.getType()))
      return function.emitOpError()
             << "the reference executor runs the values of md_dist within "
                "a function; an argument of md_dist cannot be run";
  for (Operation *op : ops) {
    if (op->getBlock() != block)
      return op->emitOpError()
             << "the reference executor runs the ops of md_dist in one "
                "block, without control flow between them";
    for (Value result : op->getResults())
      if (isDistributedType(result.getType()))
        for (Operation *user : result.getUsers())
          if (!isa_and_nonnull<MDDistDialect>(user->getDialect()))
            return user->emitOpError()
                   << "uses a value of md_dist, which the reference "
                      "executor cannot hand to an op of another dialect";
  }
  end = block->getTerminator();
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
  if (throughExec)
    declareSets();
  for (Operation *op : ops)
    if (failed(lower(op)))
      return failure();
  for (Operation *op : llvm::reverse(ops)) {
    op->dropAllUses();
    op->erase();
  }
  storage.clear();
  relations.clear();
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
    Lowering lowering(module, false, 64, 3, haloEvents);
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
