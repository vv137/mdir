// The reference executor of md_dist: every domain in one process.
//
// See docs/distributed-ml2.md, Sections 6 and 7.1 (D275). Every op
// runs for all domains in turn, in the order of the domains; collective
// ops are the points where all domains' values are complete, so running
// op by op is a bulk-synchronous schedule of the global view. Layouts and
// halos are handles of the in-process transport (runtime/mdrt_dist.c), and
// a distributed field is a buffer memref<D x rows x k x f64>.

#include "mdir/Conversion/Passes.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
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
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
namespace md = mdir::md;
namespace mdrt = mdir::mdrt;
using namespace mdir::md_dist;

namespace mdir {
#define GEN_PASS_DEF_CONVERTMDDISTTOLOOPS
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

class Lowering {
public:
  Lowering(ModuleOp module)
      : module(module), loc(UnknownLoc::get(module.getContext())) {}

  LogicalResult run(func::FuncOp function);

private:
  LogicalResult lower(Operation *op);
  LogicalResult lowerPartition(PartitionOp op);
  LogicalResult lowerDistribute(DistributeOp op);
  LogicalResult lowerCollect(CollectOp op);
  LogicalResult lowerHalo(HaloOp op);
  LogicalResult lowerHaloCopy(HaloCopyOp op);
  LogicalResult lowerHaloSum(HaloSumOp op);
  LogicalResult lowerZeros(ZerosOp op);
  LogicalResult lowerNeighborhood(NeighborhoodOp op);
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
  func::FuncOp copy = declare(
      "mdrtDistHaloCopy",
      b.getFunctionType({i64, owned.getType(), ghosts.getType(), i64}, {}),
      true);
  Value image = arith::ConstantOp::create(
      b, loc, b.getI64IntegerAttr(op.getImage() ? 1 : 0));
  func::CallOp::create(b, loc, copy,
                       ValueRange{halo.handle, owned, ghosts, image});
  storage[op.getResult()].buffer = ghosts;
  return success();
}

LogicalResult Lowering::lowerHaloSum(HaloSumOp op) {
  OpBuilder b(op);
  Storage &halo = storage[op.getHalo()];
  Storage &layout = storage[layoutOf(op.getHalo())];
  Value owned = storage[op.getOwned()].buffer;
  Value ghosts = storage[op.getGhosts()].buffer;
  int64_t k = op.getResult().getType().getNumComponents();
  Value result = allocate(b, layout.capacity, k);
  memref::CopyOp::create(b, loc, owned, result);
  func::FuncOp sum = declare(
      "mdrtDistHaloSum",
      b.getFunctionType({b.getI64Type(), ghosts.getType(), result.getType()},
                        {}),
      true);
  func::CallOp::create(b, loc, sum, ValueRange{halo.handle, ghosts, result});
  storage[op.getResult()].buffer = result;
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
              Value xi = load(bi, xo, d, i, vec3);
              SmallVector<Value> gi;
              for (unsigned g = 0; g < ownedFields.size(); ++g)
                gi.push_back(load(bi, ownedFields[g], d, i, types[g]));
              Value idi = unique ? Value(memref::LoadOp::create(
                                       bi, loc, layout.ids, ValueRange{d, i}))
                                 : Value();
              auto rows = scf::ForOp::create(
                  bi, loc, c(bi, 0), local, c(bi, 1),
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
// The pass
//===----------------------------------------------------------------------===//

LogicalResult Lowering::lower(Operation *op) {
  loc = op->getLoc();
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](PartitionOp o) { return lowerPartition(o); })
      .Case([&](DistributeOp o) { return lowerDistribute(o); })
      .Case([&](CollectOp o) { return lowerCollect(o); })
      .Case([&](HaloOp o) { return lowerHalo(o); })
      .Case([&](HaloCopyOp o) { return lowerHaloCopy(o); })
      .Case([&](HaloSumOp o) { return lowerHaloSum(o); })
      .Case([&](ZerosOp o) { return lowerZeros(o); })
      .Case([&](NeighborhoodOp o) { return lowerNeighborhood(o); })
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
  for (Operation *op : ops)
    if (failed(lower(op)))
      return failure();
  for (Operation *op : llvm::reverse(ops)) {
    op->dropAllUses();
    op->erase();
  }
  storage.clear();
  decomposition = nullptr;
  return success();
}

namespace {
class ConvertMDDistToLoops
    : public mdir::impl::ConvertMDDistToLoopsBase<ConvertMDDistToLoops> {
public:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    Lowering lowering(module);
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
} // namespace
