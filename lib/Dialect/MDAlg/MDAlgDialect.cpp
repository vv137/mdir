// The md_alg dialect and the verifiers of its ops (D406). The
// operand types are checked by the constraints of MDAlgOps.td; the rules
// here are those of the execution part and of the buffers that an op
// must not share.

#include "mdir/Dialect/MDAlg/MDAlgDialect.h"
#include "mdir/Dialect/MDAlg/MDAlgOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/DialectImplementation.h"

#include <cmath>

using namespace mlir;
using namespace mdir::md_alg;

#include "mdir/Dialect/MDAlg/MDAlgDialect.cpp.inc"
#include "mdir/Dialect/MDAlg/MDAlgEnums.cpp.inc"
#include "mdir/Dialect/MDAlg/MDAlgInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "mdir/Dialect/MDAlg/MDAlgOps.cpp.inc"

void MDAlgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/MDAlg/MDAlgOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

/// The launch values of `op`: none, or `expected`.
static LogicalResult verifyLaunch(Operation *op, OperandRange launch,
                                  unsigned expected, StringRef what) {
  if (!launch.empty() && launch.size() != expected)
    return op->emitOpError() << "takes " << expected << " launch values ("
                             << what << "), not " << launch.size();
  return success();
}

LogicalResult ScanOp::verify() {
  std::optional<ScanStrategy> strategy = getStrategy();
  bool scratch = static_cast<bool>(getScratch());
  bool launch = !getLaunch().empty();
  if ((strategy.has_value() != scratch) || (scratch != launch))
    return emitOpError()
           << "takes its strategy, its scratch, and its launch together";
  if (!strategy)
    return success();
  Type element = cast<MemRefType>(getScratch().getType()).getElementType();
  bool wide = *strategy == ScanStrategy::SinglePass;
  if (element.isInteger(64) != wide)
    return emitOpError() << "with the strategy '"
                         << stringifyScanStrategy(*strategy)
                         << "' takes a scratch of " << (wide ? "i64" : "i32")
                         << ", not " << element;
  if (*strategy == ScanStrategy::Chunked) {
    if (!getCursor())
      return emitOpError()
             << "with the strategy 'chunked' requires a cursor, which it "
                "writes";
    return verifyLaunch(*this, getLaunch(), 5,
                        "chunks, chunk, grid, block, one");
  }
  return verifyLaunch(*this, getLaunch(), 3, "blocks, threads, one");
}

LogicalResult ScatterToBinsOp::verify() {
  if (getOut() == getKeys() || getOut() == getCursor())
    return emitOpError()
           << "writes its items into its keys or its cursor, which it reads";
  return verifyLaunch(*this, getLaunch(), 3, "grid, block, one");
}

LogicalResult SortBinsOp::verify() {
  std::optional<SortStrategy> strategy = getStrategy();
  if (strategy.has_value() != !getLaunch().empty())
    return emitOpError() << "takes its strategy and its launch together";
  if (Value out = getOut())
    if (out == getItems() || out == getStart() || out == getBy())
      return emitOpError()
             << "sorts into its items, its offsets, or its order, which it "
                "reads";
  if (getKeys() && !getOut())
    return emitOpError() << "takes the keys of the items only with 'into'";
  if (strategy == SortStrategy::Insertion && getOut())
    return emitOpError() << "with the strategy 'insertion' sorts in place; "
                            "it takes no 'into'";
  if (strategy == SortStrategy::StableRank) {
    if (!getOut() || !getKeys())
      return emitOpError() << "with the strategy 'stable_rank' requires "
                              "'into' and the keys of the items";
    if (getBy())
      return emitOpError()
             << "with the strategy 'stable_rank' orders by the items; it "
                "takes no 'by'";
  }
  return verifyLaunch(*this, getLaunch(), 3, "grid, block, one");
}

LogicalResult WrapOp::verify() {
  if (getSignedCompare() && !getType().isInteger(32))
    return emitOpError() << "compares signed only on i32";
  return success();
}

//===----------------------------------------------------------------------===//
// Stages (D407)
//===----------------------------------------------------------------------===//

/// Whether `v` is `item`, or `item * a + c` for constants a != 0 and c, as
/// index or integer arithmetic and casts: an index injective in the item
/// by construction.
static bool isItemSlot(Value v, Value item) {
  if (v == item)
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  auto constant = [](Value c, bool nonzero) {
    IntegerAttr attr;
    if (!matchPattern(c, m_Constant(&attr)))
      return false;
    return !nonzero || !attr.getValue().isZero();
  };
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
          arith::ExtUIOp>(def))
    return isItemSlot(def->getOperand(0), item);
  if (isa<arith::AddIOp>(def)) {
    Value x = def->getOperand(0), y = def->getOperand(1);
    return (constant(y, false) && isItemSlot(x, item)) ||
           (constant(x, false) && isItemSlot(y, item));
  }
  if (isa<arith::MulIOp>(def)) {
    Value x = def->getOperand(0), y = def->getOperand(1);
    return (constant(y, true) && isItemSlot(x, item)) ||
           (constant(x, true) && isItemSlot(y, item));
  }
  return false;
}

/// Whether `v` is computed, through integer arithmetic and casts with
/// constants, from the address of one of `buffers`
/// (`memref.extract_aligned_pointer_as_index`) plus a constant.
static bool isAddressOf(Value v, ValueRange buffers) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (auto extract = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(def))
    return llvm::is_contained(buffers, extract.getSource());
  if (isa<LLVM::IntToPtrOp, arith::IndexCastOp, arith::IndexCastUIOp>(def))
    return isAddressOf(def->getOperand(0), buffers);
  if (isa<arith::AddIOp>(def)) {
    Value x = def->getOperand(0), y = def->getOperand(1);
    return (matchPattern(y, m_Constant()) && isAddressOf(x, buffers)) ||
           (matchPattern(x, m_Constant()) && isAddressOf(y, buffers));
  }
  return false;
}

/// The number of md_alg.bin or md_alg.no_bin that every path through
/// `block` runs, or -1 where paths differ or one is nested elsewhere than
/// in the branches of scf.if.
static int64_t binsOnEveryPath(Block &block) {
  int64_t count = 0;
  for (Operation &op : block) {
    if (isa<BinOp, NoBinOp>(op)) {
      ++count;
      continue;
    }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      int64_t then = binsOnEveryPath(*branch.thenBlock());
      int64_t otherwise =
          branch.elseBlock() ? binsOnEveryPath(*branch.elseBlock()) : 0;
      if (then < 0 || otherwise < 0 || then != otherwise)
        return -1;
      count += then;
      continue;
    }
    bool nested = false;
    op.walk([&](Operation *inner) {
      if (isa<BinOp, NoBinOp>(inner))
        nested = true;
    });
    if (nested)
      return -1;
  }
  return count;
}

/// The buffer that `v` views, through the views whose elements are those
/// of their source (as the collection analysis follows them).
static Value viewedRoot(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (!isa<memref::MemorySpaceCastOp, memref::CastOp, memref::SubViewOp,
             memref::ReinterpretCastOp>(def))
      break;
    v = def->getOperand(0);
  }
  return v;
}

/// Whether `v` and one of `buffers` are, or view, one buffer (#874): the
/// checks of md_alg.sort_by_key and md_alg.bin_members that two operands,
/// or an operand and a use in the region, are not one buffer compare the
/// buffers that they view, so that a view does not pass for another buffer.
static bool viewsAnyOf(ValueRange buffers, Value v) {
  Value root = viewedRoot(v);
  return llvm::any_of(buffers,
                      [&](Value buffer) { return viewedRoot(buffer) == root; });
}

/// Whether every write of `region`, whose items run in no fixed order, is
/// a slot of its item (`item`) in a buffer of `writes` or an addition to a
/// counter of `adds` (D407), and no op uses a buffer of `own`. With
/// `strict`, a buffer of `writes` or `adds` is used only as the target of
/// such a write, so that no item reads what another wrote
/// (D409). A use of a view of a buffer, made outside the region, is a use
/// of the buffer (#874). `noun` and `name` name the region and its op in
/// the messages.
static LogicalResult verifySideOutputs(Region &region, Value item,
                                       ArrayRef<Value> own, ValueRange writes,
                                       ValueRange adds, StringRef noun,
                                       StringRef name, bool strict) {
  WalkResult result = region.walk([&](Operation *op) -> WalkResult {
    for (Value operand : op->getOperands())
      if (viewsAnyOf(own, operand))
        return op->emitOpError()
               << "uses a buffer of the '" << name << "' of its " << noun;
    if (strict)
      for (OpOperand &use : op->getOpOperands()) {
        Value v = use.get();
        if (!viewsAnyOf(writes, v) && !viewsAnyOf(adds, v))
          continue;
        bool target =
            (isa<memref::StoreOp>(op) &&
             use.getOperandNumber() ==
                 cast<memref::StoreOp>(op).getMemrefMutable()
                     .getOperandNumber()) ||
            (isa<memref::AtomicRMWOp>(op) &&
             use.getOperandNumber() ==
                 cast<memref::AtomicRMWOp>(op).getMemrefMutable()
                     .getOperandNumber()) ||
            isa<memref::ExtractAlignedPointerAsIndexOp>(op);
        if (!target)
          return op->emitOpError()
                 << "reads a side output of the " << noun << " of '" << name
                 << "', which another item may have written";
      }
    if (isa<BinOp, NoBinOp, YieldOp, scf::IfOp, scf::YieldOp>(op))
      return WalkResult::advance();
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      Value target = store.getMemRef();
      if (!llvm::is_contained(writes, target))
        return op->emitOpError() << "writes a buffer that the " << noun
                                 << " does not declare in 'writes'";
      ValueRange indices = store.getIndices();
      bool slot = !indices.empty() && isItemSlot(indices[0], item);
      for (Value other : indices.drop_front())
        slot = slot && matchPattern(other, m_Constant());
      if (!slot)
        return op->emitOpError()
               << "writes a side output at a place that is not a slot of "
                  "the item: the first index the item times a constant "
                  "other than zero plus a constant, the others constants";
      return WalkResult::advance();
    }
    if (auto rmw = dyn_cast<memref::AtomicRMWOp>(op)) {
      if (rmw.getKind() != arith::AtomicRMWKind::addi ||
          !llvm::is_contained(adds, rmw.getMemref()))
        return op->emitOpError() << "is not an addition to a counter that "
                                    "the "
                                 << noun << " declares in 'adds'";
      return WalkResult::advance();
    }
    if (auto rmw = dyn_cast<LLVM::AtomicRMWOp>(op)) {
      if (rmw.getBinOp() != LLVM::AtomicBinOp::add ||
          !isAddressOf(rmw.getPtr(), adds))
        return op->emitOpError()
               << "is not an addition at the address of a counter that the "
               << noun << " declares in 'adds', plus a constant";
      return WalkResult::advance();
    }
    if (auto extract = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(op)) {
      if (!llvm::is_contained(adds, extract.getSource()))
        return op->emitOpError() << "takes the address of a buffer that the "
                                 << noun << " does not declare in 'adds'";
      return WalkResult::advance();
    }
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects)
      return op->emitOpError() << "has effects that the " << noun << " of '"
                               << name << "' cannot state";
    SmallVector<MemoryEffects::EffectInstance> list;
    effects.getEffects(list);
    for (auto &effect : list)
      if (!isa<MemoryEffects::Read>(effect.getEffect()))
        return op->emitOpError()
               << "writes memory in the " << noun << " of '" << name
               << "' other than a declared side output";
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

/// Refuses, on `op`, a side output of `writes` or `adds` that is, or views,
/// a buffer of `own` or another side output (D409, #871): it would let one
/// item see what another wrote, or what the op itself writes or reads.
/// `owned` names the buffers of `own` in the message.
static LogicalResult verifyDistinctSideOutputs(Operation *op,
                                               ArrayRef<Value> own,
                                               ValueRange writes,
                                               ValueRange adds,
                                               StringRef owned) {
  SmallVector<Value> roots;
  for (Value v : own)
    roots.push_back(viewedRoot(v));
  for (Value side : llvm::concat<Value>(writes, adds)) {
    Value root = viewedRoot(side);
    if (llvm::is_contained(roots, root))
      return op->emitOpError() << "declares as a side output " << owned
                               << ", or another side output, or a view of "
                                  "one";
    roots.push_back(root);
  }
  return success();
}

LogicalResult SortByKeyOp::verify() {
  bool launch = !getLaunch().empty();
  bool scan = getScanStrategy().has_value();
  bool sort = getSortStrategy().has_value();
  // The sort within the bins goes with the order 'stable' (checked below).
  bool sorted = getOrder() == Order::Stable;
  if (launch != scan || (sorted && scan != sort) ||
      scan != (bool)getScanScratch())
    return emitOpError() << "takes its launch, its scan's strategy, scratch, "
                            "and launch, and its sort's strategy and launch "
                            "together";
  if (launch) {
    if (failed(verifyLaunch(*this, getLaunch(), 3, "grid, block, one")))
      return failure();
    Type element =
        cast<MemRefType>(getScanScratch().getType()).getElementType();
    bool wide = *getScanStrategy() == ScanStrategy::SinglePass;
    if (element.isInteger(64) != wide)
      return emitOpError() << "with the scan '"
                           << stringifyScanStrategy(*getScanStrategy())
                           << "' takes a scratch of " << (wide ? "i64" : "i32")
                           << ", not " << element;
    if (*getScanStrategy() == ScanStrategy::Chunked) {
      if (failed(verifyLaunch(*this, getScanLaunch(), 5,
                              "chunks, chunk, grid, block, one")))
        return failure();
    } else if (failed(verifyLaunch(*this, getScanLaunch(), 3,
                                   "blocks, threads, one"))) {
      return failure();
    }
    if (sorted &&
        failed(verifyLaunch(*this, getSortLaunch(), 3, "grid, block, one")))
      return failure();
    bool rank = sort && *getSortStrategy() == SortStrategy::StableRank;
    if (rank != (bool)getSortBuffer())
      return emitOpError() << "takes the buffer of its scatter with the sort "
                              "'stable_rank' alone";
    if (rank && getBy())
      return emitOpError()
             << "with the sort 'stable_rank' orders by the items; it takes "
                "no 'by'";
  } else if (getSortBuffer() || sort || !getScanLaunch().empty() ||
             !getSortLaunch().empty()) {
    return emitOpError() << "takes its launch, its scan's strategy, scratch, "
                            "and launch, and its sort's strategy and launch "
                            "together";
  }
  if (getOrder() == Order::Unordered) {
    if (getBy())
      return emitOpError() << "orders its bins by 'by' only with the order "
                              "'stable'";
    if (sort || !getSortLaunch().empty() || getSortBuffer())
      return emitOpError() << "with the order 'unordered' takes no sort "
                              "within its bins";
  }
  if (getEveryKey() && !getKey().empty())
    return emitOpError() << "takes 'every_key' only with keys given; with a "
                            "region the bins say it";
  SmallVector<Value> own = {getKeys(), getCounts(), getStart(), getCursor(),
                            getOut()};
  if (Value buffer = getSortBuffer())
    own.push_back(buffer);
  for (size_t i = 0; i < own.size(); ++i)
    for (size_t j = i + 1; j < own.size(); ++j)
      if (viewsAnyOf(own[j], own[i]))
        return emitOpError() << "takes one buffer for two of its keys, "
                                "counts, start, cursor, order, and scatter";
  if (Value by = getBy())
    if (viewsAnyOf(getOut(), by) ||
        (getSortBuffer() && viewsAnyOf(getSortBuffer(), by)))
      return emitOpError() << "sorts into its order 'by', which it reads";
  for (Value w : getWrites())
    if (llvm::is_contained(getAdds(), w))
      return emitOpError() << "declares a buffer both written and added to";
  if (Value by = getBy())
    own.push_back(by);
  return verifyDistinctSideOutputs(*this, own, getWrites(), getAdds(),
                                   "one of its own buffers or its 'by'");
}

LogicalResult SortByKeyOp::verifyRegions() {
  Region &region = getKey();
  if (region.empty())
    return success();
  Block &block = region.front();
  if (block.getNumArguments() != 1 || !block.getArgument(0).getType().isIndex())
    return emitOpError() << "takes a key region of one index argument, the "
                            "item";
  Value item = block.getArgument(0);
  if (binsOnEveryPath(block) != 1)
    return emitOpError() << "runs exactly one 'md_alg.bin' or "
                            "'md_alg.no_bin' on every path through its key "
                            "region, in the branches of 'scf.if' alone";
  SmallVector<Value> own = {getKeys(), getCounts(), getStart(), getCursor(),
                            getOut()};
  if (Value buffer = getSortBuffer())
    own.push_back(buffer);
  return verifySideOutputs(region, item, own, getWrites(), getAdds(),
                           "key region", "md_alg.sort_by_key",
                           /*strict=*/false);
}

/// Whether `op` is in the key region of an md_alg.sort_by_key, through the
/// branches of scf.if alone.
static LogicalResult verifyInKey(Operation *op) {
  Operation *parent = op->getParentOp();
  while (isa_and_nonnull<scf::IfOp>(parent))
    parent = parent->getParentOp();
  if (!isa_and_nonnull<SortByKeyOp>(parent))
    return op->emitOpError()
           << "stands in the key region of an 'md_alg.sort_by_key', directly "
              "or in the branches of 'scf.if'";
  return success();
}

LogicalResult BinOp::verify() { return verifyInKey(*this); }
LogicalResult NoBinOp::verify() { return verifyInKey(*this); }

LogicalResult AppendOp::verify() {
  if (getValues().empty() || getValues().size() != getInto().size())
    return emitOpError() << "takes one buffer of 'into' for each of its "
                            "values, and at least one";
  int64_t block = getBlock();
  if (block <= 0 || !llvm::isPowerOf2_64(block))
    return emitOpError() << "takes blocks of a power of two of entries, not "
                         << block;
  if ((bool)getLane() != (bool)getBelow())
    return emitOpError() << "takes its lane and the mask of the lanes below "
                            "it together";
  SmallVector<Value> buffers(getInto().begin(), getInto().end());
  buffers.append({getUnits(), getOrdinals(), getPool()});
  for (size_t i = 0; i < buffers.size(); ++i)
    for (size_t j = i + 1; j < buffers.size(); ++j)
      if (buffers[i] == buffers[j])
        return emitOpError() << "takes one buffer for two of its entries, "
                                "names, and pool";
  return success();
}

LogicalResult GridPlanOp::verify() {
  if (getLengths().empty() || getCounts().size() != getLengths().size())
    return emitOpError() << "plans along one edge or more, a count for each";
  Type real = getLengths().empty() ? Type() : getLengths().front().getType();
  for (Value v : getLengths())
    if (v.getType() != real)
      return emitOpError() << "takes its edges and width of one type";
  if (getWidth() && getWidth().getType() != real)
    return emitOpError() << "takes its edges and width of one type";
  if (getVolume() && getVolume().getType() != real)
    return emitOpError() << "takes its volume of the type of its edges";
  auto leastAttr = cast<FloatAttr>(getLeastAttr());
  auto mostAttr = cast<FloatAttr>(getMostAttr());
  if (leastAttr.getType() != real || mostAttr.getType() != real ||
      (getOccupancyAttr() &&
       cast<FloatAttr>(getOccupancyAttr()).getType() != real))
    return emitOpError() << "takes its bounds and occupancy of the type of "
                            "its edges";
  double least = leastAttr.getValueAsDouble();
  double most = mostAttr.getValueAsDouble();
  if (least != std::floor(least) || most != std::floor(most) || least < 1.0 ||
      least > most)
    return emitOpError() << "takes whole bounds with 1 <= least <= most";
  bool occupancy = static_cast<bool>(getOccupancyAttr());
  if ((bool)getWidth() == occupancy)
    return emitOpError() << "takes a width or an occupancy, one of them";
  if (occupancy &&
      !(cast<FloatAttr>(getOccupancyAttr()).getValueAsDouble() > 0.0))
    return emitOpError() << "takes an occupancy greater than zero";
  if (occupancy != (bool)getVolume() || occupancy != (bool)getPopulation())
    return emitOpError() << "takes an occupancy with its volume and its "
                            "population";
  return success();
}

//===----------------------------------------------------------------------===//
// Determinism (D408)
//===----------------------------------------------------------------------===//

StringRef mdir::md_alg::getDeterminismName(Determinism determinism) {
  return stringifyDeterminism(determinism);
}

Determinism DeterminismInterface::getRequiredDeterminism() {
  if (auto stated = (*this)->getAttrOfType<DeterminismAttr>("determinism"))
    return stated.getValue();
  return getDeterminismDefault();
}

LogicalResult mdir::md_alg::verifyDeterminism(Operation *op) {
  auto self = cast<DeterminismInterface>(op);
  auto stated = op->getAttrOfType<DeterminismAttr>("determinism");
  if (!stated)
    return success();
  Determinism bound = self.getDeterminismBound();
  if (static_cast<uint32_t>(stated.getValue()) <=
      static_cast<uint32_t>(bound))
    return success();
  return op->emitOpError()
         << "cannot meet the determinism requirement '"
         << stringifyDeterminism(stated.getValue()) << "': its contract leaves "
         << self.getFreeResults() << " free, so it admits at most '"
         << stringifyDeterminism(bound) << "'";
}

// An integer scan, a stable sort, and the grid plan fix every result: their
// serial definitions are the results on every target.
Determinism ScanOp::getDeterminismBound() { return Determinism::Bitwise; }
Determinism ScanOp::getDeterminismDefault() { return Determinism::Bitwise; }
StringRef ScanOp::getFreeResults() { return ""; }
Determinism SortBinsOp::getDeterminismBound() { return Determinism::Bitwise; }
Determinism SortBinsOp::getDeterminismDefault() {
  return Determinism::Bitwise;
}
StringRef SortBinsOp::getFreeResults() { return ""; }
Determinism GridPlanOp::getDeterminismBound() { return Determinism::Bitwise; }
Determinism GridPlanOp::getDeterminismDefault() {
  return Determinism::Bitwise;
}
StringRef GridPlanOp::getFreeResults() { return ""; }

// The order within a bin, left free, can be fixed by a schedule
// (reproducible) but is no result of the serial definition (bitwise).
Determinism ScatterToBinsOp::getDeterminismBound() {
  return Determinism::Reproducible;
}
Determinism ScatterToBinsOp::getDeterminismDefault() {
  return Determinism::Unordered;
}
StringRef ScatterToBinsOp::getFreeResults() {
  return "the order of the items within a bin";
}

Determinism SortByKeyOp::getDeterminismBound() {
  return getOrder() == Order::Stable ? Determinism::Bitwise
                                     : Determinism::Reproducible;
}
Determinism SortByKeyOp::getDeterminismDefault() {
  return getOrder() == Order::Stable ? Determinism::Bitwise
                                     : Determinism::Unordered;
}
StringRef SortByKeyOp::getFreeResults() {
  return getOrder() == Order::Stable
             ? ""
             : "the order of the items within a bin (order unordered)";
}

// The lists read through their names are fixed; where their entries are
// stored, the numbers of the blocks, is left to the order in which
// concurrent lists take them.
Determinism AppendOp::getDeterminismBound() {
  return Determinism::Reproducible;
}
Determinism AppendOp::getDeterminismDefault() {
  return Determinism::Unordered;
}
StringRef AppendOp::getFreeResults() {
  return "the numbers of the blocks";
}

//===----------------------------------------------------------------------===//
// md_alg.bin_members (D409)
//===----------------------------------------------------------------------===//

/// Whether `name` combines integers (exactly, commutatively, associatively)
/// or floats, or is no combiner (neither).
static bool isIntegerCombiner(StringRef name) {
  return llvm::is_contained({"addi", "muli", "andi", "ori", "xori", "maxsi",
                             "minsi", "maxui", "minui"},
                            name);
}
static bool isFloatCombiner(StringRef name) {
  return llvm::is_contained({"addf", "mulf", "maxnumf", "minnumf"}, name);
}

bool BinMembersOp::combinesExactly() {
  if (!getCombine())
    return true;
  return llvm::all_of(getCombine()->getAsValueRange<StringAttr>(),
                      isIntegerCombiner);
}

LogicalResult BinMembersOp::verify() {
  SmallVector<Value> reads = {getStart(), getItems()};
  if (Value by = getBy())
    reads.push_back(by);
  if (failed(verifyDistinctSideOutputs(*this, reads, getWrites(), getAdds(),
                                       "a buffer that it reads")))
    return failure();
  if (getBy() && getVisit() == Visit::Unordered)
    return emitOpError() << "visits by 'by' only in the order 'sorted' or "
                            "'stable'";
  size_t combiners = getCombine() ? getCombine()->size() : 0;
  if (combiners != getInit().size())
    return emitOpError() << "takes one combiner for each initial value";
  if (getResults().getTypes() != getInit().getTypes())
    return emitOpError() << "has results of the types of its initial values";
  if (!getCombine())
    return success();
  for (auto [name, type] :
       llvm::zip(getCombine()->getAsValueRange<StringAttr>(),
                 getInit().getTypes())) {
    bool integer = type.isIntOrIndex();
    if (!(integer ? isIntegerCombiner(name) : isFloatCombiner(name)))
      return emitOpError() << "cannot combine " << type << " by '" << name
                           << "'";
    if (!integer && getVisit() == Visit::Unordered)
      return emitOpError()
             << "with the visit 'unordered' combines integers alone: '"
             << name << "' on " << type
             << " depends on the order of the items";
  }
  return success();
}

LogicalResult BinMembersOp::verifyRegions() {
  Block &block = getBody().front();
  if (block.getNumArguments() != 1 || !block.getArgument(0).getType().isIndex())
    return emitOpError() << "takes a region of one index argument, the item";
  auto yield = block.empty() ? YieldOp() : dyn_cast<YieldOp>(block.back());
  if (!yield)
    return emitOpError() << "ends its region in 'md_alg.yield'";
  if (yield.getValues().getTypes() != getInit().getTypes())
    return emitOpError() << "yields one value of the type of each initial "
                            "value";
  SmallVector<Value> own = {getStart(), getItems()};
  if (Value by = getBy())
    own.push_back(by);
  return verifySideOutputs(getBody(), block.getArgument(0), own, getWrites(),
                           getAdds(), "region", "md_alg.bin_members",
                           /*strict=*/true);
}

Determinism BinMembersOp::getDeterminismBound() { return Determinism::Bitwise; }
Determinism BinMembersOp::getDeterminismDefault() {
  return Determinism::Bitwise;
}
StringRef BinMembersOp::getFreeResults() { return ""; }

//===----------------------------------------------------------------------===//
// md_alg.traverse_pairs (D415)
//===----------------------------------------------------------------------===//

StringRef TraversePairsOp::getExchangeOf(unsigned index) {
  return cast<StringAttr>(getExchange()[index]).getValue();
}

/// What the declarations of `op` require of its list (D413), as
/// `md_exec::getRequirement` computes it for `md_exec.pair_for`.
static mdir::mdrt::NeighborsContract getTraversalRequirement(
    TraversePairsOp op) {
  using Contract = mdir::mdrt::NeighborsContract;
  bool unique = op.getMultiplicity() == Multiplicity::Unique;
  Contract required;
  required.traversal =
      unique ? Contract::Traversal::Unique : Contract::Traversal::Directed;
  required.images = Contract::Images::Minimum;
  required.exclusions =
      unique ? Contract::Exclusions::Removed : Contract::Exclusions::Masked;
  if (op.getCombine() == Combine::Owner)
    required = Contract::withOrganization(
        required, Contract::Organization::GroupedBy, Contract::Group::Owner,
        Contract::Order::None);
  if (op.getVisit() == Visit::Stable) {
    Contract ordered = Contract::withOrganization(
        required, Contract::Organization::Stable,
        unique ? Contract::Group::Group : Contract::Group::Owner,
        Contract::Order::Search);
    if (std::optional<Contract> joined =
            Contract::joinOrganization(required, ordered))
      required = *joined;
    else
      required = ordered;
  }
  return required;
}

LogicalResult TraversePairsOp::verify() {
  // The two forms of the list.
  bool buffers = isBufferForm();
  if (buffers) {
    if (getLists().size() != getNumListBuffers())
      return emitOpError() << "takes " << getNumListBuffers()
                           << " buffers of the layout of the groups in its "
                              "buffer form, not "
                           << getLists().size();
    for (Value list : getLists()) {
      auto type = cast<MemRefType>(list.getType());
      if (type.getRank() != 1 || !type.getElementType().isInteger(32))
        return emitOpError()
               << "takes rank-1 buffers of i32 for the layout of the groups";
    }
    if (getSizes().size() != 2)
      return emitOpError() << "takes two sizes, the places and the units, in "
                              "its buffer form";
    auto contract = getContractAttr();
    if (!contract || !isa<mdir::mdrt::NeighborsType>(contract.getValue()))
      return emitOpError()
             << "states the contract of its buffers, a neighbor structure's "
                "type, in its buffer form";
    if (!getReach())
      return emitOpError() << "states the reach of its buffers in its buffer "
                              "form";
    if (getCutoff().convertToDouble() > getReach()->convertToDouble())
      return emitOpError() << "has a cutoff, " << getCutoff().convertToDouble()
                           << ", beyond the reach of its list, "
                           << getReach()->convertToDouble();
  } else if (!getLists().empty() || !getSizes().empty() || getContractAttr() ||
             getReach()) {
    return emitOpError() << "takes its list as a structure or as buffers, not "
                            "both";
  }

  // The declarations.
  unsigned numOuts = getOuts().size();
  unsigned numValues = numOuts + getReduce().size();
  if (getAlgebra().size() != numValues)
    return emitOpError() << "expected " << numValues
                         << " algebra entries, one per destination and sum, "
                            "got "
                         << getAlgebra().size();
  if (getExchange().size() != numValues)
    return emitOpError() << "expected " << numValues
                         << " exchange contracts, one per destination and "
                            "sum, got "
                         << getExchange().size();
  if (auto weights = getWeights())
    if (weights->size() != getReduce().size())
      return emitOpError() << "expected " << getReduce().size()
                           << " weights, one per sum, got "
                           << weights->size();
  if (auto overwrite = getOverwrite())
    if (overwrite->size() != numOuts)
      return emitOpError() << "expected " << numOuts
                           << " overwrite flags, one per destination, got "
                           << overwrite->size();
  if (!(getCutoff().convertToDouble() > 0.0))
    return emitOpError() << "expected a positive cutoff";

  bool unique = getMultiplicity() == Multiplicity::Unique;
  bool atomic = getCombine() == Combine::Atomic;
  if (getCombine() == Combine::Privatized)
    return emitOpError() << "takes the combination 'privatized', which is "
                            "reserved";
  if (getCombine() == Combine::Owner && unique)
    return emitOpError() << "adds each owner's row in its registers, which a "
                            "visit of each pair once does not give: the "
                            "partner's end is not the owner's row";
  if (getVisit() == Visit::Sorted)
    return emitOpError() << "visits 'unordered' or 'stable', not 'sorted'";

  for (unsigned i = 0; i != numValues; ++i) {
    StringRef algebra = cast<StringAttr>(getAlgebra()[i]).getValue();
    if (algebra != "any_order" && algebra != "any_order_exact" &&
        algebra != "fixed_order")
      return emitOpError() << "has an unknown algebra '" << algebra << "'";
    StringRef exchange = getExchangeOf(i);
    if (exchange != "none" && exchange != "symmetric" &&
        exchange != "antisymmetric")
      return emitOpError() << "has an unknown exchange contract '" << exchange
                           << "'";
    if ((unique || atomic) &&
        !(exchange == "symmetric" ||
          (i < numOuts && exchange == "antisymmetric")))
      return emitOpError()
             << "adds at both ends of a pair, which needs every destination "
                "symmetric or antisymmetric and every sum symmetric; value "
             << i << " is " << exchange;
    if (algebra != "fixed_order")
      continue;
    if (getVisit() != Visit::Stable)
      return emitOpError()
             << "visits 'unordered', which value " << i
             << ", 'fixed_order', does not allow: it needs a stable visit";
    if (atomic && i < numOuts)
      return emitOpError()
             << "adds destination " << i
             << ", 'fixed_order', by the combination 'atomic', whose order "
                "the threads decide";
    if (unique && i >= numOuts)
      return emitOpError()
             << "sums value " << i
             << ", 'fixed_order', over the units of a visit of each pair "
                "once, which fixes no order";
  }

  // What the declarations require of the list, and what it guarantees.
  auto type = cast<mdir::mdrt::NeighborsType>(
      buffers ? getContractAttr().getValue() : getNeighbors().getType());
  if (std::optional<std::string> reason =
          type.getContract().whyNot(getTraversalRequirement(*this)))
    return emitOpError() << "takes a list that does not meet its "
                            "requirement: "
                         << *reason;

  // A destination is no buffer that the op reads.
  SmallVector<Value> read = {getPositions()};
  llvm::append_range(read, getIns());
  llvm::append_range(read, getLists());
  for (auto [index, out] : llvm::enumerate(getOuts())) {
    if (viewsAnyOf(read, out))
      return emitOpError() << "has destination " << index
                           << ", which is or views a buffer that the op reads";
    for (auto [other, second] : llvm::enumerate(getOuts()))
      if (other > index && viewedRoot(second) == viewedRoot(out))
        return emitOpError() << "has destinations " << index << " and "
                             << other << ", which are one buffer";
  }
  return success();
}

LogicalResult TraversePairsOp::verifyRegions() {
  Block &block = getKernel().front();
  unsigned expected = 2 + 2 * getIns().size();
  if (block.getNumArguments() != expected)
    return emitOpError() << "takes a region of " << expected
                         << " arguments: r^2, the displacement, and two "
                            "values for each buffer of 'ins'";
  auto yield = block.empty() ? YieldOp() : dyn_cast<YieldOp>(block.back());
  if (!yield)
    return emitOpError() << "ends its region in 'md_alg.yield'";
  unsigned values = getOuts().size() + getReduce().size();
  if (yield.getValues().size() != values)
    return emitOpError() << "yields " << yield.getValues().size()
                         << " values, not one per destination and sum ("
                         << values << ")";
  // The region writes nothing: every write is the op's, through its
  // combination. It uses no destination, nor a view of one.
  WalkResult result = getKernel().walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      if (viewsAnyOf(getOuts(), operand)) {
        nested->emitOpError()
            << "uses a destination of 'md_alg.traverse_pairs' in its region, "
               "which reads or writes it outside the combination";
        return WalkResult::interrupt();
      }
    if (auto memory = dyn_cast<MemoryEffectOpInterface>(nested)) {
      if (memory.hasEffect<MemoryEffects::Write>() ||
          memory.hasEffect<MemoryEffects::Free>() ||
          memory.hasEffect<MemoryEffects::Allocate>()) {
        nested->emitOpError()
            << "writes memory in the region of 'md_alg.traverse_pairs', "
               "whose writes are the op's, through its combination";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (!nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
        !isPure(nested) && nested->getNumRegions() == 0) {
      nested->emitOpError()
          << "has effects that the verifier of 'md_alg.traverse_pairs' "
             "cannot see; its region may not write";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void TraversePairsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto add = [&](MemoryEffects::Effect *effect, OpOperand &operand) {
    effects.emplace_back(effect, &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  };
  for (OpOperand &operand : getNeighborsMutable())
    add(MemoryEffects::Read::get(), operand);
  for (OpOperand &operand : getListsMutable())
    add(MemoryEffects::Read::get(), operand);
  add(MemoryEffects::Read::get(), getPositionsMutable());
  for (OpOperand &operand : getInsMutable())
    add(MemoryEffects::Read::get(), operand);
  for (auto [index, operand] : llvm::enumerate(getOutsMutable())) {
    if (!overwrites(index))
      add(MemoryEffects::Read::get(), operand);
    add(MemoryEffects::Write::get(), operand);
  }
  for (OpOperand &operand : getScratchMutable())
    add(MemoryEffects::Write::get(), operand);
}
