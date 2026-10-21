// The calls of the templates to the emitters of CompactionEmitter.h
// (D296) and RaggedRows.h (D314):
// `func.call @mdir.primitive.<name>(...)`, each replaced by the code of its
// emitter where it stands.

#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/RaggedRows.h"
#include "mdir/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mdir;

namespace mdir {
#define GEN_PASS_DEF_EXPANDDEVICEPRIMITIVES
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

namespace {

/// What an operand of a primitive must be.
enum class Kind { I1, I32, Index, Buffer, WideBuffer, DynamicBuffer, SharedBuffer };

bool hasKind(Type type, Kind kind) {
  switch (kind) {
  case Kind::I1:
    return type.isInteger(1);
  case Kind::I32:
    return type.isInteger(32);
  case Kind::Index:
    return type.isIndex();
  case Kind::Buffer: {
    auto memref = dyn_cast<MemRefType>(type);
    return memref && memref.getRank() == 1 &&
           memref.getElementType().isInteger(32);
  }
  case Kind::WideBuffer: {
    auto memref = dyn_cast<MemRefType>(type);
    return memref && memref.getRank() == 1 &&
           memref.getElementType().isInteger(64);
  }
  case Kind::DynamicBuffer: {
    auto memref = dyn_cast<MemRefType>(type);
    return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
           memref.getElementType().isInteger(32);
  }
  case Kind::SharedBuffer: {
    auto memref = dyn_cast<MemRefType>(type);
    return memref && memref.getRank() == 1 && memref.hasStaticShape() &&
           memref.getElementType().isInteger(32) &&
           gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(memref) &&
           llvm::isPowerOf2_64(memref.getDimSize(0) - 1) &&
           memref.getDimSize(0) - 1 <= 32;
  }
  }
  return false;
}

/// The emitter of a primitive, given the call and its operands.
using Expander =
    std::function<SmallVector<Value>(OpBuilder &, func::CallOp, ValueRange)>;

struct Primitive {
  /// The operands; those past `least` may be left out.
  SmallVector<Kind> operands;
  unsigned least;
  /// The number of results, all i32 unless `resultKinds` says otherwise.
  unsigned results;
  Expander expand;
  SmallVector<Kind> resultKinds = {};
};

std::optional<Primitive> getPrimitive(StringRef name) {
  using namespace kernels;
  if (name == "warp_lanes_below")
    return Primitive{{Kind::I32, Kind::I32}, 1, 1,
                     [](OpBuilder &b, func::CallOp call, ValueRange args) {
                       return SmallVector<Value>{emitLanesBelow(
                           b, call.getLoc(), args[0],
                           args.size() > 1 ? args[1] : Value())};
                     }};
  if (name == "warp_ballot")
    return Primitive{{Kind::I1}, 1, 1,
                     [](OpBuilder &b, func::CallOp call, ValueRange args) {
                       return SmallVector<Value>{
                           emitWarpBallot(b, call.getLoc(), args[0])};
                     }};
  if (name == "warp_rank")
    return Primitive{{Kind::I32, Kind::I32}, 2, 1,
                     [](OpBuilder &b, func::CallOp call, ValueRange args) {
                       return SmallVector<Value>{
                           emitWarpRank(b, call.getLoc(), args[0], args[1])};
                     }};
  if (name == "warp_count")
    return Primitive{{Kind::I32}, 1, 1,
                     [](OpBuilder &b, func::CallOp call, ValueRange args) {
                       return SmallVector<Value>{
                           emitWarpCount(b, call.getLoc(), args[0])};
                     }};
  // warp_scan(value, lane, zero, width) {lanes = L, unsigned}: the
  // attribute `lanes` (32 if absent) and the unit attribute `unsigned`.
  if (name == "warp_scan")
    return Primitive{
        {Kind::I32, Kind::I32, Kind::I32, Kind::I32}, 4, 1,
        [](OpBuilder &b, func::CallOp call, ValueRange args) {
          int64_t lanes = 32;
          if (auto given = call->getAttrOfType<IntegerAttr>("lanes"))
            lanes = given.getInt();
          return SmallVector<Value>{emitWarpScan(
              b, call.getLoc(), args[0], args[1], args[2], args[3], lanes,
              call->hasAttr("unsigned"))};
        }};
  // block_scan(value, shared, thread) -> (before, total).
  if (name == "block_scan")
    return Primitive{{Kind::I32, Kind::SharedBuffer, Kind::Index}, 3, 2,
                     [](OpBuilder &b, func::CallOp call, ValueRange args) {
                       BlockScan scan = emitBlockScan(b, call.getLoc(),
                                                      args[0], args[1],
                                                      args[2]);
                       return SmallVector<Value>{scan.before, scan.total};
                     }};
  // device_scan(counts, start, n, sums, blocks, threads, one), and
  // device_scan_cursor with the cursor last.
  if (name == "device_scan" || name == "device_scan_cursor") {
    bool withCursor = name == "device_scan_cursor";
    SmallVector<Kind> operands = {Kind::Buffer, Kind::Buffer, Kind::Index,
                                  Kind::Buffer, Kind::Index,  Kind::Index,
                                  Kind::Index};
    if (withCursor)
      operands.push_back(Kind::Buffer);
    unsigned count = operands.size();
    return Primitive{std::move(operands), count, 0,
                     [](OpBuilder &b, func::CallOp call, ValueRange args) {
                       emitDeviceScan(b, call.getLoc(), args[0], args[1],
                                      args.size() > 7 ? args[7] : Value(),
                                      args[2], args[3], args[4], args[5],
                                      args[6]);
                       return SmallVector<Value>{};
                     }};
  }
  // The alternative single-pass strategy keeps one i64 descriptor per tile
  // and a ticket counter in the caller-owned state buffer.
  if (name == "single_pass_scan" || name == "single_pass_scan_cursor") {
    bool withCursor = name == "single_pass_scan_cursor";
    SmallVector<Kind> operands = {Kind::Buffer, Kind::Buffer, Kind::Index,
                                  Kind::WideBuffer, Kind::Index, Kind::Index,
                                  Kind::Index};
    if (withCursor)
      operands.push_back(Kind::Buffer);
    unsigned count = operands.size();
    return Primitive{std::move(operands), count, 0,
                     [](OpBuilder &b, func::CallOp call, ValueRange args) {
                       emitSinglePassScan(
                           b, call.getLoc(), args[0], args[1],
                           args.size() > 7 ? args[7] : Value(), args[2],
                           args[3], args[4], args[5], args[6]);
                       return SmallVector<Value>{};
                     }};
  }
  using K = Kind;
  // chunked_scan(counts, start, cursor, sums, n, chunks, chunk, grid,
  // block, one).
  if (name == "chunked_scan")
    return Primitive{{K::Buffer, K::Buffer, K::Buffer, K::Buffer, K::Index,
                      K::Index, K::Index, K::Index, K::Index, K::Index},
                     10, 0,
                     [](OpBuilder &b, func::CallOp call, ValueRange a) {
                       emitChunkedScan(b, call.getLoc(), a[0], a[1], a[2],
                                       a[3], a[4], a[5], a[6], a[7], a[8],
                                       a[9]);
                       return SmallVector<Value>{};
                     }};
  // cursor_scatter(keys, cursor, out, n, grid, block, one) {every_key}.
  if (name == "cursor_scatter")
    return Primitive{{K::Buffer, K::Buffer, K::Buffer, K::Index, K::Index,
                      K::Index, K::Index},
                     7, 0,
                     [](OpBuilder &b, func::CallOp call, ValueRange a) {
                       emitCursorScatter(b, call.getLoc(), a[0], a[1], a[2],
                                         a[3], a[4], a[5], a[6],
                                         call->hasAttr("every_key"));
                       return SmallVector<Value>{};
                     }};
  // sort_bins(start, items, n, grid, block, one), and sort_bins_by with
  // the buffer of the values that order the items last.
  if (name == "sort_bins" || name == "sort_bins_by") {
    SmallVector<Kind> operands = {K::Buffer, K::Buffer, K::Index,
                                  K::Index,  K::Index,  K::Index};
    if (name == "sort_bins_by")
      operands.push_back(K::Buffer);
    unsigned count = operands.size();
    return Primitive{std::move(operands), count, 0,
                     [](OpBuilder &b, func::CallOp call, ValueRange a) {
                       emitSortBins(b, call.getLoc(), a[0], a[1], a[2], a[3],
                                    a[4], a[5],
                                    a.size() > 6 ? a[6] : Value());
                       return SmallVector<Value>{};
                     }};
  }
  // rank_in_bins(items, keys, start, ranked, bins, grid, block, one).
  if (name == "rank_in_bins")
    return Primitive{{K::Buffer, K::Buffer, K::Buffer, K::Buffer, K::Index,
                      K::Index, K::Index, K::Index},
                     8, 0,
                     [](OpBuilder &b, func::CallOp call, ValueRange a) {
                       emitRankInBins(b, call.getLoc(), a[0], a[1], a[2],
                                      a[3], a[4], a[5], a[6], a[7]);
                       return SmallVector<Value>{};
                     }};
  // ragged_row(rows, row, entry size) -> (first, count): the row of a
  // buffer of ragged rows, checked (RaggedRows.h).
  if (name == "ragged_row")
    return Primitive{{K::DynamicBuffer, K::Index, K::Index},
                     3,
                     2,
                     [](OpBuilder &b, func::CallOp call, ValueRange a) {
                       RaggedRow row = emitRaggedRow(b, call.getLoc(), a[0],
                                                     a[1], a[2],
                                                     /*checked=*/true);
                       return SmallVector<Value>{row.first, row.count};
                     },
                     {K::Index, K::Index}};
  // ragged_entry(rows, first, k, column, entry size) -> i32.
  if (name == "ragged_entry")
    return Primitive{{K::DynamicBuffer, K::Index, K::Index, K::Index,
                      K::Index},
                     5, 1,
                     [](OpBuilder &b, func::CallOp call, ValueRange a) {
                       return SmallVector<Value>{emitRaggedEntry(
                           b, call.getLoc(), a[0], a[1], a[2], a[3], a[4])};
                     }};
  return std::nullopt;
}

/// Whether `call` gives `primitive` operands and results that it has; an
/// error on the call otherwise.
LogicalResult check(func::CallOp call, StringRef name,
                    const Primitive &primitive) {
  unsigned given = call.getNumOperands();
  unsigned most = primitive.operands.size();
  if (given < primitive.least || given > most)
    return call.emitOpError()
           << "gives the device primitive '" << name << "' " << given
           << " operands; it takes " << primitive.least
           << (primitive.least == most ? ""
                                       : " to " + std::to_string(most));
  for (auto [k, operand] : llvm::enumerate(call.getOperands()))
    if (!hasKind(operand.getType(), primitive.operands[k]))
      return call.emitOpError() << "gives the device primitive '" << name
                                << "' an operand " << k << " of the type "
                                << operand.getType() << ", which it does not take";
  bool resultsMatch = call.getNumResults() == primitive.results;
  for (auto [k, type] : llvm::enumerate(call.getResultTypes()))
    resultsMatch =
        resultsMatch && (primitive.resultKinds.empty()
                             ? type.isInteger(32)
                             : hasKind(type, primitive.resultKinds[k]));
  if (!resultsMatch)
    return call.emitOpError() << "declares results of the device primitive '"
                              << name << "' that it does not have";
  return success();
}

} // namespace

LogicalResult kernels::expandPrimitiveCalls(Operation *root) {
  SmallVector<func::CallOp> calls;
  root->walk([&](func::CallOp call) {
    if (call.getCallee().starts_with(kPrimitivePrefix))
      calls.push_back(call);
  });
  // Every call is checked before any is expanded, so that a module with an
  // error is left as it was.
  SmallVector<Primitive> primitives;
  for (func::CallOp call : calls) {
    StringRef name = call.getCallee().drop_front(kPrimitivePrefix.size());
    std::optional<Primitive> primitive = getPrimitive(name);
    if (!primitive)
      return call.emitOpError() << "calls an unknown device primitive '"
                                << name << "'";
    if (failed(check(call, name, *primitive)))
      return failure();
    primitives.push_back(std::move(*primitive));
  }
  for (auto [call, primitive] : llvm::zip(calls, primitives)) {
    OpBuilder builder(call);
    SmallVector<Value> results =
        primitive.expand(builder, call, call.getOperands());
    call.replaceAllUsesWith(results);
    call.erase();
  }
  // The declarations, now that nothing calls them.
  SmallVector<func::FuncOp> declarations;
  root->walk([&](func::FuncOp function) {
    if (function.getSymName().starts_with(kPrimitivePrefix))
      declarations.push_back(function);
  });
  for (func::FuncOp function : declarations) {
    if (!function.isDeclaration())
      return function.emitOpError()
             << "defines a device primitive, which the emitters give";
    function.erase();
  }
  return success();
}

namespace {
class ExpandDevicePrimitives
    : public mdir::impl::ExpandDevicePrimitivesBase<ExpandDevicePrimitives> {
public:
  void runOnOperation() final {
    if (failed(kernels::expandPrimitiveCalls(getOperation())))
      signalPassFailure();
  }
};
} // namespace
