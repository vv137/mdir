// Readiness and lifetime checks for the existing GPU copy sequence.

#include "mdir/Dialect/MDExec/Transfers.h"

#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MDExec/Independence.h"
#include "mdir/Dialect/MDExec/LoopEffects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace {

struct ByteRange {
  int64_t begin, end;
};

/// The SSA view is the exact slice, including dynamic descriptors. A static
/// byte envelope can prove separation; an unknown envelope cannot. The root
/// SSA value identifies the allocation generation within this activation
/// or loop iteration. Opaque roots are never presumed distinct.
struct Storage {
  Value slice;
  Value root;
  Value generation;
  std::optional<ByteRange> bytes;
  bool unresolvedCallLoop = false;
};

enum class Visibility { Host, Device };

struct Transfer {
  gpu::MemcpyOp copy;
  Value token;
  Storage source, destination;
  Visibility visibility;
  gpu::WaitOp completion;
};

static std::optional<ByteRange> byteRange(MemRefType type) {
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return std::nullopt;
  unsigned width = type.getElementTypeBitWidth();
  if (width == 0 || width % 8)
    return std::nullopt;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset) || offset < 0)
    return std::nullopt;
  int64_t last = offset;
  for (auto [size, stride] : llvm::zip(type.getShape(), strides)) {
    if (size == 0)
      return ByteRange{0, 0};
    int64_t extent;
    if (ShapedType::isDynamic(stride) || stride < 0 ||
        __builtin_mul_overflow(size - 1, stride, &extent) ||
        __builtin_add_overflow(last, extent, &last))
      return std::nullopt;
  }
  int64_t begin, end;
  if (__builtin_add_overflow(last, int64_t(1), &last) ||
      __builtin_mul_overflow(offset, int64_t(width / 8), &begin) ||
      __builtin_mul_overflow(last, int64_t(width / 8), &end))
    return std::nullopt;
  return ByteRange{begin, end};
}

using Substitutions = llvm::DenseMap<Value, Value>;

static Value substitute(Value value, const Substitutions &arguments) {
  // Call substitutions point towards the caller, never back into the callee.
  auto found = arguments.find(value);
  while (found != arguments.end()) {
    value = found->second;
    found = arguments.find(value);
  }
  return value;
}

static Storage storage(Value value, const Substitutions &arguments) {
  value = substitute(value, arguments);
  Value descriptor = value, root = value;
  bool exactEnvelope = true;
  while (Operation *op = root.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::CastOp>(op)) {
      root = substitute(cast.getSource(), arguments);
      if (descriptor == cast.getResult())
        descriptor = root;
    } else if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(op)) {
      root = substitute(cast.getSource(), arguments);
      if (descriptor == cast.getResult())
        descriptor = root;
    } else if (auto view = dyn_cast<ViewLikeOpInterface>(op)) {
      // A subview's inferred static layout is relative to the same base.
      // Reinterpretation and byte-shifted views need a separate range proof.
      exactEnvelope &= isa<memref::SubViewOp>(op);
      root = substitute(view.getViewSource(), arguments);
    } else {
      break;
    }
  }
  auto type = dyn_cast<MemRefType>(descriptor.getType());
  // BufferAliases can follow loop permutations in their own function.
  // Across a call its roots would still be callee arguments, not the
  // substituted caller buffers. Do not use that as a separation proof.
  bool callLoop = false;
  if (!arguments.empty()) {
    if (auto argument = dyn_cast<BlockArgument>(root))
      callLoop = isa<LoopLikeOpInterface>(argument.getOwner()->getParentOp());
    else if (Operation *op = root.getDefiningOp())
      callLoop = isa<LoopLikeOpInterface>(op);
  }
  return {value, root, root,
          exactEnvelope && type ? byteRange(type) : std::nullopt, callLoop};
}

enum class Overlap { No, Yes, Unknown };

class Checker {
public:
  LogicalResult check(gpu::MemcpyOp copy);

private:
  Overlap overlap(const Storage &a, const Storage &b, bool wholeRoot);
  LogicalResult access(Operation *op, Value value, bool writes, bool frees,
                       const Transfer &transfer,
                       const Substitutions &arguments);
  LogicalResult operation(Operation *op, const Transfer &transfer,
                          const Substitutions &arguments);
  LogicalResult unsupported(Operation *op, const Transfer &transfer,
                            StringRef reason);
  BufferAliases aliases;
  llvm::DenseSet<Operation *> calls;
};

Overlap Checker::overlap(const Storage &a, const Storage &b, bool wholeRoot) {
  if (a.root == b.root) {
    if (!wholeRoot && a.bytes && b.bytes &&
        (a.bytes->end <= b.bytes->begin || b.bytes->end <= a.bytes->begin))
      return Overlap::No;
    if (wholeRoot || a.slice == b.slice || b.slice == b.root)
      return Overlap::Yes;
    return Overlap::Unknown;
  }
  if (a.unresolvedCallLoop || b.unresolvedCallLoop)
    return Overlap::Unknown;
  if (!aliases.mayAlias(a.root, b.root))
    return Overlap::No;
  return Overlap::Unknown;
}

LogicalResult Checker::unsupported(Operation *op, const Transfer &transfer,
                                   StringRef reason) {
  auto diagnostic = op->emitOpError()
                    << "unsupported transfer contract: " << reason;
  diagnostic.attachNote(transfer.copy->getLoc()) << "copy is still in flight";
  return failure();
}

LogicalResult Checker::access(Operation *op, Value value, bool writes,
                              bool frees, const Transfer &transfer,
                              const Substitutions &arguments) {
  Storage touched = storage(value, arguments);
  for (auto [borrow, name] :
       {std::pair<const Storage *, StringRef>{&transfer.destination, "destination"},
        {&transfer.source, "source"}}) {
    // Reading the source does not conflict with its read borrow.
    if (name == "source" && !writes && !frees)
      continue;
    Overlap relation = overlap(touched, *borrow, frees);
    if (relation == Overlap::No)
      continue;
    if (relation == Overlap::Unknown)
      return unsupported(op, transfer,
                         "cannot prove storage roots or slices disjoint");
    auto diagnostic = op->emitOpError()
                      << "transfer " << name << " "
                      << (frees ? "freed or reused" : writes ? "written" : "read")
                      << " before copy completion";
    diagnostic.attachNote(transfer.copy->getLoc()) << "copy starts here";
    return failure();
  }
  return success();
}

LogicalResult Checker::operation(Operation *op, const Transfer &transfer,
                                 const Substitutions &arguments) {
  if (auto wait = dyn_cast<gpu::WaitOp>(op))
    return success(); // Only check() decides whether this is completion.
  if (auto copy = dyn_cast<gpu::MemcpyOp>(op)) {
    if (failed(access(op, copy.getSrc(), false, false, transfer, arguments)))
      return failure();
    return access(op, copy.getDst(), true, false, transfer, arguments);
  }

  if (auto call = dyn_cast<func::CallOp>(op)) {
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!function)
      return unsupported(op, transfer, "unresolved function effects");
    if (function.isExternal()) {
      auto summary = function->getAttrOfType<DictionaryAttr>("dyn.effects");
      SmallVector<MemoryEffects::EffectInstance> effects;
      std::string error;
      if (!summary || failed(dyn::parseEffectSummary(summary, effects, error)))
        return unsupported(op, transfer, "unknown declared function effects");
      // D355's named resources are distinct from buffer storage. A default
      // resource effect, however, gives no footprint for this borrow.
      for (const auto &effect : effects)
        if (effect.getResource() == SideEffects::DefaultResource::get())
          return unsupported(op, transfer,
                             "function effects have no buffer footprint");
      return success();
    }
    if (function->isAncestor(transfer.copy) || !calls.insert(function).second)
      return unsupported(op, transfer, "recursive function effects");
    if (!llvm::hasSingleElement(function.getBody())) {
      calls.erase(function);
      return unsupported(op, transfer, "unrecognized function control flow");
    }
    Substitutions mapped = arguments;
    for (auto [argument, operand] :
         llvm::zip(function.getArguments(), call.getOperands()))
      mapped[argument] = substitute(operand, arguments);
    for (Operation &nested : function.getBody().front())
      if (failed(operation(&nested, transfer, mapped))) {
        calls.erase(function);
        return failure();
      }
    calls.erase(function);
    return success();
  }

  if (auto loop = dyn_cast<LoopEffectOpInterface>(op)) {
    LoopEffectSummary summary = summarizeLoop(op);
    if (!summary.complete)
      return unsupported(op, transfer, "incomplete loop effect summary");
    for (const analysis::Access &effect : summary.accesses)
      if (failed(access(op, effect.operand->get(),
                        effect.effect != analysis::Effect::Read, false,
                        transfer, arguments)))
        return failure();
    return success();
  }

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (memory) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);
    for (const auto &effect : effects) {
      // A fresh allocation cannot alias an already-live generation. An
      // attempt to free the old generation is checked separately.
      if (isa<MemoryEffects::Allocate>(effect.getEffect()))
        continue;
      if (effect.getResource() != SideEffects::DefaultResource::get())
        continue;
      Value value = effect.getValue();
      if (!value)
        return unsupported(op, transfer, "memory effect has no storage subject");
      bool frees = isa<MemoryEffects::Free>(effect.getEffect());
      bool writes = isa<MemoryEffects::Write>(effect.getEffect());
      if (!frees && !writes && !isa<MemoryEffects::Read>(effect.getEffect()))
        return unsupported(op, transfer, "unrecognized memory effect");
      if (failed(access(op, value, writes, frees, transfer, arguments)))
        return failure();
    }
    if (op->getNumRegions() == 0)
      return success();
  }
  if (memory || op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      if (!llvm::hasSingleElement(region))
        return unsupported(op, transfer, "unrecognized region control flow");
      for (Operation &nested : region.front())
        if (failed(operation(&nested, transfer, arguments)))
          return failure();
    }
    return success();
  }
  if (isMemoryEffectFree(op))
    return success();
  return unsupported(op, transfer, "unknown operation effects");
}

LogicalResult Checker::check(gpu::MemcpyOp copy) {
  Substitutions arguments;
  auto destinationType = cast<MemRefType>(copy.getDst().getType());
  auto space = dyn_cast_or_null<IntegerAttr>(destinationType.getMemorySpace());
  if (destinationType.getMemorySpace() && !space)
    return copy.emitOpError()
           << "unsupported transfer contract: unknown visibility domain";
  Transfer transfer{copy, copy.getAsyncToken(), storage(copy.getSrc(), arguments),
                    storage(copy.getDst(), arguments),
                    space && space.getInt() != 0 ? Visibility::Device
                                                : Visibility::Host,
                    {}};
  // A synchronous copy completes as part of the operation itself.
  if (!transfer.token)
    return success();

  llvm::DenseSet<Value> completionTokens{transfer.token};
  auto covers = [&](ValueRange dependencies) {
    return llvm::any_of(dependencies, [&](Value token) {
      return completionTokens.contains(token);
    });
  };
  for (Operation *op = copy->getNextNode(); op; op = op->getNextNode()) {
    if (auto wait = dyn_cast<gpu::WaitOp>(op)) {
      if (covers(wait.getAsyncDependencies())) {
        if (Value token = wait.getAsyncToken())
          completionTokens.insert(token);
        else {
          transfer.completion = wait;
          return success();
        }
      }
    }
    if (auto next = dyn_cast<gpu::MemcpyOp>(op)) {
      if (next.getAsyncToken() && covers(next.getAsyncDependencies())) {
        // This copy's explicit dependency completes the earlier copy
        // before its own memory accesses. Its token also carries that
        // completion to a later blocking wait. Its own borrows are checked
        // independently; no host access is licensed by this async edge.
        completionTokens.insert(next.getAsyncToken());
        continue;
      }
    }
    if (failed(operation(op, transfer, arguments)))
      return failure();
  }
  return copy.emitOpError()
         << "unsupported transfer contract: no recognized same-block blocking "
            "completion; token dominance is not completion";
}

} // namespace

LogicalResult mdir::md_exec::verifyTransfers(Operation *root) {
  Checker checker;
  WalkResult result = root->walk([&](gpu::MemcpyOp copy) {
    return failed(checker.check(copy)) ? WalkResult::interrupt()
                                       : WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}
