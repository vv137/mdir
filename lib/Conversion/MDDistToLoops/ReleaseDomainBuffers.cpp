// The release of the buffers that a compiled domain procedure allocates
// (D[distribute-pair-lowering]).
//
// `convert-md-dist-to-loops{matrix-pairs=true}` outlines the body of each
// loop over domains into a procedure of its own, which the driver calls
// once per domain and step. The procedure packs its local rows into new
// buffers, and the storage of md_exec gives its fields buffers of the
// procedure's own scope; neither is freed there, since a function's
// storage otherwise lives as long as the run. Called at every step, each
// call kept them all. See the description of the pass in Passes.td.

#include "mdir/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace mdir {
#define GEN_PASS_DEF_MDDISTRELEASEDOMAINBUFFERS
#include "mdir/Conversion/Passes.h.inc"

namespace {

/// The external functions of the runtime that a domain procedure gives a
/// buffer it allocated, which use it only during the call:
/// mdrtDistMatrixRebuild (runtime/mdrt_dist.c) compares the positions with
/// the reference positions of its matrix and copies them there. The midpoint
/// filter reads current positions while emitting selected row indices.
/// Pair image quantization reads those positions into a separate buffer.
static bool usesOnlyDuringCall(StringRef name) {
  return name == "mdrtDistMatrixRebuild" || name == "mdrtDistBuildMidpoint" ||
         name == "mdrtDistFilterMatrix" || name == "mdrtDistQuantizePairImages";
}

class Release {
public:
  explicit Release(ModuleOp module) : symbols(module) {}

  /// Whether every use of `buffer` (and of its views) ends with the
  /// function: no return, yield, or call that may keep it. Otherwise
  /// reports why at the use.
  LogicalResult staysWithin(Value buffer) {
    SmallVector<Value> worklist{buffer};
    llvm::SmallPtrSet<Value, 8> seen{buffer};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
          if (view.getViewSource() == value)
            for (Value result : user->getResults())
              if (seen.insert(result).second)
                worklist.push_back(result);
          continue;
        }
        if (auto call = dyn_cast<func::CallOp>(user)) {
          auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
          if (callee && !callee.isExternal()) {
            // A function of the module: the same rules for its argument.
            Value argument =
                callee.getArgument(use.getOperandNumber());
            if (!visited.insert(argument).second)
              continue;
            if (failed(staysWithin(argument)))
              return failure();
            continue;
          }
          if (usesOnlyDuringCall(call.getCallee()))
            continue;
          return user->emitOpError()
                 << "is given a buffer that a domain procedure allocates, "
                    "and may keep it past the procedure's release of it";
        }
        if (user->hasTrait<OpTrait::IsTerminator>() ||
            isa<RegionBranchOpInterface>(user))
          return user->emitOpError()
                 << "carries a buffer that a domain procedure allocates "
                    "out of the block that releases it";
        // A raw pointer, or another memref that may alias it.
        if (isa<memref::ExtractAlignedPointerAsIndexOp>(user) ||
            llvm::any_of(user->getResultTypes(),
                         [](Type type) { return isa<BaseMemRefType>(type); }))
          return user->emitOpError()
                 << "takes an alias of a buffer that a domain procedure "
                    "allocates, which its release cannot follow";
        // Loads, stores, copies, and the queries of its extent use the
        // buffer in place.
      }
    }
    return success();
  }

  static bool isReleased(Value buffer) {
    SmallVector<Value> worklist{buffer};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      for (Operation *user : value.getUsers()) {
        if (isa<memref::DeallocOp>(user))
          return true;
        if (auto view = dyn_cast<ViewLikeOpInterface>(user))
          if (view.getViewSource() == value)
            llvm::append_range(worklist, user->getResults());
      }
    }
    return false;
  }

  SymbolTable symbols;
  llvm::SmallPtrSet<Value, 8> visited;
};

class ReleaseDomainBuffers
    : public impl::MDDistReleaseDomainBuffersBase<ReleaseDomainBuffers> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    Release release(module);
    SmallVector<memref::AllocOp> allocations;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!function->hasAttr("md_dist.domain_procedure"))
        continue;
      function->removeAttr("md_dist.domain_procedure");
      function.walk([&](memref::AllocOp alloc) {
        if (!Release::isReleased(alloc.getMemref()))
          allocations.push_back(alloc);
      });
    }
    for (memref::AllocOp alloc : allocations) {
      release.visited.clear();
      if (failed(release.staysWithin(alloc.getMemref())))
        return signalPassFailure();
      // The end of the block of the allocation is after its last use:
      // nothing carries the buffer out of the block.
      OpBuilder b(alloc->getBlock()->getTerminator());
      memref::DeallocOp::create(b, alloc.getLoc(), alloc.getMemref());
    }
  }
};
} // namespace
} // namespace mdir
