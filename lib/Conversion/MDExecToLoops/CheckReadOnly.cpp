// Proves that a function writes none of the buffers that it declares
// read-only (D[start-arguments], #326).
//
// The entry of a program declares the buffers of the model read-only, and
// a Python simulation hands it the same buffers at every start of an
// activation instead of copies. The pass follows every use of each such
// buffer in the lowered module and accepts only reads; anything it cannot
// prove a read fails the compile, so that a buffer that is proven read-only
// and one that is not are never confused.

#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mdir {

#define GEN_PASS_DEF_CHECKREADONLYARGUMENTS
#include "mdir/Conversion/Passes.h.inc"

namespace {

/// The proof for one argument: every value that may be the buffer of the
/// argument or a view of it, and every use of each.
class ReadOnlyProof {
public:
  ReadOnlyProof(func::FuncOp function, unsigned index)
      : function(function), index(index) {}

  LogicalResult run() {
    add(function.getArgument(index));
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      for (OpOperand &use : value.getUses())
        if (failed(check(use)))
          return failure();
    }
    return success();
  }

private:
  void add(Value value) {
    if (seen.insert(value).second)
      pending.push_back(value);
  }

  InFlightDiagnostic fail(Operation *op, const Twine &what) {
    InFlightDiagnostic diagnostic = op->emitError();
    diagnostic << "the argument " << index << " of '" << function.getName()
               << "' is declared read-only ('"
               << mdrt::getReadOnlyAttrName() << "'), but " << what;
    return diagnostic;
  }

  LogicalResult check(OpOperand &use) {
    Operation *user = use.getOwner();
    Value value = use.get();

    // A view of the buffer is the buffer.
    if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
      if (view.getViewSource() == value) {
        for (Value result : user->getResults())
          if (isa<BaseMemRefType>(result.getType()))
            add(result);
        return success();
      }
    }

    // A call: into the function of the module, or of a function of the
    // runtime that declares that it only reads the parameter.
    if (auto call = dyn_cast<CallOpInterface>(user)) {
      auto symbol = llvm::dyn_cast_if_present<SymbolRefAttr>(
          call.getCallableForCallee());
      auto callee = symbol ? SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                                 user, symbol)
                           : func::FuncOp();
      if (!callee)
        return fail(user, "it is passed to a call whose callee is not known");
      OperandRange arguments = call.getArgOperands();
      unsigned first = arguments.getBeginOperandIndex();
      unsigned number = use.getOperandNumber();
      if (number < first || number >= first + arguments.size())
        return fail(user, "it is an operand of a call that is not an argument");
      unsigned parameter = number - first;
      if (callee.isExternal()) {
        if (!callee.getArgAttr(parameter, mdrt::getReadOnlyAttrName()))
          return fail(user, "it is passed to '" + callee.getName() +
                                "', which does not declare its parameter " +
                                Twine(parameter) + " read-only");
        return success();
      }
      add(callee.getArgument(parameter));
      return success();
    }

    // An op whose effects on the buffer are known: reads alone.
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
      SmallVector<MemoryEffects::EffectInstance> found;
      effects.getEffectsOnValue(value, found);
      for (const MemoryEffects::EffectInstance &effect : found)
        if (!isa<MemoryEffects::Read>(effect.getEffect()))
          return fail(user, "it may be written or released here");
      if (!found.empty())
        return success();
      // No effect on the buffer: an op that reads its shape (memref.dim)
      // and gives no buffer or address out of it.
      if (isa<memref::DimOp, memref::RankOp>(user))
        return success();
      return fail(user, "the op takes it without saying how");
    }
    return fail(user, "the op takes it and its memory effects are not known");
  }

  func::FuncOp function;
  unsigned index;
  SmallVector<Value> pending;
  llvm::DenseSet<Value> seen;
};

class CheckReadOnlyArguments
    : public impl::CheckReadOnlyArgumentsBase<CheckReadOnlyArguments> {
public:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool failed = false;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      for (unsigned i = 0, e = function.getNumArguments(); i != e; ++i) {
        if (!function.getArgAttr(i, mdrt::getReadOnlyAttrName()))
          continue;
        if (!isa<BaseMemRefType>(function.getArgument(i).getType())) {
          function.emitError()
              << "declares its argument " << i
              << " read-only, which is not a buffer";
          failed = true;
          continue;
        }
        if (mlir::failed(ReadOnlyProof(function, i).run()))
          failed = true;
      }
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace
} // namespace mdir
