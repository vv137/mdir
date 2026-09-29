// Inlining of potentials, functions, and programs.

#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mdir/Dialect/Dyn/DynOps.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mdir;

/// Returns true if `op` calls a potential, a function, or a program.
static bool isCall(Operation *op) { return isa<md::CallOp, dyn::StepOp>(op); }

/// Returns true if `op` defines a potential, a function, or a program.
static bool isCallee(Operation *op) {
  return isa<md::PotentialOp, md::FunctionOp, dyn::ProgramOp>(op);
}

/// Replaces `call` with the body of `callee`.
static LogicalResult inlineCall(Operation *call, FunctionOpInterface callee) {
  if (callee.isExternal())
    return call->emitOpError() << "cannot inline a declaration";
  Region &body = callee.getFunctionBody();
  if (!llvm::hasSingleElement(body))
    return call->emitOpError()
           << "cannot inline a body with more than one block";
  Block &block = body.front();

  IRMapping mapping;
  for (unsigned i = 0, e = block.getNumArguments(); i != e; ++i)
    mapping.map(block.getArgument(i), call->getOperand(i));

  OpBuilder builder(call);
  for (Operation &op : block.without_terminator())
    builder.clone(op, mapping);

  Operation *terminator = block.getTerminator();
  for (unsigned i = 0, e = call->getNumResults(); i != e; ++i)
    call->getResult(i).replaceAllUsesWith(
        mapping.lookupOrDefault(terminator->getOperand(i)));
  call->erase();
  return success();
}

namespace mdir {
namespace md {

#define GEN_PASS_DEF_INLINE
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {
class Inline : public impl::InlineBase<Inline> {
public:
  using impl::InlineBase<Inline>::InlineBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // A body that is inlined may contain calls itself. The depth of the
    // calls bounds the number of rounds; recursion would not terminate.
    const unsigned maxRounds = 64;
    for (unsigned round = 0;; ++round) {
      SmallVector<Operation *> calls;
      module->walk([&](Operation *op) {
        if (isCall(op))
          calls.push_back(op);
      });
      if (calls.empty())
        break;
      if (round == maxRounds) {
        calls.front()->emitOpError()
            << "cannot be inlined: the calls are nested more than "
            << maxRounds << " deep or are recursive";
        return signalPassFailure();
      }

      for (Operation *call : calls) {
        auto name = call->getAttrOfType<FlatSymbolRefAttr>("callee");
        Operation *callee = SymbolTable::lookupNearestSymbolFrom(call, name);
        if (!callee || !isCallee(callee)) {
          call->emitOpError() << "'" << name.getValue()
                              << "' does not name a potential, a function, "
                                 "or a program";
          return signalPassFailure();
        }
        if (failed(inlineCall(call, cast<FunctionOpInterface>(callee))))
          return signalPassFailure();
      }
    }

    // Remove what is no longer referenced. A potential that an
    // `md.evaluate` still names stays.
    for (Operation &op : llvm::make_early_inc_range(module))
      if (isCallee(&op) && SymbolTable::symbolKnownUseEmpty(&op, module))
        op.erase();
  }
};
} // namespace

} // namespace md
} // namespace mdir
