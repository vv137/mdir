// Inlining of potentials, functions, and programs, and of the bodies of
// neural stages into the kernels that apply them (D288).

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

    // The bodies of the stages, into the kernels: the forward body, or with
    // `vjp` its vector-Jacobian product. They are functions of one element,
    // so that a kernel holds them as it holds any arithmetic.
    SmallVector<md::ApplyStageOp> applications;
    module->walk([&](md::ApplyStageOp op) { applications.push_back(op); });
    for (md::ApplyStageOp apply : applications) {
      auto stage = SymbolTable::lookupNearestSymbolFrom<md::StageOp>(
          apply, apply.getStageAttr());
      if (!stage) {
        apply.emitOpError() << "'" << apply.getStage()
                            << "' does not name a stage";
        return signalPassFailure();
      }
      // A batched body is called by the tiles of the loop that the kernel
      // becomes (D293); the application stays.
      if (stage.getRows())
        continue;
      FlatSymbolRefAttr name =
          apply.getVjp() ? stage.getVjpAttr() : stage.getForwardAttr();
      auto body = dyn_cast_or_null<FunctionOpInterface>(
          name ? SymbolTable::lookupNearestSymbolFrom(stage, name)
               : nullptr);
      if (!body) {
        apply.emitOpError() << "cannot be inlined: the stage '"
                            << apply.getStage() << "' binds no such body";
        return signalPassFailure();
      }
      if (failed(inlineCall(apply, body)))
        return signalPassFailure();
    }

    // Remove what is no longer referenced. A potential that an
    // `md.evaluate` still names stays. The stages go first, so that their
    // bodies are no longer referenced.
    SmallVector<md::StageOp> stages(module.getOps<md::StageOp>());
    for (md::StageOp stage : stages) {
      if (!SymbolTable::symbolKnownUseEmpty(stage, module))
        continue;
      SmallVector<FlatSymbolRefAttr, 2> bodies = {stage.getForwardAttr()};
      if (stage.getVjpAttr())
        bodies.push_back(stage.getVjpAttr());
      stage.erase();
      for (FlatSymbolRefAttr name : bodies) {
        Operation *body = SymbolTable::lookupSymbolIn(module, name);
        if (body && SymbolTable::symbolKnownUseEmpty(body, module))
          body->erase();
      }
    }
    for (Operation &op : llvm::make_early_inc_range(module))
      if (isCallee(&op) && SymbolTable::symbolKnownUseEmpty(&op, module))
        op.erase();
  }
};
} // namespace

} // namespace md
} // namespace mdir
