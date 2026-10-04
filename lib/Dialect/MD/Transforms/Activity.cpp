#include "mdir/Dialect/MD/Transforms/Activity.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
using namespace mlir;
using namespace mdir::md;

ActivityAnalysis::ActivityAnalysis(Value variable) : variable(variable) {
  if (auto arg = dyn_cast<BlockArgument>(variable))
    knownBlocks.insert(arg.getOwner());
  else
    knownBlocks.insert(variable.getDefiningOp()->getBlock());
}

static auto combine = [](ActivityResult &into, const ActivityResult &from) {
    if (from.dependence == Activity::Active ||
        into.dependence == Activity::Active) {
      into = {Activity::Active, ""};
      return;
    }
    if (from.dependence == Activity::Unknown &&
        into.dependence == Activity::Inactive)
      into = from;
  };

  // Ops of a kernel whose meaning the pass knows: arithmetic without
  // effects, and the values of tables.
static auto isKernelOp = [](Operation *op) {
    if (isa<LookupOp, YieldOp>(op) ||
        op->getName().getStringRef() == "md_exec.cell_edges")
      return true;
    return isa<DerivativeOpInterface>(op) &&
           op->getNumRegions() == 0 && isMemoryEffectFree(op);
  };
  // Ops over particles, tuples, or pairs whose kernels the pass knows.
static auto isSumOp = [](Operation *op) {
    return isa<SumRelationOp, GatherRelationOp, SumTuplesOp, GatherTuplesOp,
               SumParticlesOp, MapParticlesOp>(op);
  };

ActivityResult ActivityAnalysis::classify(Value value) {
    if (value == variable)
      return {Activity::Active, ""};
    auto found = verdicts.find(value);
    if (found != verdicts.end())
      return found->second;
    ActivityResult verdict;
    Operation *op = value.getDefiningOp();
    // An explicit kernel argument cannot affect its enclosing scope.
    if (auto argument = dyn_cast<BlockArgument>(variable)) {
      if (isSumOp(argument.getOwner()->getParentOp()) && op &&
          !argument.getOwner()->getParent()->isAncestor(op->getParentRegion()))
        return {Activity::Inactive, ""};
    }
    if (!op) {
      Block *owner = cast<BlockArgument>(value).getOwner();
      if (!knownBlocks.contains(owner) && !isSumOp(owner->getParentOp()) &&
          !isa<PotentialOp, FunctionOp>(owner->getParentOp()))
        verdict = {Activity::Unknown,
                   "it is an argument of a block whose meaning the pass does "
                   "not know"};
    } else {
      bool known = isKernelOp(op) || isSumOp(op) ||
                   (op->getName().getDialectNamespace() == "md" &&
                    op->getNumRegions() == 0 && isMemoryEffectFree(op));
      // An operand that depends on the variable makes the op dependent,
      // known or not.
      for (Value operand : op->getOperands())
        combine(verdict, classify(operand));
      if (!known) {
        combine(verdict,
                {Activity::Unknown,
                 ("'" + op->getName().getStringRef() +
                  "' is not an op whose dependences the pass knows")
                     .str()});
      } else {
        // What the kernels take from outside.
        op->walk([&](Operation *inner) {
          if (inner == op)
            return WalkResult::advance();
          if (!isKernelOp(inner)) {
            combine(verdict,
                    {Activity::Unknown,
                     ("its kernel holds '" + inner->getName().getStringRef() +
                      "', whose dependences the pass does not know")
                         .str()});
            return WalkResult::advance();
          }
          for (Value operand : inner->getOperands()) {
            Operation *definition = operand.getDefiningOp();
            bool outside =
                definition ? !op->isAncestor(definition)
                           : !op->isAncestor(
                                 cast<BlockArgument>(operand).getOwner()
                                     ->getParentOp());
            if (outside)
              combine(verdict, classify(operand));
          }
          return WalkResult::advance();
        });
      }
    }
    verdicts[value] = verdict;
    return verdict;
}

#include "mdir/Dialect/MD/Transforms/Passes.h"
namespace mdir { namespace md {
#define GEN_PASS_DEF_ANALYZEACTIVITY
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"
namespace {
class AnalyzeActivity : public impl::AnalyzeActivityBase<AnalyzeActivity> {
public:
  using impl::AnalyzeActivityBase<AnalyzeActivity>::AnalyzeActivityBase;
  void runOnOperation() final {
    bool invalid = false;
    getOperation()->walk([&](PotentialOp potential) {
      if (potential.isExternal()) return;
      Block &body = potential.getBody().front();
      if (argument < 0 || static_cast<unsigned>(argument) >= body.getNumArguments()) {
        potential.emitError("activity argument is out of range");
        invalid = true;
        return;
      }
      ActivityAnalysis analysis(body.getArgument(argument));
      for (Operation &op : body) for (Value result : op.getResults()) {
        auto verdict = analysis.classify(result);
        StringRef name = verdict.dependence == Activity::Active ? "active" :
            verdict.dependence == Activity::Inactive ? "proven inactive" : "unknown";
        op.emitRemark() << "activity: " << name
                        << (verdict.reason.empty() ? "" : ": " + verdict.reason);
      }
    });
    if (invalid) signalPassFailure();
  }
};
} }
}
