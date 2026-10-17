#include "mdir/Dialect/MD/Transforms/Activity.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
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
    if (isa<LookupOp, YieldOp, mdir::md_dist::YieldOp>(op) ||
        op->getName().getStringRef() == "md_exec.cell_edges")
      return true;
    return isa<DerivativeOpInterface>(op) &&
           op->getNumRegions() == 0 && isMemoryEffectFree(op);
  };
  // Ops over particles, tuples, or pairs whose kernels the pass knows.
static auto isSumOp = [](Operation *op) {
    return isa<SumRelationOp, GatherRelationOp, SumTuplesOp, GatherTuplesOp,
               SumParticlesOp, MapParticlesOp, mdir::md_dist::SumPairsOp,
               mdir::md_dist::GatherPairsOp, mdir::md_dist::SumParticlesOp,
               mdir::md_dist::MapParticlesOp>(op);
  };

  // A value of md_dist that states structure: a layout, a halo, a relation.
  // Each is a piecewise constant function of the configuration (which
  // domain owns a particle, which images are ghosts, which pairs are within
  // the cutoff), whose derivative is zero almost everywhere, as that of the
  // relation of md.neighborhood is. A value that depends on the variable
  // only through them is inactive (D[ml2-distribute]).
static auto isDistributed = [](Operation *op) {
    return isa_and_nonnull<mdir::md_dist::MDDistDialect>(op->getDialect());
  };
static auto isStructural = [](Type type) {
    return isa<mdir::md_dist::LayoutType, mdir::md_dist::HaloType,
               mdir::md_dist::RelationType>(type);
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
    } else if (isDistributed(op) && isStructural(value.getType())) {
      // Inactive: structure, not a value.
    } else {
      bool distributed = isDistributed(op);
      bool known = isKernelOp(op) || isSumOp(op) ||
                   ((op->getName().getDialectNamespace() == "md" ||
                     distributed) &&
                    op->getNumRegions() == 0 && isMemoryEffectFree(op));
      // An operand that depends on the variable makes the op dependent,
      // known or not; the structure of md_dist does not.
      for (Value operand : op->getOperands())
        if (!distributed || !isStructural(operand.getType()))
          combine(verdict, classify(operand));
      if (!known) {
        combine(verdict,
                {Activity::Unknown,
                 ("'" + op->getName().getStringRef() +
                  "' is not an op whose dependences the pass knows")
                     .str(), op});
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
                         .str(), inner});
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
