// The strategy of each scatter-accumulate on a device.
//
// See the description of the pass in Passes.td, D285 in
// docs/decisions.md, and docs/device-primitives.md. The preconditions are
// those of include/mdir/Dialect/MDExec/Scatter.h, which the lowering checks
// with the same functions; what this pass adds is the cost: which of the
// strategies that apply to take.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/DevicePolicy.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDExec/Scatter.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_CHOOSESCATTER
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class ChooseScatter : public impl::ChooseScatterBase<ChooseScatter> {
public:
  using impl::ChooseScatterBase<ChooseScatter>::ChooseScatterBase;

  void runOnOperation() final {
    Operation *module = getOperation();
    auto isSwitch = [](StringRef value) {
      return value == "auto" || value == "on" || value == "off";
    };
    if (!isSwitch(tile) || !isSwitch(hash) || !isSwitch(pairHash) ||
        !isSwitch(scatterHash)) {
      module->emitError() << "expected 'auto', 'on', or 'off' for 'tile', "
                             "'hash', 'pair-hash', and 'scatter-hash', got '"
                          << static_cast<std::string>(tile) << "', '"
                          << static_cast<std::string>(hash) << "', '"
                          << static_cast<std::string>(pairHash) << "', and '"
                          << static_cast<std::string>(scatterHash) << "'";
      return signalPassFailure();
    }
    // The values that the options leave to the policy of the
    // architecture (-1), from its entries (D420).
    policy = policy::getDevicePolicy(arch);
    auto chosen = [](int64_t option, int64_t entry) {
      return option == -1 ? entry : option;
    };
    edge = chosen(tileEdge, policy.scatter.tileEdge);
    minPoints = chosen(tileMinPoints, policy.scatter.tileMinPoints);
    least = chosen(hashLeast, policy.scatter.hashLeast);
    pairLeast = chosen(pairHashLeast, policy.scatter.pairHashLeast);
    probes = chosen(hashProbes, policy.scatter.hashProbes);
    block = chosen(blockSize, policy.launch.blockSize);
    budget = policy.launch.blockMemoryBytes;
    if (pairLeast < 0) {
      module->emitError() << "expected pair-hash-least of 0 or more";
      return signalPassFailure();
    }
    if (!scatter::isTileEdge(edge)) {
      module->emitError()
          << "expected an edge of the tiles of 4, 8, or 16 points, got "
          << edge;
      return signalPassFailure();
    }
    if (probes < 1 || least < 0 || block < 1) {
      module->emitError() << "expected hash-probes of at least 1, hash-least "
                             "of 0 or more, and a block-size of at least 1";
      return signalPassFailure();
    }
    module->walk([&](Operation *op) {
      if (auto sum = dyn_cast<ReciprocalOp>(op))
        chooseTile(sum);
      else if (auto loop = dyn_cast<TupleForOp>(op))
        chooseHash(loop);
      else if (auto pairs = dyn_cast<PairForOp>(op))
        choosePairHash(pairs);
      else if (auto add = dyn_cast<ScatterAddOp>(op))
        chooseScatterHash(add);
    });
  }

private:
  /// The entries of the policies for `arch`, and the values that the pass
  /// takes: the options, or the entries where an option is -1.
  policy::DevicePolicy policy;
  int64_t edge = 0, minPoints = 0, least = 0, pairLeast = 0, probes = 0,
          block = 0, budget = 0;

  /// `fromPolicy`: whether the choice took a value of the scatter policy
  /// that no option gave, so that the remark says where a default entry
  /// was not measured on the architecture.
  void explainChosen(Operation *op, ScatterAttr scatter, bool fromPolicy) {
    if (!explain)
      return;
    std::string text;
    llvm::raw_string_ostream os(text);
    scatter.printBody(os);
    op->emitRemark() << "scatter: " << text
                     << (fromPolicy
                             ? policy.describeFallback(policy.scatterFallback)
                             : "");
  }
  void explainRefused(Operation *op, StringRef strategy, StringRef why) {
    if (explain)
      op->emitRemark() << "scatter: no " << strategy << ": " << why;
  }

  /// The tiles of the spreading of PME (D282).
  void chooseTile(ReciprocalOp op) {
    // The storage form has its scratch already, without the bins.
    if (op.isStorageForm())
      return;
    if (tile == "off")
      return explainRefused(op, "tile", "tile=off");
    if (scatter::Refusal refused =
            scatter::checkTile(op, edge, budget))
      return explainRefused(op, "tile", *refused);
    // With `auto`, a grid in f64 keeps the spreading of before, which is
    // faster there on the devices measured (their f64 is slow, and the
    // atomics of a tile are a loop of compare-and-swap of 64 bits); so does
    // a grid of fewer than `tile-min-points` points, whose bins are too few
    // to occupy the device (on JAC, 512 bins of 8 points, the tiles were
    // slower on an RTX PRO 6000 at every order). Every other spreading
    // takes the tiles that fit.
    if (tile == "auto") {
      auto forces = cast<md::FieldType>(op.getForces().getType());
      if (!forces.getElementType().isF32())
        return explainRefused(op, "tile",
                              "the grid is in f64, which spreads faster "
                              "without tiles (tile=auto)");
      int64_t points = 1;
      for (int64_t length : op.getGrid())
        points *= length;
      if (points < minPoints)
        return explainRefused(
            op, "tile",
            "the grid has " + std::to_string(points) +
                " points, fewer than tile-min-points, too few bins to "
                "occupy a device (tile=auto)" +
                (tileMinPoints == -1
                     ? policy.describeFallback(policy.scatterFallback)
                     : ""));
    }
    ScatterAttr chosen = ScatterAttr::getTile(&getContext(), edge);
    op.setScatterAttr(chosen);
    explainChosen(op, chosen,
                  tileEdge == -1 || (tile == "auto" && tileMinPoints == -1));
  }

  /// The tables of the tuples evaluated once (D283).
  void chooseHash(TupleForOp op) {
    if (op.getOuts().empty())
      return;
    if (hash == "off")
      return explainRefused(op, "hash", "hash=off");
    // The loops whose kernel evaluates a tuple a thread and adds to the
    // members with atomics: those that may share particles and have no
    // global sums (a loop with global sums takes several tuples a thread,
    // for the rows of its sums, D103, and keeps the atomics). Whether a
    // loop overwrites a destination is known in the storage form only;
    // where it does, the lowering writes rows, and the attribute is not
    // used.
    if (op.getDisjoint())
      return explainRefused(op, "hash",
                            "the tuples share no particle, and each is "
                            "written by its own thread");
    if (!op.getReduce().empty())
      return explainRefused(op, "hash",
                            "the loop has global sums, whose kernel takes "
                            "several tuples a thread for the rows of its "
                            "sums and keeps the atomics");
    for (unsigned i = 0, e = op.getOuts().size(); i != e; ++i)
      if (op.overwrites(i))
        return explainRefused(op, "hash",
                              "the loop overwrites a destination");
    int64_t slots = scatter::getHashSlots(block, op.getArity());
    if (scatter::Refusal refused = scatter::checkHash(
            slots, op.getOuts().getTypes(),
            algebra::getDestinationAlgebras(op), budget))
      return explainRefused(op, "hash", *refused);
    ScatterAttr chosen = ScatterAttr::getHash(&getContext(), slots, probes,
                                              hash == "on" ? 0 : least);
    op.setScatterAttr(chosen);
    explainChosen(op, chosen,
                  hashProbes == -1 || (hash == "auto" && hashLeast == -1));
  }

  /// The tables of the loops over pairs that add at both ends of a pair
  /// (D291): the loops over groups, each pair once.
  void choosePairHash(PairForOp op) {
    // A loop that writes only at its centers, each by its own thread,
    // scatters nothing; nor does one with only global sums.
    if (op.getConflict() != Conflict::Atomic || op.getOuts().empty())
      return;
    if (pairHash == "off")
      return explainRefused(op, "hash", "pair-hash=off");
    // With `auto`, the atomics: the table was measured slower at every
    // loop over pairs (on an RTX 4090, 1.18 to 1.32 times the step of the
    // Amber suite over groups and 2.06 times the two-ended loops of EAM
    // over a matrix; docs/device-primitives.md, Section 2.7).
    if (pairHash == "auto")
      return explainRefused(op, "hash",
                            "the table was measured slower at every loop "
                            "over pairs, 1.18 to 2.06 times the atomics "
                            "(pair-hash=auto)");
    int64_t slots = scatter::getHashSlots(block, scatter::kPairHashReach);
    if (scatter::Refusal refused = scatter::checkHash(
            slots, op.getOuts().getTypes(),
            algebra::getDestinationAlgebras(op), budget))
      return explainRefused(op, "hash", *refused);
    ScatterAttr chosen =
        ScatterAttr::getHash(&getContext(), slots, probes, pairLeast);
    op.setScatterAttr(chosen);
    explainChosen(op, chosen, hashProbes == -1 || pairHashLeast == -1);
  }

  /// The table of a scatter-accumulate on its own (D291): a
  /// thread a value.
  void chooseScatterHash(ScatterAddOp op) {
    if (scatterHash == "off")
      return explainRefused(op, "hash", "scatter-hash=off");
    int64_t slots = scatter::getHashSlots(block, 1);
    if (scatter::Refusal refused = scatter::checkHash(
            slots, TypeRange{op.getDest().getType()},
            algebra::getDestinationAlgebras(op), budget))
      return explainRefused(op, "hash", *refused);
    ScatterAttr chosen = ScatterAttr::getHash(
        &getContext(), slots, probes, scatterHash == "on" ? 0 : least);
    op.setScatterAttr(chosen);
    explainChosen(op, chosen,
                  hashProbes == -1 ||
                      (scatterHash == "auto" && hashLeast == -1));
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
