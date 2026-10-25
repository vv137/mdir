// Conservative access summaries for execution loops (D[loop-effect-summary]).

#include "mdir/Dialect/MDExec/LoopEffects.h"

#include "mdir/Dialect/MDExec/Algebra.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;
using namespace mdir::analysis;

#include "mdir/Dialect/MDExec/LoopEffects.cpp.inc"

static Attribute setOf(Type type) {
  if (auto field = dyn_cast<md::FieldType>(type))
    return field.getParticleSet();
  if (auto neighbors = dyn_cast<mdrt::NeighborsType>(type))
    return neighbors.getParticleSet();
  if (auto incidence = dyn_cast<mdrt::IncidenceType>(type))
    return incidence.getParticleSet();
  if (auto buffer = dyn_cast<MemRefType>(type))
    if (auto layout = dyn_cast_if_present<SetLayoutAttr>(buffer.getLayout()))
      return layout.getSet();
  return {};
}

static void add(LoopEffectSummary &summary, OpOperand &operand, Effect effect,
                Rows rows) {
  Attribute set = setOf(operand.get().getType());
  summary.add(operand, effect, rows, set);
  summary.complete &= static_cast<bool>(set);
}

template <typename OpTy>
static void appendFieldAccesses(OpTy op, LoopEffectSummary &summary,
                                Rows inputRows, Rows outputRows,
                                bool accumulates,
                                function_ref<bool(unsigned)> readsOuts) {
  for (OpOperand &operand : op.getInsMutable())
    add(summary, operand, Effect::Read, inputRows);
  for (auto [index, operand] : llvm::enumerate(op.getOutsMutable())) {
    if (readsOuts(index))
      add(summary, operand, Effect::Read, outputRows);
    add(summary, operand, accumulates ? Effect::Accumulate : Effect::Write,
        outputRows);
    if (accumulates) {
      std::optional<Algebra> rule =
          algebra::getAlgebra(op.getOperation(), index);
      summary.reductions.push_back({&operand, rule, outputRows});
      summary.complete &= rule.has_value();
    }
  }
  for (OpOperand &operand : op.getScratchMutable())
    add(summary, operand, Effect::Write, Rows::Whole);
  unsigned offset = accumulates ? op.getOuts().size() : 0;
  for (auto [index, operand] : llvm::enumerate(op.getReduceMutable())) {
    std::optional<Algebra> rule =
        algebra::getAlgebra(op.getOperation(), offset + index);
    summary.reductions.push_back({&operand, rule, Rows::Whole});
    summary.complete &= rule.has_value();
  }
}

void PairForOp::getLoopEffectSummary(LoopEffectSummary &summary) {
  add(summary, getNeighborsMutable(), Effect::Read, Rows::Whole);
  add(summary, getPositionsMutable(), Effect::Read, Rows::Neighbor);
  for (OpOperand &operand : getCenterSetMutable())
    add(summary, operand, Effect::Read, Rows::Own);
  Rows destinationRows =
      getConflict() == Conflict::Atomic ? Rows::Neighbor : Rows::Own;
  appendFieldAccesses(*this, summary, Rows::Neighbor, destinationRows,
                      /*accumulates=*/true,
                      [&](unsigned index) { return !overwrites(index); });
}

void ParticleForOp::getLoopEffectSummary(LoopEffectSummary &summary) {
  appendFieldAccesses(*this, summary, Rows::Own, Rows::Own,
                      /*accumulates=*/false, [](unsigned) { return false; });
}

void TupleForOp::getLoopEffectSummary(LoopEffectSummary &summary) {
  add(summary, getIncidenceMutable(), Effect::Read, Rows::Whole);
  add(summary, getPositionsMutable(), Effect::Read, Rows::TupleMembers);
  for (OpOperand &operand : getParametersMutable())
    add(summary, operand, Effect::Read, Rows::TupleMembers);
  appendFieldAccesses(*this, summary, Rows::TupleMembers, Rows::Own,
                      /*accumulates=*/true,
                      [&](unsigned index) { return !overwrites(index); });
}

void ScatterAddOp::getLoopEffectSummary(LoopEffectSummary &summary) {
  add(summary, getValuesMutable(), Effect::Read, Rows::Own);
  add(summary, getIndicesMutable(), Effect::Read, Rows::Own);
  if (!overwrites())
    add(summary, getDestMutable(), Effect::Read, Rows::Indexed);
  add(summary, getDestMutable(), Effect::Accumulate, Rows::Indexed);
  std::optional<Algebra> rule = algebra::getAlgebra(getOperation(), 0);
  summary.reductions.push_back({&getDestMutable(), rule, Rows::Indexed});
  summary.complete &= rule.has_value();
}

LoopEffectSummary mdir::md_exec::summarizeLoop(Operation *op) {
  LoopEffectSummary summary;
  auto loop = dyn_cast<LoopEffectOpInterface>(op);
  if (!loop) {
    summary.complete = false;
    return summary;
  }
  loop.getLoopEffectSummary(summary);
  for (Region &region : op->getRegions())
    region.walk([&](Operation *nested) {
      if (auto memory = dyn_cast<MemoryEffectOpInterface>(nested)) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memory.getEffects(effects);
        // The loop supplies scalar kernel arguments for its declared rows.
        // A body memory access needs a separate footprint proof, even when
        // its buffer is a declared operand: the index may reach other rows.
        // This interface slice does not attempt that proof.
        summary.complete &= effects.empty();
      } else if (!nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
                 !isPure(nested)) {
        summary.complete = false;
      }
    });
  return summary;
}
