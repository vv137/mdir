// The algebra of each reduced value (D301); see
// include/mdir/Dialect/MDExec/Algebra.h.

#include "mdir/Dialect/MDExec/Algebra.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mdir/Dialect/MDDist/MDDistTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

/// The element type of a field, a buffer, a number, or a vector.
static Type getElement(Type type) {
  if (auto field = dyn_cast<md::FieldType>(type))
    return field.getElementType();
  return getElementTypeOrSelf(type);
}

bool algebra::reduces(Operation *op) {
  return isa<ParticleForOp, PairForOp, TupleForOp, ScatterAddOp, ReciprocalOp,
             md_dist::AllReduceOp, md_dist::HaloSumOp>(op);
}

bool algebra::isDistributed(Operation *op) {
  return isa<md_dist::AllReduceOp, md_dist::HaloSumOp>(op);
}

void algebra::stateDistributed(Operation *op) {
  SmallVector<Algebra> values(getReducedTypes(op).size(), Algebra::FixedOrder);
  setAlgebras(op, values);
}

SmallVector<Type> algebra::getReducedTypes(Operation *op) {
  SmallVector<Type> types;
  auto add = [&](ValueRange values) {
    for (Value value : values)
      types.push_back(getElement(value.getType()));
  };
  llvm::TypeSwitch<Operation *>(op)
      .Case([&](ParticleForOp loop) { add(loop.getReduce()); })
      .Case([&](PairForOp loop) {
        add(loop.getOuts());
        add(loop.getReduce());
      })
      .Case([&](TupleForOp loop) {
        add(loop.getOuts());
        add(loop.getReduce());
      })
      .Case([&](ScatterAddOp scatter) { add(scatter.getDest()); })
      .Case([&](md_dist::AllReduceOp sum) {
        types.push_back(getElementTypeOrSelf(sum.getResult().getType()));
      })
      .Case([&](md_dist::HaloSumOp sum) {
        types.push_back(
            cast<md_dist::FieldType>(sum.getResult().getType()).getElementType());
      })
      .Case([&](ReciprocalOp sum) {
        // The grid is in the type of the forces (D271).
        Value forces = sum.isStorageForm() ? Value(sum.getOut())
                                           : Value(sum.getForces());
        types.push_back(getElement(forces ? forces.getType()
                                          : sum.getPositions().getType()));
        types.push_back(sum.getEnergy().getType());
        types.push_back(getElement(sum.getVirial().getType()));
      });
  return types;
}

bool algebra::needsAlgebra(Operation *op) {
  return reduces(op) && !getReducedTypes(op).empty();
}

void algebra::carry(Operation *from, Operation *to) {
  if (Attribute list = from->getAttr(kAlgebraAttrName))
    to->setAttr(kAlgebraAttrName, list);
}

Algebra algebra::getAssigned(Type element, bool deterministic) {
  if (isa<IntegerType>(element))
    return Algebra::AnyOrderExact;
  return deterministic ? Algebra::FixedOrder : Algebra::AnyOrder;
}

void algebra::setAlgebras(Operation *op, ArrayRef<Algebra> values) {
  op->setAttr(kAlgebraAttrName,
              AlgebraListAttr::get(op->getContext(), values));
}

std::optional<SmallVector<Algebra>> algebra::getAlgebras(Operation *op) {
  auto list = op->getAttrOfType<AlgebraListAttr>(kAlgebraAttrName);
  if (!list)
    return std::nullopt;
  return SmallVector<Algebra>(list.getValues());
}

std::string algebra::getMissingMessage() {
  return "states no algebra of the values it reduces; run "
         "'md-exec-assign-algebra'";
}

std::optional<Algebra> algebra::getAlgebra(Operation *op, unsigned index,
                                           std::string *refusal) {
  std::optional<SmallVector<Algebra>> values = getAlgebras(op);
  if (!values || index >= values->size()) {
    if (refusal)
      *refusal = getMissingMessage();
    return std::nullopt;
  }
  return (*values)[index];
}

std::optional<SmallVector<Algebra>>
algebra::getDestinationAlgebras(Operation *op) {
  std::optional<SmallVector<Algebra>> values = getAlgebras(op);
  if (!values)
    return std::nullopt;
  unsigned count = llvm::TypeSwitch<Operation *, unsigned>(op)
                       .Case([](PairForOp loop) { return loop.getOuts().size(); })
                       .Case([](TupleForOp loop) {
                         return loop.getOuts().size();
                       })
                       .Case([](ScatterAddOp) { return 1u; })
                       .Default([](Operation *) { return 0u; });
  values->truncate(std::min<size_t>(count, values->size()));
  return values;
}

bool algebra::allowReorder(ArrayRef<Algebra> values) {
  return llvm::all_of(values, allowsReorder);
}

LogicalResult algebra::verifyDeterministic(Operation *module) {
  WalkResult result = module->walk([&](Operation *op) -> WalkResult {
    std::optional<SmallVector<Algebra>> values = getAlgebras(op);
    if (!values)
      return WalkResult::advance();
    SmallVector<Type> types = getReducedTypes(op);
    for (auto [index, value] : llvm::enumerate(*values))
      if (value == Algebra::AnyOrder && index < types.size() &&
          isa<FloatType>(types[index])) {
        op->emitOpError()
            << "states the algebra 'any_order' for reduced value " << index
            << " of " << types[index]
            << " in a program built for the deterministic mode, whose "
               "floating-point sums are 'fixed_order'";
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult algebra::verify(Operation *op, Attribute attr) {
  auto list = dyn_cast<AlgebraListAttr>(attr);
  if (!list)
    return op->emitError() << "expects '" << kAlgebraAttrName
                           << "' to be an #md_exec.algebra";
  if (!reduces(op))
    return op->emitError() << "states an algebra but reduces no values";
  SmallVector<Type> types = getReducedTypes(op);
  if (list.getValues().size() != types.size())
    return op->emitOpError()
           << "states " << list.getValues().size()
           << " algebras for the " << types.size() << " values it reduces";
  for (auto [index, value, type] :
       llvm::enumerate(list.getValues(), types))
    if (value == Algebra::AnyOrderExact && !isa<IntegerType>(type))
      return op->emitOpError()
             << "states the algebra 'any_order_exact' for reduced value " << index
             << " of " << type
             << ", whose sum rounds by the order of its terms";
  return success();
}
