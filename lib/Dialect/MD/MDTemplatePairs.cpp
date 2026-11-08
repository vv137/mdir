// Structural domains for the template window's explicit tuple arithmetic.
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mdir::md;

LogicalResult TemplatePairsOp::verify() {
  auto universe = cast<FieldType>(getUniverse().getType());
  auto molecules = cast<RelationType>(getMolecules().getType());
  if (universe.getNumComponents() != 1 ||
      !universe.getElementType().isInteger(32))
    return emitOpError("expected a scalar i32 universe field; only its cardinality is used");
  if (molecules.getParticleSet() != universe.getParticleSet() ||
      (molecules.getArity() != 3 && molecules.getArity() != 4) ||
      molecules.getOrientation() != Orientation::Ordered ||
      !molecules.getTupleSet())
    return emitOpError("expected named ordered three- or four-site molecules of the universe");
  constexpr unsigned arities[] = {6, 4, 2};
  for (unsigned i = 0; i != 3; ++i) {
    unsigned arity = arities[i];
    auto type = cast<RelationType>((*this)->getResult(i).getType());
    auto orientation = arity == 2 ? Orientation::Unordered : Orientation::Ordered;
    if (type.getParticleSet() != universe.getParticleSet() ||
        type.getArity() != arity || type.getOrientation() != orientation ||
        !type.getTupleSet())
      return emitOpError("expected named WW(6, ordered), SW(4, ordered), SS(2, unordered) domains of the universe");
    if (type.getTupleSet() == molecules.getTupleSet())
      return emitOpError("a class-pair domain cannot alias the molecule domain");
  }
  if (Value excluded = getExcluded()) {
    auto type = cast<RelationType>(excluded.getType());
    if (type.getParticleSet() != universe.getParticleSet() || type.getArity() != 2 ||
        type.getOrientation() != Orientation::Unordered)
      return emitOpError("expected unordered excluded pairs of the universe");
    if (type.getTupleSet() &&
        type.getTupleSet() == cast<RelationType>(getSoluteSolute().getType()).getTupleSet())
      return emitOpError("the SS domain cannot alias the excluded-pair domain");
  }
  return success();
}

LogicalResult TemplatePairsOp::verifySymbolUses(SymbolTableCollection &symbols) {
  if (failed(verify()))
    return failure();
  auto type = cast<RelationType>(getMolecules().getType());
  if (!type.getTupleSet())
    return emitOpError("expected a named molecule tuple set");
  auto set = symbols.lookupNearestSymbolFrom<TupleSetOp>(*this, type.getTupleSet());
  if (!set || !set.getDisjoint())
    return emitOpError("expected a declared disjoint molecule tuple set");
  if (set.getParticleSetAttr() != type.getParticleSet() ||
      set.getArity() != type.getArity() ||
      set.getOrientation() != type.getOrientation())
    return emitOpError("molecule tuple declaration and operand type must agree");
  for (Value result : getResults()) {
    auto resultType = cast<RelationType>(result.getType());
    auto domain = symbols.lookupNearestSymbolFrom<TupleSetOp>(
        *this, resultType.getTupleSet());
    if (!domain || domain.getParticleSetAttr() != resultType.getParticleSet() ||
        domain.getArity() != resultType.getArity() ||
        domain.getOrientation() != resultType.getOrientation())
      return emitOpError("class-pair domain declaration and result type must agree");
  }
  return success();
}
