// The records of the profile map (D398, #840); see
// include/mdir/Dialect/MDExec/ProfileMap.h.

#include "mdir/Dialect/MDExec/ProfileMap.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir;

Location profile::inPotential(Location loc, StringRef potential) {
  return NameLoc::get(
      StringAttr::get(loc.getContext(), kPotentialPrefix + potential), loc);
}

/// The set of the rows of a buffer in the storage form, or "".
static StringRef getSet(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  if (!buffer)
    return {};
  if (auto layout = dyn_cast<md_exec::SetLayoutAttr>(buffer.getLayout()))
    return layout.getSet().getValue();
  return {};
}

/// The tuple set of the tuples that `incidence` lists the members of.
static StringRef getTupleSet(Operation *op, Value incidence) {
  if (auto set = op->getAttrOfType<FlatSymbolRefAttr>("md_exec.tuple_set"))
    return set.getValue();
  if (Operation *definition = incidence.getDefiningOp()) {
    if (auto set = definition->getAttrOfType<FlatSymbolRefAttr>(
            "md_exec.tuple_set"))
      return set.getValue();
    if (definition->getNumOperands() > 0)
      if (StringRef set = getSet(definition->getOperand(0).getType());
          !set.empty())
        return set;
  }
  return {};
}

/// The set of the rows that `op` runs over.
static StringRef getRows(Operation *op) {
  if (auto loop = dyn_cast<md_exec::TupleForOp>(op)) {
    if (StringRef set = getTupleSet(op, loop.getIncidence()); !set.empty())
      return set;
    for (Value parameter : loop.getParameters())
      if (StringRef set = getSet(parameter.getType()); !set.empty())
        return set;
    return {};
  }
  if (auto loop = dyn_cast<md_exec::PairForOp>(op))
    return getSet(loop.getPositions().getType());
  for (Type type : op->getOperandTypes())
    if (StringRef set = getSet(type); !set.empty())
      return set;
  for (Type type : op->getResultTypes())
    if (StringRef set = getSet(type); !set.empty())
      return set;
  return {};
}

/// The names of the potentials and of the terms that the locations under
/// `op` are in.
static void collectNames(Operation *op, llvm::SetVector<StringRef> &potentials,
                         llvm::SetVector<StringRef> &terms) {
  auto visit = [&](Location loc) {
    loc->walk([&](Location inner) {
      if (auto name = dyn_cast<NameLoc>(inner)) {
        StringRef text = name.getName().getValue();
        if (text.consume_front(profile::kPotentialPrefix))
          potentials.insert(text);
        else if (text.consume_front(profile::kTermPrefix))
          terms.insert(text);
      }
      return WalkResult::advance();
    });
  };
  op->walk([&](Operation *inner) { visit(inner->getLoc()); });
}

/// The names, sorted: the order in which a kernel's ops come is not part of
/// the record.
static ArrayAttr getStrings(Builder &builder,
                            const llvm::SetVector<StringRef> &names) {
  SmallVector<StringRef> sorted(names.begin(), names.end());
  llvm::sort(sorted);
  SmallVector<Attribute> values;
  for (StringRef name : sorted)
    values.push_back(builder.getStringAttr(name));
  return builder.getArrayAttr(values);
}

DictionaryAttr profile::describe(Operation *op) {
  Builder builder(op->getContext());
  SmallVector<NamedAttribute> entries;
  StringRef source = op->getName().stripDialect();
  entries.push_back(builder.getNamedAttr("source",
                                         builder.getStringAttr(source)));
  if (StringRef set = getRows(op); !set.empty())
    entries.push_back(builder.getNamedAttr("set", builder.getStringAttr(set)));
  llvm::SetVector<StringRef> potentials, terms;
  collectNames(op, potentials, terms);
  // The potential of a term alone is `term_<name>`, which the builder
  // makes for the energy of each term; that of the sum of the terms
  // (`energy`) names none of them.
  for (StringRef potential : potentials)
    if (potential.starts_with("term_"))
      terms.insert(potential.drop_front(5));
  if (!potentials.empty())
    entries.push_back(builder.getNamedAttr(
        "potentials", getStrings(builder, potentials)));
  if (!terms.empty())
    entries.push_back(
        builder.getNamedAttr("terms", getStrings(builder, terms)));
  return builder.getDictionaryAttr(entries);
}

Location profile::attach(Location loc, DictionaryAttr record) {
  // The record replaces one that the location has, so that a location is
  // never of two records.
  Location inner = loc;
  if (auto fused = dyn_cast<FusedLoc>(loc))
    if (auto old = dyn_cast_or_null<DictionaryAttr>(fused.getMetadata()))
      if (old.get(kRecordName))
        inner = fused.getLocations().size() == 1
                    ? fused.getLocations().front()
                    : FusedLoc::get(loc.getContext(), fused.getLocations());
  Builder builder(loc.getContext());
  auto metadata = builder.getDictionaryAttr(
      builder.getNamedAttr(kRecordName, record));
  return FusedLoc::get(loc.getContext(), {inner}, metadata);
}

DictionaryAttr profile::find(Location loc) {
  DictionaryAttr record;
  loc->walk([&](Location inner) {
    if (auto fused = dyn_cast<FusedLoc>(inner))
      if (auto metadata =
              dyn_cast_or_null<DictionaryAttr>(fused.getMetadata()))
        if (auto found = metadata.getAs<DictionaryAttr>(kRecordName)) {
          record = found;
          return WalkResult::interrupt();
        }
    return WalkResult::advance();
  });
  return record;
}

Location profile::refine(Location loc, StringRef stage) {
  DictionaryAttr record = find(loc);
  if (!record)
    return loc;
  NamedAttrList entries(record);
  entries.set("stage", StringAttr::get(loc.getContext(), stage));
  return attach(loc, entries.getDictionary(loc.getContext()));
}

void profile::tagOps(Operation *root) {
  // The ops of a kernel are copied into the loops of the op that holds
  // them and keep their own locations.
  root->walk<WalkOrder::PreOrder>([](Operation *op) {
    if (!isa_and_nonnull<md_exec::MDExecDialect>(op->getDialect()))
      return WalkResult::advance();
    if (!find(op->getLoc()))
      op->setLoc(attach(op->getLoc(), describe(op)));
    return WalkResult::skip();
  });
}

void profile::tagTemplate(Operation *function) {
  auto name = function->getAttrOfType<StringAttr>(
      SymbolTable::getSymbolAttrName());
  if (!name)
    return;
  Builder builder(function->getContext());
  DictionaryAttr record = builder.getDictionaryAttr(
      {builder.getNamedAttr("source", builder.getStringAttr("template")),
       builder.getNamedAttr("template", name)});
  function->walk([&](scf::ParallelOp loop) {
    if (!find(loop.getLoc()))
      loop->setLoc(attach(loop.getLoc(), record));
  });
}

/// The floating-point type of the elements of `type`, or null.
static FloatType getFloat(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    type = vector.getElementType();
  return dyn_cast<FloatType>(type);
}

std::string profile::getPrecision(Region &region) {
  // The arithmetic: an op whose operands and result are all of one
  // floating-point type. A load, a conversion, a constant, and a comparison
  // are not; their types are those of what the arithmetic reads and writes.
  bool f32 = false, f64 = false;
  region.walk([&](Operation *op) {
    if (op->getNumResults() != 1 || op->getNumOperands() == 0)
      return;
    Type type = op->getResult(0).getType();
    FloatType real = getFloat(type);
    if (!real)
      return;
    if (!llvm::all_of(op->getOperandTypes(),
                      [&](Type operand) { return operand == type; }))
      return;
    f32 |= real.getWidth() == 32;
    f64 |= real.getWidth() == 64;
  });
  return f32 && f64 ? "f32+f64" : f32 ? "f32" : f64 ? "f64" : "";
}
