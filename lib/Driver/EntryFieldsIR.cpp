// Host buffers enter the native entry in their original operation order.
#include "EntryFieldsIR.h"
#include "ReorderIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/IR/Builders.h"
#include <cassert>
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;
Value mdir::driver::builder::buildEntryField(OpBuilder &builder, Value buffer,
    unsigned components, bool integer, llvm::StringRef set) {
  Type element = integer ? Type(builder.getI32Type()) : Type(builder.getF64Type());
  auto type = mdir::md::FieldType::get(builder.getContext(),
      FlatSymbolRefAttr::get(builder.getContext(), set), components, element);
  return mdir::mdrt::FromBufferOp::create(builder, builder.getUnknownLoc(), type, buffer);
}
EntryFieldsIR mdir::driver::builder::buildEntryFields(OpBuilder &builder,
    const Program &program, const EntryArgumentsIR &arguments,
    bool positions, bool givenForces) {
  EntryFieldsIR fields;
  auto loc = builder.getUnknownLoc();
  auto *context = builder.getContext();
  assert(program.fields.size() == arguments.fields.size());
  assert(program.tables.size() == arguments.tables.size());
  assert(program.tupleSets.size() == arguments.tuples.size());
  if (positions) fields.positions = buildEntryField(builder, arguments.positions, 3, false);
  fields.velocities = buildEntryField(builder, arguments.velocities, 3, false);
  fields.masses = buildEntryField(builder, arguments.masses, 1, false);
  for (auto [field, input] : llvm::zip(program.fields, arguments.fields))
    fields.fields.push_back(buildEntryField(builder, input, 1, field.isInteger));
  for (auto [table, input] : llvm::zip(program.tables, arguments.tables)) {
    auto type = mdir::md::TableType::get(context, 2, builder.getF64Type(), table.isSquare());
    fields.tables.push_back(mdir::mdrt::FromBufferOp::create(builder, loc, type, input));
  }
  for (auto [tuple, inputs] : llvm::zip(program.tupleSets, arguments.tuples)) {
    auto orientation = tuple.arity == 2 && !tuple.oriented
        ? mdir::md::Orientation::Unordered
        : tuple.reversible ? mdir::md::Orientation::Reversal : mdir::md::Orientation::Ordered;
    auto type = mdir::md::RelationType::get(context, FlatSymbolRefAttr::get(context, "atoms"),
        tuple.arity, orientation, FlatSymbolRefAttr::get(context, tuple.name));
    EntryTupleBuffersIR values;
    values.members = mdir::mdrt::FromBufferOp::create(builder, loc, type, inputs.members);
    assert(tuple.fields.size() == inputs.fields.size());
    for (Value input : inputs.fields)
      values.fields.push_back(buildEntryField(builder, input, 1, false, tuple.name));
    fields.tuples.push_back(std::move(values));
  }
  fields.identities = buildEntryField(builder, arguments.identities, 1, true);
  if (givenForces) {
    assert(arguments.forces && "given forces need an entry buffer");
    fields.forces = buildEntryField(builder, arguments.forces, 3, false);
  }
  return fields;
}

SmallVector<Value> mdir::driver::builder::buildEntryPotentialFields(const EntryFieldsIR &fields) {
  SmallVector<Value> values;
  llvm::append_range(values, fields.fields);
  llvm::append_range(values, fields.tables);
  for (const auto &tuple : fields.tuples) {
    values.push_back(tuple.members);
    llvm::append_range(values, tuple.fields);
  }
  return values;
}
EntryFieldsIR mdir::driver::builder::buildEntryReorder(OpBuilder &builder, const Program &program,
    EntryFieldsIR fields, Value cell) {
  if (!program.reorders) return fields;
  SmallVector<ReorderFieldIR> parameters;
  for (auto [index, field] : llvm::enumerate(program.fields)) {
    std::optional<unsigned> source;
    if (field.name == "pme_listed" || field.name == "ljpme_listed") {
      StringRef name = field.name == "pme_listed" ? "q" : "ljpme_c";
      auto found = llvm::find_if(program.fields,
          [&](const auto &candidate) { return candidate.name == name; });
      unsigned slot = std::distance(program.fields.begin(), found);
      assert(slot < index && "derived list must follow its source field");
      source = slot;
    }
    parameters.push_back({source ? Value{} : fields.fields[index], source});
  }
  SmallVector<Value> relations;
  ReorderIRConfig config{program.orderWidth, {}};
  for (auto [index, tuple] : llvm::enumerate(program.tupleSets)) {
    relations.push_back(fields.tuples[index].members);
    if (tuple.name == "anchors") config.anchorRelation = index;
  }
  ReorderIRInputs input{fields.positions, fields.velocities, {}, fields.masses,
      fields.identities, cell, parameters, relations};
  auto result = buildReorder(builder, input, config);
  fields.positions = result.positions;
  fields.velocities = result.velocities;
  fields.masses = result.masses;
  fields.identities = result.ids;
  fields.fields = std::move(result.fields);
  for (auto [index, relation] : llvm::enumerate(result.relations))
    fields.tuples[index].members = relation;
  return fields;
}
