// Shared projected-force traces preserve the entry's original group order.
#include "EntryConstraintIR.h"
#include "DynamicsIR.h"
using namespace mlir;
using namespace mdir::driver::builder;
llvm::SmallVector<RigidGroupIR> mdir::driver::builder::buildEntryGroups(ValueRange fields,
    llvm::ArrayRef<EntryConstraintIR> groups) {
  llvm::SmallVector<RigidGroupIR> values;
  for (const auto &group : groups) values.push_back({fields[group.relation], group.arity});
  return values;
}
Value mdir::driver::builder::buildEntryConstraintTrace(OpBuilder &builder, const EntryFieldsIR &physical,
    const PotentialStateIR &state, Value forces, Value trace,
    llvm::ArrayRef<EntryConstraintIR> groups, bool axes) {
  Value descent = buildDescentDirection(builder, forces, physical.masses);
  for (const auto &group : groups) {
    SmallVector<Value> lengths;
    for (unsigned slot : group.parameters) lengths.push_back(state.fields[slot]);
    ConstraintVelocityInputs input{state.fields[group.relation], physical.positions,
        state.cell, descent, physical.masses, lengths, {}, {}};
    descent = group.water ? buildSettleVelocities(builder, input).velocities
                          : buildShakeVelocities(builder, input, group.arity).velocities;
  }
  for (const auto &group : groups)
    trace = buildConstraintTrace(builder,
        {state.fields[group.relation], physical.positions, state.cell, descent,
         forces, physical.velocities, physical.masses, trace}, group.arity, axes);
  return trace;
}
