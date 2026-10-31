// Evaluations retain request/result order and the original physical state.
#include "PotentialIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
using namespace mlir;
using namespace mdir::driver::builder;
SmallVector<Value> mdir::driver::builder::potentialArguments(const PotentialStateIR &state) {
  SmallVector<Value> values{state.positions, state.cell};
  llvm::append_range(values, state.fields);
  if (state.time)
    values.push_back(state.time);
  return values;
}
EnergyResults mdir::driver::builder::buildEnergyEvaluation(
    OpBuilder &builder, const PotentialStateIR &state, const EnergyRequests &request) {
  assert((request.energy || request.forces || request.virial) && "evaluation needs a result");
  auto loc = builder.getUnknownLoc();
  SmallVector<Type> types;
  SmallVector<int32_t> kinds;
  if (request.energy) {
    types.push_back(builder.getF64Type());
    kinds.push_back(int32_t(mdir::md::Request::Energy));
  }
  if (request.forces) {
    types.push_back(state.positions.getType());
    kinds.push_back(int32_t(mdir::md::Request::Forces));
  }
  if (request.virial) {
    types.push_back(VectorType::get({9}, builder.getF64Type()));
    kinds.push_back(int32_t(mdir::md::Request::Virial));
  }
  SmallVector<int64_t> arguments(kinds.size(), -1);
  auto evaluation = mdir::md::EvaluateOp::create(
      builder, loc, types, FlatSymbolRefAttr::get(builder.getContext(), "energy"),
      potentialArguments(state), builder.getDenseI32ArrayAttr(kinds),
      builder.getDenseI64ArrayAttr(arguments));
  size_t index = 0;
  EnergyResults result;
  if (request.energy) result.energy = evaluation.getResult(index++);
  if (request.forces) result.forces = evaluation.getResult(index++);
  if (request.virial) result.virial = evaluation.getResult(index++);
  return result;
}
Value mdir::driver::builder::buildSimulationTime(OpBuilder &builder, Value step, Value interval) {
  auto loc = builder.getUnknownLoc();
  Value real = arith::SIToFPOp::create(builder, loc, builder.getF64Type(), step);
  return arith::MulFOp::create(builder, loc, real, interval);
}

SmallVector<Type> mdir::driver::builder::buildParameterTypes(
    MLIRContext &context, const mdir::driver::Program &program) {
  auto atoms = FlatSymbolRefAttr::get(&context, "atoms");
  auto f64 = Float64Type::get(&context);
  auto i32 = IntegerType::get(&context, 32);
  SmallVector<Type> types;
  for (const auto &field : program.fields)
    types.push_back(mdir::md::FieldType::get(&context, atoms, 1,
        field.isInteger ? Type(i32) : Type(f64)));
  for (const auto &table : program.tables)
    types.push_back(mdir::md::TableType::get(&context, 2, f64, table.isSquare()));
  for (const auto &set : program.tupleSets) {
    auto symbol = FlatSymbolRefAttr::get(&context, set.name);
    auto orientation = set.arity == 2 && !set.oriented
                           ? mdir::md::Orientation::Unordered
                           : set.reversible ? mdir::md::Orientation::Reversal
                                            : mdir::md::Orientation::Ordered;
    types.push_back(mdir::md::RelationType::get(&context, atoms, set.arity, orientation, symbol));
    for (unsigned i = 0; i != set.fields.size(); ++i)
      types.push_back(mdir::md::FieldType::get(&context, symbol, 1, f64));
  }
  return types;
}

mdir::md::PotentialOp mdir::driver::builder::buildScalarPotential(
    OpBuilder &builder, llvm::StringRef name, TypeRange inputs,
    llvm::function_ref<Value(OpBuilder &, ValueRange)> body) {
  auto loc = builder.getUnknownLoc();
  auto potential = mdir::md::PotentialOp::create(builder, loc, name,
      builder.getFunctionType(inputs, {builder.getF64Type()}), StringAttr{}, ArrayAttr{}, ArrayAttr{});
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Location> locations(inputs.size(), loc);
  auto args = builder.createBlock(&potential.getBody(), {}, inputs, locations)->getArguments();
  Value energy = body(builder, args);
  mdir::md::ReturnOp::create(builder, loc, ValueRange{energy});
  return potential;
}
