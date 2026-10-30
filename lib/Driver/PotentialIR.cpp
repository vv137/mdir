// Evaluations retain request/result order and the original physical state.
#include "PotentialIR.h"
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
