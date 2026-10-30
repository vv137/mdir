// Output evaluations retain request, table, and host-call order.
#include "OutputIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mdir::driver::builder;

namespace {
class OutputBuilder {
public:
  explicit OutputBuilder(OpBuilder &builder)
      : builder(builder), loc(builder.getUnknownLoc()) {}
  Value real(double value) {
    return arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(value));
  }
  Value index(size_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value);
  }
  Value allocate(ArrayRef<int64_t> shape) {
    return memref::AllocaOp::create(
        builder, loc, MemRefType::get(shape, builder.getF64Type()));
  }
  void store(Value value, Value memory, ValueRange slots) {
    memref::StoreOp::create(builder, loc, value, memory, slots);
  }
  mdir::md::EvaluateOp evaluate(StringRef name, ValueRange operands,
                                 ArrayRef<int32_t> kinds,
                                 ArrayRef<int64_t> derivatives) {
    SmallVector<Type> types(kinds.size(), builder.getF64Type());
    return mdir::md::EvaluateOp::create(
        builder, loc, types, FlatSymbolRefAttr::get(builder.getContext(), name),
        operands, builder.getDenseI32ArrayAttr(kinds),
        builder.getDenseI64ArrayAttr(derivatives));
  }
  void write(StringRef name, Value memory, Value step) {
    Value cast = memref::CastOp::create(
        builder, loc, MemRefType::get({ShapedType::kDynamic}, builder.getF64Type()),
        memory);
    SmallVector<Value> values;
    if (step)
      values.push_back(step);
    values.push_back(cast);
    func::CallOp::create(builder, loc, name, TypeRange{}, values);
  }

private:
  OpBuilder &builder;
  Location loc;
};
} // namespace

void mdir::driver::builder::buildFreeEnergyOutput(
    OpBuilder &builder, const PotentialStateIR &state, Value step,
    Value zeroIndex, Value oneIndex, const FreeEnergyOutputConfig &config) {
  auto loc = builder.getUnknownLoc();
  OutputBuilder out(builder);
  size_t components = config.lambdas.size();
  size_t states = config.lambdas.empty() ? 0 : config.lambdas.front().size();
  if (states == 1)
    states = 0;
  Value memory = out.allocate({static_cast<int64_t>(components + states)});
  SmallVector<Value> values = potentialArguments(state);
  for (const auto &component : config.lambdas)
    values.push_back(out.real(component[config.state]));
  SmallVector<int32_t> kinds(components, int32_t(mdir::md::Request::Derivative));
  SmallVector<int64_t> derivatives;
  for (size_t c = 0; c != components; ++c)
    derivatives.push_back(config.baseArgument + c);
  auto derivative = out.evaluate("alchemical", values, kinds, derivatives);
  for (size_t c = 0; c != components; ++c) {
    Value slot = out.index(c);
    out.store(derivative.getResult(c), memory, ValueRange{slot});
  }
  if (states == 0) {
    out.write("mdrtWriteFreeEnergy", memory, step);
    return;
  }
  Value table = out.allocate({static_cast<int64_t>(states),
                               static_cast<int64_t>(components)});
  for (size_t k = 0; k != states; ++k)
    for (size_t c = 0; c != components; ++c) {
      Value value = out.real(config.lambdas[c][k]);
      Value row = out.index(k), column = out.index(c);
      out.store(value, table, ValueRange{row, column});
    }
  Value count = out.index(states), first = out.index(components);
  auto loop = scf::ForOp::create(builder, loc, zeroIndex, count, oneIndex);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    SmallVector<Value> loaded = potentialArguments(state);
    for (size_t c = 0; c != components; ++c) {
      Value column = out.index(c);
      Value value = memref::LoadOp::create(
          builder, loc, table, ValueRange{loop.getInductionVar(), column});
      loaded.push_back(value);
    }
    auto energy = out.evaluate("alchemical", loaded,
                                {int32_t(mdir::md::Request::Energy)}, {-1});
    Value slot = arith::AddIOp::create(builder, loc, first, loop.getInductionVar());
    out.store(energy.getResult(0), memory, ValueRange{slot});
  }
  out.write("mdrtWriteFreeEnergy", memory, step);
}

void mdir::driver::builder::buildObservablesOutput(
    OpBuilder &builder, const PotentialStateIR &state, Value step, size_t count,
    unsigned baseArgument, ArrayRef<ObservableOutputTermIR> terms,
    ValueRange tunables) {
  OutputBuilder out(builder);
  Value memory = out.allocate({static_cast<int64_t>(count)});
  for (const auto &term : terms) {
    SmallVector<Value> values = potentialArguments(state);
    for (const auto &constant : term.constants)
      values.push_back(constant.tunable ? tunables[*constant.tunable]
                                       : out.real(constant.value));
    SmallVector<int32_t> kinds;
    SmallVector<int64_t> derivatives;
    for (auto [argument, slot] : term.columns) {
      kinds.push_back(int32_t(argument < 0 ? mdir::md::Request::Energy
                                          : mdir::md::Request::Derivative));
      derivatives.push_back(argument < 0 ? -1 : baseArgument + argument);
    }
    auto result = out.evaluate(term.potential, values, kinds, derivatives);
    for (auto [i, column] : llvm::enumerate(term.columns)) {
      Value slot = out.index(column.second);
      out.store(result.getResult(i), memory, ValueRange{slot});
    }
  }
  out.write("mdrtWriteObservables", memory, step);
}

void mdir::driver::builder::buildTunableGradientOutput(
    OpBuilder &builder, const PotentialStateIR &state, Value asked,
    Value zeroIndex, Value oneIndex, Value ids, ValueRange constants,
    ValueRange seeds, const TunableGradientOutputConfig &config) {
  auto loc = builder.getUnknownLoc();
  OutputBuilder out(builder);
  Value zero = out.real(0.0);
  Value test = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::ONE,
                                      asked, zero);
  Value trips = arith::SelectOp::create(builder, loc, test, oneIndex, zeroIndex);
  auto loop = scf::ForOp::create(builder, loc, zeroIndex, trips, oneIndex);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  size_t count = config.scalarArguments.size();
  Value memory = out.allocate({static_cast<int64_t>(count + 1)});
  SmallVector<Value> values = potentialArguments(state);
  llvm::append_range(values, constants);
  llvm::append_range(values, seeds);
  SmallVector<Type> types(count + 1, builder.getF64Type());
  SmallVector<int32_t> kinds{int32_t(mdir::md::Request::Energy)};
  SmallVector<int64_t> derivatives{-1};
  for (unsigned argument : config.scalarArguments) {
    kinds.push_back(int32_t(mdir::md::Request::Derivative));
    derivatives.push_back(argument);
  }
  for (unsigned argument : config.fieldArguments) {
    kinds.push_back(int32_t(mdir::md::Request::Derivative));
    derivatives.push_back(argument);
    types.push_back(values[argument].getType());
  }
  auto result = mdir::md::EvaluateOp::create(
      builder, loc, types, FlatSymbolRefAttr::get(builder.getContext(), "tunable"),
      values, builder.getDenseI32ArrayAttr(kinds),
      builder.getDenseI64ArrayAttr(derivatives));
  auto energy = out.evaluate("energy", potentialArguments(state),
                              {int32_t(mdir::md::Request::Energy)}, {-1});
  Value shift = arith::SubFOp::create(builder, loc, result.getResult(0),
                                       energy.getResult(0));
  Value last = out.index(count);
  out.store(shift, memory, ValueRange{last});
  for (size_t c = 0; c != config.fieldArguments.size(); ++c) {
    Value number = arith::ConstantIntOp::create(builder, loc, c, 64);
    mdir::mdrt::HostCallOp::create(
        builder, loc, TypeRange{},
        FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteTunableGradientField"),
        ValueRange{number, result.getResult(count + 1 + c), ids}, ValueRange{},
        UnitAttr{});
  }
  for (size_t i = 0; i != count; ++i) {
    Value slot = out.index(i);
    out.store(result.getResult(i + 1), memory, ValueRange{slot});
  }
  out.write("mdrtWriteTunableGradient", memory, {});
}

void mdir::driver::builder::buildWriteSolvent(OpBuilder &builder, Value sums) {
  auto loc = builder.getUnknownLoc();
  Value full = vector::ExtractOp::create(builder, loc, sums, int64_t{0});
  Value half = vector::ExtractOp::create(builder, loc, sums, int64_t{1});
  func::CallOp::create(builder, loc, "mdrtWriteSolvent", TypeRange{},
                       ValueRange{full, half});
}

Value mdir::driver::builder::buildTermEnergyBuffer(
    OpBuilder &builder, const PotentialStateIR &state, ArrayRef<std::string> potentials,
    size_t count) {
  OutputBuilder out(builder);
  Value memory = out.allocate({static_cast<int64_t>(count)});
  for (auto [index, potential] : llvm::enumerate(potentials)) {
    auto result = out.evaluate(potential, potentialArguments(state),
                                {int32_t(mdir::md::Request::Energy)}, {-1});
    Value slot = out.index(index);
    out.store(result.getResult(0), memory, ValueRange{slot});
  }
  return memory;
}

void mdir::driver::builder::buildWriteTerms(OpBuilder &builder, Value memory) {
  OutputBuilder(builder).write("mdrtWriteTerms", memory, {});
}
