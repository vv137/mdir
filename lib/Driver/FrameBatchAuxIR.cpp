#include "FrameBatchAuxIR.h"
#include "FrameBatchEvaluationIR.h"
#include "FrameBatchOutputIR.h"
#include "SimpleFrameEntryIR.h"
#include "EntryFieldsIR.h"
#include "KineticIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;
llvm::Error mdir::driver::builder::buildFrameBatchAux(OpBuilder &builder,
    const Program &program, const SimpleFrameEntryIRConfig &config,
    const EntryFieldsIR &physical, const PotentialStateIR &state, EnergyResults energy,
    const FrameBatchEvaluationPoint &point, Value volume, Value volumeScale,
    Value dispersionVirial, Value coulombVirial) {
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value bytes = memref::DimOp::create(builder, loc, point.aux, zero);
  Value asked = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt, bytes, zero);
  Value trips = arith::SelectOp::create(builder, loc, asked, one, zero);
  auto output = buildEntryLoop(builder, "device frame auxiliary outputs", zero, trips, one, {},
      [&](OpBuilder &at, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
    unsigned base = config.dependent ? 1 : 2;
    Value columns = arith::ConstantIndexOp::create(at, loc, base + config.output.observableCount);
    Value stride = arith::ConstantIndexOp::create(at, loc, 8 * (base + config.output.observableCount));
    Value offset = arith::MulIOp::create(at, loc, point.frame, stride);
    auto type = MemRefType::get({ShapedType::kDynamic}, at.getF64Type(),
        MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
    Value row = memref::ViewOp::create(at, loc, type, point.aux, offset, ValueRange{columns});
    auto write = [&](Value value, unsigned column) -> llvm::Error {
      Value index = arith::ConstantIndexOp::create(at, loc, column);
      return buildFrameBatchScalarOutput(at, value, row, point.frame, point.status, 6, index);
    };
    if (!config.dependent) {
      Value trace = buildVirialTrace(at, energy.virial);
      if (!config.constraints.empty())
        trace = buildEntryConstraintTrace(at, physical, state, energy.forces,
            trace, config.constraints, false);
      Value correction = arith::AddFOp::create(at, loc, dispersionVirial, coulombVirial);
      trace = arith::AddFOp::create(at, loc, trace, correction);
      if (auto error = write(trace, 0)) return error;
    }
    if (auto error = write(volume, base - 1)) return error;
    for (const auto &term : config.output.observables) {
      SmallVector<Value> arguments = potentialArguments(state);
      for (const auto &constant : term.constants) {
        Value value;
        if (constant.tunable)
          value = point.startValues[config.output.observableTunables[*constant.tunable]];
        else
          value = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(constant.value));
        arguments.push_back(value);
      }
      SmallVector<int32_t> kinds;
      SmallVector<int64_t> derivatives;
      for (auto [argument, column] : term.columns) {
        kinds.push_back(int32_t(argument < 0 ? mdir::md::Request::Energy : mdir::md::Request::Derivative));
        derivatives.push_back(argument < 0 ? -1 : config.output.observableBaseArgument + argument);
      }
      SmallVector<Type> types(term.columns.size(), at.getF64Type());
      auto observed = mdir::md::EvaluateOp::create(at, loc, types,
          FlatSymbolRefAttr::get(at.getContext(), term.potential), arguments,
          at.getDenseI32ArrayAttr(kinds), at.getDenseI64ArrayAttr(derivatives));
      for (auto [i, column] : llvm::enumerate(term.columns)) {
        Value value = observed.getResult(i);
        auto constant = [&](Value memory) -> Value {
          Value index = arith::ConstantIndexOp::create(at, loc, column.second);
          auto loaded = memref::LoadOp::create(at, loc, memory, index);
          loaded->setAttr("mdir.frame_parameter", at.getUnitAttr());
          return loaded;
        };
        if (column.second < program.observableVolumeConstants.size()) {
          Value scaled = arith::MulFOp::create(at, loc, constant(point.observableVolume), volumeScale);
          value = arith::AddFOp::create(at, loc, value, scaled);
        }
        if (column.second < program.observableFixedConstants.size())
          value = arith::AddFOp::create(at, loc, value, constant(point.observableFixed));
        if (auto error = write(value, base + column.second)) return error;
      }
    }
    return SmallVector<Value>{};
  });
  if (!output) return output.takeError();
  return llvm::Error::success();
}
