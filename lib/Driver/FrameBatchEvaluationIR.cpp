#include "FrameBatchEvaluationIR.h"
#include "FrameBatchOutputIR.h"
#include "FrameBatchTunableIR.h"
#include "SimpleFrameEntryIR.h"
#include "EntryFieldsIR.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

llvm::Error mdir::driver::builder::buildFrameBatchEvaluation(OpBuilder &builder,
    const Program &program, const SimpleFrameEntryIRConfig &config,
    const EntryFieldsIR &physical, const PotentialStateIR &state,
    const FrameBatchEvaluationPoint &point,
    llvm::function_ref<EntryCompletionIR(OpBuilder &, const PotentialStateIR &,
                                        EnergyResults, Value)> complete,
    Value scale) {
  auto loc = builder.getUnknownLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  auto real = [&](OpBuilder &at, double value) -> Value {
    return arith::ConstantOp::create(at, loc, at.getF64FloatAttr(value));
  };
  auto iteration = buildEntryLoop(builder, "device frame evaluation", zero, one, one, {},
      [&](OpBuilder &body, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
    auto evaluated = buildEntryEvaluation(body, state, {true, true, true}, config.sites, complete, scale);
    if (evaluated.storedTerms.size() != config.storedTerms)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
          "native device frame evaluation returned an unexpected stored-term count");
    auto energy = evaluated.physical;
    auto volume = arith::MulFOp::create(body, loc, point.edges[0], point.edges[1]);
    Value cellVolume = arith::MulFOp::create(body, loc, volume, point.edges[2]);
    double reference = config.batchParameters ? config.batchParameters->referenceVolume : 1.0;
    Value firstVolume = real(body, reference);
    auto correction = [&](unsigned index) -> Value {
      Value slot = arith::ConstantIndexOp::create(body, loc, index);
      auto loaded = memref::LoadOp::create(body, loc, point.corrections, slot);
      loaded->setAttr("mdir.frame_parameter", body.getUnitAttr());
      Value value = loaded;
      if (config.batchParameters && config.batchParameters->lj)
        value = arith::AddFOp::create(body, loc, value,
            real(body, config.batchParameters->correctionOffsets[index]));
      return value;
    };
    Value disp = correction(0), fixed = correction(1);
    Value shift = correction(2), shiftFixed = correction(3), dispVirial = correction(4);
    auto atVolume = [&](Value value) -> Value {
      Value numerator = arith::MulFOp::create(body, loc, value, firstVolume);
      return arith::DivFOp::create(body, loc, numerator, cellVolume);
    };
    Value dispersion = arith::AddFOp::create(body, loc, atVolume(disp), fixed);
    Value self = real(body, program.coulombSelfEnergy);
    Value background = arith::SubFOp::create(body, loc, real(body, program.coulombConstantEnergy), self);
    Value coulomb = arith::AddFOp::create(body, loc, self, atVolume(background));
    Value constants = arith::AddFOp::create(body, loc, dispersion, coulomb);
    constants = arith::AddFOp::create(body, loc, constants, real(body, program.ljpmeSelfEnergy));
    Value potential = arith::AddFOp::create(body, loc, energy.energy, constants);
    Value difference = real(body, 0);
    if (config.output.gradient) {
      const auto &gradient = *config.output.gradient;
      SmallVector<Value> arguments = potentialArguments(state);
      for (unsigned slot : gradient.constants) arguments.push_back(point.startValues[slot]);
      if (gradient.seeds)
        arguments.append(gradient.seeds, physical.fields[*gradient.seedField]);
      SmallVector<Type> types{body.getF64Type()};
      SmallVector<int32_t> kinds{int32_t(mdir::md::Request::Energy)};
      SmallVector<int64_t> derivatives{-1};
      for (unsigned argument : gradient.output.scalarArguments) {
        types.push_back(body.getF64Type());
        kinds.push_back(int32_t(mdir::md::Request::Derivative));
        derivatives.push_back(argument);
      }
      for (unsigned argument : gradient.output.fieldArguments) {
        types.push_back(arguments[argument].getType());
        kinds.push_back(int32_t(mdir::md::Request::Derivative));
        derivatives.push_back(argument);
      }
      auto tunable = mdir::md::EvaluateOp::create(body, loc, types,
          FlatSymbolRefAttr::get(body.getContext(), "tunable"), arguments,
          body.getDenseI32ArrayAttr(kinds), body.getDenseI64ArrayAttr(derivatives));
      auto unshifted = buildEnergyEvaluation(body, state, {true, false, false});
      difference = arith::SubFOp::create(body, loc, tunable.getResult(0), unshifted.energy);
      if (config.batchParameters) {
        const auto &plan = *config.batchParameters;
        size_t scalars = gradient.output.scalarArguments.size();
        for (size_t i = 0; i != scalars; ++i) {
          Value index = arith::ConstantIndexOp::create(body, loc, i);
          if (auto error = buildFrameBatchScalarOutput(body, tunable.getResult(i + 1),
                  point.raw, point.frame, point.status, 4, index)) return error;
        }
        auto idsType = MemRefType::get({ShapedType::kDynamic}, body.getI32Type(),
            MemRefLayoutAttrInterface(), body.getI64IntegerAttr(1));
        Value ids = mdir::mdrt::ToBufferOp::create(body, loc, idsType, physical.identities);
        auto fieldType = MemRefType::get({ShapedType::kDynamic}, body.getF64Type(),
            MemRefLayoutAttrInterface(), body.getI64IntegerAttr(1));
        for (size_t i = 0; i != gradient.output.fieldArguments.size(); ++i) {
          Value field = mdir::mdrt::ToBufferOp::create(body, loc, fieldType,
              tunable.getResult(scalars + 1 + i));
          Value offset = arith::ConstantIndexOp::create(body, loc, plan.fieldValues[i]);
          if (auto error = buildFrameBatchFieldOutput(body, field, ids, point.raw,
                  offset, point.frame, point.status)) return error;
        }
        Value bytes = memref::DimOp::create(body, loc, point.rows, zero);
        Value requested = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ugt, bytes, zero);
        Value trips = arith::SelectOp::create(body, loc, requested, one, zero);
        auto rows = buildEntryLoop(body, "device frame parameter rows", zero, trips, one, {},
            [&](OpBuilder &at, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
          Value count = arith::ConstantIndexOp::create(at, loc, plan.parameters);
          Value stride = arith::ConstantIndexOp::create(at, loc, plan.parameters * 8);
          Value offset = arith::MulIOp::create(at, loc, point.frame, stride);
          auto rowType = MemRefType::get({ShapedType::kDynamic}, at.getF64Type(),
              MemRefLayoutAttrInterface(), at.getI64IntegerAttr(1));
          Value row = memref::ViewOp::create(at, loc, rowType, point.rows, offset, ValueRange{count});
          if (auto error = buildFrameBatchOrderedContraction(at, point.raw,
                  point.rowOffsets, point.sourceIndices, point.weights, row)) return error;
          return SmallVector<Value>{};
        });
        if (!rows) return rows.takeError();
      }
    }
    Value total = arith::AddFOp::create(body, loc, potential, difference);
    Value ratio = arith::DivFOp::create(body, loc, firstVolume, cellVolume);
    Value scaledShift = arith::MulFOp::create(body, loc, shift, ratio);
    total = arith::AddFOp::create(body, loc, total, scaledShift);
    total = arith::AddFOp::create(body, loc, total, shiftFixed);
    if (llvm::Error error = buildFrameBatchScalarOutput(body, total,
            point.energies, point.frame, point.status, 2))
      return std::move(error);

    Value nil = real(body, 0);
    Value force = arith::CmpFOp::create(body, loc, arith::CmpFPredicate::ONE, point.asks[0], nil);
    Value virial = arith::CmpFOp::create(body, loc, arith::CmpFPredicate::ONE, point.asks[1], nil);
    Value forceCount = arith::SelectOp::create(body, loc, force, one, zero);
    Value virialCount = arith::SelectOp::create(body, loc, virial, one, zero);
    Value forceOnly = arith::SelectOp::create(body, loc, virial, zero, forceCount);
    auto writeForces = [&](OpBuilder &at, Value forces) -> llvm::Error {
      Type from = program.force == Element::F64 ? Type(at.getF64Type()) : Type(at.getF32Type());
      Type to = program.state == Element::F64 ? Type(at.getF64Type()) : Type(at.getF32Type());
      auto space = at.getI64IntegerAttr(1);
      auto inputType = MemRefType::get({ShapedType::kDynamic, 3}, from, MemRefLayoutAttrInterface(), space);
      auto identityType = MemRefType::get({ShapedType::kDynamic}, at.getI32Type(), MemRefLayoutAttrInterface(), space);
      Value values = mdir::mdrt::ToBufferOp::create(at, loc, inputType, forces);
      Value identities = mdir::mdrt::ToBufferOp::create(at, loc, identityType, physical.identities);
      Value count = memref::DimOp::create(at, loc, values, zero);
      Value width = arith::ConstantIndexOp::create(at, loc, 3 * (to.isF64() ? 8 : 4));
      Value stride = arith::MulIOp::create(at, loc, count, width);
      Value offset = arith::MulIOp::create(at, loc, point.frame, stride);
      auto outputType = MemRefType::get({ShapedType::kDynamic, 3}, to, MemRefLayoutAttrInterface(), space);
      Value destination = memref::ViewOp::create(at, loc, outputType,
          point.gradients, offset, ValueRange{count});
      return buildFrameBatchPositionOutput(at, values, identities,
          destination, point.frame, point.status);
    };
    auto forceLoop = buildEntryLoop(body, "device frame force request", zero, forceOnly, one, {},
        [&](OpBuilder &at, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
      auto result = buildEntryEvaluation(at, state, {false, true, false}, config.sites, complete, scale);
      if (llvm::Error error = writeForces(at, result.physical.forces)) return std::move(error);
      return SmallVector<Value>{};
    });
    if (!forceLoop) return forceLoop.takeError();
    auto virialLoop = buildEntryLoop(body, "device frame virial request", zero, virialCount, one, {},
        [&](OpBuilder &at, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
      auto result = buildEntryEvaluation(at, state, {false, true, true}, config.sites, complete, scale);
      auto forces = buildEntryLoop(at, "device frame shared force request", zero, forceCount, one, {},
          [&](OpBuilder &write, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
        if (llvm::Error error = writeForces(write, result.physical.forces)) return std::move(error);
        return SmallVector<Value>{};
      });
      if (!forces) return forces.takeError();
      // Keep Output::getFrameStrain's multiplication/division order.
      Value d = arith::MulFOp::create(at, loc, dispVirial, firstVolume);
      d = arith::DivFOp::create(at, loc, d, cellVolume);
      Value c = arith::MulFOp::create(at, loc, real(at, program.coulombConstantVirial), firstVolume);
      c = arith::DivFOp::create(at, loc, c, cellVolume);
      Value sum = arith::AddFOp::create(at, loc, d, c);
      Value diagonal = arith::DivFOp::create(at, loc, sum, real(at, 3));
      if (llvm::Error error = buildFrameBatchStrainOutput(at, result.physical.virial,
              diagonal, point.strain, point.frame, point.status)) return std::move(error);
      return SmallVector<Value>{};
    });
    if (!virialLoop) return virialLoop.takeError();
    return SmallVector<Value>{};
  });
  if (!iteration) return iteration.takeError();
  (*iteration)->setAttr("mdrt.segment", builder.getUnitAttr());
  return llvm::Error::success();
}
