// Shared runtime ABI declarations for native and transitional entries.
#include "EntryRuntimeIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
using namespace mdir::driver::builder;
void mdir::driver::builder::emitRuntimeDeclarations(mlir::OpBuilder &builder,
                             const RuntimeDeclarationConfig &config,
                             bool afterModels) {
  using namespace mlir;
  Type f64 = builder.getF64Type(), i64 = builder.getI64Type();
  Type state = config.state64 ? builder.getF64Type() : builder.getF32Type();
  Type force = config.force64 ? builder.getF64Type() : builder.getF32Type();
  auto buffer = [](int64_t count, Type element) {
    return MemRefType::get({count}, element);
  };
  auto vectors = [](Type element) {
    return MemRefType::get({ShapedType::kDynamic, 3}, element);
  };
  Type ids = buffer(ShapedType::kDynamic, builder.getI32Type());
  Type numbers = buffer(ShapedType::kDynamic, f64);
  Type positions = vectors(state), forces = vectors(force);
  auto signature = [&](unsigned integers, unsigned reals) {
    SmallVector<Type> types(integers, i64);
    types.append(reals, f64);
    return types;
  };
  auto declare = [&](StringRef name, TypeRange inputs,
                     TypeRange results = {}, bool cInterface = false) {
    auto function = func::FuncOp::create(
        builder, builder.getUnknownLoc(), name,
        builder.getFunctionType(inputs, results));
    function.setPrivate();
    if (cInterface)
      function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    return function;
  };
  if (!afterModels) {
    declare("mdrtWriteEnergies", signature(1, 5), {}, true);
    declare("mdrtWritePull", {i64, numbers, numbers}, {}, true);
    declare("mdrtWriteFreeEnergy", {i64, numbers}, {}, true);
    declare("mdrtWriteObservables", {i64, numbers}, {}, true);
    declare("mdrtWriteTerms", {numbers}, {}, true);
    if (config.tunableGradient) {
      declare("mdrtWriteTunableGradient", {numbers}, {}, true);
      declare("mdrtWriteTunableGradientField", {i64, numbers, ids}, {}, true);
    }
    declare("mdrtWriteVirial", signature(0, 3), {}, true);
    declare("mdrtWriteFrame", {i64, positions, ids}, {}, true);
    declare("mdrtCheckSpread", {i64, positions, ids}, {}, true);
    if (config.frames) {
      declare("mdrtFrameBoundary", {buffer(6, f64), buffer(2, f64)},
              {positions}, true);
      declare("mdrtWriteFrameForces", {forces, ids}, {}, true);
      declare("mdrtWriteFrameVirial", signature(0, 9), {}, true);
    } else if (config.segments) {
      declare("mdrtPartBoundary",
              {positions, positions, forces, ids, f64,
               buffer(config.partCounts, i64)}, {}, true);
    } else {
      declare("mdrtFinish", {positions, positions, ids}, {}, true);
    }
    return;
  }
  if (config.minimize)
    declare("mdrtWriteMinimization", {i64, f64, f64, forces, ids}, {}, true);
  if (config.stopsMinimization)
    declare("mdrtCheckMinimization", {buffer(2, i64)}, {}, true);
  if (config.solvent)
    declare("mdrtWriteSolvent", signature(0, 2));
  if (config.constraintStops) {
    auto stop = [&](StringRef name, TypeRange inputs, bool reads) {
      auto function = declare(name, inputs);
      auto resources = builder.getStrArrayAttr({"dyn.stop"});
      SmallVector<NamedAttribute> effects;
      if (reads)
        effects.push_back(builder.getNamedAttr("read", resources));
      effects.push_back(builder.getNamedAttr("write", resources));
      function->setAttr("dyn.effects", builder.getDictionaryAttr(effects));
    };
    stop("mdrtStopConstraintFailure", {i64, i64}, false);
    stop("mdrtDeferStops", {}, false);
    stop("mdrtStopPending", {}, true);
  }
  if (config.rescales && config.noseHoover)
    declare("mdrtNoseHooverFactor", signature(1, 1), {f64});
  else if (config.rescales)
    declare("mdrtBussiFactor", signature(2, 4), {f64});
  if (config.bath)
    declare("mdrtAddBath", {f64}, {}, true);
  if (config.barostat) {
    declare("mdrtBarostatStrain", signature(2, 6), {f64});
    declare("mdrtBarostatStrainArea", signature(2, 8), {f64});
    declare("mdrtBarostatStrainHeight", signature(2, 6), {f64});
    declare("mdrtBarostatStrainAxis", signature(3, 6), {f64});
    declare("mdrtSetBox", signature(0, 3), {}, true);
    declare("mdrtSetTilt", signature(0, 3), {}, true);
  }
  if (config.barostatState)
    declare("mdrtSetBarostatState", signature(0, 9), {}, true);
  if (config.checkpoints) {
    SmallVector<Type> inputs = {i64, positions, positions};
    if (config.writesForces)
      inputs.push_back(forces);
    inputs.push_back(ids);
    declare("mdrtWriteCheckpoint", inputs, {}, true);
  }
}
