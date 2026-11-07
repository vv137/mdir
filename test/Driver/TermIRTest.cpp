// Native construction retains ownership and actual SSA operands through printing.
#include "../../lib/Driver/TermIR.h"
#include "../../lib/Driver/SimpleFrameEntryIR.h"
#include "../../lib/Driver/FrameBatchTunableIR.h"
#include "../../lib/Driver/FrameBatchLJIR.h"
#include "../../lib/Driver/SimpleStepEntryIR.h"
#include "../../lib/Driver/MinimizationEntryIR.h"
#include "../../lib/Driver/PullOutputIR.h"
#include "../../lib/Driver/EntryRuntimeIR.h"
#include "../../lib/Driver/EntryLoopIR.h"
#include "../../lib/Driver/PotentialIR.h"
#include "../../lib/Driver/DynamicsIR.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "../../lib/Driver/StagedModelIR.h"
#include "../../lib/Driver/BarostatIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mdir/Dialect/ProgramDialects.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Compiler/Compile.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"
#include <cmath>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Parser/Parser.h"
#include <cstdlib>

using namespace mdir::driver::builder;
using namespace mlir;

static void require(bool condition, llvm::StringRef message) {
  if (!condition) {
    llvm::errs() << message << "\n";
    std::exit(1);
  }
}

static void requireUsedBlockResults(Block &block) {
  for (Operation &operation : block)
    if (operation.getNumResults() && operation.use_empty()) {
      llvm::errs() << "unused native entry row operation: " << operation << "\n";
      require(false, "native entry row produced an unused operation");
    }
}

static void reject(llvm::Error error, llvm::StringRef message) {
  require(bool(error), "invalid term connection was accepted");
  require(llvm::toString(std::move(error)) == message,
          "wrong term connection diagnostic");
}

static void checkModelInput(llvm::StringRef source, bool hasLiteral = false) {
  TermIRPlan plan;
  MLIRContext &context = plan.getContext();
  auto model = StagedModelIR::parse(context, source, "user-model", "model");
  require(bool(model), model ? "" : llvm::toString(model.takeError()));
  auto destination = parseSourceString<ModuleOp>(
      "module { md.particle_set @atoms "
      "md.potential @energy(%x: !md.field<@atoms, 3 x f64>, "
      "%cell: !md.cell, %z: !md.field<@atoms, i32>) -> f64 { "
      "%u = arith.constant 0.0 : f64 md.return %u : f64 } }", &context);
  require(bool(destination), "model destination did not parse");
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(destination->getBody());
  require(!(*model)->cloneDeclarations(builder), "model declarations did not clone");
  auto potential = destination->lookupSymbol<mdir::md::PotentialOp>("energy");
  Block &body = potential.getBody().front();
  builder.setInsertionPoint(&body.back());
  auto result = (*model)->cloneBody(builder, body.getArguments());
  require(bool(result), result ? "" : llvm::toString(result.takeError()));
  body.back().setOperand(0, *result);
  require(succeeded(verify(*destination)), "cloned model did not verify");
  if (hasLiteral) {
    auto attribute = result->getDefiningOp()->getAttrOfType<StringAttr>("user.note");
    require(attribute && attribute.getValue() == "%value md.potential @literal",
            "cloning changed user string contents");
  }
  // Operands from another program in the same context must not escape.
  auto foreign = destination->clone();
  auto foreignPotential = foreign.lookupSymbol<mdir::md::PotentialOp>("energy");
  auto bad = (*model)->cloneBody(builder,
                                  foreignPotential.getBody().front().getArguments());
  require(!bad, "model accepted another program's operands");
  reject(bad.takeError(),
         "staged model operands must belong to the destination program");
  foreign.erase();
  auto wrongArity = (*model)->cloneBody(builder, {});
  require(!wrongArity, "model accepted the wrong arity");
  reject(wrongArity.takeError(),
         "staged model argument count differs from its input boundary");
  SmallVector<Value> wrongTypes(body.getArguments());
  wrongTypes[0] = wrongTypes[2];
  auto wrongType = (*model)->cloneBody(builder, wrongTypes);
  require(!wrongType, "model accepted the wrong operand type");
  reject(wrongType.takeError(),
         "staged model argument type differs from its input boundary");
  TermIRPlan other;
  OpBuilder foreignBuilder(&other.getContext());
  reject((*model)->cloneDeclarations(foreignBuilder),
         "staged model and destination must share their build context");
}

// A consumer may use only exported fragment values. In particular, virial
// reporting borrows the original three extracts, never re-reads the vector.
static void checkDiagonalExports() {
  TermIRPlan plan;
  OpBuilder builder(&plan.getContext());
  auto module = plan.getModule();
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  auto f64 = builder.getF64Type();
  auto report = func::FuncOp::create(builder, loc, "report",
      builder.getFunctionType({f64, f64, f64}, {}));
  report.setPrivate();
  auto entry = func::FuncOp::create(builder, loc, "entry",
      builder.getFunctionType({VectorType::get({9}, f64)}, {VectorType::get({3}, f64)}));
  builder.setInsertionPointToEnd(entry.addEntryBlock());
  auto diagonal = buildVirialDiagonal(builder, entry.getArgument(0));
  SmallVector<Value> scalars{diagonal.xx, diagonal.yy, diagonal.zz};
  auto call = func::CallOp::create(builder, loc, "report", TypeRange{}, scalars);
  func::ReturnOp::create(builder, loc, diagonal.vector);
  mdir::driver::Program program;
  require(!serializeNativeProgram(program, plan), "diagonal exports failed construction");
  require(succeeded(verify(module)), "diagonal consumer did not verify");
  unsigned extracts = 0;
  module.walk([&](vector::ExtractOp) { ++extracts; });
  for (unsigned index = 0; index != 3; ++index)
    require(call.getOperand(index) == scalars[index],
            "virial consumer did not receive the actual scalar");
  require(extracts == 3, "diagonal consumer duplicated extracts");
}

// An attribute containing an evaluate token cannot replace an operation.
static void checkFrameEvaluations() {
  for (bool evaluates : {false, true}) {
    TermIRPlan plan;
    auto &context = plan.getContext();
    OpBuilder builder(&context);
    auto module = plan.getModule();
    builder.setInsertionPointToEnd(module.getBody());
    auto loc = builder.getUnknownLoc();
    mdir::md::ParticleSetOp::create(builder, loc, "atoms");
    auto field = mdir::md::FieldType::get(&context,
        FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
    SmallVector<Type> inputs{field, mdir::md::CellType::get(&context)};
    auto energy = buildScalarPotential(builder, "energy", inputs,
        [](OpBuilder &at, ValueRange) -> Value {
          return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.0));
        });
    require(bool(energy), "frame evaluation potential failed construction");
    auto entry = func::FuncOp::create(builder, loc, "entry", builder.getFunctionType(inputs, {}));
    builder.setInsertionPointToEnd(entry.addEntryBlock());
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    auto loop = scf::ForOp::create(builder, loc, zero, one, one);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      auto note = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
      note->setAttr("note", builder.getStringAttr("md.evaluate "));
      if (evaluates)
        mdir::md::EvaluateOp::create(builder, loc, TypeRange{builder.getF64Type()},
            FlatSymbolRefAttr::get(&context, "energy"), entry.getArguments(),
            builder.getDenseI32ArrayAttr({int32_t(mdir::md::Request::Energy)}),
            builder.getDenseI64ArrayAttr({-1}));
    }
    func::ReturnOp::create(builder, loc);
    auto error = checkFrameEvaluation(*loop.getOperation());
    if (evaluates)
      require(!error, "native frame evaluation was not recognized");
    else
      reject(std::move(error),
             "the program of frames: the start of this program evaluates "
             "nothing, so its loop over frames would give nothing (Builder::emitEntry)");
    require(succeeded(verify(module)), "frame evaluation fixture did not verify");
  }
}

// Complete native bodies keep their operations and arguments at the module
// boundary, even after the transient metadata that described them is gone.
static void checkNativeDescentBody() {
  TermIRPlan plan;
  mdir::driver::Program program;
  program.fields.emplace_back().name = "parameter";
  program.fields.emplace_back().name = "species";
  program.fields.back().isInteger = true;
  program.tables.emplace_back().name = "pairs";
  program.tables.emplace_back().name = "grid";
  program.tables.back().columns = 3;
  program.tupleSets.emplace_back().name = "bonds";
  program.tupleSets.back().arity = 2;
  program.tupleSets.back().fields.emplace_back().name = "length";
  program.tupleSets.emplace_back().name = "sites_linear";
  program.tupleSets.back().arity = 4;
  program.tupleSets.back().reversible = false;
  program.tupleSets.back().fields.emplace_back().name = "a";
  program.tupleSets.back().fields.emplace_back().name = "b";
  auto &context = plan.getContext();
  OpBuilder builder(&context);
  auto module = plan.getModule();
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::md::TupleSetOp::create(builder, loc, "bonds", "atoms", 2,
                               mdir::md::Orientation::Unordered, true);
  mdir::md::TupleSetOp::create(builder, loc, "sites_linear", "atoms", 4,
                               mdir::md::Orientation::Ordered, false);
  auto parameters = buildParameterTypes(context, program);
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  SmallVector<Type> energyInputs{vec, mdir::md::CellType::get(&context)};
  llvm::append_range(energyInputs, parameters);
  auto energy = mdir::md::PotentialOp::create(
      builder, loc, "energy", builder.getFunctionType(energyInputs, {builder.getF64Type()}),
      StringAttr{}, ArrayAttr{}, ArrayAttr{});
  {
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Location> locations(energyInputs.size(), loc);
    builder.createBlock(&energy.getBody(), {}, energyInputs, locations);
    Value zero = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
    mdir::md::ReturnOp::create(builder, loc, ValueRange{zero});
  }
  DescentConstraintGroupIR group{
      4, 2, {5}, {{0, 1, 0}}, {false, false, 1.0e-8, 1.0e-12}, false, false};
  DescentSiteIR site{6, 7, 8, false, false, "spread_sites_linear"};
  auto native = buildNativeDescent(builder, parameters, 4.0, false, {group}, {site});
  Value original = native.getBody().front().getArgument(0);
  program.fields.clear(); program.tables.clear(); program.tupleSets.clear();
  require(!serializeNativeProgram(program, plan), "owned native descent failed its module boundary");
  require(module.lookupSymbol<mdir::dyn::ProgramOp>("descend") == native,
          "native function was reparsed or cloned");
  require(native.getBody().front().getArgument(0) == original,
          "native function lost its actual block argument");
  require(succeeded(verify(module)), "native descent did not verify");
}

static void checkNativeVerletBodies() {
 for (bool timed : {false, true})
  for (bool leapfrog : {false, true})
   for (bool withEnergy : {false, true})
    for (int motion : {0, 1, 2})
     for (bool scales : {false, true})
      for (bool withVirial : {false, true})
       for (bool constrained : {false, true}) {
  if (withEnergy && !withVirial) continue;
  if (motion == 2 && (leapfrog || scales)) continue;
  TermIRPlan plan;
  mdir::driver::Program program;
  program.fields.emplace_back().name = "parameter";
  program.fields.emplace_back().name = "species";
  program.fields.back().isInteger = true;
  program.tables.emplace_back().name = "pairs";
  program.tables.emplace_back().name = "grid";
  program.tables.back().columns = 3;
  program.tupleSets.emplace_back().name = "bonds";
  program.tupleSets.back().arity = 2;
  program.tupleSets.back().fields.emplace_back().name = "length";
  program.tupleSets.emplace_back().name = "sites_linear";
  program.tupleSets.back().arity = 4;
  program.tupleSets.back().reversible = false;
  program.tupleSets.back().fields.emplace_back().name = "a";
  program.tupleSets.back().fields.emplace_back().name = "b";
  auto &context = plan.getContext();
  OpBuilder builder(&context);
  auto module = plan.getModule();
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::md::TupleSetOp::create(builder, loc, "bonds", "atoms", 2,
                               mdir::md::Orientation::Unordered, true);
  mdir::md::TupleSetOp::create(builder, loc, "sites_linear", "atoms", 4,
                               mdir::md::Orientation::Ordered, false);
  auto parameters = buildParameterTypes(context, program);
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  SmallVector<Type> energyInputs{vec, mdir::md::CellType::get(&context)};
  llvm::append_range(energyInputs, parameters);
  if (timed) energyInputs.push_back(builder.getF64Type());
  auto energy = mdir::md::PotentialOp::create(
      builder, loc, "energy", builder.getFunctionType(energyInputs, {builder.getF64Type()}),
      StringAttr{}, ArrayAttr{}, ArrayAttr{});
  {
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Location> locations(energyInputs.size(), loc);
    builder.createBlock(&energy.getBody(), {}, energyInputs, locations);
    Value zero = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
    mdir::md::ReturnOp::create(builder, loc, ValueRange{zero});
  }
  StringRef name = withEnergy ? "step_energy" : "step";
  DescentSiteIR site{6, 7, 8, false, false, "spread_sites_linear"};
  StepNoiseIR noise{motion != 0, motion == 2, {123, motion == 2 ? 1.0 : 0.8, 0.1}, 2.5};
  if (constrained) {
    auto stop = [&](StringRef symbol, TypeRange inputs, bool reads) {
      auto function = func::FuncOp::create(builder, loc, symbol, builder.getFunctionType(inputs, {}));
      function.setPrivate();
      auto resources = builder.getStrArrayAttr({"dyn.stop"});
      SmallVector<NamedAttribute> effects;
      if (reads) effects.push_back(builder.getNamedAttr("read", resources));
      effects.push_back(builder.getNamedAttr("write", resources));
      function->setAttr("dyn.effects", builder.getDictionaryAttr(effects));
    };
    auto i64 = builder.getI64Type();
    stop("mdrtDeferStops", {}, false);
    stop("mdrtStopConstraintFailure", TypeRange{i64, i64}, false);
    stop("mdrtStopPending", {}, true);
  }
  DescentConstraintGroupIR group{4, 2, {5}, {{0, 1, 0}},
      {false, false, 1.0e-8, 1.0e-12}, false, false};
  SmallVector<DescentConstraintGroupIR> groups;
  if (constrained) groups.push_back(group);
  bool measured = measuresHalfSteps(constrained, noise.brownian) && withEnergy;
  std::string combination = describeStepCombination(leapfrog, noise, constrained,
      withEnergy, measured, true);
  auto constructed = buildNativeStep(builder,
      {name.str(), withEnergy, withVirial, scales, measured, false}, parameters, leapfrog, timed,
      {site}, false, false, {}, noise, groups, false);
  if (!constructed) {
    llvm::errs() << combination << ": " << llvm::toString(constructed.takeError()) << "\n";
    std::exit(1);
  }
  auto native = *constructed;
  Value original = native.getBody().front().getArgument(0);
  program.fields.clear(); program.tables.clear(); program.tupleSets.clear();
  require(!serializeNativeProgram(program, plan), std::string("owned native Verlet failed its module boundary: ") + combination);
  require(module.lookupSymbol<mdir::dyn::ProgramOp>(name) == native,
          std::string("native function was reparsed or cloned: ") + combination);
  require(native.getBody().front().getArgument(0) == original,
          std::string("native function lost its actual block argument: ") + combination);
  require(succeeded(verify(module)), std::string("native Verlet did not verify: ") + combination);
}
}

static void checkNativeStepReturnGuard() {
  TermIRPlan plan;
  auto &context = plan.getContext();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(plan.getModule().getBody());
  mdir::md::ParticleSetOp::create(builder, builder.getUnknownLoc(), "atoms");
  StepNoiseIR noise;
  auto combination = describeStepCombination(true, noise, false, true, false, false);
  // A malformed internal request must fail construction before IR verification.
  auto result = buildNativeStep(builder, {"guard", true, false, false, false, false}, {}, true, false);
  require(!result, std::string("native return guard accepted an invalid layout: ") + combination);
  auto message = llvm::toString(result.takeError());
  require(message.find(combination) != std::string::npos,
          std::string("native return guard omitted its combination: ") + combination);
  require(message.find("return has 3 values but signature has 4 results") != std::string::npos,
          std::string("native return guard omitted its result counts: ") + combination);
}

static void checkNativeBarePotentialBody() {
  TermIRPlan plan;
  mdir::driver::Program metadata;
  metadata.fields.emplace_back().name = "parameter";
  auto &context = plan.getContext();
  OpBuilder builder(&context);
  auto module = plan.getModule();
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  auto atoms = FlatSymbolRefAttr::get(&context, "atoms");
  SmallVector<Type> inputs{mdir::md::FieldType::get(&context, atoms, 3, builder.getF64Type()),
                           mdir::md::CellType::get(&context)};
  llvm::append_range(inputs, buildParameterTypes(context, metadata));
  double constant = 2.5;
  auto builtPotential = buildScalarPotential(builder, "energy", inputs,
      [&](OpBuilder &at, ValueRange) -> Value {
        return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(constant));
      });
  require(bool(builtPotential), "native scalar potential construction failed");
  auto potential = *builtPotential;
  Value original = potential.getBody().front().getArgument(0);
  constant = -1.0;
  metadata.fields.clear();
  require(!serializeNativeProgram(metadata, plan), "owned native potential failed its boundary");
  require(module.lookupSymbol<mdir::md::PotentialOp>("energy") == potential,
          "native potential was reparsed or cloned");
  require(potential.getBody().front().getArgument(0) == original,
          "native potential lost its actual argument");
  auto value = cast<arith::ConstantOp>(potential.getBody().front().front());
  require(cast<FloatAttr>(value.getValue()).getValueAsDouble() == 2.5,
          "native potential retained transient construction state");
  require(succeeded(verify(module)), "native potential did not verify");
  builder.setInsertionPointToEnd(module.getBody());
  auto failed = buildScalarPotential(builder, "failed", inputs,
      [](OpBuilder &, ValueRange) -> llvm::Expected<Value> {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "staged construction boundary failure");
      });
  require(!failed, "native scalar potential swallowed a construction error");
  require(llvm::toString(failed.takeError()) == "staged construction boundary failure",
          "native scalar potential changed its construction error");
  require(!module.lookupSymbol<mdir::md::PotentialOp>("failed"),
          "failed scalar potential retained an incomplete body");

}


static void checkStaticRestartControl() {
  for (bool minimize : {false, true}) {
    mdir::driver::Control control;
    control.segments = true;
    control.minimize = minimize;
    control.restartInput = "checkpoint.h5";
    auto refused = checkEntryControl(control);
    require(bool(refused), "static restart control reached native construction");
    reject(std::move(refused), segmentsStaticRestartError);
    control.segments = false;
    require(!checkEntryControl(control), "ordinary restart was refused by the segment guard");
    control.segments = true;
    control.restartInput.clear();
    require(!checkEntryControl(control), "activation continuation was refused by the static guard");
  }
}

static void checkNativeThermalStepEntry(bool leapfrog, bool constrained, bool nose, bool com, bool barostat = false, bool exact = false, bool trotter = false, bool firstOrderWork = false, bool everyStep = false) {
  TermIRPlan plan;
  auto &context = plan.getContext();
  auto module = plan.getModule();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::driver::Program metadata{};
  metadata.entry = "native_thermal_steps";
  metadata.state = metadata.force = metadata.mass = metadata.parameter = mdir::driver::Element::F64;
  if (constrained) {
    mdir::md::TupleSetOp::create(builder, loc, "bonds", "atoms", 2,
        mdir::md::Orientation::Unordered, true);
    metadata.tupleSets.emplace_back();
    auto &set = metadata.tupleSets.back();
    set.name = "bonds";
    set.arity = 2;
    set.fields = {{"length", {}, false}};
  }
  if (barostat) metadata.startValues = {{"baro_constant", 0.0},
      {"rest_edge0", 1.0}, {"rest_edge1", 1.0}, {"rest_edge2", 1.0}, {"baro_energy_constant", 0.0}};
  RuntimeDeclarationConfig runtime{};
  runtime.state64 = runtime.force64 = runtime.rescales = runtime.bath = true;
  runtime.noseHoover = nose;
  runtime.barostat = barostat;
  runtime.barostatState = everyStep;
  runtime.constraintStops = constrained;
  emitRuntimeDeclarations(builder, runtime, false);
  emitRuntimeDeclarations(builder, runtime, true);
  auto parameters = buildParameterTypes(context, metadata);
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  SmallVector<Type> inputs{vec, mdir::md::CellType::get(&context)};
  llvm::append_range(inputs, parameters);
  auto energy = buildScalarPotential(builder, "energy", inputs,
      [](OpBuilder &at, ValueRange) -> Value {
        return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.125));
      });
  require(bool(energy), "native thermal potential did not construct");
  std::vector<DescentConstraintGroupIR> groups;
  if (constrained) groups.push_back({0, 2, {1}, {{0, 1, 0}},
      {false, false, 1.0e-8, 1.0e-12}, false, false});
  bool measured = measuresHalfSteps(constrained, false);
  auto step = buildNativeStep(builder, {"step"}, parameters, leapfrog, false,
      {}, barostat, false, {}, {}, groups);
  auto reporting = buildNativeStep(builder, {"step_energy", true, true, false, measured},
      parameters, leapfrog, false, {}, barostat, false, {}, {}, groups);
  require(bool(step) && bool(reporting), "native thermal programs did not construct");
  mdir::driver::Control control;
  control.precision = mdir::driver::Precision::Double;
  control.integrator = leapfrog ? mdir::driver::Integrator::Leapfrog : mdir::driver::Integrator::VelocityVerlet;
  control.numSteps = 8;
  control.energyPeriod = 4;
  control.thermostat = true;
  control.thermostatMethod = nose ? mdir::driver::ThermostatMethod::NoseHoover
      : mdir::driver::ThermostatMethod::VRescale;
  control.thermostatPeriod = everyStep ? 1 : 2;
  control.comPeriod = com ? control.thermostatPeriod : 0;
  control.barostat = barostat;
  control.barostatPeriod = control.thermostatPeriod;
  control.barostatWork = trotter ? (firstOrderWork ? mdir::driver::BarostatWork::TrotterFirstOrder : mdir::driver::BarostatWork::Trotter) : exact ? mdir::driver::BarostatWork::Exact : mdir::driver::BarostatWork::FirstOrder;
  control.anisotropic = (exact || trotter) && constrained;
  control.semiIsotropic = (exact || trotter) && !constrained && !com;
  SimpleStepEntryIRConfig config{control.getCouplingPeriod() - (trotter && !everyStep ? 2 : 1),
      control.numSteps / control.getEnergyLoopPeriod()};
  config.couplingIntervals = control.getEnergyLoopPeriod() / control.getCouplingPeriod() - 1;
  config.leapfrog = control.integrator == mdir::driver::Integrator::Leapfrog;
  config.removesMotion = control.comPeriod > 0;
  config.rescales = control.thermostat && !control.isLangevin();
  config.noseHoover = control.isNoseHoover();
  config.bathParameters = control.thermostat;
  config.seed = static_cast<int64_t>(control.seed);
  config.freedom = 3.0;
  config.targetKinetic = 0.5 * config.freedom * mdir::driver::units::boltzmann * control.temperature;
  config.decay = std::exp(-static_cast<double>(control.getCouplingPeriod()) * control.timestep / control.tauT);
  if (constrained) config.constraints = {{0, 2, {1}, false}};
  if (barostat) {
    config.barostat = true;
    config.exactBarostat = exact;
    config.trotter = trotter;
    config.countsAfterScaling = trotter && (!firstOrderWork || everyStep);
    config.everyStep = everyStep;
    config.strainCoupling = control.anisotropic ? StrainCoupling::Anisotropic
        : control.semiIsotropic ? StrainCoupling::SemiIsotropic : StrainCoupling::Isotropic;
    if (exact) config.barostatEnergyConstant = 4;
    config.barostatConstant = 0;
    config.referenceEdges = std::array<unsigned, 3>{1, 2, 3};
    double bar = 1.01325;
    config.barostatParameters = BarostatParameters{
        {control.compressibilities[0] / bar, control.compressibilities[1] / bar,
         control.compressibilities[2] / bar},
        control.pressure * bar, control.compressibility / bar,
        static_cast<double>(control.barostatPeriod) * control.timestep / control.tauP,
        mdir::driver::units::boltzmann * control.temperature, control.compressibilityZ / bar,
        control.surfaceTension * control.surfaces * mdir::driver::units::dynePerCmToBarNm};
  }
  if (constrained || barostat) {
    auto coupling = buildNativeStep(builder, {"step_thermostat", barostat && !trotter, barostat && (!trotter || !firstOrderWork || everyStep), trotter, measured},
        parameters, leapfrog, false, {}, barostat, false, {}, {}, groups);
    require(bool(coupling), "native thermal measured step did not construct");
    config.couplingStep = *coupling;
    if (trotter) {
      auto scaled = buildNativeStep(builder, {"step_trotter_energy", true, true, true, measured},
          parameters, leapfrog, false, {}, true, false, {}, {}, groups);
      require(bool(scaled), "native Trotter scaled program did not construct");
      config.scaledEnergyStep = *scaled;
      if (!everyStep) {
        auto pressure = buildNativeStep(builder, {"step_virial", false, true},
            parameters, leapfrog, false, {}, true, false, {}, {}, groups);
        require(bool(pressure), "native Trotter pressure program did not construct");
        config.pressureStep = *pressure;
        auto invalid = config;
        invalid.pressureStep = {};
        auto missing = buildSimpleStepEntry(builder, metadata, *step, *reporting, invalid);
        require(!missing, "native Trotter accepted a missing pressure step");
        reject(missing.takeError(), "native Trotter entry requires pressure and scaled energy programs");
      } else {
        auto invalid = config;
        invalid.countsAfterScaling = false;
        auto missing = buildSimpleStepEntry(builder, metadata, *step, *reporting, invalid);
        require(!missing, "native period-one Trotter accepted incomplete pressure state");
        reject(missing.takeError(), "native period-one Trotter requires pressure state after scaling");
      }
    }
    auto invalid = config;
    invalid.couplingStep = {};
    auto missing = buildSimpleStepEntry(builder, metadata, *step, *reporting, invalid);
    require(!missing, "native thermal entry accepted a missing measured program");
    reject(missing.takeError(), "native step entry thermostat results do not match its integrator");
  }
  if (barostat) {
    auto missing = config;
    missing.barostatConstant.reset();
    auto failed = buildSimpleStepEntry(builder, metadata, *step, *reporting, missing);
    require(!failed, "native first-order barostat accepted a missing constant role");
    reject(failed.takeError(), "native barostat requires coupling parameters");
  }
  auto entry = buildSimpleStepEntry(builder, metadata, *step, *reporting, config);
  if (!entry) {
    llvm::errs() << "native thermal entry: " << llvm::toString(entry.takeError()) << "\n";
    std::exit(1);
  }
  auto loop = *entry->getOps<scf::ForOp>().begin();
  unsigned factors = 0;
  entry->walk([&](func::CallOp call) {
    if (call.getCallee() != (nose ? "mdrtNoseHooverFactor" : "mdrtBussiFactor")) return;
    ++factors;
    Value completed = call.getOperand(nose ? 0 : 1);
    require(completed.getDefiningOp<arith::AddIOp>() && loop->isProperAncestor(completed.getDefiningOp()),
        "native thermal factor uses a stale closing step");
  });
  require(factors == 2, "native thermal entry lost coupling or closing-row factors");
  if (barostat) {
    unsigned strainCalls = 0;
    entry->walk([&](func::CallOp call) {
      if (call.getCallee() == "mdrtBarostatStrain" || call.getCallee() == "mdrtBarostatStrainAxis" ||
          call.getCallee() == "mdrtBarostatStrainArea" || call.getCallee() == "mdrtBarostatStrainHeight") ++strainCalls;
    });
    unsigned perStrain = control.anisotropic ? 3 : control.semiIsotropic ? 2 : 1;
    require(strainCalls == 2 * perStrain, "native barostat lost a coupling strain");
    if (exact) {
      unsigned reevaluations = 0;
      loop.walk([&](mdir::md::EvaluateOp operation) {
        if (operation.getNumResults() == 2) ++reevaluations;
      });
      require(reevaluations == 2, "native exact barostat lost a scaled-position evaluation");
    }
    entry->walk([&](mdir::dyn::StepOp call) {
      auto cell = call.getOperand(4).getDefiningOp<mdir::md::OrthorhombicCellOp>();
      bool scaled = trotter && (call.getCalleeAttr().getValue() == "step_thermostat" ||
          call.getCalleeAttr().getValue() == "step_trotter_energy");
      require(cell && (scaled ? bool(cell.getOperand(0).getDefiningOp<arith::MulFOp>())
                              : bool(cell.getOperand(0).getDefiningOp<memref::LoadOp>())),
          "native barostat step lost its live or scaled cell");
      require(loop->isProperAncestor(cell), "native first-order cell load escaped its schedule");
    });
  }
  if (everyStep) {
    auto checkState = [&](func::FuncOp function, unsigned evaluations) {
      unsigned sets = 0;
      function.walk([&](func::CallOp call) { sets += call.getCallee() == "mdrtSetBarostatState"; });
      unsigned rootEvaluations = llvm::range_size(function.getOps<mdir::md::EvaluateOp>());
      require(sets == 3, "native period-one entry lost initial or closing pressure state");
      require(rootEvaluations == evaluations, "native period-one entry evaluated the wrong restart state");
    };
    checkState(*entry, 1);
    const auto initialValues = metadata.startValues;
    for (bool stored : {false, true})
      for (bool recompute : {false, true}) {
        metadata.startValues = initialValues;
        auto restartConfig = config;
        restartConfig.restart = true;
        restartConfig.recomputeForces = recompute;
        metadata.takesForces = !recompute;
        if (stored) {
          std::array<unsigned, 9> slots;
          for (unsigned slot = 0; slot != 9; ++slot) {
            slots[slot] = metadata.startValues.size();
            metadata.startValues.push_back({"bstate" + std::to_string(slot), 0.0});
          }
          restartConfig.barostatState = slots;
        }
        metadata.entry = "native_period_one_restart_" + std::to_string(stored) + std::to_string(recompute);
        auto restarted = buildSimpleStepEntry(builder, metadata, *step, *reporting, restartConfig);
        if (!restarted) {
          llvm::errs() << metadata.entry << ": " << llvm::toString(restarted.takeError()) << "\n";
          std::exit(1);
        }
        checkState(*restarted, unsigned(recompute) + unsigned(!stored));
      }
    metadata.startValues = initialValues;
  } else {
  auto production = mdir::compiler::getSegmentsControl(control);
  require(production.energyPeriod == 0 && production.framePeriod == 0 &&
          production.checkpointPeriod == 0 && production.numSteps == production.getCouplingPeriod(),
      "native coupled parts fixture differs from production control");
  auto partsConfig = config;
  partsConfig.segments = production.segments;
  partsConfig.energyIntervals.reset();
  // Builder's appended closing nest has one energy interval, zero complete
  // coupling periods and zero plain steps. Its actual bounds come from the host.
  partsConfig.couplingIntervals = 0;
  partsConfig.plainSteps = 0;
  metadata.takesForces = true;
  metadata.entry = "native_thermal_parts";
  auto positions = MemRefType::get({ShapedType::kDynamic, 3}, builder.getF64Type());
  auto ids = MemRefType::get({ShapedType::kDynamic}, builder.getI32Type());
  auto boundary = func::FuncOp::create(builder, loc, "mdrtPartBoundary",
      builder.getFunctionType({positions, positions, positions, ids, builder.getF64Type(),
          MemRefType::get({9}, builder.getI64Type())}, {}));
  boundary.setPrivate();
  boundary->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto segmented = buildSimpleStepEntry(builder, metadata, *step, *reporting, partsConfig);
  if (!segmented) {
    llvm::errs() << "native thermal parts: " << llvm::toString(segmented.takeError()) << "\n";
    std::exit(1);
  }
  unsigned partFactors = 0;
  scf::ForOp partLoop;
  for (auto candidate : segmented->getOps<scf::ForOp>())
    if (candidate.getNumRegionIterArgs() == 3) partLoop = candidate;
  require(bool(partLoop), "native thermal parts have no carried-state loop");
  partLoop.walk([&](func::CallOp call) {
    if (call.getCallee() != (nose ? "mdrtNoseHooverFactor" : "mdrtBussiFactor")) return;
    ++partFactors;
    auto completed = call.getOperand(nose ? 0 : 1).getDefiningOp<arith::AddIOp>();
    require(completed && partLoop->isProperAncestor(completed),
        "native thermal parts use a stale closing step");
  });
  require(partFactors == 3, "native thermal parts lost an outer, inner or reporting coupling");
  bool actualBoundary = false;
  for (auto call : partLoop.getBody()->getOps<mdir::mdrt::HostCallOp>())
    if (call.getCalleeAttr().getValue() == "mdrtPartBoundary") {
      require(call.getInPlace() && llvm::equal(call.getOperands().take_front(3), partLoop.getRegionIterArgs()),
          "native thermal part boundary lost actual state or in-place resorting");
      actualBoundary = true;
    }
  require(actualBoundary, "native thermal parts have no physical boundary");
  unsigned checkedPlainRows = 0;
  partLoop.walk([&](scf::ForOp row) {
    auto cast = row.getUpperBound().getDefiningOp<arith::IndexCastOp>();
    if (!cast) return;
    auto load = cast->getOperand(0).getDefiningOp<memref::LoadOp>();
    if (!load || load.getIndices().size() != 1) return;
    auto slot = load.getIndices()[0].getDefiningOp<arith::ConstantIndexOp>();
    // Host count slot 4 selects the plain energy row, without coupling.
    if (!slot || slot.value() != 4) return;
    requireUsedBlockResults(*row.getBody());
    ++checkedPlainRows;
  });
  require(checkedPlainRows == 1, "native parts did not check their plain row results");
  if (barostat) {
    unsigned strains = 0, reevaluations = 0;
    partLoop.walk([&](func::CallOp call) {
      if (call.getCallee() == "mdrtBarostatStrain" || call.getCallee() == "mdrtBarostatStrainAxis" ||
          call.getCallee() == "mdrtBarostatStrainArea" || call.getCallee() == "mdrtBarostatStrainHeight") ++strains;
    });
    partLoop.walk([&](mdir::md::EvaluateOp op) { if (op.getNumResults() == 2) ++reevaluations; });
    unsigned perStrain = control.anisotropic ? 3 : control.semiIsotropic ? 2 : 1;
    require(strains == 3 * perStrain, "native barostat parts lost a closing strain");
    require(!exact || reevaluations == 3, "native exact parts lost a scaled evaluation");
    partLoop.walk([&](mdir::dyn::StepOp call) {
      auto cell = call.getOperand(4).getDefiningOp<mdir::md::OrthorhombicCellOp>();
      bool scaled = trotter && (call.getCalleeAttr().getValue() == "step_thermostat" ||
          call.getCalleeAttr().getValue() == "step_trotter_energy");
      require(cell && (scaled ? bool(cell.getOperand(0).getDefiningOp<arith::MulFOp>())
                              : bool(cell.getOperand(0).getDefiningOp<memref::LoadOp>())) &&
          partLoop->isProperAncestor(cell), "native barostat parts lost their live or scaled cell");
    });
  }
  if (trotter) {
    unsigned pressureCalls = 0, scaledCalls = 0;
    partLoop.walk([&](mdir::dyn::StepOp call) {
      auto name = call.getCalleeAttr().getValue();
      pressureCalls += name == "step_virial";
      scaledCalls += name == "step_thermostat" || name == "step_trotter_energy";
    });
    require(pressureCalls == 3 && scaledCalls == 3,
        "native Trotter parts lost an outer, inner or reporting pressure/scaling pair");
  }
  }
  require(succeeded(verify(module)), "native thermal module did not verify");
}

static void checkNativeCenterStepEntry() {
  TermIRPlan plan;
  auto &context = plan.getContext();
  auto module = plan.getModule();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::driver::Program metadata{};
  metadata.state = metadata.force = metadata.mass = metadata.parameter = mdir::driver::Element::F64;
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  auto energy = buildScalarPotential(builder, "energy",
      {vec, mdir::md::CellType::get(&context)},
      [](OpBuilder &at, ValueRange) -> Value {
        return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.125));
      });
  require(bool(energy), "native COM potential did not construct");
  auto step = buildNativeStep(builder, {"step"}, {}, false, false);
  auto reporting = buildNativeStep(builder, {"step_energy", true, true}, {}, false, false);
  require(bool(step) && bool(reporting), "native COM programs did not construct");
  RuntimeDeclarationConfig runtime{};
  runtime.state64 = runtime.force64 = runtime.bath = true;
  emitRuntimeDeclarations(builder, runtime, false);
  emitRuntimeDeclarations(builder, runtime, true);
  mdir::driver::Control control;
  control.numSteps = 12;
  control.energyPeriod = 6;
  control.comPeriod = 3;
  SimpleStepEntryIRConfig config{control.getCouplingPeriod() - 1,
      control.numSteps / control.getEnergyLoopPeriod()};
  config.couplingIntervals = control.getEnergyLoopPeriod() / control.getCouplingPeriod() - 1;
  metadata.entry = "native_center_steps";
  auto entry = buildSimpleStepEntry(builder, metadata, *step, *reporting, config);
  require(bool(entry), "native COM entry did not construct");
  auto energies = *entry->getOps<scf::ForOp>().begin();
  auto childLoops = energies.getBody()->getOps<scf::ForOp>();
  auto it = childLoops.begin();
  auto couples = *it++;
  auto finalPlain = *it;
  require(llvm::equal(finalPlain.getInitArgs(), couples.getResults()),
      "native COM final period starts from stale coupled state");
  auto closing = *couples.getBody()->getOps<mdir::dyn::StepOp>().begin();
  auto yielded = cast<scf::YieldOp>(couples.getBody()->getTerminator());
  auto coupledVelocity = yielded.getOperand(1).getDefiningOp<mdir::md::MapParticlesOp>();
  require(coupledVelocity && coupledVelocity.getOperand(0) == closing.getResult(1),
      "native COM removal does not consume the closing step velocity");
  metadata.entry = "native_center_without_rows";
  config.energyIntervals.reset();
  config.couplingIntervals = control.numSteps / control.getCouplingPeriod();
  auto bare = buildSimpleStepEntry(builder, metadata, *step, {}, config);
  require(bool(bare), "native COM entry without rows did not construct");
  metadata.entry = "native_center_zero_period_loop";
  control.comPeriod = control.energyPeriod;
  config.plainSteps = control.getCouplingPeriod() - 1;
  config.energyIntervals = control.numSteps / control.getEnergyLoopPeriod();
  config.couplingIntervals = control.getEnergyLoopPeriod() / control.getCouplingPeriod() - 1;
  auto zero = buildSimpleStepEntry(builder, metadata, *step, *reporting, config);
  require(bool(zero), "native COM entry with one period per row did not construct");
  require(succeeded(verify(module)), "native COM entry module did not verify");
}

static void checkNativeMinimizationEntry() {
  TermIRPlan plan;
  auto &context = plan.getContext();
  auto module = plan.getModule();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::driver::Program metadata{};
  metadata.entry = "native_minimization";
  metadata.state = metadata.force = metadata.mass = metadata.parameter = mdir::driver::Element::F64;
  metadata.reorders = true;
  metadata.orderWidth = 0.5;
  metadata.fields.push_back({"q", {}, false});
  auto parameters = buildParameterTypes(context, metadata);
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  SmallVector<Type> inputs{vec, mdir::md::CellType::get(&context)};
  llvm::append_range(inputs, parameters);
  auto energy = buildScalarPotential(builder, "energy", inputs,
      [](OpBuilder &at, ValueRange) -> Value {
        return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.125));
      });
  require(bool(energy), "native minimization entry potential did not construct");
  auto descend = buildNativeDescent(builder, parameters, 3.0, false);
  MinimizationEntryIRConfig config;
  config.steps = 8;
  config.energyPeriod = 2;
  config.framePeriod = 4;
  config.firstSize = 0.001;
  config.cap = 0.01;
  config.checkpointPeriod = 4;
  config.topology = true;
  config.terms = {"energy"};
  RuntimeDeclarationConfig runtime{};
  runtime.state64 = runtime.force64 = true;
  runtime.minimize = true;
  runtime.stopsMinimization = config.stops;
  runtime.checkpoints = config.checkpointPeriod > 0;
  runtime.writesForces = metadata.writesForces;
  emitRuntimeDeclarations(builder, runtime, false);
  emitRuntimeDeclarations(builder, runtime, true);
  auto missing = buildMinimizationEntry(builder, metadata, {}, config);
  require(!missing, "native minimization accepted a missing descent program");
  reject(missing.takeError(), "native minimization has no descent program");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry), "failed native minimization retained its entry");
  auto built = buildMinimizationEntry(builder, metadata, descend, config);
  require(bool(built), "native minimization entry did not construct");
  auto frames = *built->getOps<scf::ForOp>().begin();
  auto rows = *frames.getBody()->getOps<scf::ForOp>().begin();
  auto steps = *rows.getBody()->getOps<scf::ForOp>().begin();
  auto trial = *steps.getBody()->getOps<mdir::dyn::StepOp>().begin();
  require(llvm::equal(rows.getInitArgs(), frames.getRegionIterArgs()) &&
          llvm::equal(steps.getInitArgs(), rows.getRegionIterArgs()),
      "native minimization reads stale enclosing controller state");
  require(trial.getOperand(0) == steps.getRegionIterArgs()[0] &&
          trial.getOperand(1) == steps.getRegionIterArgs()[1] &&
          trial.getOperand(4) == steps.getRegionIterArgs()[3],
      "native minimization trial reads stale positions, forces or step length");
  metadata.entry = "native_unstopped_minimization";
  config.stops = false;
  config.framePeriod = 0;
  auto unstopped = buildMinimizationEntry(builder, metadata, descend, config);
  require(bool(unstopped), "native minimization without stop rows did not construct");
  metadata.entry = "native_parts_minimization";
  mdir::driver::Control control;
  control.minimize = true;
  auto segmentsControl = mdir::compiler::getSegmentsControl(control);
  // Match emitEntry's mapping, including stopsMinimization(). The host checks
  // convergence between parts; the entry reads counts from part memory.
  config.segments = segmentsControl.segments;
  config.steps = segmentsControl.numSteps;
  config.energyPeriod = segmentsControl.energyPeriod;
  config.framePeriod = segmentsControl.framePeriod;
  config.checkpointPeriod = segmentsControl.checkpointPeriod;
  config.stops = segmentsControl.minimize && !segmentsControl.segments;
  require(config.steps == std::max<int64_t>(1, segmentsControl.getCouplingPeriod()) &&
          config.energyPeriod == 0 && config.framePeriod == 0 &&
          config.checkpointPeriod == 0 && !config.stops,
      "native minimization fixture differs from production segments control");
  auto positions = MemRefType::get({ShapedType::kDynamic, 3}, builder.getF64Type());
  auto ids = MemRefType::get({ShapedType::kDynamic}, builder.getI32Type());
  auto partMemory = MemRefType::get({config.partCounts}, builder.getI64Type());
  auto boundary = func::FuncOp::create(builder, loc, "mdrtPartBoundary",
      builder.getFunctionType({positions, positions, positions, ids, builder.getF64Type(), partMemory}, {}));
  boundary.setPrivate();
  boundary->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto segmented = buildMinimizationEntry(builder, metadata, descend, config);
  require(bool(segmented), "native minimization parts did not construct");
  scf::ForOp parts;
  for (auto loop : segmented->getOps<scf::ForOp>())
    if (loop.getNumRegionIterArgs() == 4) parts = loop;
  require(bool(parts), "native minimization has no part state loop");
  auto partFrames = *parts.getBody()->getOps<scf::ForOp>().begin();
  require(llvm::equal(partFrames.getInitArgs(), parts.getRegionIterArgs()),
      "native minimization parts read stale controller state");
  bool currentBoundary = false;
  for (auto call : parts.getBody()->getOps<mdir::mdrt::HostCallOp>())
    if (call.getOperand(0) == parts.getRegionIterArgs()[0] &&
        call.getOperand(2) == parts.getRegionIterArgs()[1]) {
      require(call.getInPlace(), "native minimization boundary lost in-place resorting");
      currentBoundary = true;
    }
  module.walk([&](mdir::mdrt::HostCallOp call) {
    bool boundary = call.getCalleeAttr().getValue() == "mdrtPartBoundary";
    require(bool(call.getInPlace()) == boundary,
        "native minimization host call has the wrong in-place contract");
  });
  require(currentBoundary, "native minimization boundary reads stale positions or forces");
  require(parts.getInitArgs()[3] == segmented->getArguments()[segmented->getNumArguments() - 2],
      "native minimization parts ignore the incoming step size");
  metadata.entry = "native_invalid_parts_minimization";
  auto rejectConfig = [&](const MinimizationEntryIRConfig &invalid, StringRef message) {
    auto rejected = buildMinimizationEntry(builder, metadata, descend, invalid);
    require(!rejected, "native minimization parts accepted invalid mode configuration");
    reject(rejected.takeError(), message);
    require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
        "invalid native minimization parts retained an entry");
  };
  auto invalid = config;
  invalid.restart = true;
  rejectConfig(invalid, segmentsStaticRestartError);
  invalid = config;
  invalid.energyPeriod = 1;
  rejectConfig(invalid, "native minimization parts require a zero energy period");
  invalid = config;
  invalid.framePeriod = 1;
  rejectConfig(invalid, "native minimization parts require a zero frame period");
  invalid = config;
  invalid.checkpointPeriod = 1;
  rejectConfig(invalid, "native minimization parts require a zero checkpoint period");
  invalid = config;
  invalid.partCounts = 1;
  rejectConfig(invalid, "native minimization part counts require start and step slots");
  invalid = config;
  invalid.stops = true;
  rejectConfig(invalid, "native minimization parts require host convergence checks");
  config.segments = false;
  metadata.entry = "native_missing_hosted_minimization";
  config.needsHostedEvaluation = true;
  auto omitted = buildMinimizationEntry(builder, metadata, descend, config);
  require(!omitted, "native minimization silently dropped its hosted role");
  reject(omitted.takeError(), "native minimization requires the hosted-evaluation role");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry), "missing native minimization role retained an entry");
  metadata.fields.clear();
  require(succeeded(verify(module)), "native minimization module did not verify");
}

static void checkNativeConstraintStepEntry() {
  TermIRPlan plan;
  auto &context = plan.getContext();
  auto module = plan.getModule();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::md::TupleSetOp::create(builder, loc, "sites_linear", "atoms", 4,
      mdir::md::Orientation::Ordered, false);
  mdir::md::TupleSetOp::create(builder, loc, "settles", "atoms", 3,
      mdir::md::Orientation::Ordered, true);
  mdir::driver::Program metadata{};
  metadata.state = metadata.force = metadata.mass = metadata.parameter = mdir::driver::Element::F64;
  metadata.takesForces = metadata.writesForces = metadata.reorders = true;
  metadata.orderWidth = 0.5;
  metadata.tupleSets.emplace_back();
  auto &site = metadata.tupleSets.back();
  site.name = "sites_linear";
  site.arity = 4;
  site.reversible = false;
  site.fields = {{"a", {}, false}, {"b", {}, false}};
  metadata.tupleSets.emplace_back();
  auto &water = metadata.tupleSets.back();
  water.name = "settles";
  water.arity = 3;
  water.reversible = false;
  water.fields = {{"doh", {}, false}, {"dhh", {}, false}};
  auto parameters = buildParameterTypes(context, metadata);
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  SmallVector<Type> inputs{vec, mdir::md::CellType::get(&context)};
  llvm::append_range(inputs, parameters);
  auto energy = buildScalarPotential(builder, "energy", inputs,
      [](OpBuilder &at, ValueRange) -> Value {
        return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.125));
      });
  require(bool(energy), "native constrained-entry potential did not construct");
  DescentSiteIR placement{0, 1, 2, false, false, "spread_sites_linear"};
  DescentConstraintGroupIR group{3, 3, {4, 5}, {{0, 1, 0}, {0, 2, 0}, {1, 2, 1}},
      {false, false, 1.0e-8, 1.0e-12}, true, true};
  RuntimeDeclarationConfig runtime{};
  runtime.state64 = runtime.force64 = true;
  runtime.constraintStops = runtime.solvent = runtime.checkpoints = true;
  runtime.writesForces = metadata.writesForces;
  emitRuntimeDeclarations(builder, runtime, false);
  emitRuntimeDeclarations(builder, runtime, true);
  auto positions = MemRefType::get({ShapedType::kDynamic, 3}, builder.getF64Type());
  auto ids = MemRefType::get({ShapedType::kDynamic}, builder.getI32Type());
  auto boundary = func::FuncOp::create(builder, loc, "mdrtPartBoundary",
      builder.getFunctionType({positions, positions, positions, ids, builder.getF64Type(),
          MemRefType::get({9}, builder.getI64Type())}, {}));
  boundary.setPrivate();
  boundary->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  for (unsigned motion : {0u, 1u, 2u}) {
    for (bool leapfrog : {false, true}) {
      if (motion == 2 && leapfrog) continue;
      std::string suffix = std::to_string(motion) + (leapfrog ? "_lf" : "_vv");
      StepNoiseIR noise{motion != 0, motion == 2, {123, motion == 2 ? 1.0 : 0.8, 0.1}, 2.5};
      bool measured = measuresHalfSteps(true, noise.brownian);
      auto step = buildNativeStep(builder, {"step_" + suffix}, parameters, leapfrog, false,
          {placement}, false, false, {}, noise, {group});
      auto reporting = buildNativeStep(builder, {"energy_" + suffix, true, true, false, measured, measured},
          parameters, leapfrog, false, {placement}, false, false, {}, noise, {group});
      require(bool(step) && bool(reporting), "native constrained-entry programs did not construct");
      metadata.entry = "native_constraints_" + suffix;
      SimpleStepEntryIRConfig config{2, 3};
      config.checkpointIntervals = 2;
      config.brownian = noise.brownian;
      config.leapfrog = leapfrog;
      config.sites = {placement};
      config.constraints = {{3, 3, {4, 5}, true}};
      auto entry = buildSimpleStepEntry(builder, metadata, *step, *reporting, config);
      if (!entry) {
        llvm::errs() << "native constrained entry " << suffix << ": "
                     << llvm::toString(entry.takeError()) << "\n";
        std::exit(1);
      }
      auto segment = *entry->getOps<scf::ForOp>().begin();
      unsigned calls = 0, solventRows = 0;
      entry->walk([&](mdir::dyn::StepOp call) {
        ++calls;
        require(segment->isProperAncestor(call.getOperands().back().getDefiningOp()),
            "native constrained step reads stale identities");
        require(bool(call.getOperand(call.getNumOperands() - 2).getDefiningOp<arith::AddIOp>()),
            "native constrained step lost its counter without an explicit memory flag");
      });
      entry->walk([&](func::CallOp call) {
        if (call.getCallee() == "mdrtWriteSolvent") ++solventRows;
      });
      require(calls == 2 && solventRows == unsigned(measured),
          "native constrained entry lost a step or changed solvent reporting");
      for (auto row : entry->getOps<func::CallOp>()) {
        if (row.getCallee() != "mdrtWriteEnergies") continue;
        auto correction = row.getOperand(3).getDefiningOp<arith::ConstantOp>();
        require(correction && cast<FloatAttr>(correction.getValue()).getValueAsDouble() == 0.0,
            "native constrained entry estimated an unmeasured initial kinetic correction");
      }
      mdir::driver::Control control;
      control.integrator = noise.brownian ? mdir::driver::Integrator::Brownian
          : leapfrog ? mdir::driver::Integrator::Leapfrog : mdir::driver::Integrator::VelocityVerlet;
      control.thermostat = noise.active && !noise.brownian;
      control.thermostatMethod = mdir::driver::ThermostatMethod::Langevin;
      auto production = mdir::compiler::getSegmentsControl(control);
      auto partsConfig = config;
      partsConfig.segments = production.segments;
      partsConfig.plainSteps = production.numSteps;
      partsConfig.energyIntervals.reset();
      partsConfig.checkpointIntervals.reset();
      require(production.energyPeriod == 0 && production.framePeriod == 0 &&
              production.checkpointPeriod == 0 && production.getCouplingPeriod() == 0,
          "native step parts fixture differs from production control");
      metadata.entry = "native_constraint_parts_" + suffix;
      auto segmented = buildSimpleStepEntry(builder, metadata, *step, *reporting, partsConfig);
      if (!segmented) {
        llvm::errs() << "native constrained parts " << suffix << ": "
                     << llvm::toString(segmented.takeError()) << "\n";
        std::exit(1);
      }
      auto loops = segmented->getOps<scf::ForOp>();
      auto it = loops.begin();
      auto initial = *it++;
      auto parts = *it;
      require(initial->hasAttr("mdrt.segment") && initial.getNumRegionIterArgs() == 2 &&
              parts.getNumRegionIterArgs() == 3,
          "native step parts lost the first-call or part state branch");
      bool actualBoundary = false;
      for (auto call : parts.getBody()->getOps<mdir::mdrt::HostCallOp>())
        if (call.getCalleeAttr().getValue() == "mdrtPartBoundary") {
          require(call.getInPlace() && llvm::equal(call.getOperands().take_front(3), parts.getRegionIterArgs()),
              "native step part boundary lost actual carried state or in-place resorting");
          actualBoundary = true;
        }
      require(actualBoundary, "native step parts have no physical boundary");
      if (leapfrog) {
        bool beginsOnly = false;
        initial.walk([&](arith::CmpIOp test) {
          if (test.getPredicate() == arith::CmpIPredicate::eq &&
              test.getLhs() == segmented->getArguments().back()) beginsOnly = true;
        });
        require(beginsOnly, "native step parts repeat the backward kick on reevaluation");
      }
      metadata.entry = "native_invalid_parts_" + suffix;
      auto staticRestart = partsConfig;
      staticRestart.restart = true;
      auto staticError = buildSimpleStepEntry(builder, metadata, *step, *reporting, staticRestart);
      require(!staticError, "native parts accepted static restart control");
      reject(staticError.takeError(), segmentsStaticRestartError);
      auto invalid = partsConfig;
      invalid.partCounts = 8;
      auto omitted = buildSimpleStepEntry(builder, metadata, *step, *reporting, invalid);
      require(!omitted, "native step parts accepted an invalid host count layout");
      reject(omitted.takeError(), "native step parts require nine host count slots");
    }
  }
  require(succeeded(verify(module)), "native constrained-entry module did not verify");
}

static void checkNativeSiteStepEntry() {
  TermIRPlan plan;
  auto &context = plan.getContext();
  auto module = plan.getModule();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::md::TupleSetOp::create(builder, loc, "sites_linear", "atoms", 4,
      mdir::md::Orientation::Ordered, false);
  mdir::driver::Program metadata{};
  metadata.entry = "native_site_steps";
  metadata.state = metadata.force = metadata.mass = metadata.parameter = mdir::driver::Element::F64;
  metadata.takesForces = metadata.writesForces = metadata.reorders = true;
  metadata.orderWidth = 0.5;
  metadata.tupleSets.emplace_back();
  auto &site = metadata.tupleSets.back();
  site.name = "sites_linear";
  site.arity = 4;
  site.reversible = false;
  site.fields = {{"a", {}, false}, {"b", {}, false}};
  auto parameters = buildParameterTypes(context, metadata);
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  SmallVector<Type> inputs{vec, mdir::md::CellType::get(&context)};
  llvm::append_range(inputs, parameters);
  auto energy = buildScalarPotential(builder, "energy", inputs,
      [](OpBuilder &at, ValueRange) -> Value {
        return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.125));
      });
  require(bool(energy), "native site-entry potential did not construct");
  DescentSiteIR placement{0, 1, 2, false, false, "spread_sites_linear"};
  auto step = buildNativeStep(builder, {"step"}, parameters, false, false, {placement});
  auto reporting = buildNativeStep(builder, {"step_energy", true, true}, parameters, false, false, {placement});
  require(bool(step) && bool(reporting), "native site-entry programs did not construct");
  SimpleStepEntryIRConfig config{2, 2};
  config.checkpointIntervals = 2;
  config.sites = {placement};
  RuntimeDeclarationConfig runtime{};
  runtime.state64 = runtime.force64 = true;
  runtime.checkpoints = config.checkpointIntervals.has_value();
  runtime.writesForces = metadata.writesForces;
  emitRuntimeDeclarations(builder, runtime, false);
  emitRuntimeDeclarations(builder, runtime, true);
  Value completedForce;
  auto entry = buildSimpleStepEntry(builder, metadata, *step, *reporting, config,
      [&](OpBuilder &, const PotentialStateIR &, EnergyResults result, Value) {
        completedForce = result.forces;
        return EntryCompletionIR{result, {}};
      });
  require(bool(entry), "native site entry did not construct");
  auto order = *entry->getOps<mdir::md_exec::SpatialOrderOp>().begin();
  require(bool(order.getOperand(0).getDefiningOp<mdir::md::MapParticlesOp>()),
      "native site entry reordered unplaced positions");
  auto raw = *entry->getOps<mdir::md::EvaluateOp>().begin();
  auto segment = *entry->getOps<scf::ForOp>().begin();
  require(completedForce != raw.getResult(1) && segment.getInitArgs()[2] == completedForce,
      "native site entry completed or carried forces before site spreading");
  require(succeeded(verify(module)), "native site-entry module did not verify");
}

static void checkNativeTimedStepEntry() {
  TermIRPlan plan;
  auto &context = plan.getContext();
  auto module = plan.getModule();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::driver::Program metadata{};
  metadata.entry = "native_timed_steps";
  metadata.state = metadata.force = metadata.mass = metadata.parameter = mdir::driver::Element::F64;
  metadata.takesForces = metadata.reorders = metadata.writesForces = true;
  metadata.orderWidth = 0.5;
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  auto energy = buildScalarPotential(builder, "energy",
      TypeRange{vec, mdir::md::CellType::get(&context), builder.getF64Type()},
      [](OpBuilder &, ValueRange args) -> Value { return args.back(); });
  require(bool(energy), "native timed-entry energy did not construct");
  auto alchemical = buildScalarPotential(builder, "alchemical",
      TypeRange{vec, mdir::md::CellType::get(&context), builder.getF64Type(), builder.getF64Type()},
      [](OpBuilder &, ValueRange args) -> Value { return args.back(); });
  auto tunable = buildScalarPotential(builder, "tunable",
      TypeRange{vec, mdir::md::CellType::get(&context), builder.getF64Type()},
      [](OpBuilder &, ValueRange args) -> Value { return args.back(); });
  require(bool(alchemical) && bool(tunable), "native entry output potentials did not construct");
  auto step = buildNativeStep(builder, {"step"}, {}, false, true);
  auto reporting = buildNativeStep(builder, {"step_energy", true, true}, {}, false, true);
  require(bool(step) && bool(reporting), "native timed-entry dynamics did not construct");
  SimpleStepEntryIRConfig config{2, 3, false};
  config.timed = true;
  config.checkpointIntervals = 2;
  config.frameIntervals = 2;
  config.output.freeEnergy = FreeEnergyOutputConfig{{{0.0}}, 0, 3};
  config.output.observables.push_back({"energy", {}, {{-1, 0}}});
  config.output.observableCount = 1;
  config.output.observableBaseArgument = 3;
  config.output.gradient.emplace();
  metadata.startValues = {{"gradient_request", 1.0}};
  metadata.tunableGradient = true;
  RuntimeDeclarationConfig runtime{};
  runtime.state64 = runtime.force64 = true;
  runtime.checkpoints = config.checkpointIntervals.has_value();
  runtime.writesForces = metadata.writesForces;
  runtime.tunableGradient = metadata.tunableGradient;
  emitRuntimeDeclarations(builder, runtime, false);
  emitRuntimeDeclarations(builder, runtime, true);
  unsigned outputPoints = 0;
  auto built = buildSimpleStepEntry(builder, metadata, *step, *reporting, config, {},
      [&](OpBuilder &, const PotentialStateIR &state, Value) {
        ++outputPoints;
        require(bool(state.time) == (outputPoints == 1),
            "native timed entry computed an unused coordinate-output time");
      });
  require(bool(built) && outputPoints == 2, "native timed entry did not construct its output points");
  auto memory = *built->getOps<memref::AllocaOp>().begin();
  unsigned calls = 0;
  built->walk([&](mdir::dyn::StepOp call) {
    ++calls;
    auto increment = call.getOperand(6).getDefiningOp<arith::AddIOp>();
    require(bool(increment), "native timed step lost its completed-step increment");
    auto load = increment.getLhs().getDefiningOp<memref::LoadOp>();
    require(load && load.getMemRef() == memory.getResult(),
        "native timed step reads a separate or stale counter");
    auto store = dyn_cast<memref::StoreOp>(call->getPrevNode());
    require(store && store.getValueToStore() == increment.getResult() &&
            store.getMemRef() == memory.getResult(),
        "native timed step moved its counter store");
  });
  require(calls == 2, "native timed entry lost its dynamics calls");
  unsigned rows = 0, gradients = 0;
  built->walk([&](func::CallOp call) {
    if (call.getCallee() == "mdrtWriteTunableGradient") ++gradients;
    if (call.getCallee() != "mdrtWriteEnergies") return;
    ++rows;
    SmallVector<StringRef> writes;
    for (Operation *next = call->getNextNode(); next; next = next->getNextNode())
      if (auto output = dyn_cast<func::CallOp>(next)) writes.push_back(output.getCallee());
    require(llvm::equal(writes, ArrayRef<StringRef>{"mdrtWriteFreeEnergy", "mdrtWriteObservables"}),
        "native entry changed free-energy/observable row order");
  });
  require(rows == 2 && gradients == 1,
      "native entry lost output rows or repeated the initial-only gradient");
  config.triclinic = true;
  metadata.entry = "native_timed_triclinic_steps";
  auto missing = buildSimpleStepEntry(builder, metadata, *step, *reporting, config);
  require(!missing, "native triclinic entry accepted missing tilt values");
  reject(missing.takeError(), "native step entry is missing cell start value 'tilt_bx'");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
      "missing triclinic tilt values retained a partial entry");
  metadata.startValues = {{"tilt_cy", 0.3}, {"unused", 0.0}, {"tilt_bx", 0.1}, {"tilt_cx", 0.2},
      {"gradient_request", 1.0}};
  config.output.gradient->request = 4;
  auto triclinic = buildSimpleStepEntry(builder, metadata, *step, *reporting, config);
  require(bool(triclinic), "native triclinic timed entry did not construct");
  auto cell = *triclinic->getOps<mdir::md::TriclinicCellOp>().begin();
  auto values = triclinic->getArguments().take_back(metadata.startValues.size());
  require(llvm::equal(cell.getOperands().take_back(3), ValueRange{values[2], values[3], values[0]}),
      "native triclinic entry guessed tilt-value positions instead of their semantic slots");
  require(succeeded(verify(module)), "native timed/triclinic entry module did not verify");
}

static void checkNativeSimpleEntry() {
  TermIRPlan plan;
  auto &context = plan.getContext();
  auto module = plan.getModule();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::driver::Program metadata{};
  metadata.entry = "native_steps";
  metadata.state = metadata.force = metadata.mass = metadata.parameter = mdir::driver::Element::F64;
  metadata.takesForces = metadata.reorders = true;
  metadata.orderWidth = 0.5;
  metadata.fields.push_back({"q", {}, false});
  metadata.tupleSets.emplace_back();
  auto &links = metadata.tupleSets.back();
  links.name = "links";
  links.arity = 2;
  links.fields.push_back({"weight", {}, false});
  mdir::md::TupleSetOp::create(builder, loc, links.name, "atoms", links.arity,
      mdir::md::Orientation::Unordered, true);
  auto parameters = buildParameterTypes(context, metadata);
  auto vec = mdir::md::FieldType::get(&context,
      FlatSymbolRefAttr::get(&context, "atoms"), 3, builder.getF64Type());
  SmallVector<Type> inputs{vec, mdir::md::CellType::get(&context)};
  llvm::append_range(inputs, parameters);
  auto energy = buildScalarPotential(builder, "energy", inputs,
      [](OpBuilder &at, ValueRange) -> Value {
        return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.125));
      });
  require(bool(energy), "native step-entry energy did not construct");
  auto step = buildNativeStep(builder, {"step"}, parameters, false, false);
  require(bool(step), "native step-entry dynamics did not construct");
  const std::optional<int64_t> checkpointIntervals = 2;
  metadata.writesForces = true;
  RuntimeDeclarationConfig runtime{};
  runtime.state64 = runtime.force64 = true;
  runtime.checkpoints = checkpointIntervals.has_value();
  runtime.writesForces = metadata.writesForces;
  emitRuntimeDeclarations(builder, runtime, false);
  emitRuntimeDeclarations(builder, runtime, true);
  auto built = buildSimpleStepEntry(builder, metadata, *step, {}, {3});
  require(bool(built), "native plain-step entry did not construct");
  auto entry = *built;
  auto loop = *entry.getOps<scf::ForOp>().begin();
  require(loop->hasAttr("mdrt.fresh"), "native plain-step loop lost its fresh boundary");
  auto call = *loop.getBody()->getOps<mdir::dyn::StepOp>().begin();
  require(llvm::equal(call.getOperands().take_front(3), loop.getRegionIterArgs()),
      "native plain-step call reads stale outer state");
  auto energyStep = buildNativeStep(builder, {"step_energy", true, true}, parameters, false, false);
  require(bool(energyStep), "native energy-step program did not construct");
  metadata.entry = "native_energy_steps";
  auto reporting = buildSimpleStepEntry(builder, metadata, *step, *energyStep, {2, 3, false});
  require(bool(reporting), "native energy-interval entry did not construct");
  auto outer = *reporting->getOps<scf::ForOp>().begin();
  auto inner = *outer.getBody()->getOps<scf::ForOp>().begin();
  auto closing = *outer.getBody()->getOps<mdir::dyn::StepOp>().begin();
  require(llvm::equal(inner.getInitArgs(), outer.getRegionIterArgs()),
      "native energy interval's plain loop reads stale outer state");
  require(llvm::equal(closing.getOperands().take_front(3), inner.getResults()),
      "native energy step lost the plain loop's final state");
  metadata.entry = "native_trajectory_steps";
  SimpleStepEntryIRConfig trajectory{2, 3, false};
  trajectory.frameIntervals = 2;
  auto framed = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory);
  require(bool(framed), "native trajectory-interval entry did not construct");
  auto frameLoop = *framed->getOps<scf::ForOp>().begin();
  auto energyLoop = *frameLoop.getBody()->getOps<scf::ForOp>().begin();
  auto stepLoop = *energyLoop.getBody()->getOps<scf::ForOp>().begin();
  require(llvm::equal(energyLoop.getInitArgs(), frameLoop.getRegionIterArgs()) &&
          llvm::equal(stepLoop.getInitArgs(), energyLoop.getRegionIterArgs()),
      "native trajectory intervals read stale enclosing state");
  require(frameLoop->hasAttr("mdrt.fresh") && !energyLoop->hasAttr("mdrt.fresh"),
      "native trajectory intervals moved the fresh boundary");
  metadata.entry = "native_frequent_frames";
  trajectory.frameIntervals.reset();
  trajectory.framesAtEnergies = true;
  auto frequent = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory);
  require(bool(frequent), "native frames-at-energy entry did not construct");
  metadata.entry = "native_checkpoint_steps";
  trajectory.checkpointIntervals = checkpointIntervals;
  trajectory.frameIntervals = 2;
  trajectory.framesAtEnergies = false;
  auto checkpointed = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory);
  require(bool(checkpointed), "native checkpoint entry did not construct");
  auto segment = *checkpointed->getOps<scf::ForOp>().begin();
  require(segment->hasAttr("mdrt.segment") && !segment->hasAttr("mdrt.fresh"),
      "native checkpoint segment lost its neighbor boundary");
  require(segment.getRegionIterArgs().size() == 5 + metadata.fields.size(),
      "native checkpoint segment lost reordered parameter state");
  unsigned calls = 0;
  segment.walk([&](mdir::dyn::StepOp call) {
    ++calls;
    require(segment->isProperAncestor(call.getOperand(3).getDefiningOp()),
        "native checkpoint step reads stale header masses");
  });
  require(calls == 2, "native checkpoint schedule lost a dynamics call");
  SmallVector<mdir::md_exec::RenumberOp> renumbers;
  checkpointed->walk([&](mdir::md_exec::RenumberOp renumber) { renumbers.push_back(renumber); });
  require(renumbers.size() == 3, "native checkpoint entry lost relation regeneration");
  require(renumbers[0].getOperand(0) == renumbers[1].getOperand(0) &&
          renumbers[0].getOperand(0) == renumbers[2].getOperand(0) &&
          renumbers[2].getOperand(1) == segment.getResults().back(),
      "native checkpoint relation regeneration lost original imports or final identities");
  metadata.entry = "native_topology_steps";
  trajectory.topology = true;
  trajectory.terms = {"energy"};
  auto topology = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory);
  require(bool(topology), "native topology entry did not construct");
  SmallVector<StringRef> initialWrites;
  for (auto call : topology->getOps<func::CallOp>()) initialWrites.push_back(call.getCallee());
  require(llvm::equal(initialWrites, ArrayRef<StringRef>{"mdrtWriteTerms", "mdrtWriteVirial", "mdrtWriteEnergies"}),
      "native topology entry changed initial output order");
  metadata.entry = "native_topology_pull_steps";
  trajectory.needsCoordinateOutput = true;
  auto omitted = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory);
  require(!omitted, "native topology entry silently dropped its coordinate-output role");
  reject(omitted.takeError(), "native step entry requires the coordinate-output role");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
      "missing coordinate-output role retained a partial entry");
  SmallVector<Type> pullInputs(inputs);
  pullInputs.append(4, builder.getF64Type());
  auto pull = buildScalarPotential(builder, "pullterm0", pullInputs,
      [&](OpBuilder &, ValueRange args) -> Value { return args[inputs.size()]; });
  require(bool(pull), "native topology pull potential did not construct");
  unsigned outputPoints = 0;
  auto pulled = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory, {},
      [&](OpBuilder &at, const PotentialStateIR &state, Value completed) {
        ++outputPoints;
        auto row = dyn_cast<func::CallOp>(&at.getInsertionBlock()->back());
        require(row && row.getCallee() == "mdrtWriteEnergies" && row.getOperand(0) == completed,
            "native topology coordinate output moved before or away from its energy row");
        for (auto closing : at.getInsertionBlock()->getOps<mdir::dyn::StepOp>())
          require(closing.getResult(0) == state.positions,
              "native topology coordinate output reads stale step positions");
        PullOutputTermIR term{0, 2, 2, state.fields[1],
            {state.fields[2], state.fields[2], state.fields[2]}};
        buildPullOutput(at, state, completed, {term});
        auto write = dyn_cast<func::CallOp>(&at.getInsertionBlock()->back());
        require(write && write.getCallee() == "mdrtWritePull" && write.getOperand(0) == completed,
            "native topology pull term lost its completed step");
      });
  require(bool(pulled) && outputPoints == 2,
      "native topology entry lost initial or energy-boundary pull output");
  metadata.entry = "native_post_steps";
  trajectory.needsCoordinateOutput = false;
  trajectory.needsPostEvaluation = true;
  trajectory.storedTerms = 1;
  auto omittedPost = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory);
  require(!omittedPost, "native topology entry silently dropped its post-evaluation role");
  reject(omittedPost.takeError(), "native step entry requires the post-evaluation role");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
      "missing post-evaluation role retained a partial entry");
  Value storedEnergy;
  auto complete = [&](OpBuilder &at, const PotentialStateIR &, EnergyResults energy, Value scale) {
    require(!scale, "fixed-cell post evaluation acquired a reference scale");
    storedEnergy = arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.5));
    energy.energy = arith::AddFOp::create(at, at.getUnknownLoc(), energy.energy, storedEnergy);
    return EntryCompletionIR{energy, {storedEnergy}};
  };
  trajectory.storedTerms = 2;
  auto wrongCount = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory, complete);
  require(!wrongCount, "native topology entry silently changed its stored-term count");
  reject(wrongCount.takeError(),
      "native step entry post-evaluation term count does not match its configuration");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
      "post-evaluation count mismatch retained a partial entry");
  trajectory.storedTerms = 1;
  auto corrected = buildSimpleStepEntry(builder, metadata, *step, *energyStep, trajectory, complete);
  require(bool(corrected), "native topology post-evaluation entry did not construct");
  SmallVector<memref::StoreOp> termStores(corrected->getOps<memref::StoreOp>());
  require(termStores.size() == 2 && termStores.back().getValueToStore() == storedEnergy,
      "native topology entry lost the post-evaluation energy after its ordinary term");
  for (unsigned motion : {0u, 1u, 2u}) {
    for (bool leapfrog : {false, true}) {
      if (motion == 2 && leapfrog) continue;
      std::string suffix = std::to_string(motion) + (leapfrog ? "_lf" : "_vv");
      StepNoiseIR noise{motion != 0, motion == 2, {123, motion == 2 ? 1.0 : 0.8, 0.1}, 2.5};
      auto plain = buildNativeStep(builder, {"motion_" + suffix}, parameters,
          leapfrog, false, {}, false, false, {}, noise);
      auto energy = buildNativeStep(builder, {"motion_energy_" + suffix, true, true}, parameters,
          leapfrog, false, {}, false, false, {}, noise);
      require(bool(plain) && bool(energy), "native integrator-entry programs did not construct");
      metadata.entry = "native_motion_" + suffix;
      SimpleStepEntryIRConfig motionConfig{2, 3};
      motionConfig.checkpointIntervals = 2;
      motionConfig.stepMemory = noise.active;
      motionConfig.brownian = noise.brownian;
      motionConfig.leapfrog = leapfrog;
      if (motion == 0 && !leapfrog) {
        motionConfig.brownian = true;
        auto mismatch = buildSimpleStepEntry(builder, metadata, *plain, *energy, motionConfig);
        require(!mismatch, "native integrator entry accepted an incompatible energy-step result layout");
        reject(mismatch.takeError(),
            "native step entry dynamics results do not match its integrator (brownian=1, leapfrog=0, constrained=0)");
        require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
            "integrator result mismatch retained a partial entry");
        motionConfig.brownian = false;
      }
      auto integrated = buildSimpleStepEntry(builder, metadata, *plain, *energy, motionConfig);
      require(bool(integrated), "native fixed-cell integrator entry did not construct");
      auto segment = *integrated->getOps<scf::ForOp>().begin();
      require(bool(segment.getInitArgs()[1].getDefiningOp<mdir::dyn::KickOp>()) == leapfrog,
          "native integrator entry lost or introduced the initial half-step kick");
      unsigned calls = 0;
      integrated->walk([&](mdir::dyn::StepOp call) {
        ++calls;
        if (noise.active) {
          auto ids = call.getOperands().back();
          require(segment->isProperAncestor(ids.getDefiningOp()),
              "native noisy step reads identities from before segment reordering");
          require(bool(call.getOperand(call.getNumOperands() - 2).getDefiningOp<arith::AddIOp>()),
              "native noisy step lost its completed-step counter");
        }
      });
      require(calls == 2, "native integrator entry lost a plain or closing step");
      metadata.entry = "native_restart_motion_" + suffix;
      motionConfig.restart = true;
      auto continued = buildSimpleStepEntry(builder, metadata, *plain, *energy, motionConfig);
      require(bool(continued), "native restart entry did not construct");
      require(continued->getOps<mdir::md::EvaluateOp>().empty(),
          "native restart entry reevaluated checkpoint forces");
      for (auto row : continued->getOps<func::CallOp>())
        require(row.getCallee() != "mdrtWriteEnergies", "native restart entry repeated the initial output row");
      auto resumedLoop = *continued->getOps<scf::ForOp>().begin();
      require(!resumedLoop.getInitArgs()[1].getDefiningOp<mdir::dyn::KickOp>(),
          "native restart entry repeated the backward half-step");
      auto forceOrder = resumedLoop.getInitArgs()[2].getDefiningOp<mdir::md_exec::PermuteOp>();
      require(bool(forceOrder), "native restart entry did not reorder its force state");
      auto imported = forceOrder.getOperand(0).getDefiningOp<mdir::mdrt::FromBufferOp>();
      require(imported && imported.getOperand() == continued->getArgument(2),
          "native restart entry reads forces other than its actual checkpoint buffer");
      metadata.entry = "native_recomputed_motion_" + suffix;
      metadata.takesForces = false;
      motionConfig.recomputeForces = true;
      auto recomputed = buildSimpleStepEntry(builder, metadata, *plain, *energy, motionConfig);
      require(bool(recomputed), "native force-recomputing restart entry did not construct");
      auto evaluation = *recomputed->getOps<mdir::md::EvaluateOp>().begin();
      auto recomputedLoop = *recomputed->getOps<scf::ForOp>().begin();
      require(recomputedLoop.getInitArgs()[2] == evaluation.getResult(1) &&
              !recomputedLoop.getInitArgs()[1].getDefiningOp<mdir::dyn::KickOp>(),
          "native recomputing restart lost its new forces or repeated the half-step");
      for (auto row : recomputed->getOps<func::CallOp>())
        require(row.getCallee() != "mdrtWriteEnergies", "native recomputing restart repeated initial output");
      metadata.takesForces = true;
    }
  }
  metadata.entry = "native_restart_without_forces";
  metadata.takesForces = false;
  SimpleStepEntryIRConfig noForces{3};
  noForces.restart = true;
  auto noState = buildSimpleStepEntry(builder, metadata, *step, {}, noForces);
  require(!noState, "native restart entry accepted no force state and no recomputation");
  reject(noState.takeError(), "native step restart has no force state");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
      "invalid native restart retained a partial entry");
  metadata.takesForces = true;
  metadata.entry = "native_hosted_steps";
  SimpleStepEntryIRConfig hosted{2, 3};
  hosted.needsHostedEvaluation = true;
  auto omittedHost = buildSimpleStepEntry(builder, metadata, *step, *energyStep, hosted);
  require(!omittedHost, "native entry silently dropped its hosted-evaluation role");
  reject(omittedHost.takeError(), "native step entry requires the hosted-evaluation role");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
      "missing hosted-evaluation role retained a partial entry");
  Value hostedForce, hostedEnergy;
  auto completedHost = buildSimpleStepEntry(builder, metadata, *step, *energyStep, hosted,
      [&](OpBuilder &at, const PotentialStateIR &, EnergyResults physical, Value) {
        hostedForce = buildFieldDifference(at, physical.forces, physical.forces);
        Value shift = arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(0.25));
        hostedEnergy = arith::AddFOp::create(at, at.getUnknownLoc(), physical.energy, shift);
        return EntryCompletionIR{{hostedEnergy, hostedForce, physical.virial}, {}};
      });
  require(bool(completedHost), "native hosted-entry completion did not construct");
  auto hostedLoop = *completedHost->getOps<scf::ForOp>().begin();
  require(hostedLoop.getInitArgs()[2] == hostedForce,
      "native hosted entry carried forces from before hosted completion");
  unsigned hostedRows = 0;
  for (auto row : completedHost->getOps<func::CallOp>())
    if (row.getCallee() == "mdrtWriteEnergies") {
      ++hostedRows;
      require(row.getOperand(1) == hostedEnergy, "native hosted entry reported the uncompleted energy");
    }
  require(hostedRows == 1, "native hosted entry lost its initial completed row");
  metadata.fields.clear();
  metadata.tupleSets.clear();
  metadata.entry = "changed";
  require(module.lookupSymbol<func::FuncOp>("native_steps") == entry,
      "native plain-step entry retained mutable metadata");
  require(succeeded(verify(module)), "native plain-step module did not verify");
  auto missing = buildSimpleStepEntry(builder, metadata, {}, {}, {3});
  require(!missing, "native plain-step entry accepted a missing dynamics program");
  reject(missing.takeError(), "a generated step has no program definition");
  require(!module.lookupSymbol<func::FuncOp>("changed"), "failed native entry retained its function");
}

static llvm::Error verifyNativeFrameEntry(ModuleOp module) {
  if (failed(verify(module)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native frame entry did not verify");
  return llvm::Error::success();
}

static void checkNativeFrameEntry() {
  TermIRPlan plan;
  auto &context = plan.getContext();
  auto module = plan.getModule();
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());
  auto loc = builder.getUnknownLoc();
  mdir::md::ParticleSetOp::create(builder, loc, "atoms");
  mdir::driver::Program metadata{};
  metadata.entry = "native_frames";
  metadata.state = metadata.force = metadata.mass = metadata.parameter = mdir::driver::Element::F64;
  metadata.takesForces = true;
  metadata.reorders = false;
  metadata.fields.push_back({"q", {}, false});
  metadata.fields.push_back({"kind", {}, true});
  metadata.tables.push_back({"grid", 1, {0.0}, 1});
  metadata.tupleSets.emplace_back();
  auto &site = metadata.tupleSets.back();
  site.name = "sites_linear";
  site.arity = 4;
  site.reversible = false;
  site.fields.push_back({"a", {}, false});
  site.fields.push_back({"b", {}, false});
  mdir::md::TupleSetOp::create(builder, loc, site.name, "atoms", site.arity,
      mdir::md::Orientation::Ordered, false);
  metadata.tupleSets.emplace_back();
  auto &bond = metadata.tupleSets.back();
  bond.name = "bonds";
  bond.arity = 2;
  bond.fields.push_back({"d1", {}, false});
  mdir::md::TupleSetOp::create(builder, loc, bond.name, "atoms", bond.arity,
      mdir::md::Orientation::Unordered, true);
  auto atoms = FlatSymbolRefAttr::get(&context, "atoms");
  SmallVector<Type> inputs{mdir::md::FieldType::get(&context, atoms, 3, builder.getF64Type()),
      mdir::md::CellType::get(&context)};
  llvm::append_range(inputs, buildParameterTypes(context, metadata));
  auto energy = buildScalarPotential(builder, "energy", inputs,
      [](OpBuilder &at, ValueRange) -> Value {
        return arith::ConstantOp::create(at, at.getUnknownLoc(), at.getF64FloatAttr(2.5));
      });
  require(bool(energy), "native frame fixture energy did not construct");
  const bool storesBarostatState = true;
  RuntimeDeclarationConfig runtime{};
  runtime.barostatState = storesBarostatState;
  runtime.state64 = runtime.force64 = true;
  runtime.frames = runtime.segments = true;
  emitRuntimeDeclarations(builder, runtime, false);
  emitRuntimeDeclarations(builder, runtime, true);
  SimpleFrameEntryIRConfig config;
  config.branchesStart = config.givenForces = true;
  config.sites.push_back({3, 4, 5, false, false, "spread_sites_linear"});
  config.constraints.push_back({6, 2, {7}, false});
  config.stepMemory = config.leapfrog = config.coupled = true;
  config.thermostat = FrameThermostatIR{7, 9.0, 4.5, 0.9};
  config.barostat = config.changesCell = config.removesMotion = true;
  config.scalesEveryStep = storesBarostatState;
  config.barostatParameters = BarostatParameters{{0.1, 0.1, 0.1}, 1.0, 0.1, 0.2, 2.5, 0.1, 0.0};
  unsigned outputCalls = 0;
  Value outputPositions, outputCell, outputStep;
  auto built = buildSimpleFrameEntry(builder, metadata, config, {},
      [&](OpBuilder &, const PotentialStateIR &state, Value step) {
        ++outputCalls;
        outputPositions = state.positions;
        outputCell = state.cell;
        outputStep = step;
      });
  require(bool(built), "native frame entry did not construct");
  auto entry = *built;
  auto positions = entry.getArgument(0);
  require(outputCalls == 1 && outputPositions && outputCell &&
          outputStep == entry.getArgument(entry.getNumArguments() - 2),
          "native frame coordinate output lost its physical state or step");
  require(positions.use_empty(), "native frame entry read outer positions");
  require(bool(entry.getArgAttr(3, "mdrt.read_only")), "native frame entry lost read-only masses");
  require(!checkFrameEvaluation(*entry.getOperation()), "native frame entry lost its actual evaluation");
  auto frame = *entry.getOps<scf::ForOp>().begin();
  Value rawEdge = entry.getArgument(entry.getNumArguments() - 6);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(frame);
    auto derived = arith::NegFOp::create(builder, loc, rawEdge);
    builder.setInsertionPoint(frame.getBody()->getTerminator());
    auto stale = arith::NegFOp::create(builder, loc, derived.getResult());
    reject(checkFrameInputs(*frame.getOperation(), ValueRange{rawEdge}),
        "native frame iteration read entry positions or cell values");
    stale.erase();
    derived.erase();
    builder.setInsertionPoint(frame);
    auto memory = memref::AllocaOp::create(builder, loc, MemRefType::get({3}, builder.getF64Type()));
    auto alias = memref::CastOp::create(builder, loc,
        MemRefType::get({ShapedType::kDynamic}, builder.getF64Type()), memory.getResult());
    builder.setInsertionPoint(frame.getBody()->getTerminator());
    auto loaded = memref::LoadOp::create(builder, loc, alias.getResult(), frame.getLowerBound());
    reject(checkFrameInputs(*frame.getOperation(), ValueRange{memory.getResult()}),
        "native frame iteration read entry positions or cell values");
    loaded.erase();
    alias.erase();
    memory.erase();
    require(!checkFrameInputs(*frame.getOperation(), ValueRange{rawEdge}),
        "native frame input guard rejected independent header initialization");
    for (bool capturesCell : {false, true}) {
      builder.setInsertionPoint(frame);
      auto producer = scf::ExecuteRegionOp::create(builder, loc, TypeRange{builder.getF64Type()});
      {
        OpBuilder::InsertionGuard regionGuard(builder);
        builder.createBlock(&producer.getRegion());
        Value value = capturesCell ? rawEdge :
            Value(arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0)));
        scf::YieldOp::create(builder, loc, ValueRange{value});
      }
      builder.setInsertionPoint(frame.getBody()->getTerminator());
      auto captured = arith::NegFOp::create(builder, loc, producer.getResult(0));
      if (capturesCell)
        reject(checkFrameInputs(*frame.getOperation(), ValueRange{rawEdge}),
            "native frame iteration read entry positions or cell values");
      else
        require(!checkFrameInputs(*frame.getOperation(), ValueRange{rawEdge}),
            "native frame input guard rejected an independent producer region");
      require(succeeded(verify(entry)), "external producer capture fixture did not verify");
      captured.erase();
      producer.erase();
    }
  }
  auto empty = func::FuncOp::create(builder, loc, "empty_frames", builder.getFunctionType({}, {}));
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(empty.addEntryBlock());
    func::ReturnOp::create(builder, loc);
  }
  reject(checkFrameEvaluation(*empty.getOperation()),
         "the program of frames: the start of this program evaluates "
         "nothing, so its loop over frames would give nothing (Builder::emitEntry)");
  auto batchMetadata = metadata;
  batchMetadata.entry = "native_device_batch_frames";
  auto batchConfig = config;
  batchConfig.deviceBatch = batchConfig.batchOnDevice = true;
  auto batchEntry = buildSimpleFrameEntry(builder, batchMetadata, batchConfig);
  require(bool(batchEntry), "native device batch entry did not construct");
  require((*batchEntry)->hasAttr("mdir.frame_batch"),
          "native batch entry lost its lowering marker");
  unsigned batchBoundaries = 0, legacyBoundaries = 0, positionViews = 0;
  batchEntry->walk([&](func::CallOp call) {
    batchBoundaries += call.getCallee() == "mdrtFrameBatchBoundary";
    legacyBoundaries += call.getCallee() == "mdrtFrameBoundary";
  });
  batchEntry->walk([&](memref::ViewOp view) {
    auto type = cast<MemRefType>(view.getType());
    auto boundary = view.getSource().getDefiningOp<func::CallOp>();
    positionViews += type.getRank() == 2 && type.getMemorySpaceAsInt() == 1 &&
        boundary && boundary.getCallee() == "mdrtFrameBatchBoundary" &&
        view.getSource() == boundary.getResult(0);
  });
  require(batchBoundaries == 1 && legacyBoundaries == 0 && positionViews == 1,
          "native batch entry did not replace the per-frame boundary with row views");
  require(succeeded(verify(*batchEntry)), "native device batch entry is invalid");
  metadata.entry = "native_triclinic_frames";
  metadata.startValues = {{"tilt_bx", 0.1}, {"tilt_cx", 0.2}, {"tilt_cy", 0.3},
      {"rest_edge0", 3.0}, {"rest_edge1", 3.0}, {"rest_edge2", 3.0}};
  auto wideConfig = config;
  wideConfig.cellEdges = 6;
  wideConfig.referenceEdges = std::array<unsigned, 3>{3, 4, 5};
  wideConfig.topology = true;
  wideConfig.terms = {"energy"};
  wideConfig.storedTerms = 1;
  wideConfig.output.observables.push_back({"energy", {}, {{-1, 0}}});
  wideConfig.output.observableCount = 1;
  wideConfig.output.observableBaseArgument = inputs.size();
  auto wide = buildSimpleFrameEntry(builder, metadata, wideConfig,
      [](OpBuilder &at, const PotentialStateIR &, EnergyResults result, Value) -> EntryCompletionIR {
        if (!result.energy) return {result, {}};
        auto loc = at.getUnknownLoc();
        Value stored = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0.5));
        result.energy = arith::AddFOp::create(at, loc, result.energy, stored);
        return {result, {stored}};
      });
  require(bool(wide), "native triclinic frame entry did not construct");
  auto wideFrame = *wide->getOps<scf::ForOp>().begin();
  SmallVector<Value> tilts(wide->getArguments().drop_back(3).take_back(3));
  for (Value tilt : tilts)
    require(!tilt.use_empty(), "native triclinic header lost its initial cell state");
  require(!checkFrameInputs(*wideFrame.getOperation(), tilts),
      "native triclinic frame entry read an outer tilt");
  metadata.entry = "missing_stored_frame_term";
  auto missing = buildSimpleFrameEntry(builder, metadata, wideConfig);
  require(!missing, "native frame entry accepted a missing stored contribution");
  reject(missing.takeError(), "native frame evaluation returned an unexpected stored-term count");
  require(!module.lookupSymbol<func::FuncOp>(metadata.entry),
          "failed native frame entry retained an incomplete function");
  metadata.fields.clear();
  metadata.tables.clear();
  metadata.entry = "changed";
  config.branchesStart = config.givenForces = false;
  require(module.lookupSymbol<func::FuncOp>("native_frames") == entry,
          "native frame entry was reparsed or lost transient metadata ownership");
  require(entry.getArgument(0) == positions, "native frame entry replaced its actual argument");
  auto declaration = module.lookupSymbol<func::FuncOp>("mdrtSetBarostatState");
  require(bool(declaration), "barostat fixture did not declare its state callback");
  declaration->remove();
  std::string diagnostic;
  {
    ScopedDiagnosticHandler capture(&context, [&](Diagnostic &message) {
      llvm::raw_string_ostream stream(diagnostic);
      message.print(stream);
      return success();
    });
    reject(verifyNativeFrameEntry(module), "native frame entry did not verify");
  }
  module.getBody()->push_back(declaration.getOperation());
  require(StringRef(diagnostic).contains("mdrtSetBarostatState") &&
          StringRef(diagnostic).contains("does not reference a valid function"),
          "missing barostat declaration lost its verifier diagnostic");
  require(!verifyNativeFrameEntry(module), "native frame entry did not verify");
  builder.setInsertionPoint(entry.getBody().front().getTerminator());
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value initial = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
  auto before = std::distance(entry.getBody().front().begin(), entry.getBody().front().end());
  auto bad = buildEntryLoop(builder, "frame construction", zero, one, one, ValueRange{initial},
      [](OpBuilder &, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
        return SmallVector<Value>{};
      });
  require(!bad, "native loop accepted a return-count mismatch");
  reject(bad.takeError(), "native frame construction loop: yield has 0 values, expected 1");
  require(std::distance(entry.getBody().front().begin(), entry.getBody().front().end()) == before,
          "failed native loop retained an incomplete region");
  require(succeeded(verify(module)), "failed native loop damaged its entry");
  auto nested = buildEntryLoop(builder, "outer frame construction", zero, one, one, {},
      [&](OpBuilder &at, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
        auto inner = buildEntryLoop(at, "inner frame construction", zero, one, one,
            ValueRange{initial},
            [](OpBuilder &, Value, ValueRange) -> llvm::Expected<SmallVector<Value>> {
              return SmallVector<Value>{};
            });
        if (!inner) return inner.takeError();
        return SmallVector<Value>{};
      });
  require(!nested, "nested native loop swallowed an inner construction error");
  reject(nested.takeError(), "native inner frame construction loop: yield has 0 values, expected 1");
  require(std::distance(entry.getBody().front().begin(), entry.getBody().front().end()) == before,
          "failed nested native loop retained an incomplete region");
  require(succeeded(verify(module)), "failed nested native loop damaged its entry");

}

static void checkNativeFrameBatchRows() {
  for (bool device : {false, true})
    for (bool wide : {false, true})
      for (unsigned edges : {3u, 6u}) {
        TermIRPlan plan;
        OpBuilder builder(&plan.getContext());
        auto module = plan.getModule();
        builder.setInsertionPointToEnd(module.getBody());
        auto loc = builder.getUnknownLoc();
        Attribute space = device ? Attribute(builder.getI64IntegerAttr(1)) : Attribute{};
        auto bytes = MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
            MemRefLayoutAttrInterface(), space);
        auto cells = MemRefType::get({ShapedType::kDynamic, 6}, builder.getF64Type());
        Type state = wide ? Type(builder.getF64Type()) : Type(builder.getF32Type());
        auto function = func::FuncOp::create(builder, loc, "batch",
            builder.getFunctionType({bytes, cells, builder.getIndexType()}, {}));
        auto args = function.addEntryBlock()->getArguments();
        builder.setInsertionPointToEnd(&function.getBody().front());
        unsigned calls = 0;
        FrameBatchRowIR captured;
        auto loop = buildFrameBatchRows(builder, args[0], args[1], args[2], state, edges,
            [&](OpBuilder &, const FrameBatchRowIR &row) -> llvm::Error {
          ++calls;
          captured = row;
          return llvm::Error::success();
        });
        require(bool(loop), "native batch row construction failed");
        require(calls == 1 && captured.index == loop->getInductionVar() &&
                    captured.edges.size() == edges,
                "native batch lost its row index or cell shape");
        auto view = captured.positions.getDefiningOp<memref::ViewOp>();
        auto positionType = cast<MemRefType>(captured.positions.getType());
        require(view && view.getSource() == args[0] &&
                    positionType.getMemorySpace() == space &&
                    positionType.getElementType() == state &&
                    positionType.getLayout().isIdentity(),
                "native batch did not create an identity view of input bytes");
        auto offset = view.getByteShift().getDefiningOp<arith::MulIOp>();
        require(offset && offset.getLhs() == captured.index,
                "native batch view does not advance with the row");
        auto stride = offset.getRhs().getDefiningOp<arith::MulIOp>();
        require(stride && stride.getLhs() == args[2] &&
                    stride.getRhs().getDefiningOp<arith::ConstantIndexOp>().value() ==
                        (wide ? 24 : 12),
                "native batch byte stride lost the particle count or width");
        for (auto [edge, value] : llvm::enumerate(captured.edges)) {
          auto load = value.getDefiningOp<memref::LoadOp>();
          require(load && load.getMemRef() == args[1] &&
                      load.getIndices()[0] == captured.index &&
                      load.getIndices()[1].getDefiningOp<arith::ConstantIndexOp>().value() ==
                          int64_t(edge),
                  "native batch loaded the wrong cell row or edge");
        }
        auto count = loop->getUpperBound().getDefiningOp<memref::DimOp>();
        require(count && count.getSource() == args[1],
                "native batch trip count does not follow its cells");
        func::ReturnOp::create(builder, loc);
        require(succeeded(verify(module)), "native batch row IR is invalid");
      }
}

static void emitFrameBatchTunableFunctions() {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
      memref::MemRefDialect, scf::SCFDialect, gpu::GPUDialect>();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  auto vector = [&](Type element) {
    return MemRefType::get({ShapedType::kDynamic}, element,
        MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(1));
  };
  builder.setInsertionPointToEnd(module.getBody());
  auto prepare = func::FuncOp::create(builder, loc, "prepare_map",
      builder.getFunctionType({}, {vector(builder.getI64Type())}));
  builder.setInsertionPointToEnd(prepare.addEntryBlock());
  SmallVector<int64_t> mapping;
  for (int64_t i = 0; i != 137; ++i)
    mapping.push_back(i % 5 == 0 ? -1 : (i * 7) % 9);
  auto deviceMap = buildFrameBatchParameterMap(builder, mapping, 9);
  require(bool(deviceMap), deviceMap ? "" : llvm::toString(deviceMap.takeError()));
  func::ReturnOp::create(builder, loc, *deviceMap);
  for (bool wide : {false, true}) {
    builder.setInsertionPointToEnd(module.getBody());
    auto entry = func::FuncOp::create(builder, loc,
        wide ? "mapped_f64" : "mapped_f32",
        builder.getFunctionType({vector(builder.getF64Type()),
            vector(builder.getI64Type()),
            vector(wide ? Type(builder.getF64Type()) : Type(builder.getF32Type()))}, {}));
    entry->setAttr("mdir.frame_batch", builder.getUnitAttr());
    builder.setInsertionPointToEnd(entry.addEntryBlock());
    require(!buildFrameBatchMappedParameters(builder, entry.getArgument(0),
        entry.getArgument(1), entry.getArgument(2)), "mapped parameter construction failed");
    func::ReturnOp::create(builder, loc);
  }
  builder.setInsertionPointToEnd(module.getBody());
  auto entry = func::FuncOp::create(builder, loc, "contract",
      builder.getFunctionType({vector(builder.getF64Type()), vector(builder.getI64Type()),
          vector(builder.getI64Type()), vector(builder.getF64Type()),
          vector(builder.getF64Type())}, {}));
  entry->setAttr("mdir.frame_batch", builder.getUnitAttr());
  builder.setInsertionPointToEnd(entry.addEntryBlock());
  require(!buildFrameBatchOrderedContraction(builder, entry.getArgument(0),
      entry.getArgument(1), entry.getArgument(2), entry.getArgument(3),
      entry.getArgument(4)), "ordered contraction construction failed");
  func::ReturnOp::create(builder, loc);
  require(succeeded(verify(module)), "device tunable helper module is invalid");
  for (Operation &operation : *module.getBody()) {
    operation.print(llvm::outs());
    llvm::outs() << "\n";
  }
}

static void emitFrameBatchLJFunctions() {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
      memref::MemRefDialect, scf::SCFDialect, gpu::GPUDialect>();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  auto vector = [&](Type element) {
    return MemRefType::get({ShapedType::kDynamic}, element,
        MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(1));
  };
  for (unsigned index = 0; index != 3; ++index) {
    builder.setInsertionPointToEnd(module.getBody());
    bool counts = index == 2;
    auto entry = func::FuncOp::create(builder, loc,
        counts ? "prepare_counts" : index == 0 ? "prepare_sigma_map" : "prepare_epsilon_map",
        builder.getFunctionType({}, {vector(counts ? Type(builder.getF64Type()) : Type(builder.getI64Type()))}));
    builder.setInsertionPointToEnd(entry.addEntryBlock());
    auto values = counts ? buildFrameBatchInitialValues(builder, {2.0})
        : buildFrameBatchParameterMap(builder, {static_cast<int64_t>(index)}, 2);
    require(bool(values), values ? "" : llvm::toString(values.takeError()));
    func::ReturnOp::create(builder, loc, *values);
  }
  builder.setInsertionPointToEnd(module.getBody());
  auto mapped = func::FuncOp::create(builder, loc, "mapped_f64",
      builder.getFunctionType({vector(builder.getF64Type()), vector(builder.getI64Type()),
          vector(builder.getF64Type())}, {}));
  builder.setInsertionPointToEnd(mapped.addEntryBlock());
  require(!buildFrameBatchMappedParameters(builder, mapped.getArgument(0),
      mapped.getArgument(1), mapped.getArgument(2)), "LJ parameter map construction failed");
  func::ReturnOp::create(builder, loc);
  builder.setInsertionPointToEnd(module.getBody());
  auto entry = func::FuncOp::create(builder, loc, "lj_constants",
      builder.getFunctionType({vector(builder.getF64Type()), vector(builder.getF64Type()),
          vector(builder.getF64Type()), vector(builder.getI64Type()),
          vector(builder.getF64Type())}, {}));
  builder.setInsertionPointToEnd(entry.addEntryBlock());
  require(!buildFrameBatchLJConstants(builder, entry.getArgument(0), entry.getArgument(1),
      entry.getArgument(2), entry.getArgument(3), entry.getArgument(4),
      {1, 2.0, 2.0, 1.0, 64.0, false}), "LJ device constant construction failed");
  func::ReturnOp::create(builder, loc);
  require(succeeded(verify(module)), "native LJ correction helper module is invalid");
  for (Operation &operation : *module.getBody()) {
    operation.print(llvm::outs());
    llvm::outs() << "\n";
  }
}

int main(int argc, char **argv) {
  if (argc == 2 && llvm::StringRef(argv[1]) == "--emit-frame-batch-lj-functions") {
    emitFrameBatchLJFunctions();
    return 0;
  }
  if (argc == 2 && llvm::StringRef(argv[1]) == "--emit-frame-batch-tunable-functions") {
    emitFrameBatchTunableFunctions();
    return 0;
  }
  checkNativeFrameBatchRows();
  for (bool leapfrog : {false, true})
    for (bool constrained : {false, true})
      for (bool nose : {false, true})
        for (bool com : {false, true}) {
          checkNativeThermalStepEntry(leapfrog, constrained, nose, com);
          checkNativeThermalStepEntry(leapfrog, constrained, nose, com, true);
          checkNativeThermalStepEntry(leapfrog, constrained, nose, com, true, true);
          checkNativeThermalStepEntry(leapfrog, constrained, nose, com, true, false, true);
          checkNativeThermalStepEntry(leapfrog, constrained, nose, com, true, false, true, true);
          checkNativeThermalStepEntry(leapfrog, constrained, nose, com, true, false, true, false, true);
          checkNativeThermalStepEntry(leapfrog, constrained, nose, com, true, false, true, true, true);
        }
  checkStaticRestartControl();
  checkNativeCenterStepEntry();
  checkNativeMinimizationEntry();
  checkNativeConstraintStepEntry();
  checkNativeSiteStepEntry();
  checkNativeTimedStepEntry();
  checkNativeSimpleEntry();
  checkNativeFrameEntry();
  checkDiagonalExports();
  checkFrameEvaluations();
  checkNativeDescentBody();
  checkNativeVerletBodies();
  checkNativeStepReturnGuard();
  checkNativeBarePotentialBody();
  require(argc == 2, "expected the frozen staged-model fixture");
  auto fixture = llvm::MemoryBuffer::getFile(argv[1]);
  require(bool(fixture), "cannot read the frozen staged-model fixture");
  checkModelInput((*fixture)->getBuffer());
  checkModelInput(
      "md.potential @unusual(%positions: !md.field<@atoms, 3 x f64>, "
      "%box: !md.cell, %numbers: !md.field<@atoms, i32>) -> f64 { "
      "%value = arith.constant {user.note = \"%value md.potential @literal\"} "
      "1.0 : f64 md.return %value : f64 }", true);
  {
    TermIRPlan plan;
    auto bad = StagedModelIR::parse(plan.getContext(), "module {}", "input", "bad");
    require(!bad, "empty model was accepted");
    reject(bad.takeError(), "the model in stages 'bad' must hold one 'md.potential'");
  }

  {
    TermIRPlan plan;
    auto wrongSignature = StagedModelIR::parse(
        plan.getContext(),
        "md.potential @bad(%x: f64, %cell: !md.cell, "
        "%z: !md.field<@atoms, i32>) -> f64 { md.return %x : f64 }",
        "input", "bad");
    require(!wrongSignature, "model accepted a nonfield position argument");
    reject(wrongSignature.takeError(),
           "the potential of the model in stages 'bad' must take the positions, "
           "the cell, and the atomic numbers");
    auto wrongReturn = StagedModelIR::parse(
        plan.getContext(),
        "md.potential @bad(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell, "
        "%z: !md.field<@atoms, i32>) -> i32 { "
        "%u = arith.constant 0 : i32 md.return %u : i32 }", "input", "bad");
    require(!wrongReturn, "model accepted a nonenergy result");
    reject(wrongReturn.takeError(),
           "the potential of the model in stages 'bad' must return one f64 energy");
  }
  for (llvm::StringRef source : {
           "\"md.potential\"() {sym_name = \"bad\"} : () -> ()",
           "\"md.potential\"() ({^bb0(%x: f64, %cell: !md.cell, "
           "%z: !md.field<@atoms, i32>): md.return %x : f64}) "
           "{sym_name = \"bad\", function_type = "
           "(!md.field<@atoms, 3 x f64>, !md.cell, !md.field<@atoms, i32>) "
           "-> f64} : () -> ()",
           "\"builtin.module\"() : () -> ()"}) {
    TermIRPlan plan;
    auto malformed = StagedModelIR::parse(plan.getContext(), source, "input", "bad");
    require(!malformed, "malformed generic model input was accepted");
    llvm::consumeError(malformed.takeError());
  }
  {
    TermIRPlan plan;
    auto native = plan.getModule();
    require(native.getContext() == &plan.getContext(),
            "native program lost its owned context");
    OpBuilder builder(&plan.getContext());
    builder.setInsertionPointToEnd(native.getBody());
    auto atoms = mdir::md::ParticleSetOp::create(
        builder, builder.getUnknownLoc(), "atoms");
    Operation *original = atoms.getOperation();
    auto entry = func::FuncOp::create(builder, builder.getUnknownLoc(), "entry",
        builder.getFunctionType({}, {builder.getF64Type()}));
    builder.setInsertionPointToEnd(entry.addEntryBlock());
    Value energy = arith::ConstantOp::create(builder, builder.getUnknownLoc(),
        builder.getF64FloatAttr(1.0));
    auto result = func::ReturnOp::create(builder, builder.getUnknownLoc(), energy);
    mdir::driver::Program program;
    // Existing serialized content is output storage, never construction input.
    program.module = "this is not assembly";
    require(!serializeNativeProgram(program, plan),
            "native program did not serialize");
    require(plan.getModule() == native && &native.getBody()->front() == original &&
                result.getOperand(0) == energy,
            "native ownership or SSA identity changed at serialization");
    require(native.getBody()->getOperations().size() == 2 &&
                succeeded(verify(native)),
            "native program lost declaration order or validity");
    require(bool(parseSourceString<ModuleOp>(program.module, &plan.getContext())),
            "completed native program did not print valid IR");
  }
  llvm::outs() << "native term IR construction checks passed\n";
}
