// Internal term connections must disappear before ordinary verification.
#include "../../lib/Driver/TermIR.h"
#include "../../lib/Driver/PotentialIR.h"
#include "../../lib/Driver/DynamicsIR.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "../../lib/Driver/StagedModelIR.h"
#include "../../lib/Driver/BarostatIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/ProgramDialects.h"
#include "mdir/Driver/Builder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
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

static void reject(llvm::Error error, llvm::StringRef message) {
  require(bool(error), "invalid term connection was accepted");
  require(llvm::toString(std::move(error)) == message,
          "wrong term connection diagnostic");
}

static TermIRValues scalar(OpBuilder &builder, ValueRange) {
  return {arith::ConstantFloatOp::create(builder, builder.getUnknownLoc(),
                                         builder.getF64Type(),
                                         llvm::APFloat(1.0))};
}

static void registerRecipe(TermIRPlan &plan, TermIRRecipe recipe = scalar) {
  plan.emit(llvm::nulls(), "", {"%result"}, {"f64"}, {}, {}, std::move(recipe));
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
  for (bool exportScalars : {false, true}) {
    TermIRPlan plan;
    mdir::driver::Program program;
    llvm::raw_string_ostream stream(program.module);
    stream << "module { func.func private @report(f64, f64, f64) "
              "func.func @entry(%w: vector<9xf64>) -> vector<3xf64> { ";
    llvm::SmallVector<llvm::StringRef> names{"%diagonal"};
    llvm::SmallVector<llvm::StringRef> types{"vector<3xf64>"};
    if (exportScalars) {
      names.append({"%xx", "%yy", "%zz"});
      types.append({"f64", "f64", "f64"});
    }
    std::array<Value, 3> scalars;
    plan.emit(stream, "", names, types, {"%w"}, {"vector<9xf64>"},
              [exportScalars, &scalars](OpBuilder &builder, ValueRange values) {
                auto result = buildVirialDiagonal(builder, values[0]);
                scalars = {result.xx, result.yy, result.zz};
                if (exportScalars)
                  return TermIRValues{result.vector, result.xx, result.yy, result.zz};
                return TermIRValues{result.vector};
              });
    stream << "func.call @report(%xx, %yy, %zz) : (f64, f64, f64) -> () "
              "return %diagonal : vector<3xf64> } }";
    stream.flush();
    if (!exportScalars) {
      reject(finishTermIRProgram(program, plan),
             "the builder produced invalid assembly");
      continue;
    }
    require(!finishTermIRProgram(program, plan),
            "explicit diagonal exports failed construction");
    require(succeeded(verify(plan.getModule())), "diagonal consumer did not verify");
    unsigned extracts = 0, calls = 0;
    plan.getModule().walk([&](vector::ExtractOp) { ++extracts; });
    plan.getModule().walk([&](func::CallOp call) {
      ++calls;
      for (unsigned i = 0; i != 3; ++i)
        require(call.getOperand(i) == scalars[i],
                "virial consumer did not receive the exported scalar");
    });
    require(extracts == 3 && calls == 1,
            "diagonal exports duplicated extracts or lost the consumer");
  }
}

// Native recipes carry no printed evaluate token. Conversely, an attribute
// containing that token cannot stand in for an actual evaluation.
static void checkFrameEvaluations() {
  for (bool evaluates : {false, true}) {
    TermIRPlan plan;
    mdir::driver::Program program;
    llvm::raw_string_ostream stream(program.module);
    stream << "module { md.potential @energy(%x: !md.field<@atoms, 3 x f64>, "
              "%cell: !md.cell) -> f64 { %zero = arith.constant 0.0 : f64 "
              "md.return %zero : f64 } "
              "func.func @entry(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) { "
              "%c0 = arith.constant 0 : index %c1 = arith.constant 1 : index "
              "scf.for %i = %c0 to %c1 step %c1 { "
              "%note = arith.constant {note = \"md.evaluate \"} 0.0 : f64 ";
    if (evaluates)
      plan.emit(stream, "", {"%u"}, {"f64"}, {"%x", "%cell"},
                {"!md.field<@atoms, 3 x f64>", "!md.cell"},
                [](OpBuilder &builder, ValueRange values) -> TermIRValues {
                  return {mdir::md::EvaluateOp::create(
                      builder, builder.getUnknownLoc(), TypeRange{builder.getF64Type()},
                      FlatSymbolRefAttr::get(builder.getContext(), "energy"), values,
                      builder.getDenseI32ArrayAttr({int32_t(mdir::md::Request::Energy)}),
                      builder.getDenseI64ArrayAttr({-1})).getResult(0)};
                });
    plan.emit(stream, "", {}, {}, {}, {},
              [](OpBuilder &builder, ValueRange) -> llvm::Expected<TermIRValues> {
                if (llvm::Error error = checkFrameEvaluation(
                        builder.getInsertionBlock()->getParentOp()))
                  return std::move(error);
                return TermIRValues{};
              });
    stream << "} return } }";
    stream.flush();
    if (evaluates)
      require(!finishTermIRProgram(program, plan), "native frame evaluation was not recognized");
    else
      reject(finishTermIRProgram(program, plan),
             "the program of frames: the start of this program evaluates "
             "nothing, so its loop over frames would give nothing (Builder::emitEntry)");
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
  llvm::raw_string_ostream stream(program.module);
  stream << "module { ";
  plan.emit(stream, "", {}, {}, {}, {},
      [native](OpBuilder &where, ValueRange) -> TermIRValues {
        native->moveBefore(where.getInsertionBlock(), where.getInsertionPoint());
        return {};
      });
  stream << "}";
  stream.flush();
  require(!finishTermIRProgram(program, plan), "owned native descent failed its module boundary");
  require(module.lookupSymbol<mdir::dyn::ProgramOp>("descend") == native,
          "native function was reparsed or cloned");
  require(native.getBody().front().getArgument(0) == original,
          "native function lost its actual block argument");
  require(succeeded(verify(module)), "native descent did not verify");
}

int main(int argc, char **argv) {
  checkDiagonalExports();
  checkFrameEvaluations();
  checkNativeDescentBody();
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
  DialectRegistry registry;
  mdir::insertProgramDialects(registry);
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  ScopedDiagnosticHandler quiet(&context,
                                [](Diagnostic &) { return success(); });
  auto parse = [&](const llvm::Twine &source) {
    auto module = parseSourceString<ModuleOp>(
        source.str(), ParserConfig(&context, /*verifyAfterParse=*/false));
    require(bool(module), "term connection test did not parse");
    return module;
  };
  constexpr llvm::StringLiteral connection =
      "module { %x = builtin.unrealized_conversion_cast to f64 "
      "{mdir.term_ir = 0 : i64} }";

  {
    TermIRPlan plan;
    std::string source;
    llvm::raw_string_ostream stream(source);
    plan.emit(stream, "", {}, {}, {}, {},
              [](OpBuilder &, ValueRange) { return TermIRValues{}; });
    stream.flush();
    auto module = parse("module { " + source + " }");
    require(!plan.materialize(*module), "zero-result connection was rejected");
    require(module->getBody()->empty(), "zero-result connection survived");
    require(succeeded(verify(*module)), "zero-result module did not verify");
  }
  {
    TermIRPlan plan;
    registerRecipe(plan, [](OpBuilder &, ValueRange) -> llvm::Expected<TermIRValues> {
      return llvm::createStringError(llvm::inconvertibleErrorCode(), "input clone failed");
    });
    auto module = parse(connection);
    reject(plan.materialize(*module), "input clone failed");
  }
  {
    TermIRPlan plan;
    registerRecipe(plan);
    auto module = parse("module {}");
    reject(plan.materialize(*module),
           "a term IR recipe has no assembly connection");
  }
  {
    TermIRPlan plan;
    std::string source;
    llvm::raw_string_ostream stream(source);
    plan.emit(stream, "", {}, {}, {"%input"}, {"f64"},
              [](OpBuilder &builder, ValueRange values) -> TermIRValues {
                require(values.size() == 1 &&
                            values[0].getType() == builder.getF64Type(),
                        "zero-result connection lost its operand");
                return {};
              });
    stream.flush();
    auto module =
        parse("module { %input = arith.constant 1.0 : f64 " + source + " }");
    require(!plan.materialize(*module),
            "zero-result operand connection was rejected");
    require(module->getBody()->getOperations().size() == 1,
            "zero-result operand connection survived");
    require(succeeded(verify(*module)),
            "zero-result operand module did not verify");
  }
  for (llvm::StringRef attribute : {"0 : i64", "-1 : i64", "\"bad\""}) {
    TermIRPlan plan;
    auto module = parse("module { %x = builtin.unrealized_conversion_cast "
                        "to f64 {mdir.term_ir = " +
                        attribute + "} }");
    reject(plan.materialize(*module), "invalid internal term IR connection");
  }
  {
    TermIRPlan plan;
    registerRecipe(plan);
    auto module =
        parse("module { %x = builtin.unrealized_conversion_cast to f64 "
              "{mdir.term_ir = 0 : i64} "
              "%y = builtin.unrealized_conversion_cast to f64 "
              "{mdir.term_ir = 0 : i64} }");
    reject(plan.materialize(*module), "invalid internal term IR connection");
  }
  {
    TermIRPlan plan;
    registerRecipe(plan,
                   [](OpBuilder &, ValueRange) { return TermIRValues{}; });
    auto module = parse(connection);
    reject(plan.materialize(*module),
           "term IR result count does not match its connection");
  }
  {
    TermIRPlan plan;
    registerRecipe(plan, [](OpBuilder &builder, ValueRange) -> TermIRValues {
      return {arith::ConstantIntOp::create(builder, builder.getUnknownLoc(), 1,
                                           64)};
    });
    auto module = parse(connection);
    reject(plan.materialize(*module),
           "term IR result type does not match its connection");
  }
  {
    TermIRPlan plan;
    registerRecipe(plan, [](OpBuilder &builder, ValueRange values) {
      auto result = scalar(builder, values);
      result.front().getDefiningOp()->setAttr("mdir.term_ir",
                                              builder.getI64IntegerAttr(0));
      return result;
    });
    auto module = parse(connection);
    reject(plan.materialize(*module),
           "internal term IR connection survived materialization");
  }
  {
    // Even an identity cast could otherwise disappear in reconciliation.
    TermIRPlan plan;
    auto module =
        parse("module { %x = arith.constant 1.0 : f64 "
              "%y = builtin.unrealized_conversion_cast %x : f64 to f64 "
              "{mdir.term_ir = 0 : i64} }");
    reject(plan.materialize(*module), "invalid internal term IR connection");
  }
  {
    TermIRPlan plan;
    registerRecipe(plan);
    auto module = parse(connection);
    require(!plan.materialize(*module), "valid term connection was rejected");
    require(plan.empty(), "materialized plan retained recipes");
    require(succeeded(verify(*module)), "materialized module did not verify");
    module->walk([](Operation *op) {
      require(!op->hasAttr("mdir.term_ir"), "materialized marker survived");
    });
  }
  {
    // Invalid builder assembly fails at its producer, before a consumer sees
    // it.
    TermIRPlan plan;
    registerRecipe(plan);
    mdir::driver::Program program;
    program.module = "module { %x = builtin.unrealized_conversion_cast to f64 "
                     "{mdir.term_ir = 0 : i64} "
                     "%y = arith.addf %missing, %x : f64 }";
    auto original = program.module;
    reject(finishTermIRProgram(program, plan),
           "the builder produced invalid assembly");
    require(program.module == original, "invalid assembly was changed");
    require(!parseSourceString<ModuleOp>(program.module, &context),
            "invalid assembly became a valid consumer input");
  }
  {
    TermIRPlan empty;
    mdir::driver::Program program;
    program.module = "module { %bad = arith.addf %missing, %missing : f64 }";
    auto original = program.module;
    reject(finishTermIRProgram(program, empty),
           "the builder produced invalid assembly");
    require(program.module == original,
            "invalid assembly without recipes was changed");
  }
  {
    // Exported staged functions must not trigger the legacy raw-text fallback.
    constexpr llvm::StringLiteral helper =
        "func.func private @tensor_helper() -> tensor<1xf64> { "
        "%c = arith.constant 0.0 : f64 "
        "%t = tensor.from_elements %c : tensor<1xf64> "
        "return %t : tensor<1xf64> }";
    mdir::driver::Program canonical;
    canonical.module = ("module { " + helper + " }").str();
    auto original = canonical.module;
    mdir::driver::printParsedModule(canonical);
    require(canonical.module != original,
            "valid tensor input skipped canonical printing");
    require(bool(parseSourceString<ModuleOp>(canonical.module, &context)),
            "canonical tensor program did not verify");
    TermIRPlan empty;
    mdir::driver::Program plain;
    plain.module = ("module { " + helper + " }").str();
    require(!finishTermIRProgram(plain, empty),
            "valid tensor input without recipes failed construction");
    require(bool(parseSourceString<ModuleOp>(plain.module, &context)),
            "tensor program without recipes did not verify");
    TermIRPlan plan;
    registerRecipe(plan);
    mdir::driver::Program program;
    program.module = ("module { " + helper +
                      " %x = builtin.unrealized_conversion_cast to f64 "
                      "{mdir.term_ir = 0 : i64} }")
                         .str();
    require(!finishTermIRProgram(program, plan),
            "valid staged tensor input failed construction");
    require(bool(parseSourceString<ModuleOp>(program.module, &context)),
            "materialized staged tensor program did not verify");
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
    mdir::driver::Program program;
    program.module = "func.func @entry() { return }";
    require(!finishTermIRProgram(program, plan),
            "native program declarations did not join remaining assembly");
    require(plan.getModule() == native && &native.getBody()->front() == original,
            "native declarations were serialized and reparsed");
    require(native.getBody()->getOperations().size() == 2 &&
                succeeded(verify(native)),
            "native program lost declaration order or validity");
    require(bool(parseSourceString<ModuleOp>(program.module, &plan.getContext())),
            "completed native program did not print valid IR");
  }
  llvm::outs() << "term IR connection checks passed\n";
}
