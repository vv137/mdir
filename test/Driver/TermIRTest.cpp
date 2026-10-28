// Internal term connections must disappear before ordinary verification.
#include "../../lib/Driver/TermIR.h"
#include "mdir/Driver/Builder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
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

int main() {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
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
    // The legacy error path retains only assembly that cannot be parsed.
    // This is the missing SSA value class of failure tracked by #706.
    TermIRPlan plan;
    registerRecipe(plan);
    mdir::driver::Program program;
    program.module = "module { %x = builtin.unrealized_conversion_cast to f64 "
                     "{mdir.term_ir = 0 : i64} "
                     "%y = arith.addf %missing, %x : f64 }";
    auto original = program.module;
    require(!finishTermIRProgram(program, plan), "legacy error path changed");
    require(program.module == original, "invalid assembly was changed");
    require(!parseSourceString<ModuleOp>(program.module, &context),
            "invalid assembly became a valid consumer input");
  }
  llvm::outs() << "term IR connection checks passed\n";
}
