// Typed fragments become operations in their original shared regions (#568).
#include "TermIR.h"
#include "mdir/Dialect/ProgramDialects.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"

using namespace mdir::driver::builder;

namespace {
constexpr llvm::StringLiteral recipeAttribute = "mdir.term_ir";
}

struct TermIRPlan::OwnedProgram {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

TermIRPlan::TermIRPlan() = default;
TermIRPlan::~TermIRPlan() = default;

mlir::MLIRContext &TermIRPlan::getContext() {
  if (!program) {
    program = std::make_unique<OwnedProgram>();
    mlir::DialectRegistry registry;
    mdir::insertProgramDialects(registry);
    program->context = std::make_unique<mlir::MLIRContext>(
        registry, mlir::MLIRContext::Threading::DISABLED);
    program->context->loadAllAvailableDialects();
  }
  return *program->context;
}

mlir::ModuleOp TermIRPlan::getModule() {
  mlir::MLIRContext &context = getContext();
  if (!program->module)
    program->module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  return *program->module;
}

void TermIRPlan::emit(llvm::raw_ostream &os, llvm::StringRef indent,
                      llvm::ArrayRef<llvm::StringRef> results,
                      llvm::ArrayRef<llvm::StringRef> resultTypes,
                      llvm::ArrayRef<llvm::StringRef> operands,
                      llvm::ArrayRef<llvm::StringRef> operandTypes,
                      TermIRRecipe recipe) {
  assert(results.size() == resultTypes.size());
  assert(operands.size() == operandTypes.size());
  size_t index = recipes.size();
  recipes.push_back(std::move(recipe));
  os << indent;
  if (!results.empty()) {
    llvm::interleaveComma(results, os);
    os << " = ";
  }
  if (resultTypes.empty()) {
    // The custom cast parser requires a nonempty output type list. Generic
    // syntax can express the zero-result connection removed at materialization.
    os << "\"builtin.unrealized_conversion_cast\"(";
    llvm::interleaveComma(operands, os);
    os << ") {" << recipeAttribute << " = " << index << " : i64} : (";
    llvm::interleaveComma(operandTypes, os);
    os << ") -> ()\n";
    return;
  }
  os << "builtin.unrealized_conversion_cast";
  if (!operands.empty()) {
    os << " ";
    llvm::interleaveComma(operands, os);
    os << " : ";
    llvm::interleaveComma(operandTypes, os);
  }
  os << " to ";
  if (!resultTypes.empty())
    llvm::interleaveComma(resultTypes, os);
  os << " {" << recipeAttribute << " = " << index << " : i64}\n";
}

llvm::Error TermIRPlan::materialize(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> connections;
  module.walk([&](mlir::UnrealizedConversionCastOp op) {
    if (op->hasAttr(recipeAttribute))
      connections.push_back(op);
  });
  std::vector<bool> seen(recipes.size(), false);
  mlir::OpBuilder builder(module.getContext());
  for (auto connection : connections) {
    auto index = connection->getAttrOfType<mlir::IntegerAttr>(recipeAttribute);
    if (!index || index.getInt() < 0 ||
        uint64_t(index.getInt()) >= recipes.size() || seen[index.getInt()])
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid internal term IR connection");
    seen[index.getInt()] = true;
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(connection);
    auto result = recipes[index.getInt()](builder, connection.getOperands());
    if (!result)
      return result.takeError();
    const auto &values = *result;
    if (values.size() != connection.getNumResults())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "term IR result count does not match its connection");
    for (auto [oldValue, newValue] :
         llvm::zip(connection.getResults(), values)) {
      if (!newValue || oldValue.getType() != newValue.getType())
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "term IR result type does not match its connection");
      oldValue.replaceAllUsesWith(newValue);
    }
    connection.erase();
  }
  if (llvm::is_contained(seen, false))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "a term IR recipe has no assembly connection");
  // Recipes may construct nested operations. Check the whole result, including
  // non-cast operations, so reconciliation cannot silently consume a marker.
  bool remaining = false;
  module.walk(
      [&](mlir::Operation *op) { remaining |= op->hasAttr(recipeAttribute); });
  if (remaining)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "internal term IR connection survived materialization");
  recipes.clear();
  return llvm::Error::success();
}
