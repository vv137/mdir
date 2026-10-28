// Typed fragments become operations in their original shared regions (#568).
#include "TermIR.h"
#include "BuilderImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mdir::driver::builder;

namespace {
constexpr llvm::StringLiteral recipeAttribute = "mdir.term_ir";
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
  os << "builtin.unrealized_conversion_cast";
  if (!operands.empty()) {
    os << " ";
    llvm::interleaveComma(operands, os);
    os << " : ";
    llvm::interleaveComma(operandTypes, os);
  }
  os << " to ";
  if (resultTypes.empty())
    os << "()";
  else
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
    auto values = recipes[index.getInt()](builder, connection.getOperands());
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
  recipes.clear();
  return llvm::Error::success();
}

std::string mdir::driver::builder::emitExpressionIR(
    TermEmission &context, const mdir::driver::Expression &expression,
    const llvm::StringMap<std::string> &bindings, llvm::StringRef prefix,
    llvm::StringRef indent, llvm::StringRef termName) {
  auto program = expression.prepareIR();
  llvm::SmallVector<llvm::StringRef> supplied;
  for (const auto &entry : bindings)
    supplied.push_back(entry.getKey());
  auto inputs = program.getInputs(supplied);
  std::vector<std::string> names;
  llvm::SmallVector<llvm::StringRef> types;
  for (const auto &name : inputs.names) {
    auto found = bindings.find(name);
    assert(found != bindings.end() && "missing expression binding");
    names.push_back(found->second);
    types.push_back("f64");
  }
  for (const auto &name : inputs.tables) {
    names.push_back("%t_" + name);
    types.push_back("!grid");
  }
  llvm::SmallVector<llvm::StringRef> operands;
  for (const auto &name : names)
    operands.push_back(name);
  std::string result = prefix.str() + "ir_result";
  context.emitIRFragment(
      indent, {result}, {"f64"}, operands, types,
      [program, inputs, term = termName.str()](
          mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        llvm::StringMap<mlir::Value> bindings, tables;
        for (size_t k = 0; k != inputs.names.size(); ++k)
          bindings[inputs.names[k]] = values[k];
        for (size_t k = 0; k != inputs.tables.size(); ++k)
          tables[inputs.tables[k]] = values[inputs.names.size() + k];
        return {program.build(builder, bindings, tables, term)};
      });
  return result;
}
