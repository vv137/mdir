//===- ProgramModule.cpp - The module of a program, parsed and printed ----===//
//
// The builder owns a native module for its entire construction lifetime.
// serializeNativeProgram prints it once for the serialized consumer boundary.
// printModuleForReading and printParsedModule are consumers of serialized IR;
// they may parse it for display and keep user-supplied expression locations.
// The equivalence gate uses the default printer without locations.
//
//===----------------------------------------------------------------------===//

#include "TermIR.h"
#include "mdir/Dialect/ProgramDialects.h"
#include "mdir/Driver/Builder.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include <optional>

using namespace mdir;

namespace {
mlir::DialectRegistry getProgramDialects() {
  mlir::DialectRegistry registry;
  insertProgramDialects(registry);
  return registry;
}

std::optional<std::string>
reprint(llvm::StringRef text, mlir::OpPrintingFlags flags) {
  mlir::MLIRContext context(getProgramDialects(),
                            mlir::MLIRContext::Threading::DISABLED);
  mlir::ScopedDiagnosticHandler quiet(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    return std::nullopt;
  std::string printed;
  llvm::raw_string_ostream os(printed);
  module->print(os, flags);
  os << "\n";
  os.flush();
  return printed;
}
} // namespace

std::string driver::printModuleForReading(const Program &program) {
  mlir::MLIRContext context(getProgramDialects(),
                            mlir::MLIRContext::Threading::DISABLED);
  mlir::ScopedDiagnosticHandler quiet(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(program.module, &context);
  if (!module)
    return program.module;
  // The positions in the builder's text are file locations; the locations
  // that the builder gives (the terms of custom expressions) are names.
  auto unknown = mlir::UnknownLoc::get(&context);
  auto plain = [&](mlir::Location location) {
    return mlir::isa<mlir::FileLineColLoc>(location) ? unknown : location;
  };
  module->getOperation()->walk([&](mlir::Operation *op) {
    op->setLoc(plain(op->getLoc()));
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          argument.setLoc(plain(argument.getLoc()));
  });
  // In local scope the locations are printed where they are, not as
  // aliases at the end.
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  flags.useLocalScope();
  std::string printed;
  llvm::raw_string_ostream os(printed);
  module->print(os, flags);
  os << "\n";
  os.flush();
  // An unknown location says nothing to a reader.
  std::string text;
  llvm::StringRef rest = printed;
  for (size_t at = rest.find(" loc(unknown)"); at != llvm::StringRef::npos;
       at = rest.find(" loc(unknown)")) {
    text += rest.substr(0, at).str();
    rest = rest.substr(at + 13);
  }
  text += rest.str();
  return text;
}

void driver::printParsedModule(Program &program) {
  // A text that does not parse is left as it is: the lowering parses it
  // again and reports the error, as it did before the text was printed.
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  if (auto printed = reprint(program.module, flags))
    program.module = std::move(*printed);
}

llvm::Error mdir::driver::builder::serializeNativeProgram(Program &program,
                                                       TermIRPlan &plan) {
  auto native = plan.getModule();
  std::string printed;
  llvm::raw_string_ostream os(printed);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  native.print(os, flags);
  os << "\n";
  os.flush();
  program.module = std::move(printed);
  return llvm::Error::success();
}
