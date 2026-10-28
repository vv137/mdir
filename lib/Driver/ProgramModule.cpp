//===- ProgramModule.cpp - The module of a program, parsed and printed ----===//
//
// The builder writes the module of a program as MLIR text (Builder.cpp).
// The program keeps it as MLIR's printer prints it once the text is parsed
// (#568): with the SSA names, the numbers, and the layout of the printer,
// so that the text no longer depends on how the builder spells it, and the
// parts of the builder that build IR with OpBuilder instead of text keep
// the same program.
//
// The text keeps the locations of the ops, printed as aliases at its end
// (`loc(#loc12)`, `#loc12 = loc(...)`). The ops of a custom expression
// carry the name of their term (`loc("control-file term: wall")`,
// Expression.cpp), which the diagnostics of the passes name; the other
// ops carry their position in the builder's text, which nothing reads
// since the kernels are named without it (D354) and which
// diagnostics may still point at. `mdir emit --stage=module` and Python's
// `Program.ir` show the
// module for reading (printModuleForReading): the locations that the
// builder gives inline, as its text had them, and the positions in its
// text left out. The equivalence check of the builder (docs/driver.md,
// Section 2.11) compares the module as the default printer prints it,
// without locations, and the lowered program.
//
//===----------------------------------------------------------------------===//

#include "TermIR.h"
#include "mdir/Dialect/ProgramDialects.h"
#include "mdir/Driver/Builder.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

/// The module of `text` as the printer prints it with `flags`, or nothing
/// if the text does not parse.
/// The loops that carry values into code that the builder wrote outside
/// them (#568): the branches of the start of a program of segments
/// (Builder::emitStartBranch) and the loop over its parts
/// (Builder::emitPartLoop). The builder writes that code inside the loop
/// as it is, using the values given, and marks the loop `mdir.carries`.
/// Inside the loop those values are the loop's arguments, which the given
/// values initialize. In the loop over parts, a cast marked
/// `mdir.part_start` takes the step that the part begins after and the
/// %start of the entry: inside the loop, the uses of the second are the
/// first, and the cast goes. The builder renamed these values in its text
/// before (`%x0` to `%x0_a` and `%x0$part`, `%start` to `%start$part`;
/// renameValue).
void carryLoopValues(mlir::ModuleOp module) {
  module.walk([](mlir::scf::ForOp loop) {
    if (!loop->hasAttr("mdir.carries"))
      return;
    loop->removeAttr("mdir.carries");
    auto inside = [&](mlir::OpOperand &use) {
      return loop->isProperAncestor(use.getOwner());
    };
    for (auto [given, argument] :
         llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs()))
      given.replaceUsesWithIf(argument, inside);
    llvm::SmallVector<mlir::UnrealizedConversionCastOp> marks;
    loop.getBody()->walk([&](mlir::UnrealizedConversionCastOp cast) {
      if (cast->hasAttr("mdir.part_start"))
        marks.push_back(cast);
    });
    for (mlir::UnrealizedConversionCastOp mark : marks) {
      mlir::Value part = mark.getOperand(0), start = mark.getOperand(1);
      mark.erase();
      start.replaceUsesWithIf(part, inside);
    }
  });
}

std::optional<std::string>
reprint(llvm::StringRef text, mlir::OpPrintingFlags flags, bool carry = false) {
  mlir::MLIRContext context(getProgramDialects(),
                            mlir::MLIRContext::Threading::DISABLED);
  mlir::ScopedDiagnosticHandler quiet(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    return std::nullopt;
  if (carry)
    carryLoopValues(*module);
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
  if (auto printed = reprint(program.module, flags, /*carry=*/true))
    program.module = std::move(*printed);
}

llvm::Error mdir::driver::builder::finishTermIRProgram(Program &program,
                                                       TermIRPlan &plan) {
  if (plan.empty()) {
    mdir::driver::printParsedModule(program);
    return llvm::Error::success();
  }
  mlir::MLIRContext context(getProgramDialects(),
                            mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  mlir::ScopedDiagnosticHandler quiet(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  // The transitional connections are not executable scalar operations.
  // Materialize their checked bodies before the ordinary verification boundary.
  mlir::ParserConfig config(&context, /*verifyAfterParse=*/false);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(program.module, config);
  if (!module) {
    // Preserve the original parser diagnostic for already-invalid assembly
    // (including #706). The reading printer may return this invalid text for
    // inspection, but canonical printing and lowering parse it again and fail.
    // No valid module reaches cast reconciliation on this path.
    return llvm::Error::success();
  }
  if (llvm::Error error = plan.materialize(*module))
    return error;
  carryLoopValues(*module);
  std::string printed;
  llvm::raw_string_ostream os(printed);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  module->print(os, flags);
  os << "\n";
  os.flush();
  program.module = std::move(printed);
  return llvm::Error::success();
}
