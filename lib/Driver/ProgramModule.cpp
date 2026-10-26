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

#include "mdir/Driver/Builder.h"
#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include <optional>

using namespace mdir;

namespace {
mlir::DialectRegistry getProgramDialects() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                  dyn::DynDialect, md::MDDialect, md_exec::MDExecDialect,
                  mdrt::MDRTDialect>();
  return registry;
}

/// The module of `text` as the printer prints it with `flags`, or nothing
/// if the text does not parse.
/// The loops of the start of a program of segments that carry values
/// (Builder::emitStartBranch, #568): the builder writes the code of the
/// start inside such a loop as it is, using the values given, and marks
/// the loop `mdir.start_branch`. Inside the loop those values are the
/// loop's arguments, which the given values initialize; the builder
/// renamed them in its text before (`%x0` to `%x0_a`, renameValue).
void carryStartBranches(mlir::ModuleOp module) {
  module.walk([](mlir::scf::ForOp loop) {
    if (!loop->hasAttr("mdir.start_branch"))
      return;
    loop->removeAttr("mdir.start_branch");
    for (auto [given, argument] :
         llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs()))
      given.replaceUsesWithIf(argument, [&](mlir::OpOperand &use) {
        return loop->isProperAncestor(use.getOwner());
      });
  });
}

std::optional<std::string> reprint(llvm::StringRef text,
                                   mlir::OpPrintingFlags flags,
                                   bool carry = false) {
  mlir::MLIRContext context(getProgramDialects(),
                            mlir::MLIRContext::Threading::DISABLED);
  mlir::ScopedDiagnosticHandler quiet(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    return std::nullopt;
  if (carry)
    carryStartBranches(*module);
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

