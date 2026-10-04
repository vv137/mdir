// Structured diagnostics preserve locations without writing a reproducer.
#include "mdir/Compiler/Compile.h"
#include "llvm/Support/raw_ostream.h"
int main() {
  mdir::driver::Control control;
  mdir::model::Execution execution;
  mdir::driver::Program program{};
  program.skin = 0.1;
  program.neighborWidth = 32;
  program.module = "module { invalid.op }";
  auto invalid = mdir::compiler::lower(control, program, execution);
  if (invalid) return 1;
  bool typed = false;
  llvm::handleAllErrors(invalid.takeError(), [&](const mdir::compiler::CompileError &e) {
    typed = e.diagnostic.find("loc(\"-\":1:") != std::string::npos &&
            e.diagnostic.find("invalid.op") != std::string::npos;
    if (!typed) llvm::errs() << e.diagnostic;
  });
  if (!typed) return 2;
  program.module = "module { func.func @run() { return } }";
  auto recovered = mdir::compiler::lower(control, program, execution);
  if (!recovered) {
    llvm::errs() << llvm::toString(recovered.takeError()); return 3;
  }
  if (recovered->loweredIR.find("llvm.func @run") == std::string::npos) return 4;
  llvm::outs() << "typed located diagnostic and compiler recovery passed\n";
}
