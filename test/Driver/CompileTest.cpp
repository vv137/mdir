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

  // A derivative without a rule is an error of the input before anything
  // is lowered, which names the op, the argument of the potential, and the
  // tunable that the argument is (D230, #256): here a field that a map
  // computes another field from.
  auto potential = [](const char *body) {
    return std::string(R"(
!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
md.particle_set @atoms
md.potential @tunable(%x: !vec, %cell: !md.cell, %q: !real) -> f64 {
)") + body + R"(
  md.return %s : f64
}
md.function @entry(%x: !vec, %cell: !md.cell, %q: !real) -> !real {
  %g = md.evaluate @tunable(%x, %cell, %q) request [derivative(2)]
      : (!vec, !md.cell, !real) -> !real
  md.return %g : !real
}
)";
  };
  control.tunableDeclarations.push_back({"charges", {}, 0, "charge"});
  program.tunableGradient = true;
  program.gradientArguments = {{2, 0}};
  program.module = potential(R"(
  %h = md.map_particles gather(%q : !real) {
  ^bb0(%q_i: f64):
    %k = arith.mulf %q_i, %q_i : f64
    md.yield %k : f64
  } : !real
  %s = md.sum_particles gather(%h : !real) {
  ^bb0(%h_i: f64):
    md.yield %h_i : f64
  } : f64)");
  std::string message;
  llvm::handleAllErrors(mdir::compiler::checkDerivatives(control, program),
                        [&](const mdir::model::ModelError &e) {
    if (e.kind == mdir::model::ModelError::Input) message = e.message;
  });
  for (const char *part :
       {"the program asks for a derivative that has no rule "
        "(System.tunable_gradient)",
        "'md.map_particles' takes argument 2 of 'tunable', a field, and has "
        "no rule for the derivative in it",
        "argument 2 of 'tunable' is the tunable 'charges'"})
    if (message.find(part) == std::string::npos) {
      llvm::errs() << "expected '" << part << "' in: " << message << "\n";
      return 5;
    }
  // With a rule, nothing is reported, and nothing is lowered.
  program.module = potential(R"(
  %s = md.sum_particles gather(%q : !real) {
  ^bb0(%q_i: f64):
    %k = arith.mulf %q_i, %q_i : f64
    md.yield %k : f64
  } : f64)");
  if (llvm::Error error = mdir::compiler::checkDerivatives(control, program)) {
    llvm::errs() << llvm::toString(std::move(error)); return 6;
  }
  llvm::outs() << "a derivative without a rule is an input error that names "
                  "the op and the tunable\n";
}
