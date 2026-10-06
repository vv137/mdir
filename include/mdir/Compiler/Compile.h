// Shared lowering service for the CLI and embedded front ends.
#ifndef MDIR_COMPILER_COMPILE_H
#define MDIR_COMPILER_COMPILE_H
#include "mdir/Compiler/CompileCache.h"
#include "mdir/Driver/Model.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
namespace llvm { class ThreadPoolInterface; }
namespace mdir { namespace compiler {
class CompileError : public llvm::ErrorInfo<CompileError> {
public:
  static char ID;
  explicit CompileError(std::string diagnostic) : diagnostic(std::move(diagnostic)) {}
  void log(llvm::raw_ostream &os) const override { os << diagnostic; }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
  std::string diagnostic;
};
std::string getPipeline(const driver::Control &, const driver::Program &);
struct CompiledProgram {
  driver::Program program;
  model::Execution execution;
  std::string pipeline, loweredIR;
};
/// Owns all returned data; initializes no runtime and creates no files.
/// Lower already-built IR; used by compiler integration and diagnostic tests.
llvm::Expected<CompiledProgram> lower(const driver::Control &, driver::Program,
                                      const model::Execution &);
llvm::Expected<CompiledProgram> compile(const model::PreparedModel &);
/// The dialects, extensions and translations of a lowering; registers the
/// passes once per process.
mlir::DialectRegistry getRegistry();
/// The threads that the MLIR contexts of the lowerings of a process share,
/// one pool for the process, so that no context keeps a pool of its own
/// (D211). A child forked after a lowering makes a
/// pool of its own; the threads of its parent's do not exist in it.
llvm::ThreadPoolInterface &getThreadPool();
/// Makes `context`, created with threading disabled, lower with the threads
/// of getThreadPool.
void shareThreadPool(mlir::MLIRContext &context);
/// Parses and lowers `program` in `context`, for a front end that runs the
/// result (D196). Errors are CompileError with diagnostics. What the
/// serialization of the GPU modules did is added to `stats`, if given
/// (D[gpu-module-compile]).
llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
lowerModule(mlir::MLIRContext &context, const driver::Control &,
            const driver::Program &, CompileStats *stats = nullptr);
/// The machine that compiles the host code of a program for the JIT, at the
/// one code-generation level of both front ends (#90).
llvm::Expected<std::unique_ptr<llvm::TargetMachine>> createHostMachine();
} }
#endif
