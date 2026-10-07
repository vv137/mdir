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
/// The pipeline of `program`. On a GPU, `gpuOptions` (getGpuOptions) are
/// the options of the serialization of its kernels; empty, they are PTX
/// for the default architecture (D214).
std::string getPipeline(const driver::Control &, const driver::Program &,
                        llvm::StringRef gpuOptions = {});
/// Points CUDA_ROOT at the toolkit whose libdevice and ptxas the kernels of
/// a GPU take: the one the environment names (CUDA_ROOT, CUDA_HOME, or
/// CUDA_PATH), else that of the build, as for `mdir run`
/// (tools/mdir/BugReport.cpp, getCudaToolkitRoot).
void useCudaToolkit();
/// For a GPU program, the options of the serialization of its kernels for
/// the visible device `device` (mdir::getGpuPipelineOptions); else empty.
/// Without `cache`, the serialization bypasses the compile cache
/// (D217). Errors are CompileError.
llvm::Expected<std::string> getGpuOptions(const driver::Control &,
                                          int64_t device, bool cache = true);
struct CompiledProgram {
  driver::Program program;
  model::Execution execution;
  /// `loweredIR` is empty in the result of plan.
  std::string pipeline, loweredIR;
  /// The options of the serialization of the GPU modules (getGpuOptions).
  std::string gpuOptions;
};
/// Owns all returned data; initializes no runtime and creates no files.
/// Lower already-built IR; used by compiler integration and diagnostic tests.
/// Without `cache`, the lowering reads and writes no entry of the compile
/// cache (D217).
llvm::Expected<CompiledProgram> lower(const driver::Control &, driver::Program,
                                      const model::Execution &,
                                      bool cache = true);
llvm::Expected<CompiledProgram> compile(const model::PreparedModel &,
                                        bool cache = true);
/// Builds `prepared` and sets up its pipeline without running it, so that
/// `loweredIR` stays empty (D[compile-once]): what `mdir.compile` does, for
/// a simulation lowers programs of its own. The errors of the build, of a
/// GPU program in a build without CUDA, and of the GPU options are those of
/// compile. Without `cache`, the lowering of the result reads and writes no
/// entry of the compile cache (D217).
llvm::Expected<CompiledProgram> plan(const model::PreparedModel &,
                                     bool cache = true);
/// The text of the lowering of `compiled`, a result of plan, in a context of
/// its own. Errors are CompileError with diagnostics.
llvm::Expected<std::string> lowerToText(const driver::Control &,
                                        const CompiledProgram &compiled);
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
/// (D214).
llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
lowerModule(mlir::MLIRContext &context, const driver::Control &,
            const driver::Program &, CompileStats *stats = nullptr,
            llvm::StringRef gpuOptions = {});
/// The machine that compiles the host code of a program for the JIT, at the
/// one code-generation level of both front ends (#90).
llvm::Expected<std::unique_ptr<llvm::TargetMachine>> createHostMachine();
} }
#endif
