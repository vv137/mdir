#include "mdir/Compiler/Compile.h"
#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/Transforms/Passes.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include <mutex>
using namespace mdir;
using namespace mdir::driver;
using llvm::StringRef;
char compiler::CompileError::ID = 0;
std::string mdir::compiler::getPipeline(const Control &control,
                               const Program &program) {
  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  os << "md-check-exchange,md-differentiate,md-expand-truncation,md-inline,"
     << "md-bypass-updates,";
  os << "convert-md-to-md-exec{skin=" << program.skin
     << " width=" << program.neighborWidth << "},";
  os << "md-exec-reuse-neighbors";
  if (program.pruneSkin > 0.0)
    os << "{prune-skin=" << program.pruneSkin << "}";
  os << ",";
  // Opt-in only (D88): the structures may miss pairs; the run warns.
  if (control.rebuildPeriod > 0)
    os << "md-exec-rebuild-at-interval{interval=" << control.rebuildPeriod
       << "},";
  os << "md-exec-expose-validity,"
     << "md-exec-fuse-loops,md-exec-accumulate-destinations,"
     << "md-exec-narrow-sums,";
  if (control.fastMath)
    os << "md-exec-simplify-distance{radial=true},";
  os << "canonicalize,cse,md-exec-fold-tables,canonicalize,cse,";
  // Before the precision: a loop over groups keeps the positions as they
  // are stored (D95).
  if (control.target == Target::GPU &&
      control.neighborStructure == NeighborStructure::Groups)
    os << "md-exec-choose-neighbors{kind=groups},";

  StringRef mode = control.precision == Precision::Single
                       ? "single"
                       : control.precision == Precision::Mixed ? "mixed"
                                                               : "double";
  // In the deterministic mode the fields of the steps are stored as their
  // roles say, whatever buffers of host calls the loops reach (#102).
  os << "md-exec-assign-precision{mode=" << mode
     << (control.deterministic ? " deterministic=true" : "") << "},";
  if (control.fastMath)
    os << "md-exec-approximate,md-exec-expand-radial,canonicalize,cse,";

  if (control.target == Target::GPU) {
    // Kernels in f32 look their tables up in f32.
    os << "md-exec-assign-storage{memory=device"
       << (control.precision == Precision::Double ? "" : " tables=f32")
       << "},md-exec-assign-streams,convert-md-exec-to-gpu{"
       // In the deterministic mode the loops of the constraints are not
       // joined with the loops over particles around them into one kernel
       // (D110): a step of energy, whose virial reads the changes of the
       // constraints, cannot join them, and the joined kernel rounds the
       // velocities of SETTLE otherwise by an ulp of f32 (#102).
       //
       // Nor does the arithmetic of a step depend on what else the step
       // computes: LLVM contracts a product into a fused multiply-add only
       // where it has one use, and the energies and the virial of a step
       // that writes them give products more uses, which changed the
       // rounding of its forces (#97). LLVM does not contract there; every
       // sum with a product as an operand becomes a fused multiply-add
       // instead, by the formula alone (#102).
       << (control.deterministic
               ? "deterministic=true fuse-integration=false explicit-fma=true "
               : "")
       << (control.fastMath && !control.deterministic ? "" : "contract=false")
       << "},"
       << "gpu-lower-to-nvvm-pipeline{cubin-format=isa},"
       << "reconcile-unrealized-casts";
    return pipeline;
  }

  // Reductions summed over fixed chunks in a fixed order: the same bits
  // from run to run and for any number of threads (D171).
  os << "md-exec-assign-storage,convert-md-exec-to-loops,fixed-order-reductions,";
  bool threaded = control.threads > 1;
  if (threaded)
    os << "convert-scf-to-openmp,hoist-static-allocas,canonicalize,";
  os << "convert-scf-to-cf,convert-math-to-llvm,convert-math-to-libm,"
     << "convert-vector-to-llvm,expand-strided-metadata,"
     << "finalize-memref-to-llvm,convert-arith-to-llvm,"
     << "convert-func-to-llvm,convert-cf-to-llvm,convert-ub-to-llvm,";
  if (threaded)
    os << "convert-openmp-to-llvm,";
  os << "reconcile-unrealized-casts";
  return pipeline;
}


llvm::Expected<compiler::CompiledProgram>
compiler::compile(const model::PreparedModel &prepared) {
  auto program = prepared.build();
  if (!program)
    return program.takeError();
  return lower(prepared.control, std::move(*program), prepared.execution);
}
mlir::DialectRegistry compiler::getRegistry() {
  // Process-wide registries are initialized once; each lowering owns a context.
  static std::once_flag registration;
  std::call_once(registration, [] {
    mlir::registerAllPasses();
    md::registerMDPasses();
    md_exec::registerMDExecPasses();
    registerMDIRConversionPasses();
  });
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  md::registerDerivativeInterfaces(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  registry.insert<dyn::DynDialect, md::MDDialect, md_exec::MDExecDialect,
                  mdrt::MDRTDialect>();
  return registry;
}
llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
compiler::lowerModule(mlir::MLIRContext &context, const driver::Control &control,
                      const driver::Program &program) {
  std::string diagnostics;
  llvm::raw_string_ostream diagnosticStream(diagnostics);
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
    d.getLocation().print(diagnosticStream);
    diagnosticStream << ": ";
    d.print(diagnosticStream);
    diagnosticStream << "\n";
    return mlir::success();
  });
  auto module = mlir::parseSourceString<mlir::ModuleOp>(program.module, &context);
  if (!module)
    return llvm::make_error<CompileError>("cannot parse generated IR: " + diagnostics);
  std::string pipeline = getPipeline(control, program);
  mlir::PassManager manager(&context, mlir::ModuleOp::getOperationName(),
                             mlir::PassManager::Nesting::Implicit);
  if (mlir::failed(mlir::parsePassPipeline(pipeline, manager, diagnosticStream)))
    return llvm::make_error<CompileError>("cannot set up lowering: " + diagnostics);
  if (mlir::failed(manager.run(*module)))
    return llvm::make_error<CompileError>("cannot lower program: " + diagnostics);
  return std::move(module);
}
llvm::Expected<compiler::CompiledProgram>
compiler::lower(const driver::Control &control, driver::Program program,
                const model::Execution &execution) {
#if !MDIR_HAS_CUDA
  if (execution.target == Target::GPU)
    return llvm::make_error<model::ModelError>(model::ModelError::Unsupported,
                                              "GPU compilation requires a CUDA-enabled build");
#endif
  mlir::MLIRContext context(getRegistry());
  auto module = lowerModule(context, control, program);
  if (!module)
    return module.takeError();
  std::string lowered;
  llvm::raw_string_ostream os(lowered);
  (*module)->print(os);
  std::string pipeline = getPipeline(control, program);
  return CompiledProgram{std::move(program), execution,
                         std::move(pipeline), std::move(lowered)};
}

// MLIR's ExecutionEngine compiles with the machine it is given, or with one
// of the default level that it makes itself; its option
// `jitCodeGenOptLevel` sets the level of a builder that it does not use.
// The level is therefore that of this machine. `Aggressive` measured no
// gain over `Default` on the Amber suite (GPU, within 0.3%) nor on JAC on
// the CPU (2%, within the noise), so the host code takes `Default` (#90).
llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
compiler::createHostMachine() {
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder)
    return builder.takeError();
  builder->setCodeGenOptLevel(llvm::CodeGenOptLevel::Default);
  return builder->createTargetMachine();
}
