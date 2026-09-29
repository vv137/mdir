// mdir-run: reads a control file, compiles the run, and executes it.
//
// See docs/driver-m0.md.

#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/Transforms/Passes.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/Output.h"
#include "mdir/Driver/System.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace mdir;
using namespace mdir::driver;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;

namespace {
enum class Emit { Module, Lowered, Run };
} // namespace

static llvm::cl::opt<std::string>
    controlFile(llvm::cl::Positional, llvm::cl::desc("<control file>"));

static llvm::cl::opt<std::string> templateName(
    "template",
    llvm::cl::desc("Print a control file with every keyword: md"),
    llvm::cl::value_desc("kind"));

static llvm::cl::opt<Emit> emit(
    "emit", llvm::cl::desc("What to do with the program of the run"),
    llvm::cl::values(
        clEnumValN(Emit::Module, "mlir", "Print it as it is built"),
        clEnumValN(Emit::Lowered, "lowered",
                   "Print it as it is executed, in the LLVM dialect"),
        clEnumValN(Emit::Run, "run", "Execute it (default)")),
    llvm::cl::init(Emit::Run));

/// The passes that compile the program of a run.
static std::string getPipeline(const Control &control,
                               const Program &program) {
  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  os << "md-check-exchange,md-differentiate,md-expand-truncation,md-inline,";
  os << "convert-md-to-md-exec{skin=" << program.skin
     << " width=" << program.neighborWidth << "},";
  os << "md-exec-reuse-neighbors,md-exec-fuse-loops,";
  if (control.fastMath)
    os << "md-exec-simplify-distance,";
  os << "canonicalize,cse,";

  StringRef mode = control.precision == Precision::Single
                       ? "single"
                       : control.precision == Precision::Mixed ? "mixed"
                                                               : "double";
  os << "md-exec-assign-precision{mode=" << mode << "},";

  if (control.target == Target::GPU) {
    os << "md-exec-assign-storage{memory=device},convert-md-exec-to-gpu,"
       << "gpu-lower-to-nvvm-pipeline{cubin-format=isa},"
       << "reconcile-unrealized-casts";
    return pipeline;
  }

  os << "md-exec-assign-storage,convert-md-exec-to-loops,";
  bool threaded = control.threads > 1;
  if (threaded)
    os << "convert-scf-to-openmp,canonicalize,";
  os << "convert-scf-to-cf,convert-math-to-llvm,convert-math-to-libm,"
     << "convert-vector-to-llvm,expand-strided-metadata,"
     << "finalize-memref-to-llvm,convert-arith-to-llvm,"
     << "convert-func-to-llvm,convert-cf-to-llvm,";
  if (threaded)
    os << "convert-openmp-to-llvm,";
  os << "reconcile-unrealized-casts";
  return pipeline;
}

/// A buffer of the host, in the type that the program takes, and its
/// descriptor.
template <unsigned Rank>
struct Buffer {
  Buffer(const std::vector<double> &values, Element element, size_t count,
         double scale = 1.0) {
    size_t width = element == Element::F32 ? sizeof(float) : sizeof(double);
    storage.resize(values.size() * width);
    for (size_t i = 0, e = values.size(); i != e; ++i) {
      if (element == Element::F32) {
        float value = static_cast<float>(scale * values[i]);
        std::memcpy(&storage[i * width], &value, width);
      } else {
        double value = scale * values[i];
        std::memcpy(&storage[i * width], &value, width);
      }
    }
    descriptor.basePtr = descriptor.data = storage.data();
    descriptor.offset = 0;
    descriptor.sizes[0] = count;
    descriptor.strides[Rank - 1] = 1;
    if (Rank == 2) {
      descriptor.sizes[1] = 3;
      descriptor.strides[0] = 3;
    }
  }

  /// Adds the buffer to the arguments of a function, as the execution
  /// engine passes a buffer: the parts of the descriptor, one by one.
  void addTo(SmallVectorImpl<void *> &arguments) {
    arguments.push_back(&descriptor.basePtr);
    arguments.push_back(&descriptor.data);
    arguments.push_back(&descriptor.offset);
    for (unsigned i = 0; i != Rank; ++i)
      arguments.push_back(&descriptor.sizes[i]);
    for (unsigned i = 0; i != Rank; ++i)
      arguments.push_back(&descriptor.strides[i]);
  }

  std::vector<char> storage;
  StridedMemRefType<char, Rank> descriptor;
};

static int fail(llvm::Error error) {
  llvm::errs() << "mdir-run: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

static int fail(const llvm::Twine &message) {
  llvm::errs() << "mdir-run: " << message << "\n";
  return 1;
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "MDIR: compiles and runs a molecular dynamics run\n");

  if (!templateName.empty()) {
    if (templateName != "md")
      return fail("expected the template 'md', got '" + templateName + "'");
    llvm::outs() << getControlTemplate();
    return 0;
  }
  if (controlFile.empty())
    return fail("expected a control file; see --help");

  //===--------------------------------------------------------------------===//
  // Read
  //===--------------------------------------------------------------------===//

  auto control = readControl(controlFile);
  if (!control)
    return fail(control.takeError());
  auto system = readSystem(*control);
  if (!system)
    return fail(system.takeError());
  assignVelocities(*control, *system);

  auto program = buildProgram(*control, *system);
  if (!program)
    return fail(program.takeError());
  if (emit == Emit::Module) {
    llvm::outs() << program->module;
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Compile
  //===--------------------------------------------------------------------===//

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::registerAllPasses();
  md::registerMDPasses();
  md_exec::registerMDExecPasses();
  registerMDIRConversionPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  registry.insert<dyn::DynDialect, md::MDDialect, md_exec::MDExecDialect,
                  mdrt::MDRTDialect>();
  mlir::MLIRContext context(registry);

  // The kernels for a GPU take their math functions from the CUDA toolkit.
  if (control->target == Target::GPU && !std::getenv("CUDA_ROOT") &&
      !std::getenv("CUDA_HOME") && !std::getenv("CUDA_PATH") &&
      StringRef(MDIR_CUDA_ROOT) != "")
    setenv("CUDA_ROOT", MDIR_CUDA_ROOT, /*overwrite=*/0);

  auto begin = std::chrono::steady_clock::now();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(program->module, &context);
  if (!module)
    return fail("the program of the run does not parse; this is a defect "
                "of mdir-run");

  // Passes on functions are nested where they occur, as on the command
  // line of mlir-opt.
  mlir::PassManager manager(&context,
                            mlir::ModuleOp::getOperationName(),
                            mlir::PassManager::Nesting::Implicit);
  std::string pipeline = getPipeline(*control, *program);
  if (mlir::failed(mlir::parsePassPipeline(pipeline, manager, llvm::errs())))
    return fail("cannot set up the passes");
  if (mlir::failed(manager.run(*module)))
    return fail("cannot compile the run");

  if (emit == Emit::Lowered) {
    module->print(llvm::outs());
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Load
  //===--------------------------------------------------------------------===//

  // The runtime is next to the driver: <prefix>/bin and <prefix>/lib.
  std::string executable =
      llvm::sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)&main);
  llvm::SmallString<256> libraries(llvm::sys::path::parent_path(
      llvm::sys::path::parent_path(executable)));
  llvm::sys::path::append(libraries, "lib");
  auto getLibrary = [](StringRef directory, StringRef name) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, name);
    return std::string(path);
  };

  std::vector<std::string> paths = {getLibrary(libraries, "libmdrt.so")};
  if (control->target == Target::GPU) {
    paths.push_back(getLibrary(libraries, "libmdrt_cuda.so"));
  } else if (control->threads > 1) {
    setenv("OMP_NUM_THREADS", std::to_string(control->threads).c_str(),
           /*overwrite=*/1);
    paths.push_back(getLibrary(MDIR_LLVM_LIBRARY_DIR, "libomp.so"));
  }
  for (const std::string &path : paths)
    if (!llvm::sys::fs::exists(path))
      return fail("cannot find '" + path + "'");

  mlir::ExecutionEngineOptions options;
  SmallVector<StringRef> sharedLibraries(paths.begin(), paths.end());
  options.sharedLibPaths = sharedLibraries;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  auto engine = mlir::ExecutionEngine::create(*module, options);
  if (!engine)
    return fail(engine.takeError());

  bool single = program->state == Element::F32;
  (*engine)->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    auto add = [&](StringRef name, void *function) {
      symbols[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(function),
                                 llvm::JITSymbolFlags::Exported};
    };
    add("_mlir_ciface_mdrtWriteEnergies",
        (void *)&_mlir_ciface_mdrtWriteEnergies);
    add("_mlir_ciface_mdrtWriteFrame",
        single ? (void *)&_mlir_ciface_mdrtWriteFrame_f32
               : (void *)&_mlir_ciface_mdrtWriteFrame_f64);
    add("_mlir_ciface_mdrtFinish",
        single ? (void *)&_mlir_ciface_mdrtFinish_f32
               : (void *)&_mlir_ciface_mdrtFinish_f64);
    return symbols;
  });

  // Loads the kernels of a GPU. The functions of the driver must be known
  // by now.
  (*engine)->initialize();

  auto function = (*engine)->lookupPacked(program->entry);
  if (!function)
    return fail(function.takeError());
  double compileTime = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - begin)
                           .count();

  //===--------------------------------------------------------------------===//
  // Run
  //===--------------------------------------------------------------------===//

  size_t count = system->getNumParticles();
  Buffer<2> positions(system->positions, program->state, count);
  Buffer<2> velocities(system->velocities, program->state, count);
  Buffer<1> masses(system->masses, program->mass, count);
  std::vector<std::unique_ptr<Buffer<1>>> fields;
  for (const Program::Field &field : program->fields)
    fields.push_back(std::make_unique<Buffer<1>>(field.values,
                                                 program->parameter, count));

  double box[3] = {system->box[0], system->box[1], system->box[2]};
  double timestep = control->timestep;
  SmallVector<void *> arguments;
  positions.addTo(arguments);
  velocities.addTo(arguments);
  masses.addTo(arguments);
  for (auto &field : fields)
    field->addTo(arguments);
  for (double &edge : box)
    arguments.push_back(&edge);
  arguments.push_back(&timestep);

  Output output;
  output.timestep = control->timestep;
  output.degreesOfFreedom = system->getDegreesOfFreedom();
  output.system = &*system;
  if (control->framePeriod > 0) {
    if (llvm::Error error = output.trajectory.open(
            control->dcdFile, count, control->framePeriod, control->timestep,
            control->box))
      return fail(std::move(error));
    output.hasTrajectory = true;
  }
  setOutput(&output);

  std::fprintf(output.log, "MDIR: %zu particles, %lld steps of %g ps\n",
               count, static_cast<long long>(control->numSteps),
               control->timestep);
  std::fprintf(output.log, "MDIR: compiled in %.2f s\n", compileTime);
  writeLogHeader(output);

  begin = std::chrono::steady_clock::now();
  (*function)(arguments.data());
  double runTime = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - begin)
                       .count();
  output.trajectory.close();

  std::fprintf(output.log, "MDIR: ran in %.2f s", runTime);
  if (control->numSteps > 0 && runTime > 0.0) {
    double simulated = control->timestep * control->numSteps * 1.0e-3;
    std::fprintf(output.log, ", %.2f ms per step, %.1f ns per day",
                 1.0e3 * runTime / control->numSteps,
                 simulated * 86400.0 / runTime);
  }
  std::fprintf(output.log, "\n");
  if (output.hasEnergies && output.firstTotal != 0.0)
    std::fprintf(output.log,
                 "MDIR: the total energy changed by %.3e of its value\n",
                 std::fabs((output.lastTotal - output.firstTotal) /
                           output.firstTotal));
  return 0;
}
