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
#include "mdir/Driver/Checkpoint.h"
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
  os << "md-exec-reuse-neighbors,md-exec-expose-validity,"
     << "md-exec-fuse-loops,";
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

  bool writesCheckpoints = control->checkpointPeriod > 0;
  bool isRestart = !control->restartInput.empty();
  if ((writesCheckpoints || isRestart) && !hasCheckpointSupport())
    return fail("this build of MDIR has no HDF5, which checkpoints need");

  StringRef integrator =
      control->integrator == Integrator::Leapfrog ? "LEAP" : "VVER";
  double velocityOffset =
      control->integrator == Integrator::Leapfrog ? -0.5 : 0.0;

  // A run that continues an earlier one takes its state from the
  // checkpoint. The file of the positions gives the types.
  int64_t firstStep = 0;
  double firstTime = 0.0;
  std::vector<double> forces;
  if (isRestart) {
    auto checkpoint = readCheckpoint(control->restartInput);
    if (!checkpoint)
      return fail(checkpoint.takeError());
    const std::string &path = control->restartInput;
    if (checkpoint->getNumParticles() != system->getNumParticles())
      return fail("'" + path + "' holds " +
                  llvm::Twine(checkpoint->getNumParticles()) +
                  " particles, but '" + control->pdbFile + "' holds " +
                  llvm::Twine(system->getNumParticles()));
    for (size_t i = 0, e = system->getNumParticles(); i != e; ++i)
      if (checkpoint->species[i] != static_cast<int32_t>(system->types[i]))
        return fail("particle " + llvm::Twine(i + 1) + " has another type "
                    "in '" + path + "' than in '" + control->pdbFile + "'");
    if (checkpoint->integrator != integrator)
      return fail("'" + path + "' was written with the integrator " +
                  checkpoint->integrator + ", and the run uses " +
                  integrator + "; the velocities of the two are not of the "
                  "same time");
    for (int i = 0; i != 3; ++i)
      if (checkpoint->box[i] != system->box[i])
        return fail("the box of '" + path + "' differs from that of "
                    "[boundary]");
    if (control->integrator == Integrator::VelocityVerlet &&
        checkpoint->forces.empty())
      return fail("'" + path + "' holds no forces, which velocity Verlet "
                  "begins a step with");

    system->positions = checkpoint->positions;
    system->velocities = checkpoint->velocities;
    forces = checkpoint->forces;
    firstStep = checkpoint->step;
    firstTime = checkpoint->time;
  } else {
    assignVelocities(*control, *system);
  }

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

  (*engine)->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    auto add = [&](StringRef name, void *function) {
      symbols[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(function),
                                 llvm::JITSymbolFlags::Exported};
    };
    add("_mlir_ciface_mdrtWriteEnergies",
        (void *)&_mlir_ciface_mdrtWriteEnergies);
    add("_mlir_ciface_mdrtWriteFrame", (void *)&_mlir_ciface_mdrtWriteFrame);
    add("_mlir_ciface_mdrtFinish", (void *)&_mlir_ciface_mdrtFinish);
    add("_mlir_ciface_mdrtWriteCheckpoint",
        program->writesForces
            ? (void *)&_mlir_ciface_mdrtWriteCheckpointWithForces
            : (void *)&_mlir_ciface_mdrtWriteCheckpoint);
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
  if (program->takesForces && forces.size() != 3 * count)
    return fail("the run takes forces, but has none; this is a defect of "
                "mdir-run");
  Buffer<2> given(program->takesForces ? forces
                                       : std::vector<double>(3 * count),
                  program->force, count);
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
  if (program->takesForces)
    given.addTo(arguments);
  masses.addTo(arguments);
  for (auto &field : fields)
    field->addTo(arguments);
  for (double &edge : box)
    arguments.push_back(&edge);
  arguments.push_back(&timestep);
  arguments.push_back(&firstStep);

  Output output;
  output.state = program->state;
  output.force = program->force;
  output.firstStep = firstStep;
  output.firstTime = firstTime;
  output.timestep = control->timestep;
  output.degreesOfFreedom = system->getDegreesOfFreedom();
  output.system = &*system;
  if (control->framePeriod > 0) {
    if (llvm::Error error = output.trajectory.open(
            control->dcdFile, count, firstStep + control->framePeriod,
            control->framePeriod, control->timestep, control->box))
      return fail(std::move(error));
    output.hasTrajectory = true;
  }
  if (writesCheckpoints) {
    output.checkpointPath = control->restartOutput;
    Checkpoint &checkpoint = output.checkpoint;
    checkpoint.masses = system->masses;
    checkpoint.species.assign(system->types.begin(), system->types.end());
    for (int i = 0; i != 3; ++i)
      checkpoint.box[i] = system->box[i];
    checkpoint.integrator = integrator.str();
    checkpoint.velocityOffset = velocityOffset;
    checkpoint.precision =
        control->precision == Precision::Single
            ? "single"
            : control->precision == Precision::Mixed ? "mixed" : "double";
    checkpoint.timestep = control->timestep;
    checkpoint.seed = control->seed;
  }
  setOutput(&output);

  std::fprintf(output.log, "MDIR: %zu particles, %lld steps of %g ps\n",
               count, static_cast<long long>(control->numSteps),
               control->timestep);
  if (isRestart)
    std::fprintf(output.log, "MDIR: continues after step %lld, from '%s'\n",
                 static_cast<long long>(firstStep),
                 control->restartInput.c_str());
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
  if (writesCheckpoints)
    std::fprintf(output.log, "MDIR: wrote %lld checkpoints to '%s'\n",
                 static_cast<long long>(output.numCheckpoints),
                 output.checkpointPath.c_str());

  // The runtime has counted the builds of the neighbor structures.
  if (auto count = (*engine)->lookup("mdrtGetBuildCount")) {
    auto getCount = reinterpret_cast<int64_t (*)()>(*count);
    int64_t builds = getCount();
    std::fprintf(output.log, "MDIR: neighbor structures were built %lld "
                             "times",
                 static_cast<long long>(builds));
    if (builds > 1 && control->numSteps > 0)
      std::fprintf(output.log, ", every %.1f steps on average",
                   static_cast<double>(control->numSteps) /
                       static_cast<double>(builds - 1));
    std::fprintf(output.log, "\n");
  } else {
    llvm::consumeError(count.takeError());
  }
  if (output.hasEnergies && output.firstTotal != 0.0)
    std::fprintf(output.log,
                 "MDIR: the total energy changed by %.3e of its value\n",
                 std::fabs((output.lastTotal - output.firstTotal) /
                           output.firstTotal));
  return 0;
}
