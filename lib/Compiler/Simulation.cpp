// A simulation that persists across runs (D196,
// docs/python-segments.md). It compiles one program of segments, whose
// entry begins the run as `mdir run` does on its first call and continues
// the state that the last call left on the others, as `mdir run --continue`
// continues a checkpoint (D211); the entry takes the
// counts of its loops, so that the program runs any number of steps from any
// step of the period of coupling.
#include "mdir/Compiler/Simulation.h"
#include "mdir/Compiler/Compile.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Driver/Checkpoint.h"
#include "mdir/Driver/Fingerprint.h"
#include "mdir/Driver/Output.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "JITEngine.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <numeric>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>
#include <unordered_set>

using namespace mdir;
using namespace mdir::driver;
using compiler::Simulation;
using compiler::SimulationState;
using llvm::StringRef;

char compiler::SimulationError::ID = 0;

namespace {
/// A buffer of the host in the type that the program takes, as the buffers
/// of `mdir run` are (tools/mdir/Run.cpp).
template <unsigned Rank> struct HostBuffer {
  HostBuffer(const std::vector<double> &values, Element element, size_t count) {
    size_t width = element == Element::F32 ? sizeof(float) : sizeof(double);
    storage.resize(values.size() * width);
    for (size_t i = 0, e = values.size(); i != e; ++i) {
      if (element == Element::F32) {
        float value = static_cast<float>(values[i]);
        std::memcpy(&storage[i * width], &value, width);
      } else {
        std::memcpy(&storage[i * width], &values[i], width);
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
  void addTo(std::vector<void *> &arguments) {
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

template <class T, int Rank> void addDescriptor(std::vector<void *> &arguments,
                                                StridedMemRefType<T, Rank> &d) {
  arguments.push_back(&d.basePtr);
  arguments.push_back(&d.data);
  arguments.push_back(&d.offset);
  for (int i = 0; i != Rank; ++i)
    arguments.push_back(&d.sizes[i]);
  for (int i = 0; i != Rank; ++i)
    arguments.push_back(&d.strides[i]);
}

llvm::Error simulationError(const llvm::Twine &message) {
  return llvm::make_error<compiler::SimulationError>(message.str());
}
llvm::Error inputError(const llvm::Twine &message) {
  return llvm::make_error<model::ModelError>(model::ModelError::Input,
                                             message.str());
}
llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::make_error<model::ModelError>(model::ModelError::Unsupported,
                                             message.str());
}

/// One simulation runs at a time in a process: the output that compiled code
/// reports to and the streams and allocations of the runtime are the
/// process's (docs/python-segments.md, Ownership).
std::mutex &getRunMutex() {
  static std::mutex mutex;
  return mutex;
}
/// The device of the GPU simulations of the process, once one has run.
int64_t usedDevice = -1;

/// The first failure that the runtime reports during a part: a build of the
/// neighbor structures that found positions that are not numbers
/// (mdrtStopNotNumbers). The part goes on to its end and is discarded.
std::string stopMessage;
extern "C" void stopPart(const char *message) {
  if (stopMessage.empty())
    stopMessage = message;
}

/// The memory of the host that compiled code allocates (`malloc` and `free`
/// of its module), by activation (D215): what an activation
/// leaves allocated is its own and is freed when it ends (#110). The code of
/// an activation runs while its set is current. A pointer that compiled code
/// did not allocate goes to `free` as it is.
std::mutex &getAllocationMutex() {
  static std::mutex mutex;
  return mutex;
}
std::unordered_set<void *> *currentAllocations = nullptr;
extern "C" void *allocateForCode(size_t size) {
  void *pointer = std::malloc(size);
  if (pointer) {
    std::lock_guard<std::mutex> lock(getAllocationMutex());
    if (currentAllocations)
      currentAllocations->insert(pointer);
  }
  return pointer;
}
extern "C" void freeForCode(void *pointer) {
  if (!pointer)
    return;
  {
    std::lock_guard<std::mutex> lock(getAllocationMutex());
    if (currentAllocations)
      currentAllocations->erase(pointer);
  }
  std::free(pointer);
}

/// A function of a runtime library, or null if the library is not loaded.
template <typename Function> Function *findRuntime(const char *name) {
  return reinterpret_cast<Function *>(
      llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name));
}

/// The directory of the runtime libraries: `MDIR_RUNTIME_DIR`, or `lib` next
/// to the directory of the module that holds this code, as a build tree and
/// an installed prefix place them, or that of the build.
std::string findRuntimeDirectory() {
  std::vector<std::string> candidates;
  if (const char *named = std::getenv("MDIR_RUNTIME_DIR"); named && *named)
    candidates.push_back(named);
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&findRuntimeDirectory), &info) &&
      info.dli_fname) {
    // build/python/mdir*.so takes build/lib; <prefix>/lib/pythonX.Y/
    // site-packages/mdir*.so takes <prefix>/lib.
    llvm::SmallString<256> here(llvm::sys::path::parent_path(info.dli_fname));
    for (int up = 0; up != 3 && !here.empty(); ++up) {
      llvm::SmallString<256> parent(llvm::sys::path::parent_path(here));
      llvm::SmallString<256> lib(parent);
      llvm::sys::path::append(lib, "lib");
      candidates.push_back(std::string(lib));
      if (llvm::sys::path::filename(parent) == "lib")
        candidates.push_back(std::string(parent));
      here = parent;
    }
  }
  candidates.push_back(MDIR_BUILD_LIBRARY_DIR);
  for (const std::string &directory : candidates) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, "libmdrt.so");
    if (llvm::sys::fs::exists(path))
      return directory;
  }
  return "";
}

/// The option of LLVM that selects the scheduler of the host code; see
/// tools/mdir/Run.cpp (D150).
llvm::cl::Option *getSchedulerOption() {
  auto &options = llvm::cl::getRegisteredOptions();
  auto option = options.find("pre-RA-sched");
  return option == options.end() ? nullptr : option->second;
}
} // namespace

/// A compiled program of segments and what its entry takes besides the state.
struct Simulation::Engine {
  Control control;
  Program program;
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unique_ptr<compiler::JITEngine> engine;
  void (*function)(void **) = nullptr;
  /// What compiling it cost (D212).
  compiler::CompileStats stats;
  /// The volume that the constants of the program are for.
  double volume = 0.0;
  /// The masses of the particles, in the order of the input.
  std::vector<double> masses;
};

namespace {
/// The buffers that a call of an entry takes, made anew for each call: on a
/// device the entry may write what it copies back into the buffers of the
/// host that it was given (AssignStorage), which `mdir run` gives it once.
struct Arguments {
  std::vector<std::unique_ptr<HostBuffer<2>>> vectors;
  std::vector<std::unique_ptr<HostBuffer<1>>> reals;
  std::vector<std::vector<int32_t>> integers;
  std::vector<std::vector<double>> doubles;
  // (Their storage outlives the call; a moved vector keeps it.)
  std::vector<std::unique_ptr<StridedMemRefType<int32_t, 1>>> integerFields;
  std::vector<std::unique_ptr<StridedMemRefType<double, 2>>> tables;
  std::vector<std::unique_ptr<StridedMemRefType<int32_t, 2>>> members;
  std::vector<std::unique_ptr<StridedMemRefType<double, 1>>> tupleFields;
  StridedMemRefType<int32_t, 1> identities;
  std::vector<void *> pointers;
};
} // namespace

/// The activation of the entry of a program of segments that runs the parts
/// of a simulation (D215). The entry runs on a stack of its
/// own; at the end of each part it hands the host its state where it is
/// (mdrtPartBoundary) and waits there, holding its buffers, its order of the
/// particles, and its neighbor structures, until the host gives it the
/// next part. It never returns: the host ends it at a boundary by freeing
/// what it allocated, which its records list, and its stack. The code holds
/// nothing else on its stack.
struct compiler::Activation {
  Simulation::Engine *engine = nullptr;
  bool onDevice = false;
  /// What the entry was given at its start: the buffers of the host, which
  /// the program works in on the CPU, and the scalars.
  Arguments arguments;
  double box[3] = {0.0, 0.0, 0.0};
  double timestep = 0.0, firstSize = 0.0, baroConstant = 0.0,
         baroEnergyConstant = 0.0;
  int64_t start = 0, firstCall = 0;
  /// The records of what the activation allocates: of the runtime of the
  /// host, of that of the device, and of the memory of the host that its
  /// code takes.
  void *hostRecord = nullptr, *deviceRecord = nullptr;
  std::unordered_set<void *> allocations;
  /// Where the activation runs, and where the host waits for it.
  ucontext_t context, host;
  void *stack = nullptr;
  size_t stackSize = 0;
  bool atBoundary = false, returned = false;
  /// The state at the last boundary, where it is, in the order of the
  /// program, and the number of its components that are not numbers.
  StridedMemRefType<char, 2> x{}, v{}, f{};
  StridedMemRefType<int32_t, 1> id{};
  double notNumbers = 0.0;
  /// The step that the next part begins after, and its counts.
  std::array<int64_t, 9> next{};
  /// The particle at each place of the program, once read.
  std::vector<int32_t> places;
};

namespace {
/// The activation whose code runs, and one that is about to begin.
compiler::Activation *runningActivation = nullptr;
compiler::Activation *startingActivation = nullptr;

/// The stack of an activation. The entry ran on the stack of the thread that
/// called it, of 8 MiB on Linux; its pages are committed as they are used.
constexpr size_t activationStack = size_t(64) << 20;

void runActivation() {
  compiler::Activation *a = startingActivation;
  a->engine->function(a->arguments.pointers.data());
  // The loop over parts has no end; the host ends an activation at a
  // boundary.
  a->returned = true;
}

/// The end of a part, or the end of the start of an activation: the state
/// where it is, and the memory where the next part's step and counts go.
extern "C" void _mlir_ciface_mdrtPartBoundary(StridedMemRefType<char, 2> *x,
                                              StridedMemRefType<char, 2> *v,
                                              StridedMemRefType<char, 2> *f,
                                              StridedMemRefType<int32_t, 1> *id,
                                              double notNumbers,
                                              StridedMemRefType<int64_t, 1> *counts) {
  compiler::Activation *a = runningActivation;
  a->x = *x;
  a->v = *v;
  a->f = *f;
  a->id = *id;
  a->notNumbers = notNumbers;
  a->atBoundary = true;
  swapcontext(&a->context, &a->host);
  // The next part.
  for (size_t k = 0; k != a->next.size(); ++k)
    counts->data[counts->offset + k * counts->strides[0]] = a->next[k];
}
} // namespace

Simulation::~Simulation() {
  std::lock_guard<std::mutex> lock(getRunMutex());
  // The files of the reports are complete when the simulation ends.
  if (output) {
    output->energies.close();
    if (output->trajectory)
      output->trajectory->close();
  }
  // The activation runs the code of the engine, and is ended first
  // (D199).
  endActivation();
  freeSnapshot();
  compiled.reset();
}

static llvm::Expected<std::unique_ptr<Simulation::Engine>>
compileEngine(const Control &control, const System &system,
              const model::Execution &execution, bool cache) {
  auto engine = std::make_unique<Simulation::Engine>();
  engine->control = control;
  auto program = buildProgram(control, system);
  if (!program)
    return program.takeError();
  engine->program = std::move(*program);
  engine->volume = system.box[0] * system.box[1] * system.box[2];
  if (control.target == Target::GPU)
    compiler::useCudaToolkit();
  // The context lowers with the threads of the process, so that a
  // simulation keeps no pool of its own (D211).
  engine->context = std::make_unique<mlir::MLIRContext>(
      compiler::getRegistry(), mlir::MLIRContext::Threading::DISABLED);
  compiler::shareThreadPool(*engine->context);
  auto seconds = [start = std::chrono::steady_clock::now()] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start).count();
  };
  auto gpu = compiler::getGpuOptions(control, execution.device, cache);
  if (!gpu)
    return gpu.takeError();
  auto module = compiler::lowerModule(*engine->context, control,
                                      engine->program, &engine->stats, *gpu);
  if (!module)
    return module.takeError();
  engine->module = std::move(*module);
  engine->stats.pipelineSeconds = seconds();

  static std::once_flag native;
  std::call_once(native, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
  std::string directory = findRuntimeDirectory();
  if (directory.empty())
    return unsupported("cannot find the MDIR runtime (libmdrt.so); set "
                       "MDIR_RUNTIME_DIR to the directory that holds it");
  auto library = [&](StringRef dir, StringRef name) {
    llvm::SmallString<256> path(dir);
    llvm::sys::path::append(path, name);
    return std::string(path);
  };
  std::vector<std::string> paths = {library(directory, "libmdrt.so")};
  if (control.target == Target::GPU) {
    paths.push_back(library(directory, "libmdrt_cuda.so"));
  } else if (control.threads > 1) {
    setenv("OMP_NUM_THREADS", std::to_string(control.threads).c_str(), 1);
    std::string omp = library(directory, "libomp.so");
    if (!llvm::sys::fs::exists(omp))
      omp = library(MDIR_LLVM_LIBRARY_DIR, "libomp.so");
    paths.push_back(omp);
  }
  for (const std::string &path : paths)
    if (!llvm::sys::fs::exists(path))
      return unsupported("cannot find '" + path + "'");

  auto targetMachine = compiler::createHostMachine();
  if (!targetMachine)
    return targetMachine.takeError();
  llvm::cl::Option *scheduler = getSchedulerOption();
  if (scheduler)
    (void)scheduler->addOccurrence(0, "pre-RA-sched", "fast");
  struct ResetScheduler {
    llvm::cl::Option *option;
    ~ResetScheduler() {
      if (option)
        (void)option->addOccurrence(0, "pre-RA-sched", "default");
    }
  } resetScheduler{scheduler};
  double jitStart = seconds();
  auto created = compiler::JITEngine::create(
      *engine->module, std::move(*targetMachine), paths, engine->program.entry,
      scheduler ? "pre-RA-sched=fast" : "", cache);
  if (!created) {
    return llvm::make_error<compiler::CompileError>(
        "cannot compile the program for execution: " +
        llvm::toString(created.takeError()));
  }
  engine->engine = std::move(*created);
  bool writesForces = engine->program.writesForces;
  auto symbolError = engine->engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    auto add = [&](StringRef name, void *function) {
      symbols[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(function),
                                 llvm::JITSymbolFlags::Exported};
    };
    add("_mlir_ciface_mdrtWriteEnergies", (void *)&_mlir_ciface_mdrtWriteEnergies);
    add("_mlir_ciface_mdrtWriteFrame", (void *)&_mlir_ciface_mdrtWriteFrame);
    add("_mlir_ciface_mdrtCheckSpread", (void *)&_mlir_ciface_mdrtCheckSpread);
    add("_mlir_ciface_mdrtWriteTerms", (void *)&_mlir_ciface_mdrtWriteTerms);
    add("_mlir_ciface_mdrtWriteVirial", (void *)&_mlir_ciface_mdrtWriteVirial);
    add("_mlir_ciface_mdrtAddBath", (void *)&_mlir_ciface_mdrtAddBath);
    add("mdrtNoseHooverFactor", (void *)&mdrtNoseHooverFactor);
    add("mdrtWriteSolvent", (void *)&mdrtWriteSolvent);
    add("_mlir_ciface_mdrtWritePull", (void *)&_mlir_ciface_mdrtWritePull);
    add("_mlir_ciface_mdrtWriteFreeEnergy",
        (void *)&_mlir_ciface_mdrtWriteFreeEnergy);
    add("_mlir_ciface_mdrtWriteObservables",
        (void *)&_mlir_ciface_mdrtWriteObservables);
    add("_mlir_ciface_mdrtSetBox", (void *)&_mlir_ciface_mdrtSetBox);
    add("_mlir_ciface_mdrtSetTilt", (void *)&_mlir_ciface_mdrtSetTilt);
    add("_mlir_ciface_mdrtSetBarostatState",
        (void *)&_mlir_ciface_mdrtSetBarostatState);
    add("_mlir_ciface_mdrtWriteMinimization",
        (void *)&_mlir_ciface_mdrtWriteMinimization);
    add("_mlir_ciface_mdrtFinish", (void *)&_mlir_ciface_mdrtFinish);
    // The memory of the host that the code allocates, freed after each call.
    add("malloc", (void *)&allocateForCode);
    add("free", (void *)&freeForCode);
    add("_mlir_ciface_mdrtPartBoundary",
        (void *)&_mlir_ciface_mdrtPartBoundary);
    add("_mlir_ciface_mdrtWriteCheckpoint",
        writesForces ? (void *)&_mlir_ciface_mdrtWriteCheckpointWithForces
                     : (void *)&_mlir_ciface_mdrtWriteCheckpoint);
    return symbols;
  });

  if (symbolError)
    return std::move(symbolError);

  // The runtime takes the failures that it cannot return from to the
  // simulation, and a GPU simulation the device that the process uses.
  if (auto set = reinterpret_cast<void (*)(void (*)(const char *))>(
          llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
              "mdrtSetStopHandler")))
    set(&stopPart);
  if (control.target == Target::GPU) {
    if (usedDevice >= 0 && usedDevice != execution.device) {
      return unsupported("this process runs its GPU simulations on device " +
                         llvm::Twine(usedDevice) + "; a simulation on device " +
                         llvm::Twine(execution.device) +
                         " needs a process of its own");
    }
    if (usedDevice < 0) {
      auto select = reinterpret_cast<void (*)(int32_t)>(
          llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
              "mgpuSetDefaultDevice"));
      if (!select) {
        return unsupported("the GPU runtime does not select devices");
      }
      select(static_cast<int32_t>(execution.device));
      usedDevice = execution.device;
    }
  }
  // Loads the kernels of a GPU; the functions of the driver are known.
  if (auto error = engine->engine->initialize())
    return llvm::make_error<compiler::CompileError>(llvm::toString(std::move(error)));
  auto function = engine->engine->lookupPacked(engine->program.entry);
  if (!function)
    return llvm::make_error<compiler::CompileError>(
        llvm::toString(function.takeError()));
  engine->function = *function;
  engine->stats += engine->engine->getCompileStats();
  engine->stats.programs = 1;
  engine->stats.bypassed = cache ? 0 : 1;
  engine->stats.engineSeconds = seconds() - jitStart;

  engine->masses = system.masses;
  return std::move(engine);
}

llvm::Expected<std::unique_ptr<Simulation>>
Simulation::create(const model::PreparedModel &prepared, bool cache) {
  std::unique_lock<std::mutex> lock(getRunMutex());
  const Control &given = prepared.control;
  bool trotter = given.barostat &&
                 (given.barostatWork == BarostatWork::Trotter ||
                  given.barostatWork == BarostatWork::TrotterFirstOrder);
  if (trotter && given.barostatPeriod == 1)
    return unsupported("a simulation with a barostat that scales the cell "
                       "every step (coupling period 1) is not supported yet");
  const System &start = prepared.system;
  if (given.barostat &&
      (start.tilt[0] != 0.0 || start.tilt[1] != 0.0 || start.tilt[2] != 0.0))
    return unsupported("a simulation with a barostat in a triclinic cell is "
                       "not supported yet");
  std::unique_ptr<Simulation> simulation(new Simulation());
  struct UnlockBeforeCleanup {
    std::unique_lock<std::mutex> &lock;
    ~UnlockBeforeCleanup() { if (lock.owns_lock()) lock.unlock(); }
  } unlockBeforeCleanup{lock};
  simulation->prepared = prepared;
  Control &control = simulation->prepared.control;
  // A minimization takes the steps of its schedule unless told otherwise,
  // and begins with the step of the control (D202).
  simulation->minimizationSteps = control.numSteps;
  simulation->minimizationSize = control.minimizeStep * units::length;
  // A tolerance is checked every energy period of the program, at the
  // steps where `mdir run` writes its rows (D[minimize-tolerance]).
  simulation->minimizationCheckPeriod = control.energyPeriod;
  control.segments = true;
  control.energyPeriod = 0;
  control.framePeriod = 0;
  control.checkpointPeriod = 0;
  control.numSteps = std::max<int64_t>(1, control.getCouplingPeriod());
  System &system = simulation->system;
  system = start;
  size_t count = system.getNumParticles();
  // A state without velocities begins at rest.
  if (system.velocities.size() != 3 * count)
    system.velocities.assign(3 * count, 0.0);
  // The reference of restraints is the model's (D198),
  // or the positions it begins at; its cell is that of the start.
  if (system.referencePositions.size() != system.positions.size())
    system.referencePositions = system.positions;
  for (int k = 0; k != 3; ++k)
    system.inputBox[k] = system.box[k];

  // The values of the tunables are put into the system that the program is
  // built from to build them anew (D213).
  simulation->compiledSystem = system;
  simulation->tunableValues = prepared.tunables.values;
  simulation->tunablesHistory = {{0, 0}};
  auto engine = compileEngine(control, system, prepared.execution, cache);
  if (!engine) {
    lock.unlock();
    return engine.takeError();
  }
  simulation->compiled = std::move(*engine);

  auto output = std::make_unique<Output>();
  output->log.quiet = true;
  output->embedded = true;
  output->timestep = control.timestep;
  output->couples = control.getCouplingPeriod() > 0 && !control.isLangevin();
  output->changesCell = control.barostat;
  output->bathKinetic = 0.5 * system.getDegreesOfFreedom() *
                        units::boltzmann * control.temperature;
  output->leastEdge = 2.0 * control.cutoffDistance * units::length;
  output->degreesOfFreedom = system.getDegreesOfFreedom();
  output->solventFreedom = system.getSolventDegreesOfFreedom();
  output->periodic = control.periodic;
  output->listReach = control.pairlistDistance * units::length;
  for (int k = 0; k != 3; ++k)
    output->box[k] = system.box[k];
  output->volume = system.box[0] * system.box[1] * system.box[2];
  output->endStep = 0;
  output->tunablesVersion = prepared.tunables.empty() ? -1 : 0;
  simulation->output = std::move(output);
  lock.unlock();
  return std::move(simulation);
}

compiler::CompileStats Simulation::getCompileStats() const {
  CompileStats stats;
  if (compiled)
    stats += compiled->stats;
  return stats;
}

llvm::Expected<Simulation::Engine *> Simulation::getEngine() {
  return compiled.get();
}

llvm::Error Simulation::startActivation(int64_t firstCall) {
  Engine &engine = *compiled;
  const Program &p = engine.program;
  auto a = std::make_unique<Activation>();
  a->engine = &engine;
  a->onDevice = engine.control.target == Target::GPU;
  size_t count = system.getNumParticles();
  // The arguments in the order of the entry (Builder.h), from the state of
  // the host. The first call begins the run and ignores the forces that it
  // is given (D211); a later one continues those given.
  Arguments &args = a->arguments;
  auto vector = [&](const std::vector<double> &values, Element element) {
    args.vectors.push_back(
        std::make_unique<HostBuffer<2>>(values, element, count));
    args.vectors.back()->addTo(args.pointers);
  };
  vector(system.positions, p.state);
  vector(system.velocities, p.state);
  if (p.takesForces)
    vector(hasRun ? forces : std::vector<double>(3 * count, 0.0), p.force);
  args.reals.push_back(
      std::make_unique<HostBuffer<1>>(engine.masses, p.mass, count));
  args.reals.back()->addTo(args.pointers);
  for (const Program::Field &field : p.fields) {
    if (!field.isInteger) {
      args.reals.push_back(
          std::make_unique<HostBuffer<1>>(field.values, p.parameter, count));
      args.reals.back()->addTo(args.pointers);
      continue;
    }
    args.integers.emplace_back(field.values.begin(), field.values.end());
    auto d = std::make_unique<StridedMemRefType<int32_t, 1>>();
    d->basePtr = d->data = args.integers.back().data();
    d->offset = 0;
    d->sizes[0] = count;
    d->strides[0] = 1;
    addDescriptor(args.pointers, *d);
    args.integerFields.push_back(std::move(d));
  }
  for (const Program::Table &table : p.tables) {
    args.doubles.push_back(table.values);
    auto d = std::make_unique<StridedMemRefType<double, 2>>();
    d->basePtr = d->data = args.doubles.back().data();
    d->offset = 0;
    d->sizes[0] = table.count;
    d->sizes[1] = table.getColumns();
    d->strides[0] = table.getColumns();
    d->strides[1] = 1;
    addDescriptor(args.pointers, *d);
    args.tables.push_back(std::move(d));
  }
  // Moving a vector keeps its storage, so the descriptors stay valid as
  // the lists grow.
  for (const Program::TupleSet &set : p.tupleSets) {
    args.integers.push_back(set.members);
    auto m = std::make_unique<StridedMemRefType<int32_t, 2>>();
    m->basePtr = m->data = args.integers.back().data();
    m->offset = 0;
    m->sizes[0] = set.size();
    m->sizes[1] = set.arity;
    m->strides[0] = set.arity;
    m->strides[1] = 1;
    addDescriptor(args.pointers, *m);
    args.members.push_back(std::move(m));
    for (const Program::Field &field : set.fields) {
      args.doubles.push_back(field.values);
      auto v = std::make_unique<StridedMemRefType<double, 1>>();
      v->basePtr = v->data = args.doubles.back().data();
      v->offset = 0;
      v->sizes[0] = set.size();
      v->strides[0] = 1;
      addDescriptor(args.pointers, *v);
      args.tupleFields.push_back(std::move(v));
    }
  }
  args.integers.emplace_back(count);
  std::vector<int32_t> &numbers = args.integers.back();
  for (size_t i = 0; i != count; ++i)
    numbers[i] = static_cast<int32_t>(i);
  args.identities.basePtr = args.identities.data = numbers.data();
  args.identities.offset = 0;
  args.identities.sizes[0] = count;
  args.identities.strides[0] = 1;
  addDescriptor(args.pointers, args.identities);
  for (int k = 0; k != 3; ++k)
    a->box[k] = output->box[k];
  a->timestep = engine.control.timestep;
  a->start = step;
  a->firstCall = firstCall;
  a->firstSize = minimizationSize;
  a->baroConstant = p.baroConstant;
  a->baroEnergyConstant = p.baroEnergyConstant;
  for (double &edge : a->box)
    args.pointers.push_back(&edge);
  args.pointers.push_back(&a->timestep);
  args.pointers.push_back(&a->start);
  if (engine.control.minimize)
    args.pointers.push_back(&a->firstSize);
  args.pointers.push_back(&a->firstCall);
  if (p.takesConstants) {
    args.pointers.push_back(&a->baroConstant);
    args.pointers.push_back(&a->baroEnergyConstant);
  }

  // A stack of its own, below which a page that is not mapped stops an
  // overflow.
  void *stack = mmap(nullptr, activationStack, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK,
                     -1, 0);
  if (stack == MAP_FAILED)
    return simulationError("cannot map the stack of the program: " +
                           llvm::Twine(std::strerror(errno)));
  size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  mprotect(stack, page, PROT_NONE);
  a->stack = stack;
  a->stackSize = activationStack;
  if (auto open = findRuntime<void *()>("mdrtActivationOpen"))
    a->hostRecord = open();
  if (a->onDevice)
    if (auto open = findRuntime<void *()>("mdrtDeviceActivationOpen"))
      a->deviceRecord = open();
  getcontext(&a->context);
  a->context.uc_stack.ss_sp = stack;
  a->context.uc_stack.ss_size = activationStack;
  a->context.uc_link = &a->host;
  makecontext(&a->context, runActivation, 0);
  startingActivation = a.get();
  activation = std::move(a);
  resumeActivation();
  startingActivation = nullptr;
  return llvm::Error::success();
}

void Simulation::resumeActivation() {
  Activation &a = *activation;
  waitForConsumers();
  ++generation;
  a.atBoundary = false;
  if (auto enter = findRuntime<void(void *)>("mdrtActivationEnter"))
    enter(a.hostRecord);
  if (a.onDevice)
    if (auto enter = findRuntime<void(void *)>("mdrtDeviceActivationEnter"))
      enter(a.deviceRecord);
  {
    std::lock_guard<std::mutex> lock(getAllocationMutex());
    currentAllocations = &a.allocations;
  }
  runningActivation = &a;
  swapcontext(&a.host, &a.context);
  runningActivation = nullptr;
  {
    std::lock_guard<std::mutex> lock(getAllocationMutex());
    currentAllocations = nullptr;
  }
  if (auto leave = findRuntime<void()>("mdrtActivationLeave"))
    leave();
  if (a.onDevice)
    if (auto leave = findRuntime<void()>("mdrtDeviceActivationLeave"))
      leave();
}

void Simulation::endActivation() {
  if (!activation)
    return;
  Activation &a = *activation;
  waitForConsumers();
  ++generation;
  // Its code waits at a boundary and never runs again. What it allocated
  // is freed: its blocks of the device return to the pool of the runtime
  // (#110).
  if (a.deviceRecord)
    if (auto close = findRuntime<void(void *)>("mdrtDeviceActivationClose"))
      close(a.deviceRecord);
  if (a.hostRecord)
    if (auto close = findRuntime<void(void *)>("mdrtActivationClose"))
      close(a.hostRecord);
  {
    std::lock_guard<std::mutex> lock(getAllocationMutex());
    for (void *pointer : a.allocations)
      std::free(pointer);
    a.allocations.clear();
  }
  if (a.stack)
    munmap(a.stack, a.stackSize);
  activation.reset();
}

namespace {
size_t getWidth(Element element) {
  return element == Element::F32 ? sizeof(float) : sizeof(double);
}
/// The address of the first element of a buffer that the program gave.
template <typename T, int Rank>
const char *getStart(const StridedMemRefType<T, Rank> &buffer, size_t width) {
  return reinterpret_cast<const char *>(buffer.data) + buffer.offset * width;
}

/// The state of the program, in its order and in the types it is stored
/// in, as the state of the host: in the order of the input and in double
/// precision.
void readState(compiler::Activation &a, const Program &p, const void *x,
               const void *v, const void *f, bool onDevice,
               std::vector<double> &positions,
               std::vector<double> &velocities,
               std::vector<double> &forces) {
  size_t count = static_cast<size_t>(a.x.sizes[0]);
  auto copy = [&](void *to, const void *from, size_t bytes) {
    if (!onDevice) {
      std::memcpy(to, from, bytes);
      return;
    }
    static auto toHost = findRuntime<void(void *, const void *, uint64_t)>(
        "mdrtDeviceCopyToHost");
    toHost(to, from, bytes);
  };
  if (a.places.size() != count) {
    a.places.resize(count);
    copy(a.places.data(), getStart(a.id, sizeof(int32_t)),
         count * sizeof(int32_t));
  }
  auto read = [&](const void *source, Element element,
                  std::vector<double> &values) {
    size_t width = getWidth(element);
    std::vector<char> raw(3 * count * width);
    copy(raw.data(), source, raw.size());
    values.assign(3 * count, 0.0);
    for (size_t i = 0; i != count; ++i) {
      size_t place = static_cast<size_t>(a.places[i]);
      for (size_t c = 0; c != 3; ++c) {
        double value;
        if (element == Element::F32) {
          float narrow;
          std::memcpy(&narrow, &raw[(3 * i + c) * width], sizeof narrow);
          value = narrow;
        } else {
          std::memcpy(&value, &raw[(3 * i + c) * width], sizeof value);
        }
        values[3 * place + c] = value;
      }
    }
  };
  read(x, p.state, positions);
  read(v, p.state, velocities);
  read(f, p.force, forces);
}
} // namespace

void Simulation::downloadState() const {
  if (hostCurrent || !activation)
    return;
  Activation &a = *activation;
  const Program &p = compiled->program;
  readState(a, p, getStart(a.x, getWidth(p.state)),
            getStart(a.v, getWidth(p.state)), getStart(a.f, getWidth(p.force)),
            a.onDevice, system.positions, system.velocities, forces);
  hostCurrent = true;
}

void Simulation::takeSnapshot() {
  Activation &a = *activation;
  const Program &p = compiled->program;
  size_t count = system.getNumParticles();
  size_t stateBytes = 3 * count * getWidth(p.state);
  size_t forceBytes = 3 * count * getWidth(p.force);
  if (!snapshot.positions) {
    snapshot.onDevice = a.onDevice;
    snapshot.stateBytes = stateBytes;
    snapshot.forceBytes = forceBytes;
    if (a.onDevice) {
      static auto keep =
          findRuntime<void *(uint64_t)>("mdrtDeviceAllocateKept");
      snapshot.positions = keep(stateBytes);
      snapshot.velocities = keep(stateBytes);
      snapshot.forces = keep(forceBytes);
    } else {
      snapshot.host.resize(2 * stateBytes + forceBytes);
      snapshot.positions = snapshot.host.data();
      snapshot.velocities = snapshot.host.data() + stateBytes;
      snapshot.forces = snapshot.host.data() + 2 * stateBytes;
    }
  }
  const void *sources[3] = {getStart(a.x, getWidth(p.state)),
                            getStart(a.v, getWidth(p.state)),
                            getStart(a.f, getWidth(p.force))};
  void *targets[3] = {snapshot.positions, snapshot.velocities,
                      snapshot.forces};
  size_t bytes[3] = {stateBytes, stateBytes, forceBytes};
  for (int k = 0; k != 3; ++k) {
    if (a.onDevice) {
      // On the stream of the kernels, before the work of the next part;
      // the host does not wait.
      static auto within = findRuntime<void(void *, const void *, uint64_t)>(
          "mdrtDeviceCopyWithin");
      within(targets[k], sources[k], bytes[k]);
    } else {
      std::memcpy(targets[k], sources[k], bytes[k]);
    }
  }
  snapshot.valid = true;
}

void Simulation::restoreSnapshot() {
  readState(*activation, compiled->program, snapshot.positions,
            snapshot.velocities, snapshot.forces, snapshot.onDevice,
            system.positions, system.velocities, forces);
  hostCurrent = true;
}

void Simulation::freeSnapshot() {
  if (snapshot.onDevice && snapshot.positions) {
    if (auto release = findRuntime<void(void *)>("mdrtDeviceFreeKept"))
      for (void *pointer :
           {snapshot.positions, snapshot.velocities, snapshot.forces})
        release(pointer);
  }
  snapshot = Snapshot();
}

llvm::Error Simulation::runPart(Engine &engine, Part part) {
  std::lock_guard<std::mutex> lock(getRunMutex());
  Output &out = *output;
  const Program &p = engine.program;
  out.state = p.state;
  out.force = p.force;
  out.firstVolume = engine.volume;
  out.dispersionEnergy = p.dispersionEnergy;
  out.dispersionVirial = p.dispersionVirial;
  out.observableVolumeConstants = p.observableVolumeConstants;
  out.pme = p.pme;
  out.reactionField = p.reactionField;
  out.coulombConstantEnergy = p.coulombConstantEnergy;
  out.coulombConstantVirial = p.coulombConstantVirial;
  out.coulombSelfEnergy = p.coulombSelfEnergy;
  out.ljpme = p.ljpme;
  out.ljpmeSelfEnergy = p.ljpmeSelfEnergy;
  int64_t closing = p.segmentPeriod ? p.closingSteps : 0;
  // An interval of the second nest: periods of coupling, or plain steps
  // and the step of energy.
  int64_t interval = p.segmentPeriod
                         ? (part.closePeriods + 1) * (part.closeInner + closing)
                         : part.closeInner + 1;
  out.endStep = step + part.outer * (p.segmentPeriod ? part.inner + closing : 1) +
                part.tail + part.plain + part.close * interval;

  // What a failed part leaves besides the state (D196).
  double boxBefore[3] = {out.box[0], out.box[1], out.box[2]};
  double bathBefore = out.bath;
  Output::MinimizationRow rowBefore = out.lastMinimization;

  std::string failure;
  auto fail = [&](const std::string &message) {
    if (failure.empty())
      failure = message;
  };
  out.fail = fail;
  out.system = &system;
  setOutput(&out);
  out.quietStep = refreshing ? step : -1;
  out.tunablesVersion = prepared.tunables.empty() ? -1 : tunablesVersion;
  stopMessage.clear();

  // A part continues the activation that the last one left, with its
  // order of the particles and its neighbor structures; the first part,
  // and the first after the values of the program have changed, begin one
  // from the state of the host (D215). A part that begins
  // an activation and fails leaves the state of the host as it was; a later
  // one, the state at the end of the last part (the snapshot).
  bool begins = !activation;
  if (begins) {
    size_t count = system.getNumParticles();
    if (p.takesForces && hasRun && forces.size() != 3 * count) {
      out.fail = nullptr;
      return simulationError("the segment takes forces, but has none; this "
                             "is a defect of mdir");
    }
    // 2: the forces of the state given anew, after an update of the
    // tunables (D213), or at the start of a stage of other physics
    // (D[python-checkpoints]).
    int64_t firstCall = hasRun ? (refreshing || startRefresh ? 2 : 0) : 1;
    if (startRefresh)
      out.quietStep = step;
    if (llvm::Error error = startActivation(firstCall)) {
      out.fail = nullptr;
      return error;
    }
  }
  Activation &a = *activation;
  if (out.endStep > step && a.atBoundary) {
    a.next = {step,       part.outer, part.inner,      part.tail,
              part.plain, part.close, part.closeInner, part.closePeriods,
              reports.framePeriod};
    resumeActivation();
  }
  out.fail = nullptr;
  if (failure.empty())
    failure = stopMessage;
  size_t count = system.getNumParticles();
  if (failure.empty() &&
      (a.returned || !a.atBoundary ||
       static_cast<size_t>(a.x.sizes[0]) != count || a.x.strides[0] != 3 ||
       a.x.strides[1] != 1 || a.v.strides[0] != 3 || a.f.strides[0] != 3 ||
       a.id.strides[0] != 1))
    failure = "the program gave no state at the end of the part; this is a "
              "defect of mdir";
  // On the CPU the structures do not test the positions (D107 does on a
  // device): a state that is not numbers ends the part here.
  if (failure.empty() && a.notNumbers > 0.0)
    failure = "the state at its end is not numbers (a time step too long, "
              "a bad contact, or a defect of mdir)";
  // Without a periodic cell, whether the particles have spread too far at
  // the end of the part (D142).
  std::vector<double> x, v, f;
  if (failure.empty() && !engine.control.periodic) {
    readState(a, p, getStart(a.x, getWidth(p.state)),
              getStart(a.v, getWidth(p.state)),
              getStart(a.f, getWidth(p.force)), a.onDevice, x, v, f);
    out.fail = fail;
    checkParticleSpread(out, x, out.endStep);
    out.fail = nullptr;
  }
  if (!failure.empty()) {
    if (!begins)
      restoreSnapshot();
    endActivation();
    for (int k = 0; k != 3; ++k)
      out.box[k] = boxBefore[k];
    out.volume = out.box[0] * out.box[1] * out.box[2];
    out.bath = bathBefore;
    out.lastMinimization = rowBefore;
    failed = true;
    return simulationError("the simulation failed after step " +
                           llvm::Twine(step) + ": " + failure +
                           "; it keeps the state of step " + llvm::Twine(step));
  }
  takeSnapshot();
  startRefresh = false;
  if (!x.empty()) {
    system.positions = std::move(x);
    system.velocities = std::move(v);
    forces = std::move(f);
    hostCurrent = true;
  } else {
    hostCurrent = false;
  }
  step = out.endStep;
  if (engine.control.minimize)
    minimizationSize = out.lastMinimization.stepSize;
  hasRun = true;
  return llvm::Error::success();
}

llvm::Expected<int64_t> Simulation::run(int64_t count,
                                        const std::function<bool()> &poll,
                                        bool energy) {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  struct Release {
    std::atomic<bool> &flag;
    ~Release() { flag = false; }
  } release{busy};
  if (llvm::Error error = checkLeases("run"))
    return std::move(error);
  if (failed)
    return simulationError("the simulation failed earlier; it keeps the "
                           "state of step " + llvm::Twine(step) +
                           " and runs no further");
  if (prepared.control.minimize)
    return inputError("this simulation minimizes: its program's integrator "
                      "minimizes; call minimize(steps)");
  if (count < 0)
    return inputError("run takes a nonnegative number of steps");
  stopRequested = false;
  if (count == 0)
    return 0;
  // The files of the checkpoint continue in the reporters of the first run
  // only; a reporter added later begins its file (D[python-checkpoints]).
  keepThrough.reset();
  continuedTrajectory.clear();
  continuedFrames = 0;
  int64_t period = compiled->program.segmentPeriod;
  int64_t closing = compiled->program.closingSteps;
  // The two steps that close a period of the barostat of Trotter type are
  // one: the first gives the strain that the second applies (D92).
  if (period > 0 && closing == 2 && (step + count) % period == period - 1)
    return inputError(
        "the run would end at step " + llvm::Twine(step + count) +
        ", between the two steps that close a period of coupling of the "
        "barostat of Trotter type (every " + llvm::Twine(period) +
        " steps); take one step more or fewer");

  int64_t unit = std::max<int64_t>(1, period);
  // A part of `remaining` steps at most, as D196 cuts a run, whose last
  // step is a step of energy if `energyAtEnd` and it reaches the end.
  auto planPart = [&](int64_t remaining, bool energyAtEnd) {
    Part part;
    part.inner = period > 0 ? period - closing : 0;
    int64_t partSteps = getPartSteps(unit);
    if (period == 0) {
      part.outer = std::min(remaining, partSteps);
    } else {
      int64_t phase = step % period;
      if (phase != 0) {
        // The rest of the period under way: its plain steps, and the steps
        // that close it if the run reaches them.
        int64_t plain = period - closing - phase;
        if (remaining < plain + closing) {
          part.tail = remaining;
        } else {
          part.outer = 1;
          part.inner = plain;
        }
      } else {
        int64_t periods = remaining / period;
        int64_t most = std::max<int64_t>(1, partSteps / period);
        part.outer = std::min(periods, most);
        if (part.outer == periods)
          part.tail = remaining % period;
      }
    }
    int64_t planned = part.outer * (period > 0 ? part.inner + closing : 1) +
                      part.tail;
    // The last step of the run is a step of energy, as `mdir run` takes at
    // a row of its log: a plain one, or one that closes a period.
    if (energyAtEnd && planned == remaining) {
      if (period == 0) {
        --part.outer;
        part.plain = 1;
      } else if (part.tail > 0) {
        --part.tail;
        part.plain = 1;
      } else {
        --part.outer;
        part.close = 1;
        part.closeInner = part.inner;
      }
    }
    return part;
  };
  // The interval of the reports inside a part (D207): the
  // greatest common divisor of their periods, if it holds whole periods of
  // coupling; otherwise each report ends a part.
  int64_t interval = 0;
  for (int64_t p : {reports.energyPeriod, reports.framePeriod})
    if (p > 0)
      interval = interval ? std::gcd(interval, p) : p;
  if (interval % unit != 0)
    interval = 0;
  int64_t end = step + count;
  int64_t taken = 0;
  while (taken < count) {
    if (taken > 0 && (stopRequested || (poll && poll())))
      break;
    int64_t remaining = count - taken;
    Part part;
    if (interval > 0 && step % interval == 0 && remaining >= interval) {
      // Intervals of the reports, each ending with a step of energy, inside
      // one call.
      int64_t most = std::max<int64_t>(1, getPartSteps(interval) / interval);
      part.close = std::min(remaining / interval, most);
      if (period > 0) {
        part.closePeriods = interval / period - 1;
        part.closeInner = period - closing;
      } else {
        part.closeInner = interval - 1;
      }
    } else {
      // Up to the next report, which a step of energy takes, or the end.
      int64_t next = getNextReport(step);
      if (next > 0 && next < end)
        part = planPart(next - step, true);
      else
        part = planPart(remaining, energy || next == end);
    }
    auto engine = getEngine();
    if (!engine)
      return engine.takeError();
    auto began = std::chrono::steady_clock::now();
    int64_t before = step;
    if (llvm::Error error = runPart(**engine, part))
      return std::move(error);
    int64_t steps = step - before;
    taken += steps;
    double seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - began)
                         .count();
    if (steps >= 8 && seconds > 0.0)
      secondsPerStep = seconds / steps;
  }
  return taken;
}

// The steps of a part: about `partSeconds` long, in whole multiples of
// `unit` (periods of coupling), from the time that steps took; 100 steps
// until that is known.
int64_t Simulation::getPartSteps(int64_t unit) const {
  double steps = secondsPerStep > 0.0 ? partSeconds / secondsPerStep : 100.0;
  if (!(steps < 1e15))
    steps = 1e15;
  int64_t whole = static_cast<int64_t>(steps) / unit * unit;
  return std::max(unit, whole);
}

llvm::Expected<int64_t> Simulation::minimize(std::optional<int64_t> count,
                                             const std::function<bool()> &poll,
                                             std::optional<double> tolerance) {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  struct Release {
    std::atomic<bool> &flag;
    ~Release() { flag = false; }
  } release{busy};
  if (llvm::Error error = checkLeases("minimize"))
    return std::move(error);
  if (failed)
    return simulationError("the simulation failed earlier; it keeps the "
                           "state of step " + llvm::Twine(step) +
                           " and runs no further");
  if (!prepared.control.minimize)
    return inputError("this simulation runs dynamics: minimize() takes a "
                      "program whose integrator minimizes");
  int64_t total = count ? *count : minimizationSteps;
  if (total < 0)
    return inputError("minimize takes a nonnegative number of steps");
  int64_t period = minimizationCheckPeriod;
  if (tolerance && !(std::isfinite(*tolerance) && *tolerance > 0.0))
    return inputError("minimize takes a positive, finite tolerance");
  if (tolerance && period <= 0)
    return inputError("a tolerance is checked every energy period of the "
                      "program, whose Schedule.energy_period is 0");
  stopRequested = false;
  // With a tolerance (D[minimize-tolerance]), the row at the
  // end of each part is compared with it, as the host of `mdir run` compares
  // each row of its log, and the parts end at the steps of those rows.
  // An activation begins with a row at its first step, which a part of no
  // step writes.
  const Output::MinimizationRow &row = output->lastMinimization;
  auto below = [&]() {
    return row.step == step && row.maxForce < *tolerance;
  };
  if (tolerance) {
    minimizationConverged = false;
    minimizationCheckedStep = step;
    if (!activation || row.step != step) {
      auto engine = getEngine();
      if (!engine)
        return engine.takeError();
      if (llvm::Error error = runPart(**engine, Part()))
        return std::move(error);
    }
    if (below()) {
      minimizationConverged = true;
      return 0;
    }
  } else {
    minimizationConverged.reset();
  }
  // Parts as those of run: the program of the parts after the first takes
  // the positions and the length of the step that the last left.
  int64_t taken = 0;
  while (taken < total) {
    if (taken > 0 && (stopRequested || (poll && poll())))
      break;
    Part part;
    part.outer = std::min(total - taken, getPartSteps(1));
    if (tolerance)
      part.outer = std::min(part.outer, (step / period + 1) * period - step);
    auto engine = getEngine();
    if (!engine)
      return engine.takeError();
    auto began = std::chrono::steady_clock::now();
    int64_t before = step;
    if (llvm::Error error = runPart(**engine, part))
      return std::move(error);
    int64_t steps = step - before;
    taken += steps;
    double seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - began)
                         .count();
    if (steps >= 8 && seconds > 0.0)
      secondsPerStep = seconds / steps;
    if (tolerance) {
      minimizationCheckedStep = step;
      if (below()) {
        minimizationConverged = true;
        break;
      }
    }
  }
  return taken;
}

int64_t Simulation::getNextReport(int64_t from) const {
  int64_t next = -1;
  for (int64_t p : {reports.energyPeriod, reports.framePeriod})
    if (p > 0) {
      int64_t due = (from / p + 1) * p;
      next = next < 0 ? due : std::min(next, due);
    }
  return next;
}

llvm::Error Simulation::setReports(const Reports &given) {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  struct Release {
    std::atomic<bool> &flag;
    ~Release() { flag = false; }
  } release{busy};
  if (prepared.control.minimize && (given.energyPeriod || given.framePeriod))
    return inputError("a minimization takes no reporters");
  if (given.energyPeriod < 0 || given.framePeriod < 0)
    return inputError("a reporter's period must be positive");
  if ((given.energyPeriod > 0) == given.energyPath.empty() ||
      (given.framePeriod > 0) == given.trajectoryPath.empty())
    return inputError("a reporter needs both a file and a period");
  // The steps of energy of the reports must not fall between the two steps
  // that close a period of the barostat of Trotter type (D92).
  int64_t period = compiled->program.segmentPeriod;
  if (period > 0 && compiled->program.closingSteps == 2)
    for (int64_t p : {given.energyPeriod, given.framePeriod})
      if (p > 0 && p % period != 0)
        return inputError("with the barostat of Trotter type a reporter's "
                          "period must be a multiple of the coupling period, " +
                          llvm::Twine(period));
  std::lock_guard<std::mutex> lock(getRunMutex());
  Output &out = *output;
  // The files of a run that a checkpoint continues: the energy file keeps
  // its rows up to the checkpoint and the trajectory that it counts the
  // frames of is cut to them, and both are appended to, as `mdir run
  // --continue` continues them (D129, D130, D149).
  std::string energyFile = getReportPath(given.energyPath);
  std::string trajectoryFile = getReportPath(given.trajectoryPath);
  bool appendsEnergies = false, appendsFrames = false;
  if (keepThrough && given.energyPath != reports.energyPath &&
      !given.energyPath.empty())
    appendsEnergies = llvm::sys::fs::exists(energyFile);
  if (keepThrough && given.trajectoryPath != reports.trajectoryPath &&
      !given.trajectoryPath.empty() && !continuedTrajectory.empty()) {
    StringRef name = llvm::sys::path::filename(trajectoryFile);
    if (name != continuedTrajectory)
      return inputError("the checkpoint counts the frames of '" +
                        continuedTrajectory + "', which is not the "
                        "trajectory of the reporter, '" + name + "'; "
                        "continue with append=False to write the frames "
                        "that follow to a part of their own");
    if (llvm::sys::fs::exists(trajectoryFile))
      appendsFrames = true;
    else if (continuedFrames > 0)
      return inputError("'" + trajectoryFile + "' is missing, and the "
                        "checkpoint counts " + llvm::Twine(continuedFrames) +
                        " frames in it; continue with append=False to write "
                        "the frames that follow to a part of their own");
  }
  std::vector<std::string> opened;
  if (given.energyPath != reports.energyPath && !given.energyPath.empty() &&
      !appendsEnergies)
    opened.push_back(energyFile);
  if (given.trajectoryPath != reports.trajectoryPath &&
      !given.trajectoryPath.empty() && !appendsFrames)
    opened.push_back(trajectoryFile);
  // None is moved unless all can be, as `mdir run` backs up (D149).
  for (const std::string &file : opened)
    if (llvm::Error error = checkBackup(file))
      return inputError(llvm::toString(std::move(error)));
  for (const std::string &file : opened) {
    auto backup = backUpOutput(file);
    if (!backup)
      return inputError(llvm::toString(backup.takeError()));
  }
  if (given.energyPath != reports.energyPath) {
    out.energies.close();
    if (!given.energyPath.empty())
      if (llvm::Error error = out.energies.open(
              energyFile, getEnergyColumns(out),
              appendsEnergies ? keepThrough : std::nullopt))
        return inputError(llvm::toString(std::move(error)));
  }
  out.energyPeriod = given.energyPeriod;
  if (given.trajectoryPath != reports.trajectoryPath ||
      given.trajectoryFormat != reports.trajectoryFormat ||
      given.framePeriod != reports.framePeriod) {
    if (out.trajectory)
      out.trajectory->close();
    out.trajectory.reset();
    out.hasTrajectory = false;
    if (!given.trajectoryPath.empty()) {
      double cell[3], tilts[3];
      for (int k = 0; k != 3; ++k) {
        cell[k] = out.box[k] / units::length;
        tilts[k] = system.tilt[k] / units::length;
      }
      auto writer = createTrajectoryWriter(given.trajectoryFormat);
      writer->setPeriodic(prepared.control.periodic);
      int64_t firstFrame = (step / given.framePeriod + 1) * given.framePeriod;
      if (appendsFrames) {
        auto removed = writer->append(trajectoryFile, system.getNumParticles(),
                                      continuedFrames, given.framePeriod,
                                      prepared.control.timestep, cell);
        if (!removed)
          return inputError(llvm::toString(removed.takeError()));
      } else if (llvm::Error error = writer->open(
                     trajectoryFile, system.getNumParticles(), firstFrame,
                     given.framePeriod, prepared.control.timestep, cell)) {
        return inputError(llvm::toString(std::move(error)));
      }
      writer->setTilt(tilts);
      out.trajectory = std::move(writer);
      out.hasTrajectory = true;
    }
  }
  reports = given;
  return llvm::Error::success();
}

void Simulation::closeReports() {
  std::lock_guard<std::mutex> lock(getRunMutex());
  output->energies.close();
  if (output->trajectory)
    output->trajectory->close();
  output->trajectory.reset();
  output->hasTrajectory = false;
  output->energyPeriod = 0;
  reports = Reports();
}

llvm::Error Simulation::checkLeases(const char *operation) const {
  int64_t live = leases;
  if (live <= 0)
    return llvm::Error::success();
  return simulationError(llvm::Twine(operation) + " is refused while " +
                         llvm::Twine(live) +
                         (live == 1 ? " lease of a view" : " leases of views") +
                         " of this simulation " + (live == 1 ? "is" : "are") +
                         " alive: release the view (View.release()) and "
                         "delete the tensors taken from it first");
}

void Simulation::waitForConsumers() {
  // Work of a consumer on any stream of the context may still read the
  // buffers that it was given (D[python-dlpack]).
  if (!exported.exchange(false))
    return;
  if (auto wait = findRuntime<void()>("mdrtDeviceWaitAll"))
    wait();
}

llvm::Expected<compiler::SimulationView> Simulation::takeView() {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  struct Release {
    std::atomic<bool> &flag;
    ~Release() { flag = false; }
  } release{busy};
  if (!activation || !activation->atBoundary)
    return simulationError(
        failed ? llvm::Twine("the simulation failed; its state is on the "
                             "host only, which state() copies")
               : llvm::Twine("the simulation has no state where its program "
                             "keeps it before its first run; run it, or "
                             "evaluate it with run(0, energy=True), first"));
  Activation &a = *activation;
  const Program &p = compiled->program;
  compiler::SimulationView view;
  view.stateWidth = getWidth(p.state);
  view.forceWidth = getWidth(p.force);
  view.positions = getStart(a.x, view.stateWidth);
  view.velocities = getStart(a.v, view.stateWidth);
  view.forces = getStart(a.f, view.forceWidth);
  view.ids = reinterpret_cast<const int32_t *>(getStart(a.id, sizeof(int32_t)));
  view.count = static_cast<size_t>(a.x.sizes[0]);
  view.onDevice = a.onDevice;
  if (a.onDevice) {
    std::lock_guard<std::mutex> lock(getRunMutex());
    static auto ordinal = findRuntime<int32_t()>("mdrtDeviceOrdinal");
    view.device = ordinal ? ordinal() : 0;
  }
  view.step = step;
  view.time = getTime();
  view.velocityOffset =
      hasRun && prepared.control.integrator == Integrator::Leapfrog ? -0.5
                                                                    : 0.0;
  view.generation = generation;
  ++leases;
  return view;
}

void Simulation::handOff(uint64_t consumer) {
  if (!activation || !activation->onDevice)
    return;
  exported = true;
  std::lock_guard<std::mutex> lock(getRunMutex());
  static auto handoff = findRuntime<void(uint64_t)>("mdrtDeviceHandOff");
  if (handoff)
    handoff(consumer);
}

double Simulation::getTime() const {
  // A minimization has no time.
  if (prepared.control.minimize)
    return 0.0;
  // From the time of a checkpoint that the simulation continues
  // (D[python-checkpoints]).
  return output->getTime(step);
}

llvm::Expected<SimulationState> Simulation::getState() const {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  SimulationState state;
  {
    // The copy of the state, from where the program keeps it.
    std::lock_guard<std::mutex> lock(getRunMutex());
    downloadState();
  }
  state.step = step;
  state.time = getTime();
  state.positions = system.positions;
  state.velocities = system.velocities;
  state.forces = forces;
  state.velocityOffset =
      hasRun && prepared.control.integrator == Integrator::Leapfrog ? -0.5
                                                                    : 0.0;
  for (int k = 0; k != 3; ++k) {
    state.box[k] = output->box[k];
    state.tilt[k] = system.tilt[k];
  }
  const auto &least = output->lastMinimization;
  if (prepared.control.minimize && hasRun && least.step == step)
    state.minimization =
        SimulationMinimization{least.energy, least.rmsForce, least.maxForce,
                               least.stepSize, least.maxForceParticle,
                               minimizationCheckedStep == step
                                   ? minimizationConverged
                                   : std::nullopt};
  state.tunablesVersion = tunablesVersion;
  const auto &row = output->lastEnergies;
  if (hasRun && !prepared.control.minimize && row.step == step)
    state.energies = SimulationEnergies{row.potential, row.kinetic, row.total,
                                        row.conserved, row.temperature,
                                        row.virial, row.pressure, row.volume};
  busy = false;
  return state;
}

llvm::Error Simulation::updateTunables(
    const std::vector<std::pair<std::string, std::vector<double>>> &changes) {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  struct Release {
    std::atomic<bool> &flag;
    ~Release() { flag = false; }
  } release{busy};
  if (llvm::Error error = checkLeases("an update of the tunables"))
    return std::move(error);
  if (failed)
    return simulationError("the simulation failed earlier; it keeps the "
                           "state of step " + llvm::Twine(step) +
                           " and takes no new values");
  const model::TunableSet &set = prepared.tunables;
  if (set.empty())
    return inputError("the program of this simulation declares no tunable "
                      "parameters (System.tunables)");
  std::vector<std::vector<double>> values = tunableValues;
  for (const auto &[name, given] : changes) {
    int k = set.find(name);
    if (k < 0)
      return inputError("this simulation has no tunable named '" + name + "'");
    values[k] = given;
  }
  auto program = rebuildTunables(values);
  if (!program)
    return program.takeError();
  // The program takes the new values from the state of the host, in an
  // activation of its own (D215).
  {
    std::lock_guard<std::mutex> lock(getRunMutex());
    downloadState();
    endActivation();
  }
  std::swap(compiled->program, *program);
  std::vector<std::vector<double>> before = std::move(tunableValues);
  tunableValues = std::move(values);
  // The forces that the next step begins with, at the new values; a
  // minimization takes none.
  if (hasRun && !prepared.control.minimize) {
    if (llvm::Error error = evaluatePart()) {
      std::swap(compiled->program, *program);
      tunableValues = std::move(before);
      failed = false;
      output->lastEnergies.step = -1;
      return simulationError("the new values of the tunables: " +
                             llvm::toString(std::move(error)) +
                             "; the update is undone");
    }
  } else {
    output->lastEnergies.step = -1;
  }
  ++tunablesVersion;
  tunablesHistory.push_back({step, tunablesVersion});
  output->tunablesVersion = tunablesVersion;
  return llvm::Error::success();
}

llvm::Expected<Program>
Simulation::rebuildTunables(const std::vector<std::vector<double>> &values) {
  const model::TunableSet &set = prepared.tunables;
  // The values of the program, built from the model with the new values by
  // the code that compiled it: everything that depends on them, and the
  // same program, or the change is structural.
  Control control = compiled->control;
  System system = compiledSystem;
  if (llvm::Error error = model::applyTunables(set, values, control, system))
    return error;
  // The width of the neighbor structures is the compiled program's: counting
  // the neighbors anew took 0.8 s of the 0.81 s of an update on JAC.
  control.neighborWidth = compiled->program.neighborWidth;
  auto program = buildProgram(control, system);
  if (!program)
    return inputError("the new values of the tunables: " +
                      llvm::toString(program.takeError()));
  if (program->module != compiled->program.module) {
    StringRef was = compiled->program.module, now = program->module;
    size_t at = 0;
    while (at < was.size() && at < now.size() && was[at] == now[at])
      ++at;
    size_t begin = was.rfind('\n', at);
    begin = begin == StringRef::npos ? 0 : begin + 1;
    StringRef line = was.substr(begin).split('\n').first.trim();
    return inputError("the new values of the tunables change the program, "
                      "not only its values (at '" + line + "'): compile it "
                      "with them (Tunable values=...)");
  }
  return std::move(*program);
}

llvm::Error Simulation::evaluatePart() {
  // On the first call, the start of the run; on a later one, the forces of
  // the state given anew (%first_call 2), which only a program with
  // tunables takes without the half kick back of leapfrog.
  if (hasRun && prepared.control.integrator == Integrator::Leapfrog &&
      !compiled->program.tunable)
    return unsupported("an evaluation without a step after the first run "
                       "takes velocity Verlet, or a program with tunable "
                       "parameters: leapfrog's velocities are half a step "
                       "behind the positions");
  bool refresh = hasRun;
  // The evaluation begins an activation from the state of the host.
  {
    std::lock_guard<std::mutex> lock(getRunMutex());
    downloadState();
    endActivation();
  }
  refreshing = refresh;
  llvm::Error error = runPart(*compiled, Part());
  refreshing = false;
  if (error)
    return error;
  // Leapfrog's velocities are half a step behind the positions after the
  // first call: a row of energies would take them for those of the step.
  if (refresh && prepared.control.integrator == Integrator::Leapfrog)
    output->lastEnergies.step = -1;
  return llvm::Error::success();
}

llvm::Error Simulation::evaluate() {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  struct Release {
    std::atomic<bool> &flag;
    ~Release() { flag = false; }
  } release{busy};
  if (llvm::Error error = checkLeases("an evaluation"))
    return std::move(error);
  if (failed)
    return simulationError("the simulation failed earlier; it keeps the "
                           "state of step " + llvm::Twine(step) +
                           " and runs no further");
  if (prepared.control.minimize)
    return inputError("this simulation minimizes: minimize(0) evaluates "
                      "nothing; call minimize(steps)");
  return evaluatePart();
}

//===----------------------------------------------------------------------===//
// Checkpoints (D[python-checkpoints], docs/python-checkpoints.md)
//===----------------------------------------------------------------------===//

namespace {
const char *getIntegratorName(const Control &control) {
  return control.minimize                             ? "MINIMIZATION"
         : control.integrator == Integrator::Leapfrog ? "LEAPFROG"
         : control.integrator == Integrator::Brownian ? "BROWNIAN"
                                                      : "VELOCITY_VERLET";
}

const char *getPrecisionName(Precision precision) {
  return precision == Precision::Single  ? "single"
         : precision == Precision::Mixed ? "mixed"
                                         : "double";
}

std::string hashText(StringRef text) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)),
                     /*LowerCase=*/true);
}

std::string describeChanges(const std::vector<FingerprintChange> &changes) {
  std::string text;
  for (const FingerprintChange &change : changes)
    text += "\n  " + change.name + ": " + change.before + " -> " +
            change.after;
  return text;
}
} // namespace

std::string Simulation::getReportPath(const std::string &path) const {
  if (path.empty() || outputsPart <= 0)
    return path;
  return getPartPath(path, outputsPart);
}

llvm::Error Simulation::saveCheckpoint(const std::string &path,
                                       const std::string &creatorVersion) {
  if (!hasCheckpointSupport())
    return unsupported("this build of MDIR has no HDF5, which checkpoints "
                       "need");
  if (path.empty())
    return inputError("save_checkpoint takes the name of a file");
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  struct Release {
    std::atomic<bool> &flag;
    ~Release() { flag = false; }
  } release{busy};
  if (failed)
    return simulationError("the simulation failed earlier; it keeps the "
                           "state of step " + llvm::Twine(step) +
                           " and writes no checkpoint");
  const Control &control = prepared.control;
  // A step begins with forces: a simulation that has not run evaluates
  // its start, as its first run would.
  if (!hasRun && !control.minimize)
    if (llvm::Error error = evaluatePart())
      return error;
  // The next part begins from the state written, as a simulation that
  // continues the checkpoint begins (and as `mdir run` builds its
  // structures anew at each checkpoint): with new neighbor structures, so
  // that a simulation that goes on and one that continues the file take the
  // same steps to the bit in the deterministic mode.
  {
    std::lock_guard<std::mutex> lock(getRunMutex());
    downloadState();
    endActivation();
  }
  Checkpoint checkpoint;
  checkpoint.step = step;
  checkpoint.time = getTime();
  checkpoint.positions = system.positions;
  checkpoint.velocities = system.velocities;
  if (hasRun && forces.size() == system.positions.size())
    checkpoint.forces = forces;
  checkpoint.masses = system.masses;
  checkpoint.species.assign(system.types.begin(), system.types.end());
  for (int k = 0; k != 3; ++k) {
    checkpoint.box[k] = output->box[k];
    checkpoint.tilt[k] = system.tilt[k];
  }
  checkpoint.periodic = control.periodic;
  checkpoint.integrator = getIntegratorName(control);
  checkpoint.velocityOffset =
      hasRun && control.integrator == Integrator::Leapfrog && !control.minimize
          ? -0.5
          : 0.0;
  checkpoint.precision = getPrecisionName(control.precision);
  checkpoint.timestep = control.timestep;
  checkpoint.seed = control.seed;
  checkpoint.firstStep = runFirstStep;
  checkpoint.part = runPartNumber;
  checkpoint.outputsPart = outputsPart;
  if (output->hasTrajectory && output->trajectory) {
    checkpoint.trajectory =
        llvm::sys::path::filename(getReportPath(reports.trajectoryPath)).str();
    checkpoint.frames = output->trajectory->getNumFrames();
  }
  checkpoint.bath = output->bath;
  checkpoint.fingerprint = prepared.fingerprint;
  checkpoint.creatorVersion = creatorVersion;
  // The additional entries: the front end, the hashes of the model and of
  // the plan, and the tunables.
  checkpoint.frontEnd = "python";
  std::string model;
  for (const FingerprintEntry &entry : prepared.fingerprint)
    if (entry.group != "execution")
      model += entry.group + "\t" + entry.name + "\t" + entry.value + "\n";
  model += model::describeTunables(prepared.tunables);
  checkpoint.modelHash = hashText(model);
  const Program &program = compiled->program;
  std::string plan =
      std::string("target=") +
      (control.target == Target::GPU ? "gpu" : "cpu") +
      "\nprecision=" + getPrecisionName(control.precision) +
      "\ndeterministic=" + (control.deterministic ? "true" : "false") +
      "\nreorders=" + (program.reorders ? "true" : "false") +
      "\nentry=" + program.entry + "\nstate=" +
      (program.state == Element::F64 ? "float64" : "float32") +
      "\nforce=" + (program.force == Element::F64 ? "float64" : "float32") +
      "\npme=" + (program.pme ? "true" : "false") + "\npme_grid=" +
      std::to_string(program.pmeGrid[0]) + "," +
      std::to_string(program.pmeGrid[1]) + "," +
      std::to_string(program.pmeGrid[2]) + "\n" + program.module;
  checkpoint.planHash = hashText(plan);
  if (!prepared.tunables.empty()) {
    checkpoint.tunableDeclarations = model::describeTunables(prepared.tunables);
    for (size_t k = 0; k != prepared.tunables.tunables.size(); ++k)
      checkpoint.tunables.emplace_back(prepared.tunables.tunables[k].name,
                                       tunableValues[k]);
    checkpoint.tunablesVersion = tunablesVersion;
    checkpoint.tunablesHistory = tunablesHistory;
  }
  if (llvm::Error error = writeCheckpoint(path, checkpoint))
    return simulationError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::string>>
Simulation::continueFrom(const Checkpoint &checkpoint, bool stage,
                         bool append) {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  struct Release {
    std::atomic<bool> &flag;
    ~Release() { flag = false; }
  } release{busy};
  if (hasRun || step != 0 || activation)
    return inputError("a simulation takes a checkpoint before its first run");
  const Control &control = prepared.control;
  std::vector<std::string> notes;
  size_t count = system.getNumParticles();
  if (checkpoint.getNumParticles() != count)
    return inputError("the checkpoint holds " +
                      llvm::Twine(checkpoint.getNumParticles()) +
                      " particles, and the program " + llvm::Twine(count));
  for (size_t i = 0; i != count; ++i)
    if (checkpoint.species[i] != static_cast<int32_t>(system.types[i]))
      return inputError("particle " + llvm::Twine(i + 1) +
                        " has another type in the checkpoint than in the "
                        "program");
  auto takeCell = [&] {
    for (int k = 0; k != 3; ++k) {
      system.box[k] = output->box[k] = checkpoint.box[k];
      system.tilt[k] = checkpoint.tilt[k];
    }
    output->volume = output->box[0] * output->box[1] * output->box[2];
  };
  // A minimization takes the positions and the cell of any checkpoint, and
  // a run of dynamics those of a minimization, and begins anew (as
  // `mdir run` does).
  if (control.minimize || checkpoint.integrator == "MINIMIZATION") {
    system.positions = checkpoint.positions;
    takeCell();
    notes.push_back("the simulation begins at the positions and the cell of "
                    "the checkpoint, at step 0");
    return notes;
  }
  std::string integrator = getIntegratorName(control);
  if (checkpoint.integrator != integrator)
    return inputError("the checkpoint was written with the integrator " +
                      checkpoint.integrator + ", and the program uses " +
                      integrator + "; the velocities of the two are not of "
                      "the same time");

  // What defined the run (D172): the same run continues only with the same
  // physics and coupling; a stage takes the forces only then.
  const Fingerprint &current = prepared.fingerprint;
  std::vector<FingerprintChange> changes =
      compareFingerprints(checkpoint.fingerprint, current, "physics");
  std::vector<FingerprintChange> coupling =
      compareFingerprints(checkpoint.fingerprint, current, "coupling");
  changes.insert(changes.end(), coupling.begin(), coupling.end());
  std::vector<FingerprintChange> execution =
      compareFingerprints(checkpoint.fingerprint, current, "execution");
  std::string declarations = model::describeTunables(prepared.tunables);
  bool sameTunables = checkpoint.tunableDeclarations == declarations;
  if (!stage) {
    if (!changes.empty())
      return inputError(
          "the checkpoint was written by a run of other physics or coupling, "
          "which a continuation of the same run does not change; begin a "
          "new stage from it (stage=True) instead. What differs (checkpoint "
          "-> program):" + describeChanges(changes));
    if (!sameTunables)
      return inputError("the checkpoint declares other tunable parameters "
                        "than the program; begin a new stage from it "
                        "(stage=True) instead");
    if (checkpoint.forces.empty())
      return inputError("the checkpoint holds no forces, which a step "
                        "begins with");
  }
  if (!execution.empty())
    notes.push_back("the run continues with other settings of its "
                    "execution, whose bits follow them (checkpoint -> "
                    "program):" + describeChanges(execution));
  bool takesForces = changes.empty() && !checkpoint.forces.empty();
  if (stage && !takesForces) {
    // The forces are evaluated at the first step without the half kick back
    // of leapfrog, whose velocities the checkpoint holds half a step behind.
    if (changes.empty())
      notes.push_back("the checkpoint holds no forces; the simulation "
                      "evaluates them at its first step");
    else
      notes.push_back("the checkpoint was written by a run of other physics "
                      "or coupling; the simulation evaluates the forces at "
                      "its first step (checkpoint -> program):" +
                      describeChanges(changes));
  }

  // The values of the tunables of the checkpoint, where it declares the
  // same; a continuation takes their version and history as well.
  if (!prepared.tunables.empty() && sameTunables &&
      !checkpoint.tunables.empty()) {
    std::vector<std::vector<double>> values = tunableValues;
    for (const auto &[name, given] : checkpoint.tunables) {
      int k = prepared.tunables.find(name);
      if (k < 0)
        return inputError("the checkpoint has values of the tunable '" +
                          name + "', which the program does not declare");
      values[k] = given;
    }
    if (llvm::Error error =
            model::checkTunableValues(prepared.tunables, values))
      return std::move(error);
    if (values != tunableValues) {
      auto program = rebuildTunables(values);
      if (!program)
        return program.takeError();
      std::swap(compiled->program, *program);
      tunableValues = std::move(values);
    }
    if (!stage) {
      tunablesVersion = checkpoint.tunablesVersion;
      tunablesHistory = checkpoint.tunablesHistory;
    } else {
      tunablesHistory = {{checkpoint.step, 0}};
    }
    output->tunablesVersion = tunablesVersion;
  }

  system.positions = checkpoint.positions;
  system.velocities = checkpoint.velocities;
  takeCell();
  if (takesForces)
    forces = checkpoint.forces;
  else
    forces.assign(3 * count, 0.0);
  startRefresh = !takesForces;
  hasRun = true;
  hostCurrent = true;
  step = checkpoint.step;
  // The time continues from that of the checkpoint, whose run may have
  // had another time step.
  output->firstStep = 0;
  output->firstTime =
      checkpoint.time - static_cast<double>(checkpoint.step) * control.timestep;
  if (!stage) {
    // The same run: its bath, the step it began at, its part, and its
    // files (D129, D130, D149).
    output->bath = checkpoint.bath;
    runFirstStep = checkpoint.firstStep;
    runPartNumber = checkpoint.part + 1;
    outputsPart = append ? checkpoint.outputsPart : runPartNumber;
    if (append) {
      keepThrough = checkpoint.step;
      continuedTrajectory = checkpoint.trajectory;
      continuedFrames = checkpoint.frames;
    }
  } else {
    output->bath = 0.0;
    runFirstStep = checkpoint.step;
    runPartNumber = 1;
    outputsPart = 0;
  }
  return notes;
}
