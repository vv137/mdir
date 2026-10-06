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
#include "mdir/Driver/Output.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "JITEngine.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <numeric>
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
/// of its module): what a call of an entry leaves allocated is the call's
/// own, since every buffer of a part is made anew, and is freed when the
/// call returns (#110). A pointer that compiled code did not allocate goes to
/// `free` as it is.
std::mutex &getAllocationMutex() {
  static std::mutex mutex;
  return mutex;
}
std::unordered_set<void *> &getAllocations() {
  static std::unordered_set<void *> allocations;
  return allocations;
}
extern "C" void *allocateForCode(size_t size) {
  void *pointer = std::malloc(size);
  if (pointer) {
    std::lock_guard<std::mutex> lock(getAllocationMutex());
    getAllocations().insert(pointer);
  }
  return pointer;
}
extern "C" void freeForCode(void *pointer) {
  if (!pointer)
    return;
  {
    std::lock_guard<std::mutex> lock(getAllocationMutex());
    getAllocations().erase(pointer);
  }
  std::free(pointer);
}
void freeAllocationsOfCall() {
  std::lock_guard<std::mutex> lock(getAllocationMutex());
  for (void *pointer : getAllocations())
    std::free(pointer);
  getAllocations().clear();
}

/// Opens or closes a call of an entry in a runtime library (`mdrtBeginCall`,
/// `mdrtDeviceEndCall`, ...), which frees what the call made; nothing if the
/// library is not loaded.
void callRuntime(const char *name) {
  if (auto function = reinterpret_cast<void (*)()>(
          llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name)))
    function();
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

Simulation::~Simulation() {
  std::lock_guard<std::mutex> lock(getRunMutex());
  // The files of the reports are complete when the simulation ends.
  if (output) {
    output->energies.close();
    if (output->trajectory)
      output->trajectory->close();
  }
  compiled.reset();
}

static llvm::Expected<std::unique_ptr<Simulation::Engine>>
compileEngine(const Control &control, const System &system,
              const model::Execution &execution) {
  auto engine = std::make_unique<Simulation::Engine>();
  engine->control = control;
  auto program = buildProgram(control, system);
  if (!program)
    return program.takeError();
  engine->program = std::move(*program);
  engine->volume = system.box[0] * system.box[1] * system.box[2];
  // The kernels of a GPU take their math functions from libdevice: that of
  // the toolkit the environment names, else that of the build, as for
  // `mdir run` (tools/mdir/BugReport.cpp, getCudaToolkitRoot).
  if (control.target == Target::GPU) {
    bool named = false;
    for (const char *name : {"CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"})
      if (const char *root = std::getenv(name); root && *root) {
        setenv("CUDA_ROOT", root, /*overwrite=*/0);
        named = true;
        break;
      }
    if (!named && *MDIR_CUDA_ROOT)
      setenv("CUDA_ROOT", MDIR_CUDA_ROOT, /*overwrite=*/0);
  }
  // The context lowers with the threads of the process, so that a
  // simulation keeps no pool of its own (D211).
  engine->context = std::make_unique<mlir::MLIRContext>(
      compiler::getRegistry(), mlir::MLIRContext::Threading::DISABLED);
  compiler::shareThreadPool(*engine->context);
  auto seconds = [start = std::chrono::steady_clock::now()] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start).count();
  };
  auto module = compiler::lowerModule(*engine->context, control,
                                      engine->program);
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
      scheduler ? "pre-RA-sched=fast" : "");
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
    add("_mlir_ciface_mdrtFinishForces", (void *)&_mlir_ciface_mdrtFinishForces);
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
  engine->stats.engineSeconds = seconds() - jitStart;

  engine->masses = system.masses;
  return std::move(engine);
}

llvm::Expected<std::unique_ptr<Simulation>>
Simulation::create(const model::PreparedModel &prepared) {
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
  // built from to build them anew (D[python-tunable]).
  simulation->compiledSystem = system;
  simulation->tunableValues = prepared.tunables.values;
  simulation->tunablesHistory = {{0, 0}};
  auto engine = compileEngine(control, system, prepared.execution);
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

  // The state before the part, which a failed part leaves.
  System before = system;
  std::vector<double> forcesBefore = forces;
  double boxBefore[3] = {out.box[0], out.box[1], out.box[2]};
  double bathBefore = out.bath;

  std::string failure;
  out.fail = [&](const std::string &message) {
    if (failure.empty())
      failure = message;
  };
  out.system = &system;
  out.finalForces.clear();
  setOutput(&out);

  size_t count = system.getNumParticles();
  // The first call begins the run and ignores the forces that it is given
  // (D211); a later call continues those that the
  // last left.
  if (p.takesForces && hasRun && forces.size() != 3 * count)
    return simulationError("the segment takes forces, but has none; this "
                           "is a defect of mdir");
  // The arguments in the order of the entry (Builder.h).
  Arguments a;
  auto vector = [&](const std::vector<double> &values, Element element) {
    a.vectors.push_back(std::make_unique<HostBuffer<2>>(values, element, count));
    a.vectors.back()->addTo(a.pointers);
  };
  vector(system.positions, p.state);
  vector(system.velocities, p.state);
  if (p.takesForces)
    vector(hasRun ? forces : std::vector<double>(3 * count, 0.0), p.force);
  a.reals.push_back(std::make_unique<HostBuffer<1>>(engine.masses, p.mass, count));
  a.reals.back()->addTo(a.pointers);
  for (const Program::Field &field : p.fields) {
    if (!field.isInteger) {
      a.reals.push_back(
          std::make_unique<HostBuffer<1>>(field.values, p.parameter, count));
      a.reals.back()->addTo(a.pointers);
      continue;
    }
    a.integers.emplace_back(field.values.begin(), field.values.end());
    auto d = std::make_unique<StridedMemRefType<int32_t, 1>>();
    d->basePtr = d->data = a.integers.back().data();
    d->offset = 0;
    d->sizes[0] = count;
    d->strides[0] = 1;
    addDescriptor(a.pointers, *d);
    a.integerFields.push_back(std::move(d));
  }
  for (const Program::Table &table : p.tables) {
    a.doubles.push_back(table.values);
    auto d = std::make_unique<StridedMemRefType<double, 2>>();
    d->basePtr = d->data = a.doubles.back().data();
    d->offset = 0;
    d->sizes[0] = table.count;
    d->sizes[1] = table.getColumns();
    d->strides[0] = table.getColumns();
    d->strides[1] = 1;
    addDescriptor(a.pointers, *d);
    a.tables.push_back(std::move(d));
  }
  // Moving a vector keeps its storage, so the descriptors stay valid as
  // the lists grow.
  for (const Program::TupleSet &set : p.tupleSets) {
    a.integers.push_back(set.members);
    auto m = std::make_unique<StridedMemRefType<int32_t, 2>>();
    m->basePtr = m->data = a.integers.back().data();
    m->offset = 0;
    m->sizes[0] = set.size();
    m->sizes[1] = set.arity;
    m->strides[0] = set.arity;
    m->strides[1] = 1;
    addDescriptor(a.pointers, *m);
    a.members.push_back(std::move(m));
    for (const Program::Field &field : set.fields) {
      a.doubles.push_back(field.values);
      auto v = std::make_unique<StridedMemRefType<double, 1>>();
      v->basePtr = v->data = a.doubles.back().data();
      v->offset = 0;
      v->sizes[0] = set.size();
      v->strides[0] = 1;
      addDescriptor(a.pointers, *v);
      a.tupleFields.push_back(std::move(v));
    }
  }
  a.integers.emplace_back(count);
  std::vector<int32_t> &numbers = a.integers.back();
  for (size_t i = 0; i != count; ++i)
    numbers[i] = static_cast<int32_t>(i);
  a.identities.basePtr = a.identities.data = numbers.data();
  a.identities.offset = 0;
  a.identities.sizes[0] = count;
  a.identities.strides[0] = 1;
  addDescriptor(a.pointers, a.identities);
  double box[3] = {out.box[0], out.box[1], out.box[2]};
  double timestep = engine.control.timestep;
  int64_t begin = step;
  for (double &edge : box)
    a.pointers.push_back(&edge);
  a.pointers.push_back(&timestep);
  a.pointers.push_back(&begin);
  for (int64_t *count : {&part.outer, &part.inner, &part.tail, &part.plain,
                         &part.close, &part.closeInner})
    a.pointers.push_back(count);
  double firstSize = minimizationSize;
  int64_t framePeriod = reports.framePeriod;
  if (engine.control.minimize) {
    a.pointers.push_back(&firstSize);
  } else {
    a.pointers.push_back(&part.closePeriods);
    a.pointers.push_back(&framePeriod);
  }
  // 2: the forces of the state given anew, after an update of the
  // tunables (D[python-tunable]).
  int64_t firstCall = hasRun ? (refreshing ? 2 : 0) : 1;
  a.pointers.push_back(&firstCall);
  double baroConstant = p.baroConstant, baroEnergyConstant = p.baroEnergyConstant;
  if (p.takesConstants) {
    a.pointers.push_back(&baroConstant);
    a.pointers.push_back(&baroEnergyConstant);
  }
  out.quietStep = refreshing ? step : -1;
  out.tunablesVersion = prepared.tunables.empty() ? -1 : tunablesVersion;
  Output::MinimizationRow rowBefore = out.lastMinimization;

  stopMessage.clear();
  // What the call allocates is its own and is freed when it returns (#110).
  callRuntime("mdrtBeginCall");
  callRuntime("mdrtDeviceBeginCall");
  engine.function(a.pointers.data());
  callRuntime("mdrtDeviceEndCall");
  callRuntime("mdrtEndCall");
  freeAllocationsOfCall();
  out.fail = nullptr;
  if (failure.empty())
    failure = stopMessage;
  if (failure.empty() && out.finalForces.size() != 3 * count)
    failure = "the segment returned no forces; this is a defect of mdir";
  // On the CPU the structures do not test the positions (D107 does on a
  // device): a state that is not numbers ends the part here, before the
  // next part would put such particles in order.
  if (failure.empty()) {
    auto finite = [](const std::vector<double> &values) {
      return llvm::all_of(values, [](double v) { return std::isfinite(v); });
    };
    if (!finite(system.positions) || !finite(system.velocities) ||
        !finite(out.finalForces))
      failure = "the state at its end is not numbers (a time step too long, "
                "a bad contact, or a defect of mdir)";
  }
  if (!failure.empty()) {
    system = std::move(before);
    forces = std::move(forcesBefore);
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
  forces = std::move(out.finalForces);
  out.finalForces.clear();
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
                                             const std::function<bool()> &poll) {
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
                           " and runs no further");
  if (!prepared.control.minimize)
    return inputError("this simulation runs dynamics: minimize() takes a "
                      "program whose integrator minimizes");
  int64_t total = count ? *count : minimizationSteps;
  if (total < 0)
    return inputError("minimize takes a nonnegative number of steps");
  stopRequested = false;
  // Parts as those of run: the program of the parts after the first takes
  // the positions and the length of the step that the last left.
  int64_t taken = 0;
  while (taken < total) {
    if (taken > 0 && (stopRequested || (poll && poll())))
      break;
    Part part;
    part.outer = std::min(total - taken, getPartSteps(1));
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
  std::vector<std::string> opened;
  if (given.energyPath != reports.energyPath && !given.energyPath.empty())
    opened.push_back(given.energyPath);
  if (given.trajectoryPath != reports.trajectoryPath &&
      !given.trajectoryPath.empty())
    opened.push_back(given.trajectoryPath);
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
      if (llvm::Error error = out.energies.open(given.energyPath,
                                                getEnergyColumns(out), {}))
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
      if (llvm::Error error = writer->open(
              given.trajectoryPath, system.getNumParticles(), firstFrame,
              given.framePeriod, prepared.control.timestep, cell))
        return inputError(llvm::toString(std::move(error)));
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

double Simulation::getTime() const {
  // A minimization has no time.
  if (prepared.control.minimize)
    return 0.0;
  return static_cast<double>(step) * prepared.control.timestep;
}

llvm::Expected<SimulationState> Simulation::getState() const {
  if (busy.exchange(true))
    return simulationError("another operation is under way on this "
                           "simulation");
  SimulationState state;
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
                               least.stepSize, least.maxForceParticle};
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
  if (failed)
    return simulationError("the simulation failed earlier; it keeps the "
                           "state of step " + llvm::Twine(step) +
                           " and runs no further");
  if (prepared.control.minimize)
    return inputError("this simulation minimizes: minimize(0) evaluates "
                      "nothing; call minimize(steps)");
  return evaluatePart();
}
