// A simulation that persists across runs (D196,
// docs/python-segments.md). It compiles the program of its first segment,
// which begins as `mdir run` does, and the program of the segments after it,
// which continue the state as `mdir run --continue` continues a checkpoint;
// the entries of both take the counts of their loops, so that one program
// runs any number of steps from any step of the period of coupling.
#include "mdir/Compiler/Simulation.h"
#include "mdir/Compiler/Compile.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Driver/Output.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
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
  std::unique_ptr<mlir::ExecutionEngine> engine;
  void (*function)(void **) = nullptr;
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

Simulation::~Simulation() = default;

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
  // One module is lowered at a time: a pool of threads for each engine
  // would stay for the life of the simulation.
  engine->context = std::make_unique<mlir::MLIRContext>(
      compiler::getRegistry(), mlir::MLIRContext::Threading::DISABLED);
  auto module = compiler::lowerModule(*engine->context, control,
                                      engine->program);
  if (!module)
    return module.takeError();
  engine->module = std::move(*module);

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

  auto targetBuilder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!targetBuilder)
    return targetBuilder.takeError();
  auto targetMachine = targetBuilder->createTargetMachine();
  if (!targetMachine)
    return targetMachine.takeError();
  // Match the default section of the functions ORC adds after the transformer.
  // On x86-64 ELF the large code model uses .ltext (with SHF_X86_64_LARGE),
  // including for ORC's synthesized initialization and deinitialization entry.
  StringRef textSection =
      (*targetMachine)->getTargetTriple().getArch() == llvm::Triple::x86_64 &&
              (*targetMachine)->getCodeModel() == llvm::CodeModel::Large
          ? ".ltext"
          : ".text";

  llvm::cl::Option *scheduler = getSchedulerOption();
  if (scheduler)
    (void)scheduler->addOccurrence(0, "pre-RA-sched", "fast");
  mlir::ExecutionEngineOptions options;
  llvm::SmallVector<StringRef> shared(paths.begin(), paths.end());
  options.sharedLibPaths = shared;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  // A frame registration describes the bounding PC range of its functions.
  // Keep the GPU module's host entry, constructors, and destructors together:
  // separately mapped text sections can enclose another engine's code, and
  // libgcc's interval index can retain a freed registration on deregistration.
  auto keepHostCodeTogether = [textSection](llvm::Module *module) {
    if (module->getTargetTriple().isOSBinFormatELF())
      for (llvm::Function &function : *module)
        if (!function.isDeclaration())
          function.setSection(textSection);
    return llvm::Error::success();
  };
  options.transformer = keepHostCodeTogether;
  auto created = mlir::ExecutionEngine::create(
      *engine->module, options, std::move(*targetMachine));
  if (!created) {
    if (scheduler)
      (void)scheduler->addOccurrence(0, "pre-RA-sched", "default");
    return llvm::make_error<compiler::CompileError>(
        "cannot compile the program for execution: " +
        llvm::toString(created.takeError()));
  }
  engine->engine = std::move(*created);
  bool writesForces = engine->program.writesForces;
  engine->engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
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
    add("_mlir_ciface_mdrtFinishForces", (void *)&_mlir_ciface_mdrtFinishForces);
    add("_mlir_ciface_mdrtWriteCheckpoint",
        writesForces ? (void *)&_mlir_ciface_mdrtWriteCheckpointWithForces
                     : (void *)&_mlir_ciface_mdrtWriteCheckpoint);
    return symbols;
  });

  // The runtime takes the failures that it cannot return from to the
  // simulation, and a GPU simulation the device that the process uses.
  if (auto set = reinterpret_cast<void (*)(void (*)(const char *))>(
          llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
              "mdrtSetStopHandler")))
    set(&stopPart);
  if (control.target == Target::GPU) {
    if (usedDevice >= 0 && usedDevice != execution.device) {
      if (scheduler)
        (void)scheduler->addOccurrence(0, "pre-RA-sched", "default");
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
        if (scheduler)
          (void)scheduler->addOccurrence(0, "pre-RA-sched", "default");
        return unsupported("the GPU runtime does not select devices");
      }
      select(static_cast<int32_t>(execution.device));
      usedDevice = execution.device;
    }
  }
  // Loads the kernels of a GPU; the functions of the driver are known.
  engine->engine->initialize();
  auto function = engine->engine->lookupPacked(engine->program.entry);
  if (scheduler)
    (void)scheduler->addOccurrence(0, "pre-RA-sched", "default");
  if (!function)
    return llvm::make_error<compiler::CompileError>(
        llvm::toString(function.takeError()));
  engine->function = *function;

  engine->masses = system.masses;
  return std::move(engine);
}

llvm::Expected<std::unique_ptr<Simulation>>
Simulation::create(const model::PreparedModel &prepared) {
  const Control &given = prepared.control;
  if (given.minimize)
    return unsupported("a simulation runs dynamics; minimization follows "
                       "later");
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
  simulation->prepared = prepared;
  Control &control = simulation->prepared.control;
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
  system.referencePositions = system.positions;
  for (int k = 0; k != 3; ++k)
    system.inputBox[k] = system.box[k];

  simulation->initial = system;
  auto engine = compileEngine(control, system, prepared.execution);
  if (!engine)
    return engine.takeError();
  simulation->first = std::move(*engine);

  auto output = std::make_unique<Output>();
  output->log.quiet = true;
  output->timestep = control.timestep;
  output->couples = control.getCouplingPeriod() > 0 && !control.isLangevin();
  output->changesCell = control.barostat;
  output->bathKinetic = 0.5 * system.getDegreesOfFreedom() *
                        units::boltzmann * control.temperature;
  output->leastEdge = 2.0 * control.cutoffDistance * units::length;
  output->degreesOfFreedom = system.getDegreesOfFreedom();
  output->periodic = control.periodic;
  output->listReach = control.pairlistDistance * units::length;
  for (int k = 0; k != 3; ++k)
    output->box[k] = system.box[k];
  output->volume = system.box[0] * system.box[1] * system.box[2];
  output->endStep = 0;
  simulation->output = std::move(output);
  return std::move(simulation);
}

llvm::Expected<Simulation::Engine *> Simulation::getEngine() {
  if (!hasRun)
    return first.get();
  if (!continued) {
    // The segments after the first continue the state that the last left,
    // as a run continues its checkpoint: they take its forces and its cell.
    // The program is built from the system at its first step, which was
    // checked, so that its constants and neighbor structures are those of
    // the first program whatever state the run has reached.
    Control control = prepared.control;
    control.continuesSegment = true;
    auto engine = compileEngine(control, initial, prepared.execution);
    if (!engine)
      return engine.takeError();
    continued = std::move(*engine);
  }
  return continued.get();
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
  out.pme = p.pme;
  out.reactionField = p.reactionField;
  out.coulombConstantEnergy = p.coulombConstantEnergy;
  out.coulombConstantVirial = p.coulombConstantVirial;
  out.coulombSelfEnergy = p.coulombSelfEnergy;
  out.ljpme = p.ljpme;
  out.ljpmeSelfEnergy = p.ljpmeSelfEnergy;
  int64_t closing = p.segmentPeriod ? p.closingSteps : 0;
  out.endStep = step + part.outer * (p.segmentPeriod ? part.inner + closing : 1) +
                part.tail + part.plain +
                part.close * (part.closeInner + closing);

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
  if (p.takesForces && forces.size() != 3 * count)
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
    vector(forces, p.force);
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

  stopMessage.clear();
  engine.function(a.pointers.data());
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
    failed = true;
    return simulationError("the simulation failed after step " +
                           llvm::Twine(step) + ": " + failure +
                           "; it keeps the state of step " + llvm::Twine(step));
  }
  forces = std::move(out.finalForces);
  out.finalForces.clear();
  step = out.endStep;
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
  if (count < 0)
    return inputError("run takes a nonnegative number of steps");
  stopRequested = false;
  if (count == 0)
    return 0;
  int64_t period = first->program.segmentPeriod;
  int64_t closing = first->program.closingSteps;
  // The two steps that close a period of the barostat of Trotter type are
  // one: the first gives the strain that the second applies (D92).
  if (period > 0 && closing == 2 && (step + count) % period == period - 1)
    return inputError(
        "the run would end at step " + llvm::Twine(step + count) +
        ", between the two steps that close a period of coupling of the "
        "barostat of Trotter type (every " + llvm::Twine(period) +
        " steps); take one step more or fewer");

  // The steps of a part: about `partSeconds` long, in whole periods, from
  // the time that steps took; 100 steps until that is known.
  int64_t unit = std::max<int64_t>(1, period);
  auto getPartSteps = [&]() -> int64_t {
    double steps = secondsPerStep > 0.0 ? partSeconds / secondsPerStep : 100.0;
    if (!(steps < 1e15))
      steps = 1e15;
    int64_t whole = static_cast<int64_t>(steps) / unit * unit;
    return std::max(unit, whole);
  };
  int64_t taken = 0;
  while (taken < count) {
    if (taken > 0 && (stopRequested || (poll && poll())))
      break;
    int64_t remaining = count - taken;
    Part part;
    part.inner = period > 0 ? period - closing : 0;
    int64_t partSteps = getPartSteps();
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
    if (energy && planned == remaining) {
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

double Simulation::getTime() const {
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
  const auto &row = output->lastEnergies;
  if (hasRun && row.step == step)
    state.energies = SimulationEnergies{row.potential, row.kinetic, row.total,
                                        row.conserved, row.temperature,
                                        row.virial, row.pressure, row.volume};
  busy = false;
  return state;
}
