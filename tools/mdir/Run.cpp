// `mdir run` and `mdir emit`: read a control file, compile the run, and
// execute it or print it.
//
// See docs/driver-m0.md.

#include "Commands.h"

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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>

using namespace mdir;
using namespace mdir::driver;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;

/// The passes that compile the program of a run.
static std::string getPipeline(const Control &control,
                               const Program &program) {
  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  os << "md-check-exchange,md-differentiate,md-expand-truncation,md-inline,"
     << "md-bypass-updates,";
  os << "convert-md-to-md-exec{skin=" << program.skin
     << " width=" << program.neighborWidth << "},";
  os << "md-exec-reuse-neighbors,";
  // Opt-in only (D88): the structures may miss pairs; the run warns.
  if (control.rebuildPeriod > 0)
    os << "md-exec-rebuild-at-interval{interval=" << control.rebuildPeriod
       << "},";
  os << "md-exec-expose-validity,"
     << "md-exec-fuse-loops,md-exec-accumulate-destinations,";
  if (control.fastMath)
    os << "md-exec-simplify-distance,";
  os << "canonicalize,cse,";

  StringRef mode = control.precision == Precision::Single
                       ? "single"
                       : control.precision == Precision::Mixed ? "mixed"
                                                               : "double";
  os << "md-exec-assign-precision{mode=" << mode << "},";
  if (control.fastMath)
    os << "md-exec-approximate,canonicalize,cse,";

  if (control.target == Target::GPU) {
    // Kernels in f32 look their tables up in f32.
    if (control.neighborStructure == NeighborStructure::Groups)
      os << "md-exec-choose-neighbors{kind=groups},";
    os << "md-exec-assign-storage{memory=device"
       << (control.precision == Precision::Double ? "" : " tables=f32")
       << "},md-exec-assign-streams,convert-md-exec-to-gpu{"
       << (control.deterministic ? "deterministic=true " : "")
       << (control.fastMath ? "" : "contract=false") << "},"
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
     << "convert-func-to-llvm,convert-cf-to-llvm,convert-ub-to-llvm,";
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
  llvm::errs() << "mdir: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

/// Reports something that the run goes on with but that its user should
/// know, on the standard error.
static void warn(const llvm::Twine &message) {
  llvm::errs() << "mdir: warning: " << message << "\n";
}

static int fail(const llvm::Twine &message) {
  llvm::errs() << "mdir: " << message << "\n";
  return 1;
}

/// Warns if the cell of `checkpoint`, which the run takes, differs from that
/// of the input. Always, on the standard error, so that it shows when the
/// log goes to a file.
static void warnAboutCell(const Checkpoint &checkpoint, const System &system,
                          StringRef path) {
  bool same = true;
  for (int i = 0; i != 3; ++i)
    same &= checkpoint.box[i] == system.box[i];
  if (same)
    return;
  warn(llvm::formatv("the cell of '{0}', {1:F4} {2:F4} {3:F4} Å, differs "
                     "from that of the input, {4:F4} {5:F4} {6:F4} Å; the "
                     "run takes that of the checkpoint",
                     path, checkpoint.box[0] / units::length,
                     checkpoint.box[1] / units::length,
                     checkpoint.box[2] / units::length,
                     system.box[0] / units::length,
                     system.box[1] / units::length,
                     system.box[2] / units::length)
           .str());
}

/// A function of this program, whose address tells where the program is.
static void anchor() {}

int mdir::tool::runControl(StringRef controlFile, Emit emit,
                           const char *argv0) {
  //===--------------------------------------------------------------------===//
  // Read
  //===--------------------------------------------------------------------===//

  auto control = readControl(controlFile);
  if (!control)
    return fail(control.takeError());
  // The policy of a fixed interval of rebuilds is opt-in and not a
  // default: the structures are not tested between builds and may leave out
  // pairs within the cutoff, whose forces are then missing (D88). Warn at
  // the start, always, on the standard error.
  if (control->rebuildPeriod > 0)
    warn(llvm::formatv(
        "'rebuild_interval = {0}': the neighbor structures are rebuilt every "
        "{0} steps and not tested in between; they may miss pairs within "
        "the cutoff. This is not a default of MDIR; the run counts the "
        "rebuilds that found a structure no longer valid",
        control->rebuildPeriod));
  auto system = readSystem(*control);
  if (!system)
    return fail(system.takeError());

  system->referencePositions = system->positions;
  bool writesCheckpoints = control->checkpointPeriod > 0;
  bool isRestart = !control->restartInput.empty();
  if ((writesCheckpoints || isRestart) && !hasCheckpointSupport())
    return fail("this build of MDIR has no HDF5, which checkpoints need");

  StringRef integrator =
      control->minimize ? "MINIMIZATION"
      : control->integrator == Integrator::Leapfrog ? "LEAPFROG"
                                                   : "VELOCITY_VERLET";
  double velocityOffset =
      control->integrator == Integrator::Leapfrog ? -0.5 : 0.0;

  // A run that continues an earlier one takes its state from the
  // checkpoint. The file of the positions gives the types.
  int64_t firstStep = 0;
  double firstTime = 0.0;
  std::vector<double> forces;
  std::optional<Checkpoint> checkpoint;
  bool fromPositions = false;
  if (isRestart) {
    auto read = readCheckpoint(control->restartInput);
    if (!read)
      return fail(read.takeError());
    checkpoint = std::move(*read);
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
    // A minimization takes the positions and the cell of any checkpoint,
    // and a run of dynamics those of a minimization, and begins anew.
    if (control->minimize || checkpoint->integrator == "MINIMIZATION") {
      system->positions = checkpoint->positions;
      warnAboutCell(*checkpoint, *system, path);
      for (int i = 0; i != 3; ++i)
        system->box[i] = checkpoint->box[i];
      std::fprintf(stdout, "MDIR: begins at the positions of '%s'\n",
                   path.c_str());
      isRestart = false;
      fromPositions = true;
    }
  }
  if (isRestart) {
    const std::string &path = control->restartInput;
    if (checkpoint->integrator != integrator)
      return fail("'" + path + "' was written with the integrator " +
                  checkpoint->integrator + ", and the run uses " +
                  integrator + "; the velocities of the two are not of the "
                  "same time");
    // With a barostat the cell of the checkpoint is where the run left it;
    // otherwise it is that of the input.
    // The cell is where the run that wrote the checkpoint left it, which a
    // barostat may have changed; the input keeps its own for what depends
    // on it (the grid of PME, the reference of restraints).
    warnAboutCell(*checkpoint, *system, path);
    for (int i = 0; i != 3; ++i) {
      system->inputBox[i] = system->box[i];
      system->box[i] = checkpoint->box[i];
    }
    if (checkpoint->forces.empty())
      return fail("'" + path + "' holds no forces, which a step begins "
                  "with");

    system->positions = checkpoint->positions;
    system->velocities = checkpoint->velocities;
    forces = checkpoint->forces;
    firstStep = checkpoint->step;
    firstTime = checkpoint->time;
  } else if (control->minimize) {
    system->velocities.assign(3 * system->getNumParticles(), 0.0);
  } else if (!system->givenVelocities || fromPositions) {
    assignVelocities(*control, *system);
  }
  // The input of the build: a run that begins at the positions of a
  // checkpoint does not continue it.
  if (fromPositions)
    control->restartInput.clear();

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
                "of mdir. Please report it with the directory that `mdir "
                "bug-report " + controlFile + "` writes");

  // Passes on functions are nested where they occur, as on the command
  // line of mlir-opt.
  mlir::PassManager manager(&context,
                            mlir::ModuleOp::getOperationName(),
                            mlir::PassManager::Nesting::Implicit);
  std::string pipeline = getPipeline(*control, *program);
  // MDIR_PIPELINE replaces the pipeline, to try another order of passes or
  // to stop part of the way.
  if (const char *replaced = std::getenv("MDIR_PIPELINE"))
    pipeline = replaced;
  if (emit == Emit::Pipeline) {
    llvm::outs() << pipeline << "\n";
    return 0;
  }
  if (mlir::failed(mlir::parsePassPipeline(pipeline, manager, llvm::errs())))
    return fail("cannot set up the passes");
  // With MDIR_PRINT_AFTER set to the name of a pass of the pipeline, the
  // module is printed to the standard error after each run of that pass,
  // to see what the kernels are.
  if (const char *after = std::getenv("MDIR_PRINT_AFTER")) {
    context.disableMultithreading();
    std::string name = after;
    manager.enableIRPrinting(
        /*shouldPrintBeforePass=*/nullptr,
        [name](mlir::Pass *pass, mlir::Operation *) {
          return pass->getArgument() == name;
        },
        /*printModuleScope=*/true, /*printAfterOnlyOnChange=*/false);
  }
  // If a pass fails or crashes, the module that it began with and the
  // pipeline are written where MDIR_REPRODUCER says, so that `mdir-opt
  // --run-reproducer` repeats the failure.
  std::string reproducer = "mdir-reproducer.mlir";
  if (const char *path = std::getenv("MDIR_REPRODUCER"))
    reproducer = path;
  manager.enableCrashReproducerGeneration(reproducer);
  if (mlir::failed(manager.run(*module)))
    return fail("cannot compile the run; this is a defect of mdir. The "
                "input of the passes is in '" + reproducer + "'. Please "
                "report it with the directory that `mdir bug-report " +
                controlFile + "` writes");

  if (emit == Emit::Lowered) {
    module->print(llvm::outs());
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Load
  //===--------------------------------------------------------------------===//

  // The runtime is next to the driver: <prefix>/bin and <prefix>/lib.
  std::string executable =
      llvm::sys::fs::getMainExecutable(argv0, (void *)(intptr_t)&anchor);
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
    add("_mlir_ciface_mdrtWriteTerms", (void *)&_mlir_ciface_mdrtWriteTerms);
    add("_mlir_ciface_mdrtAddBath", (void *)&_mlir_ciface_mdrtAddBath);
    add("_mlir_ciface_mdrtSetBox", (void *)&_mlir_ciface_mdrtSetBox);
    add("_mlir_ciface_mdrtWriteMinimization",
        (void *)&_mlir_ciface_mdrtWriteMinimization);
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
                "mdir");
  Buffer<2> given(program->takesForces ? forces
                                       : std::vector<double>(3 * count),
                  program->force, count);
  Buffer<1> masses(system->masses, program->mass, count);
  // The numbers of the particles: their places in the files of the run.
  std::vector<int32_t> numbers(count);
  for (size_t i = 0; i != count; ++i)
    numbers[i] = static_cast<int32_t>(i);
  StridedMemRefType<int32_t, 1> identities;
  identities.basePtr = identities.data = numbers.data();
  identities.offset = 0;
  identities.sizes[0] = count;
  identities.strides[0] = 1;
  std::vector<std::unique_ptr<Buffer<1>>> fields;
  std::vector<std::vector<int32_t>> integerValues;
  std::vector<std::unique_ptr<StridedMemRefType<int32_t, 1>>> integerFields;
  for (const Program::Field &field : program->fields) {
    if (!field.isInteger) {
      fields.push_back(std::make_unique<Buffer<1>>(field.values,
                                                   program->parameter, count));
      continue;
    }
    integerValues.emplace_back(field.values.begin(), field.values.end());
    auto descriptor = std::make_unique<StridedMemRefType<int32_t, 1>>();
    descriptor->basePtr = descriptor->data = integerValues.back().data();
    descriptor->offset = 0;
    descriptor->sizes[0] = count;
    descriptor->strides[0] = 1;
    integerFields.push_back(std::move(descriptor));
  }
  // The tables of pairs of types, n by n, in f64.
  std::vector<std::unique_ptr<StridedMemRefType<double, 2>>> tables;
  for (const Program::Table &table : program->tables) {
    auto descriptor = std::make_unique<StridedMemRefType<double, 2>>();
    descriptor->basePtr = descriptor->data =
        const_cast<double *>(table.values.data());
    descriptor->offset = 0;
    descriptor->sizes[0] = table.count;
    descriptor->sizes[1] = table.getColumns();
    descriptor->strides[0] = table.getColumns();
    descriptor->strides[1] = 1;
    tables.push_back(std::move(descriptor));
  }

  double box[3] = {system->box[0], system->box[1], system->box[2]};
  double timestep = control->timestep;
  SmallVector<void *> arguments;
  positions.addTo(arguments);
  velocities.addTo(arguments);
  if (program->takesForces)
    given.addTo(arguments);
  masses.addTo(arguments);
  // The tuples of a topology: their members, and their fields.
  std::vector<std::unique_ptr<StridedMemRefType<int32_t, 2>>> memberBuffers;
  std::vector<std::unique_ptr<StridedMemRefType<double, 1>>> tupleFields;
  for (const Program::TupleSet &set : program->tupleSets) {
    auto members = std::make_unique<StridedMemRefType<int32_t, 2>>();
    members->basePtr = members->data =
        const_cast<int32_t *>(set.members.data());
    members->offset = 0;
    members->sizes[0] = set.size();
    members->sizes[1] = set.arity;
    members->strides[0] = set.arity;
    members->strides[1] = 1;
    memberBuffers.push_back(std::move(members));
    for (const Program::Field &field : set.fields) {
      auto values = std::make_unique<StridedMemRefType<double, 1>>();
      values->basePtr = values->data =
          const_cast<double *>(field.values.data());
      values->offset = 0;
      values->sizes[0] = set.size();
      values->strides[0] = 1;
      tupleFields.push_back(std::move(values));
    }
  }

  // The fields in the order of the program, then the tables.
  size_t realField = 0, integerField = 0;
  for (const Program::Field &field : program->fields) {
    if (!field.isInteger) {
      fields[realField++]->addTo(arguments);
      continue;
    }
    StridedMemRefType<int32_t, 1> &descriptor = *integerFields[integerField++];
    arguments.push_back(&descriptor.basePtr);
    arguments.push_back(&descriptor.data);
    arguments.push_back(&descriptor.offset);
    arguments.push_back(&descriptor.sizes[0]);
    arguments.push_back(&descriptor.strides[0]);
  }
  for (auto &table : tables) {
    arguments.push_back(&table->basePtr);
    arguments.push_back(&table->data);
    arguments.push_back(&table->offset);
    arguments.push_back(&table->sizes[0]);
    arguments.push_back(&table->sizes[1]);
    arguments.push_back(&table->strides[0]);
    arguments.push_back(&table->strides[1]);
  }
  size_t tupleField = 0;
  for (auto [index, set] : llvm::enumerate(program->tupleSets)) {
    StridedMemRefType<int32_t, 2> &members = *memberBuffers[index];
    arguments.push_back(&members.basePtr);
    arguments.push_back(&members.data);
    arguments.push_back(&members.offset);
    arguments.push_back(&members.sizes[0]);
    arguments.push_back(&members.sizes[1]);
    arguments.push_back(&members.strides[0]);
    arguments.push_back(&members.strides[1]);
    for (size_t k = 0, e = set.fields.size(); k != e; ++k) {
      StridedMemRefType<double, 1> &values = *tupleFields[tupleField++];
      arguments.push_back(&values.basePtr);
      arguments.push_back(&values.data);
      arguments.push_back(&values.offset);
      arguments.push_back(&values.sizes[0]);
      arguments.push_back(&values.strides[0]);
    }
  }
  arguments.push_back(&identities.basePtr);
  arguments.push_back(&identities.data);
  arguments.push_back(&identities.offset);
  arguments.push_back(&identities.sizes[0]);
  arguments.push_back(&identities.strides[0]);
  for (double &edge : box)
    arguments.push_back(&edge);
  arguments.push_back(&timestep);
  arguments.push_back(&firstStep);

  Output output;
  output.state = program->state;
  output.force = program->force;
  output.firstStep = firstStep;
  output.energyPeriod = control->energyPeriod;
  output.firstTime = firstTime;
  output.timestep = control->timestep;
  output.couples = control->getCouplingPeriod() > 0;
  output.changesCell = control->barostat;
  output.minimizes = control->minimize;
  output.leastEdge = 2.0 * control->cutoffDistance * units::length;
  output.degreesOfFreedom = system->getDegreesOfFreedom();
  output.volume = system->box[0] * system->box[1] * system->box[2];
  output.firstVolume = output.volume;
  for (int k = 0; k != 3; ++k)
    output.box[k] = system->box[k];
  output.dispersionEnergy = program->dispersionEnergy;
  output.dispersionVirial = program->dispersionVirial;
  output.pme = program->pme;
  output.pmeConstantEnergy = program->pmeConstantEnergy;
  output.pmeConstantVirial = program->pmeConstantVirial;
  output.pmeSelfEnergy = program->pmeSelfEnergy;
  output.system = &*system;
  if (control->framePeriod > 0) {
    // The cell in Å, from the system: a topology gives it with the
    // coordinates, not the control file.
    double cell[3];
    for (int k = 0; k != 3; ++k)
      cell[k] = system->box[k] / units::length;
    if (llvm::Error error = output.trajectory.open(
            control->dcdFile, count, firstStep + control->framePeriod,
            control->framePeriod, control->timestep, cell))
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

  if (control->minimize)
    std::fprintf(output.log,
                 "MDIR: %zu particles, %lld steps of steepest descent\n",
                 count, static_cast<long long>(control->numSteps));
  else
    std::fprintf(output.log, "MDIR: %zu particles, %lld steps of %g ps\n",
                 count, static_cast<long long>(control->numSteps),
                 control->timestep);
  if (control->rebuildPeriod > 0)
    std::fprintf(output.log,
                 "MDIR: warning: the neighbor structures are rebuilt every "
                 "%lld steps and not tested in between (rebuild_interval, "
                 "opt-in); they may miss pairs within the cutoff\n",
                 static_cast<long long>(control->rebuildPeriod));
  if (isRestart)
    std::fprintf(output.log, "MDIR: continues after step %lld, from '%s'\n",
                 static_cast<long long>(firstStep),
                 control->restartInput.c_str());
  else if (system->givenVelocities)
    std::fprintf(output.log,
                 "MDIR: the velocities are those of the file of "
                 "coordinates\n");
  std::fprintf(output.log, "MDIR: compiled in %.2f s\n", compileTime);
  writeLogHeader(output);

  begin = std::chrono::steady_clock::now();
  (*function)(arguments.data());
  double runTime = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - begin)
                       .count();
  output.trajectory.close();

  std::fprintf(output.log, "MDIR: ran in %.2f s", runTime);
  if (!control->minimize && control->numSteps > 0 && runTime > 0.0) {
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
  // At a fixed interval, the builds that found a structure no longer valid:
  // pairs within the cutoff may have been missed before each (D88).
  if (control->rebuildPeriod > 0) {
    if (auto count = (*engine)->lookup("mdrtGetLateBuildCount")) {
      int64_t late = reinterpret_cast<int64_t (*)()>(*count)();
      std::fprintf(output.log,
                   "MDIR: %lld rebuilds found a neighbor structure no "
                   "longer valid\n",
                   static_cast<long long>(late));
      if (late > 0)
        warn(llvm::formatv(
            "{0} rebuilds at the interval of {1} steps found a neighbor "
            "structure no longer valid: pairs within the cutoff may have "
            "been missed; use a shorter 'rebuild_interval', a longer "
            "'pairlist_distance', or none",
            late, control->rebuildPeriod));
    } else {
      llvm::consumeError(count.takeError());
    }
  }
  // The momentum of the state at the end, which the removal of the motion
  // of the center of mass keeps at 0.
  double momentum[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0, e = system->getNumParticles(); i != e; ++i)
    for (int c = 0; c != 3; ++c)
      momentum[c] += system->masses[i] * system->velocities[3 * i + c];
  std::fprintf(output.log,
               "MDIR: the momentum at the end is %.3e amu nm/ps\n",
               std::sqrt(momentum[0] * momentum[0] +
                         momentum[1] * momentum[1] +
                         momentum[2] * momentum[2]));
  if (control->minimize) {
    if (output.hasEnergies)
      std::fprintf(output.log,
                   "MDIR: the potential energy went from %.4f to %.4f "
                   "kcal/mol\n",
                   output.firstTotal / units::energy,
                   output.lastTotal / units::energy);
    return 0;
  }
  if (output.hasEnergies && output.firstTotal != 0.0)
    std::fprintf(output.log,
                 "MDIR: the %s energy changed by %.3e of its value\n",
                 output.couples ? "conserved" : "total",
                 std::fabs((output.lastTotal - output.firstTotal) /
                           output.firstTotal));
  return 0;
}
