// `mdir run` and `mdir emit`: read a control file, compile the run, and
// execute it or print it.
//
// See docs/driver.md.

#include "BuildInfo.h"
#include "Commands.h"
#include "DistributionLog.h"
#include "Manifest.h"
#include "TowersShape.h"

#include "mdir/Compiler/Compile.h"
#include "mdir/Compiler/JITEngine.h"
#include "mdir/Compiler/ProfileMap.h"
#include "mdir/Compiler/Skins.h"
#include "mdir/Conversion/AxisFFTEmitter.h"
#include "mdir/Conversion/GPUToBinary.h"
#include "mdir/Conversion/MDAlgLowering.h"
#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MDDist/Grid.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDExec/Transforms/Passes.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Driver/Builder.h"
#include "mdir/Driver/Checkpoint.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/Fingerprint.h"
#include "mdir/Driver/H5MD.h"
#include "mdir/Driver/ModelTerm.h"
#include "mdir/Driver/Output.h"
#include "mdir/Driver/Prolate.h"
#include "mdir/Driver/RigidUnits.h"
#include "mdir/Driver/System.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <unistd.h>

using namespace mdir;
using namespace mdir::driver;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;

/// The passes that compile the program of a run.
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
static void warnAboutCell(const Control &control, const Checkpoint &checkpoint,
                          const System &system,
                          StringRef path) {
  bool same = true;
  for (int i = 0; i != 3; ++i)
    same &= periodicComponent(control, checkpoint.box[i]) == system.box[i] &&
            periodicComponent(control, checkpoint.tilt[i]) == system.tilt[i];
  if (same)
    return;
  warn(llvm::formatv("the cell of '{0}', {1:F4} {2:F4} {3:F4} Ang, differs "
                     "from that of the input, {4:F4} {5:F4} {6:F4} Ang; the "
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

double mdir::tool::parseWalltime(StringRef text) {
  text = text.trim();
  if (!text.contains(':')) {
    double hours;
    if (text.getAsDouble(hours) || !(hours > 0.0))
      return -1.0;
    return 3600.0 * hours;
  }
  SmallVector<StringRef, 3> parts;
  text.split(parts, ':');
  if (parts.size() > 3)
    return -1.0;
  double seconds = 0.0;
  for (auto [index, part] : llvm::enumerate(parts)) {
    unsigned value;
    if (part.getAsInteger(10, value) || (index != 0 && value >= 60))
      return -1.0;
    seconds += value * (index == 0 ? 3600.0 : index == 1 ? 60.0 : 1.0);
  }
  return seconds;
}

/// Asks the run to stop at its next checkpoint (D131). The handler is reset
/// when it runs, so that a second signal of the same kind ends the run at
/// once. Only what is safe in a handler: a flag and write(2).
static void requestStop(int signal) {
  driver::stopSignal = signal;
  static const char message[] =
      "mdir: stopping at the next checkpoint; signal again to stop now\n";
  ssize_t written = ::write(STDERR_FILENO, message, sizeof(message) - 1);
  (void)written;
}

/// Observe the resolved structure kinds immediately before lowering (D168).
/// A requested GROUPS structure may stay a matrix if its loops require one.
class ManifestNeighbors : public mlir::PassInstrumentation {
public:
  explicit ManifestNeighbors(std::set<std::string> &kinds) : kinds(kinds) {}
  void runBeforePass(mlir::Pass *pass, mlir::Operation *operation) override {
    if (pass->getArgument() != "convert-md-exec-to-gpu" &&
        pass->getArgument() != "convert-md-exec-to-loops")
      return;
    if (operation->hasAttr("mdir.towers_grid")) {
      kinds.insert("towers");
      return;
    }
    auto add = [&](md_exec::NeighborKind kind) {
      kinds.insert(kind == md_exec::NeighborKind::Groups ? "groups" : "matrix");
    };
    operation->walk([&](md_exec::EmptyNeighborsOp op) { add(op.getKind()); });
    operation->walk([&](md_exec::BuildNeighborsOp op) { add(op.getKind()); });
  }
private:
  std::set<std::string> &kinds;
};

/// The policy that each md_alg op with a determinism aspect was lowered
/// with (D408), taken off the module after the passes, so that the
/// module is what it was without the record: the entries for the manifest
/// and the line of the log, which gives each distinct policy once with the
/// count of its ops.
static std::pair<llvm::json::Array, std::string>
takeAlgPolicy(mlir::ModuleOp module) {
  llvm::json::Array entries;
  auto recorded =
      module->getAttrOfType<mlir::ArrayAttr>(kernels::kPolicyAttrName);
  if (!recorded)
    return {std::move(entries), std::string()};
  module->removeAttr(kernels::kPolicyAttrName);
  static const char *const fields[] = {"determinism", "order", "strategy",
                                       "launch", "scan", "scan_launch",
                                       "sort", "sort_launch", "block"};
  SmallVector<std::pair<std::string, unsigned>> distinct;
  for (mlir::Attribute attribute : recorded) {
    auto entry = mlir::cast<mlir::DictionaryAttr>(attribute);
    llvm::json::Object object;
    for (mlir::NamedAttribute field : entry)
      object[field.getName().getValue()] =
          mlir::cast<mlir::StringAttr>(field.getValue()).getValue().str();
    std::string text =
        "md_alg." +
        mlir::cast<mlir::StringAttr>(entry.get("op")).getValue().str() + "{";
    bool first = true;
    for (const char *name : fields)
      if (auto value = entry.getAs<mlir::StringAttr>(name)) {
        text += (first ? "" : " ") + std::string(name) + "=" +
                value.getValue().str();
        first = false;
      }
    text += "}";
    auto *found = llvm::find_if(
        distinct, [&](const auto &seen) { return seen.first == text; });
    if (found == distinct.end())
      distinct.push_back({text, 1});
    else
      ++found->second;
    entries.push_back(std::move(object));
  }
  std::string line = llvm::formatv("MDIR: md_alg policy: {0} ops: ",
                                   recorded.size());
  llvm::interleave(
      distinct,
      [&](const auto &policy) {
        line += policy.first + " x" + std::to_string(policy.second);
      },
      [&] { line += "; "; });
  return {std::move(entries), line + "\n"};
}

namespace mdir::tool {
struct CompiledRun {
  std::unique_ptr<mlir::ExecutionEngine> engine;
  void (*entry)(void **) = nullptr;
  std::unique_ptr<compiler::JITEngine> trialEngine;
  llvm::Expected<void *> lookup(StringRef name) const {
    return trialEngine ? trialEngine->lookup(name) : engine->lookup(name);
  }
};

static llvm::Expected<CompiledRun> compileEntry(mlir::ModuleOp module,
                                                 llvm::ArrayRef<std::string> paths,
                                                 const Program &program, bool trial) {
  // The code of the host is scheduled by the fast list scheduler of LLVM.
  // The default one, at each call, follows the chains of the block up
  // through every path of their token factors, which grows exponentially
  // with the calls of one block: the host code of a term over the centers
  // of a dihedral on JAC, a block of 95 calls among 1,400 stores of the
  // arguments of kernels, did not compile in ten minutes; the fast one
  // compiles it in 2.6 s (D150). The kernels were compiled in the pass
  // pipeline, with the default; it is set back once the host code is. A
  // scheduler of the command line is kept. A skin search before restored
  // the option as it found it (#829).
  compiler::FastHostScheduler scheduler(/*keepCommandLine=*/true);
  mlir::ExecutionEngineOptions engineOptions;
  SmallVector<StringRef> sharedLibraries(paths.begin(), paths.end());
  engineOptions.sharedLibPaths = sharedLibraries;
  auto hostMachine = compiler::createHostMachine();
  if (!hostMachine)
    return hostMachine.takeError();
  auto runtimeSymbols = [&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    auto add = [&](StringRef name, void *function) {
      symbols[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(function),
                                 llvm::JITSymbolFlags::Exported};
    };
    add("_mlir_ciface_mdrtWriteEnergies",
        (void *)&_mlir_ciface_mdrtWriteEnergies);
    add("_mlir_ciface_mdrtWriteFrame", (void *)&_mlir_ciface_mdrtWriteFrame);
    add("_mlir_ciface_mdrtStageFrame", (void *)&_mlir_ciface_mdrtStageFrame);
    add("_mlir_ciface_mdrtCheckSpread",
        (void *)&_mlir_ciface_mdrtCheckSpread);
    add("_mlir_ciface_mdrtWriteTerms", (void *)&_mlir_ciface_mdrtWriteTerms);
    add("_mlir_ciface_mdrtWriteVirial", (void *)&_mlir_ciface_mdrtWriteVirial);
    add("_mlir_ciface_mdrtModelEvaluate",
        (void *)&_mlir_ciface_mdrtModelEvaluate);
    add("_mlir_ciface_mdrtModelEvaluateEnergy",
        (void *)&_mlir_ciface_mdrtModelEvaluateEnergy);
    add("_mlir_ciface_mdrtModelEvaluateEdges",
        (void *)&_mlir_ciface_mdrtModelEvaluateEdges);
    add("_mlir_ciface_mdrtModelEvaluateEdgesEnergy",
        (void *)&_mlir_ciface_mdrtModelEvaluateEdgesEnergy);
    add("_mlir_ciface_mdrtAddBath", (void *)&_mlir_ciface_mdrtAddBath);
    add("mdrtNoseHooverFactor", (void *)&mdrtNoseHooverFactor);
    add("mdrtWriteSolvent", (void *)&mdrtWriteSolvent);
    add("_mlir_ciface_mdrtWritePull", (void *)&_mlir_ciface_mdrtWritePull);
    add("_mlir_ciface_mdrtWriteFreeEnergy",
        (void *)&_mlir_ciface_mdrtWriteFreeEnergy);
    add("_mlir_ciface_mdrtWriteObservables",
        (void *)&_mlir_ciface_mdrtWriteObservables);
    add("_mlir_ciface_mdrtScaleMasterEdge",
        (void *)&_mlir_ciface_mdrtScaleMasterEdge);
    add("_mlir_ciface_mdrtSetBox", (void *)&_mlir_ciface_mdrtSetBox);
    add("_mlir_ciface_mdrtSetTilt", (void *)&_mlir_ciface_mdrtSetTilt);
    add("_mlir_ciface_mdrtSetBarostatState",
        (void *)&_mlir_ciface_mdrtSetBarostatState);
    add("_mlir_ciface_mdrtWriteMinimization",
        (void *)&_mlir_ciface_mdrtWriteMinimization);
    add("_mlir_ciface_mdrtCheckMinimization",
        (void *)&_mlir_ciface_mdrtCheckMinimization);
    add("_mlir_ciface_mdrtFinish", (void *)&_mlir_ciface_mdrtFinish);
    add("_mlir_ciface_mdrtWriteCheckpoint",
        program.writesForces
            ? (void *)&_mlir_ciface_mdrtWriteCheckpointWithForces
            : (void *)&_mlir_ciface_mdrtWriteCheckpoint);
    return symbols;
  };

  const auto &triple = (*hostMachine)->getTargetTriple();
  if (trial && triple.isOSBinFormatELF() && (triple.getArch() == llvm::Triple::x86_64 ||
                                             triple.getArch() == llvm::Triple::aarch64)) {
    auto engine = compiler::JITEngine::create(
        module, std::move(*hostMachine), paths, program.entry,
        ("host-opt=" + compiler::hostOptimization()).str());
    if (!engine)
      return engine.takeError();
    if (auto error = (*engine)->registerSymbols(runtimeSymbols))
      return std::move(error);
    if (auto error = (*engine)->initialize())
      return std::move(error);
    auto entry = (*engine)->lookupPacked(program.entry);
    if (!entry)
      return entry.takeError();
    return CompiledRun{nullptr, *entry, std::move(*engine)};
  }

  // LLVM's module pipeline on the host code before the code generator, as
  // a simulation runs it (D269). The options hold a reference to the
  // transformer, which lives until the engine is made.
  llvm::TargetMachine *machine = hostMachine->get();
  std::function<llvm::Error(llvm::Module *)> optimize =
      [machine](llvm::Module *module) {
        compiler::optimizeHostModule(*module, *machine);
        return llvm::Error::success();
      };
  engineOptions.transformer = optimize;
  auto engine = mlir::ExecutionEngine::create(module, engineOptions,
                                              std::move(*hostMachine));
  if (!engine)
    return engine.takeError();

  (*engine)->registerSymbols(runtimeSymbols);

  // Loads the kernels of a GPU. The functions of the driver must be known
  // by now.
  (*engine)->initialize();

  auto function = (*engine)->lookupPacked(program.entry);
  scheduler.restore();
  if (!function)
    return function.takeError();
  return CompiledRun{std::move(*engine), *function};
}

// Both candidates have the same passes up to the graph transformation.
// Retain the input as well: a nonaligned restart can build different loops.
struct TrialLowering {
  mlir::MLIRContext context{compiler::getRegistry(), mlir::MLIRContext::Threading::DISABLED};
  mlir::OwningOpRef<mlir::ModuleOp> input, lowered;
  std::string pipeline;

  TrialLowering() { compiler::shareThreadPool(context); }

  mlir::LogicalResult prepare(mlir::OwningOpRef<mlir::ModuleOp> &module,
                              StringRef passes, bool &reused) {
    // These two attributes are consumed by the graph pass in the suffix.
    auto dt = (*module)->removeAttr("mdir.towers_dt");
    auto start = (*module)->removeAttr("mdir.towers_start_time");
    if (input && passes == pipeline && mlir::OperationEquivalence::isEquivalentTo(
            *module, *input, mlir::OperationEquivalence::IgnoreLocations)) {
      module = lowered->clone();
      reused = true;
    } else {
      mlir::PassManager manager(&context, mlir::ModuleOp::getOperationName(),
                                mlir::PassManager::Nesting::Implicit);
      if (mlir::failed(mlir::parsePassPipeline(passes, manager)))
        return mlir::failure();
      input = module->clone();
      pipeline = passes.str();
      if (mlir::failed(manager.run(*module)))
        return mlir::failure();
      lowered = module->clone();
    }
    if (dt) (*module)->setAttr("mdir.towers_dt", dt);
    if (start) (*module)->setAttr("mdir.towers_start_time", start);
    return mlir::success();
  }
};

struct StepGraphTrial {
  TrialLowering &lowering;
  const TowersSetup &setup;
  StringRef path, restart;
  double us = 0;
  int64_t measured = 0, cold = 0;
  std::string refusal;
  CompiledRun compiled;
  double compileSeconds = 0, instantiateSeconds = 0, warmSeconds = 0, timedSeconds = 0;
  compiler::CompileStats cache;
  bool sharedLowering = false;

  llvm::json::Object timings() const {
    return llvm::json::Object{{"compile_s", compileSeconds}, {"instantiate_s", instantiateSeconds},
            {"warm_s", warmSeconds}, {"timed_s", timedSeconds},
            {"shared_lowering", sharedLowering},
            {"host_hits", cache.hits}, {"host_compiled", cache.compiled},
            {"ptx_hits", cache.gpuPtxHits}, {"ptx_compiled", cache.gpuPtxCompiled},
            {"cubin_hits", cache.gpuCubinHits}, {"cubin_compiled", cache.gpuCubinCompiled}};
  }
};

static int runControlImpl(StringRef controlFile, Emit emit,
                           const char *argv0, const RunOptions &options, StepGraphTrial *trial) {
  auto began = std::chrono::steady_clock::now();

  //===--------------------------------------------------------------------===//
  // Read
  //===--------------------------------------------------------------------===//

  // The refusal of a distributed setup by [distribute], the builder, or the
  // passes of md_dist: the reason goes to the log and the manifest that the
  // control names as well. It may precede the checks of the paths of the
  // ordinary run: an existing diagnostic history is kept, and the inputs,
  // the trajectory, and the checkpoints are not overwritten.
  auto refuse = [&](const Control &c, llvm::Error e) {
    std::string reason = llvm::toString(std::move(e));
    if (emit == Emit::Run && (!c.logFile.empty() || !c.manifestFile.empty())) {
      auto inputs = getManifestInputs(controlFile, c, System{});
      auto diagnosticPath = [&](const std::string &path,
                                const std::string &other) {
        if (path.empty())
          return false;
        std::vector<std::string> reserved{
            other,      c.energyFile,     c.trajectoryFile, c.observablesFile,
            c.pullFile, c.freeEnergyFile, c.profileMapFile, c.restartOutput};
        if (!c.restartOutput.empty()) {
          reserved.push_back(getPreviousCheckpointPath(c.restartOutput));
          reserved.push_back(c.restartOutput + ".partial");
        }
        if (auto error = checkManifestPath(path, inputs, reserved)) {
          warn(llvm::toString(std::move(error)));
          return false;
        }
        return true;
      };
      if (diagnosticPath(c.logFile, c.manifestFile)) {
        // A rejected control may point at an unrelated file. Only append
        // to a recognizable MDIR log or create an unused path.
        const char *mode = "wx";
        if (llvm::sys::fs::exists(c.logFile)) {
          std::ifstream previous(c.logFile);
          char signature[5]{};
          if (previous.read(signature, 4) && StringRef(signature) == "MDIR")
            mode = "a";
        }
        if (FILE *file = std::fopen(c.logFile.c_str(), mode)) {
          std::fprintf(file, "MDIR: distributed setup refused: %s\n",
                       reason.c_str());
          std::fclose(file);
        } else {
          warn("cannot record refusal in log '" + c.logFile +
               "': " + std::strerror(errno));
        }
      }
      Manifest refusal;
      if (diagnosticPath(c.manifestFile, c.logFile))
        if (auto err = refusal.refuse(c.manifestFile, reason))
          warn(llvm::toString(std::move(err)));
    }
    return fail(reason);
  };
  Control rejected;
  auto control = readControl(controlFile, &rejected);
  if (!control)
    return refuse(rejected, control.takeError());
  // What the run writes (D149). Its log goes to the standard output from
  // here, and to the file of [output] as well once that is open.
  Output output;
  if (trial) {
    // An independent invocation owns its state, callbacks and banks. Use the
    // production shape and checkpoint, with no files or checkpoints to write.
    output.log.quiet = true;
    control->stepGraph = trial->path.str();
    control->restartInput = trial->restart.str();
    for (auto *file : {&control->restartOutput, &control->energyFile,
                       &control->trajectoryFile, &control->manifestFile, &control->logFile,
                       &control->profileMapFile, &control->freeEnergyFile, &control->pullFile,
                       &control->observablesFile})
      file->clear();
    control->framePeriod = control->checkpointPeriod = 0;
    auto period = std::min<int64_t>(control->getCouplingPeriod(), 17);
    control->thermostatPeriod = period;
    if (control->comPeriod) control->comPeriod = period;
    // Leave a long sequence of ordinary chunks between reporting sites.
    control->energyPeriod = period * 32;
    control->numSteps = control->energyPeriod * 2;
  }
  Manifest manifest;
  llvm::json::Array manifestInputs;

  // A stop lands on a checkpoint, from which the run continues exactly
  // (D131); a run without checkpoints has nowhere to stop.
  bool stops = control->checkpointPeriod > 0 && !control->minimize;
  if (options.maxWalltime > 0.0 && !stops)
    return fail("--max-walltime stops a run of dynamics at a checkpoint, "
                "and this run writes none ('checkpoint_interval' in "
                "[output])");

  // `mdir run --continue` (D129): the run goes on from its own checkpoint
  // until it has taken `steps` steps from the step it began at, which the
  // checkpoint records. Without a checkpoint it begins, so that the same
  // command line starts a run and continues it until it is complete.
  std::optional<Checkpoint> own;
  int64_t total = control->numSteps;
  if (options.continues) {
    if (control->restartOutput.empty())
      return fail("--continue goes on from the checkpoint of the run, and "
                  "[output] names no 'checkpoint'");
    const std::string &path = control->restartOutput;
    std::string previous = getPreviousCheckpointPath(path);
    if (llvm::sys::fs::exists(path)) {
      auto read = readCheckpoint(path);
      if (!read)
        return fail(llvm::toString(read.takeError()) + "; '" + previous +
                    "' holds the checkpoint before it, if there is one");
      own = std::move(*read);
      int64_t end = own->firstStep + total;
      if (own->step >= end) {
        output.log.print(
                     "MDIR: the run is complete: '%s' holds step %lld, and "
                     "the run began at step %lld and takes %lld steps\n",
                     path.c_str(), static_cast<long long>(own->step),
                     static_cast<long long>(own->firstStep),
                     static_cast<long long>(total));
        return 0;
      }
      // The physics and the coupling of the run are compared with those of
      // the checkpoint once the system is read (D172).
      // What remains must hold whole intervals of each output and of the
      // coupling, which are counted from where the run began.
      int64_t remaining = end - own->step;
      for (auto [name, period] :
           {std::pair<StringRef, int64_t>{"'energy_interval'",
                                          control->energyPeriod},
            {"'trajectory_interval'", control->framePeriod},
            {"'checkpoint_interval'", control->checkpointPeriod},
            {"the interval of coupling", control->getCouplingPeriod()}})
        if (period > 0 && remaining % period != 0)
          return fail(llvm::formatv(
                          "the {0} steps that remain after step {1} of '{2}' "
                          "are not a multiple of {3}, {4}",
                          remaining, own->step, path, name, period)
                          .str());
      control->restartInput = path;
      control->numSteps = remaining;
    } else if (llvm::sys::fs::exists(previous)) {
      return fail("'" + path + "' is missing, but '" + previous + "' is "
                  "there: the run stopped while it replaced its checkpoint. "
                  "Rename '" + previous + "' to '" + path + "' to continue "
                  "from it");
    } else {
      output.log.print("MDIR: no checkpoint '%s' yet; the run begins\n",
                       path.c_str());
    }
  }
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
  for (const auto &[code, message] : system->warnings)
    warn(message);
  // `neighbor_structure = "AUTO"` is resolved for the system before
  // anything is built from it (D261).
  resolveNeighborStructure(*control, *system);

  auto graphRefusal = control->stepGraphRefusal(system->numConstraints);
  if (options.segments) graphRefusal = "GRAPH does not support segmented execution";
  if (control->stepGraph == "GRAPH" && !graphRefusal.empty())
    return fail("step_graph: " + graphRefusal);
  if (control->stepGraph == "SPANS" && control->stepGraphGiven &&
      !control->spansRefusal().empty())
    return fail("step_graph: " + control->spansRefusal());
  InputPaths inputPaths = getManifestInputs(controlFile, *control, *system);
  if (!control->manifestFile.empty() && emit == Emit::Run) {
    auto hashes = hashManifestInputs(inputPaths);
    if (!hashes)
      return fail(hashes.takeError());
    manifestInputs = std::move(*hashes);
  }

  system->referencePositions = system->positions;
  bool writesCheckpoints = control->checkpointPeriod > 0;
  const auto inputBoxRounding = system->boxRounding;
  bool isRestart = !control->restartInput.empty();
  if ((writesCheckpoints || isRestart) && !hasCheckpointSupport())
    return fail("this build of MDIR has no HDF5, which checkpoints need");
  if (control->framePeriod > 0 &&
      control->trajectoryFormat == TrajectoryFormat::H5MD &&
      !hasCheckpointSupport())
    return fail("this build of MDIR has no HDF5, which a trajectory in H5MD "
                "needs");

  // What defines this run, which its checkpoints record and a checkpoint
  // that it takes is compared with (D172).
  Fingerprint fingerprint;
  if (writesCheckpoints || isRestart) {
    auto computed = getRunFingerprint(controlFile, *control, *system);
    if (!computed)
      return fail(computed.takeError());
    fingerprint = std::move(*computed);
  }
  // Notes for the log, which is not open yet.
  std::vector<std::string> notes;
  auto describe = [](const std::vector<FingerprintChange> &changes) {
    std::string text;
    for (const FingerprintChange &change : changes)
      text += "\n  " + change.name + ": " + change.before + " -> " +
              change.after;
    return text;
  };
  // The differences that `--continue --force` overrides (D447):
  // the log notes them, the manifest and the new checkpoint record them.
  std::vector<FingerprintChange> forcedChanges;
  bool forcedRecompute = false;
  if (own) {
    // A run continues as it was: another physics or coupling is another
    // run, which begins from the checkpoint as 'checkpoint' of [input].
    // --force continues it anyway, evaluating its forces and coupling state
    // anew as such a run does.
    std::vector<FingerprintChange> changes =
        compareFingerprints(own->fingerprint, fingerprint, "physics");
    std::vector<FingerprintChange> coupling =
        compareFingerprints(own->fingerprint, fingerprint, "coupling");
    changes.insert(changes.end(), coupling.begin(), coupling.end());
    if (!changes.empty()) {
      if (!options.forced)
        return fail("'" + control->restartInput +
                    "' was written by a run of "
                    "other physics or coupling, which --continue does not "
                    "change; begin a new run from it as 'checkpoint' of "
                    "[input] instead, or override it with --continue "
                    "--force. What differs (checkpoint -> control file):" +
                    describe(changes));
      forcedChanges.insert(forcedChanges.end(), changes.begin(), changes.end());
      forcedRecompute = true;
    }
    // A deterministic distributed run gives the same bits on every grid,
    // so --continue may take another (D447); without
    // [distribute], or with it where the run had none, the sums are of
    // another arithmetic, and the run would not reproduce its uninterrupted
    // course, which a deterministic continuation promises.
    auto isDistribution = [](const FingerprintEntry &entry) {
      return entry.group == "execution" &&
             StringRef(entry.name).starts_with("[distribute]");
    };
    auto distributed = [&](const Fingerprint &entries) {
      return llvm::any_of(entries, isDistribution);
    };
    auto deterministic = [](const Fingerprint &entries) {
      return llvm::any_of(entries, [](const FingerprintEntry &entry) {
        return entry.name == "[execution] deterministic" &&
               entry.value == "true";
      });
    };
    std::vector<FingerprintChange> execution =
        compareFingerprints(own->fingerprint, fingerprint, "execution");
    bool bitwise = deterministic(own->fingerprint) && control->deterministic;
    if (bitwise && distributed(own->fingerprint) != distributed(fingerprint)) {
      if (!options.forced)
        return fail("'" + control->restartInput + "' was written by a " +
                    "deterministic run " +
                    (distributed(own->fingerprint) ? "with" : "without") +
                    " [distribute], whose fixed-point sums --continue keeps; "
                    "continue it " +
                    (distributed(own->fingerprint)
                         ? "with [distribute], on any grid,"
                         : "without [distribute]") +
                    " or begin a new run from it as 'checkpoint' of [input], "
                    "or override it with --continue --force");
      for (const FingerprintChange &change : execution)
        if (StringRef(change.name).starts_with("[distribute]"))
          forcedChanges.push_back(change);
    }
    // Another grid under the deterministic contract keeps the bits.
    std::vector<FingerprintChange> grid, other;
    for (const FingerprintChange &change : execution)
      (bitwise && change.name == "[distribute] domains" &&
               change.before != "(none)" && change.after != "(none)"
           ? grid
           : other)
          .push_back(change);
    if (!grid.empty())
      notes.push_back("the run continues on another grid of domains, whose "
                      "bits the deterministic contract keeps (checkpoint -> "
                      "control file):" +
                      describe(grid));
    if (!other.empty())
      notes.push_back("the run continues with other settings of its "
                      "execution, whose bits follow them (checkpoint -> "
                      "control file):" +
                      describe(other));
    if (!forcedChanges.empty())
      notes.push_back("--force continues the run past the differences of its "
                      "fingerprint that --continue refuses; the checkpoints "
                      "and the manifest record them (checkpoint -> control "
                      "file):" +
                      describe(forcedChanges));
  } else if (options.forced) {
    notes.push_back("--force overrides nothing: there is no checkpoint to "
                    "continue");
  }

  StringRef integrator =
      control->minimize ? "MINIMIZATION"
      : control->integrator == Integrator::Leapfrog ? "LEAPFROG"
      : control->integrator == Integrator::Brownian ? "BROWNIAN"
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
      return fail(llvm::toString(read.takeError()) + "; '" +
                  getPreviousCheckpointPath(control->restartInput) +
                  "' holds the checkpoint before it, if there is one");
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
      warnAboutCell(*control, *checkpoint, *system, path);
      system->setInitialCell(*control, checkpoint->box, checkpoint->tilt);
      output.log.print("MDIR: begins at the positions of '%s'\n",
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
    // The cell is where the run that wrote the checkpoint left it, which a
    // barostat may have changed; the input keeps its own for what depends
    // on it (the grid of PME, the reference of restraints). A run's own
    // barostat moving its cell is no news.
    if (!own)
      warnAboutCell(*control, *checkpoint, *system, path);
    std::copy_n(system->box, 3, system->inputBox);
    system->setInitialCell(*control, checkpoint->box, checkpoint->tilt);
    if (checkpoint->forces.empty())
      return fail("'" + path + "' holds no forces, which a step begins "
                  "with");

    system->positions = checkpoint->positions;
    system->velocities = checkpoint->velocities;
    system->barostatState = checkpoint->barostatState;
    system->thermostatState = checkpoint->thermostatState;
    forces = checkpoint->forces;
    // The checkpoint of another run: its forces, the state of its barostat,
    // and its thermostat chain are those of its own physics and coupling,
    // and are taken only where this run has the same
    // (D172). Otherwise the run evaluates them at its
    // first step, as production after equilibration with restraints must.
    if (!own || forcedRecompute) {
      std::vector<FingerprintChange> physics =
          compareFingerprints(checkpoint->fingerprint, fingerprint, "physics");
      std::vector<FingerprintChange> coupling = compareFingerprints(
          checkpoint->fingerprint, fingerprint, "coupling");
      if (!physics.empty() || !coupling.empty()) {
        std::vector<FingerprintChange> changes = physics;
        changes.insert(changes.end(), coupling.begin(), coupling.end());
        notes.push_back("'" + path + "' was written by a run of other "
                        "physics or coupling; the run evaluates the forces "
                        "and the state of the barostat at its first step "
                        "(checkpoint -> control file):" +
                        describe(changes));
        control->restartRecomputes = true;
        forces.clear();
        system->barostatState.clear();
      }
      bool otherThermostat = llvm::any_of(
          coupling, [](const FingerprintChange &change) {
            return StringRef(change.name).starts_with("[thermostat]") ||
                   StringRef(change.name).starts_with("[ensemble]");
          });
      if (otherThermostat && !system->thermostatState.empty()) {
        notes.push_back("the thermostat differs from that of '" + path +
                        "'; its chain begins at rest");
        system->thermostatState.clear();
      }
    }
    if (llvm::any_of(system->boxRounding, [](double d) { return d != 0; })) {
      control->restartRecomputes = true;
      forces.clear();
      system->barostatState.clear();
      notes.push_back("the checkpoint cell was rounded to binary32; "
                      "the run recomputes its forces and barostat state");
    }
    firstStep = checkpoint->step;
    control->restoredStep = firstStep;
    firstTime = checkpoint->time;
  } else if (control->minimize) {
    system->velocities.assign(3 * system->getNumParticles(), 0.0);
  } else if (!system->givenVelocities || fromPositions) {
    assignVelocities(*control, *system);
  }
  StringRef previousTowersHeight =
      isRestart ? StringRef(checkpoint->towersCellHeight) : StringRef();
  bool usesTowers = control->neighborStructure == NeighborStructure::Towers;
  llvm::json::Object resolvedChoices;
  llvm::json::Value towersShapeReport = nullptr;
  std::string towersCellHeight;
  if (trial) {
    control->towersSetup = trial->setup;
    towersCellHeight = trial->setup.launch.cellHeight;
  } else if (control->towersShapeGiven || usesTowers) {
    auto shape =
        checkTowersShape(*control, *system, /*setup=*/true,
                         usesTowers ? previousTowersHeight : StringRef(),
                         usesTowers ? &control->towersSetup : nullptr);
    if (!shape)
      return fail(shape.takeError());
    if (usesTowers)
      towersCellHeight =
          shape->getObject("effective")->getString("cell_height")->str();
    if (usesTowers) resolvedChoices["towers_shape"] = llvm::json::Object(*shape->getObject("effective"));
    towersShapeReport = std::move(*shape);
  }
  auto graphChoice = initialStepGraphChoice(control->stepGraph, graphRefusal,
                                            system->getNumParticles(),
                                            control->stepGraphGiven,
                                            control->spansRefusal());
  control->stepGraph = graphChoice.getString("value")->str();
  if (control->stepGraph == "AUTO") {
    auto start = std::chrono::steady_clock::now();
    auto recordElapsed = llvm::scope_exit([&] {
      graphChoice["setup_ms"] = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
    });
    TrialLowering lowering;
    StepGraphTrial launch{lowering, control->towersSetup, "LAUNCH", control->restartInput};
    StepGraphTrial graph{lowering, control->towersSetup, "GRAPH", control->restartInput};
    constexpr int rounds = 4;
    constexpr int noiseFactor = 2;
    llvm::json::Array differences;
    double launchSum = 0, graphSum = 0;
    int64_t launchChunks = 0, graphChunks = 0, cold = 0;
    for (int i = 0; i < rounds; ++i) {
      auto *first = i % 2 ? &graph : &launch;
      auto *second = i % 2 ? &launch : &graph;
      for (auto *candidate : {first, second}) {
        if (runControlImpl(controlFile, Emit::Run, argv0, {}, candidate))
          return fail("step_graph setup trial could not measure the ordinary steps");
        if (!candidate->refusal.empty())
          break;
      }
      if (!launch.refusal.empty() || !graph.refusal.empty())
        break;
      launchSum += launch.us;
      graphSum += graph.us;
      differences.push_back(launch.us - graph.us);
      launchChunks += launch.measured;
      graphChunks += graph.measured;
      cold += graph.cold;
    }
    graphChoice["launch_setup"] = launch.timings();
    graphChoice["graph_setup"] = graph.timings();
    control->stepGraph = "LAUNCH";
    if (!launch.refusal.empty() || !graph.refusal.empty()) {
      graphChoice["reason"] = launch.refusal.empty() ? graph.refusal : launch.refusal;
    } else {
      double gain = (launchSum - graphSum) / rounds;
      double squared = 0;
      for (const auto &difference : differences) {
        double residual = *difference.getAsNumber() - gain;
        squared += residual * residual;
      }
      double noise = std::sqrt(squared / (rounds - 1));
      bool selectGraph = gain > noiseFactor * noise;
      control->stepGraph = selectGraph ? "GRAPH" : "LAUNCH";
      graphChoice["reason"] = selectGraph ? "GRAPH gain exceeds noise threshold"
                                          : "within noise";
      graphChoice["launch_us"] = launchSum / rounds;
      graphChoice["graph_us"] = graphSum / rounds;
      graphChoice["paired_gain_us"] = std::move(differences);
      graphChoice["margin_us"] = std::abs(gain);
      graphChoice["margin_percent"] = 100 * std::abs(gain) * rounds / std::max(launchSum, graphSum);
      graphChoice["launch_measured_chunks"] = launchChunks;
      graphChoice["graph_measured_chunks"] = graphChunks;
      graphChoice["graph_construction_chunks_excluded"] = cold;
      graphChoice["noise_us"] = noise;
      graphChoice["noise_factor"] = noiseFactor;
      graphChoice["below_noise"] = !selectGraph;
    }
  }
  graphChoice["value"] = control->stepGraph;
  addSpansChoice(graphChoice, control->target == Target::GPU, trial != nullptr);
  bool spansOn = *graphChoice.getBoolean("spans");
  std::string graphReport = llvm::formatv("{0}", llvm::json::Value(llvm::json::Object(graphChoice))).str();
  // The stored forces use the writer's strict pair predicate. Rebuild
  // them at the restored positions if the requested traversal or its
  // cell origins have changed.
  if (isRestart && previousTowersHeight != towersCellHeight) {
    control->restartRecomputes = true;
    forces.clear();
    system->barostatState.clear();
    notes.push_back("the tower geometry differs from the checkpoint; "
                    "the run evaluates the forces again at its first step");
  }
  // The input of the build: a run that begins at the positions of a
  // checkpoint does not continue it.
  if (fromPositions)
    control->restartInput.clear();
  // `rigid_units` is resolved for the system at the positions and the cell
  // that the run begins with, the input's or the checkpoint's
  // (D424).
  if (llvm::Error error = resolveRigidUnits(*control, *system))
    return fail(std::move(error));
  for (const auto &[code, message] : control->rigidUnitsRoute.warnings)
    warn(message);
  // The reaches that are "AUTO" (D281): the defaults, the cache, or
  // candidates measured from the state the run begins with, before its own
  // program is built. What the search printed goes to the log once its
  // header is written.
  std::string skinLines;
  compiler::SkinChoice skinChoice;
  {
    llvm::raw_string_ostream lines(skinLines);
    skinChoice = compiler::chooseSkins(*control, *system,
                                       /*search=*/emit == Emit::Run, &lines);
  }
  // The step that the run began at, from which `steps` counts, and its
  // part: a run that begins from the checkpoint of another begins at its
  // step, whose counter and time it continues (D129).
  int64_t runFirstStep = own ? own->firstStep : firstStep;
  int64_t part = own ? own->part + 1 : 1;

  // The files of the outputs (D149): those of the control file, or those
  // of the part that the checkpoint of a continued run records, or with
  // --no-append those of its own part.
  int64_t outputsPart = !own ? 0 : options.appends ? own->outputsPart : part;
  auto getOutputPath = [&](const std::string &name) {
    return outputsPart > 0 ? getPartPath(name, outputsPart) : name;
  };

  if (llvm::any_of(control->functions, [](const auto &function) {
        return !function.valuesFile.empty();
      })) {
    std::vector<std::string> outputs;
    for (const auto &path : {control->logFile, control->energyFile,
                             control->pullFile, control->observablesFile,
                             control->trajectoryFile, control->manifestFile})
      if (!path.empty())
        outputs.push_back(getOutputPath(path));
    if (!control->restartOutput.empty()) {
      outputs.push_back(control->restartOutput);
      outputs.push_back(getPreviousCheckpointPath(control->restartOutput));
    }
    if (llvm::Error error = checkTabulatedInputs(inputPaths, outputs))
      return fail(std::move(error));
  }

  if (!control->manifestFile.empty()) {
    std::vector<std::string> otherOutputs;
    for (const auto &path : {control->logFile, control->energyFile,
                             control->pullFile, control->observablesFile,
                             control->trajectoryFile})
      if (!path.empty())
        otherOutputs.push_back(getOutputPath(path));
    if (!control->restartOutput.empty()) {
      otherOutputs.push_back(control->restartOutput);
      otherOutputs.push_back(getPreviousCheckpointPath(control->restartOutput));
    }
    if (llvm::Error error = checkManifestPath(getOutputPath(control->manifestFile),
                                              inputPaths, otherOutputs))
      return fail(std::move(error));
  }

  // The program of a model term asks the plugin for its cutoff
  // (D277); the plugin is beside the runtime libraries,
  // <prefix>/lib.
  {
    std::string executable =
        llvm::sys::fs::getMainExecutable(argv0, (void *)(intptr_t)&anchor);
    llvm::SmallString<256> directory(llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(executable)));
    llvm::sys::path::append(directory, "lib");
    setPluginDirectory(directory);
  }
  // The program of segments of a Python simulation of the same model,
  // which is built with the control of a simulation and lowered by it
  // (#568, the equivalence check of the builder).
  if (options.segments)
    *control = compiler::getSegmentsControl(*control);
  if (llvm::Error error = checkDistributedSystem(*control, *system))
    return refuse(*control, std::move(error));
  auto program = buildProgram(*control, *system);
  if (!program)
    return fail(program.takeError());
  if (emit == Emit::Module) {
    llvm::outs() << printModuleForReading(*program);
    // The profile map is of the lowered program (D398).
    if (options.profileMap.empty())
      return 0;
  }

  //===--------------------------------------------------------------------===//
  // Compile
  //===--------------------------------------------------------------------===//

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // The passes are registered once a process: a search of the skins may
  // have lowered programs before (D281).
  // The threads of the process's pool, at most MDIR_COMPILE_THREADS, as a
  // Python simulation lowers (D211); the GPU modules are serialized on them
  // (D214).
  std::unique_ptr<mlir::MLIRContext> productionContext;
  if (!trial) {
    productionContext = std::make_unique<mlir::MLIRContext>(
        compiler::getRegistry(), mlir::MLIRContext::Threading::DISABLED);
    compiler::shareThreadPool(*productionContext);
  }
  auto &context = trial ? trial->lowering.context : *productionContext;

  // The kernels for a GPU take their math functions from libdevice: that
  // of the toolkit the environment names, else the copy installed with
  // mdir, else that of the toolkit of the build.
  if (control->target == Target::GPU) {
    std::string toolkit = getCudaToolkitRoot();
    if (!toolkit.empty())
      setenv("CUDA_ROOT", toolkit.c_str(), /*overwrite=*/0);
  }

  auto begin = std::chrono::steady_clock::now();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(program->module, &context);
  if (!module)
    return fail("the program of the run does not parse; this is a defect "
                "of mdir. Please report it with the directory that `mdir "
                "bug-report " + controlFile + "` writes");

  if (trial)
    (*module)->setAttr("mdir.step_graph_trial", mlir::UnitAttr::get(&context));
  if ((*module)->hasAttr("mdir.towers_dt"))
    (*module)->setAttr("mdir.towers_start_time",
                       mlir::FloatAttr::get(mlir::Float64Type::get(&context), firstTime));
  // Passes on functions are nested where they occur, as on the command
  // line of mlir-opt.
  std::set<std::string> neighborKinds;
  mlir::PassManager manager(&context,
                            mlir::ModuleOp::getOperationName(),
                            mlir::PassManager::Nesting::Implicit);
  if (!control->manifestFile.empty())
    manager.addInstrumentation(std::make_unique<ManifestNeighbors>(neighborKinds));
  // The kernels are compiled for the device that runs them: the first
  // visible one, or the one MDRT_DEVICE names (D214).
  auto gpu = compiler::getGpuOptions(*control, /*device=*/0);
  if (!gpu)
    return fail(gpu.takeError());
  std::string pipeline = compiler::getPipeline(*control, *program, *gpu);
  // MDIR_PIPELINE replaces the pipeline, to try another order of passes or
  // to stop part of the way.
  if (const char *replaced = std::getenv("MDIR_PIPELINE"))
    pipeline = replaced;
  if (emit == Emit::Pipeline) {
    llvm::outs() << pipeline << "\n";
    if (options.profileMap.empty())
      return 0;
  }
  // Custom pipelines and pass dumps retain their original pass sequence.
  if (trial && !trial->compiled.entry && !std::getenv("MDIR_PIPELINE") &&
      !std::getenv("MDIR_PRINT_AFTER")) {
    auto split = pipeline.find(",mdir-gpu-lower-to-nvvm{");
    if (split != std::string::npos) {
      if (mlir::failed(trial->lowering.prepare(module, StringRef(pipeline).take_front(split),
                                               trial->sharedLowering)))
        return fail("cannot lower the common step_graph trial program");
      pipeline.erase(0, split + 1);
    }
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
  std::string distDiagnostic;
  std::optional<mlir::ScopedDiagnosticHandler> handler;
  if (control->distribution)
    handler.emplace(&context, [&](mlir::Diagnostic &diagnostic) {
      if (diagnostic.getSeverity() == mlir::DiagnosticSeverity::Error &&
          distDiagnostic.empty())
        distDiagnostic = diagnostic.str();
      return mlir::failure(); // Also emit the diagnostic at its IR location.
    });
  // Unsupported ordinary-loop layouts are choices for AUTO. Keep genuine
  // compiler errors on the usual error path.
  mlir::ScopedDiagnosticHandler graphDiagnostics(&context, [&](mlir::Diagnostic &diag) {
    std::string message = diag.str();
    if (!trial || !StringRef(message).starts_with("step_graph: "))
      return mlir::failure();
    trial->refusal = message;
    return mlir::success();
  });
  if (!trial) manager.enableCrashReproducerGeneration(reproducer);
  // Later rounds run the same candidate with freshly initialized data.
  // Its executable already contains the lowered kernels and host entry.
  if ((!trial || !trial->compiled.entry) && mlir::failed(manager.run(*module))) {
    if (trial && !trial->refusal.empty()) return 0;
    if (!distDiagnostic.empty())
      return refuse(*control,
                    llvm::createStringError("%s", distDiagnostic.c_str()));
    return fail("cannot compile the run; this is a defect of mdir. The "
                "input of the passes is in '" + reproducer + "'. Please "
                "report it with the directory that `mdir bug-report " +
                controlFile + "` writes");
  }
  handler.reset();
  // The execution parameters of the md_alg ops, for the log and the
  // manifest (D408).
  auto [algPolicy, algPolicyLine] = takeAlgPolicy(*module);

  // Read the compiled choice so custom pipelines report the path they emit.
  bool hasFusedInverse =
      module->lookupSymbol("_mlir_ciface_mdrtCudaFFTGaussianBackwardF32");
  const char *pmeForceProduct = hasFusedInverse ? "inverse_load" : "separate";
  llvm::json::Array fftPlans;
  if (auto plans =
          (*module)->getAttrOfType<mlir::ArrayAttr>(kernels::kAxisFFTPlans)) {
    for (mlir::Attribute plan : plans) {
      llvm::json::Object record;
      for (mlir::NamedAttribute field :
           mlir::cast<mlir::DictionaryAttr>(plan)) {
        StringRef key = field.getName().getValue();
        if (auto array =
                mlir::dyn_cast<mlir::DenseI64ArrayAttr>(field.getValue())) {
          llvm::json::Array values;
          for (int64_t value : array.asArrayRef())
            values.push_back(value);
          record[key] = std::move(values);
        } else {
          record[key] =
              mlir::cast<mlir::StringAttr>(field.getValue()).getValue().str();
        }
      }
      if (!hasFusedInverse && record.getString("backend") == "axis" &&
          record.getString("evaluation") == "forces")
        pmeForceProduct = "axis_x";
      fftPlans.push_back(std::move(record));
    }
    (*module)->removeAttr(kernels::kAxisFFTPlans);
  }
  std::string fftPlanLine;
  if (!fftPlans.empty())
    fftPlanLine = llvm::formatv("MDIR: PME FFT plans {0}\n",
                                llvm::json::Value(llvm::json::Array(fftPlans)))
                      .str();

  // The profile map of the lowered program, which names the source of each
  // parallel region of the host (D398).
  auto writeProfileMap = [&](const std::string &path) -> llvm::Error {
    auto map = compiler::getProfileMap(*module, *control);
    if (!map)
      return map.takeError();
    std::error_code error;
    llvm::raw_fd_ostream os(path, error);
    if (error)
      return llvm::createStringError(error, "cannot write the profile map '" +
                                                path + "': " + error.message());
    os << *map;
    return llvm::Error::success();
  };
  if (emit != Emit::Run) {
    if (emit == Emit::Lowered)
      module->print(llvm::outs());
    if (!options.profileMap.empty())
      if (llvm::Error error = writeProfileMap(options.profileMap))
        return fail(std::move(error));
    return 0;
  }
  // Where the compile cache is, or why there is none, and what it gave the
  // GPU modules of the run and the host objects of the skin search
  // (D397). The host code of the run itself is not
  // cached until it moves onto the owned engine (#99).
  compiler::CompileCacheLocation cacheLocation =
      compiler::CompileCacheConfig::locate();
  compiler::CompileStats gpuCache = mdir::takeGpuModuleStats(context);
  compiler::CompileStats searchCache;
  for (const compiler::SkinTrial &trial : skinChoice.trials)
    searchCache += trial.cache;
  std::string cacheLine = "MDIR: compile cache: ";
  if (cacheLocation.config)
    cacheLine += "on, '" + cacheLocation.config->directory + "' (" +
                 cacheLocation.source + ")";
  else
    cacheLine += "off (" + cacheLocation.reason + ")";
  if (gpuCache.gpuModules)
    cacheLine += llvm::formatv(
        "; GPU modules {0}: PTX {1} compiled, {2} read; cubins {3} compiled, "
        "{4} read; {5} stored",
        gpuCache.gpuModules, gpuCache.gpuPtxCompiled, gpuCache.gpuPtxHits,
        gpuCache.gpuCubinCompiled, gpuCache.gpuCubinHits, gpuCache.gpuStored);
  if (!skinChoice.trials.empty())
    cacheLine += llvm::formatv(
        "; the skin search: host objects {0} compiled, {1} read, {2} stored",
        searchCache.compiled, searchCache.hits, searchCache.stored);
  cacheLine += "\n";
  auto optionalText = [](const std::string &text) {
    return text.empty() ? llvm::json::Value(nullptr) : llvm::json::Value(text);
  };
  llvm::json::Object cacheRecord{
      {"enabled", cacheLocation.config.has_value()},
      {"directory", cacheLocation.config
                        ? llvm::json::Value(cacheLocation.config->directory)
                        : llvm::json::Value(nullptr)},
      {"source", optionalText(cacheLocation.source)},
      {"reason", optionalText(cacheLocation.reason)},
      {"gpu_modules", gpuCache.gpuModules},
      {"ptx_compiled", gpuCache.gpuPtxCompiled},
      {"ptx_read", gpuCache.gpuPtxHits},
      {"cubin_compiled", gpuCache.gpuCubinCompiled},
      {"cubin_read", gpuCache.gpuCubinHits},
      {"gpu_stored", gpuCache.gpuStored},
      {"gpu_rejected", gpuCache.gpuRejected},
      {"skin_search_host_compiled", searchCache.compiled},
      {"skin_search_host_read", searchCache.hits},
      {"skin_search_host_stored", searchCache.stored}};
  // A run that is not continued keeps the outputs of an earlier run that
  // has the same names as `#<name>.<n>#` before it writes its own (D149);
  // those of a continued run are its own.
  if (!options.continues) {
    std::vector<std::string> written = {control->logFile, control->energyFile,
                                        control->pullFile, control->manifestFile,
                                        control->profileMapFile,
                                        control->freeEnergyFile,
                                        control->observablesFile};
    if (control->framePeriod > 0)
      written.push_back(control->trajectoryFile);
    if (control->checkpointPeriod > 0) {
      written.push_back(control->restartOutput);
      written.push_back(getPreviousCheckpointPath(control->restartOutput));
    }
    // None is moved unless all can be.
    for (const std::string &file : written)
      if (!file.empty())
        if (llvm::Error error = checkBackup(file))
          return fail(std::move(error));
    for (const std::string &file : written) {
      if (file.empty())
        continue;
      auto backup = backUpOutput(file);
      if (!backup)
        return fail(backup.takeError());
      if (!backup->empty())
        output.log.print("MDIR: backed up '%s' as '%s'\n", file.c_str(),
                         backup->c_str());
    }
  }
  // The log file holds what the run has printed so far; a continued run
  // appends to it. Without one the log keeps nothing from here on: this
  // is the only place where its file opens (#304).
  if (!control->logFile.empty()) {
    if (llvm::Error error =
            output.log.open(getOutputPath(control->logFile), own.has_value()))
      return fail(std::move(error));
  } else {
    output.log.keepNothing();
  }
  if (!control->profileMapFile.empty())
    if (llvm::Error error = writeProfileMap(control->profileMapFile))
      return fail(std::move(error));

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
    // An installed tree carries the OpenMP runtime in its lib; a build tree
    // takes the one of the LLVM it was built with.
    std::string omp = getLibrary(libraries, "libomp.so");
    if (!llvm::sys::fs::exists(omp))
      omp = getLibrary(MDIR_LLVM_LIBRARY_DIR, "libomp.so");
    paths.push_back(omp);
  }
  for (const std::string &path : paths)
    if (!llvm::sys::fs::exists(path))
      return fail("cannot find '" + path + "'");

  CompiledRun production;
  auto &compiled = trial ? trial->compiled : production;
  bool needsCompile = !compiled.entry;
  if (needsCompile) {
    auto result = compileEntry(*module, paths, *program, trial != nullptr);
    if (!result)
      return fail(result.takeError());
    compiled = std::move(*result);
  }
  // The learned potentials (D270): the plugin beside the
  // runtime, or at MDIR_MLIP_PLUGIN, on the device of the runtime, which
  // the program has loaded.
  if (!control->models.empty()) {
    auto models = ModelTerms::load(*control, *system, libraries);
    if (!models)
      return fail(models.takeError());
    output.models = std::move(*models);
  }
  double compileTime = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - begin)
                           .count();

  //===--------------------------------------------------------------------===//
  // Run
  //===--------------------------------------------------------------------===//

  if (trial && needsCompile) {
    trial->compileSeconds = compileTime;
    trial->cache = gpuCache;
    if (compiled.trialEngine)
      trial->cache += compiled.trialEngine->getCompileStats();
  }
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
  // The values of the state at the start that the entry takes rather than
  // its text (D227).
  SmallVector<double> startValues;
  for (const Program::StartValue &value : program->startValues)
    startValues.push_back(value.value);
  for (double &value : startValues)
    arguments.push_back(&value);

  int distRank = 0, distRanks = 1;
  llvm::json::Object distribution;
  if (const auto &dist = control->distribution) {
    const auto &grid = program->domainGrid;
    const char *source = dist->domains
                             ? "minimum surface area; lexicographic ties"
                         : dist->gridGiven ? "explicit grid"
                                           : "default one domain";
    llvm::json::Array candidates;
    if (dist->domains)
      mdir::md_dist::selectGrid(
          dist->domains, system->box,
          [&](int64_t x, int64_t y, int64_t z, long double score) {
            candidates.push_back(
                llvm::json::Object{{"grid", llvm::json::Array{x, y, z}},
                                   {"score_per_nm", double(score)}});
            output.log.print(
                "MDIR: grid candidate %lld x %lld x %lld score %.17g /nm\n",
                (long long)x, (long long)y, (long long)z, double(score));
          });
    distribution = llvm::json::Object{
        {"grid", llvm::json::Array{grid[0], grid[1], grid[2]}},
        {"grid_reason", source},
        {"requested_count", dist->domains ? llvm::json::Value(dist->domains)
                                          : llvm::json::Value(nullptr)},
        {"candidates", std::move(candidates)},
        {"transport", dist->transport},
        {"transport_source", dist->transportGiven ? "requested" : "default"},
        {"traversal", dist->traversal},
        {"traversal_source", dist->traversalGiven ? "requested" : "default"},
        {"ranks", distRanks},
        {"cutoff_angstrom", control->cutoffDistance},
        {"skin_angstrom", control->pairlistDistance - control->cutoffDistance},
        {"reach_angstrom", control->pairlistDistance}};
    output.log.print(
        "MDIR: grid choice %s, requested count %lld, ranks %d; %s transport "
        "(%s), %s traversal (%s); reach %g Ang = cutoff %g Ang + skin %g Ang\n",
        source, (long long)dist->domains, distRanks, dist->transport.c_str(),
        dist->transportGiven ? "requested" : "default", dist->traversal.c_str(),
        dist->traversalGiven ? "requested" : "default",
        control->pairlistDistance, control->cutoffDistance,
        control->pairlistDistance - control->cutoffDistance);
  }
  if (!control->manifestFile.empty()) {
    auto device = getManifestDevice(control->target);
    if (!device)
      return fail(device.takeError());
    llvm::json::Object outputs;
    for (auto [name, path] : {
        std::pair<const char *, std::string>{"log", control->logFile},
        {"energy", control->energyFile}, {"pull", control->pullFile},
        {"observables", control->observablesFile},
        {"trajectory", control->framePeriod > 0 ? control->trajectoryFile : ""},
        {"manifest", control->manifestFile},
        {"profile_map", control->profileMapFile}})
      if (!path.empty())
        outputs[name] = getOutputPath(path);
    if (!control->restartOutput.empty())
      outputs["checkpoint"] = control->restartOutput;
    auto element = [](Element type) {
      return type == Element::F32 ? "f32" : "f64";
    };
    llvm::json::Array neighbors, warnings;
    for (const auto &kind : neighborKinds)
      neighbors.push_back(kind);
    for (const auto &[code, message] : system->warnings)
      warnings.push_back(llvm::json::Object{{"code", code}, {"message", message}});
    for (const auto &[code, message] : control->rigidUnitsRoute.warnings)
      warnings.push_back(llvm::json::Object{{"code", code}, {"message", message}});
    llvm::json::Value pme = nullptr;
    if (program->pme)
      pme = llvm::json::Object{{"beta_inverse_angstrom", program->pmeBeta * units::length},
          {"grid", llvm::json::Array{program->pmeGrid[0], program->pmeGrid[1],
                                    program->pmeGrid[2]}},
          {"order", program->pmeOrder},
          {"force_only_product_rule", pmeForceProduct},
          {"kernel", program->pmeProlate ? "PROLATE" : "GAUSSIAN"}};
    if (program->pme && !fftPlans.empty())
      (*pme.getAsObject())["fft_plans"] = std::move(fftPlans);
    // The prolate kernel's own parameters (D266).
    if (program->pme && program->pmeProlate) {
      llvm::json::Object &object = *pme.getAsObject();
      object["bandwidth"] = program->pmeBandwidth;
      object["continuity"] = control->pmeContinuity;
      if (control->pmeBasisTerms)
        object["basis_terms"] = control->pmeBasisTerms;
      object["window"] = control->pmeWindow;
      if (control->pmeWindow == "PROLATE") {
        object["window_bandwidth"] = program->pmeWindowBandwidth;
        if (!control->pmeBasisTerms)
          object["aliasing_estimate"] = program->pmeAliasing;
      } else {
        // The mesh error estimate of the B-spline window on this grid, and
        // the bound the automatic grid meets (D444).
        object["aliasing_estimate"] = program->pmeAliasing;
        object["aliasing_bound"] = mdir::prolate::getSplineBound(
            control->pmeAlphaTolerance, control->pmeBasisTerms > 0);
        object["grid_source"] = control->pmeGrid[0]          ? "grid"
                                : control->pmeMaxSpacing > 0.0 ? "max_spacing"
                                                               : "automatic";
      }
    }
    llvm::json::Object effective{
        {"seed", std::to_string(control->seed)}, {"time_step_ps", control->timestep},
        {"trajectory_format", control->framePeriod == 0 ? llvm::json::Value(nullptr) :
            llvm::json::Value(getTrajectoryFormatName(control->trajectoryFormat))},
        {"neighbor_structure_requested",
         getNeighborStructureName(control->neighborStructureRequested)},
        {"neighbor_structure", getNeighborStructureName(control->neighborStructure)},
        {"neighbor_structure_reason", control->neighborStructureReason},
        // The route of the rigid units and why (D424).
        {"rigid_units_requested", getRigidUnitsName(control->rigidUnits)},
        {"rigid_units",
         control->rigidUnitsRoute.units ? "units" : "general"},
        {"rigid_units_reason", control->rigidUnitsRoute.reason},
        {"rigid_units_route", getRigidUnitsJSON(*control)},
        {"pairlist_distance_angstrom", control->pairlistDistance},
        {"pruned_distance_angstrom", control->prunedDistance != 0.0
             ? llvm::json::Value(control->prunedDistance) : llvm::json::Value(nullptr)},
        // Where each reach came from, and the search if there was one
        // (D281).
        {"pairlist_distance_source",
         control->getSkinSource(/*outer=*/true)},
        {"pruned_distance_source",
         control->getSkinSource(/*outer=*/false)},
        {"skin_search", control->isSkinAuto()
             ? skinChoice.toJSON(*control) : llvm::json::Value(nullptr)},
        {"neighbor_structures", std::move(neighbors)},
        // The policy each md_alg op was lowered with (D408): a
        // bitwise comparison of two runs compares these first.
        {"md_alg_policy", std::move(algPolicy)},
        {"buffer_precision", llvm::json::Object{{"state", element(program->state)},
            {"force", element(program->force)}, {"mass", element(program->mass)},
            {"parameter", element(program->parameter)}}}, {"pme", std::move(pme)}};
    if (control->periodic && control->precision != Precision::Double) {
      auto deltas = [](const std::array<double, 6> &changes) {
        llvm::json::Array result;
        for (double d : changes)
          result.push_back(d);
        return result;
      };
      effective["box_rounding_delta_nm"] = deltas(system->boxRounding);
      if (checkpoint)
        effective["input_box_rounding_delta_nm"] = deltas(inputBoxRounding);
    }
    if (control->distribution)
      effective["distribute"] = std::move(distribution);
    if (towersShapeReport.getAsObject())
      effective["towers_shape"] = towersShapeReport;
    if (!towersCellHeight.empty())
      effective["towers_cell_height"] = towersCellHeight;
    resolvedChoices["step_graph"] = std::move(graphChoice);
    resolvedChoices["term_streams"] = termStreamsChoice(*control);
    llvm::json::Object metadata{
        {"build", getManifestBuild()}, {"effective", std::move(effective)},
        {"resolved_choices", std::move(resolvedChoices)},
        {"warnings", std::move(warnings)}, {"pipeline", pipeline},
        {"inputs", std::move(manifestInputs)},
        {"target", control->target == Target::GPU ? "gpu" : "cpu"},
        {"precision", control->precision == Precision::Double ? "double" :
                      control->precision == Precision::Mixed ? "mixed" : "single"},
        {"threads", control->threads}, {"device", std::move(*device)},
        {"outputs", std::move(outputs)}, {"first_step", firstStep},
        {"requested_end_step", firstStep + control->numSteps},
        {"part", part}, {"continued", options.continues},
        {"compile_seconds", compileTime},
        {"compile_cache", std::move(cacheRecord)}};
    if (!forcedChanges.empty()) {
      llvm::json::Array overridden;
      for (const FingerprintChange &change : forcedChanges)
        overridden.push_back(llvm::json::Object{{"group", change.group},
                                                {"name", change.name},
                                                {"checkpoint", change.before},
                                                {"control", change.after}});
      metadata["forced_overrides"] = std::move(overridden);
    }
    if (llvm::Error error = manifest.start(getOutputPath(control->manifestFile),
                                            options.continues, std::move(metadata)))
      return fail(std::move(error));
    output.recordStop = [&](int64_t step, const char *reason) {
      return manifest.finish("stopped", step, reason);
    };
  }

  output.state = program->state;
  output.force = program->force;
  output.onDevice = control->target == Target::GPU;
  output.firstStep = firstStep;
  output.energyPeriod = control->energyPeriod;
  // What the summary at the end takes of the rows, in constant memory
  // (#300): the rate from the first row at or after the middle of the
  // run, and the temperatures of the solute and of the solvent in ten
  // blocks of the rows after the first, which has none.
  output.rateStep = firstStep + control->numSteps / 2;
  int64_t temperatureRows =
      control->energyPeriod > 0 ? control->numSteps / control->energyPeriod : 0;
  for (BlockMean *mean : {&output.soluteOptimal, &output.solventOptimal,
                          &output.soluteFull, &output.solventFull})
    mean->plan(temperatureRows);
  output.firstTime = firstTime;
  output.timestep = control->timestep;
  // Langevin dynamics exchanges energy with the bath in every step, which
  // the run does not count (D135): its log has no conserved energy.
  output.distributedWordEnergy =
      control->distribution && control->deterministic;
  output.couples = control->getCouplingPeriod() > 0 && !control->isLangevin();
  output.changesCell = control->barostat;
  if (control->barostat && control->precision != Precision::Double) {
    output.checkpoint.masterCell.assign(system->inputMasterCell.begin(),
                                        system->inputMasterCell.end());
    if (checkpoint && !checkpoint->masterCell.empty())
      output.checkpoint.masterCell = checkpoint->masterCell;
  }
  output.overdamped = control->isBrownian();
  output.halfTemperature = control->isLangevin();
  output.bathKinetic = 0.5 * system->getDegreesOfFreedom() *
                       units::boltzmann * control->temperature;
  output.minimizes = control->minimize;
  output.minimizeTolerance =
      control->minimizeTolerance * units::energy / units::length;
  output.minimizeStepFloor = control->getMinimizeStepFloor() * units::length;
  output.minimizeFirstStep = control->minimizeStep * units::length;
  output.leastEdge = 2.0 * control->cutoffDistance * units::length;
  output.leastReachEdge =
      control->neighborStructure == driver::NeighborStructure::Groups
          ? control->pairlistDistance * units::length / 0.999
          : 0.0;
  output.degreesOfFreedom = system->getDegreesOfFreedom();
  output.solventFreedom = system->getSolventDegreesOfFreedom();
  output.volume = system->box[0] * system->box[1] * system->box[2];
  output.firstVolume = output.volume;
  for (int k = 0; k != 3; ++k)
    output.box[k] = system->box[k];
  output.dispersionEnergy = program->dispersionEnergy;
  output.dispersionVirial = program->dispersionVirial;
  output.observableVolumeConstants = program->observableVolumeConstants;
  output.observableFixedConstants = program->observableFixedConstants;
  output.dispersionFixedEnergy = program->dispersionFixedEnergy;
  output.pme = program->pme;
  output.periodic = control->periodic;
  output.listReach = control->pairlistDistance * units::length;
  output.reactionField = program->reactionField;
  output.coulombConstantEnergy = program->coulombConstantEnergy;
  output.coulombConstantVirial = program->coulombConstantVirial;
  output.coulombSelfEnergy = program->coulombSelfEnergy;
  output.ljpme = program->ljpme;
  output.ljpmeSelfEnergy = program->ljpmeSelfEnergy;
  output.system = &*system;
  // The files of columns (D149): a continued run keeps their rows up to
  // its checkpoint and appends.
  std::optional<int64_t> keepThrough;
  if (own)
    keepThrough = own->step;
  if (!control->energyFile.empty())
    if (llvm::Error error =
            output.energies.open(getOutputPath(control->energyFile),
                                 getEnergyColumns(output), keepThrough))
      return fail(std::move(error));
  if (control->framePeriod > 0) {
    // The cell in Å, from the system: a topology gives it with the
    // coordinates, not the control file.
    double cell[3];
    for (int k = 0; k != 3; ++k)
      cell[k] = system->box[k] / units::length;
    // A continued run appends its frames to the trajectory of the part of
    // its outputs, which its checkpoint counts them in, cut to those
    // frames, or writes them to a part of their own (D130, D149).
    std::string trajectory = getOutputPath(control->trajectoryFile);
    bool appends = false;
    if (own && options.appends && !own->trajectory.empty()) {
      StringRef name = llvm::sys::path::filename(trajectory);
      if (own->trajectory != name)
        return fail("'" + control->restartOutput + "' counts the frames of '" +
                    own->trajectory + "', which is not the trajectory that "
                    "the run writes, '" + name + "'");
      if (llvm::sys::fs::exists(trajectory)) {
        appends = true;
      } else if (own->frames > 0) {
        return fail("'" + trajectory + "' is missing, and '" +
                    control->restartOutput + "' counts " +
                    llvm::Twine(own->frames) + " frames in it; --no-append "
                    "writes the frames that follow to a part of their own");
      }
    }
    if (control->trajectoryFormat == TrajectoryFormat::H5MD) {
      // The frames of the state without loss (D239): the
      // potential energy of a frame is that of the row of its step, where
      // every frame is a step of energy.
      H5MDOptions options;
      options.omitObjectTimes = output.distributedWordEnergy;
      options.single = control->trajectorySingle;
      options.variableStrings = control->trajectoryVariableStrings;
      options.energy = !control->minimize && control->energyPeriod > 0 &&
                       control->framePeriod % control->energyPeriod == 0;
      options.triclinic = system->tilt[0] != 0.0 || system->tilt[1] != 0.0 ||
                          system->tilt[2] != 0.0;
      options.masses = system->masses;
      options.species.assign(system->types.begin(), system->types.end());
      options.creatorVersion = MDIR_VERSION;
      options.precision =
          control->precision == Precision::Single
              ? "single"
              : control->precision == Precision::Mixed ? "mixed" : "double";
      output.trajectory = std::make_unique<H5MDWriter>(std::move(options));
      output.trajectory->setExactEdges(system->box);
      output.trajectory->setExactTilt(system->tilt);
    } else {
      output.trajectory = createTrajectoryWriter(control->trajectoryFormat);
    }
    output.trajectory->setPeriodic(control->periodic);
    if (appends) {
      auto removed = output.trajectory->append(
          trajectory, count, own->frames, control->framePeriod,
          control->timestep, cell);
      if (!removed)
        return fail(removed.takeError());
      if (auto last = output.trajectory->getLastStep();
          last && *last > own->step)
        return fail("'" + trajectory + "' holds a frame of step " +
                    llvm::Twine(*last) + " among the " +
                    llvm::Twine(own->frames) + " that the checkpoint counts, "
                    "past its step, " + llvm::Twine(own->step) + ": it is not "
                    "the trajectory of this run");
      if (*removed > 0)
        output.log.print("MDIR: removed %lld frames past the checkpoint "
                         "from '%s'\n",
                         static_cast<long long>(*removed),
                         trajectory.c_str());
    } else if (llvm::Error error = output.trajectory->open(
                   trajectory, count, firstStep + control->framePeriod,
                   control->framePeriod, control->timestep, cell)) {
      return fail(std::move(error));
    }
    output.trajectoryName = llvm::sys::path::filename(trajectory).str();
    double tilts[3];
    for (int k = 0; k != 3; ++k)
      tilts[k] = system->tilt[k] / units::length;
    output.trajectory->setTilt(tilts);
    output.hasTrajectory = true;
  }
  // The terms over centers (D145), at every energy of the log: for each
  // its coordinates, its energy, and its force, over two centers the
  // component along the distance and the vector on the second center, of
  // an angle or a dihedral −∂E/∂θ; the coordinates and forces of pulling
  // that GROMACS writes to its pullx and pullf files, in one file.
  // [free_energy] (D161): at every energy of the log, dH/dλ of each
  // component and the difference of the energy to each state, the input of
  // thermodynamic integration and of MBAR. With one state the differences
  // are identically 0 and are left out (D190).
  if (!control->freeEnergyFile.empty()) {
    std::vector<ColumnFile::Column> columns = {{"step", "-", true},
                                               {"time", "ps"}};
    for (const auto &[name, values] : control->freeEnergy.lambdas)
      columns.push_back({"dHdl." + name, "kcal/mol"});
    size_t states = control->freeEnergy.getNumStates();
    for (size_t k = 0; states > 1 && k != states; ++k)
      columns.push_back({"dU." + std::to_string(k), "kcal/mol"});
    output.freeEnergyState = control->freeEnergy.state;
    output.stateFixedEnergies = program->stateFixedEnergies;
    output.stateVolumeEnergies = program->stateVolumeEnergies;
    output.lambdaFixedDerivatives = program->lambdaFixedDerivatives;
    output.lambdaVolumeDerivatives = program->lambdaVolumeDerivatives;
    if (llvm::Error error = output.freeEnergy.open(
            getOutputPath(control->freeEnergyFile), columns, keepThrough))
      return fail(std::move(error));
  }
  // [output] observe (D189): the energy of a term, `<term>.energy`, or its
  // derivative in a constant of its, `<term>.d_<constant>`, per unit of the
  // constant, at every energy of the log.
  if (!control->observablesFile.empty()) {
    std::vector<ColumnFile::Column> columns = {{"step", "-", true},
                                               {"time", "ps"}};
    for (const auto &[name, unit] : control->getObservableColumns())
      columns.push_back({name, unit});
    if (llvm::Error error = output.observables.open(
            getOutputPath(control->observablesFile), columns, keepThrough))
      return fail(std::move(error));
  }
  if (!control->pullFile.empty()) {
    std::vector<ColumnFile::Column> columns = {{"step", "-", true},
                                               {"time", "ps"}};
    for (const TupleTerm &term : system->topology->tupleTerms) {
      if (!term.isCentroid())
        continue;
      auto add = [&](StringRef quantity, StringRef unit) {
        columns.push_back({(term.name + "." + quantity).str(), unit.str()});
      };
      if (term.arity == 2) {
        // The line of units of the file, not printed (D396).
        for (StringRef quantity : {"r", "dx", "dy", "dz"})
          add(quantity, "Å"); // ascii-exempt
        add("energy", "kcal/mol");
        for (StringRef quantity : {"f_r", "fx", "fy", "fz"})
          add(quantity, "kcal/mol/Å"); // ascii-exempt
      } else {
        add("theta", "rad");
        add("energy", "kcal/mol");
        add("f_theta", "kcal/mol/rad");
      }
      output.pullCounts.push_back(term.arity == 2 ? 4 : 1);
    }
    if (llvm::Error error = output.pull.open(
            getOutputPath(control->pullFile), columns, keepThrough))
      return fail(std::move(error));
  }
  if (writesCheckpoints) {
    output.checkpointPath = control->restartOutput;
    Checkpoint &checkpoint = output.checkpoint;
    checkpoint.omitObjectTimes = output.distributedWordEnergy;
    // A forced continuation is recorded in every checkpoint that follows
    // it, with the earlier ones of the run.
    if (own)
      checkpoint.forcedContinuations = own->forcedContinuations;
    for (const FingerprintChange &change : forcedChanges)
      checkpoint.forcedContinuations +=
          "part " + std::to_string(part) + ": " + change.group + " " +
          change.name + ": " + change.before + " -> " + change.after + "\n";
    checkpoint.masses = system->masses;
    checkpoint.species.assign(system->types.begin(), system->types.end());
    for (int i = 0; i != 3; ++i) {
      checkpoint.box[i] = system->box[i];
      checkpoint.tilt[i] = system->tilt[i];
    }
    checkpoint.periodic = control->periodic;
    checkpoint.integrator = integrator.str();
    checkpoint.towersCellHeight = towersCellHeight;
    checkpoint.velocityOffset = velocityOffset;
    checkpoint.precision =
        control->precision == Precision::Single
            ? "single"
            : control->precision == Precision::Mixed ? "mixed" : "double";
    checkpoint.timestep = control->timestep;
    checkpoint.seed = control->seed;
    checkpoint.firstStep = runFirstStep;
    checkpoint.part = part;
    checkpoint.outputsPart = outputsPart;
    checkpoint.trajectory = output.trajectoryName;
    checkpoint.fingerprint = fingerprint;
    checkpoint.creatorVersion = getBuildVersion();
    if (control->hasFreeEnergy) {
      checkpoint.freeEnergy = control->freeEnergy.describe();
      checkpoint.freeEnergyState = control->freeEnergy.state;
      for (const auto &[name, values] : control->freeEnergy.lambdas)
        checkpoint.freeEnergyLambda.push_back(
            values[control->freeEnergy.state]);
    }
  }
  output.endStep = firstStep + control->numSteps;
  // A continued run counts the energy that the coupling has taken from
  // where its checkpoint left it, so that its conserved energy continues.
  if (own)
    output.bath = own->bath;
  // A Nose-Hoover chain (D163a), at rest or as the checkpoint left it.
  output.setChain(*control, *system);
  output.began = began;
  output.maxWalltime = options.maxWalltime;
  setOutput(&output);

  // The version and the commit come first, so that a log names the build
  // that wrote it.
  output.log.print("MDIR %s, commit %.12s%s\n", MDIR_VERSION, MDIR_GIT_COMMIT,
                   StringRef(MDIR_GIT_DIRTY) == "yes"
                       ? " with uncommitted changes"
                       : "");
  if (control->minimize && control->minimizeTolerance > 0.0)
    output.log.print("MDIR: %zu particles, at most %lld steps of steepest "
                     "descent, until the largest force is below %g "
                     "kcal/mol/Ang\n",
                     count, static_cast<long long>(control->numSteps),
                     control->minimizeTolerance);
  else if (control->minimize)
    output.log.print(
                 "MDIR: %zu particles, %lld steps of steepest descent\n",
                 count, static_cast<long long>(control->numSteps));
  else
    output.log.print("MDIR: %zu particles, %lld steps of %g ps\n",
                 count, static_cast<long long>(control->numSteps),
                 control->timestep);
  if (control->distribution) {
    const auto &grid = program->domainGrid;
    output.log.print(
        "MDIR: domain grid %lld x %lld x %lld, %s transport, "
        "%s traversal, halo reach %g Ang\n",
        static_cast<long long>(grid[0]), static_cast<long long>(grid[1]),
        static_cast<long long>(grid[2]),
        control->distribution->transport.c_str(),
        control->distribution->traversal.c_str(), control->pairlistDistance);
  }
  if (program->pme)
    output.log.print("MDIR: PME force-only product rule: %s\n", pmeForceProduct);
  // The parameters that the prolate kernel derives (D266).
  if (program->pmeProlate && (control->pmeBasisTerms || control->pmeWindow == "BSPLINE")) {
    output.log.print("MDIR: prolate PME, continuity %lld, c = %.6g, %s window, order %lld\n",
                     static_cast<long long>(control->pmeContinuity), program->pmeBandwidth,
                     control->pmeWindow.c_str(), static_cast<long long>(program->pmeOrder));
    if (control->pmeBasisTerms)
      output.log.print("MDIR: %lld even Legendre terms\n",
                       static_cast<long long>(control->pmeBasisTerms));
    if (control->pmeWindow == "BSPLINE")
      output.log.print(
          "MDIR: a grid of %lld x %lld x %lld (%s), the mesh error estimate "
          "%.2g, the bound of the automatic grid %.2g (D444)\n",
          static_cast<long long>(program->pmeGrid[0]),
          static_cast<long long>(program->pmeGrid[1]),
          static_cast<long long>(program->pmeGrid[2]),
          control->pmeGrid[0]           ? "given"
          : control->pmeMaxSpacing > 0.0 ? "from max_spacing"
                                         : "automatic",
          program->pmeAliasing,
          mdir::prolate::getSplineBound(control->pmeAlphaTolerance,
                                        control->pmeBasisTerms > 0));
  } else if (program->pmeProlate)
    output.log.print("MDIR: the prolate kernel of PME: c = %.6g, c1 = %.6g, "
                     "%lld points of the window, a grid of %lld x %lld x "
                     "%lld, the aliasing estimate %.2g\n",
                     program->pmeBandwidth, program->pmeWindowBandwidth,
                     static_cast<long long>(program->pmeOrder),
                     static_cast<long long>(program->pmeGrid[0]),
                     static_cast<long long>(program->pmeGrid[1]),
                     static_cast<long long>(program->pmeGrid[2]),
                     program->pmeAliasing);
  if (!control->periodic)
    output.log.print(
                 "MDIR: no periodic cell; the particles are in a cell of "
                 "%.4f %.4f %.4f Ang, whose images stay beyond the reach of "
                 "the neighbor structures, %.4f Ang, while the particles "
                 "spread less than %.4f %.4f %.4f Ang (D142)\n",
                 system->box[0] / units::length, system->box[1] / units::length,
                 system->box[2] / units::length, control->pairlistDistance,
                 (system->box[0] - output.listReach) / units::length,
                 (system->box[1] - output.listReach) / units::length,
                 (system->box[2] - output.listReach) / units::length);
  if (control->rebuildPeriod > 0)
    output.log.print(
                 "MDIR: warning: the neighbor structures are rebuilt every "
                 "%lld steps and not tested in between (rebuild_interval, "
                 "opt-in); they may miss pairs within the cutoff\n",
                 static_cast<long long>(control->rebuildPeriod));
  if (own)
    output.log.print(
                 "MDIR: continues the run after step %lld, from '%s', to "
                 "step %lld (part %lld)\n",
                 static_cast<long long>(firstStep),
                 control->restartInput.c_str(),
                 static_cast<long long>(output.endStep),
                 static_cast<long long>(part));
  else if (isRestart)
    output.log.print("MDIR: continues after step %lld, from '%s'\n",
                 static_cast<long long>(firstStep),
                 control->restartInput.c_str());
  else if (system->givenVelocities)
    output.log.print(
                 "MDIR: the velocities are those of the file of "
                 "coordinates\n");
  output.log.print("MDIR: step_graph %s\n", graphReport.c_str());
  output.log.print(
      "MDIR: term_streams %s\n",
      llvm::formatv("{0}", llvm::json::Value(termStreamsChoice(*control)))
          .str()
          .c_str());
  if (towersShapeReport.getAsObject())
    output.log.print("MDIR: TOWERS launch shape %s\n",
                     llvm::formatv("{0}", towersShapeReport).str().c_str());
  if (control->periodic && control->precision != Precision::Double) {
    auto logDelta = [&](const char *source, const std::array<double, 6> &d) {
      output.log.print("MDIR: %sbinary32 box delta (nm, ax by cz bx cx cy): "
                       "%+.17g %+.17g %+.17g %+.17g %+.17g %+.17g\n",
                       source, d[0], d[1], d[2], d[3], d[4], d[5]);
    };
    if (checkpoint)
      logDelta("coordinate input ", inputBoxRounding);
    logDelta("", system->boxRounding);
  }
  for (const std::string &note : notes)
    output.log.print("MDIR: note: %s\n", note.c_str());
  // The structure that `AUTO` took, and why (D261); a
  // structure that was given is not repeated.
  if (control->neighborStructureRequested == NeighborStructure::Auto) {
    if (control->neighborStructure == NeighborStructure::Groups)
      output.log.print("MDIR: neighbor structure: groups of 16 with the "
                       "reach %g Ang", control->pairlistDistance);
    else
      output.log.print("MDIR: neighbor structure: the matrix with the "
                       "reach %g Ang", control->pairlistDistance);
    if (control->prunedDistance != 0.0)
      output.log.print(", pruned to %g Ang", control->prunedDistance);
    output.log.print(" (AUTO: %s)\n",
                     control->neighborStructureReason.c_str());
  }
  // The route of the rigid units, where the key was given or the units
  // are taken, with the radii widened at the input (D424).
  if (control->rigidUnitsGiven || control->rigidUnitsRoute.units) {
    output.log.print("MDIR: rigid units: %s\n",
                     describeRigidUnitsRoute(*control).c_str());
    if (control->rigidUnitsRoute.units)
      for (const std::string &line : describeRigidUnitsKinds(*control))
        output.log.print("MDIR:   %s\n", line.c_str());
    for (const auto &[code, message] : control->rigidUnitsRoute.warnings)
      output.log.print("MDIR: warning: %s\n", message.c_str());
  }
  // The reaches that "AUTO" took, and how (D281): given ones and
  // the defaults of keys not given are not repeated.
  // Whether the choice was searched now or reused from the cache, and the
  // budget of the search (D401).
  bool skinsShown = control->skinSource == "cached" ||
                    control->skinSource == "measured" ||
                    control->skinsRequested || !skinChoice.refused.empty();
  if (control->isSkinAuto() && skinsShown) {
    output.log.print("%s", skinLines.c_str());
    output.log.print("MDIR: skins: the reach %g Ang",
                     control->pairlistDistance);
    if (control->prunedDistance != 0.0)
      output.log.print(", pruned to %g Ang", control->prunedDistance);
    output.log.print(" (%s: %s", control->skinSource.c_str(),
                     control->skinReason.c_str());
    if (skinChoice.recorded) {
      const llvm::json::Array *candidates =
          skinChoice.recorded->getArray("candidates");
      output.log.print("; reused, not searched: the search of a budget of "
                       "%lld recorded %zu candidates in %.1f s",
                       static_cast<long long>(
                           skinChoice.recorded->getInteger("budget")
                               .value_or(0)),
                       candidates ? candidates->size() : size_t(0),
                       skinChoice.recorded->getNumber("seconds").value_or(0.0));
    } else if (skinChoice.searched) {
      output.log.print("; searched now: a budget of %lld, %zu candidates in "
                       "%.1f s",
                       static_cast<long long>(skinChoice.budget.value_or(0)),
                       skinChoice.trials.size(), skinChoice.seconds);
    } else if (!skinChoice.trials.empty()) {
      output.log.print("; not searched: the reference alone measured, in "
                       "%.1f s", skinChoice.seconds);
      if (skinChoice.budget)
        output.log.print(", a budget of %lld",
                         static_cast<long long>(*skinChoice.budget));
    }
    if (!skinChoice.notCached.empty())
      output.log.print("; not cached: %s", skinChoice.notCached.c_str());
    if (!skinChoice.cachePath.empty())
      output.log.print("; %s '%s'", skinChoice.recorded ? "from" : "stored in",
                       skinChoice.cachePath.c_str());
    output.log.print(")\n");
  }
  output.log.print("MDIR: compiled in %.2f s\n", compileTime);
  output.log.print("%s", cacheLine.c_str());
  output.log.print("%s", fftPlanLine.c_str());
  output.log.print("%s", algPolicyLine.c_str());
  if (output.models)
    output.log.print("%s", output.models->describe().c_str());
  writeLogHeader(output);

  // SIGTERM and SIGINT ask the run to stop at its next checkpoint (D131).
  if (stops) {
    struct sigaction action = {};
    action.sa_handler = requestStop;
    action.sa_flags = SA_RESETHAND;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGINT, &action, nullptr);
  }

  // The store of the process for the structures of the tuples (D253, #316):
  // a segment begins in a new order of the particles and builds them in the
  // memory of the segment before, and takes them as they are only where the
  // identities and the members compare equal. It is open while the entry
  // runs.
  bool onDevice = control->target == Target::GPU;
  auto findRuntime = [&](StringRef host, StringRef device) -> void * {
    auto symbol = compiled.lookup(onDevice ? device : host);
    if (!symbol) {
      llvm::consumeError(symbol.takeError());
      return nullptr;
    }
    return *symbol;
  };
  void *tupleStore = nullptr;
  if (void *open = findRuntime("mdrtTuplesOpen", "mdrtDeviceTuplesOpen"))
    tupleStore = reinterpret_cast<void *(*)()>(open)();
  if (tupleStore)
    if (void *enter = findRuntime("mdrtTuplesEnter", "mdrtDeviceTuplesEnter"))
      reinterpret_cast<void (*)(void *)>(enter)(tupleStore);

  std::optional<DistributionLog> distLog;
  using Observe = void (*)(DistObserver *);
  Observe observe = nullptr;
  if (control->distribution) {
    const auto &g = program->domainGrid;
    distLog.emplace(output.log, manifest, distRank, distRanks,
                    g[0] * g[1] * g[2], firstStep, control->energyPeriod,
                    control->pairlistDistance);
    bool opened = distLog->open(std::getenv("MDIR_DIST_STEP_TRACE"),
                                control->restartOutput);
    if (!opened) {
      output.log.print("MDIR: distributed setup refused: %s\n",
                       distLog->error.c_str());
      if (auto e = manifest.finish("refused", firstStep, distLog->error))
        return fail(std::move(e));
      return fail(distLog->error);
    }
    observe = reinterpret_cast<Observe>(
        findRuntime("mdrtDistObserve", "mdrtDistObserve"));
    if (!observe)
      return fail("distributed statistics require mdrtDistObserve");
    observe(&distLog->observer);
    output.finishDistribution = [&] {
      observe(nullptr);
      distLog->finish();
    };
  }
  // The work between host boundaries is recorded and a sequence that
  // repeats is replayed as a CUDA graph (D448).
  // The choice, with the overrides of the environment, is spansOn.
  bool spans = onDevice && spansOn;
  using SpansEnable = void (*)(int32_t);
  auto spansEnable = reinterpret_cast<SpansEnable>(
      onDevice ? findRuntime("", "mdrtSpansEnable") : nullptr);
  if (spansEnable)
    spansEnable(spans);

  begin = std::chrono::steady_clock::now();
  output.lastCheckpoint = begin;
  compiled.entry(arguments.data());
  if (spansEnable)
    spansEnable(0);
  if (distLog) {
    output.finishDistribution();
  }
  // The wait of the end of the run: the checkpoint at the last step is
  // written by a thread of its own (D336), and the run ends
  // once it has passed all three of its events (CheckpointThread in
  // lib/Driver/Output.cpp), the last its commit to stable storage.
  waitForCheckpoint();
  // And for the frames that the thread of the frames still holds
  // (D400).
  finishFrames(output);
  if (distLog && !distLog->error.empty())
    return fail(distLog->error);
  double runTime = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - begin)
                       .count();
  // How many structures of the tuples the starts of the segments took as
  // they were kept and how many they built.
  int64_t tupleCounts[2] = {-1, -1};
  if (tupleStore) {
    if (void *read = findRuntime("mdrtTuplesCounts", "mdrtDeviceTuplesCounts"))
      reinterpret_cast<void (*)(void *, int64_t *)>(read)(tupleStore,
                                                          tupleCounts);
    if (void *leave = findRuntime("mdrtTuplesLeave", "mdrtDeviceTuplesLeave"))
      reinterpret_cast<void (*)()>(leave)();
    if (void *close = findRuntime("mdrtTuplesClose", "mdrtDeviceTuplesClose"))
      reinterpret_cast<void (*)(void *)>(close)(tupleStore);
  }
  if (trial) {
    auto result = compiled.lookup("mdrtStepTrialResult");
    if (!result) return fail(result.takeError());
    if (!reinterpret_cast<int (*)(double *, int64_t *, int64_t *, double *, double *, double *)>(*result)(
            &trial->us, &trial->measured, &trial->cold, &trial->instantiateSeconds,
            &trial->warmSeconds, &trial->timedSeconds))
      trial->refusal = "too few warmed ordinary chunks to measure replay";
    // Trial builds and prunes belong to the disposable run.
    if (auto reset = findRuntime("mdrtResetCounts", "mdrtResetCounts"))
      reinterpret_cast<void (*)()>(reset)();
    setOutput(nullptr);
  }
  if (output.trajectory)
    output.trajectory->close();
  if (llvm::Error error = manifest.finish("completed", control->minimize
                                              ? output.lastMinimizationStep
                                              : output.endStep))
    return fail(std::move(error));


  output.log.print("MDIR: ran in %.2f s", runTime);
  if (!control->minimize && control->numSteps > 0 && runTime > 0.0) {
    double simulated = control->timestep * control->numSteps * 1.0e-3;
    output.log.print(", %.2f ms per step, %.1f ns per day",
                 1.0e3 * runTime / control->numSteps,
                 simulated * 86400.0 / runTime);
  }
  output.log.print("\n");
  if (output.models)
    output.log.print("%s", output.models->describeUse().c_str());
  // The rate past the start: from the first output of the energies at or
  // after half the steps to the last, which wait for the device.
  if (!control->minimize && output.numEnergyTimes >= 2) {
    const auto &from = output.rateFrom;
    const auto &last = output.rateLast;
    if (from && from->first < last.first) {
      double seconds = last.second - from->second;
      int64_t steps = last.first - from->first;
      double simulated = control->timestep * steps * 1.0e-3;
      output.log.print(
                   "MDIR: from step %lld to step %lld, %.3f ms per step, "
                   "%.1f ns per day\n",
                   static_cast<long long>(from->first),
                   static_cast<long long>(last.first),
                   1.0e3 * seconds / steps, simulated * 86400.0 / seconds);
    }
  }
  if (writesCheckpoints)
    output.log.print("MDIR: wrote %lld checkpoints to '%s'\n",
                 static_cast<long long>(output.numCheckpoints),
                 output.checkpointPath.c_str());

  // How much of the blocks of the lists of groups the builds took.
  auto readCount = [&](StringRef name) -> int64_t {
    auto symbol = compiled.lookup(name);
    if (!symbol) {
      llvm::consumeError(symbol.takeError());
      return 0;
    }
    return reinterpret_cast<int64_t (*)()>(*symbol)();
  };
  if (int64_t used = readCount("mdrtGetGroupsBlocks"))
    output.log.print(
                 "MDIR: the lists of groups took at most %lld of %lld "
                 "blocks of 64 entries; the longest held %lld\n",
                 static_cast<long long>(used),
                 static_cast<long long>(readCount("mdrtGetGroupsCapacity")),
                 static_cast<long long>(readCount("mdrtGetGroupsLongest")));
  // The first build of a structure sizes its lists (D331); a
  // build that needed more room later made it and was made again.
  if (int64_t grew = readCount("mdrtGetGroupsGrowths"))
    output.log.print(
                 "MDIR: the lists of groups grew %lld times after the first "
                 "build of their structure, which sizes them\n",
                 static_cast<long long>(grew));
  if (int64_t over = readCount("mdrtGetGroupsOverflow"))
    output.log.print(
                 "MDIR: %lld groups had more partners of excluded pairs than "
                 "the memory of a warp holds, 256, and took their excluded "
                 "pairs from the rows\n",
                 static_cast<long long>(over));

  // What the spans issued (D448).
  if (void *read = onDevice ? findRuntime("", "mdrtSpansCounts") : nullptr) {
    int64_t counts[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    reinterpret_cast<void (*)(int64_t *)>(read)(counts);
    if (counts[0] + counts[1] + counts[8] > 0)
      output.log.print(
          "MDIR: spans: %lld issued as CUDA graphs and %lld as recorded, "
          "%lld graphs made, %lld captures refused, %lld graphs released, "
          "%lld kernel nodes updated, %lld updates refused; %lld kernels and "
          "transforms recorded, %lld launches not recorded for their "
          "arguments\n",
          static_cast<long long>(counts[0]), static_cast<long long>(counts[1]),
          static_cast<long long>(counts[2]), static_cast<long long>(counts[3]),
          static_cast<long long>(counts[5]), static_cast<long long>(counts[6]),
          static_cast<long long>(counts[7]), static_cast<long long>(counts[4]),
          static_cast<long long>(counts[8]));
  }

  // The runtime has counted the builds of the neighbor structures.
  if (auto count = compiled.lookup("mdrtGetBuildCount")) {
    auto getCount = reinterpret_cast<int64_t (*)()>(*count);
    int64_t builds = getCount();
    output.log.print("MDIR: neighbor structures were built %lld "
                             "times",
                 static_cast<long long>(builds));
    if (builds > 1 && control->numSteps > 0)
      output.log.print(", every %.1f steps on average",
                   static_cast<double>(control->numSteps) /
                       static_cast<double>(builds - 1));
    output.log.print("\n");
  } else {
    llvm::consumeError(count.takeError());
  }
  // A diagnostic of the check of the kept structures (the maintainer's
  // ruling on PR #327): a run without the variable prints what it printed.
  if (tupleCounts[0] + tupleCounts[1] > 0 && getenv("MDRT_CHECK_KEPT_TUPLES"))
    output.log.print("MDIR: the structures of the tuples were built %lld "
                     "times and taken as they were kept %lld times\n",
                     static_cast<long long>(tupleCounts[1]),
                     static_cast<long long>(tupleCounts[0]));
  // The prunings of the inner lists of dual lists (D114).
  if (auto count = compiled.lookup("mdrtGetPruneCount")) {
    int64_t prunes = reinterpret_cast<int64_t (*)()>(*count)();
    if (prunes > 0) {
      output.log.print("MDIR: the inner lists were pruned %lld times",
                   static_cast<long long>(prunes));
      if (prunes > 1 && control->numSteps > 0)
        output.log.print(", every %.1f steps on average",
                     static_cast<double>(control->numSteps) /
                         static_cast<double>(prunes - 1));
      // Those that the loop over pairs did (D284).
      if (auto fused = compiled.lookup("mdrtGetFusedPruneCount")) {
        int64_t inLoop = reinterpret_cast<int64_t (*)()>(*fused)();
        if (inLoop > 0)
          output.log.print(", %lld of them in the loop over pairs",
                           static_cast<long long>(inLoop));
      } else {
        llvm::consumeError(fused.takeError());
      }
      output.log.print("\n");
    }
  } else {
    llvm::consumeError(count.takeError());
  }
  // At a fixed interval, the builds that found a structure no longer valid:
  // pairs within the cutoff may have been missed before each (D88).
  if (control->rebuildPeriod > 0) {
    if (auto count = compiled.lookup("mdrtGetLateBuildCount")) {
      int64_t late = reinterpret_cast<int64_t (*)()>(*count)();
      output.log.print(
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
  // The pressures that semi-isotropic coupling took, which a bilayer at
  // its tension balances (D119).
  if (control->barostat && control->semiIsotropic) {
    if (auto get = compiled.lookup("mdrtGetSemiPressures")) {
      double p[6];
      int64_t n = reinterpret_cast<int64_t (*)(double *)>(*get)(p);
      output.log.print(
                   "MDIR: the pressures of the barostat over %lld periods, "
                   "in bar: x and y %.2f +/- %.2f, z %.2f +/- %.2f, the "
                   "difference %.2f +/- %.2f\n",
                   static_cast<long long>(n), p[0], p[3], p[1], p[4], p[2],
                   p[5]);
    } else {
      llvm::consumeError(get.takeError());
    }
  }
  // The pressures of the axes that anisotropic coupling took (D163c).
  if (control->barostat && control->anisotropic) {
    if (auto get = compiled.lookup("mdrtGetAxisPressures")) {
      double p[6];
      int64_t n = reinterpret_cast<int64_t (*)(double *)>(*get)(p);
      output.log.print(
          "MDIR: the pressures of the barostat over %lld periods, in bar: "
          "x %.2f +/- %.2f, y %.2f +/- %.2f, z %.2f +/- %.2f\n",
          static_cast<long long>(n), p[0], p[3], p[1], p[4], p[2], p[5]);
    } else {
      llvm::consumeError(get.takeError());
    }
  }
  // The temperatures of the solute and of the solvent, the rigid waters,
  // over the rows of the log: their means, with errors from ten blocks of
  // rows (D203). A gradient between them at a time step
  // is the hot-solvent/cold-solute problem of Lingenheil et al. (2008).
  // They are sums taken as the rows came (#300), with blocks of a tenth of
  // the rows that the run was to write.
  if (!output.solventOptimal.empty()) {
    auto summarize = [](const BlockMean &values) {
      return std::make_pair(values.getMean(), values.getError());
    };
    auto solvent = summarize(output.solventOptimal);
    auto solventFull = summarize(output.solventFull);
    if (!output.soluteOptimal.empty()) {
      auto solute = summarize(output.soluteOptimal);
      auto soluteFull = summarize(output.soluteFull);
      output.log.print(
          "MDIR: the temperatures over %zu rows, in K: of the solute "
          "%.2f +/- %.2f, of the solvent %.2f +/- %.2f; from the velocities of "
          "the steps alone, %.2f +/- %.2f and %.2f +/- %.2f\n",
          static_cast<size_t>(output.solventOptimal.count), solute.first,
          solute.second,
          solvent.first, solvent.second, soluteFull.first, soluteFull.second,
          solventFull.first, solventFull.second);
    } else {
      output.log.print(
          "MDIR: the temperature of the solvent over %zu rows, in K: "
          "%.2f +/- %.2f; from the velocities of the steps alone, %.2f +/- "
          "%.2f\n",
          static_cast<size_t>(output.solventOptimal.count), solvent.first,
          solvent.second, solventFull.first, solventFull.second);
    }
    // The blocks are those of the rows only if the run wrote the rows
    // that it planned.
    if (!output.solventOptimal.isComplete())
      output.log.print(
          "MDIR: the run wrote %lld rows of these temperatures and planned "
          "%lld, so their errors are of %lld blocks of %lld rows; this is "
          "a defect of mdir\n",
          static_cast<long long>(output.solventOptimal.count),
          static_cast<long long>(output.solventOptimal.planned),
          static_cast<long long>(output.solventOptimal.blocksDone),
          static_cast<long long>(output.solventOptimal.blockSize));
  }
  // The momentum of the state at the end, which the removal of the motion
  // of the center of mass keeps at 0.
  double momentum[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0, e = system->getNumParticles(); i != e; ++i)
    for (int c = 0; c != 3; ++c)
      momentum[c] += system->masses[i] * system->velocities[3 * i + c];
  output.log.print(
               "MDIR: the momentum at the end is %.3e amu nm/ps\n",
               std::sqrt(momentum[0] * momentum[0] +
                         momentum[1] * momentum[1] +
                         momentum[2] * momentum[2]));
  if (control->minimize) {
    if (output.hasEnergies)
      output.log.print(
                   "MDIR: the potential energy went from %.4f to %.4f "
                   "kcal/mol\n",
                   output.firstTotal / units::energy,
                   output.lastTotal / units::energy);
    // Whether the largest force fell below the tolerance
    // (D219), at a row of the energies, or the step below its
    // floor (D274).
    double largest = output.lastMinimization.maxForce /
                     (units::energy / units::length);
    if (output.stalledStep >= 0) {
      output.log.print(
          "MDIR: stopped at step %lld: the step has shrunk to %.3e Ang, "
          "below %.0e Ang (%.0e of its cap of %g Ang), where a trial no longer "
          "lowers the energy by more than its rounding; the largest force "
          "is %.4f kcal/mol/Ang",
          static_cast<long long>(output.stalledStep),
          output.lastMinimization.stepSize / units::length,
          control->getMinimizeStepFloor(),
          control->getMinimizeStepFloor() / Control::minimizeStepCap,
          Control::minimizeStepCap, largest);
      if (control->minimizeTolerance > 0.0)
        output.log.print(", not below %g", control->minimizeTolerance);
      output.log.print("\n");
    } else if (control->minimizeTolerance > 0.0) {
      if (output.convergedStep >= 0)
        output.log.print("MDIR: converged at step %lld: the largest force, "
                         "%.4f kcal/mol/Ang, is below %g\n",
                         static_cast<long long>(output.convergedStep),
                         largest, control->minimizeTolerance);
      else
        output.log.print("MDIR: not converged in %lld steps: the largest "
                         "force, %.4f kcal/mol/Ang, is not below %g\n",
                         static_cast<long long>(control->numSteps), largest,
                         control->minimizeTolerance);
    }
    return 0;
  }
  if (control->isLangevin() || control->isBrownian())
    return 0;
  if (output.hasEnergies && output.firstTotal != 0.0)
    output.log.print(
                 "MDIR: the %s energy changed by %.3e of its value\n",
                 output.couples ? "conserved" : "total",
                 std::fabs((output.lastTotal - output.firstTotal) /
                           output.firstTotal));
  return 0;
}

int runControl(StringRef controlFile, Emit emit, const char *argv0, const RunOptions &options) {
  return runControlImpl(controlFile, emit, argv0, options, nullptr);
}
} // namespace mdir::tool
