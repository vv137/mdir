//===- ModelTerm.cpp - Learned potentials through the plugin --------------===//
//
// The core's side of D270 (docs/metatomic-ml1.md, Section 6):
// the plugin is loaded with dlopen (RTLD_LOCAL) the first time a run has a
// term of [[energy.model]], and stays loaded: libtorch is not unloaded. Its
// work goes to the stream of the kernels of MDIR's runtime, whose functions
// are found among the libraries that the program of the run has loaded.
//
//===----------------------------------------------------------------------===//

#include "mdir/Driver/ModelTerm.h"

#include "mdir/Driver/Output.h"
#include "mdir/MLIP/Plugin.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>

using namespace mdir;
using namespace mdir::driver;

namespace {

/// The functions of the plugin, found once for the process.
struct Plugin {
  std::string path;
  void *library = nullptr;
  int32_t (*abiVersion)(void) = nullptr;
  const char *(*buildVersions)(void) = nullptr;
  const char *(*torchVersion)(void) = nullptr;
  const char *(*lastError)(void) = nullptr;
  int (*load)(const mdir_mlip_load_options *, mdir_mlip_model **) = nullptr;
  int (*cutoff)(const mdir_mlip_load_options *, double *) = nullptr;
  int (*describe)(const mdir_mlip_model *, mdir_mlip_description *) = nullptr;
  int (*evaluate)(mdir_mlip_model *, const mdir_mlip_inputs *,
                  mdir_mlip_outputs *) = nullptr;
  int (*memoryStats)(mdir_mlip_model *, int32_t, mdir_mlip_memory *) = nullptr;
  void (*free)(mdir_mlip_model *) = nullptr;
};

std::mutex pluginMutex;
Plugin *loadedPlugin = nullptr;

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 message.str());
}

llvm::Expected<Plugin *> openPlugin(llvm::StringRef path) {
  std::lock_guard<std::mutex> lock(pluginMutex);
  if (loadedPlugin) {
    if (loadedPlugin->path != path)
      return makeError("the plugin '" + loadedPlugin->path +
                       "' is loaded in this process already; '" + path +
                       "' cannot be loaded beside it");
    return loadedPlugin;
  }
  // torch's allocator splits its segments where they can grow, unless the
  // user configured it (Section 6.5 of the design); it reads the variable
  // when it first allocates.
  setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", 0);
  void *library = dlopen(path.str().c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library)
    return makeError("cannot load the plugin of learned potentials '" + path +
                     "': " + dlerror() +
                     "; build MDIR with -DMDIR_ENABLE_METATOMIC=ON, or set "
                     "MDIR_MLIP_PLUGIN");
  auto plugin = std::make_unique<Plugin>();
  plugin->path = path.str();
  plugin->library = library;
  bool missing = false;
  auto find = [&](auto &function, const char *name) {
    function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
        dlsym(library, name));
    missing |= !function;
  };
  find(plugin->abiVersion, "mdir_mlip_abi_version");
  find(plugin->buildVersions, "mdir_mlip_build_versions");
  find(plugin->torchVersion, "mdir_mlip_torch_version");
  find(plugin->lastError, "mdir_mlip_last_error");
  find(plugin->load, "mdir_mlip_load");
  find(plugin->cutoff, "mdir_mlip_cutoff");
  find(plugin->describe, "mdir_mlip_describe");
  find(plugin->evaluate, "mdir_mlip_evaluate");
  find(plugin->memoryStats, "mdir_mlip_memory_stats");
  find(plugin->free, "mdir_mlip_free");
  // The version first: a plugin of another version may lack functions of
  // this one.
  if (plugin->abiVersion && plugin->abiVersion() != MDIR_MLIP_ABI_VERSION)
    return makeError(llvm::formatv("the plugin '{0}' has the boundary of "
                                   "version {1}; this MDIR takes {2}",
                                   path, plugin->abiVersion(),
                                   MDIR_MLIP_ABI_VERSION)
                         .str());
  if (missing)
    return makeError("'" + path + "' is not a plugin of learned potentials "
                     "of MDIR: a function of include/mdir/MLIP/Plugin.h is "
                     "missing");
  loadedPlugin = plugin.release();
  return loadedPlugin;
}

/// A function of MDIR's runtime for NVIDIA GPUs, which the run has loaded.
template <typename T> T findRuntime(const char *name) {
  return reinterpret_cast<T>(
      llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name));
}

/// The models of the simulation that is running.
ModelTerms *getRunning() {
  Output *output = getOutput();
  return output ? output->models.get() : nullptr;
}

} // namespace

namespace {
/// The directory of the runtime libraries (setPluginDirectory).
std::string pluginDirectory;
} // namespace

void mdir::driver::setPluginDirectory(llvm::StringRef directory) {
  std::lock_guard<std::mutex> lock(pluginMutex);
  pluginDirectory = directory.str();
}

llvm::Expected<double>
mdir::driver::getModelCutoff(const Control::ModelTerm &term) {
  std::string directory;
  {
    std::lock_guard<std::mutex> lock(pluginMutex);
    directory = pluginDirectory;
  }
  const char *given = std::getenv("MDIR_MLIP_PLUGIN");
  if (directory.empty() && !(given && *given))
    return makeError("the plugin of learned potentials is not found: the "
                     "directory of the runtime libraries is not known; set "
                     "MDIR_MLIP_PLUGIN");
  llvm::Expected<Plugin *> opened = openPlugin(getPluginPath(directory));
  if (!opened)
    return opened.takeError();
  mdir_mlip_load_options options{term.path.c_str(), term.backend.c_str(), -1,
                                 term.centersPerCall};
  double cutoff = 0.0;
  if ((*opened)->cutoff(&options, &cutoff) != MDIR_MLIP_OK)
    return makeError("cannot read the cutoff of the model '" + term.name +
                     "' from '" + term.path + "': " +
                     (*opened)->lastError());
  return cutoff;
}

llvm::Expected<int> mdir::driver::getPairsMode() {
  const char *given = std::getenv("MDIR_MLIP_PAIRS");
  llvm::StringRef mode = given ? given : "";
  if (mode.empty() || mode == "exact")
    return MDIR_MLIP_PAIRS_EXACT;
  if (mode == "padded")
    return MDIR_MLIP_PAIRS_PADDED;
  if (mode == "search")
    return MDIR_MLIP_PAIRS_SEARCH;
  return makeError("MDIR_MLIP_PAIRS is '" + mode +
                   "'; it takes 'exact', 'padded', or 'search'");
}

std::string mdir::driver::getPluginPath(llvm::StringRef directory) {
  if (const char *given = std::getenv("MDIR_MLIP_PLUGIN"); given && *given)
    return given;
  llvm::SmallString<256> path(directory);
  llvm::sys::path::append(path, "libmdir_metatomic.so");
  return std::string(path);
}

llvm::Expected<std::string>
mdir::driver::getPluginTorchVersion(llvm::StringRef path) {
  llvm::Expected<Plugin *> plugin = openPlugin(path);
  if (!plugin)
    return plugin.takeError();
  return std::string((*plugin)->torchVersion());
}

llvm::Expected<std::unique_ptr<ModelTerms>>
ModelTerms::load(const Control &control, const System &system,
                 llvm::StringRef directory) {
  std::string path = getPluginPath(directory);
  if (!system.topology)
    return makeError("a term of [[energy.model]] needs a topology, which "
                     "gives the elements of the particles");
  const Topology &topology = *system.topology;
  llvm::Expected<Plugin *> opened = openPlugin(path);
  if (!opened)
    return opened.takeError();
  Plugin &plugin = **opened;
  auto allocate = findRuntime<void *(*)(uint64_t)>("mdrtDeviceAllocateRecorded");
  auto ordinal = findRuntime<int32_t (*)(void)>("mdrtDeviceOrdinal");
  if (!allocate || !ordinal)
    return makeError("a term of [[energy.model]] runs on a GPU, and the "
                     "runtime for NVIDIA GPUs is not loaded");

  int device = ordinal();
  std::unique_ptr<ModelTerms> terms(new ModelTerms());
  terms->count = static_cast<int64_t>(topology.getNumParticles());
  llvm::Expected<int> mode = getPairsMode();
  if (!mode)
    return mode.takeError();
  terms->pairsMode = *mode;

  for (const Control::ModelTerm &term : control.models) {
    Model model;
    model.name = term.name;
    mdir_mlip_load_options options{term.path.c_str(), term.backend.c_str(),
                                   device, term.centersPerCall};
    if (plugin.load(&options, &model.handle) != MDIR_MLIP_OK)
      return makeError("cannot load the model '" + term.name + "' from '" +
                       term.path + "': " + plugin.lastError());
    terms->models.push_back(model);
    Model &added = terms->models.back();
    mdir_mlip_description d;
    if (plugin.describe(added.handle, &d) != MDIR_MLIP_OK)
      return makeError("cannot describe the model '" + term.name +
                       "': " + plugin.lastError());

    // Every particle with mass is the model's, with an element it takes
    // (Section 6.3 of the design).
    std::vector<int64_t> known(d.types, d.types + d.num_types);
    if (topology.atomicNumbers.size() != topology.getNumParticles())
      return makeError("the topology gives no elements, which the model '" +
                       term.name + "' takes");
    for (size_t i = 0; i != topology.getNumParticles(); ++i) {
      int number = topology.atomicNumbers[i];
      if (topology.masses[i] <= 0.0)
        return makeError(llvm::formatv(
            "particle {0} has no mass; the model '{1}' takes every particle, "
            "and ML1 has no particle without mass beside it",
            i + 1, term.name));
      if (number <= 0)
        return makeError(llvm::formatv(
            "particle {0} has no element, which the model '{1}' takes",
            i + 1, term.name));
      if (!llvm::is_contained(known, number))
        return makeError(llvm::formatv(
            "particle {0} is of the element {1}, which the model '{2}' does "
            "not take",
            i + 1, number, term.name));
    }

    added.description = llvm::formatv(
        "MDIR: model '{0}' ({1}, {2}): {12}, {3}; cutoff {4} nm, "
        "{5} pair list; inputs of {6} bits; units {7} and {8}, "
        "factors {9} and {10} to nm and kJ/mol; {11}; {13}\n",
        term.name, d.name, term.path, d.framework, d.cutoff,
        d.full_list ? "full" : "half", d.dtype_bits, d.length_unit,
        d.energy_unit, d.length_factor, d.energy_factor,
        d.centers_per_call > 0
            ? llvm::formatv("{0} centers per call", d.centers_per_call).str()
        : d.local_centers ? std::string("local, the whole system in one call")
        : term.centersPerCall > 0
            ? std::string("not declared local: the whole system in one "
                          "call, whatever 'centers_per_call' asks")
            : std::string("the whole system in one call"),
        term.backend == "AOTI" ? "AOTInductor package" : "metatomic",
        // How the model takes its pair list (D277).
        terms->pairsMode == MDIR_MLIP_PAIRS_EXACT
            ? "the pair list MDIR's, its pairs within the cutoff"
        : terms->pairsMode == MDIR_MLIP_PAIRS_PADDED
            ? "the pair list MDIR's, every slot within the cutoff and the "
              "skin (MDIR_MLIP_PAIRS=padded)"
            : "the plugin's own pair search (MDIR_MLIP_PAIRS=search)");

    added.results = static_cast<double *>(allocate(sizeof(double) * 10));
  }
  return std::move(terms);
}

ModelTerms::~ModelTerms() {
  auto release = findRuntime<void (*)(void *)>("mdrtDeviceFreeRecorded");
  for (Model &model : models) {
    if (release)
      release(model.results);
    if (loadedPlugin && model.handle)
      loadedPlugin->free(model.handle);
  }
}

std::string ModelTerms::evaluate(int64_t index, const void *positions,
                                 int bits, const int32_t *numbers, int64_t n,
                                 const double cell[9], void *forces,
                                 int forceBits, const int32_t *slots,
                                 int64_t numSlots, const int32_t *pairCounts,
                                 const int32_t *pairOffsets, int64_t builds) {
  static auto stream = findRuntime<void *(*)(void)>("mdrtDeviceExternalStream");
  if (!stream)
    return "the runtime for NVIDIA GPUs is not loaded";
  Model &model = models[index];
  if (n != count)
    return llvm::formatv("the model '{0}' is given {1} particles, not {2}",
                         model.name, n, count)
        .str();
  mdir_mlip_inputs inputs{};
  inputs.count = n;
  inputs.positions = positions;
  inputs.position_bits = bits;
  inputs.numbers = numbers;
  std::copy(cell, cell + 9, inputs.cell);
  inputs.periodic[0] = inputs.periodic[1] = inputs.periodic[2] = 1;
  inputs.stream = stream();
  // MDIR's pair list (D277), where the program built it.
  if (pairCounts) {
    inputs.pairs.mode = pairsMode;
    inputs.pairs.slots = slots;
    inputs.pairs.num_slots = numSlots;
    inputs.pairs.counts = pairCounts;
    inputs.pairs.offsets = pairOffsets;
    inputs.pairs.builds = builds;
  } else {
    inputs.pairs.mode = MDIR_MLIP_PAIRS_SEARCH;
  }
  // The plugin adds its forces to those of the program, where they are.
  mdir_mlip_outputs outputs{};
  outputs.forces = forces;
  outputs.force_bits = forceBits;
  outputs.energy = model.results;
  outputs.virial = model.results + 1;
  if (loadedPlugin->evaluate(model.handle, &inputs, &outputs) != MDIR_MLIP_OK)
    return "the model '" + model.name +
           "' failed: " + loadedPlugin->lastError();
  model.pairs = outputs.pairs;
  model.calls = outputs.calls;
  ++model.evaluations;
  return std::string();
}

void ModelTerms::readLast(int64_t index, double &energy,
                          double virial[9]) const {
  static auto copy =
      findRuntime<void (*)(void *, const void *, uint64_t)>("mdrtDeviceCopyToHost");
  double results[10] = {};
  if (copy && models[index].evaluations > 0)
    copy(results, models[index].results, sizeof results);
  energy = results[0];
  std::copy(results + 1, results + 10, virial);
}

void ModelTerms::read(double &energy, std::array<double, 9> &virial) const {
  static auto copy =
      findRuntime<void (*)(void *, const void *, uint64_t)>("mdrtDeviceCopyToHost");
  energy = 0.0;
  virial.fill(0.0);
  for (const Model &model : models) {
    double results[10] = {};
    if (copy && model.evaluations > 0)
      copy(results, model.results, sizeof results);
    energy += results[0];
    for (int k = 0; k != 9; ++k)
      virial[k] += results[1 + k];
  }
}

std::string ModelTerms::describe() const {
  std::string text;
  for (const Model &model : models)
    text += model.description;
  return text;
}

std::string ModelTerms::describeUse() const {
  std::string text;
  for (const Model &model : models) {
    mdir_mlip_memory memory{};
    if (loadedPlugin->memoryStats(model.handle, 0, &memory) != MDIR_MLIP_OK)
      continue;
    text += llvm::formatv(
        "MDIR: model '{0}': {1} evaluations, {2} pairs and {3} calls in the "
        "last; the plugin's allocator held at most {4:F1} MiB allocated, "
        "{5:F1} MiB reserved\n",
        model.name, model.evaluations, model.pairs, model.calls,
        memory.peak_allocated / 1048576.0, memory.peak_reserved / 1048576.0);
  }
  return text;
}

std::vector<std::string> ModelTerms::getNames() const {
  std::vector<std::string> names;
  for (const Model &model : models)
    names.push_back(model.name);
  return names;
}

//===----------------------------------------------------------------------===//
// The functions that the program calls
//===----------------------------------------------------------------------===//

namespace {
/// Evaluates model `index` for the program (D270); with
/// `energy` and `virial`, writes its energy and its virial there.
void evaluateModel(int64_t index, void *given, void *atomic, double lx,
                   double ly, double lz, double bx, double cx, double cy,
                   int64_t bits, int64_t forceBits, void *added,
                   void *energy, void *virial, void *slotsGiven = nullptr,
                   void *countsGiven = nullptr, void *offsetsGiven = nullptr,
                   int64_t builds = 0) {
  auto *positions = static_cast<StridedMemRefType<char, 2> *>(given);
  // MDIR's pair list (D277): buffers of the device.
  const int32_t *slots = nullptr, *pairCounts = nullptr,
                *pairOffsets = nullptr;
  int64_t numSlots = 0;
  if (countsGiven) {
    auto *s = static_cast<StridedMemRefType<int32_t, 2> *>(slotsGiven);
    auto *c = static_cast<StridedMemRefType<int32_t, 1> *>(countsGiven);
    auto *o = static_cast<StridedMemRefType<int32_t, 1> *>(offsetsGiven);
    slots = s->data + s->offset;
    numSlots = s->sizes[0];
    pairCounts = c->data + c->offset;
    pairOffsets = o->data + o->offset;
  }
  auto *numbers = static_cast<StridedMemRefType<int32_t, 1> *>(atomic);
  auto *forces = static_cast<StridedMemRefType<char, 2> *>(added);
  // The edge vectors a = (l_x, 0, 0), b = (b_x, l_y, 0), c = (c_x, c_y, l_z),
  // row by row.
  double cell[9] = {lx, 0.0, 0.0, bx, ly, 0.0, cx, cy, lz};
  ModelTerms *models = getRunning();
  std::string failure =
      models ? models->evaluate(
                   index, positions->data + positions->offset * (bits / 8),
                   static_cast<int>(bits), numbers->data + numbers->offset,
                   positions->sizes[0], cell,
                   forces->data + forces->offset * (forceBits / 8),
                   static_cast<int>(forceBits), slots, numSlots, pairCounts,
                   pairOffsets, builds)
             : std::string("a model term is evaluated, but no model is loaded");
  if (!failure.empty())
    stopRun(failure);
  if (!energy)
    return;
  auto *u = static_cast<StridedMemRefType<double, 1> *>(energy);
  auto *w = static_cast<StridedMemRefType<double, 1> *>(virial);
  double tensor[9];
  models->readLast(index, u->data[u->offset], tensor);
  for (int k = 0; k != 9; ++k)
    w->data[w->offset + k * w->strides[0]] = tensor[k];
}
} // namespace

extern "C" void _mlir_ciface_mdrtModelEvaluate(
    int64_t index, void *positions, void *numbers, double lx, double ly,
    double lz, double bx, double cx, double cy, int64_t bits,
    int64_t forceBits, void *forces) {
  evaluateModel(index, positions, numbers, lx, ly, lz, bx, cx, cy, bits,
                forceBits, forces, nullptr, nullptr);
}

extern "C" void _mlir_ciface_mdrtModelEvaluateEnergy(
    int64_t index, void *positions, void *numbers, double lx, double ly,
    double lz, double bx, double cx, double cy, int64_t bits,
    int64_t forceBits, void *forces, void *energy, void *virial) {
  evaluateModel(index, positions, numbers, lx, ly, lz, bx, cx, cy, bits,
                forceBits, forces, energy, virial);
}

extern "C" void _mlir_ciface_mdrtModelEvaluateEdges(
    int64_t index, void *positions, void *numbers, void *slots, void *counts,
    void *offsets, int64_t builds, double lx, double ly, double lz,
    double bx, double cx, double cy, int64_t bits, int64_t forceBits,
    void *forces) {
  evaluateModel(index, positions, numbers, lx, ly, lz, bx, cx, cy, bits,
                forceBits, forces, nullptr, nullptr, slots, counts, offsets,
                builds);
}

extern "C" void _mlir_ciface_mdrtModelEvaluateEdgesEnergy(
    int64_t index, void *positions, void *numbers, void *slots, void *counts,
    void *offsets, int64_t builds, double lx, double ly, double lz,
    double bx, double cx, double cy, int64_t bits, int64_t forceBits,
    void *forces, void *energy, void *virial) {
  evaluateModel(index, positions, numbers, lx, ly, lz, bx, cx, cy, bits,
                forceBits, forces, energy, virial, slots, counts, offsets,
                builds);
}
