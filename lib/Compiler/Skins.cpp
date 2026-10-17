// The automatic choice of the skins of the neighbor lists (D281):
// a cost model narrows the reaches to a few candidates, each is compiled
// and measured on the system for a few blocks of steps, and the fastest is
// kept, unless it is within the noise of the reference. The rebuilds and
// prunings are exact (a list is rebuilt when a particle has moved half its
// skin, D80, D114), so every candidate takes the same pairs within the
// cutoff: the choice changes the speed, not the forces.
//
// See docs/neighbors-m0.md, "The choice of the skins".

#include "mdir/Compiler/Skins.h"
#include "mdir/Compiler/Simulation.h"
#include "mdir/Driver/Model.h"
#include "mdir/Driver/Topology.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>

using namespace mdir;
using namespace mdir::driver;
using llvm::StringRef;

namespace {

/// The product of particles and steps from which a run is long enough to
/// search: about five hours of an RTX 4090 at about 4 ns a particle and a
/// step (the Amber suite: JAC 5.5, Cellulose 3.6, STMV 4.3), so that a
/// search of 1.5 to 4 minutes, a compilation of 13 to 35 s a candidate,
/// costs at most about 1% of the run and gains 1 to 4% (D281).
constexpr double searchParticleSteps = 5e12;

/// The constants of the cost model of a step, fitted to the grid of #406 on
/// an RTX 4090 (480 runs of the Amber suite in mixed precision, outer skins
/// of 1.5 to 6 Å and inner ones of 0.45 to 1.0 Å at 2 fs, the cutoff 8 Å;
/// the root mean square of the relative residuals 1.0%), per particle at
/// the density of water, `referenceDensity`:
///   t / N = a + c1 ρ' (r_c + s_in)^3 + c2 ρ' (r_c + s_out)^3 / n_p
///             + (c3 + c4 ρ' (r_c + s_out)^3) / n_b + L / (N n_b),
/// ρ' = ρ / referenceDensity, t in µs: the loop over the inner list, the
/// pruning of the outer one every n_p steps, and its build every n_b steps
/// with a fixed cost L of its launches. On another device the constants are
/// used as they are; the measurement decides.
struct CostModel {
  static constexpr double referenceDensity = 0.1; // particles per Å^3
  static constexpr double c1 = 1.3079e-6, c2 = 0.3653e-6, c3 = 2.1188e-3,
                          c4 = 0.0814e-6, launch = 74.41;
  /// The steps between builds and between prunings grow as powers of the
  /// skins (the means of the fits within each system of the grid).
  static constexpr double buildPower = 1.57, prunePower = 0.76;

  double particles, density, cutoff;
  /// The reference's skins and its measured intervals.
  double outer0, inner0, buildInterval0, pruneInterval0;

  /// The time of a step in µs, but for the constant a.
  double predict(double outer, double inner) const {
    double scale = density / referenceDensity;
    double so = outer - cutoff;
    double nb = std::max(1.0, buildInterval0 *
                                  std::pow(so / (outer0 - cutoff), buildPower));
    double cube = std::pow(outer, 3.0);
    double t = (c3 * particles + c4 * scale * particles * cube) / nb +
               launch / nb;
    if (inner == 0.0)
      return t + c1 * scale * particles * cube;
    double si = inner - cutoff;
    double np = pruneInterval0 > 0.0 && inner0 > 0.0
                    ? std::max(1.0, pruneInterval0 *
                                        std::pow(si / (inner0 - cutoff),
                                                 prunePower))
                    : 1.0;
    return t + c1 * scale * particles * std::pow(inner, 3.0) +
           c2 * scale * particles * cube / np;
  }
};

/// MDIR_SKIN_SEARCH: "always", "never", or "auto" (the default).
StringRef getSearchPolicy() {
  const char *value = std::getenv("MDIR_SKIN_SEARCH");
  return value && *value ? StringRef(value) : StringRef("auto");
}

/// The directory of the cache, or empty if it is off.
std::string getCacheDirectory() {
  if (const char *off = std::getenv("MDIR_SKIN_CACHE"); off && StringRef(off) == "off")
    return "";
  if (const char *dir = std::getenv("MDIR_SKIN_CACHE_DIR"); dir && *dir)
    return dir;
  llvm::SmallString<256> path;
  if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
    path = xdg;
  } else if (const char *home = std::getenv("HOME"); home && *home) {
    path = home;
    llvm::sys::path::append(path, ".cache");
  } else {
    return "";
  }
  llvm::sys::path::append(path, "mdir", "skins");
  return std::string(path);
}

/// The name of the device that a GPU run takes: the visible one that
/// MDRT_DEVICE names, else the first; empty if the driver cannot tell.
std::string getDeviceName() {
  auto library = llvm::sys::DynamicLibrary::getPermanentLibrary("libcuda.so.1");
  if (!library.isValid())
    return "";
  auto init = reinterpret_cast<int (*)(unsigned)>(
      library.getAddressOfSymbol("cuInit"));
  auto get = reinterpret_cast<int (*)(int *, int)>(
      library.getAddressOfSymbol("cuDeviceGet"));
  auto name = reinterpret_cast<int (*)(char *, int, int)>(
      library.getAddressOfSymbol("cuDeviceGetName"));
  if (!init || !get || !name || init(0))
    return "";
  int ordinal = 0;
  if (const char *chosen = std::getenv("MDRT_DEVICE"); chosen && *chosen)
    ordinal = std::atoi(chosen);
  int device = 0;
  char text[256] = {};
  if (get(&device, ordinal) || name(text, sizeof(text), device))
    return "";
  return text;
}

std::string format(double value) {
  return llvm::formatv("{0}", llvm::format("%.6g", value)).str();
}

/// A function of the runtime that compiled code loaded, or null.
template <typename F> F *findRuntime(const char *name) {
  return reinterpret_cast<F *>(
      llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name));
}

int64_t readCount(const char *name) {
  auto *get = findRuntime<int64_t()>(name);
  return get ? get() : 0;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  size_t n = values.size();
  return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

/// The steps that a part of a trial run must hold whole: the periods of
/// the coupling and of the barostat.
int64_t getStepUnit(const Control &control) {
  int64_t unit = 1;
  for (int64_t period : {control.getCouplingPeriod(), control.barostatPeriod})
    if (period > 0)
      unit = std::lcm(unit, period);
  return unit;
}

int64_t roundUp(int64_t steps, int64_t unit) {
  return (steps + unit - 1) / unit * unit;
}

/// Compiles and runs one candidate from the state of `system`: a warm-up,
/// then blocks of about half a second each, timed one by one.
compiler::SkinTrial measure(const Control &control, const System &system,
                            double outer, double inner) {
  compiler::SkinTrial trial;
  trial.outer = outer;
  trial.inner = inner;
  model::PreparedModel prepared;
  prepared.control = control;
  prepared.control.pairlistDistance = outer;
  prepared.control.prunedDistance = inner;
  prepared.control.pairlistDistanceGiven = true;
  prepared.control.prunedDistanceGiven = inner != 0.0;
  prepared.control.pairlistDistanceAuto = false;
  prepared.control.prunedDistanceAuto = false;
  prepared.system = system;
  model::Execution &execution = prepared.execution;
  execution.target = control.target;
  execution.precision = control.precision;
  execution.threads = control.threads;
  execution.deterministic = control.deterministic;
  execution.reorder = control.reorder;
  execution.fastMath = control.fastMath;
  execution.neighborCapacity = control.neighborWidth;
  execution.neighborStructure = control.neighborStructure;

  auto began = std::chrono::steady_clock::now();
  auto seconds = [](std::chrono::steady_clock::time_point from) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         from)
        .count();
  };
  // Nothing is kept or cached: each candidate is compiled anew, which is
  // what the search costs (D281).
  auto simulation = compiler::Simulation::create(prepared, /*cache=*/true);
  if (!simulation) {
    trial.failure = llvm::toString(simulation.takeError());
    return trial;
  }
  trial.compileSeconds = seconds(began);
  compiler::CompileStats stats = (*simulation)->getCompileStats();
  trial.pipelineSeconds = stats.pipelineSeconds;
  trial.kernelSeconds = stats.gpuSerializeSeconds;
  trial.hostSeconds = stats.compileSeconds;
  int64_t unit = getStepUnit(control);
  auto run = [&](int64_t steps) -> double {
    auto from = std::chrono::steady_clock::now();
    auto taken = (*simulation)->run(steps);
    if (!taken) {
      trial.failure = llvm::toString(taken.takeError());
      return -1.0;
    }
    // A part ends at a boundary that reads the test of the state for
    // numbers from the device (mdrtPartBoundary), so it has waited for it.
    return seconds(from);
  };
  // The warm-up: the first build, the loading of the kernels, and the
  // first parts, whose length the simulation learns.
  int64_t warm = roundUp(200, unit);
  double warmSeconds = run(warm);
  if (warmSeconds < 0.0)
    return trial;
  warmSeconds = run(warm);
  if (warmSeconds < 0.0)
    return trial;
  double perStep = warmSeconds / warm;
  int64_t block =
      roundUp(std::max<int64_t>(200, static_cast<int64_t>(0.5 / perStep)), unit);
  constexpr int blocks = 6;
  std::vector<double> times;
  int64_t builds = readCount("mdrtGetBuildCount");
  int64_t prunes = readCount("mdrtGetPruneCount");
  auto from = std::chrono::steady_clock::now();
  for (int b = 0; b != blocks; ++b) {
    double t = run(block);
    if (t < 0.0)
      return trial;
    times.push_back(1e6 * t / block);
  }
  trial.runSeconds = seconds(from);
  builds = readCount("mdrtGetBuildCount") - builds;
  prunes = readCount("mdrtGetPruneCount") - prunes;
  int64_t steps = block * blocks;
  trial.buildInterval = static_cast<double>(steps) / std::max<int64_t>(1, builds);
  trial.pruneInterval =
      inner != 0.0 ? static_cast<double>(steps) / std::max<int64_t>(1, prunes)
                   : 0.0;
  trial.microseconds = median(times);
  double mean = std::accumulate(times.begin(), times.end(), 0.0) / blocks;
  double variance = 0.0;
  for (double t : times)
    variance += (t - mean) * (t - mean);
  variance /= blocks - 1;
  trial.error = std::sqrt(variance / blocks) / mean;
  return trial;
}

/// The candidates of the cost model besides the reference: its best pair,
/// and its best pair whose outer skin is at least 1 Å from that one.
std::vector<std::pair<double, double>>
proposeCandidates(const Control &control, const CostModel &model,
                  double outerReference, double innerReference,
                  double leastEdge) {
  double cutoff = control.cutoffDistance;
  std::vector<double> outers, inners;
  if (control.pairlistDistanceAuto) {
    for (double skin = 1.5; skin <= 6.0 + 1e-9; skin += 0.5)
      outers.push_back(cutoff + skin);
  } else {
    outers.push_back(outerReference);
  }
  bool dual = innerReference != 0.0;
  if (control.prunedDistanceAuto && dual) {
    // Up to 1.5 Å beyond the cutoff, by 0.05 Å.
    for (int k = 6; k <= 30; ++k)
      inners.push_back(cutoff + 0.05 * k);
  } else {
    inners.push_back(innerReference);
  }
  struct Scored {
    double outer, inner, time;
  };
  std::vector<Scored> scored;
  for (double outer : outers) {
    // The reach of the groups is below the least edge of the cell over
    // 0.999 (D242), and at most 127 times the cutoff (I2).
    if (outer > 0.999 * leastEdge || outer > 127.0 * cutoff)
      continue;
    for (double inner : inners) {
      if (inner != 0.0 && !(inner > cutoff && inner < outer - 0.25))
        continue;
      bool reference = std::fabs(outer - outerReference) < 1e-9 &&
                       std::fabs(inner - innerReference) < 1e-9;
      if (!reference)
        scored.push_back({outer, inner, model.predict(outer, inner)});
    }
  }
  std::sort(scored.begin(), scored.end(),
            [](const Scored &a, const Scored &b) { return a.time < b.time; });
  std::vector<std::pair<double, double>> chosen;
  if (scored.empty())
    return chosen;
  chosen.push_back({scored[0].outer, scored[0].inner});
  // A second candidate elsewhere: the landscape is not smooth (#406), so
  // the second is at least 1 Å away in the outer skin from the first and
  // from the reference where the outer is chosen, or 0.2 Å in the inner
  // one from both where only it is.
  for (const Scored &s : scored) {
    bool far =
        control.pairlistDistanceAuto
            ? std::fabs(s.outer - scored[0].outer) >= 1.0 - 1e-9 &&
                  std::fabs(s.outer - outerReference) >= 1.0 - 1e-9
            : std::fabs(s.inner - scored[0].inner) >= 0.2 - 1e-9 &&
                  std::fabs(s.inner - innerReference) >= 0.2 - 1e-9;
    if (far) {
      chosen.push_back({s.outer, s.inner});
      break;
    }
  }
  return chosen;
}

/// Reads the entry of the cache at `path` if its key is `key`.
std::optional<std::pair<double, double>> readCache(StringRef path,
                                                   StringRef key) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return std::nullopt;
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return std::nullopt;
  auto text = object->getString("key");
  auto outer = object->getNumber("pairlist_distance_angstrom");
  auto inner = object->getNumber("pruned_distance_angstrom");
  if (!text || *text != key || !outer || !inner)
    return std::nullopt;
  return std::make_pair(*outer, *inner);
}

void writeCache(StringRef path, StringRef key, const Control &control,
                const compiler::SkinChoice &choice) {
  llvm::SmallString<256> directory(llvm::sys::path::parent_path(path));
  if (llvm::sys::fs::create_directories(directory))
    return;
  // Written beside and renamed, so that a reader sees a whole entry.
  llvm::SmallString<256> temporary;
  int fd;
  if (llvm::sys::fs::createUniqueFile(
          llvm::Twine(path) + ".%%%%%%.tmp", fd, temporary))
    return;
  {
    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
    llvm::json::Object entry{
        {"key", key},
        {"pairlist_distance_angstrom", control.pairlistDistance},
        {"pruned_distance_angstrom", control.prunedDistance},
        {"search", choice.toJSON(control)}};
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(entry))) << "\n";
  }
  if (llvm::sys::fs::rename(temporary, path))
    llvm::sys::fs::remove(temporary);
}

} // namespace

bool compiler::canSearchSkins(const Control &control, std::string &why) {
  if (!control.isSkinAuto()) {
    why = "the reaches are given";
    return false;
  }
  if (control.target != Target::GPU) {
    why = "the CPU";
    return false;
  }
  // A search chooses by timing; the deterministic mode keeps the same bits
  // from run to run.
  if (control.deterministic) {
    why = "the deterministic mode";
    return false;
  }
  if (control.neighborStructure != NeighborStructure::Groups) {
    why = "the matrix, which is not searched";
    return false;
  }
  if (control.minimize) {
    why = "a minimization";
    return false;
  }
  if (control.rebuildPeriod > 0) {
    why = "'rebuild_interval' is given";
    return false;
  }
  if (!control.models.empty()) {
    why = "a learned potential, which is not searched";
    return false;
  }
  return true;
}

std::string compiler::getSkinCacheKey(const Control &control,
                                      const System &system) {
  // The composition: the types, masses, and charges of the particles.
  llvm::SHA256 hash;
  auto add = [&](const void *data, size_t bytes) {
    hash.update(llvm::ArrayRef<uint8_t>(static_cast<const uint8_t *>(data),
                                        bytes));
  };
  add(system.types.data(), system.types.size() * sizeof(unsigned));
  add(system.masses.data(), system.masses.size() * sizeof(double));
  if (system.topology)
    add(system.topology->charges.data(),
        system.topology->charges.size() * sizeof(double));
  std::string composition = llvm::toHex(hash.final(), /*LowerCase=*/true);
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "mdir skins 1\n"
     << "version " << MDIR_VERSION << "\n"
     << "particles " << system.getNumParticles() << "\n"
     << "composition " << composition << "\n"
     << "cutoff " << format(control.cutoffDistance) << "\n"
     << "structure "
     << (control.neighborStructure == NeighborStructure::Groups ? "groups"
                                                                 : "matrix")
     << "\n"
     << "pairlist_distance "
     << (control.pairlistDistanceAuto ? std::string("AUTO")
                                      : format(control.pairlistDistance))
     << "\n"
     << "pruned_distance "
     << (control.prunedDistanceAuto ? std::string("AUTO")
                                    : format(control.prunedDistance))
     << "\n"
     << "device " << getDeviceName() << "\n"
     << "precision "
     << (control.precision == Precision::Double  ? "double"
         : control.precision == Precision::Mixed ? "mixed"
                                                 : "single")
     << "\n"
     << "time_step " << format(control.timestep) << "\n"
     << "ensemble "
     << (control.barostat ? "NPT" : control.thermostat ? "NVT" : "NVE")
     << "\n"
     << "temperature " << format(control.temperature) << "\n";
  return text;
}

llvm::json::Value
compiler::SkinChoice::toJSON(const driver::Control &control) const {
  llvm::json::Array candidates;
  for (const SkinTrial &trial : trials) {
    llvm::json::Object object{
        {"pairlist_distance_angstrom", trial.outer},
        {"pruned_distance_angstrom",
         trial.inner != 0.0 ? llvm::json::Value(trial.inner)
                            : llvm::json::Value(nullptr)},
        {"predicted_relative", trial.predicted},
        {"compile_seconds", trial.compileSeconds},
        {"run_seconds", trial.runSeconds}};
    if (trial.failure.empty()) {
      object["microseconds_per_step"] = trial.microseconds;
      object["relative_error"] = trial.error;
      object["steps_per_build"] = trial.buildInterval;
      if (trial.inner != 0.0)
        object["steps_per_pruning"] = trial.pruneInterval;
    } else {
      object["failure"] = trial.failure;
    }
    candidates.push_back(std::move(object));
  }
  return llvm::json::Object{
      {"source", control.skinSource},
      {"reason", control.skinReason},
      {"cache", cachePath.empty() ? llvm::json::Value(nullptr)
                                  : llvm::json::Value(cachePath)},
      {"seconds", seconds},
      {"candidates", std::move(candidates)}};
}

compiler::SkinChoice compiler::chooseSkins(Control &control,
                                           const System &system, bool search,
                                           llvm::raw_ostream *log) {
  SkinChoice choice;
  if (!control.isSkinAuto()) {
    control.skinSource = "given";
    control.skinReason.clear();
    return choice;
  }
  // The defaults that resolveNeighborStructure set (D261).
  control.skinSource = "default";
  std::string why;
  if (!canSearchSkins(control, why)) {
    control.skinReason = why;
    return choice;
  }

  std::string key = getSkinCacheKey(control, system);
  std::string directory = getCacheDirectory();
  std::string path;
  if (!directory.empty()) {
    llvm::SmallString<256> file(directory);
    llvm::SHA256 hash;
    hash.update(key);
    llvm::sys::path::append(
        file, llvm::toHex(hash.final(), true).substr(0, 32) + ".json");
    path = std::string(file);
    if (auto cached = readCache(path, key)) {
      control.pairlistDistance = cached->first;
      control.prunedDistance = cached->second;
      control.skinSource = "cached";
      control.skinReason = "the choice of an earlier run of the same key";
      choice.cachePath = path;
      return choice;
    }
  }

  StringRef policy = getSearchPolicy();
  double particleSteps = static_cast<double>(system.getNumParticles()) *
                         static_cast<double>(control.numSteps);
  if (policy == "never") {
    control.skinReason = "MDIR_SKIN_SEARCH=never";
    return choice;
  }
  bool longEnough =
      policy == "always" || particleSteps >= searchParticleSteps;
  if (!search && longEnough) {
    control.skinReason = "`mdir run` measures candidates; this does not";
    return choice;
  }
  if (!longEnough) {
    control.skinReason = llvm::formatv("a short run: {0} particles times {1} "
                                       "steps, below {2:e0}",
                                       system.getNumParticles(),
                                       control.numSteps, searchParticleSteps)
                             .str();
    return choice;
  }

  auto began = std::chrono::steady_clock::now();
  double outer0 = control.pairlistDistance, inner0 = control.prunedDistance;
  // The reference of a search is the dual list of the groups of D261 for
  // the reaches that are chosen.
  if (control.pairlistDistanceAuto)
    outer0 = control.cutoffDistance + 3.0;
  if (control.prunedDistanceAuto)
    inner0 = control.cutoffDistance +
             0.6 * std::max(1.0, control.timestep / 0.002);
  if (inner0 >= outer0)
    inner0 = 0.0;
  double leastEdge = std::min({system.box[0], system.box[1], system.box[2]}) /
                     units::length;
  if (!control.periodic)
    leastEdge = HUGE_VAL;
  if (outer0 > 0.999 * leastEdge) {
    control.skinReason = "the cell is too small for the reference";
    return choice;
  }
  auto note = [&](const SkinTrial &trial) {
    if (!log)
      return;
    *log << llvm::formatv("MDIR: skins: candidate {0} / {1} Å: ",
                          format(trial.outer),
                          trial.inner != 0.0 ? format(trial.inner) : "-");
    if (trial.failure.empty())
      *log << llvm::formatv("{0:f2} µs a step ± {1:f2}%, builds every {2:f1} "
                            "steps, compiled in {3:f1} s (the pipeline "
                            "{4:f1} s, of which the kernels {5:f1} s; the "
                            "host code {6:f1} s)\n",
                            trial.microseconds, 100.0 * trial.error,
                            trial.buildInterval, trial.compileSeconds,
                            trial.pipelineSeconds, trial.kernelSeconds,
                            trial.hostSeconds);
    else
      *log << "failed: " << trial.failure << "\n";
    log->flush();
  };

  SkinTrial reference = measure(control, system, outer0, inner0);
  note(reference);
  choice.trials.push_back(reference);
  auto finish = [&] {
    choice.seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - began)
                         .count();
    // The trial runs' counts and the handler of failures of the simulations
    // are not the run's.
    if (auto *reset = findRuntime<void()>("mdrtResetCounts"))
      reset();
    if (auto *set = findRuntime<void(void (*)(const char *))>(
            "mdrtSetStopHandler"))
      set(nullptr);
    if (auto *set = findRuntime<void(void (*)(const char *))>(
            "mdrtCudaSetStopHandler"))
      set(nullptr);
  };
  if (!reference.failure.empty()) {
    finish();
    control.skinReason = "the reference could not be measured: " +
                         reference.failure;
    return choice;
  }

  double density = static_cast<double>(system.getNumParticles()) /
                   (system.box[0] * system.box[1] * system.box[2] /
                    std::pow(units::length, 3.0));
  CostModel model{static_cast<double>(system.getNumParticles()),
                  control.periodic ? density : CostModel::referenceDensity,
                  control.cutoffDistance,
                  outer0,
                  inner0,
                  reference.buildInterval,
                  reference.pruneInterval};
  double base = model.predict(outer0, inner0);
  choice.trials.front().predicted = 1.0;
  // The candidates the run can afford: a compilation and its steps each,
  // as the reference took them, within 2% of the time of the run at the
  // reference's rate, and at most `maxTrials` (MDIR_SKIN_SEARCH=always: the
  // most). A large system compiles for longer: STMV took 86 to 130 s a
  // candidate on an RTX 4090's node, JAC 11 to 13 s.
  constexpr size_t maxTrials = 7;
  size_t budget = maxTrials;
  if (policy != "always") {
    double runSeconds = 1e-6 * reference.microseconds *
                        static_cast<double>(control.numSteps);
    double trialSeconds = reference.compileSeconds + reference.runSeconds;
    budget = std::min<size_t>(
        maxTrials, static_cast<size_t>(0.02 * runSeconds / trialSeconds));
  }
  if (budget < 3) {
    finish();
    control.pairlistDistance = outer0;
    control.prunedDistance = inner0;
    // Not cached: a longer run of the same key may afford the search.
    control.skinReason = llvm::formatv(
        "a candidate costs {0:f0} s, and the run of about {1:f0} s affords "
        "{2} within 2% of it, fewer than 3",
        reference.compileSeconds + reference.runSeconds,
        1e-6 * reference.microseconds * static_cast<double>(control.numSteps),
        budget).str();
    return choice;
  }
  for (auto [outer, inner] :
       proposeCandidates(control, model, outer0, inner0, leastEdge)) {
    SkinTrial trial = measure(control, system, outer, inner);
    trial.predicted = model.predict(outer, inner) / base;
    note(trial);
    choice.trials.push_back(trial);
  }
  // Then a climb from the fastest so far, while it gains: steps of 0.5 Å of
  // the outer skin away from the reference (toward the model's best where
  // the reference is the fastest), then of 0.1 Å of the inner
  // one, up, or down if the first step up does not gain; at most
  // `maxTrials` candidates in all. The cost model narrows; the landscape
  // is not smooth (#406), and on the grid of #406 one step of the outer
  // skin took the choice to the grid's best on four systems of five, from
  // up to 1.2% from it. The inner skin is a step of 0.1 Å, a finer grid
  // than #406's, whose best prunings the model places low on JAC (0.7 Å
  // against 0.8, 2% on an RTX 4090).
  auto fastest = [&]() -> const SkinTrial & {
    const SkinTrial *f = &choice.trials.front();
    for (const SkinTrial &trial : choice.trials)
      if (trial.failure.empty() && trial.microseconds < f->microseconds)
        f = &trial;
    return *f;
  };
  // Measures (outer, inner) if it is a valid candidate not measured yet;
  // whether it was the fastest.
  auto attempt = [&](double outer, double inner) {
    double cutoff = control.cutoffDistance;
    bool valid = choice.trials.size() < budget &&
                 outer - cutoff >= 1.5 - 1e-9 && outer - cutoff <= 6.0 + 1e-9 &&
                 outer <= 0.999 * leastEdge && outer <= 127.0 * cutoff &&
                 (inner == 0.0 ||
                  (inner >= cutoff + 0.3 - 1e-9 && inner < outer - 0.25));
    for (const SkinTrial &trial : choice.trials)
      valid &= std::fabs(trial.outer - outer) > 1e-9 ||
               std::fabs(trial.inner - inner) > 1e-9;
    if (!valid)
      return false;
    double before = fastest().microseconds;
    SkinTrial trial = measure(control, system, outer, inner);
    trial.predicted = model.predict(outer, inner) / base;
    note(trial);
    choice.trials.push_back(trial);
    return trial.failure.empty() && trial.microseconds < before;
  };
  if (control.pairlistDistanceAuto) {
    // Away from the reference, or, where the reference is the fastest,
    // toward the model's best.
    double toward = choice.trials.size() > 1 ? choice.trials[1].outer : outer0;
    while (true) {
      const SkinTrial &f = fastest();
      double away = std::fabs(f.outer - outer0) > 1e-9 ? f.outer - outer0
                                                       : toward - outer0;
      if (std::fabs(away) < 1e-9 ||
          !attempt(f.outer + (away > 0.0 ? 0.5 : -0.5), f.inner))
        break;
    }
  }
  if (control.prunedDistanceAuto && inner0 != 0.0) {
    bool gained = false;
    while (true) {
      const SkinTrial &f = fastest();
      if (!attempt(f.outer, f.inner + 0.1))
        break;
      gained = true;
    }
    while (!gained) {
      const SkinTrial &f = fastest();
      if (!attempt(f.outer, f.inner - 0.1))
        break;
    }
  }
  finish();

  // The fastest, unless it is within the noise of the reference: twice the
  // standard error of the difference, and at least 0.3%.
  const SkinTrial *best = &choice.trials.front();
  for (const SkinTrial &trial : choice.trials)
    if (trial.failure.empty() && trial.microseconds < best->microseconds)
      best = &trial;
  const SkinTrial &first = choice.trials.front();
  double gain = 1.0 - best->microseconds / first.microseconds;
  double noise = std::max(
      0.003, 2.0 * std::hypot(first.error, best->error));
  if (best != &first && gain > noise) {
    control.pairlistDistance = best->outer;
    control.prunedDistance = best->inner;
    control.skinReason = llvm::formatv("{0:f1}% faster than {1} / {2} Å, "
                                       "beyond the noise of {3:f1}%",
                                       100.0 * gain, format(first.outer),
                                       first.inner != 0.0 ? format(first.inner)
                                                          : "-",
                                       100.0 * noise)
                             .str();
  } else {
    control.pairlistDistance = first.outer;
    control.prunedDistance = first.inner;
    control.skinReason =
        best == &first
            ? std::string("the reference was the fastest")
            : llvm::formatv("the fastest, {0} / {1} Å, is {2:f1}% faster, "
                            "within the noise of {3:f1}%",
                            format(best->outer),
                            best->inner != 0.0 ? format(best->inner) : "-",
                            100.0 * gain, 100.0 * noise)
                  .str();
  }
  control.skinSource = "measured";
  if (!path.empty()) {
    choice.cachePath = path;
    writeCache(path, key, control, choice);
  }
  return choice;
}
