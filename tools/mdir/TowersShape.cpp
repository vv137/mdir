#include "TowersShape.h"
#include "Commands.h"
#include "TowersCell.h"
#include "TowersHeight.h"
#include "TowersSpecialization.h"
#include "TowersTrial.h"
#include "TowersTrialRecord.h"
#include "mdir/Compiler/CompileCache.h"
#include "mdir/Dialect/MDExec/DevicePolicy.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>
#include <set>
#if MDIR_HAS_CUDA
#include <cuda.h>
#endif

using namespace mdir::driver;
using llvm::json::Object;
using llvm::json::Value;
namespace {
llvm::Error refusal(llvm::StringRef reason) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "towers_shape: %s", reason.str().c_str());
}
struct Limits {
  bool supportsMma;
  int major, minor;
  std::string uuid;
  int driver;
  int device, blockBytes, smBytes, reserved, threads, smThreads, blocks;
};
llvm::Expected<std::optional<Limits>> deviceLimits(bool requireMma) {
#if MDIR_HAS_CUDA
  auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary("libcuda.so.1");
  if (lib.isValid()) {
    auto init =
        reinterpret_cast<decltype(&cuInit)>(lib.getAddressOfSymbol("cuInit"));
    auto get = reinterpret_cast<decltype(&cuDeviceGet)>(
        lib.getAddressOfSymbol("cuDeviceGet"));
    auto attr = reinterpret_cast<decltype(&cuDeviceGetAttribute)>(
        lib.getAddressOfSymbol("cuDeviceGetAttribute"));
    CUdevice device;
    const char *ordinal = std::getenv("MDRT_DEVICE");
    if (init && get && attr && init(0) == CUDA_SUCCESS &&
        get(&device, ordinal ? std::atoi(ordinal) : 0) == CUDA_SUCCESS) {
      int major, minor;
      if (attr(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) !=
              CUDA_SUCCESS ||
          attr(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) !=
              CUDA_SUCCESS)
        return refusal("could not query CUDA compute capability");
      Limits l{};
      l.major = major;
      l.minor = minor;
      l.supportsMma = major > 7 || (major == 7 && minor >= 5);
      if (requireMma && !l.supportsMma)
        return refusal("screen MMA requires mma.sync m16n8k8 f16 (sm_75 or newer)");
      for (auto [field, key] :
           {std::pair{&l.blockBytes,
                      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN},
            {&l.smBytes,
             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR},
            {&l.reserved, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK},
            {&l.threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK},
            {&l.smThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR},
            {&l.blocks, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR}}) {
        if (attr(field, key, device) != CUDA_SUCCESS)
          return refusal("could not query CUDA launch limits");
      }
      CUuuid uuid;
      auto id = reinterpret_cast<decltype(&cuDeviceGetUuid)>(
          lib.getAddressOfSymbol("cuDeviceGetUuid"));
      auto version = reinterpret_cast<decltype(&cuDriverGetVersion)>(
          lib.getAddressOfSymbol("cuDriverGetVersion"));
      if (!id || !version || id(&uuid, device) != CUDA_SUCCESS ||
          version(&l.driver) != CUDA_SUCCESS)
        return refusal("could not query CUDA device identity");
      l.uuid = llvm::toHex(llvm::StringRef(uuid.bytes, sizeof(uuid.bytes)));
      l.device = device;
      return l;
    }
  }
#endif
  return std::nullopt;
}

#if MDIR_HAS_CUDA
llvm::Error occupancyRefusal(int registers, int blocks, int64_t resident) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "towers_shape: compiled pair uses %d registers per thread and admits "
      "%d resident blocks, below blocks_per_sm=%lld",
      registers, blocks, static_cast<long long>(resident));
}
llvm::Expected<Object> checkRegisters(int device, const TowersShape &shape,
                                      const std::string &image, bool pressure,
                                      bool require = true) {
  auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary("libcuda.so.1");
#define LOAD(name, api, symbol)                                                \
  auto name = reinterpret_cast<decltype(&api)>(lib.getAddressOfSymbol(symbol))
  LOAD(retain, cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
  LOAD(release, cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
  LOAD(push, cuCtxPushCurrent, "cuCtxPushCurrent_v2");
  LOAD(pop, cuCtxPopCurrent, "cuCtxPopCurrent_v2");
  LOAD(load, cuModuleLoadData, "cuModuleLoadData");
  LOAD(unload, cuModuleUnload, "cuModuleUnload");
  LOAD(function, cuModuleGetFunction, "cuModuleGetFunction");
  LOAD(configure, cuFuncSetAttribute, "cuFuncSetAttribute");
  LOAD(attribute, cuFuncGetAttribute, "cuFuncGetAttribute");
  LOAD(occupancy, cuOccupancyMaxActiveBlocksPerMultiprocessor,
       "cuOccupancyMaxActiveBlocksPerMultiprocessor");
#undef LOAD
  if (!retain || !release || !push || !pop || !load || !unload || !function ||
      !attribute || !configure || !occupancy)
    return refusal("missing CUDA function for compiled register check");
  CUcontext ctx;
  if (retain(&ctx, device) != CUDA_SUCCESS)
    return refusal("could not retain device context for register check");
  auto drop = llvm::scope_exit([&] { release(device); });
  if (push(ctx) != CUDA_SUCCESS)
    return refusal("could not enter device context for register check");
  auto restore = llvm::scope_exit([&] { pop(&ctx); });
  CUmodule module;
  if (load(&module, image.data()) != CUDA_SUCCESS)
    return refusal("could not load compiled pair specialization");
  auto cleanup = llvm::scope_exit([&] { unload(module); });
  CUfunction pair;
  int registers, blocks;
  if (function(&pair, module,
               pressure ? "towers_force_virial" : "towers_force") !=
          CUDA_SUCCESS ||
      attribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, pair) != CUDA_SUCCESS ||
      configure(pair, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                8 * shape.records) != CUDA_SUCCESS ||
      occupancy(&blocks, pair, shape.threads, 8 * shape.records) !=
          CUDA_SUCCESS)
    return refusal(
        "could not query compiled pair register count and occupancy");
  if (require && blocks < shape.blocksPerSM)
    return occupancyRefusal(registers, blocks, shape.blocksPerSM);
  return Object{{"registers_per_thread", registers},
                {"resident_blocks", blocks}};
}
#endif

using Cell = std::array<int, 3>;
struct FrameSpans {
  int64_t column = 1, plate = 1;
  int capacity = 32, cells = 1;
};

llvm::Expected<FrameSpans> measureSpans(const System &system, double cutoff,
                                        int reach, bool variableBox) {
  mdir::tool::TowersCell geometry(system);
  FrameSpans maximum;
  Cell dims;
  for (int i = 0; i < 3; ++i) {
    double n = geometry.bins(i, reach, cutoff, variableBox);
    if (!(n >= (i == 2 ? 2 * reach + 1 : 3) && n < INT32_MAX / 4))
      return refusal("cell height is unsupported for this box and cutoff");
    dims[i] = int(n);
  }
  // Only centers whose column or plate touches an occupied cell affect either
  // maximum.
  std::map<Cell, std::array<int64_t, 2>> spans;
  std::map<Cell, int> counts;
  for (size_t i = 0; i < system.getNumParticles(); ++i) {
    Cell cell;
    auto point = geometry.coordinates(&system.positions[3 * i]);
    for (int a = 0; a < 3; ++a) {
      double length = geometry.triclinic ? 1 : system.box[a];
      double x = std::fmod(point[a], length);
      if (x < 0)
        x += length;
      cell[a] = std::min(dims[a] - 1, int(x / (length / dims[a])));
    }
    maximum.capacity = std::max(maximum.capacity, ++counts[cell]);
  }
  for (const auto &[cell, count] : counts) {
    auto add = [&](int x, int y, int z, int plane) {
      Cell owner{(cell[0] + dims[0] + x) % dims[0],
                 (cell[1] + dims[1] + y) % dims[1],
                 (cell[2] + dims[2] + z) % dims[2]};
      spans[owner][plane] += count;
    };
    for (int z = -reach; z <= reach; ++z)
      add(0, 0, z, 0);
    for (Cell offset : {Cell{0, 0, 0}, Cell{1, 0, 0}, Cell{1, 1, 0},
                        Cell{0, 1, 0}, Cell{-1, 1, 0}})
      add(offset[0], offset[1], 0, 1);
  }
  for (const auto &entry : spans) {
    maximum.column = std::max(maximum.column, entry.second[0]);
    maximum.plate = std::max(maximum.plate, entry.second[1]);
  }
  maximum.capacity = llvm::alignTo(maximum.capacity, 8);
  // Slot offsets, including the growth pool, are signed 32-bit indices.
  int64_t limit =
      (INT32_MAX - int64_t(system.getNumParticles())) / maximum.capacity;
  for (int bins : dims) {
    if (maximum.cells > limit / bins) {
      maximum.cells = 0;
      break;
    }
    maximum.cells *= bins;
  }
  return maximum;
}

llvm::Expected<TowersShape>
derive(const TowersShape &given, const System &system, int reach,
       const FrameSpans &maximum, const Limits &limits, bool halfStripe = false,
       bool halfRecords = false) {
  TowersShape s = given;
  if (s.threads == 0)
    s.threads = 512;
  if (s.blocksPerSM == 0)
    s.blocksPerSM = 2;
  if (s.split == 0)
    s.split = 1;
  s.cellHeight = reach == 2 ? "HALF" : "FULL";
  if (s.threads > limits.threads ||
      s.threads * s.blocksPerSM > limits.smThreads ||
      s.blocksPerSM > std::min(4, limits.blocks))
    return refusal(
        "thread count or resident blocks exceeds the block or SM launch limit");
  if (!s.panel || !s.stripe) {
    auto roundWarp = [](int64_t n) { return (n + 31) / 32 * 32; };
    if (!s.panel)
      s.panel = std::min({roundWarp((maximum.column + s.split - 1) / s.split),
                          int64_t(512), 2 * s.threads});
    if (!s.stripe)
      s.stripe = std::min(roundWarp(maximum.plate), s.threads);
    if (!given.stripe && halfStripe)
      s.stripe = roundWarp((s.stripe + 1) / 2);
  }
  if (s.panel > 2 * s.threads || s.stripe > s.threads)
    return refusal("panel or stripe exceeds the supported worker spans");
  auto tableTypes = system.topology
                        ? system.topology->getNumTypes()
                        : (system.types.empty()
                               ? 0
                               : size_t(*std::max_element(system.types.begin(),
                                                          system.types.end())) +
                                     1);
  int64_t table = tableTypes <= 32 ? int64_t(8 * tableTypes * tableTypes) : 0;
  auto fixed = [&]() {
    int64_t tile = 48 * (s.panel + s.stripe) + 16;
    return tile + (tile + table <= 48 * 1024 ? table : 0);
  };
  auto available = [&]() {
    return std::min<int64_t>(limits.smBytes / s.blocksPerSM,
                             limits.blockBytes) -
           limits.reserved - fixed();
  };
  auto minimum = [&]() {
    return 8 * (given.records ? given.records : s.panel / 32);
  };
  while (!given.panel && s.panel > 32 && fixed() > 48 * 1024)
    s.panel -= 32;
  while (!given.stripe && s.stripe > 32 &&
         (s.stripe * s.panel >= (1 << 18) || fixed() > 48 * 1024 ||
          available() < minimum()))
    s.stripe -= 32;
  if (s.stripe * s.panel >= (1 << 18))
    return refusal("panel times stripe exceeds the record prefix");
  if (!s.records)
    s.records = std::min((available() - 128) / 8,
                         s.stripe * (s.panel / 32) / (halfRecords ? 2 : 1));
  int64_t bytes =
      fixed() + 8 * std::max(s.panel / 32, s.records) + limits.reserved;
  if (fixed() > 48 * 1024 || bytes > limits.blockBytes)
    return refusal("shared memory per block exceeds the tile or device limit");
  if (bytes * s.blocksPerSM > limits.smBytes)
    return refusal(
        "shared memory per SM cannot hold the requested resident blocks");
  if (s.records < s.panel / 32 || s.records > s.stripe * (s.panel / 32))
    return refusal("record capacity is outside the panel and stripe limits");
  return s;
}

// Enumerate at one already resolved height. A supplied field stays fixed
// throughout, including partial pins. Resource rounding can make distinct
// proposals coincide, especially on small frames.
std::vector<TowersShape> launchCandidates(const TowersShape &given,
                                          const System &system, int reach,
                                          const FrameSpans &maximum,
                                          const Limits &limits) {
  std::vector<TowersShape> candidates;
  std::set<std::array<int64_t, 6>> seen;
  for (int64_t threads : {128, 256, 512})
    for (int64_t residency : {1, 2})
      for (int64_t split : {1, 2}) {
        TowersShape proposal = given;
        if (!proposal.threads)
          proposal.threads = threads;
        if (!proposal.blocksPerSM)
          proposal.blocksPerSM = residency;
        if (!proposal.split)
          proposal.split = split;
        for (bool halfStripe : {false, true})
          for (bool halfRecords : {false, true}) {
            auto shape = derive(proposal, system, reach, maximum, limits,
                                halfStripe, halfRecords);
            if (!shape) {
              llvm::consumeError(shape.takeError());
              continue;
            }
            if (seen.insert({shape->threads, shape->blocksPerSM, shape->split,
                             shape->panel, shape->stripe, shape->records})
                    .second)
              candidates.push_back(*shape);
          }
      }
  return candidates;
}

Object fields(const TowersShape &s, bool provenance) {
  Object result;
  for (auto [key, field] : {std::pair{"threads", s.threads},
                            {"blocks_per_sm", s.blocksPerSM},
                            {"stripe", s.stripe},
                            {"panel", s.panel},
                            {"records", s.records},
                            {"split", s.split}})
    result[key] = provenance ? Value(field ? "control" : "AUTO")
                  : field    ? Value(field)
                             : Value("AUTO");
  result["pack"] = provenance ? Value(s.pack.empty() ? "AUTO" : "control")
                              : Value(s.pack.empty() ? "AUTO" : s.pack);
  result["screen"] = provenance ? Value(s.screen == "AUTO" ? "AUTO" : "control")
                                : Value(s.screen);
  result["cell_height"] =
      provenance ? Value(s.cellHeight.empty() ? "AUTO" : "control")
                 : Value(s.cellHeight.empty() ? "AUTO" : s.cellHeight);
  return result;
}
#if MDIR_HAS_CUDA
std::string resolvePack(const TowersShape &given, mdir::tool::TowersPopulation pop,
                        Object &report) {
  report["population_source"] = "device spans";
  report["maximum_population"] = pop.maximum;
  report["mean_population"] = pop.mean;
  report["pack_rule"] = "AUTO selects WIDE when maximum_population > 128";
  if (!given.pack.empty())
    return given.pack;
  return pop.maximum > 128 ? "WIDE" : "COMPACT";
}
llvm::Expected<TowersShape>
timeCandidates(const Control &control, const System &system,
               const Limits &limits, const FrameSpans &spans,
               const mdir::tool::TowersEmitter &emit,
               std::vector<TowersShape> &candidates, Object &report,
               const TowersShape *untimed = nullptr) {
  using namespace mdir;
  using namespace mdir::tool;
  auto start = std::chrono::steady_clock::now();
  if (candidates.empty())
    return refusal("no launch satisfies the supplied pins and device limits");
  const auto &first = candidates.front();
  TowersBuffers parameters;
  auto firstImage = emit(first, &parameters);
  if (!firstImage)
    return firstImage.takeError();
  double cutoff = control.cutoffDistance * units::length;
  int reach = first.cellHeight == "HALF" ? 2 : 1;
  auto frame = createTowersTrial(limits.device, system, spans.cells, reach,
                                 spans.capacity, parameters);
  if (!frame)
    return frame.takeError();
  auto population = readTowersPopulation(**frame, *firstImage);
  if (!population)
    return population.takeError();
  auto pack = resolvePack(control.towersShape, *population, report);
  bool recompile = (first.pack == "WIDE") != (pack == "WIDE");
  for (auto &candidate : candidates)
    candidate.pack = pack;
  if (candidates.size() == 1) {
    report["selection"] = "single candidate";
    report["winner"] = fields(candidates.front(), false);
    report["cells"] = llvm::json::Array{Object{
        {"shape", fields(candidates.front(), false)}, {"median_us", nullptr}}};
    report["margin_us"] = nullptr;
    report["below_noise"] = nullptr;
    report["trial_cost_ms"] = 0.0;
    return candidates.front();
  }
  if (recompile) {
    firstImage = emit(first, nullptr);
    if (!firstImage)
      return firstImage.takeError();
  }
  // Include the compiled pair body, the device identity, the box and the
  // feasible proposals. Frames on the same grid can reuse the winning launch.
  std::string key = compiler::getCacheSalt();
  llvm::raw_string_ostream os(key);
  os << "\ntowers-moving-split-timing-v2 " << limits.uuid << " "
     << limits.driver << " " << compiler::getCacheEntryName(*firstImage, "")
     << " " << spans.cells << " "
     << llvm::format_hex(llvm::bit_cast<uint64_t>(cutoff), 18);
  for (double length : system.box)
    os << " " << llvm::format_hex(llvm::bit_cast<uint64_t>(length), 18);
  for (double tilt : system.tilt)
    os << " " << llvm::format_hex(llvm::bit_cast<uint64_t>(tilt), 18);
  for (const auto &s : candidates)
    os << "\n" << Value(fields(s, false));
  auto cache = compiler::CompileCacheConfig::fromEnvironment();
  auto name = compiler::getCacheEntryName(key, ".shape");
  if (cache && cache->opened) {
    double seconds = 0;
    bool rejected = false;
    auto hit = compiler::readCacheEntry(*cache->opened, "gpu", name, key,
                                        seconds, rejected);
    if (hit) {
      auto data = llvm::json::parse(hit->getBuffer());
      if (!data)
        llvm::consumeError(data.takeError());
      else if (auto *saved = data->getAsObject()) {
        auto index = saved->getInteger("winner_index");
        if (index && size_t(*index) < candidates.size() &&
            saved->getArray("cells")) {
          compiler::touchCacheEntry(*cache->opened, "gpu", name);
          double cost = saved->getNumber("trial_cost_ms").value_or(0);
          for (auto &kv : *saved)
            report[kv.first] = std::move(kv.second);
          report["selection"] = "grid cache";
          report["recorded_trial_cost_ms"] = cost;
          report["trial_cost_ms"] = 0.0;
          return candidates[*index];
        }
      }
    }
  }
  // TOWERS taken by `neighbor_structure = "AUTO"` without a cached trial
  // (D[best-defaults]): the derived launch, untimed. The trial costs about
  // a second for each candidate it compiles, tens of seconds for a system,
  // against a gain of about a microsecond a step (D432).
  if (untimed) {
    TowersShape shape = *untimed;
    shape.pack = pack;
    report["selection"] = "derived (AUTO structure, no cached trial)";
    report["winner"] = fields(shape, false);
    report["margin_us"] = nullptr;
    report["below_noise"] = nullptr;
    report["trial_cost_ms"] = 0.0;
    return shape;
  }

  struct Candidate {
    unsigned index;
    std::string image;
    TowersTiming initial;
    std::array<TowersTiming, 5> repeat{};
  };
  std::vector<Candidate> trials;
  std::string firstRefusal;
  auto measure = [&](const Candidate &c, TowersTiming &us) -> llvm::Error {
    auto result = measureTowersTrial(**frame, c.image, candidates[c.index],
                                     control.barostat);
    if (!result)
      return result.takeError();
    us = *result;
    return llvm::Error::success();
  };
  for (auto [idx, s] : llvm::enumerate(candidates)) {
    auto image = idx == 0 ? std::move(firstImage) : emit(s, nullptr);
    if (!image)
      return image.takeError();
    auto registers =
        checkRegisters(limits.device, s, *image, control.barostat, false);
    if (!registers)
      return registers.takeError();
    int blocks = registers->getInteger("resident_blocks").value();
    if (blocks < s.blocksPerSM) {
      if (firstRefusal.empty())
        firstRefusal = llvm::toString(occupancyRefusal(
            registers->getInteger("registers_per_thread").value(), blocks,
            s.blocksPerSM));
      continue;
    }
    Candidate trial{unsigned(idx), std::move(*image), {}};
    if (auto err = measure(trial, trial.initial))
      return std::move(err);
    trials.push_back(std::move(trial));
  }
  if (trials.empty())
    return llvm::createStringError("%s", firstRefusal.c_str());
  auto measurement = [&](const TowersTiming &t, const TowersShape &shape,
                         llvm::StringRef totalKey) {
    Object record{{"shape", fields(shape, false)},
                  {"pair_us", t.pairUs},
                  {"rebuild_us", t.rebuildUs},
                  {"stripe_retries", t.stripeRetries}};
    record[totalKey] = t.total();
    return record;
  };
  int64_t timed = trials.size();
  std::map<int, Object> splits;
  for (auto &trial : trials) {
    const auto &shape = candidates[trial.index];
    auto &group = splits[shape.split];
    if (!group.getArray("candidates"))
      group["candidates"] = llvm::json::Array{};
    group.getArray("candidates")
        ->push_back(measurement(trial.initial, shape, "initial_us"));
  }
  std::stable_sort(trials.begin(), trials.end(),
                   [](const auto &a, const auto &b) {
                     return a.initial.total() < b.initial.total();
                   });
  // Report a repeated measurement for each feasible split. If the five
  // fastest all share a split, retain the fastest launch of the other split.
  size_t keep = std::min<size_t>(5, trials.size());
  auto split = [&](const Candidate &c) { return candidates[c.index].split; };
  if (splits.size() == 2 &&
      llvm::all_of(llvm::ArrayRef(trials).take_front(keep),
                   [&](const Candidate &c) {
                     return split(c) == split(trials.front());
                   })) {
    auto other = llvm::find_if(trials, [&](const Candidate &c) {
      return split(c) != split(trials.front());
    });
    std::iter_swap(trials.begin() + keep, other);
    ++keep;
  }
  trials.resize(keep);
  std::vector<unsigned> order(trials.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 shuffle(0);
  for (unsigned rep = 0; rep < 5; ++rep) {
    std::shuffle(order.begin(), order.end(), shuffle);
    for (unsigned i : order)
      if (auto err = measure(trials[i], trials[i].repeat[rep]))
        return std::move(err);
  }
  llvm::json::Array finalists;
  std::map<int, const Candidate *> winners;
  for (auto &c : trials) {
    llvm::stable_sort(c.repeat,
                      [](auto &a, auto &b) { return a.total() < b.total(); });
    const auto &shape = candidates[c.index];
    finalists.push_back(measurement(c.repeat[2], shape, "median_us"));
    auto &best = winners[shape.split];
    if (!best || c.repeat[2].total() < best->repeat[2].total())
      best = &c;
  }
  llvm::json::Array records;
  std::vector<std::array<double, 5>> times;
  std::vector<unsigned> indices;
  for (auto [split, best] : winners) {
    auto &group = splits.at(split);
    for (auto &kv :
         measurement(best->repeat[2], candidates[best->index], "median_us"))
      group[kv.first] = std::move(kv.second);
    records.push_back(std::move(group));
    auto &totals = times.emplace_back();
    llvm::transform(best->repeat, totals.begin(),
                    [](auto &sample) { return sample.total(); });
    indices.push_back(best->index);
  }
  Object summary = towersTrialRecord(std::move(records), std::move(times));
  unsigned best = indices[*summary.getInteger("winner_cell")];
  summary["winner_index"] = int64_t(best);
  summary["finalists"] = std::move(finalists);
  summary["timed_candidates"] = timed;
  summary["trial_cost_ms"] = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
  if (cache && cache->opened) {
    uint64_t bytes = 0;
    if (auto err = compiler::writeCacheEntry(
            *cache->opened, "gpu", name, key,
            llvm::formatv("{0}", Value(Object(summary))).str(), 0, &bytes))
      report["cache_error"] = llvm::toString(std::move(err));
    else
      compiler::noteCacheStores(*cache->opened, bytes, cache->maxBytes);
  }
  for (auto &kv : summary)
    report[kv.first] = std::move(kv.second);
  report["selection"] = "setup trial";
  return candidates[best];
}
#endif
} // namespace

llvm::Expected<Object>
mdir::tool::checkTowersShape(const Control &control, const System &system,
                             bool setup, llvm::StringRef checkpointHeight,
                             TowersSetup *resolved, bool untimed) {
  if (!control.periodic)
    return refusal("cell height requires a periodic box");
  TowersCell geometry(system);
  auto limits = deviceLimits(control.towersShape.screen == "MMA");
  if (!limits)
    return limits.takeError();
  auto given = control.towersShape;
  Object result{{"requested", control.towersShapeTable
                                  ? Value(fields(given, false))
                                  : Value("AUTO")},
                {"source", fields(given, true)}};
  if (*limits) {
    if (given.screen == "AUTO") {
      given.screen = (**limits).supportsMma ? "MMA" : "SCALAR";
      result["screen_reason"] = (**limits).supportsMma
                                    ? "device supports m16n8k8"
                                    : "device lacks m16n8k8";
    } else {
      result["screen_reason"] = "control";
    }
  }
  if (!checkpointHeight.empty()) {
    if (checkpointHeight != "FULL" && checkpointHeight != "HALF")
      return refusal("checkpoint tower height must be FULL or HALF");
    given.cellHeight = checkpointHeight.str();
    (*result.getObject("source"))["cell_height"] = "checkpoint";
  }
  double cutoff = control.cutoffDistance * units::length;
  int reach = given.cellHeight == "HALF" ? 2 : 1;
  if (given.cellHeight.empty()) {
    reach = selectTowersReach(geometry.heights.data(), cutoff,
                              system.getNumParticles());
  }
  auto choices = towersCells(geometry.heights.data(), cutoff,
                             reach == 1 ? "FULL" : "HALF", given.split);
  llvm::json::Array planned;
  for (auto cell : choices)
    planned.push_back(Object{{"cell_height", cell.reach == 1 ? "FULL" : "HALF"},
                             {"split", cell.split}});
  result["candidate_cells"] = std::move(planned);
  if (choices.empty())
    return refusal("cell height is unsupported for this box and cutoff");
  if (!*limits) {
    if (setup)
      return refusal(
          "a visible CUDA device is required to resolve launch resources");
    TowersShape partial = given;
    partial.cellHeight = reach == 1 ? "FULL" : "HALF";
    result["effective"] = fields(partial, false);
    result["register_check"] = "register check runs at setup";
    return result;
  }
  auto maximum = measureSpans(system, cutoff, reach, control.barostat);
  if (!maximum)
    return maximum.takeError();
  if (!maximum->cells)
    return refusal("tower grid exceeds the signed 32-bit slot range");
  auto candidates = launchCandidates(given, system, reach, *maximum, **limits);
  llvm::json::Array proposals;
  for (const auto &candidate : candidates)
    proposals.push_back(fields(candidate, false));
  result["launch_candidates"] = std::move(proposals);
  auto shape = derive(given, system, reach, *maximum, **limits);
  if (!shape && !candidates.empty()) {
    llvm::consumeError(shape.takeError());
    shape = candidates.front();
  }
  if (!shape)
    return shape.takeError();
#if MDIR_HAS_CUDA
  llvm::SmallString<256> deviceLibrary(getCudaToolkitRoot());
  llvm::sys::path::append(deviceLibrary, "nvvm", "libdevice",
                          "libdevice.10.bc");
  if (!setup && !llvm::sys::fs::exists(deviceLibrary)) {
    result["effective"] = fields(*shape, false);
    result["register_check"] = "register check runs at setup";
    return result;
  }
  std::string rebuildOnly;
  auto emit = prepareTowersSpecialization(control, system, reach,
                                          maximum->capacity, (*limits)->device,
                                          setup ? &rebuildOnly : nullptr);
  if (!emit)
    return emit.takeError();
  if (!*emit) {
    if (setup) {
      auto frame = createTowersTrial((*limits)->device, system, maximum->cells,
                                     reach, maximum->capacity, {});
      if (!frame)
        return frame.takeError();
      auto population = readTowersPopulation(**frame, rebuildOnly);
      if (!population)
        return population.takeError();
      shape->pack = resolvePack(given, *population, result);
    }
    result["selection"] = "derived shape (no force-only region)";
    result["register_check"] = "no force-only region";
  } else {
    result["force_kernel"] =
        control.barostat ? "towers_force_virial" : "towers_force";
    if (setup) {
      TowersShape derived = *shape;
      shape = timeCandidates(control, system, **limits, *maximum, *emit,
                             candidates, result, untimed ? &derived : nullptr);
      if (!shape)
        return shape.takeError();
    } else {
      result["selection"] =
          candidates.size() == 1 ? "single candidate" : "setup trial pending";
    }
    auto image = (*emit)(*shape, nullptr);
    if (!image)
      return image.takeError();
    auto compiled =
        checkRegisters((*limits)->device, *shape, *image, control.barostat);
    if (!compiled)
      return compiled.takeError();
    result["register_check"] = std::move(*compiled);
  }
  if (resolved) {
    resolved->launch = *shape;
    resolved->capacity = maximum->capacity;
    for (int a = 0; a < 3; ++a)
      resolved->bins[a] =
          int(geometry.bins(a, reach, cutoff, control.barostat));
  }
#endif
  result["effective"] = fields(*shape, false);
  return result;
}

mdir::tool::AutoTowers
mdir::tool::resolveAutoTowers(const Control &control, const System &system) {
  AutoTowers result;
  auto no = [&](std::string why) {
    result.reason = std::move(why);
    return result;
  };
  if (control.neighborStructureRequested != NeighborStructure::Auto)
    return no("the structure was given");
  // TOWERS's own requirements (D428), and the inputs whose loops or
  // keys belong to the structures of D261: these keep that rule.
  if (control.target != Target::GPU)
    return no("the target is the CPU");
  if (control.precision != Precision::Mixed)
    return no("TOWERS takes mixed precision");
  if (control.deterministic)
    return no("the deterministic mode");
  if (!control.periodic)
    return no("the cell is not periodic");
  if (system.tilt[0] != 0.0 || system.tilt[1] != 0.0 || system.tilt[2] != 0.0)
    return no("a triclinic cell, not measured with TOWERS");
  if (control.minimize)
    return no("a minimization");
  if (control.frames || control.segments)
    return no("a program of frames or segments");
  if (!control.stagedModels.empty() || !control.models.empty() ||
      !control.externalTerms.empty())
    return no("a model or external term");
  if (control.implicitSolvent != Control::ImplicitSolvent::None)
    return no("implicit solvent");
  if (control.tunableGradient)
    return no("the derivative in the tunables");
  if (control.rebuildPeriod > 0)
    return no("'rebuild_interval' is given");
  if (control.pairlistDistanceGiven || control.prunedDistanceGiven)
    return no("a list reach is given, which TOWERS does not use");
  if (control.rigidUnits == RigidUnits::On)
    return no("'rigid_units = \"ON\"' takes lists of units");
  auto limits = deviceLimits(/*requireMma=*/false);
  if (!limits) {
    std::string message = llvm::toString(limits.takeError());
    return no("the device query failed: " + message);
  }
  if (!*limits)
    return no("no visible CUDA device");
  std::string arch = "sm_" + std::to_string((**limits).major) +
                     std::to_string((**limits).minor);
  auto policy = md_exec::policy::getDevicePolicy(arch);
  int64_t least = policy.traversal.towersLeast;
  result.record = Object{
      {"rule", "towers_least"},
      {"device_architecture", arch},
      {"towers_least", least < 0 ? Value(nullptr) : Value(least)},
      {"fallback", policy.traversalFallback},
      {"basis", Object{{"decision", policy.traversalBasis.decision},
                       {"measured", policy.traversalBasis.measured},
                       {"date", policy.traversalBasis.date}}}};
  size_t count = system.getNumParticles();
  if (least < 0 || policy.traversalFallback)
    return no("TOWERS was not measured as the choice of AUTO on " + arch);
  if (count < size_t(least))
    return no(llvm::formatv("{0} particles, fewer than {1} on {2}", count,
                            least, arch)
                  .str());
  result.take = true;
  result.reason = llvm::formatv("{0} particles, at least {1} on {2} ({3})",
                                count, least, arch,
                                policy.traversalBasis.decision)
                      .str();
  return result;
}

llvm::json::Object mdir::tool::initialStepGraphChoice(llvm::StringRef request,
    llvm::StringRef refusal, size_t particles) {
  Object choice{{"requested", request.str()}, {"value", request.str()},
                {"category", "setup trial"}, {"reason", "fixed choice"}};
  if (request != "AUTO")
    return choice;
  // Smallest measured LAUNCH winner in the six-input RTX 4090 comparison.
  constexpr size_t threshold = 408609;
  if (!refusal.empty()) {
    choice["value"] = "LAUNCH";
    choice["reason"] = refusal.str();
  } else if (particles >= threshold) {
    choice["value"] = "LAUNCH";
    choice["category"] = "rule";
    choice["reason"] = "particle count reaches the LAUNCH threshold";
    choice["inputs"] = Object{{"particles", particles}, {"threshold", threshold}};
  } else {
    choice["reason"] = "setup trial pending";
  }
  return choice;
}
