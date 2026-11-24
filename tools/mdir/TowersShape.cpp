#include "TowersShape.h"
#include "Commands.h"
#include "TowersCell.h"
#include "TowersHeight.h"
#include "TowersSpecialization.h"
#include "TowersTrial.h"
#include "TowersTrialRecord.h"
#include "mdir/Compiler/CompileCache.h"
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
  bool supportsTowers = true;
  std::string uuid;
  int driver;
  int device, blockBytes, smBytes, reserved, threads, smThreads, blocks;
};
llvm::Expected<std::optional<Limits>> deviceLimits() {
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
      if (major < 7 || (major == 7 && minor < 5)) {
        l.supportsTowers = false;
        return l;
      }
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
  result["cell_height"] =
      provenance ? Value(s.cellHeight.empty() ? "AUTO" : "control")
                 : Value(s.cellHeight.empty() ? "AUTO" : s.cellHeight);
  return result;
}
#if MDIR_HAS_CUDA
struct CellTrial {
  std::vector<TowersShape> candidates;
  mdir::tool::TowersEmitter emit;
  mdir::tool::TrialFrame *frame = nullptr;
  std::string firstImage;
  int cells, reach, capacity;
  unsigned winner = 0;
  std::string winnerImage, refusal;
  std::array<float, 5> steps{};
  Object report;
};

llvm::Error rankCell(const Control &control, const System &system,
                     const Limits &limits, CellTrial &cell) {
  using namespace mdir::tool;
  const auto &candidates = cell.candidates;
  struct Candidate {
    unsigned index;
    std::string image;
    float initial;
    std::array<float, 5> repeat{};
  };
  std::vector<Candidate> trials;
  std::string firstRefusal;
  auto measure = [&](const Candidate &c, float &us) -> llvm::Error {
    auto result = measureTowersTrial(**cell.frame, c.image, candidates[c.index],
                                     control.barostat);
    if (!result)
      return result.takeError();
    us = *result;
    return llvm::Error::success();
  };
  for (auto [idx, s] : llvm::enumerate(candidates)) {
    llvm::Expected<std::string> image =
        idx == 0 ? llvm::Expected<std::string>(std::move(cell.firstImage))
                 : cell.emit(s, nullptr);
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
    Candidate trial{unsigned(idx), std::move(*image), 0};
    if (auto err = measure(trial, trial.initial))
      return std::move(err);
    trials.push_back(std::move(trial));
  }
  if (trials.empty()) {
    cell.refusal = firstRefusal;
    return llvm::Error::success();
  }
  cell.report["timed_candidates"] = int64_t(trials.size());
  llvm::json::Array initial;
  for (const auto &candidate : trials)
    initial.push_back(
        Object{{"shape", fields(candidates[candidate.index], false)},
               {"initial_us", double(candidate.initial)}});
  cell.report["candidates"] = std::move(initial);
  std::stable_sort(
      trials.begin(), trials.end(),
      [](const auto &a, const auto &b) { return a.initial < b.initial; });
  trials.resize(std::min<size_t>(5, trials.size()));
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
  for (auto &c : trials) {
    llvm::sort(c.repeat);
    finalists.push_back(Object{{"shape", fields(candidates[c.index], false)},
                               {"median_us", double(c.repeat[2])}});
  }
  auto best = std::min_element(
      trials.begin(), trials.end(),
      [](const auto &a, const auto &b) { return a.repeat[2] < b.repeat[2]; });
  cell.report["finalists"] = std::move(finalists);
  cell.winner = best->index;
  cell.winnerImage = std::move(best->image);
  return llvm::Error::success();
}

llvm::Expected<TowersShape>
timeCells(const Control &control, const System &system, const Limits &limits,
          std::vector<CellTrial> &cells, Object &report) {
  using namespace mdir;
  auto start = std::chrono::steady_clock::now();
  std::string key = compiler::getCacheSalt();
  llvm::raw_string_ostream os(key);
  os << "\ntowers-moving-v2 " << limits.uuid << " " << limits.driver;
  for (double x : system.box)
    os << " " << llvm::format_hex(llvm::bit_cast<uint64_t>(x), 18);
  for (double x : system.tilt)
    os << " " << llvm::format_hex(llvm::bit_cast<uint64_t>(x), 18);
  std::map<int, mdir::tool::TowersBuffers> inputs;
  for (auto &cell : cells) {
    bool first = !inputs.count(cell.reach);
    auto image = cell.emit(cell.candidates.front(),
                           first ? &inputs[cell.reach] : nullptr);
    if (!image)
      return image.takeError();
    cell.firstImage = std::move(*image);
    os << " " << compiler::getCacheEntryName(cell.firstImage, "");
    for (const auto &s : cell.candidates)
      os << "\n" << Value(fields(s, false));
  }
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
        auto cell = saved->getInteger("cache_cell");
        auto index = saved->getInteger("winner_index");
        if (cell && index && size_t(*cell) < cells.size() &&
            size_t(*index) < cells[*cell].candidates.size() &&
            saved->getArray("cells")) {
          compiler::touchCacheEntry(*cache->opened, "gpu", name);
          for (auto &kv : *saved)
            report[kv.first] = std::move(kv.second);
          report["selection"] = "grid cache";
          double originalCost = report.getNumber("trial_cost_ms").value_or(0);
          report["recorded_trial_cost_ms"] = originalCost;
          report["trial_cost_ms"] = 0.0;
          return cells[*cell].candidates[*index];
        }
      }
    }
  }
  std::map<int, tool::TrialFrame> frames;
  std::vector<size_t> feasible;
  for (auto [i, cell] : llvm::enumerate(cells)) {
    if (!frames.count(cell.reach)) {
      auto frame =
          tool::createTowersTrial(limits.device, system, cell.cells, cell.reach,
                                  cell.capacity, inputs.at(cell.reach));
      if (!frame)
        return frame.takeError();
      frames[cell.reach] = std::move(*frame);
      inputs.erase(cell.reach);
    }
    cell.frame = &frames.at(cell.reach);
    if (auto error = rankCell(control, system, limits, cell))
      return std::move(error);
    if (cell.refusal.empty())
      feasible.push_back(i);
  }
  if (feasible.empty())
    return refusal(cells.front().refusal);
  // Compare the cell winners in fresh shuffled rounds after compilation.
  std::vector<size_t> order = feasible;
  std::mt19937 shuffle(1);
  for (unsigned rep = 0; rep < 5; ++rep) {
    std::shuffle(order.begin(), order.end(), shuffle);
    for (size_t i : order) {
      auto &cell = cells[i];
      auto us = tool::measureTowersTrial(**cell.frame, cell.winnerImage,
                                         cell.candidates[cell.winner],
                                         control.barostat);
      if (!us)
        return us.takeError();
      cell.steps[rep] = *us;
    }
  }
  llvm::json::Array records;
  std::vector<std::array<float, 5>> times;
  int64_t total = 0;
  for (size_t i : feasible) {
    auto &cell = cells[i];
    total += *cell.report.getInteger("timed_candidates");
    cell.report["shape"] = fields(cell.candidates[cell.winner], false);
    records.push_back(std::move(cell.report));
    times.push_back(cell.steps);
  }
  Object summary = tool::towersTrialRecord(std::move(records), times);
  size_t best = feasible[*summary.getInteger("winner_cell")];
  auto &winner = cells[best];
  summary["cache_cell"] = int64_t(best);
  summary["winner_index"] = int64_t(winner.winner);
  summary["timed_candidates"] = total;
  summary["trial_cost_ms"] = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
  if (cache && cache->opened) {
    uint64_t bytes = 0;
    std::string data = llvm::formatv("{0}", Value(Object(summary))).str();
    if (auto err = compiler::writeCacheEntry(*cache->opened, "gpu", name, key,
                                             data, 0, &bytes))
      report["cache_error"] = llvm::toString(std::move(err));
    else
      compiler::noteCacheStores(*cache->opened, bytes, cache->maxBytes);
  }
  for (auto &kv : summary)
    report[kv.first] = std::move(kv.second);
  report["selection"] = "setup trial";
  return winner.candidates[winner.winner];
}
#endif
} // namespace

llvm::Expected<Object>
mdir::tool::checkTowersShape(const Control &control, const System &system,
                             bool setup, llvm::StringRef checkpointHeight,
                             TowersSetup *resolved) {
  if (!control.periodic)
    return refusal("cell height requires a periodic box");
  TowersCell geometry(system);
  auto limits = deviceLimits();
  if (!limits)
    return limits.takeError();
  auto given = control.towersShape;
  Object result{{"requested", control.towersShapeTable
                                  ? Value(fields(given, false))
                                  : Value("AUTO")},
                {"source", fields(given, true)}};
  if (*limits && !(**limits).supportsTowers) {
    constexpr auto reason =
        "TOWERS requires mma.sync m16n8k8 f16 (sm_75 or newer)";
    if (control.neighborStructure == NeighborStructure::Towers)
      return refusal(reason);
    result["selection"] = reason;
    return result;
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
  auto choices = towersCells(geometry.heights.data(), cutoff, given.cellHeight,
                             given.split);
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
  std::map<int, FrameSpans> spans;
  std::vector<TowersShape> candidates;
  llvm::json::Array proposals;
  for (auto cell : choices) {
    if (!spans.count(cell.reach)) {
      auto maximum = measureSpans(system, cutoff, cell.reach, control.barostat);
      if (!maximum)
        return maximum.takeError();
      spans[cell.reach] = *maximum;
    }
    if (!spans.at(cell.reach).cells)
      continue;
    TowersShape pins = given;
    pins.split = cell.split;
    for (auto &s : launchCandidates(pins, system, cell.reach, spans[cell.reach],
                                    **limits)) {
      proposals.push_back(fields(s, false));
      candidates.push_back(s);
    }
  }
  result["launch_candidates"] = std::move(proposals);
  if (!spans.count(reach) || !spans.at(reach).cells) {
    auto feasible = llvm::find_if(
        spans, [](const auto &entry) { return entry.second.cells > 0; });
    if (feasible == spans.end())
      return refusal("tower grid exceeds the signed 32-bit slot range");
    reach = feasible->first;
  }
  auto shape = derive(given, system, reach, spans[reach], **limits);
  if (!shape && !candidates.empty()) {
    llvm::consumeError(shape.takeError());
    shape = candidates.front();
  }
  if (!shape)
    return shape.takeError();
  reach = shape->cellHeight == "HALF" ? 2 : 1;
#if MDIR_HAS_CUDA
  llvm::SmallString<256> deviceLibrary(getCudaToolkitRoot());
  llvm::sys::path::append(deviceLibrary, "nvvm", "libdevice",
                          "libdevice.10.bc");
  if (!setup && !llvm::sys::fs::exists(deviceLibrary)) {
    result["effective"] = fields(*shape, false);
    result["register_check"] = "register check runs at setup";
    return result;
  }
  std::map<int, TowersEmitter> emitters;
  for (auto &[r, maximum] : spans) {
    if (!maximum.cells || (!setup && r != reach))
      continue;
    auto emit = prepareTowersSpecialization(
        control, system, r, maximum.capacity, (*limits)->device);
    if (!emit)
      return emit.takeError();
    emitters[r] = std::move(*emit);
  }
  if (!emitters[reach]) {
    result["selection"] = "derived shape (no force-only region)";
    result["register_check"] = "no force-only region";
  } else {
    result["force_kernel"] =
        control.barostat ? "towers_force_virial" : "towers_force";
    if (candidates.empty())
      return refusal("no launch satisfies the supplied pins and device limits");
    if (candidates.size() == 1) {
      *shape = candidates.front();
      result["selection"] = "single candidate";
      result["winner"] = fields(*shape, false);
      result["cells"] = llvm::json::Array{
          Object{{"shape", fields(*shape, false)}, {"median_us", nullptr}}};
      result["margin_us"] = nullptr;
      result["below_noise"] = nullptr;
      result["trial_cost_ms"] = 0.0;
    } else if (setup) {
      std::vector<CellTrial> cells;
      for (auto choice : choices) {
        CellTrial cell;
        cell.reach = choice.reach;
        cell.cells = spans.at(cell.reach).cells;
        if (!cell.cells)
          continue;
        cell.capacity = spans.at(cell.reach).capacity;
        cell.emit = emitters[cell.reach];
        for (auto &s : candidates)
          if (s.split == choice.split &&
              (s.cellHeight == "HALF") == (cell.reach == 2))
            cell.candidates.push_back(s);
        if (!cell.candidates.empty())
          cells.push_back(std::move(cell));
      }
      shape = timeCells(control, system, **limits, cells, result);
      if (!shape)
        return shape.takeError();
    } else {
      result["selection"] = "setup trial pending";
    }
    reach = shape->cellHeight == "HALF" ? 2 : 1;
    auto image = emitters[reach](*shape, nullptr);
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
    resolved->capacity = spans[reach].capacity;
    for (int a = 0; a < 3; ++a)
      resolved->bins[a] =
          int(geometry.bins(a, reach, cutoff, control.barostat));
  }
#endif
  result["effective"] = fields(*shape, false);
  return result;
}
