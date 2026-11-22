#include "TowersShape.h"
#include "llvm/Support/DynamicLibrary.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
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
  int blockBytes, smBytes, reserved, threads, smThreads, blocks;
};
llvm::Expected<Limits> deviceLimits() {
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
      Limits l{};
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
      return l;
    }
  }
#endif
  return refusal(
      "a visible CUDA device is required to resolve launch resources");
}

using Cell = std::array<int, 3>;
llvm::Expected<TowersShape> derive(const TowersShape &given,
                                   const System &system, double cutoff,
                                   int reach, const Limits &limits) {
  TowersShape s = given;
  Cell dims;
  for (int i = 0; i < 3; ++i) {
    double n = std::floor(system.box[i] * (i == 2 ? reach : 1) / cutoff);
    if (!(n >= (i == 2 ? 2 * reach + 1 : 3) && n < INT32_MAX / 4))
      return refusal("cell height is unsupported for this box and cutoff");
    dims[i] = int(n);
  }
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
    // Only centers whose column or plate touches an occupied cell affect either
    // maximum.
    std::map<Cell, std::array<int64_t, 2>> spans;
    std::map<Cell, int> counts;
    for (size_t i = 0; i < system.getNumParticles(); ++i) {
      Cell cell;
      for (int a = 0; a < 3; ++a) {
        double x = std::fmod(system.positions[3 * i + a], system.box[a]);
        if (x < 0)
          x += system.box[a];
        cell[a] = std::min(dims[a] - 1, int(x / (system.box[a] / dims[a])));
      }
      ++counts[cell];
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
    int64_t column = 1, plate = 1;
    for (const auto &entry : spans) {
      column = std::max(column, entry.second[0]);
      plate = std::max(plate, entry.second[1]);
    }
    auto roundWarp = [](int64_t n) { return (n + 31) / 32 * 32; };
    if (!s.panel)
      s.panel = std::min({roundWarp((column + s.split - 1) / s.split),
                          int64_t(512), 2 * s.threads});
    if (!s.stripe)
      s.stripe = std::min(roundWarp(plate), s.threads);
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
    s.records = std::min((available() - 128) / 8, s.stripe * (s.panel / 32));
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
} // namespace

llvm::Expected<Object> mdir::tool::checkTowersShape(const Control &control,
                                                    const System &system) {
  if (!control.periodic ||
      std::any_of(std::begin(system.tilt), std::end(system.tilt),
                  [](double x) { return x != 0; }))
    return refusal("cell height requires a periodic rectangular box");
  auto limits = deviceLimits();
  if (!limits)
    return limits.takeError();
  const auto &given = control.towersShape;
  Object result{{"requested", control.towersShapeTable
                                  ? Value(fields(given, false))
                                  : Value("AUTO")},
                {"source", fields(given, true)}};
  auto resolve = [&](int reach) {
    return derive(given, system, control.cutoffDistance * units::length, reach,
                  *limits);
  };
  if (!given.cellHeight.empty()) {
    auto shape = resolve(given.cellHeight == "FULL" ? 1 : 2);
    if (!shape)
      return shape.takeError();
    result["effective"] = fields(*shape, false);
  } else {
    // The 4n direct-launch defaults take a geometry specialization as input.
    // Retain both valid candidates until a geometry is selected.
    Object candidates;
    std::string reason;
    for (int reach : {1, 2}) {
      auto shape = resolve(reach);
      if (shape)
        candidates[shape->cellHeight] = fields(*shape, false);
      else
        reason = llvm::toString(shape.takeError());
    }
    if (candidates.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                     reason.c_str());
    result["effective"] = candidates.size() == 1
                              ? std::move(candidates.begin()->second)
                              : Value(nullptr);
    if (candidates.size() > 1)
      result["candidates"] = std::move(candidates);
  }
  return result;
}
