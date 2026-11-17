// The tuning policies of the device lowering (D[tuning-policies]).
//
// See include/mdir/Dialect/MDExec/DevicePolicy.h and
// docs/device-primitives.md, Section 8. Adding an architecture adds
// entries here; the choosers and the lowerings read only the selected
// entries.

#include "mdir/Dialect/MDExec/DevicePolicy.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using namespace mdir::md_exec::policy;

namespace {
/// An entry of a table: its key, its values, the measurement behind
/// them, and the architectures that the measurement covered (a major
/// version and a minor one, -1 for any minor).
template <typename Values> struct Entry {
  DeviceKey key;
  Values values;
  Measurement basis;
};

/// The architectures that a default entry was measured on.
struct Measured {
  int64_t major, minor;
};

/// The entry of `table` for `key`: the architecture and class, then the
/// architecture, then its major version; nothing where none matches.
template <typename Values>
const Entry<Values> *find(ArrayRef<Entry<Values>> table,
                          std::optional<DeviceKey> key) {
  if (!key)
    return nullptr;
  const Entry<Values> *best = nullptr;
  int rank = 0;
  for (const Entry<Values> &entry : table) {
    if (entry.key.major != key->major)
      continue;
    if (entry.key.minor != -1 && entry.key.minor != key->minor)
      continue;
    if (!entry.key.deviceClass.empty() &&
        entry.key.deviceClass != key->deviceClass)
      continue;
    int own = (entry.key.minor != -1 ? 2 : 1) +
              (!entry.key.deviceClass.empty() ? 4 : 0);
    if (own > rank) {
      best = &entry;
      rank = own;
    }
  }
  return best;
}

bool covers(ArrayRef<Measured> measured, std::optional<DeviceKey> key) {
  if (!key)
    return false;
  for (const Measured &m : measured)
    if (m.major == key->major && (m.minor == -1 || m.minor == key->minor))
      return true;
  return false;
}
} // namespace

//===----------------------------------------------------------------------===//
// The pruning in the loop over pairs (D284)
//===----------------------------------------------------------------------===//

// The groups from which the pruning in the loop took no more time a step
// than the pruning of the refresh (D284, recorded 2026-10-18; mixed
// precision, scripts/benchmarks/fused-prune/steps.py). On an RTX 4090
// (sm_89), water boxes of 52,686 and 103,158 sites: 3,292 groups equal,
// 6,447 groups 4% less; a warp a group leaves the device partly idle
// below about 3,300 groups.
const PruningPolicy mdir::md_exec::policy::kDefaultPruning = {
    /*fusedGroups=*/4096};
static const Measurement kPruningDefaultBasis = {
    "D284", "TIP3P water boxes of 52,686 and 103,158 sites, RTX 4090",
    "2026-10-18"};
static const Measured kPruningDefaultMeasured[] = {{8, 9}};

// On an RTX PRO 6000 Blackwell Max-Q (sm_120) at its cap of 300 W the
// kernel takes the sum of the two it replaces, and a step is equal from
// about 19,000 groups (+1.0% at 6,447, +0.2% at 19,367, -0.2% at 25,538,
// -1.1% at 66,693); Factor IX (5,681 groups) took 1 to 2% more from 4096.
// Every device of the major version 12 takes the entry, the RTX 5090
// included, which was not measured (#433).
static const Entry<PruningPolicy> kPruningTable[] = {
    {{12, -1, ""},
     {/*fusedGroups=*/20480},
     {"D284",
      "TIP3P water boxes of 52,686 to 309,876 sites, Factor IX, Cellulose, "
      "and STMV, RTX PRO 6000 Blackwell Max-Q",
      "2026-10-18"}},
};

//===----------------------------------------------------------------------===//
// The scatter-accumulates (D282, D283, D291)
//===----------------------------------------------------------------------===//

// The tiles of PME (D282, recorded 2026-10-18): every grid of at least
// 500,000 points (about 1,000 bins of 8 points) spread faster with tiles on
// both devices measured, the Amber suite at orders 4, 6, and 8 and the
// prolate kernel on an RTX 4090 and an RTX PRO 6000; bins of 8 points, the
// tile kernel 67.6 µs a step on Cellulose against 86.1 with bins of 16
// (RTX 4090). The tables of the tuples (D283, recorded 2026-10-18): a
// launch of fewer than 100,000 tuples keeps the atomics (JAC, 55,000
// tuples, 1.5% slower with the table on an RTX 4090 and 3% on an RTX PRO
// 6000; Factor IX, 165,000, 2.3% and 1% faster); a search gives up after
// 8 slots (a block of JAC with 494 particles in 512 slots took the kernel
// from 12 to 42 µs without the bound). The loops over pairs (D291): the
// table at any count where it is asked for (`pair-hash=on`); `auto` takes
// the atomics, measured faster.
const ScatterPolicy mdir::md_exec::policy::kDefaultScatter = {
    /*tileMinPoints=*/500000, /*tileEdge=*/8, /*hashLeast=*/100000,
    /*hashProbes=*/8, /*pairHashLeast=*/0};
static const Measurement kScatterDefaultBasis = {
    "D282, D283",
    "the Amber suite (JAC, Factor IX, Cellulose, STMV), RTX 4090 and RTX "
    "PRO 6000 Blackwell Max-Q",
    "2026-10-18"};
static const Measured kScatterDefaultMeasured[] = {{8, 9}, {12, 0}};
static const ArrayRef<Entry<ScatterPolicy>> kScatterTable = {};

//===----------------------------------------------------------------------===//
// The launches of the kernels of the loops
//===----------------------------------------------------------------------===//

// Not measured as a choice: the blocks of 128 threads and the 16 lanes a
// particle of the first device lowering (the loops over tuples by rows,
// D103; the matrix's rows, docs/paper/08-gpu.md), the blocks of 256
// threads of the batched bodies (D293), the static limit of the block
// memory of a block on every device of compute capability 7.0 or later
// (48 KiB), and the lanes of the gathering of PME, 4 a particle up to the
// order 4 (the stored places, D102) and 8 above it (in the prototype of
// #334, Cellulose on an RTX 4090, the gathering at order 8 took 358 µs
// with 8 lanes and 286 with 32, issue #411, which is open).
const LaunchPolicy mdir::md_exec::policy::kDefaultLaunch = {
    /*blockSize=*/128,       /*rowLanes=*/16,
    /*stageBlockSize=*/256,  /*blockMemoryBytes=*/48 * 1024,
    /*gatherSmallLanes=*/4,  /*gatherSmallOrder=*/4,
    /*gatherLanes=*/8};
static const Measurement kLaunchDefaultBasis = {
    "D102, D103, D293, #411",
    "not measured as a choice; the gathering's lanes in the prototype of "
    "#334 (Cellulose, RTX 4090)",
    "2026-10-18"};
static const ArrayRef<Entry<LaunchPolicy>> kLaunchTable = {};

//===----------------------------------------------------------------------===//
// Selection
//===----------------------------------------------------------------------===//

std::optional<DeviceKey> mdir::md_exec::policy::parseArchitecture(
    StringRef arch) {
  if (!arch.consume_front("sm_"))
    return std::nullopt;
  StringRef digits = arch.take_while([](char c) { return c >= '0' && c <= '9'; });
  StringRef suffix = arch.drop_front(digits.size());
  if (digits.size() < 2 ||
      !llvm::all_of(suffix, [](char c) { return c >= 'a' && c <= 'z'; }))
    return std::nullopt;
  unsigned number = 0;
  if (digits.getAsInteger(10, number))
    return std::nullopt;
  DeviceKey key;
  key.major = number / 10;
  key.minor = number % 10;
  return key;
}

std::string mdir::md_exec::policy::getTargetArchitecture(StringRef gpuOptions) {
  SmallVector<StringRef> words;
  gpuOptions.split(words, ' ', -1, /*KeepEmpty=*/false);
  for (StringRef word : words)
    if (word.consume_front("cubin-chip="))
      return word.str();
  return "";
}

std::string DevicePolicy::describeFallback(bool fallback) const {
  if (!fallback)
    return "";
  if (arch.empty())
    return " (the default of the policy: no architecture given)";
  return " (the default of the policy: " + arch + " was not measured)";
}

DevicePolicy mdir::md_exec::policy::getDevicePolicy(StringRef arch,
                                                    StringRef deviceClass) {
  DevicePolicy policy;
  policy.arch = arch.str();
  policy.deviceClass = deviceClass.str();
  std::optional<DeviceKey> key = parseArchitecture(arch);
  if (key)
    key->deviceClass = deviceClass;

  policy.pruning = kDefaultPruning;
  policy.pruningBasis = kPruningDefaultBasis;
  policy.pruningFallback = !covers(kPruningDefaultMeasured, key);
  if (const auto *entry = find<PruningPolicy>(kPruningTable, key)) {
    policy.pruning = entry->values;
    policy.pruningBasis = entry->basis;
    policy.pruningFallback = false;
  }

  policy.scatter = kDefaultScatter;
  policy.scatterBasis = kScatterDefaultBasis;
  policy.scatterFallback = !covers(kScatterDefaultMeasured, key);
  if (const auto *entry = find<ScatterPolicy>(kScatterTable, key)) {
    policy.scatter = entry->values;
    policy.scatterBasis = entry->basis;
    policy.scatterFallback = false;
  }

  policy.launch = kDefaultLaunch;
  policy.launchBasis = kLaunchDefaultBasis;
  policy.launchFallback = true;
  if (const auto *entry = find<LaunchPolicy>(kLaunchTable, key)) {
    policy.launch = entry->values;
    policy.launchBasis = entry->basis;
    policy.launchFallback = false;
  }
  return policy;
}
