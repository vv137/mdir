// The tuning policies of the device lowering (D420).
//
// See include/mdir/Dialect/MDExec/DevicePolicy.h and
// docs/device-primitives.md, Section 8. Adding an architecture adds
// entries here; the choosers and the lowerings read only the selected
// entries.

#include "mdir/Dialect/MDExec/DevicePolicy.h"

#include "llvm/ADT/ArrayRef.h"

using namespace llvm;
using namespace mdir::md_exec::policy;

namespace {
/// An architecture that a measurement covered: a major version and a
/// minor one, -1 for any minor.
struct Measured {
  int64_t major, minor;
};

/// An entry of a table: its key, its values, the measurement behind
/// them, and the architectures that the measurement covered. An
/// architecture that the key matches but the measurement did not cover
/// takes the entry as a fallback (DevicePolicy::fallback).
template <typename Values> struct Entry {
  DeviceKey key;
  Values values;
  Measurement basis;
  ArrayRef<Measured> measured;
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
// Every device of the major version 12 takes the entry, as the string
// test on `cubin-chip=sm_12` gave it before (#433). The measurement covered
// sm_120 alone, on that device: sm_121 takes the entry as a fallback, and
// so would the RTX 5090 once a device class tells it apart.
static const Measured kPruning12Measured[] = {{12, 0}};
static const Entry<PruningPolicy> kPruningTable[] = {
    {{12, -1, ""},
     {/*fusedGroups=*/20480},
     {"D284",
      "TIP3P water boxes of 52,686 to 309,876 sites, Factor IX, Cellulose, "
      "and STMV, RTX PRO 6000 Blackwell Max-Q",
      "2026-10-18"},
     kPruning12Measured},
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
// The SIMT schedule of a list of units (D421)
//===----------------------------------------------------------------------===//

// The block and the rotation are those of #795's schedule at 745c9f12,
// whose mechanisms are requirements (D417): blocks of 128 threads and the
// 16 units of a group turned through the 16 lanes of a half-warp. The
// units of a column of the build over units were swept over 64 (the
// groups' value, D404), 96, 128, 192, and 256 on the six systems of #795;
// 128 gave the least time a call of the loop over pairs on each of them,
// or one within 0.3% of it, with the places held raised from 55-66% to
// 70-85% and the build's time a step unchanged. Measured with the timing
// protocol of D417 on RTX 4090s (sm_89), 2026-11-18; every other
// architecture takes it as a fallback.
const SegmentsPolicy mdir::md_exec::policy::kDefaultSegments = {
    /*blockSize=*/128, /*rotationLanes=*/16, /*columnUnits=*/128.0};
static const Measurement kSegmentsDefaultBasis = {
    "D421",
    "the six systems of #795 (rhombic dodecahedra of OPC and TIP3P, "
    "ubiquitin in OPC and TIP3P, a box of OPC, a CHARMM protein in TIP3P), "
    "RTX 4090",
    "2026-11-18"};
static const Measured kSegmentsDefaultMeasured[] = {{8, 9}};
static const ArrayRef<Entry<SegmentsPolicy>> kSegmentsTable = {};

//===----------------------------------------------------------------------===//
// The build of the groups (D423)
//===----------------------------------------------------------------------===//

// The default is D89's value, 64 particles a column, the first build's and
// not measured as a choice: an architecture without an entry of its own
// keeps what its programs did before the table.
const GroupsPolicy mdir::md_exec::policy::kDefaultGroups = {
    /*columnParticles=*/64.0};
static const Measurement kGroupsDefaultBasis = {
    "D89", "not measured as a choice: the value of the first build",
    "2026-10-01"};
static const ArrayRef<Measured> kGroupsDefaultMeasured = {};

// On sm_89 the particles of a column were swept over 64, 80, 96, 112, 128,
// and 192 (#912), with the timing protocol, in NPT on JAC, Factor IX,
// Cellulose, STMV, and a TIP3P box of 309,876 sites (JAC and Cellulose at a
// relative RMS force error of about 4e-4). On RTX 4090s, 96 took the least
// time a step on each system or one within 0.1% of it: 0.967 to 0.979 of
// 64's step on the four of the suite, 0.992 on the water box, with fewer
// entries (2 to 3%, and so fewer atomic reductions, which bound the loop
// over pairs, #912) and a faster build.
static const Measured kGroups89Measured[] = {{8, 9}};
static const Entry<GroupsPolicy> kGroupsTable[] = {
    {{8, 9, ""},
     {/*columnParticles=*/96.0},
     {"D423",
      "JAC, Factor IX, Cellulose, STMV, and TIP3P (309,876 sites), NPT, RTX "
      "4090",
      "2026-11-18"},
     kGroups89Measured},
};

//===----------------------------------------------------------------------===//
// The traversal of AUTO (D[best-defaults])
//===----------------------------------------------------------------------===//

// The default keeps the rule of D261 (MATRIX or GROUPS by size): TOWERS
// was not measured against them as a default on the architecture.
const TraversalPolicy mdir::md_exec::policy::kDefaultTraversal = {
    /*towersLeast=*/-1};
static const Measurement kTraversalDefaultBasis = {
    "D261", "not measured as a choice of AUTO: the rule of D261 is kept",
    "2026-10-13"};
static const ArrayRef<Measured> kTraversalDefaultMeasured = {};

// On sm_89 TOWERS with its derived launch took less time a step than the
// structure of D261 from 4,608 atoms on (D[best-defaults], RTX 4090,
// mixed precision, 2 fs, two jobs in opposite orders): TIP3P boxes of
// 4,608, 6,912, 9,989 and 19,880 atoms, JAC, Factor IX, Cellulose and STMV.
// A box of 3,072 atoms ran 3% slower than the matrix, so the entry starts
// at the smallest box measured faster.
static const Measured kTraversal89Measured[] = {{8, 9}};
// The entry is keyed by the device model, not the architecture alone: a
// performance threshold of one model is not borrowed by another of the same
// compute capability (L4, L40), which keeps the default.
static const Entry<TraversalPolicy> kTraversalTable[] = {
    {{8, 9, "RTX 4090"},
     {/*towersLeast=*/4608},
     {"D[best-defaults]",
      "main 03506508 and the branch at c0c015d8: TIP3P boxes of 3,072 to "
      "19,880 atoms, JAC, Factor IX, Cellulose and STMV, RTX 4090",
      "2026-11-27"},
     kTraversal89Measured},
};

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

std::string DevicePolicy::describeFallback(bool fallback) const {
  // Without an architecture (mdir-opt, PTX for the default architecture)
  // the remarks are those of before the tables.
  if (!fallback || arch.empty())
    return "";
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
    policy.pruningFallback = !covers(entry->measured, key);
  }

  policy.scatter = kDefaultScatter;
  policy.scatterBasis = kScatterDefaultBasis;
  policy.scatterFallback = !covers(kScatterDefaultMeasured, key);
  if (const auto *entry = find<ScatterPolicy>(kScatterTable, key)) {
    policy.scatter = entry->values;
    policy.scatterBasis = entry->basis;
    policy.scatterFallback = !covers(entry->measured, key);
  }

  policy.launch = kDefaultLaunch;
  policy.launchBasis = kLaunchDefaultBasis;
  policy.launchFallback = true;
  if (const auto *entry = find<LaunchPolicy>(kLaunchTable, key)) {
    policy.launch = entry->values;
    policy.launchBasis = entry->basis;
    policy.launchFallback = !covers(entry->measured, key);
  }

  policy.segments = kDefaultSegments;
  policy.segmentsBasis = kSegmentsDefaultBasis;
  policy.segmentsFallback = !covers(kSegmentsDefaultMeasured, key);
  if (const auto *entry = find<SegmentsPolicy>(kSegmentsTable, key)) {
    policy.segments = entry->values;
    policy.segmentsBasis = entry->basis;
    policy.segmentsFallback = !covers(entry->measured, key);
  }

  policy.groups = kDefaultGroups;
  policy.groupsBasis = kGroupsDefaultBasis;
  policy.groupsFallback = !covers(kGroupsDefaultMeasured, key);
  if (const auto *entry = find<GroupsPolicy>(kGroupsTable, key)) {
    policy.groups = entry->values;
    policy.groupsBasis = entry->basis;
    policy.groupsFallback = !covers(entry->measured, key);
  }

  policy.traversal = kDefaultTraversal;
  policy.traversalBasis = kTraversalDefaultBasis;
  policy.traversalFallback = !covers(kTraversalDefaultMeasured, key);
  if (const auto *entry = find<TraversalPolicy>(kTraversalTable, key)) {
    policy.traversal = entry->values;
    policy.traversalBasis = entry->basis;
    policy.traversalFallback = !covers(entry->measured, key);
  }
  return policy;
}
