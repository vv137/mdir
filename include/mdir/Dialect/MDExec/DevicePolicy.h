// The tuning policies of the device lowering (D[tuning-policies],
// issue #433, docs/device-primitives.md, Section 8).
//
// Every threshold and launch parameter that a chooser or a lowering of a
// device program takes from a measurement is an entry of one table for
// each strategy, after the dispatch policies of CUB (one policy record for
// each algorithm and architecture, selected for the target; the
// algorithm reads its parameters only from the selected record). A table
// is keyed by the target's architecture and by a device class, where the
// measurements differ within one architecture. An architecture without
// an entry of its own takes the table's default entry. Each entry names
// the measurement behind it (the decision, the systems, the devices, and
// the date) and the architectures that it covered; a selected entry that
// was not measured on the architecture is a fallback, and the selection
// says so (`DevicePolicy::pruningFallback` and the others).
//
// The architecture is the name that the target states (`sm_89`,
// `sm_120a`); the tables do not assume a vendor beyond that name, and a
// name that they do not parse takes the defaults. Capabilities of the
// device (the subgroup width, atomics) are not tuning and are not here
// (#558).

#ifndef MDIR_DIALECT_MDEXEC_DEVICEPOLICY_H
#define MDIR_DIALECT_MDEXEC_DEVICEPOLICY_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mdir {
namespace md_exec {
namespace policy {

/// The measurement behind an entry: the decision that states it, and the
/// systems, devices, and date of the measurement.
struct Measurement {
  llvm::StringRef decision;
  llvm::StringRef measured;
  llvm::StringRef date;
};

/// The pruning of a dual list of groups in the loop over pairs that
/// follows its refresh (D284).
struct PruningPolicy {
  /// The groups of the last build from which the loop prunes.
  int64_t fusedGroups;
};

/// The strategies of the scatter-accumulates (D282, D283, D291),
/// which `md-exec-choose-scatter` reads.
struct ScatterPolicy {
  /// With `tile=auto`, the fewest points of a grid that is tiled (D282).
  int64_t tileMinPoints;
  /// The points of an edge of a bin of the tiles (D282).
  int64_t tileEdge;
  /// With `hash=auto`, the fewest tuples (or values of a
  /// scatter-accumulate) of a launch that take the table (D283).
  int64_t hashLeast;
  /// The slots that a search of the table tries before an atomic (D283).
  int64_t hashProbes;
  /// The fewest units of work of a launch over groups that take the table
  /// with `pair-hash=on` (D291).
  int64_t pairHashLeast;
};

/// The launches of the kernels of the loops (D283, D293, D102, #411).
struct LaunchPolicy {
  /// The threads of a block of the loops over tuples, pairs, and
  /// particles (`convert-md-exec-to-gpu`), and so the slots of the tables
  /// of D283, which `md-exec-choose-scatter` sizes for it.
  int64_t blockSize;
  /// The threads that share the neighbors or tuples of a particle.
  int64_t rowLanes;
  /// The threads of a block of a kernel of the batched bodies of stages
  /// (D293).
  int64_t stageBlockSize;
  /// The block memory that a kernel of a strategy may take: the static
  /// limit of a block, which also keeps a block's occupancy where the
  /// measurements of D282 and D283 put it.
  int64_t blockMemoryBytes;
  /// The lanes of a particle in the gathering of PME on a device: these up
  /// to the order `gatherSmallOrder` (D102), `gatherLanes` above it
  /// (#411).
  int64_t gatherSmallLanes;
  int64_t gatherSmallOrder;
  int64_t gatherLanes;

  /// The lanes of a particle in a gathering of order `order`.
  int64_t getGatherLanes(int64_t order) const {
    return order <= gatherSmallOrder ? gatherSmallLanes : gatherLanes;
  }
};

/// The key of an entry: an architecture, as a major version and a minor
/// one (`sm_120` is 12 and 0), the minor -1 for every minor version of
/// the major; and a device class, empty for every device of the
/// architecture.
struct DeviceKey {
  int64_t major = -1;
  int64_t minor = -1;
  llvm::StringRef deviceClass;
};

/// The architecture that `arch` names, `sm_<major><minor>` with a
/// suffix of letters (`sm_120a`), or nothing for another name or none.
std::optional<DeviceKey> parseArchitecture(llvm::StringRef arch);

/// The entries of every table selected for one architecture and device
/// class.
struct DevicePolicy {
  PruningPolicy pruning;
  ScatterPolicy scatter;
  LaunchPolicy launch;
  /// The measurement behind each selected entry.
  Measurement pruningBasis, scatterBasis, launchBasis;
  /// The tables whose selected entry was not measured on the
  /// architecture: the default where it has no entry of its own, or an
  /// entry of its major version that covered another minor one.
  bool pruningFallback = true, scatterFallback = true, launchFallback = true;
  /// The architecture and class as given.
  std::string arch, deviceClass;

  /// The words of an explain remark for a choice from the table of
  /// `fallback`: nothing where the entry was measured on the
  /// architecture or no architecture is given, else that the default was
  /// taken.
  std::string describeFallback(bool fallback) const;
};

/// The entries for the architecture `arch` and the device class
/// `deviceClass` (empty: any device of the architecture). An exact entry
/// of the architecture and class wins, then one of the architecture, then
/// one of its major version, then the default.
DevicePolicy getDevicePolicy(llvm::StringRef arch,
                             llvm::StringRef deviceClass = "");

/// The default entries, which an architecture without one of its own
/// takes. The pass options that take a value from the table default to
/// -1, "the entry of the target's architecture".
extern const PruningPolicy kDefaultPruning;
extern const ScatterPolicy kDefaultScatter;
extern const LaunchPolicy kDefaultLaunch;

} // namespace policy
} // namespace md_exec
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_DEVICEPOLICY_H
