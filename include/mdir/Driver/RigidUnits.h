// The route of the rigid units (D424; D417, slice S4b).
//
// `[performance] rigid_units` decides whether the pair term is built over
// the segments of the rigid kinds of D403 (D405, D417). The resolution runs
// once, before anything is built, next to that of the neighbor structure
// (D261): it chooses the kinds, refuses `ON` where the route cannot apply,
// measures the covering radii at the positions that the run begins with,
// and records why, for `mdir check`, the log, and the manifest.
#ifndef MDIR_DRIVER_RIGIDUNITS_H
#define MDIR_DRIVER_RIGIDUNITS_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/RigidGroups.h"
#include "mdir/Driver/System.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <array>
#include <vector>

namespace mdir {
namespace driver {

/// The numerical margin of a covering radius over the exact distance of the
/// places from their anchor, in nm (#795, ruling 5: 10^-3 Å).
constexpr double rigidUnitsMargin = 1e-4;

/// Resolves `control.rigidUnits` for `system` at its positions and cell:
/// fills `control.rigidUnitsRoute`, or refuses `ON` where the route cannot
/// apply, with the reason. `AUTO` takes the general route until its rule is
/// measured; `OFF` is the general route. Call it after
/// resolveNeighborStructure and once the positions that the run begins
/// with are known.
llvm::Error resolveRigidUnits(Control &control, const System &system);

/// What the build of the pair term over the segments takes from a resolved
/// route: the taken kinds, their covering radii, and the anchored members
/// of the rest with its radius.
struct RigidUnitsPlan {
  RigidGroups groups;
  /// Indices into `groups.kinds`, in the order of the segments.
  std::vector<unsigned> taken;
  std::vector<double> radii;
  /// (member, anchor) pairs of the rest (#799).
  std::vector<std::array<unsigned, 2>> anchored;
  double restRadius = 0.0;
};

/// The plan of `route`, which must take the units, for `topology`.
RigidUnitsPlan getRigidUnitsPlan(const RigidUnitsRoute &route,
                                 const Topology &topology);

/// The first line of the report of a resolved route, after "rigid units: ",
/// and the lines of its kinds, for `mdir check` and the log.
std::string describeRigidUnitsRoute(const Control &control);
std::vector<std::string> describeRigidUnitsKinds(const Control &control);
/// The same as an object of `mdir check`'s JSON report and of the
/// manifest, lengths in Å.
llvm::json::Object getRigidUnitsJSON(const Control &control);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_RIGIDUNITS_H
