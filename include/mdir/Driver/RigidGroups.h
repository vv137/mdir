// The rigid groups of a topology and their kinds (D403).
//
// Two rules, stated on the constraints and the site definitions alone:
//
//   R1. A molecule is a rigid group when a constraint fixes the distance
//       of every pair of its massive atoms, its frame. Rigid groups with
//       the same number of particles, the same charge and type of each,
//       the same frame distances, and the same site weights are copies of
//       one kind.
//   R2. A massless site of a rigid group is a combination of the frame
//       atoms with constant weights that sum to 1,
//       r_s = sum_k w_k r_k.
//
// docs/constraints.md, Section 3, has the derivations.

#ifndef MDIR_DRIVER_RIGIDGROUPS_H
#define MDIR_DRIVER_RIGIDGROUPS_H

#include "mdir/Driver/Topology.h"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace mdir {
namespace driver {

/// One kind of rigid group and its copies. A molecule's particles are
/// taken in the order of their numbers; a place is a position in that
/// order, from 0.
struct RigidKind {
  /// The particles of one molecule, frame atoms and sites.
  unsigned size = 0;
  /// The places of the massive atoms, ascending: the frame.
  std::vector<unsigned> frame;
  /// The charge (e) and the type of each place.
  std::vector<double> charges;
  std::vector<unsigned> types;
  /// The constrained distances of the frame (nm), one for each pair of
  /// entries (a, b) of `frame`, a < b, in lexicographic order:
  /// (0, 1), (0, 2), ..., (1, 2), ...
  std::vector<double> distances;
  /// A massless site: its place and its weights over the atoms of
  /// `frame`, in the order of `frame`. They sum to 1.
  struct Site {
    unsigned place = 0;
    std::vector<double> weights;
  };
  std::vector<Site> sites;
  /// The particles of the copies, `size` for each, the copies in the order
  /// of their first particle.
  std::vector<unsigned> members;

  /// Facts that a consumer may ask, as counts of copies; none of them
  /// decides whether a molecule is a rigid group or which kind it is.
  /// Copies whose particles are not consecutive numbers.
  size_t scattered = 0;
  /// Copies with a pair of their own particles that the topology does not
  /// exclude.
  size_t incompleteExclusions = 0;
  /// Copies with an excluded pair whose other particle is outside them.
  size_t outsideExclusions = 0;

  size_t getCount() const { return size ? members.size() / size : 0; }
};

/// Why a molecule is not a rigid group.
enum class RigidReason {
  /// Some pair of its massive atoms has no constrained distance.
  FlexibleFrame,
  /// One of its massless sites is no constant-weight combination of the
  /// frame atoms.
  SiteNotConstantWeight,
};
constexpr unsigned NumRigidReasons = 2;

/// The words of a reason in a report, and its key in JSON.
const char *getRigidReasonText(RigidReason reason);
const char *getRigidReasonKey(RigidReason reason);

struct RigidGroups {
  /// Most populous first; kinds of equal count in the order of their first
  /// particle.
  std::vector<RigidKind> kinds;
  /// The molecules of the topology: the connected components of its bonds,
  /// its constrained distances, and the links of sites to their atoms.
  size_t molecules = 0;
  /// The molecules that are no rigid group, by reason.
  std::array<size_t, NumRigidReasons> leftOut{};
};

/// The weights over (i, j, k) of the site
/// `x_i + a (u + v) / |u + v|`, with `u` and `v` the unit vectors from `i`
/// to `j` and to `k`, on a frame that holds the distances `ij`, `ik`, and
/// `jk`:
///   w_j = a sqrt(ik / ij) / sqrt((ij + ik)^2 - jk^2),
///   w_k = a sqrt(ij / ik) / sqrt((ij + ik)^2 - jk^2),
///   w_i = 1 - w_j - w_k.
/// None where the form has no direction, `jk >= ij + ik`, or a length is
/// not positive and finite.
std::optional<std::array<double, 3>>
getBisectorWeights(double a, double ij, double ik, double jk);

/// The rigid groups of a validated topology (validateTopology), prepared
/// or not: a topology whose constraints are not resolved has none beyond
/// its single atoms. Reads the topology only.
RigidGroups findRigidGroups(const Topology &topology);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_RIGIDGROUPS_H
