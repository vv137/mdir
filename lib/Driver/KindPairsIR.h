// The segments of the pairs of rigid kinds and their site windows, built
// from the kinds of D403 (D[kind-pairs]).
#ifndef MDIR_LIB_DRIVER_KINDPAIRSIR_H
#define MDIR_LIB_DRIVER_KINDPAIRSIR_H

#include "mdir/Driver/RigidGroups.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir {
class Block;
class OpBuilder;
} // namespace mlir

namespace mdir::driver {

/// The enabling condition of D[kind-pairs]: the generator takes at most
/// this many kinds. The ops and their verifier take any number.
inline constexpr unsigned MaxPairKinds = 2;

/// A kind of D403 as the pair term takes it: its places, and for each
/// place the data that decide its site pairs.
struct PairKind {
  /// The places of the frame, ascending; the first is the anchor.
  std::vector<unsigned> frame;
  /// The massless sites and their weights over the frame (rule R2).
  std::vector<RigidKind::Site> sites;
  /// The charge (e) and the type of each place.
  std::vector<double> charges;
  std::vector<unsigned> types;
  /// Whether the type of each place has Lennard-Jones with any type.
  std::vector<bool> lennardJones;
  /// The covering radius of the places about the anchor (nm) that the
  /// consumer checks.
  double extent = 0.0;

  unsigned getSize() const { return charges.size(); }
};

/// The taken kinds, numbered 0, ..., K - 1, and the Lennard-Jones table's
/// zero entries. The rest is kind K.
struct KindPairsPlan {
  std::vector<PairKind> kinds;
  unsigned typeCount = 0;
  /// Whether epsilon(a, b) is nonzero, for types a and b.
  std::vector<bool> lennardJonesPairs;

  bool hasLennardJones(unsigned a, unsigned b) const {
    return lennardJonesPairs[a * typeCount + b];
  }
};

/// The pair data of the kinds that a consumer takes, in its order, with
/// their covering radii. Refused, with the reason: more than MaxPairKinds
/// kinds, or a kind whose copies have one of their own pairs not excluded
/// or an exclusion with another particle (D403's facts).
llvm::Expected<KindPairsPlan>
planKindPairs(const Topology &, llvm::ArrayRef<const RigidKind *> kinds,
              llvm::ArrayRef<double> extents);

namespace builder {

/// The names of the tuple sets of kind f and of segment (f, g).
std::string getKindTupleSetName(unsigned f);
std::string getSegmentTupleSetName(unsigned f, unsigned g);
/// Declares the tuple sets of the taken kinds (disjoint) and of the
/// segments on the particle set `atoms`, at the builder's position.
void declareKindPairsTupleSets(mlir::OpBuilder &, const KindPairsPlan &,
                               llvm::StringRef atoms);

/// One site pair that a segment takes. The displacement is x_p - x_q in
/// the anchor image of the window. Each pair still needs its own site
/// cutoff; a covering bound on the anchors only selects candidates.
struct KindSitePair {
  mlir::Value displacement, typeI, typeJ, chargeI, chargeJ;
  bool lennardJones = false, coulomb = false;
  mlir::Value alchemicalI, alchemicalJ;
};

struct KindPairsInputs {
  /// The relations of the taken kinds, in the plan's order, and those of
  /// md.kind_pairs.
  llvm::SmallVector<mlir::Value> kinds;
  mlir::Value excluded, anchored;
  double radius = 0.0;
  mlir::Value positions, cell, types, charges;
  double cutoff = 0.0;
  bool lennardJones = true, coulomb = true;
  mlir::Value alchemical;
};

/// The pair energy of one site pair: the caller clones the system's pair
/// term (cloneKindPairEnergy).
using PairEnergy =
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, const KindSitePair &)>;

/// The pair term over all segments of md.kind_pairs, summed: one
/// md.sum_segment_window for each segment that has a site pair.
mlir::Value buildKindPairsTerm(mlir::OpBuilder &, const KindPairsPlan &,
                               const KindPairsInputs &, PairEnergy);

/// Clones the existing scalar primary pair body for one site pair. The
/// pair's values replace its gathered values; captured mixing tables and
/// parameters keep their actual SSA values. No other Lennard-Jones or
/// Coulomb expression is written.
mlir::Value cloneKindPairEnergy(mlir::OpBuilder &, mlir::Block &,
                                const KindSitePair &);

} // namespace builder
} // namespace mdir::driver
#endif
