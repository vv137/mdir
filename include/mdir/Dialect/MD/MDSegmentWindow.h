// The segments of rigid kinds and the site geometry of their windows
// (D405).
#ifndef MDIR_DIALECT_MD_MDSEGMENTWINDOW_H
#define MDIR_DIALECT_MD_MDSEGMENTWINDOW_H

#include "mdir/Dialect/MD/MDTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include <array>
#include <optional>
#include <utility>

namespace mlir {
class Block;
class Location;
class OpBuilder;
class Operation;
} // namespace mlir

namespace mdir::md {

/// Rule R3: the number of the segment of the kinds g <= f,
/// s(f, g) = f (f + 1) / 2 + g.
inline unsigned getSegmentNumber(unsigned f, unsigned g) {
  return f * (f + 1) / 2 + g;
}
/// The (K + 1)(K + 2)/2 segments of K taken kinds and the rest.
inline unsigned getNumSegments(unsigned takenKinds) {
  return (takenKinds + 1) * (takenKinds + 2) / 2;
}
/// The kinds (f, g) of the segment numbered s.
std::pair<unsigned, unsigned> getSegmentKinds(unsigned s);

/// One molecule of a segment's tuple: a taken kind, or the rest (no
/// descriptor), whose molecules are single particles.
struct SegmentSide {
  RigidKindAttr kind;
  /// The frame size m: the molecule's members in a tuple.
  unsigned getFrameSize() const { return kind ? kind.getFrame().size() : 1; }
  /// The size n: frame atoms and sites.
  unsigned getSize() const { return kind ? kind.getSize() : 1; }
};

/// The shape of the tuples and window arguments of a segment (f, g):
/// side `a` is the molecule of f, side `b` that of g.
struct SegmentShape {
  SegmentSide a, b;
  unsigned getArity() const { return a.getFrameSize() + b.getFrameSize(); }
  /// The anchor displacement and the frame offsets of a, then of b.
  unsigned getNumCoordinates() const { return getArity() - 1; }
  /// The force slots of one site pair: the tuple's frame atoms, or with
  /// site forces every place of the two molecules.
  unsigned getNumSlots(bool siteForces) const {
    return siteForces ? a.getSize() + b.getSize() : getArity();
  }
  /// For each window coordinate, the tuple members (positive, negative):
  /// (a_0, b_0) for the anchor displacement, (a_k, a_0) and (b_k, b_0)
  /// for the frame offsets.
  llvm::SmallVector<std::array<unsigned, 2>> getCoordinateMembers() const;
};

/// The shape of segment (f, g) of `pairs` (an md.kind_pairs), g <= f <= K.
SegmentShape getSegmentShape(mlir::Operation *pairs, unsigned f, unsigned g);
/// The shape of the segment that a segment window reads. The window must
/// have been verified.
SegmentShape getSegmentShape(mlir::Operation *window);

/// The places (p, q) of the site pair whose displacement this is, proved
/// symbolically from the window's arithmetic: p a place of a and q of b,
/// in the slot numbering of site forces (q counted from a's size). None if
/// the displacement is not of the form d_0 + c_p - c_q (MDKindPairs.cpp).
std::optional<std::array<unsigned, 2>>
getSegmentWindowSites(mlir::Value displacement, mlir::Block &,
                      const SegmentShape &);

/// The constant transpose of a molecule's site placement: from one force
/// per place (n values), the force on each frame atom (m values),
/// F_{a_k} = F_{a_k} + sum_s w_{s,k} F_s. A molecule without sites is
/// unchanged.
llvm::SmallVector<mlir::Value>
emitSiteForceTranspose(mlir::OpBuilder &, mlir::Location, mlir::ValueRange,
                       const SegmentSide &);

} // namespace mdir::md
#endif
