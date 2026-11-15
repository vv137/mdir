// The regions of the segments of a list of units (D[md-alg-segments]):
// their syntax and their checks, shared by md_exec.segment_for and
// md_alg.traverse_pairs.
//
//   segment(f, g) sites [(p, q) [a, ...], ...] { ^bb0(...): ... }
//
// One region for each segment s(f, g) that the op visits, in increasing
// s(f, g). Its table names, for each slice of the region, the place p of
// the unit of kind f, the place q of the unit of kind g, and the values a
// (destinations, then sums) that the slice yields: its mask. The region
// takes, for each of its M site pairs, r^2 and the displacement, of one
// float type T, then for each field of `ins` its values at the m_f members
// of the first unit and the m_g of the second. It yields, site pair by site
// pair, the values of each mask in increasing order.

#ifndef MDIR_DIALECT_MDRT_SEGMENTREGIONS_H
#define MDIR_DIALECT_MDRT_SEGMENTREGIONS_H

#include "mdir/Dialect/MDRT/MDRTTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mdir::mdrt {

/// One entry of a segment's table: its places and its mask.
struct SitePair {
  unsigned p = 0, q = 0;
  llvm::SmallVector<unsigned, 4> values;
};

/// Parses the regions and their tables into `segments` (one [f, g] each)
/// and `sites` (one array of [p, q, a...] each).
mlir::ParseResult
parseSegmentRegions(mlir::OpAsmParser &parser, mlir::ArrayAttr &segments,
                    mlir::ArrayAttr &sites,
                    llvm::SmallVectorImpl<std::unique_ptr<mlir::Region>> &regions);
void printSegmentRegions(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                         mlir::ArrayAttr segments, mlir::ArrayAttr sites,
                         mlir::MutableArrayRef<mlir::Region> regions);

/// The kinds (f, g) of region `index`.
std::pair<unsigned, unsigned> getSegmentKinds(mlir::ArrayAttr segments,
                                              unsigned index);
/// The table of region `index`.
llvm::SmallVector<SitePair> getSitePairs(mlir::ArrayAttr sites,
                                         unsigned index);
/// The attributes of a segment and of its table.
mlir::Attribute getSegmentAttr(mlir::MLIRContext *context, unsigned f,
                               unsigned g);
mlir::Attribute getSitesAttr(mlir::MLIRContext *context,
                             llvm::ArrayRef<SitePair> pairs);

/// What a region of a segment takes and yields.
struct SegmentSignature {
  /// The element type of each field of `ins`, one argument per member.
  llvm::SmallVector<mlir::Type> fields;
  /// The type of each value, the destinations, then the sums.
  llvm::SmallVector<mlir::Type> values;
  /// The name of the op that ends a region.
  llvm::StringRef terminator;
};

/// The arguments of a region with `pairs` site pairs in the float type
/// `real`, over units of `members` and `partners` members.
llvm::SmallVector<mlir::Type>
getSegmentArgumentTypes(mlir::Type real, unsigned pairs, unsigned members,
                        unsigned partners, const SegmentSignature &signature);

/// Checks the regions of `op` against the list `type` (rules V2 to V7 of
/// D[md-alg-segments], Section 5): the segments, their tables and masks,
/// the arguments, the yields, and that the slice of a site pair reads no
/// argument of another.
mlir::LogicalResult verifySegmentRegions(mlir::Operation *op,
                                         SegmentedPairsType type,
                                         mlir::ArrayAttr segments,
                                         mlir::ArrayAttr sites,
                                         mlir::MutableArrayRef<mlir::Region> regions,
                                         const SegmentSignature &signature);

/// The ops of `block` that the yields of site pair `pair` depend on, in
/// the order of the block, and those yields. The slices are separable
/// (verifySegmentRegions).
llvm::SmallVector<mlir::Operation *>
getSiteSlice(mlir::Block &block, unsigned pair, unsigned yieldBegin,
             unsigned yieldEnd);

} // namespace mdir::mdrt

#endif // MDIR_DIALECT_MDRT_SEGMENTREGIONS_H
