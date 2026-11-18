// The device build of the groups of 16 particles that share a list of
// neighbors (D89, docs/neighbor-groups.md), written by an emitter
// (D404).

#ifndef MDIR_CONVERSION_GROUPSBUILD_H
#define MDIR_CONVERSION_GROUPSBUILD_H

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace mdir {
namespace kernels {

/// The name of the function that builds the groups:
/// `mdrt_gpu_build_neighbors_groups`, with `_triclinic` for a triclinic
/// cell.
std::string getGroupsBuildName(bool triclinic);

/// Appends to `module` the build of the groups on a device and what it
/// calls: the helpers `mdrt_gpu_groups_grid`, `mdrt_gpu_groups_partner`, and
/// `mdrt_gpu_groups_image` (with `triclinic`, also
/// `mdrt_gpu_groups_image_triclinic` and `mdrt_gpu_groups_offset_triclinic`),
/// and the build, named by getGroupsBuildName.
///
/// `triclinic` builds in the brick of a lower-triangular cell, a
/// `vector<6xf64>` (docs/triclinic.md, P3); otherwise the cell is the
/// `vector<3xf64>` of the edges of an orthorhombic one. `singlePassScan`
/// takes the decoupled look-back scan for the three counting sorts (D321)
/// in place of the three-launch scan. `columnParticles` is the number of
/// particles of a column of the compact order, in a cube of the column's
/// width (GroupsPolicy, D423).
///
/// Its scans, scatters, sorts within bins, and reads of the ragged rows are
/// left as md_alg ops (D406), which kernels::lowerAlgToSIMT
/// (MDAlgLowering.h) replaces with the code of their emitters.
void emitGroupsBuild(mlir::ModuleOp module, bool triclinic,
                     bool singlePassScan, double columnParticles);

/// The name of the build over the units of `kinds` taken rigid kinds and
/// the rest (D419):
/// `mdrt_gpu_build_segmented_pairs_k<kinds>`, with `_triclinic`.
std::string getSegmentsBuildName(bool triclinic, unsigned kinds);

/// Appends to `module` the build over units: the groups build of D404 with
/// units for particles, each at its reference particle, a group of one
/// kind, a search for each segment s(f, g) with its reach R + e_f + e_g,
/// and the records in the order of the segments (D417, Sections 1 and 2;
/// S1's layout). The helpers that the groups' build shares are added only
/// if the module does not hold them.
///
/// Its arguments, after those of the groups' build up to the reach: the
/// covering reach of each segment in the order of s (getSegmentCoveringReach
/// of NeighborCoverage.h), and the end of the units of each rank (the rest,
/// then the kinds from K - 1 to 0); then the excluded pairs, the
/// order of the places by unit, the place of each unit, the staging
/// entries and masks, the count of each list (group g, segment s at
/// g S + s), the blocks' names and ordinals, the shifts of the places, the
/// reference particle and the value of `order` of each unit, the unit of
/// each particle of the rest, the last block of each list, the order of
/// S1's layout, the records' entries, masks, groups, and counts, the
/// offsets of the segments, and the sizes on the host.
/// `columnUnits` is the number of units of a column of the plan of the
/// columns, which the cubes cut along z (SegmentsPolicy, D420).
void emitSegmentsBuild(mlir::ModuleOp module, bool triclinic,
                       bool singlePassScan, unsigned kinds,
                       double columnUnits);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_GROUPSBUILD_H
