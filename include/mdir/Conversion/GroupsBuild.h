// The device build of the groups of 16 particles that share a list of
// neighbors (D89, docs/neighbor-groups.md), written by an emitter
// (D[groups-emitter]).

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
/// calls: the declarations of the device primitives (CompactionEmitter.h),
/// the helpers `mdrt_gpu_groups_grid`, `mdrt_gpu_groups_partner`, and
/// `mdrt_gpu_groups_image` (with `triclinic`, also
/// `mdrt_gpu_groups_image_triclinic` and `mdrt_gpu_groups_offset_triclinic`),
/// and the build, named by getGroupsBuildName.
///
/// `triclinic` builds in the brick of a lower-triangular cell, a
/// `vector<6xf64>` (docs/triclinic.md, P3); otherwise the cell is the
/// `vector<3xf64>` of the edges of an orthorhombic one. `singlePassScan`
/// takes the decoupled look-back scan for the three counting sorts (D321)
/// in place of the three-launch scan.
///
/// The calls of the device primitives are left as calls, which
/// expandPrimitiveCalls replaces with the code of their emitters.
void emitGroupsBuild(mlir::ModuleOp module, bool triclinic,
                     bool singlePassScan);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_GROUPSBUILD_H
