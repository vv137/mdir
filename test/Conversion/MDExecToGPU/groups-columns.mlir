// The particles of a column of the compact order of the groups' build are an
// entry of the tuning policies (GroupsPolicy, D[groups-columns]): 96 by
// default and on sm_89, where they were measured; the option
// `groups-column-particles` overrides the entry.
//
// RUN: mdir-opt %S/../../Integration/eam.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-precision="mode=mixed" --md-exec-assign-algebra \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-md-alg \
// RUN:     --convert-md-exec-to-gpu="arch=sm_89" | FileCheck %s --check-prefix=POLICY
// RUN: mdir-opt %S/../../Integration/eam.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-precision="mode=mixed" --md-exec-assign-algebra \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-md-alg \
// RUN:     --convert-md-exec-to-gpu | FileCheck %s --check-prefix=POLICY
// RUN: mdir-opt %S/../../Integration/eam.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-precision="mode=mixed" --md-exec-assign-algebra \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-md-alg \
// RUN:     --convert-md-exec-to-gpu="arch=sm_89 groups-column-particles=64" | FileCheck %s --check-prefix=GIVEN

// POLICY: func.func private @mdrt_gpu_build_neighbors_groups(
// POLICY: md_alg.grid_plan %{{.*}}, %{{.*}} : f64, f64 volume %{{.*}} : f64 occupancy 9.600000e+01 : f64 of
// GIVEN: func.func private @mdrt_gpu_build_neighbors_groups(
// GIVEN: md_alg.grid_plan %{{.*}}, %{{.*}} : f64, f64 volume %{{.*}} : f64 occupancy 6.400000e+01 : f64 of
