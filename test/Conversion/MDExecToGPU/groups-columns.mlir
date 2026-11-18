// The particles of a column of the compact order of the groups' build are an
// entry of the tuning policies (GroupsPolicy, D423): 96 on
// sm_89, where they were measured, and D89's 64 everywhere else (sm_86,
// sm_120, no architecture), which no measurement covered. The option
// `groups-column-particles` overrides the entry; it takes -1 (the entry) or
// a positive count and refuses anything else.
//
// RUN: mdir-opt %S/../../Integration/eam.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-precision="mode=mixed" --md-exec-assign-algebra \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-md-alg -o %t.mlir
// RUN: mdir-opt %t.mlir --convert-md-exec-to-gpu="arch=sm_89" | FileCheck %s --check-prefix=C96
// RUN: mdir-opt %t.mlir --convert-md-exec-to-gpu="arch=sm_89 groups-column-particles=-1" | FileCheck %s --check-prefix=C96
// RUN: mdir-opt %t.mlir --convert-md-exec-to-gpu="arch=sm_86" | FileCheck %s --check-prefix=C64
// RUN: mdir-opt %t.mlir --convert-md-exec-to-gpu="arch=sm_120" | FileCheck %s --check-prefix=C64
// RUN: mdir-opt %t.mlir --convert-md-exec-to-gpu | FileCheck %s --check-prefix=C64
// RUN: mdir-opt %t.mlir --convert-md-exec-to-gpu="arch=sm_89 groups-column-particles=64" | FileCheck %s --check-prefix=C64
// RUN: mdir-opt %t.mlir --convert-md-exec-to-gpu="arch=sm_120 groups-column-particles=96" | FileCheck %s --check-prefix=C96
//
// The option refuses 0 and a negative count other than -1.
// RUN: not mdir-opt %t.mlir --convert-md-exec-to-gpu="arch=sm_89 groups-column-particles=0" -o /dev/null 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not mdir-opt %t.mlir --convert-md-exec-to-gpu="arch=sm_89 groups-column-particles=-5" -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEG

// C96: func.func private @mdrt_gpu_build_neighbors_groups(
// C96: md_alg.grid_plan %{{.*}}, %{{.*}} : f64, f64 volume %{{.*}} : f64 occupancy 9.600000e+01 : f64 of
// C64: func.func private @mdrt_gpu_build_neighbors_groups(
// C64: md_alg.grid_plan %{{.*}}, %{{.*}} : f64, f64 volume %{{.*}} : f64 occupancy 6.400000e+01 : f64 of
// ZERO: error: convert-md-exec-to-gpu: groups-column-particles=0 is not a number of particles: it takes -1 (the entry of the tuning policy for the architecture) or a positive count
// NEG: error: convert-md-exec-to-gpu: groups-column-particles=-5 is not a number of particles
