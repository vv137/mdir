// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true \
// RUN:   --md-exec-assign-storage="memory=device" | FileCheck %s --check-prefix=FIXED

// Gaussian cubic forces use the same packed cell workspace for both input
// scatter choices. Fixed-order spreading retains its integer grid.

!vec = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>
!grid = !md.table<2, f64>
md.particle_set @atoms

// CHECK-LABEL: func.func @tiled(
// CHECK-NOT: gpu.memset
// CHECK: md_exec.reciprocal {{.*}} scratch({{[^:]+}} : memref<?xi64, 1>, memref<?xf32, 1>, memref<?xf32, 1>, memref<?xf64, 1>, memref<?x4xf32, 1>, memref<?xf32, 1>, memref<?x4xf32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi64, 1>)
// CHECK-SAME: scatter(tile(8))
func.func @tiled(%x: !vec, %q: !real, %cell: !md.cell,
                 %b: memref<?x?xf64>) -> (f64, vector<9xf64>, !vec) {
  %moduli = mdrt.from_buffer %b : memref<?x?xf64> to !grid
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(4) beta(3.0) coulomb(138.9)
      scatter(tile(8))
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %w, %f : f64, vector<9xf64>, !vec
}

// CHECK-LABEL: func.func @non_tiled(
// CHECK-NOT: gpu.memset
// CHECK: md_exec.reciprocal {{.*}} scratch({{[^:]+}} : memref<?xi64, 1>, memref<?xf32, 1>, memref<?xf32, 1>, memref<?xf64, 1>, memref<?x4xf32, 1>, memref<?xf32, 1>, memref<?x4xf32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi64, 1>)
// FIXED-LABEL: func.func @non_tiled(
// FIXED-NOT: gpu.memset
// FIXED: md_exec.reciprocal {{.*}} scratch({{[^:]+}} : memref<?xi64, 1>, memref<?xf32, 1>, memref<?xf32, 1>, memref<?xf64, 1>, memref<?x4xf32, 1>, memref<?xf32, 1>, memref<?x4xf32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi64, 1>)
func.func @non_tiled(%x: !vec, %q: !real, %cell: !md.cell,
                     %b: memref<?x?xf64>) -> (f64, vector<9xf64>, !vec) {
  %moduli = mdrt.from_buffer %b : memref<?x?xf64> to !grid
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %w, %f : f64, vector<9xf64>, !vec
}
