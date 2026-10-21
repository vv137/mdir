// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" | FileCheck %s

// A tiled reciprocal sum takes bins but no bricks. Its weighted gather is
// selected by the lowering from the sum's order, grid type, and algebra.

!vec = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>
!grid = !md.table<2, f64>
md.particle_set @atoms

// CHECK-LABEL: func.func @tiled(
// CHECK-NOT: gpu.memset
// CHECK: md_exec.reciprocal {{.*}} scratch(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<?xi64, 1>, memref<?xf32, 1>, memref<?xf32, 1>, memref<?xf64, 1>, memref<?xf32, 1>, memref<?xf32, 1>, memref<?xi32, 1>)
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
