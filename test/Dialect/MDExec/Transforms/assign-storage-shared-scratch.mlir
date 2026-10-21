// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-storage | FileCheck %s --check-prefix=HOST

// The reciprocal sums of a function share their scratch
// (D315): each buffer is allocated once in the body of
// the function and taken by every sum that needs one of its shape. Two
// sums on the grid of 8^3 points take the same buffers; the sum on the
// grid of 6^3 points takes grids of its own and shares the buffers whose
// length does not depend on the grid: the places of the particles. The
// bricks (D108), which a sum finds zero and leaves zero, are set to zero
// once, and only shared with bricks. No sum takes one buffer twice.

!vec = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>
!grid = !md.table<2, f64>
md.particle_set @atoms

// CHECK-LABEL: func.func @three(
// CHECK:         gpu.memset async [%{{[0-9]+}}] %[[BRICKS:[a-z0-9_]+]],
// CHECK:         md_exec.reciprocal {{.*}} scratch(%[[FIXED:[a-z0-9_]+]], %[[REAL:[a-z0-9_]+]], %[[COMPLEX:[a-z0-9_]+]], %[[ROWS:[a-z0-9_]+]], %[[PLACES:[a-z0-9_]+]], %[[TABLES:[a-z0-9_]+]], %[[BRICKS]] :
// CHECK-SAME:    grid([8, 8, 8])
// CHECK-NOT:     gpu.alloc (%c
// CHECK-NOT:     gpu.memset
// CHECK:         md_exec.reciprocal {{.*}} scratch(%[[FIXED]], %[[REAL]], %[[COMPLEX]], %[[ROWS]], %[[PLACES]], %[[TABLES]], %[[BRICKS]] :
// CHECK-SAME:    grid([8, 8, 8])
// CHECK:         gpu.memset async [%{{[0-9]+}}] %[[BRICKS6:[a-z0-9_]+]],
// CHECK:         md_exec.reciprocal {{.*}} scratch(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[PLACES]], %{{[a-z0-9_]+}}, %[[BRICKS6]] :
// CHECK-SAME:    grid([6, 6, 6])
// CHECK-NOT:     md_exec.reciprocal
// HOST-LABEL: func.func @three(
// HOST:         md_exec.reciprocal {{.*}} scratch(%[[F:[a-z0-9_]+]], %[[R:[a-z0-9_]+]], %[[C:[a-z0-9_]+]], %[[T:[a-z0-9_]+]] :
// HOST:         md_exec.reciprocal {{.*}} scratch(%[[F]], %[[R]], %[[C]], %[[T]] :
// HOST:         md_exec.reciprocal {{.*}} scratch(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} :
// HOST-SAME:    grid([6, 6, 6])
func.func @three(%x: !vec, %q: !real, %cell: !md.cell,
                 %b: memref<?x?xf64>, %b6: memref<?x?xf64>)
    -> (f64, f64, f64, !vec, !vec, !vec) {
  %moduli = mdrt.from_buffer %b : memref<?x?xf64> to !grid
  %moduli6 = mdrt.from_buffer %b6 : memref<?x?xf64> to !grid
  %u1, %w1, %f1 = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  %u2, %w2, %f2 = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.5) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  %u3, %w3, %f3 = md_exec.reciprocal %x, %q, %cell, %moduli6
      grid([6, 6, 6]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u1, %u2, %u3, %f1, %f2, %f3 : f64, f64, f64, !vec, !vec, !vec
}
