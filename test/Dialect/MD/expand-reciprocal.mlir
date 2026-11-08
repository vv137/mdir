// RUN: mdir-opt %s --md-expand-reciprocal=functions=selected --verify-each | FileCheck %s
// RUN: not mdir-opt %s --md-expand-reciprocal=functions=bad 2>&1 | FileCheck %s --check-prefix=BAD
// BAD: selected reciprocal expansion requires unlisted order-four rectangular Gaussian Coulomb SPME/OPTIMAL
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!m = !md.table<2, f64>
md.particle_set @atoms

// CHECK-LABEL: func.func @selected
// CHECK: md.checked_rectangular_cell
// CHECK: md.grid_stencil
// CHECK: md.scatter_stencil
// CHECK: md.linear_transform
// CHECK-SAME: direction("forward")
// CHECK: md.map_grid
// CHECK: md.sum_grid
// CHECK: md.linear_transform
// CHECK-SAME: direction("backward")
// CHECK: md.reduce_stencil
// CHECK: md.gather_stencil
// CHECK-NOT: md.reciprocal
// CHECK: return
func.func @selected(%x: !x, %h: !md.cell, %q: !q, %m: !m) -> (f64, !x, vector<9xf64>, !q) {
  %u, %f, %w, %p = md.reciprocal %x, %q, %h, %m grid([4, 5, 6]) order(4) beta(3.12) coulomb(138.935) : !x, !q, !m -> f64, !x, vector<9xf64>, !q
  return %u, %f, %w, %p : f64, !x, vector<9xf64>, !q
}

// CHECK-LABEL: func.func @unselected
// CHECK: md.reciprocal
func.func @unselected(%x: !x, %h: !md.cell, %q: !q, %m: !m) -> f64 {
  %u, %f, %w = md.reciprocal %x, %q, %h, %m grid([4, 5, 6]) order(4) beta(3.12) coulomb(138.935) : !x, !q, !m -> f64, !x, vector<9xf64>
  return %u : f64
}

func.func @bad(%x: !x, %h: !md.cell, %q: !q, %m: !m) -> f64 {
  %u, %f, %w = md.reciprocal %x, %q, %h, %m grid([4, 5, 6]) order(4) beta(3.12) coulomb(138.935) p3m : !x, !q, !m -> f64, !x, vector<9xf64>
  return %u : f64
}
