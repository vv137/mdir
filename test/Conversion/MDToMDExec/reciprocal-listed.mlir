// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.25 width=48" | FileCheck %s
// RUN: mdir-opt %s | mdir-opt | FileCheck %s --check-prefix=ROUND

// A reciprocal sum over a list of the particles whose charge can be
// nonzero (D[pme-skip-zero]): the sum of md_exec that gives the forces
// takes the list; the one that gives the potential at the particles takes
// none, since that potential is not zero where the charge is.

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!ids = !md.field<@atoms, i32>
!grid = !md.table<2, f64>
md.particle_set @atoms

// ROUND-LABEL: md.function @forces(
// ROUND: md.reciprocal %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} listed(%{{.*}}) grid([8, 8, 8])
// ROUND-SAME: : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>, !md.table<2, f64>, !md.field<@atoms, i32> ->
// CHECK-LABEL: md.function @forces(
// CHECK: %[[L:.*]] = md_exec.nonzero %{{.*}} : !md.field<@atoms, f64> -> !md.field<@atoms, i32>
// CHECK: md_exec.reciprocal %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} grid([8, 8, 8]) order(4)
// CHECK-SAME: listed(%[[L]] : !md.field<@atoms, i32>)
// CHECK-SAME: -> f64, vector<9xf64>, !md.field<@atoms, 3 x f64>
md.function @forces(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %l = md_exec.nonzero %q : !real -> !ids
  %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli listed(%l)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid, !ids -> f64, !vec, vector<9xf64>
  md.return %u, %f : f64, !vec
}

// CHECK-LABEL: md.function @both(
// CHECK: md_exec.reciprocal
// CHECK-SAME: listed(
// CHECK-SAME: -> f64, vector<9xf64>, !md.field<@atoms, 3 x f64>
// CHECK: md_exec.reciprocal
// CHECK-NOT: listed(
// CHECK-SAME: potential
md.function @both(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid,
                  %l: !ids) -> (f64, !vec, !real) {
  %u, %f, %w, %p = md.reciprocal %x, %q, %cell, %moduli listed(%l)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid, !ids -> f64, !vec, vector<9xf64>, !real
  md.return %u, %f, %p : f64, !vec, !real
}
