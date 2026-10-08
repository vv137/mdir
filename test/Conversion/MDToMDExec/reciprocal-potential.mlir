// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.25 width=48" | FileCheck %s

// The fourth result of md.reciprocal, the potential of the grid at the
// particles (D[tunable-gradient-pme]): a sum of md_exec that yields it in
// place of the forces. Where the forces are not taken, that sum gives the
// energy and the virial too; where they are, the sum of the forces gives
// them and the two sums stand side by side.

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!grid = !md.table<2, f64>
md.particle_set @atoms

// CHECK-LABEL: md.function @alone(
// CHECK: %[[U:.*]], %[[W:.*]], %[[P:.*]] = md_exec.reciprocal %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} grid([8, 8, 8]) order(4)
// CHECK-SAME: potential
// CHECK-SAME: -> f64, vector<9xf64>, !md.field<@atoms, f64>
// CHECK-NOT: md_exec.reciprocal
// CHECK: md.return %[[U]], %[[P]]
md.function @alone(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !real) {
  %u, %f, %w, %p = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>, !real
  md.return %u, %p : f64, !real
}

// CHECK-LABEL: md.function @both(
// CHECK: %[[U:.*]], %{{.*}}, %[[F:.*]] = md_exec.reciprocal
// CHECK-NOT: potential
// CHECK-SAME: -> f64, vector<9xf64>, !md.field<@atoms, 3 x f64>
// CHECK: %{{.*}}, %{{.*}}, %[[P:.*]] = md_exec.reciprocal
// CHECK-SAME: potential
// CHECK: md.return %[[U]], %[[F]], %[[P]]
md.function @both(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec, !real) {
  %u, %f, %w, %p = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>, !real
  md.return %u, %f, %p : f64, !vec, !real
}
