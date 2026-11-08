// RUN: mdir-opt %s --md-differentiate | FileCheck %s

// The derivative of a potential along directions of its arguments and its
// forces (D389, #726): the pair sum gathers the
// direction of the charges beside them and yields the derivative of its
// kernel along the number and along the values at both ends; a term that
// reads no directed argument has no term; the reciprocal sum takes two sums
// at the charges shifted by their direction and back; no potential along
// directions is left.

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!grid = !md.table<2, f64>
md.particle_set @atoms

md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %q: !real, %k: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.4) : !vec -> !pairs
  %up = md.sum_relation %n, %x, %cell gather(%q : !real) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
    %qq = arith.mulf %qi, %qj : f64
    %c = arith.divf %qq, %r : f64
    %e = arith.mulf %eps, %c : f64
    md.yield %e : f64
  } : !pairs, !vec -> f64
  // Reads k alone, which is not directed.
  %ux = md.sum_particles gather(%x : !vec) {
  ^bb0(%xi: vector<3xf64>):
    %x2 = arith.mulf %xi, %xi : vector<3xf64>
    %s = vector.reduction <add>, %x2 : vector<3xf64> into f64
    %e = arith.mulf %k, %s : f64
    md.yield %e : f64
  } : f64
  %u = arith.addf %up, %ux : f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @lj.direction_2_3.energy_forces
// CHECK-SAME: (%[[X:.*]]: !md.field<@atoms, 3 x f64>, %{{.*}}: !md.cell, %[[EPS:.*]]: f64, %[[Q:.*]]: !md.field<@atoms, f64>, %{{.*}}: f64, %[[DEPS:.*]]: f64, %[[DQ:.*]]: !md.field<@atoms, f64>)
// CHECK: md.sum_relation %{{.*}} gather(%[[Q]], %[[DQ]] : {{.*}}) exchange(symmetric, derived)
// CHECK-NEXT: ^bb0(%{{.*}}: f64, %{{.*}}: vector<3xf64>, %{{.*}}: f64, %{{.*}}: f64, %{{.*}}: f64, %{{.*}}: f64):
// CHECK: arith.mulf %[[DEPS]]
// CHECK: md.gather_relation %{{.*}} gather(%[[Q]], %[[DQ]] : {{.*}}) exchange(antisymmetric, derived)
// CHECK-NOT: md.sum_particles
// CHECK: md.return

md.potential @mesh(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid) -> f64 {
  %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  md.return %u : f64
}

// CHECK-LABEL: md.function @mesh.direction_2.energy_forces
// CHECK: md.map_particles gather
// CHECK: arith.addf
// CHECK: md.reciprocal
// CHECK: md.map_particles gather
// CHECK: arith.subf
// CHECK: md.reciprocal
// CHECK-NOT: md.reciprocal
// CHECK: md.return

// CHECK-NOT: md.potential @lj.direction
// CHECK-NOT: md.potential @mesh.direction

md.function @f(%x: !vec, %cell: !md.cell, %eps: f64, %q: !real, %k: f64,
               %deps: f64, %dq: !real, %moduli: !grid) -> (f64, !vec, f64, !vec) {
  %p, %fp = md.evaluate @lj(%x, %cell, %eps, %q, %k, %deps, %dq)
      request [direction, direction_forces] {directions = array<i64: 2, 3>}
      : (!vec, !md.cell, f64, !real, f64, f64, !real) -> (f64, !vec)
  %m, %fm = md.evaluate @mesh(%x, %cell, %q, %moduli, %dq)
      request [direction, direction_forces] {directions = array<i64: 2>}
      : (!vec, !md.cell, !real, !grid, !real) -> (f64, !vec)
  md.return %p, %fp, %m, %fm : f64, !vec, f64, !vec
}
