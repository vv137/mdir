// RUN: mdir-opt %s --md-differentiate | FileCheck %s

// Force evaluation inside a program is replaced like any other.

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @square(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @square.forces(
// CHECK:         md.gather_relation

// CHECK-LABEL: dyn.program @leapfrog(
// CHECK:         %[[F:[0-9]+]] = md.call @square.forces(%{{[a-z0-9]+}}, %{{[a-z0-9]+}})
// CHECK-NOT:     md.evaluate
// CHECK:         dyn.kick %{{[a-z0-9]+}}, %[[F]],
dyn.program @leapfrog(%x: !vec, %v: !vec, %m: !real, %cell: !md.cell,
                      %dt: f64) -> (!vec, !vec)
    attributes {velocity_offset = -0.5} {
  %f = md.evaluate @square(%x, %cell) request [forces]
      : (!vec, !md.cell) -> !vec
  %v1 = dyn.kick %v, %f, %m, %dt : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  dyn.return %x1, %v1 : !vec, !vec
}
