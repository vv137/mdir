// RUN: mdir-opt %s --md-differentiate | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @square(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  %s = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.mulf %a, %s : f64
  md.return %u : f64
}

// Energy and forces. With u = r², K(i, j) = −a · (r + r) / r · d.
//
// CHECK-LABEL: md.function @square.energy_forces(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell, %[[A:[a-z0-9]+]]: f64)
// CHECK-SAME:    -> (f64, !md.field<@atoms, 3 x f64>)
// CHECK:         %[[N:[0-9]+]] = md.neighborhood %[[X]], %[[CELL]]
// CHECK:         %[[S:[0-9]+]] = md.sum_relation %[[N]], %[[X]], %[[CELL]] exchange(symmetric)
// CHECK:         %[[U:[0-9]+]] = arith.mulf %[[A]], %[[S]]
// CHECK:         %[[F:[0-9]+]] = md.gather_relation %[[N]], %[[X]], %[[CELL]] exchange(antisymmetric, derived) {
// CHECK-NEXT:    ^bb0(%[[R:[a-z0-9]+]]: f64, %[[D:[a-z0-9]+]]: vector<3xf64>):
// CHECK-NEXT:      %[[SLOPE:[0-9]+]] = arith.addf %[[R]], %[[R]]
// CHECK-NEXT:      %[[WEIGHTED:[0-9]+]] = arith.mulf %[[A]], %[[SLOPE]]
// CHECK-NEXT:      %[[RATIO:[0-9]+]] = arith.divf %[[WEIGHTED]], %[[R]]
// CHECK-NEXT:      %[[FACTOR:[0-9]+]] = arith.negf %[[RATIO]]
// CHECK-NEXT:      %[[BROADCAST:[0-9]+]] = vector.broadcast %[[FACTOR]] : f64 to vector<3xf64>
// CHECK-NEXT:      %[[K:[0-9]+]] = arith.mulf %[[BROADCAST]], %[[D]]
// CHECK-NEXT:      md.yield %[[K]] : vector<3xf64>
// CHECK:         md.return %[[U]], %[[F]]

// The energy is not requested, so its sum is not computed.
//
// CHECK-LABEL: md.function @square.forces(
// CHECK-NOT:     md.sum_relation
// CHECK:         md.gather_relation
// CHECK-NOT:     md.sum_relation
// CHECK:         md.return

// W = Σ d ⊗ K(i, j)
//
// CHECK-LABEL: md.function @square.virial(
// CHECK:         %[[W:[0-9]+]] = md.sum_relation %{{.*}} exchange(symmetric, derived)
// CHECK:           vector.extract %{{[a-z0-9]+}}[0]
// CHECK:           vector.extract %{{[a-z0-9]+}}[1]
// CHECK:           vector.extract %{{[a-z0-9]+}}[2]
// CHECK:           %[[OUTER:[0-9]+]] = vector.from_elements
// CHECK:           md.yield %[[OUTER]] : vector<9xf64>
// CHECK:         md.return %[[W]] : vector<9xf64>

// ∂U/∂a = Σ r²: the sum itself.
//
// CHECK-LABEL: md.function @square.derivative2(
// CHECK:         %[[S:[0-9]+]] = md.sum_relation
// CHECK-NOT:     md.sum_relation
// CHECK:         md.return %[[S]] : f64

// CHECK-LABEL: md.function @caller(
md.function @caller(%x: !vec, %cell: !md.cell, %a: f64)
    -> (f64, !vec, !vec, vector<9xf64>, f64) {
  // CHECK: %{{[0-9]+}}:2 = md.call @square.energy_forces(%{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}})
  // CHECK-SAME: : (!md.field<@atoms, 3 x f64>, !md.cell, f64) -> (f64, !md.field<@atoms, 3 x f64>)
  %u, %f = md.evaluate @square(%x, %cell, %a) request [energy, forces]
      : (!vec, !md.cell, f64) -> (f64, !vec)
  // CHECK: md.call @square.forces(
  %g = md.evaluate @square(%x, %cell, %a) request [forces]
      : (!vec, !md.cell, f64) -> !vec
  // CHECK: md.call @square.virial(
  %w = md.evaluate @square(%x, %cell, %a) request [virial]
      : (!vec, !md.cell, f64) -> vector<9xf64>
  // CHECK: md.call @square.derivative2(
  %da = md.evaluate @square(%x, %cell, %a) request [derivative(2)]
      : (!vec, !md.cell, f64) -> f64
  // CHECK-NOT: md.evaluate
  md.return %u, %f, %g, %w, %da : f64, !vec, !vec, vector<9xf64>, f64
}

// The same request twice uses one generated function.
//
// CHECK-LABEL: md.function @twice(
md.function @twice(%x: !vec, %cell: !md.cell, %a: f64) -> (!vec, !vec) {
  // CHECK: md.call @square.forces(
  // CHECK: md.call @square.forces(
  %f = md.evaluate @square(%x, %cell, %a) request [forces]
      : (!vec, !md.cell, f64) -> !vec
  %g = md.evaluate @square(%x, %cell, %a) request [forces]
      : (!vec, !md.cell, f64) -> !vec
  md.return %f, %g : !vec, !vec
}

// Two sums, one with per-particle parameters. The forces of the two are
// added particle by particle.
md.potential @two(%x: !vec, %cell: !md.cell, %q: !real) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0) : !vec -> !pairs
  %u1 = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u2 = md.sum_relation %n, %x, %cell gather(%q : !real) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %q1: f64, %q2: f64):
    %qq = arith.mulf %q1, %q2 : f64
    %k = arith.divf %qq, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %u1, %u2 : f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @two.forces(
// CHECK:         %[[F1:[0-9]+]] = md.gather_relation %{{[0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} exchange(antisymmetric, derived)
// CHECK:         %[[F2:[0-9]+]] = md.gather_relation %{{[0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} gather(%{{[a-z0-9]+}} : !md.field<@atoms, f64>) exchange(antisymmetric, derived)
// CHECK:         %[[F:[0-9]+]] = md.map_particles gather(%[[F1]], %[[F2]] :
// CHECK-NEXT:    ^bb0(%[[A:[a-z0-9]+]]: vector<3xf64>, %[[B:[a-z0-9]+]]: vector<3xf64>):
// CHECK-NEXT:      %[[SUM:[0-9]+]] = arith.addf %[[A]], %[[B]]
// CHECK-NEXT:      md.yield %[[SUM]]
// CHECK:         md.return %[[F]]

md.function @caller_two(%x: !vec, %cell: !md.cell, %q: !real) -> !vec {
  %f = md.evaluate @two(%x, %cell, %q) request [forces]
      : (!vec, !md.cell, !real) -> !vec
  md.return %f : !vec
}

// Truncation is expanded in the generated function. The potential keeps its
// attribute.
//
// CHECK-LABEL: md.potential @truncated(
// CHECK:         truncation(shift)
md.potential @truncated(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) truncation(shift) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @truncated.energy(
// CHECK-NOT:     truncation
// CHECK:         arith.constant 1.500000e+00
// CHECK:         md.return

md.function @caller_truncated(%x: !vec, %cell: !md.cell) -> f64 {
  %u = md.evaluate @truncated(%x, %cell) request [energy]
      : (!vec, !md.cell) -> f64
  md.return %u : f64
}
