// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.25 width=48" | FileCheck %s
// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.25 width=48 cells=1" \
// RUN: | FileCheck %s --check-prefix=WIDE

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

// A sum over an unordered relation becomes a loop over directed pairs with
// the weight one half. The kernel is written in terms of the distance; the
// loop provides the squared distance.
//
// CHECK-LABEL: md.function @energy(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell, %[[A:[a-z0-9]+]]: f64)
md.function @energy(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  // The cells are a third as wide as the cutoff plus the skin, or wider.
  //
  // CHECK:      %[[CELLS:[0-9]+]] = md_exec.build_cells %[[X]], %[[CELL]] width(0.58333333333333337)
  // WIDE:       md_exec.build_cells %{{[a-z0-9]+}}, %{{[a-z0-9]+}} width(1.750000e+00)
  // CHECK:      %[[NL:[0-9]+]] = md_exec.build_neighbors %[[CELLS]], %[[X]], %[[CELL]]
  // CHECK-SAME:   cutoff(1.500000e+00) skin(2.500000e-01) kind(matrix) width(48)
  // CHECK-NOT:  md.neighborhood
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs

  // CHECK:      %[[ZERO:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f64
  // CHECK:      %[[U:[0-9]+]] = md_exec.pair_for %[[NL]], %[[X]], %[[CELL]]
  // CHECK-SAME:   reduce(%[[ZERO]] : f64) cutoff(1.500000e+00) weights [5.000000e-01]
  // CHECK-SAME:   exchange [symmetric] policy(directed, owner_only) {
  // CHECK-NEXT: ^bb0(%[[R2:[a-z0-9]+]]: f64, %{{[a-z0-9]+}}: vector<3xf64>):
  // CHECK-NEXT:   %[[R:[0-9]+]] = math.sqrt %[[R2]] : f64
  // CHECK-NEXT:   %[[K:[0-9]+]] = arith.divf %[[A]], %[[R]] : f64
  // CHECK-NEXT:   md_exec.yield %[[K]] : f64
  // CHECK-NEXT: } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> f64
  // CHECK-NOT:  md.sum_relation
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.divf %a, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64

  // CHECK:      md.return %[[U]] : f64
  md.return %u : f64
}

// A gather accumulates from zero. One neighbor structure serves both loops.
// Each loop carries the exchange contract of its kernel.
//
// CHECK-LABEL: md.function @energy_forces(
md.function @energy_forces(%x: !vec, %cell: !md.cell, %q: !real)
    -> (f64, !vec) {
  // CHECK:      %[[NL:[0-9]+]] = md_exec.build_neighbors
  // CHECK-NOT:  md_exec.build_neighbors
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs

  // CHECK:      %[[U:[0-9]+]] = md_exec.pair_for %[[NL]],
  // CHECK-SAME:   ins(%{{[a-z0-9]+}} : !md.field<@atoms, f64>)
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64):
  %u = md.sum_relation %n, %x, %cell gather(%q : !real) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %q1: f64, %q2: f64):
    %qq = arith.mulf %q1, %q2 : f64
    %k = arith.divf %qq, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64

  // The kernel does not use the distance, so no square root is taken.
  //
  // CHECK:      %[[F0:[0-9]+]] = md_exec.zeros : !md.field<@atoms, 3 x f64>
  // CHECK:      %[[F:[0-9]+]] = md_exec.pair_for %[[NL]],
  // CHECK-SAME:   outs(%[[F0]] : !md.field<@atoms, 3 x f64>) cutoff(1.500000e+00)
  // CHECK-NOT:    weights
  // CHECK-SAME:   exchange [antisymmetric] policy(directed, owner_only) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %[[D:[a-z0-9]+]]: vector<3xf64>):
  // CHECK-NEXT:   md_exec.yield %[[D]] : vector<3xf64>
  %f = md.gather_relation %n, %x, %cell exchange(antisymmetric, derived) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %d : vector<3xf64>
  } : !pairs, !vec -> !vec

  // CHECK:      md.return %[[U]], %[[F]]
  md.return %u, %f : f64, !vec
}

// CHECK-LABEL: md.function @particles(
md.function @particles(%v: !vec, %f1: !vec, %f2: !vec) -> (f64, !vec) {
  // CHECK:      %[[ZERO:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f64
  // CHECK:      %[[S:[0-9]+]] = md_exec.particle_for ins(%{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>) reduce(%[[ZERO]] : f64) {
  // CHECK:      } -> f64
  %s = md.sum_particles gather(%v : !vec) {
  ^bb0(%v_i: vector<3xf64>):
    %sq = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    md.yield %v2 : f64
  } : f64

  // CHECK:      %[[EMPTY:[0-9]+]] = md_exec.empty : !md.field<@atoms, 3 x f64>
  // CHECK:      %[[F:[0-9]+]] = md_exec.particle_for ins(%{{[a-z0-9]+}}, %{{[a-z0-9]+}} :
  // CHECK-SAME:   outs(%[[EMPTY]] : !md.field<@atoms, 3 x f64>) {
  // CHECK:      } -> !md.field<@atoms, 3 x f64>
  %f = md.map_particles gather(%f1, %f2 : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %t = arith.addf %a, %b : vector<3xf64>
    md.yield %t : vector<3xf64>
  } : !vec

  // CHECK:      md.return %[[S]], %[[F]]
  md.return %s, %f : f64, !vec
}

// v' = v + dt · f / m, then x' = x + dt · v'.
//
// CHECK-LABEL: dyn.program @leapfrog(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[V:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>,
// CHECK-SAME:    %[[F:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[M:[a-z0-9]+]]: !md.field<@atoms, f64>, %[[DT:[a-z0-9]+]]: f64)
dyn.program @leapfrog(%x: !vec, %v: !vec, %f: !vec, %m: !real, %dt: f64)
    -> (!vec, !vec) attributes {velocity_offset = -0.5} {
  // CHECK:      %[[V1:[0-9]+]] = md_exec.particle_for ins(%[[V]], %[[F]], %[[M]] :
  // CHECK-NEXT: ^bb0(%[[VI:[a-z0-9]+]]: vector<3xf64>, %[[FI:[a-z0-9]+]]: vector<3xf64>, %[[MI:[a-z0-9]+]]: f64):
  // A particle of mass 0 is not kicked.
  // CHECK-NEXT:   %[[ZERO:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f64
  // CHECK-NEXT:   %[[MASSLESS:[0-9]+]] = arith.cmpf oeq, %[[MI]], %[[ZERO]] : f64
  // CHECK-NEXT:   %[[QUOTIENT:[0-9]+]] = arith.divf %[[DT]], %[[MI]] : f64
  // CHECK-NEXT:   %[[FACTOR:[0-9]+]] = arith.select %[[MASSLESS]], %[[ZERO]], %[[QUOTIENT]] : f64
  // CHECK-NEXT:   %[[FV:[0-9]+]] = vector.broadcast %[[FACTOR]] : f64 to vector<3xf64>
  // CHECK-NEXT:   %[[DV:[0-9]+]] = arith.mulf %[[FV]], %[[FI]] : vector<3xf64>
  // CHECK-NEXT:   %[[VN:[0-9]+]] = arith.addf %[[VI]], %[[DV]] : vector<3xf64>
  // CHECK-NEXT:   md_exec.yield %[[VN]] : vector<3xf64>
  // CHECK-NOT:  dyn.kick
  %v1 = dyn.kick %v, %f, %m, %dt : !vec

  // CHECK:      %[[X1:[0-9]+]] = md_exec.particle_for ins(%[[X]], %[[V1]] :
  // CHECK-NEXT: ^bb0(%[[XI:[a-z0-9]+]]: vector<3xf64>, %[[VI:[a-z0-9]+]]: vector<3xf64>):
  // CHECK-NEXT:   %[[DTV:[0-9]+]] = vector.broadcast %[[DT]] : f64 to vector<3xf64>
  // CHECK-NEXT:   %[[DX:[0-9]+]] = arith.mulf %[[DTV]], %[[VI]] : vector<3xf64>
  // CHECK-NEXT:   %[[XN:[0-9]+]] = arith.addf %[[XI]], %[[DX]] : vector<3xf64>
  // CHECK-NEXT:   md_exec.yield %[[XN]] : vector<3xf64>
  // CHECK-NOT:  dyn.drift
  %x1 = dyn.drift %x, %v1, %dt : !vec

  // CHECK:      dyn.return %[[X1]], %[[V1]]
  dyn.return %x1, %v1 : !vec, !vec
}
