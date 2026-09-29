// RUN: mdir-opt %s | mdir-opt | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

// CHECK: md.particle_set @atoms
md.particle_set @atoms

// The Lennard-Jones potential with a switching function.
//
// CHECK-LABEL: md.potential @lj(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell,
// CHECK-SAME:    %[[EPS:[a-z0-9]+]]: f64, %[[SIGMA:[a-z0-9]+]]: f64) -> f64
md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  // CHECK: %[[N:[a-z0-9]+]] = md.neighborhood %[[X]], %[[CELL]] cutoff(2.500000e+00)
  // CHECK-SAME: : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  %n = md.neighborhood %x, %cell cutoff(2.5) : !vec -> !pairs

  // CHECK: %[[U:[a-z0-9]+]] = md.sum_relation %[[N]], %[[X]], %[[CELL]]
  // CHECK-SAME: exchange(symmetric) truncation(switch, from = 2.000000e+00) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>):
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(switch, from = 2.0) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    // CHECK: md.yield %{{.*}} : f64
    md.yield %k : f64
  // CHECK: } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  } : !pairs, !vec -> f64

  // CHECK: md.return %[[U]] : f64
  md.return %u : f64
}

// Per-particle parameters with combining rules.
//
// CHECK-LABEL: md.potential @lj_mixed(
md.potential @lj_mixed(%x: !vec, %cell: !md.cell,
                       %eps: !real, %sigma: !real) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0) : !vec -> !pairs

  // CHECK: md.sum_relation
  // CHECK-SAME: gather(%[[EPSF:[a-z0-9]+]], %[[SIGF:[a-z0-9]+]] : !md.field<@atoms, f64>, !md.field<@atoms, f64>)
  // CHECK-SAME: exchange(symmetric, asserted) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64):
  %u = md.sum_relation %n, %x, %cell gather(%eps, %sigma : !real, !real)
         exchange(symmetric, asserted) {
  ^bb0(%r: f64, %d: vector<3xf64>,
       %eps1: f64, %eps2: f64, %sigma1: f64, %sigma2: f64):
    %half = arith.constant 0.5 : f64
    %c4   = arith.constant 4.0 : f64
    %i6   = arith.constant 6 : i32
    %ssum = arith.addf %sigma1, %sigma2 : f64
    %s    = arith.mulf %half, %ssum : f64
    %eprd = arith.mulf %eps1, %eps2 : f64
    %e    = math.sqrt %eprd : f64
    %sr   = arith.divf %s, %r : f64
    %s6   = math.fpowi %sr, %i6 : f64, i32
    %s12  = arith.mulf %s6, %s6 : f64
    %t    = arith.subf %s12, %s6 : f64
    %e4   = arith.mulf %c4, %e : f64
    %k    = arith.mulf %e4, %t : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// The shape of a derivative program: energy and forces.
//
// CHECK-LABEL: md.function @lj.energy_forces(
// CHECK-SAME:    -> (f64, !md.field<@atoms, 3 x f64>)
md.function @lj.energy_forces(%x: !vec, %cell: !md.cell,
                              %eps: f64, %sigma: f64) -> (f64, !vec) {
  %n = md.neighborhood %x, %cell cutoff(2.5) : !vec -> !pairs

  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64

  // K(i, j) = (24 eps / r^2) (2 (sigma/r)^12 - (sigma/r)^6) d_ij
  //
  // CHECK: %[[F:[a-z0-9]+]] = md.gather_relation
  // CHECK-SAME: exchange(antisymmetric, derived) {
  // CHECK: md.yield %{{.*}} : vector<3xf64>
  // CHECK: } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f64>
  %f = md.gather_relation %n, %x, %cell exchange(antisymmetric, derived) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c24 = arith.constant 24.0 : f64
    %c2  = arith.constant 2.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a   = arith.mulf %c2, %s12 : f64
    %b   = arith.subf %a, %s6 : f64
    %e24 = arith.mulf %c24, %eps : f64
    %num = arith.mulf %e24, %b : f64
    %r2  = arith.mulf %r, %r : f64
    %g   = arith.divf %num, %r2 : f64
    %gv  = vector.broadcast %g : f64 to vector<3xf64>
    %k   = arith.mulf %gv, %d : vector<3xf64>
    md.yield %k : vector<3xf64>
  } : !pairs, !vec -> !vec

  // CHECK: md.return %{{[a-z0-9]+}}, %[[F]] : f64, !md.field<@atoms, 3 x f64>
  md.return %u, %f : f64, !vec
}

// Field types.
//
// CHECK-LABEL: md.function @fields(
// CHECK-SAME: !md.field<@atoms, f64>
// CHECK-SAME: !md.field<@atoms, i32>
// CHECK-SAME: !md.field<@atoms, i64>
// CHECK-SAME: !md.relation<@atoms, 2, ordered>
md.function @fields(%a: !md.field<@atoms, f64>, %b: !md.field<@atoms, i32>,
                    %c: !md.field<@atoms, i64>,
                    %e: !md.relation<@atoms, 2, ordered>) {
  md.return
}

// Sums and maps over particles.
//
// CHECK-LABEL: md.function @particles(
md.function @particles(%v: !vec, %m: !real, %f1: !vec, %f2: !vec)
    -> (f64, !vec) {
  // CHECK: %[[KE:[a-z0-9]+]] = md.sum_particles gather(%{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64):
  // CHECK: } : f64
  %ke = md.sum_particles gather(%v, %m : !vec, !real) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %half = arith.constant 0.5 : f64
    %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
    %mv2  = arith.mulf %m_i, %v2 : f64
    %k    = arith.mulf %half, %mv2 : f64
    md.yield %k : f64
  } : f64

  // CHECK: %[[F:[a-z0-9]+]] = md.map_particles gather(%{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>) {
  // CHECK: } : !md.field<@atoms, 3 x f64>
  %f = md.map_particles gather(%f1, %f2 : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    md.yield %s : vector<3xf64>
  } : !vec

  // CHECK: md.return %[[KE]], %[[F]]
  md.return %ke, %f : f64, !vec
}

// Evaluation requests and calls.
//
// CHECK-LABEL: md.function @requests(
md.function @requests(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64)
    -> (f64, !vec, vector<9xf64>, f64, f64) {
  // CHECK: md.evaluate @lj(%{{.*}}) request [energy, forces, virial, derivative(3)]
  // CHECK-SAME: : (!md.field<@atoms, 3 x f64>, !md.cell, f64, f64) -> (f64, !md.field<@atoms, 3 x f64>, vector<9xf64>, f64)
  %u, %f, %w, %du = md.evaluate @lj(%x, %cell, %eps, %sigma)
      request [energy, forces, virial, derivative(3)]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec, vector<9xf64>, f64)

  // CHECK: md.call @lj(%{{.*}}) : (!md.field<@atoms, 3 x f64>, !md.cell, f64, f64) -> f64
  %e = md.call @lj(%x, %cell, %eps, %sigma)
      : (!vec, !md.cell, f64, f64) -> f64

  md.return %u, %f, %w, %du, %e : f64, !vec, vector<9xf64>, f64, f64
}
