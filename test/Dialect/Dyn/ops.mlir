// RUN: mdir-opt %s | mdir-opt | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.5) : !vec -> !pairs
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
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// CHECK-LABEL: dyn.program @velocity_verlet(
// CHECK-SAME:    -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>)
// CHECK-SAME:    attributes {provides = ["symplectic", "time_reversible"]}
dyn.program @velocity_verlet(%x: !vec, %v: !vec, %f: !vec, %m: !real,
                             %cell: !md.cell, %dt: f64,
                             %eps: f64, %sigma: f64) -> (!vec, !vec, !vec)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %c    = arith.constant 0.5 : f64
  %half = arith.mulf %c, %dt : f64
  // CHECK: %[[V1:[0-9]+]] = dyn.kick %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %[[HALF:[0-9]+]] : !md.field<@atoms, 3 x f64>
  %v1 = dyn.kick %v, %f, %m, %half : !vec
  // CHECK: %[[X1:[0-9]+]] = dyn.drift %{{[a-z0-9]+}}, %[[V1]], %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>
  %x1 = dyn.drift %x, %v1, %dt : !vec
  // CHECK: %[[F1:[0-9]+]] = md.evaluate @lj(%[[X1]],
  %f1 = md.evaluate @lj(%x1, %cell, %eps, %sigma) request [forces]
      : (!vec, !md.cell, f64, f64) -> !vec
  // CHECK: %[[V2:[0-9]+]] = dyn.kick %[[V1]], %[[F1]], %{{[a-z0-9]+}}, %[[HALF]]
  %v2 = dyn.kick %v1, %f1, %m, %half : !vec
  // CHECK: dyn.return %[[X1]], %[[V2]], %[[F1]]
  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec
}

// CHECK-LABEL: dyn.program @leapfrog(
// CHECK-SAME:    attributes {provides = ["symplectic", "time_reversible"], velocity_offset = -5.000000e-01 : f64}
dyn.program @leapfrog(%x: !vec, %v: !vec, %m: !real, %cell: !md.cell,
                      %dt: f64, %eps: f64, %sigma: f64) -> (!vec, !vec)
    attributes {velocity_offset = -0.5,
                provides = ["symplectic", "time_reversible"]} {
  %f = md.evaluate @lj(%x, %cell, %eps, %sigma) request [forces]
      : (!vec, !md.cell, f64, f64) -> !vec
  %v1 = dyn.kick %v, %f, %m, %dt : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  dyn.return %x1, %v1 : !vec, !vec
}

// The step loop carries the state as loop-carried values.
//
// CHECK-LABEL: func.func @run_segment(
func.func @run_segment(%x0: !vec, %v0: !vec, %f0: !vec, %m: !real,
                       %cell: !md.cell, %dt: f64, %eps: f64, %sigma: f64,
                       %n: index) -> (!vec, !vec, !vec) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.for
  // CHECK-SAME: iter_args(%[[XA:[a-z0-9]+]] = %{{[a-z0-9]+}}, %[[VA:[a-z0-9]+]] = %{{[a-z0-9]+}}, %[[FA:[a-z0-9]+]] = %{{[a-z0-9]+}})
  %x, %v, %f = scf.for %s = %c0 to %n step %c1
      iter_args(%xa = %x0, %va = %v0, %fa = %f0) -> (!vec, !vec, !vec) {
    // CHECK: %[[STATE:[0-9]+]]:3 = dyn.step @velocity_verlet(%[[XA]], %[[VA]], %[[FA]],
    %xb, %vb, %fb = dyn.step @velocity_verlet(
        %xa, %va, %fa, %m, %cell, %dt, %eps, %sigma)
        : (!vec, !vec, !vec, !real, !md.cell, f64, f64, f64)
        -> (!vec, !vec, !vec)
    // CHECK: scf.yield %[[STATE]]#0, %[[STATE]]#1, %[[STATE]]#2
    scf.yield %xb, %vb, %fb : !vec, !vec, !vec
  }
  return %x, %v, %f : !vec, !vec, !vec
}
