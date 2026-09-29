// RUN: mdir-opt %s --md-exec-fuse-loops | FileCheck %s

!vec  = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!ions = !md.field<@ions, 3 x f64>
!nl   = !mdrt.neighbors<@atoms>

md.particle_set @atoms
md.particle_set @ions

// A kick and the drift that follows it become one loop. The drift reads the
// velocity that the kick has yielded for the particle.
//
// CHECK-LABEL: func.func @kick_drift(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[V:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[F:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[M:[a-z0-9]+]]: !md.field<@atoms, f64>, %[[DT:[a-z0-9]+]]: f64)
func.func @kick_drift(%x: !vec, %v: !vec, %f: !vec, %m: !real, %dt: f64)
    -> (!vec, !vec) {
  // CHECK:      %[[E0:[0-9]+]] = md_exec.empty
  // CHECK:      %[[E1:[0-9]+]] = md_exec.empty
  // CHECK:      %[[LOOP:[0-9]+]]:2 = md_exec.particle_for
  // CHECK-SAME:   ins(%[[V]], %[[F]], %[[M]], %[[X]] :
  // CHECK-SAME:   outs(%[[E0]], %[[E1]] :
  // CHECK-NEXT: ^bb0(%[[VI:[a-z0-9]+]]: vector<3xf64>, %[[FI:[a-z0-9]+]]: vector<3xf64>, %[[MI:[a-z0-9]+]]: f64, %[[XI:[a-z0-9]+]]: vector<3xf64>):
  // CHECK:        %[[VN:[0-9]+]] = arith.addf %[[VI]], %{{[0-9]+}}
  // CHECK:        %[[DX:[0-9]+]] = arith.mulf %{{[0-9]+}}, %[[VN]]
  // CHECK:        %[[XN:[0-9]+]] = arith.addf %[[XI]], %[[DX]]
  // CHECK:        md_exec.yield %[[VN]], %[[XN]] : vector<3xf64>, vector<3xf64>
  // CHECK-NOT:  md_exec.particle_for
  // CHECK:      return %[[LOOP]]#1, %[[LOOP]]#0
  %e0 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f, %m : !vec, !vec, !real)
      outs(%e0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
    %s  = arith.divf %dt, %m_i : f64
    %sv = vector.broadcast %s : f64 to vector<3xf64>
    %dv = arith.mulf %sv, %f_i : vector<3xf64>
    %vn = arith.addf %v_i, %dv : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec

  %e1 = md_exec.empty : !vec
  %x1 = md_exec.particle_for ins(%x, %v1 : !vec, !vec) outs(%e1 : !vec) {
  ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
    %tv = vector.broadcast %dt : f64 to vector<3xf64>
    %dx = arith.mulf %tv, %v_i : vector<3xf64>
    %xn = arith.addf %x_i, %dx : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec
  return %x1, %v1 : !vec, !vec
}

// A field that only the second loop reads is not stored.
//
// CHECK-LABEL: func.func @not_stored(
func.func @not_stored(%x: !vec, %v: !vec, %dt: f64) -> !vec {
  // CHECK:      %[[LOOP:[0-9]+]] = md_exec.particle_for
  // CHECK-SAME:   ins(%{{[a-z0-9]+}}, %{{[a-z0-9]+}} :
  // CHECK-SAME:   outs(%{{[0-9]+}} : !md.field<@atoms, 3 x f64>)
  // CHECK:        md_exec.yield %{{[0-9]+}} : vector<3xf64>
  // CHECK-NOT:  md_exec.particle_for
  // CHECK:      return %[[LOOP]]
  %e0 = md_exec.empty : !vec
  %dx = md_exec.particle_for ins(%v : !vec) outs(%e0 : !vec) {
  ^bb0(%v_i: vector<3xf64>):
    %tv = vector.broadcast %dt : f64 to vector<3xf64>
    %d  = arith.mulf %tv, %v_i : vector<3xf64>
    md_exec.yield %d : vector<3xf64>
  } -> !vec

  %e1 = md_exec.empty : !vec
  %x1 = md_exec.particle_for ins(%x, %dx : !vec, !vec) outs(%e1 : !vec) {
  ^bb0(%x_i: vector<3xf64>, %d_i: vector<3xf64>):
    %xn = arith.addf %x_i, %d_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec
  return %x1 : !vec
}

// A kick and the kinetic energy of the new velocities. The fields come
// first among the results, then the global sums.
//
// CHECK-LABEL: func.func @kick_energy(
func.func @kick_energy(%v: !vec, %f: !vec, %m: !real) -> (!vec, f64) {
  // CHECK:      %[[LOOP:[0-9]+]]:2 = md_exec.particle_for
  // CHECK-SAME:   ins(%{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} :
  // CHECK-SAME:   outs(%{{[0-9]+}} : !md.field<@atoms, 3 x f64>)
  // CHECK-SAME:   reduce(%{{[a-z0-9_]+}} : f64)
  // CHECK:        %[[VN:[0-9]+]] = arith.addf
  // CHECK:        %[[SQ:[0-9]+]] = arith.mulf %[[VN]], %[[VN]]
  // CHECK:        md_exec.yield %[[VN]], %{{[0-9]+}} : vector<3xf64>, f64
  // CHECK-NOT:  md_exec.particle_for
  // CHECK:      return %[[LOOP]]#0, %[[LOOP]]#1
  %e0 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f : !vec, !vec) outs(%e0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>):
    %vn = arith.addf %v_i, %f_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec

  %k0 = arith.constant 0.0 : f64
  %ke = md_exec.particle_for ins(%v1, %m : !vec, !real) reduce(%k0 : f64) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %sq = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %k  = arith.mulf %m_i, %v2 : f64
    md_exec.yield %k : f64
  } -> f64
  return %v1, %ke : !vec, f64
}

// Loops that have nothing to do with one another are fused as well: they
// run over the same particles.
//
// CHECK-LABEL: func.func @independent(
// CHECK:         md_exec.particle_for
// CHECK-NOT:     md_exec.particle_for
// CHECK:         return
func.func @independent(%x: !vec, %v: !vec) -> (!vec, !vec) {
  %e0 = md_exec.empty : !vec
  %x1 = md_exec.particle_for ins(%x : !vec) outs(%e0 : !vec) {
  ^bb0(%x_i: vector<3xf64>):
    %xn = arith.addf %x_i, %x_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec
  %e1 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v : !vec) outs(%e1 : !vec) {
  ^bb0(%v_i: vector<3xf64>):
    %vn = arith.addf %v_i, %v_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  return %x1, %v1 : !vec, !vec
}

// The second loop scales by a global sum of the first. The sum is complete
// only when the first loop is, so the two stay apart.
//
// CHECK-LABEL: func.func @uses_sum(
// CHECK:         md_exec.particle_for
// CHECK:         md_exec.particle_for
func.func @uses_sum(%v: !vec) -> !vec {
  %k0 = arith.constant 0.0 : f64
  %ke = md_exec.particle_for ins(%v : !vec) reduce(%k0 : f64) {
  ^bb0(%v_i: vector<3xf64>):
    %sq = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    md_exec.yield %v2 : f64
  } -> f64

  %e0 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v : !vec) outs(%e0 : !vec) {
  ^bb0(%v_i: vector<3xf64>):
    %s  = vector.broadcast %ke : f64 to vector<3xf64>
    %vn = arith.mulf %s, %v_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  return %v1 : !vec
}

// The forces between the drift and the kick need the positions of all
// particles, and the kick needs the forces. Neither loop can move to the
// other.
//
// CHECK-LABEL: func.func @forces_between(
// CHECK:         md_exec.particle_for
// CHECK:         md_exec.pair_for
// CHECK:         md_exec.particle_for
func.func @forces_between(%x: !vec, %v: !vec, %cell: !md.cell, %nl: !nl)
    -> (!vec, !vec) {
  %e0 = md_exec.empty : !vec
  %x1 = md_exec.particle_for ins(%x, %v : !vec, !vec) outs(%e0 : !vec) {
  ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
    %xn = arith.addf %x_i, %v_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec

  %f0 = md_exec.zeros : !vec
  %f = md_exec.pair_for %nl, %x1, %cell outs(%f0 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !nl, !vec -> !vec

  %e1 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f : !vec, !vec) outs(%e1 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>):
    %vn = arith.addf %v_i, %f_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  return %x1, %v1 : !vec, !vec
}

// Loops over the particles of different sets stay apart.
//
// CHECK-LABEL: func.func @different_sets(
// CHECK:         md_exec.particle_for
// CHECK:         md_exec.particle_for
func.func @different_sets(%x: !vec, %y: !ions) -> (!vec, !ions) {
  %e0 = md_exec.empty : !vec
  %x1 = md_exec.particle_for ins(%x : !vec) outs(%e0 : !vec) {
  ^bb0(%x_i: vector<3xf64>):
    %xn = arith.addf %x_i, %x_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec
  %e1 = md_exec.empty : !ions
  %y1 = md_exec.particle_for ins(%y : !ions) outs(%e1 : !ions) {
  ^bb0(%y_i: vector<3xf64>):
    %yn = arith.addf %y_i, %y_i : vector<3xf64>
    md_exec.yield %yn : vector<3xf64>
  } -> !ions
  return %x1, %y1 : !vec, !ions
}

// Loops in the storage form are left alone.
//
// CHECK-LABEL: func.func @storage(
// CHECK:         md_exec.particle_for
// CHECK:         md_exec.particle_for
func.func @storage(%x: memref<?x3xf64>, %v: memref<?x3xf64>) {
  md_exec.particle_for ins(%v : memref<?x3xf64>) outs(%v : memref<?x3xf64>) {
  ^bb0(%v_i: vector<3xf64>):
    %vn = arith.addf %v_i, %v_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  }
  md_exec.particle_for ins(%x, %v : memref<?x3xf64>, memref<?x3xf64>)
      outs(%x : memref<?x3xf64>) {
  ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
    %xn = arith.addf %x_i, %v_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  }
  return
}
