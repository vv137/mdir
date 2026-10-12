// RUN: mdir-opt %s --md-exec-fuse-loops | FileCheck %s --check-prefix=FUSED
// RUN: mdir-opt %s --md-exec-fuse-loops=integration-runs | FileCheck %s

// The loop over particles that closes a run of loops over disjoint tuples
// and the one that opens the next (D110, D250). A
// device joins a loop over particles, the loops over disjoint tuples that
// read it, and the loop over particles that reads them into one kernel,
// and a loop is the loop after of one run or the loop before of the next,
// not both. With `integration-runs` the two stay apart, so that each run
// has its loops; without it they are one loop, as on the CPU.

!vec = !md.field<@atoms, 3 x f64>
!groups = !md.relation<@atoms, 2, unordered, @groups>
!groups_inc = !mdrt.incidence<@atoms, 2, @groups>

md.particle_set @atoms
md.tuple_set @groups on(@atoms) arity(2) orientation(unordered) disjoint

// A kick, the projection of its velocities, a drift with them, and the
// constraints of its positions: the loop that adds the change of the
// projection closes the first run, and the drift opens the second.
//
// FUSED-LABEL: func.func @two_runs(
// FUSED:         md_exec.particle_for
// FUSED:         md_exec.tuple_for
// FUSED:         md_exec.particle_for
// FUSED:         md_exec.tuple_for
// FUSED:         md_exec.particle_for
// FUSED-NOT:     md_exec.particle_for
//
// CHECK-LABEL: func.func @two_runs(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[V:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[F:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>,
// CHECK:         %[[KICK:[0-9]+]] = md_exec.particle_for ins(%[[V]], %[[F]] :
// CHECK:         %[[DV:[0-9]+]] = md_exec.tuple_for
// CHECK-SAME:      ins(%[[KICK]] :
// CHECK:         %[[PROJECTED:[0-9]+]] = md_exec.particle_for ins(%[[KICK]], %[[DV]] :
// CHECK:         %[[DRIFT:[0-9]+]] = md_exec.particle_for ins(%[[X]], %[[PROJECTED]] :
// CHECK:         %[[DX:[0-9]+]] = md_exec.tuple_for
// CHECK-SAME:      ins(%[[DRIFT]] :
// CHECK:         %[[CONSTRAINED:[0-9]+]] = md_exec.particle_for ins(%[[DRIFT]], %[[DX]] :
// CHECK-NOT:     md_exec.particle_for
// CHECK:         return %[[CONSTRAINED]], %[[PROJECTED]]
func.func @two_runs(%x: !vec, %v: !vec, %f: !vec, %members: memref<?x2xi32>,
                    %cell: !md.cell) -> (!vec, !vec) {
  %pairs = mdrt.from_buffer %members : memref<?x2xi32> to !groups
  %inc = md_exec.build_incidence %pairs : !groups -> !groups_inc
  %e0 = md_exec.empty : !vec
  %v0u = md_exec.particle_for ins(%v, %f : !vec, !vec) outs(%e0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>):
    %vn = arith.addf %v_i, %f_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  %z0 = md_exec.zeros : !vec
  %dv = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(1, 0))
          ins(%v0u : !vec) outs(%z0 : !vec) arity(2) disjoint {
  ^bb0(%r: vector<3xf64>, %a0: vector<3xf64>, %a1: vector<3xf64>):
    %d = arith.subf %a1, %a0 : vector<3xf64>
    %g = arith.mulf %d, %r : vector<3xf64>
    %h = arith.negf %g : vector<3xf64>
    md_exec.yield %g, %h : vector<3xf64>, vector<3xf64>
  } : !groups_inc, !vec -> !vec
  %e1 = md_exec.empty : !vec
  %v0 = md_exec.particle_for ins(%v0u, %dv : !vec, !vec) outs(%e1 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %dv_i: vector<3xf64>):
    %vn = arith.addf %v_i, %dv_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  %e2 = md_exec.empty : !vec
  %x1d = md_exec.particle_for ins(%x, %v0 : !vec, !vec) outs(%e2 : !vec) {
  ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
    %xn = arith.addf %x_i, %v_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec
  %z1 = md_exec.zeros : !vec
  %dx = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(1, 0))
          ins(%x1d : !vec) outs(%z1 : !vec) arity(2) disjoint {
  ^bb0(%r: vector<3xf64>, %a0: vector<3xf64>, %a1: vector<3xf64>):
    %d = arith.subf %a1, %a0 : vector<3xf64>
    %g = arith.mulf %d, %r : vector<3xf64>
    %h = arith.negf %g : vector<3xf64>
    md_exec.yield %g, %h : vector<3xf64>, vector<3xf64>
  } : !groups_inc, !vec -> !vec
  %e3 = md_exec.empty : !vec
  %x1 = md_exec.particle_for ins(%x1d, %dx : !vec, !vec) outs(%e3 : !vec) {
  ^bb0(%x_i: vector<3xf64>, %dx_i: vector<3xf64>):
    %xn = arith.addf %x_i, %dx_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec
  return %x1, %v0 : !vec, !vec
}

// With a third run, here loops over tuples that take the constrained
// positions as the placement of virtual sites does, the loop after of the
// second run would open the third, and the runs that are one kernel would
// only change places: the loops are fused as without the option.
//
// FUSED-LABEL: func.func @three_runs(
// CHECK-LABEL: func.func @three_runs(
// CHECK:         md_exec.particle_for
// CHECK:         md_exec.tuple_for
// CHECK:         md_exec.particle_for
// CHECK-NOT:     md_exec.particle_for
// CHECK:         md_exec.tuple_for
// CHECK:         md_exec.particle_for
// CHECK-NOT:     md_exec.particle_for
// CHECK:         md_exec.tuple_for
func.func @three_runs(%x: !vec, %v: !vec, %f: !vec, %members: memref<?x2xi32>,
                    %cell: !md.cell) -> (!vec, !vec, !vec) {
  %pairs = mdrt.from_buffer %members : memref<?x2xi32> to !groups
  %inc = md_exec.build_incidence %pairs : !groups -> !groups_inc
  %e0 = md_exec.empty : !vec
  %v0u = md_exec.particle_for ins(%v, %f : !vec, !vec) outs(%e0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>):
    %vn = arith.addf %v_i, %f_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  %z0 = md_exec.zeros : !vec
  %dv = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(1, 0))
          ins(%v0u : !vec) outs(%z0 : !vec) arity(2) disjoint {
  ^bb0(%r: vector<3xf64>, %a0: vector<3xf64>, %a1: vector<3xf64>):
    %d = arith.subf %a1, %a0 : vector<3xf64>
    %g = arith.mulf %d, %r : vector<3xf64>
    %h = arith.negf %g : vector<3xf64>
    md_exec.yield %g, %h : vector<3xf64>, vector<3xf64>
  } : !groups_inc, !vec -> !vec
  %e1 = md_exec.empty : !vec
  %v0 = md_exec.particle_for ins(%v0u, %dv : !vec, !vec) outs(%e1 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %dv_i: vector<3xf64>):
    %vn = arith.addf %v_i, %dv_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  %e2 = md_exec.empty : !vec
  %x1d = md_exec.particle_for ins(%x, %v0 : !vec, !vec) outs(%e2 : !vec) {
  ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
    %xn = arith.addf %x_i, %v_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec
  %z1 = md_exec.zeros : !vec
  %dx = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(1, 0))
          ins(%x1d : !vec) outs(%z1 : !vec) arity(2) disjoint {
  ^bb0(%r: vector<3xf64>, %a0: vector<3xf64>, %a1: vector<3xf64>):
    %d = arith.subf %a1, %a0 : vector<3xf64>
    %g = arith.mulf %d, %r : vector<3xf64>
    %h = arith.negf %g : vector<3xf64>
    md_exec.yield %g, %h : vector<3xf64>, vector<3xf64>
  } : !groups_inc, !vec -> !vec
  %e3 = md_exec.empty : !vec
  %x1 = md_exec.particle_for ins(%x1d, %dx : !vec, !vec) outs(%e3 : !vec) {
  ^bb0(%x_i: vector<3xf64>, %dx_i: vector<3xf64>):
    %xn = arith.addf %x_i, %dx_i : vector<3xf64>
    md_exec.yield %xn : vector<3xf64>
  } -> !vec
  %z2 = md_exec.zeros : !vec
  %s = md_exec.tuple_for %inc, %x1, %cell coordinates(displacement(1, 0))
         outs(%z2 : !vec) arity(2) disjoint {
  ^bb0(%r: vector<3xf64>):
    %h = arith.negf %r : vector<3xf64>
    md_exec.yield %r, %h : vector<3xf64>, vector<3xf64>
  } : !groups_inc, !vec -> !vec
  return %x1, %v0, %s : !vec, !vec, !vec
}


// A loop that closes a run still takes the loops after it that open none:
// the kick that follows the projection is fused with it.
//
// FUSED-LABEL: func.func @closes_only(
// CHECK-LABEL: func.func @closes_only(
// CHECK:         md_exec.particle_for
// CHECK:         md_exec.tuple_for
// CHECK:         md_exec.particle_for
// CHECK-NOT:     md_exec.particle_for
func.func @closes_only(%x: !vec, %v: !vec, %f: !vec,
                       %members: memref<?x2xi32>, %cell: !md.cell) -> !vec {
  %pairs = mdrt.from_buffer %members : memref<?x2xi32> to !groups
  %inc = md_exec.build_incidence %pairs : !groups -> !groups_inc
  %e0 = md_exec.empty : !vec
  %v0u = md_exec.particle_for ins(%v, %f : !vec, !vec) outs(%e0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>):
    %vn = arith.addf %v_i, %f_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  %z0 = md_exec.zeros : !vec
  %dv = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(1, 0))
          ins(%v0u : !vec) outs(%z0 : !vec) arity(2) disjoint {
  ^bb0(%r: vector<3xf64>, %a0: vector<3xf64>, %a1: vector<3xf64>):
    %d = arith.subf %a1, %a0 : vector<3xf64>
    %g = arith.mulf %d, %r : vector<3xf64>
    %h = arith.negf %g : vector<3xf64>
    md_exec.yield %g, %h : vector<3xf64>, vector<3xf64>
  } : !groups_inc, !vec -> !vec
  %e1 = md_exec.empty : !vec
  %v0 = md_exec.particle_for ins(%v0u, %dv : !vec, !vec) outs(%e1 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %dv_i: vector<3xf64>):
    %vn = arith.addf %v_i, %dv_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  %e2 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v0, %f : !vec, !vec) outs(%e2 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>):
    %vn = arith.addf %v_i, %f_i : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  return %v1 : !vec
}
