// The writer of a list of units (D422): for a refresh
// of a list of units in a loop over steps, md-exec-expose-validity makes the
// writer right after the op that gives the refresh's positions, the last op
// that writes them, so that it comes before the ops that read them (the
// reciprocal sum of a step) and the refresh takes its test; a refresh of an empty list takes none. Nothing
// changes for a refresh of groups (expose-validity.mlir).
//
// RUN: mdir-opt %s --md-exec-expose-validity | FileCheck %s

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!vec = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f32>
!k = !md.relation<@atoms, 4, ordered, @kind>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint

// CHECK-LABEL: func.func @steps(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[V:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell
func.func @steps(%x: !vec, %v: !vec, %cell: !md.cell, %n: index, %t: !i, %kind: !k) -> !vec {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:      %[[L0:[0-9]+]] = md_exec.empty_segmented_pairs
  // CHECK-NEXT: md_exec.refresh_segmented_pairs %[[L0]] at(%[[X]], %[[CELL]]) cutoff
  %l0 = md_exec.empty_segmented_pairs %t kinds(%kind : !k) cutoff(0.9) skin(0.1) : !i -> !sp
  %first = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, !vec
  // CHECK:      scf.for
  // CHECK-SAME:   iter_args(%[[XA:[a-z0-9]+]] = %[[X]], %[[LA:[a-z0-9]+]] = %{{[0-9]+}})
  %xe, %le = scf.for %step = %c0 to %n step %c1
      iter_args(%xa = %x, %la = %first) -> (!vec, !sp) {
    // CHECK:      %[[X1:[0-9]+]] = md_exec.particle_for ins(%[[XA]], %[[V]] :
    // CHECK:      } -> !md.field<@atoms, 3 x f64>
    // CHECK-NEXT: %[[MOVED:[a-z0-9_]+]], %[[STALE:[a-z0-9_]+]] = md_exec.write_unit_frames %[[LA]] at(%[[X1]], %[[CELL]]) cutoff(9.000000e-01) skin(1.000000e-01) prune_skin(5.000000e-02)
    // CHECK-NEXT: md_exec.empty
    // CHECK-NEXT: md_exec.particle_for ins(%[[X1]] :
    // CHECK:      } -> !md.field<@atoms, 3 x f64>
    // CHECK-NEXT: md_exec.refresh_segmented_pairs %[[LA]] at(%[[X1]], %[[CELL]]) moved(%[[MOVED]]) stale(%[[STALE]]) cutoff(9.000000e-01) skin(1.000000e-01) prune_skin(5.000000e-02)
    %e = md_exec.empty : !vec
    %x1 = md_exec.particle_for ins(%xa, %v : !vec, !vec) outs(%e : !vec) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
      %xn = arith.addf %x_i, %v_i : vector<3xf64>
      md_exec.yield %xn : vector<3xf64>
    } -> !vec
    %e2 = md_exec.empty : !vec
    %w = md_exec.particle_for ins(%x1 : !vec) outs(%e2 : !vec) {
    ^bb0(%x_i: vector<3xf64>):
      md_exec.yield %x_i : vector<3xf64>
    } -> !vec
    %l1 = md_exec.refresh_segmented_pairs %la at(%x1, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !vec
    scf.yield %x1, %l1 : !vec, !sp
  }
  return %xe : !vec
}
