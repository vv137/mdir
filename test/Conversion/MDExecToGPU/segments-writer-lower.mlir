// The allowed order of the writer of a list of units
// (D422): the op that writes the positions, then the
// writer, the refresh that takes its test, and the loop that reads the
// frames. It lowers, the writer to a kernel and flags read at the refresh,
// and the loop reads the list's frames (segments-writer-invalid.mlir holds
// the refused orders).
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
// CHECK-LABEL: func.func @allowed
// CHECK:       call @mdrtFlagStart
// CHECK:       call @mdrtFlagFinish
// CHECK:       call @mdrtSegmentsStopExtent
// CHECK:       call @mdrtGroupsFrame
// CHECK-NOT:   md_exec.write_unit_frames
func.func @allowed(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>, %s0: memref<?xf64, 1>, %s1: memref<?xf64, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  md_exec.particle_for outs(%x : memref<?x3xf64, 1>) {
    %z = arith.constant dense<1.0> : vector<3xf64>
    md_exec.yield %z : vector<3xf64>
  }
  %m, %s = md_exec.write_unit_frames %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) moved(%m) stale(%s) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      scratch(%s0, %s1 : memref<?xf64, 1>, memref<?xf64, 1>)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
    segment(1, 1) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f32, %d: vector<3xf32>):
      %e = arith.extf %r2 : f32 to f64
      md_alg.yield %d, %e : vector<3xf32>, f64
    } -> f64
  return %u : f64
}
