// The refusals of the rule of the writer of a list of units
// (D[md-alg-segments-writer]), which convert-md-exec-to-gpu checks before it
// lowers any op: the writer comes after the last writer of the positions and
// before the first loop that reads the frames. Each names the op that writes
// the positions. The allowed order lowers (segments-writer-lower.mlir).
//
// RUN: mdir-opt %s --split-input-file --convert-md-exec-to-gpu --verify-diagnostics

// -----

// (a) An op that writes the positions between the writer and the refresh
// that takes its test, as a later pass that moved a constraint or a
// barostat's scaling past the writer would leave.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @between_writer_and_refresh(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>, %s0: memref<?xf64, 1>, %s1: memref<?xf64, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  // expected-error @+1 {{is followed by an op that writes the positions before the refresh that takes its test; the writer comes after the last writer of the positions}}
  %m, %s = md_exec.write_unit_frames %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  // expected-note @+1 {{the op that writes them}}
  md_exec.particle_for outs(%x : memref<?x3xf64, 1>) {
    %z = arith.constant dense<1.0> : vector<3xf64>
    md_exec.yield %z : vector<3xf64>
  }
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

// -----

// (b) An op that writes the positions between the refresh and the loop that
// reads the frames, with the writer present.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @between_refresh_and_loop(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>, %s0: memref<?xf64, 1>, %s1: memref<?xf64, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %m, %s = md_exec.write_unit_frames %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) moved(%m) stale(%s) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  // expected-note @+1 {{the op that writes them}}
  md_exec.particle_for outs(%x : memref<?x3xf64, 1>) {
    %z = arith.constant dense<1.0> : vector<3xf64>
    md_exec.yield %z : vector<3xf64>
  }
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{reads the frames of a list of units whose positions an op writes after its writer; the writer of the list comes after the last writer of the positions}}
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

// -----

// A writer whose test no refresh in its block takes.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @no_refresh(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>, %s0: memref<?xf64, 1>, %s1: memref<?xf64, 1>) {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  // expected-error @+1 {{has no refresh in its block that takes its test}}
  %m, %s = md_exec.write_unit_frames %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  return
}
