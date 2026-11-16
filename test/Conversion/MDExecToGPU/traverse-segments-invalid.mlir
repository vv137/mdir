// The refusals of convert-md-exec-to-gpu for a list of units and the loops
// over its segments (D417, D[md-alg-segments-build]), each with its own
// message. The lists are lowered; their SIMT traversal is slice S3's, and the
// serial definition runs over them with traverse-oracle=serial.
//
// RUN: mdir-opt %s --split-input-file --convert-md-exec-to-gpu --verify-diagnostics

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @traverse(%l: !sp, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{visits the segments of a list of units, whose SIMT schedule this pass does not lower yet (D417, slice S3)}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
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

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @list(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>) -> !sp {
  // expected-error @+1 {{has no refresh, which gives the positions it is built at}}
  %l = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  return %l : !sp
}

// -----

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @build(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell) -> !sp {
  // expected-error @+1 {{builds a list of units in place; run 'md-exec-reuse-neighbors' to give it an empty list and a refresh}}
  %l = md_exec.build_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) at(%x, %cell) cutoff(0.9) skin(0.1) : memref<?xi32, 1>, memref<?x3xf64, 1> -> !sp
  return %l : !sp
}

// -----

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @single(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf32, 1>, %cell: !md.cell) -> !sp {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  // expected-error @+1 {{builds a list of units from positions in 'f32'; lists of units are built from positions in f64}}
  %l1 = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf32, 1>
  return %l1 : !sp
}
