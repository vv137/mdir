// convert-md-exec-to-gpu does not lower a list of units or a loop over its
// segments yet (D417): the build is slice S2, the SIMT
// schedule slice S3. It refuses them by name, the list's ops and the
// traversal, instead of reaching the schedule of the groups.
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
  // expected-error @+1 {{is an op of a list of units, which this pass does not lower yet (D417, slices S2 and S3)}}
  %l = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  return %l : !sp
}
