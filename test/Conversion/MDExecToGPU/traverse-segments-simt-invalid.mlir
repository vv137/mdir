// The refusals of the SIMT schedule of a list of units
// (D421), each with its own message: what the schedule
// does not take is refused by name, never lowered otherwise.
//
// RUN: mdir-opt %s --split-input-file --convert-md-exec-to-gpu --verify-diagnostics

// -----

// Site pairs in f64: the schedule is lowered for mixed precision.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @double(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>, %s0: memref<?xf64, 1>, %s1: memref<?xf64, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{computes its site pairs of segment (1, 1) in 'f64'; the SIMT schedule of a list of units takes f32 (mixed precision), and the serial definition any type}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      scratch(%s0, %s1 : memref<?xf64, 1>, memref<?xf64, 1>)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
    segment(1, 1) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      %v = arith.truncf %d : vector<3xf64> to vector<3xf32>
      md_alg.yield %v, %r2 : vector<3xf32>, f64
    } -> f64
  return %u : f64
}

// -----

// A rotation through 8 lanes.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @rotated(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>, %s0: memref<?xf64, 1>, %s1: memref<?xf64, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{turns its partners through 8 lanes; the schedule of a list of units holds 16 units a half-warp and turns them through 16}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      scratch(%s0, %s1 : memref<?xf64, 1>, memref<?xf64, 1>)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5] rotate(8)
    segment(1, 1) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f32, %d: vector<3xf32>):
      %e = arith.extf %r2 : f32 to f64
      md_alg.yield %d, %e : vector<3xf32>, f64
    } -> f64
  return %u : f64
}

// -----

// A field of 64 bits that a slice reads.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @wide(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>, %q: memref<?xf64, 1>, %s0: memref<?xf64, 1>, %s1: memref<?xf64, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{reads a field of 'f64' in segment (1, 1); the SIMT schedule of a list of units turns fields of 32 bits}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      ins(%q : memref<?xf64, 1>)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      scratch(%s0, %s1 : memref<?xf64, 1>, memref<?xf64, 1>)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
    segment(1, 1) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f32, %d: vector<3xf32>, %qi: f64, %qj: f64):
      %e = arith.extf %r2 : f32 to f64
      %p = arith.mulf %qi, %qj : f64
      %s = arith.addf %e, %p : f64
      md_alg.yield %d, %s : vector<3xf32>, f64
    } -> f64
  return %u : f64
}

// -----

// Three taken kinds: the schedule is measured with the generator's two
// (D405), and refuses more.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>, #md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>, #md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group), (2, 0) unique, minimum, removed, grouped_by(group), (2, 1) unique, minimum, removed, grouped_by(group), (2, 2) unique, minimum, removed, grouped_by(group), (3, 0) unique, minimum, removed, grouped_by(group), (3, 1) unique, minimum, removed, grouped_by(group), (3, 2) unique, minimum, removed, grouped_by(group), (3, 3) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @three(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>, %s0: memref<?xf64, 1>, %s1: memref<?xf64, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k, %k, %k : memref<?x3xi32, 1>, memref<?x3xi32, 1>, memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{takes 3 kinds; the pair segments take at most 2 kinds of rigid groups (D405), and the schedule is measured with those}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      scratch(%s0, %s1 : memref<?xf64, 1>, memref<?xf64, 1>)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
    segment(3, 3) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f32, %d: vector<3xf32>):
      %e = arith.extf %r2 : f32 to f64
      md_alg.yield %d, %e : vector<3xf32>, f64
    } -> f64
  return %u : f64
}
