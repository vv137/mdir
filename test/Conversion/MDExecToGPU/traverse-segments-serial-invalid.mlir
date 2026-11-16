// The refusals of the serial definition over a list of units that the
// device builds (convert-md-exec-to-gpu with traverse-oracle=serial;
// D[md-alg-segments-build]), each with its own message: the coverage query
// of each segment (D357, D417 Section 3), and the records that a loop reads,
// the outer ones of a list or the inner ones of a dual list (D114).
//
// RUN: mdir-opt %s --split-input-file --convert-md-exec-to-gpu="traverse-oracle=serial" --verify-diagnostics

// The loop's cutoff beyond the cutoff that the list was built with: segment
// (1, 1), the rest with itself (radius 0), needs pairs within 1.0.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @beyond(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{segment (1, 1) needs pairs of reference points within 1, beyond the covering reach 0.9 of the list of units}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(1.0) weights [0.5]
    segment(1, 1) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f32, %d: vector<3xf32>):
      %e = arith.extf %r2 : f32 to f64
      md_alg.yield %d, %e : vector<3xf32>, f64
    } -> f64
  return %u : f64
}

// -----

// The water with itself, segment (0, 0), needs 0.9 + 2 (0.1); the list's
// covering reach for it is 0.85 + 2 (0.1).
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @water(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.85) skin(0.15) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.85) skin(0.15) prune_skin(0.15) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{segment (0, 0) needs pairs of reference points within 1.1, beyond the covering reach 1.05 of the list of units}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
    segment(0, 0) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f32, %d: vector<3xf32>):
      %e = arith.extf %r2 : f32 to f64
      md_alg.yield %d, %e : vector<3xf32>, f64
    } -> f64
  return %u : f64
}

// -----

// A loop over a list that no refresh has built.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @empty(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  // expected-note @+1 {{unresolved coverage boundary}}
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{cannot prove neighbor coverage across md_exec.empty_segmented_pairs boundary}}
  %u = md_alg.traverse_pairs structure %l0 : !sp
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

// The inner records of a list that its refresh does not keep dual.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @inner(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{reads the inner records of a list of units that its refresh does not keep dual}}
  %u = md_alg.traverse_pairs structure %l : !sp inner
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

// The outer records of a dual list.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @outer(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{reads the outer records of a dual list of units, which the loops take through the inner records}}
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

// Covered: the water with itself at the cutoff that the list was built
// with, lowered to the serial definition with no diagnostic. A covering
// reach that leaves out a radius (reach + e_f) fails here.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @covered(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
    segment(0, 0) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f32, %d: vector<3xf32>):
      %e = arith.extf %r2 : f32 to f64
      md_alg.yield %d, %e : vector<3xf32>, f64
    } -> f64
  return %u : f64
}
