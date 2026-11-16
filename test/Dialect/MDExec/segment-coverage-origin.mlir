// The coverage query of lists of units, md_exec::verifySegmentCoverage,
// apart from the lowering (#898): it proves a list that begins at its
// empty list, and refuses with a diagnostic, not an assertion, a list whose
// refreshes begin at a function argument or at the result of another op,
// which give no radius of the rest.
//
// RUN: mdir-opt %s --split-input-file --test-md-exec-segment-coverage --verify-diagnostics

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @covered(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-remark @+1 {{segment coverage proved}}
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

// The refreshes begin at an argument of the function.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
// expected-note @+1 {{the list of units begins here}}
func.func @argument(%l0: !sp, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{takes a list of units whose empty list or build is not found: its refreshes begin at an argument of 'func.func', which does not give the radius of the rest}}
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

// The refreshes begin at a result of scf.if.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @region(%c: i1, %t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> f64 {
  %a = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %b = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.2) : memref<?xi32, 1> -> !sp
  // expected-note @+1 {{the list of units begins here}}
  %l0 = scf.if %c -> (!sp) {
    scf.yield %a : !sp
  } else {
    scf.yield %b : !sp
  }
  %l = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{takes a list of units whose empty list or build is not found: its refreshes begin at a result of 'scf.if', which does not give the radius of the rest}}
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
