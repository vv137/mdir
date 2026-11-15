// The refusals of the verifier of md_alg.traverse_pairs over the segments of
// a list of units (D417, Section 5, rules V1 to V13), one case
// each. Each case is the same valid op, a list of units of one kind of four
// places (a frame of three and one site) and the rest, with one change.
// D415's refusals are in traverse-invalid.mlir.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics

// V1: the regions of segments over a structure of groups.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64, %g: !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>) {
  // expected-error @+1 {{takes the regions of segments, which a list of units has}}
  %u = md_alg.traverse_pairs structure %g : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V1: one region over a list of units.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes a list of units, whose segments each take a region}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  {
  ^bb0(%r2: f32, %d: vector<3xf32>, %qi: f32, %qj: f32):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V2: a segment whose kinds are in the wrong order.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has segment(0, 1): its kinds must have g <= f}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(0, 1) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V2: a segment twice.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has segment(1, 0) after segment 1: each segment once, in increasing s(f, g)}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V2: the segments out of order.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has segment(1, 0) after segment 2}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  }
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V2: a segment of a kind the list does not have.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has segment(2, 0), but the list's kinds are 0 to K = 1}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(2, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V3: a place after the last of the unit (the site is place 3).
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has the site pair (0, 4) in segment(1, 0), whose units have 1 and 4 places}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 4) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V3: a site pair twice.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has the site pair (0, 3) twice in segment(1, 0)}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 3) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V3: a place after the last of the first unit (the rest has one place).
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has the site pair (1, 0) in segment(1, 0), whose units have 1 and 4 places}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (1, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V3: a segment with an empty table.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{expected at least one site pair in the table of segment(1, 1)}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [] {
  ^bb0(%qi: f32, %qj: f32):
    md_alg.yield
  } -> f64
  return
}

// -----

// V4: a value the op does not have.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has value 2 in the mask of (0, 0), but the op has 2 values}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [2]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V4: a site pair that yields nothing.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{has the site pair (0, 0) with no value}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) []] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V4: a mask out of order.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{expected the mask of (0, 3) in increasing order}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [1, 0], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V5: one site pair in another type.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{expected the region of segment(1, 0) to take r^2 and the displacement for each of its 2 site pairs, in one type}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f64, %d1: vector<3xf64>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.addf %r1, %r1 : f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V5: the values of a field at the places, the formed site counted, not at the members.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{then 1 + 3 values for each field of 'ins'}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32, %qj3: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V5: two segments in two types.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes r^2 of segment(1, 1) in 'f64' and of segment(1, 0) in 'f32': the site arithmetic of every segment is in one type}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f64, %d0: vector<3xf64>, %qi: f32, %qj: f32):
    %e0 = arith.addf %r0, %r0 : f64
    %w = arith.truncf %d0 : vector<3xf64> to vector<3xf32>
    md_alg.yield %w, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V6: one value fewer.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{expected the region of segment(1, 0) to yield the values of each mask}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V6: a force in another type than its destination.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{expected the region of segment(1, 0) to yield the values of each mask}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    %w = arith.extf %d0 : vector<3xf32> to vector<3xf64>
    md_alg.yield %w, %e0, %e1 : vector<3xf64>, f64, f64
  } -> f64
  return
}

// -----

// V7: the slice of a site pair reads the distance of another.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    // expected-error @+1 {{reads, in the slice of the site pair (0, 0) of segment(1, 0), the distance of another site pair}}
    %s = arith.addf %r0, %r1 : f32
    %e1 = arith.extf %s : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V7: a site pair yields the displacement of another.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{yields for the site pair (0, 3) of segment(1, 0) a value of another site pair}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d1, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V8: a store in the region of a segment.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64, %t: memref<?xf32>) {
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{writes memory in the region of 'md_alg.traverse_pairs'}}
    memref.store %r0, %t[%c0] : memref<?xf32>
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  } -> f64
  return
}

// -----

// V9: a segment that visits each pair in both orders.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) directed, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes a list whose segment(1, 0) does not meet its requirement}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V9: a segment whose exclusions are only marked.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, masked, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes a list whose segment(1, 1) does not meet its requirement}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V10: a fixed order of the forces under atomic additions.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, stable(group, search), (1, 0) unique, minimum, removed, stable(group, search), (1, 1) unique, minimum, removed, stable(group, search)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{adds destination 0, 'fixed_order', by the combination 'atomic'}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit stable combine atomic algebra ["fixed_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V11: a type that lacks a segment.
// expected-error @+1 {{expected 3 segments for 1 taken kinds, got 2}}
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V11: a kind whose site's weights do not sum to 1.
// expected-error @+1 {{expected the weights of each site to sum to 1}}
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.24], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V11: a kind that is not a #md.rigid_kind.
// expected-error @+1 {{expected each taken kind to be a '#md.rigid_kind', got 3 : i64}}
!sp = !mdrt.segmented_pairs<@atoms, kinds [3 : i64], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V11: the segments of a type out of order.
// expected-error @+1 {{expected segment 1 in the order of s(f, g) = f(f+1)/2 + g, got (1, 1)}}
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V12: the buffers of the groups alone, without the segments and the members.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64, %b: memref<?xi32>, %n: index) {
  // expected-error @+1 {{takes 9 buffers of the layout of the groups in its buffer form, not 7}}
  %u = md_alg.traverse_pairs lists(%b, %b, %b, %b, %b, %b, %b : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%n, %n) contract(!sp) reach(1.0)
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V12: the inner list of the buffer form without its five buffers.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64, %b: memref<?xi32>, %n: index) {
  // expected-error @+1 {{takes 14 buffers of the layout of the groups in its buffer form, not 9}}
  %u = md_alg.traverse_pairs lists(%b, %b, %b, %b, %b, %b, %b, %b, %b : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%n, %n) inner contract(!sp) reach(1.0)
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V1: the regions of segments in the buffer form of a structure of groups.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64, %b: memref<?xi32>, %n: index) {
  // expected-error @+1 {{takes the regions of segments, which a list of units has}}
  %u = md_alg.traverse_pairs lists(%b, %b, %b, %b, %b, %b, %b, %b, %b : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%n, %n) contract(!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>) reach(1.0)
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V12: a cutoff beyond the reach of the buffers.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64, %b: memref<?xi32>, %n: index) {
  // expected-error @+1 {{has a cutoff, 9.000000e-01, beyond the reach of its list, 5.000000e-01}}
  %u = md_alg.traverse_pairs lists(%b, %b, %b, %b, %b, %b, %b, %b, %b : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%n, %n) contract(!sp) reach(0.5)
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5]
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V13: the table of a block with segments.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes no table of a block (scatter) with segments}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5] scatter(#md_exec.scatter<hash(512, 4, 2000)>)
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V13: a rotation that is not a power of 2.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @f(%l: !sp, %x: !x, %cell: !md.cell, %q: !q, %f: !f, %u0: f64) {
  // expected-error @+1 {{rotates the partners through 12 lanes: a power of 2 up to 16}}
  %u = md_alg.traverse_pairs structure %l : !sp
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5] rotate(12)
  segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>, %qi: f32, %qj0: f32, %qj1: f32, %qj2: f32):
    %e0 = arith.extf %r0 : f32 to f64
    %e1 = arith.extf %r1 : f32 to f64
    md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
  }
  segment(1, 1) sites [(0, 0) [0, 1]] {
  ^bb0(%r0: f32, %d0: vector<3xf32>, %qi: f32, %qj: f32):
    %e0 = arith.extf %r0 : f32 to f64
    md_alg.yield %d0, %e0 : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// V13: a rotation, an execution parameter of the segments' schedule, on an op over groups.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
func.func @f(%g: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes sites and a rotation only with segments}}
  %u = md_alg.traverse_pairs structure %g : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5] rotate(16) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}
