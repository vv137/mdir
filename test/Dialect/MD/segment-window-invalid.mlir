// RUN: mdir-opt %s -split-input-file -verify-diagnostics
// The refusals of md.sum_segment_window and md.gather_segment_window
// (D405), one case for each.
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// The relation is a segment of md.kind_pairs.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind, %other: !s00) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected a segment of 'md.kind_pairs'}}
  %u = md.sum_segment_window %other, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// g <= f: the segments are numbered for g <= f only.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected g <= f in segment(0, 1)}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 1) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// f <= K: kind 1 is the rest; there is no kind 2.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected kinds 0 to K = 1 in segment(f, g)}}
  %u = md.sum_segment_window %s00, %x, %h segment(2, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// A wrong segment index: (0, 0) is s(0, 0) = 0, but the relation is result 1, s(1, 0).
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{segment(0, 0) is s(f, g) = 0, not result 1 of 'md.kind_pairs'}}
  %u = md.sum_segment_window %s10, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s10, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// The wrong site count: segment (0, 0) of two frames of three atoms has four frame offsets, not three.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected the anchor displacement, 4 frame offsets of the segment's kinds, then 6 values of each gathered field}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// Weights that differ from the kind: the site is formed with 0.625, the kind has 0.75.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{cannot prove the site geometry of the segment: a displacement is not the anchor displacement plus a place of each kind, with the kinds' weights}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.625 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %d = arith.addf %base, %pa : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// A transposed weight row: the kind places its site at 0.5 x_0 + 0.3 x_1 + 0.2 x_2; the body forms 0.2 h_1 + 0.3 h_2.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.3, 0.2], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{cannot prove the site geometry of the segment: a displacement is not the anchor displacement plus a place of each kind, with the kinds' weights}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %w1 = arith.constant 0.2 : f64
    %w2 = arith.constant 0.3 : f64
    %s1 = vector.broadcast %w1 : f64 to vector<3xf64>
    %s2 = vector.broadcast %w2 : f64 to vector<3xf64>
    %p1 = arith.mulf %s1, %a1 : vector<3xf64>
    %p2 = arith.mulf %s2, %a2 : vector<3xf64>
    %pa = arith.addf %p1, %p2 : vector<3xf64>
    %d = arith.addf %base, %pa : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// A positive cutoff.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected a positive finite cutoff}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(-1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// Site forces are a layout of the gather.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{site forces are a layout of 'md.gather_segment_window' only}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) site_forces {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// A sum is an energy or a strain.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected f64 energy or vector<9xf64> strain}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> vector<3xf64>
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// A gather is a force field.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected a force field with the positions' type}}
  %u = md.gather_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// Contributions, then as many displacements.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected site contributions followed by their site displacements}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    md.yield %r2, %d, %d : f64, vector<3xf64>, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// The body is straight-line.
md.function @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{expected a straight-line, effect-free window body}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %e = scf.execute_region -> f64 {
      %r = arith.constant 1.0 : f64
      scf.yield %r : f64
    }
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return
}
