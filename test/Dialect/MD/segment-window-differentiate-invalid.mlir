// RUN: mdir-opt %s -split-input-file --md-differentiate -verify-diagnostics
// The refusals of the derivative of a segment window (D405,
// those of #795), one case for each.
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 3, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// Higher derivatives and tunables are not admitted.
md.potential @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind, %q: f64) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{supports energy, forces and virial only; higher derivatives and tunables require the general path}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return %u : f64
}
md.function @parameter(%x: !x, %h: !md.cell, %types: !i, %kind: !kind, %q: f64) -> f64 {
  %g = md.evaluate @window(%x, %h, %types, %kind, %q) request [derivative(4)] : (!x, !md.cell, !i, !kind, f64) -> f64
  md.return %g : f64
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 3, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// A gathered field may not be the positions.
md.potential @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error@+1 {{cannot differentiate with respect to the positions through this use}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) gather(%x : !x) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>, %p0: vector<3xf64>, %p1: vector<3xf64>, %p2: vector<3xf64>, %p3: vector<3xf64>, %p4: vector<3xf64>, %p5: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return %u : f64
}
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> (f64, !x, vector<9xf64>) {
  %u, %f, %w = md.evaluate @window(%x, %h, %types, %kind) request [energy, forces, virial] : (!x, !md.cell, !i, !kind) -> (f64, !x, vector<9xf64>)
  md.return %u, %f, %w : f64, !x, vector<9xf64>
}

// -----
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 3, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
// A capture may not hide a dependence on the cell.
md.potential @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>] : (!i, !kind) -> (!s00, !s10, !s11)
  %edges = md_exec.cell_edges %h : vector<3xf64>
  // expected-error@+1 {{segment-window captures must not hide position or cell dependence}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    %edge = vector.extract %edges[0] : f64 from vector<3xf64>
    %withCell = arith.addf %e, %edge : f64
    md.yield %withCell, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return %u : f64
}
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> (f64, !x, vector<9xf64>) {
  %u, %f, %w = md.evaluate @window(%x, %h, %types, %kind) request [energy, forces, virial] : (!x, !md.cell, !i, !kind) -> (f64, !x, vector<9xf64>)
  md.return %u, %f, %w : f64, !x, vector<9xf64>
}
