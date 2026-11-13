// RUN: mdir-opt %s -split-input-file --md-differentiate --verify-each | FileCheck %s
// Window shapes of segments that the verifier proves and the derivative
// takes (D[kind-pairs]): the rest against a kind with a site, the rest
// with itself, and two kinds, one with a site whose weights are not
// symmetric.
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
// CHECK-LABEL: md.function @rest_kind.
// The rest's particle (slot 0) against the kind's site (slot 4).
// CHECK: md.gather_segment_window %{{.*}} segment(1, 0) cutoff(1.000000e+00) site_forces {
// CHECK: md.yield %{{[0-9]+}}, %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %{{[0-9]+}}, %{{[0-9]+}} :
md.potential @rest_kind(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  %u = md.sum_segment_window %s10, %x, %h segment(1, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %d = arith.subf %base, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    md.yield %r2, %d : f64, vector<3xf64>
  } : !s10, !x -> f64
  md.return %u : f64
}
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> (f64, !x) {
  %u, %f = md.evaluate @rest_kind(%x, %h, %types, %kind) request [energy, forces] : (!x, !md.cell, !i, !kind) -> (f64, !x)
  md.return %u, %f : f64, !x
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
// The rest with itself: the displacement is the anchor displacement.
// CHECK-LABEL: md.function @rest.
// CHECK: md.gather_segment_window %{{.*}} segment(1, 1) cutoff(1.000000e+00) {
md.potential @rest(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  %u = md.sum_segment_window %s11, %x, %h segment(1, 1) cutoff(1.0) {
  ^bb0(%d: vector<3xf64>):
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    md.yield %r2, %d : f64, vector<3xf64>
  } : !s11, !x -> f64
  md.return %u : f64
}
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> (f64, !x) {
  %u, %f = md.evaluate @rest(%x, %h, %types, %kind) request [energy, forces] : (!x, !md.cell, !i, !kind) -> (f64, !x)
  md.return %u, %f : f64, !x
}
// -----
md.particle_set @atoms
md.tuple_set @a on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @b on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s20 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s21 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s22 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!a = !md.relation<@atoms, 3, ordered, @a>
!b = !md.relation<@atoms, 4, ordered, @b>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 6, ordered, @s10>
!s11 = !md.relation<@atoms, 6, ordered, @s11>
!s20 = !md.relation<@atoms, 4, ordered, @s20>
!s21 = !md.relation<@atoms, 4, ordered, @s21>
!s22 = !md.relation<@atoms, 2, unordered, @s22>
// Two kinds: the site of kind 1 at 0.5 x_0 + 0.3 x_1 + 0.2 x_2 (place 1 of
// four, between the frame atoms 0, 2 and 3) against frame atom 2 of kind 0.
// The slots are kind 1's four places, then kind 0's three.
// CHECK-LABEL: md.function @two.
// CHECK: md.gather_segment_window %{{.*}} segment(1, 0) cutoff(1.000000e+00) site_forces {
// CHECK: md.yield %cst{{[_0-9]*}}, %{{[0-9]+}}, %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %{{[0-9]+}}, %{{[0-9]+}} :
md.potential @two(%x: !x, %h: !md.cell, %types: !i, %a: !a, %b: !b) -> f64 {
  %r:6 = md.kind_pairs %types, kinds(%a, %b) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.2>, #md.rigid_kind<frame = [0, 2, 3], sites = [1], weights = [0.5, 0.3, 0.2], extent = 0.2>] : (!i, !a, !b) -> (!s00, !s10, !s11, !s20, !s21, !s22)
  %u = md.sum_segment_window %r#1, %x, %h segment(1, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %w1 = arith.constant 0.3 : f64
    %w2 = arith.constant 0.2 : f64
    %s1 = vector.broadcast %w1 : f64 to vector<3xf64>
    %s2 = vector.broadcast %w2 : f64 to vector<3xf64>
    %p1 = arith.mulf %s1, %a1 : vector<3xf64>
    %p2 = arith.mulf %s2, %a2 : vector<3xf64>
    %pa = arith.addf %p1, %p2 : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %b2 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    md.yield %r2, %d : f64, vector<3xf64>
  } : !s10, !x -> f64
  md.return %u : f64
}
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %a: !a, %b: !b) -> (f64, !x) {
  %u, %f = md.evaluate @two(%x, %h, %types, %a, %b) request [energy, forces] : (!x, !md.cell, !i, !a, !b) -> (f64, !x)
  md.return %u, %f : f64, !x
}
