// RUN: mdir-opt %s --verify-each | FileCheck %s --check-prefix=DOMAIN
// RUN: mdir-opt %s --md-differentiate --verify-each | FileCheck %s --check-prefix=DERIV --implicit-check-not=md.evaluate
// The anchor image is explicit. Local site geometry stays visible to AD
// (D[kind-pairs]). One kind of three frame atoms; segment (0, 0).
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
// DOMAIN: md.sum_segment_window %{{.*}}, %{{.*}}, %{{.*}} segment(0, 0) cutoff(1.000000e+00)
// DOMAIN: md.yield {{.*}}, {{.*}} : f64, vector<3xf64>
md.potential @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>] : (!i, !kind) -> (!s00, !s10, !s11)
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
// DERIV: md.function @window.
// DERIV: md.gather_segment_window %{{.*}} segment(0, 0) cutoff(1.000000e+00) {
// DERIV: md.sum_segment_window
// DERIV: -> vector<9xf64>
// DERIV: md.call
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> (f64, !x, vector<9xf64>) {
  %u, %f, %w = md.evaluate @window(%x, %h, %types, %kind) request [energy, forces, virial] : (!x, !md.cell, !i, !kind) -> (f64, !x, vector<9xf64>)
  md.return %u, %f, %w : f64, !x, vector<9xf64>
}
