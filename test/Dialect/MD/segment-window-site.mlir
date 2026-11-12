// RUN: mdir-opt %s --md-differentiate --verify-each | FileCheck %s
// A site formed from the kind's weights over the frame (D[kind-pairs]):
// the extra point at x_0 + 0.75 (h_1 + h_2), weights (-0.5, 0.75, 0.75).
// The radial derivative keeps one force slot for each place, the site's
// among them, and marks the layout.
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
md.potential @window(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !kind) -> (!s00, !s10, !s11)
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  md.return %u : f64
}
// CHECK: md.gather_segment_window %{{.*}} segment(0, 0) cutoff(1.000000e+00) site_forces {
// Eight slots, the site of a (place 3) and of b (place 7) among them.
// CHECK: md.yield %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %{{[0-9]+}}, %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %cst{{[_0-9]*}}, %{{[0-9]+}}, %{{[0-9]+}} : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> (f64, !x, vector<9xf64>) {
  %u, %f, %w = md.evaluate @window(%x, %h, %types, %kind) request [energy, forces, virial] : (!x, !md.cell, !i, !kind) -> (f64, !x, vector<9xf64>)
  md.return %u, %f, %w : f64, !x, vector<9xf64>
}
