// RUN: mdir-opt %s --md-differentiate --verify-each | FileCheck %s --implicit-check-not=md.evaluate
// Named segments compose with the ordinary coordinate derivative rules
// (D405). No position or cell dependence is hidden in
// md.kind_pairs itself.
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!k = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
md.potential @energy(%x: !x, %h: !md.cell, %types: !i, %kind: !k) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [-0.5, 0.75, 0.75], extent = 0.2>] : (!i, !k) -> (!s00, !s10, !s11)
  %u = md.sum_tuples %s00, %x, %h coordinates(distance(0, 3)) {
  ^bb0(%r: f64):
    %square = arith.mulf %r, %r : f64
    md.yield %square : f64
  } : !s00, !x -> f64
  md.return %u : f64
}
// CHECK: md.function @energy.
// CHECK: md.kind_pairs
// CHECK: md.gather_tuples
// CHECK: md.sum_tuples
// CHECK: md.call
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %kind: !k) -> (f64, !x, vector<9xf64>) {
  %u, %f, %v = md.evaluate @energy(%x, %h, %types, %kind) request [energy, forces, virial] : (!x, !md.cell, !i, !k) -> (f64, !x, vector<9xf64>)
  md.return %u, %f, %v : f64, !x, vector<9xf64>
}
