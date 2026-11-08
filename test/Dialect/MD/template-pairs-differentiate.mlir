// RUN: mdir-opt %s --md-differentiate --verify-each | FileCheck %s --implicit-check-not=md.evaluate
// Named class domains compose with the ordinary coordinate derivative rules.
// No position or cell dependence is hidden in md.template_pairs itself.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @ww on(@atoms) arity(6) orientation(ordered)
md.tuple_set @sw on(@atoms) arity(4) orientation(ordered)
md.tuple_set @ss on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!w = !md.relation<@atoms, 4, ordered, @water>
!ww = !md.relation<@atoms, 6, ordered, @ww>
!sw = !md.relation<@atoms, 4, ordered, @sw>
!ss = !md.relation<@atoms, 2, unordered, @ss>
md.potential @energy(%x: !x, %h: !md.cell, %types: !i, %water: !w) -> f64 {
  %ww, %sw, %ss = md.template_pairs %types, %water : (!i, !w) -> (!ww, !sw, !ss)
  %u = md.sum_tuples %ww, %x, %h coordinates(distance(0, 3)) {
  ^bb0(%r: f64):
    %square = arith.mulf %r, %r : f64
    md.yield %square : f64
  } : !ww, !x -> f64
  md.return %u : f64
}
// CHECK: md.function @energy.
// CHECK: md.template_pairs
// CHECK: md.gather_tuples
// CHECK: md.sum_tuples
// CHECK: md.call
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %water: !w) -> (f64, !x, vector<9xf64>) {
  %u, %f, %v = md.evaluate @energy(%x, %h, %types, %water) request [energy, forces, virial] : (!x, !md.cell, !i, !w) -> (f64, !x, vector<9xf64>)
  md.return %u, %f, %v : f64, !x, vector<9xf64>
}
