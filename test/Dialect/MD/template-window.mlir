// RUN: mdir-opt %s --verify-each | FileCheck %s --check-prefix=DOMAIN
// RUN: mdir-opt %s --md-differentiate --verify-each | FileCheck %s --check-prefix=DERIV --implicit-check-not=md.evaluate
// The anchor image is explicit. Local site geometry stays visible to AD.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @ww on(@atoms) arity(6) orientation(ordered)
md.tuple_set @sw on(@atoms) arity(4) orientation(ordered)
md.tuple_set @ss on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!water = !md.relation<@atoms, 3, ordered, @water>
!ww = !md.relation<@atoms, 6, ordered, @ww>
!sw = !md.relation<@atoms, 4, ordered, @sw>
!ss = !md.relation<@atoms, 2, unordered, @ss>
// DOMAIN: md.sum_template_window
// DOMAIN: md.yield {{.*}}, {{.*}} : f64, vector<3xf64>
md.potential @window(%x: !x, %h: !md.cell, %types: !i, %water: !water) -> f64 {
  %ww, %sw, %ss = md.template_pairs %types, %water : (!i, !water) -> (!ww, !sw, !ss)
  %u = md.sum_template_window %ww, %x, %h cutoff(1.0) extent(0.1) alpha(0.125) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %half = arith.constant 0.5 : f64
    %e = arith.mulf %half, %r2 : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !ww, !x -> f64
  md.return %u : f64
}
// DERIV: md.function @window.
// DERIV: md.gather_template_window
// DERIV: md.sum_template_window
// DERIV: -> vector<9xf64>
// DERIV: md.call
md.function @entry(%x: !x, %h: !md.cell, %types: !i, %water: !water) -> (f64, !x, vector<9xf64>) {
  %u, %f, %w = md.evaluate @window(%x, %h, %types, %water) request [energy, forces, virial] : (!x, !md.cell, !i, !water) -> (f64, !x, vector<9xf64>)
  md.return %u, %f, %w : f64, !x, vector<9xf64>
}
