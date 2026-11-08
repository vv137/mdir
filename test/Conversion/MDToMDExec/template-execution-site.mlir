// RUN: mdir-opt %s --md-differentiate --md-inline --convert-md-to-md-exec --test-md-exec-template-windows=cache-sites --verify-each | FileCheck %s
// RUN: mdir-opt %s --md-differentiate --md-inline --convert-md-to-md-exec --md-exec-assign-precision=mode=mixed --test-md-exec-template-windows=cache-sites --verify-each | FileCheck %s
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @ww on(@atoms) arity(6) orientation(ordered)
md.tuple_set @sw on(@atoms) arity(4) orientation(ordered)
md.tuple_set @ss on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!water = !md.relation<@atoms, 4, ordered, @water>
!ww = !md.relation<@atoms, 6, ordered, @ww>
!sw = !md.relation<@atoms, 4, ordered, @sw>
!ss = !md.relation<@atoms, 2, unordered, @ss>
md.potential @site(%x: !x, %h: !md.cell, %types: !i, %water: !water) -> f64 {
  %ww, %sw, %ss = md.template_pairs %types, %water : (!i, !water) -> (!ww, !sw, !ss)
  %u = md.sum_template_window %ww, %x, %h cutoff(1.0) extent(0.2) alpha(0.75) {
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
  } : !ww, !x -> f64
  md.return %u : f64
}
// CHECK-LABEL: func.func @window_energy
// CHECK-SAME: test.cached_sites = 2 : i32
// CHECK-LABEL: func.func @window_forces
// CHECK-SAME: test.cached_sites = 2 : i32
// CHECK-LABEL: func.func @window_virial
// CHECK-SAME: test.cached_sites = 2 : i32
func.func @entry(%x: !x, %h: !md.cell, %types: !i, %water: !water) -> (f64, !x, vector<9xf64>) {
  %u, %f, %w = md.evaluate @site(%x, %h, %types, %water) request [energy, forces, virial]
    : (!x, !md.cell, !i, !water) -> (f64, !x, vector<9xf64>)
  func.return %u, %f, %w : f64, !x, vector<9xf64>
}
