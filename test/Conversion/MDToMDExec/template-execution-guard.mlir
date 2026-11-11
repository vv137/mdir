// RUN: mdir-opt %s --md-differentiate --md-inline --convert-md-to-md-exec --md-exec-assign-precision=mode=mixed --test-md-exec-template-windows --verify-each | FileCheck %s --implicit-check-not=arith.select
// A captured scalar remains available inside the site guard. Expensive
// arithmetic in the admitted straight-line source body stays under it:
// a transcendental function is outside the select form's precondition
// (isTemplateBodySpeculatable), which template-execution-speculate.mlir
// pins with its other negative cases.
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
md.potential @guard(%x: !x, %h: !md.cell, %types: !i, %water: !water) -> f64 {
  %ww, %sw, %ss = md.template_pairs %types, %water : (!i, !water) -> (!ww, !sw, !ss)
  %bias = arith.constant 0.5 : f64
  %u = md.sum_template_window %ww, %x, %h cutoff(1.0) extent(0.1) alpha(0.75) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %d = arith.subf %base, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %shifted = arith.addf %r2, %bias : f64
    %e = math.exp %shifted : f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !ww, !x -> f64
  md.return %u : f64
}
// CHECK-LABEL: func.func @window_energy
// CHECK-NOT: math.exp
// CHECK: arith.cmpf olt
// CHECK: scf.if
// CHECK-NOT: math.exp
// CHECK: arith.addf
// CHECK: math.exp {{.*}} : f32
// CHECK: arith.extf {{.*}} : f32 to f64
// CHECK: scf.yield
// CHECK: else
func.func @entry(%x: !x, %h: !md.cell, %types: !i, %water: !water) -> f64 {
  %u = md.evaluate @guard(%x, %h, %types, %water) request [energy]
    : (!x, !md.cell, !i, !water) -> f64
  func.return %u : f64
}
