// RUN: mdir-opt %s --md-differentiate --md-inline --convert-md-to-md-exec --md-exec-fuse-loops --md-exec-accumulate-destinations --md-exec-assign-precision=mode=mixed --md-exec-assign-storage=memory=host --verify-each | FileCheck %s
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
md.potential @both(%x: !x, %h: !md.cell, %types: !i, %water: !water) -> f64 {
  %ww, %sw, %ss = md.template_pairs %types, %water : (!i, !water) -> (!ww, !sw, !ss)
  %a = md.sum_template_window %ww, %x, %h cutoff(1.0) extent(0.1) alpha(0.75) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %d = arith.subf %base, %b1 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %e = vector.reduction <add>, %square : vector<3xf64> into f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !ww, !x -> f64
  %b = md.sum_template_window %ss, %x, %h cutoff(1.0) extent(0.1) alpha(0.75) {
  ^bb0(%d: vector<3xf64>):
    %square = arith.mulf %d, %d : vector<3xf64>
    %e = vector.reduction <add>, %square : vector<3xf64> into f64
    md.yield %e, %d : f64, vector<3xf64>
  } : !ss, !x -> f64
  %u = arith.addf %a, %b : f64
  md.return %u : f64
}

// Independent classes write one force destination after fusion. Their site
// cutoffs remain separate; concatenation does not add unrelated pair arms.
// CHECK-LABEL: func.func @entry
// CHECK: md_exec.template_for
// CHECK-SAME: outs(%[[FORCE:[^ ]+]] : memref<?x3xf32
// CHECK-SAME: pairs [1, 0, 1]
// CHECK: water
// CHECK: md_exec.yield {{.*}} : vector<3xf32>, vector<3xf32>, vector<3xf32>, vector<3xf32>, vector<3xf32>, vector<3xf32>, vector<3xf32>
// CHECK: mixed
// CHECK: md_exec.yield
// CHECK: atoms
// CHECK: md_exec.yield {{.*}} : vector<3xf32>, vector<3xf32>, vector<3xf32>
// CHECK-NOT: md_exec.template_for
// CHECK: return
func.func @entry(%x: !x, %h: !md.cell, %types: !i, %water: !water) -> !x {
  %f = md.evaluate @both(%x, %h, %types, %water) request [forces]
    : (!x, !md.cell, !i, !water) -> !x
  func.return %f : !x
}
