// Native constraint provenance survives conversion, tuple fusion,
// destination merging and storage. An ordinary gather gets no marker.
// RUN: mdir-opt %s --convert-md-to-md-exec --md-exec-fuse-loops --md-exec-accumulate-destinations --md-exec-assign-storage=memory=host --verify-each | FileCheck %s
md.particle_set @atoms
md.tuple_set @rigid on(@atoms) arity(3) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!rel = !md.relation<@atoms, 3, ordered, @rigid>
!inc = !mdrt.incidence<@atoms, 3, @rigid>

// CHECK-LABEL: func.func @source_marked
// CHECK: md_exec.tuple_for
// CHECK: md.template_constraint_epoch
// CHECK: return
func.func @source_marked(%x: !x, %h: !md.cell, %r: !rel) -> !x {
  %a = md.gather_tuples %r, %x, %h coordinates(displacement(1, 0)) {
  ^bb0(%d: vector<3xf64>):
    %z = arith.constant dense<0.0> : vector<3xf64>
    md.yield %d, %z, %z : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } {md.template_constraint_epoch} : !rel, !x -> !x
  return %a : !x
}

// CHECK-LABEL: func.func @fused_marked
// CHECK: md_exec.tuple_for
// CHECK-SAME: outs(%{{[a-zA-Z0-9_]+}} :
// CHECK: md.template_constraint_epoch
// CHECK-NOT: md_exec.tuple_for
// CHECK: return
func.func @fused_marked(%x: !x, %h: !md.cell, %relation: !rel) -> !x {
  %r = md_exec.build_incidence %relation : !rel -> !inc
  %z1 = md_exec.zeros : !x
  %a = md_exec.tuple_for %r, %x, %h coordinates(displacement(1, 0))
      outs(%z1 : !x) arity(3) disjoint {
  ^bb0(%d: vector<3xf64>):
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %d, %z, %z : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } {md.template_constraint_epoch} : !inc, !x -> !x
  %z2 = md_exec.zeros : !x
  %b = md_exec.tuple_for %r, %x, %h coordinates(displacement(1, 0))
      outs(%z2 : !x) arity(3) disjoint {
  ^bb0(%d: vector<3xf64>):
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %z, %d, %z : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !inc, !x -> !x
  %e = md_exec.empty : !x
  %sum = md_exec.particle_for ins(%a, %b : !x, !x) outs(%e : !x) {
  ^bb0(%a0: vector<3xf64>, %b0: vector<3xf64>):
    %s = arith.addf %a0, %b0 : vector<3xf64>
    md_exec.yield %s : vector<3xf64>
  } -> !x
  return %sum : !x
}

// CHECK-LABEL: func.func @ordinary
// CHECK-NOT: md.template_constraint_epoch
// CHECK: md_exec.tuple_for
// CHECK-NOT: md.template_constraint_epoch
// CHECK: return
func.func @ordinary(%x: !x, %h: !md.cell, %r: !rel) -> !x {
  %a = md.gather_tuples %r, %x, %h coordinates(displacement(1, 0)) {
  ^bb0(%d: vector<3xf64>):
    %z = arith.constant dense<0.0> : vector<3xf64>
    md.yield %d, %z, %z : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !rel, !x -> !x
  return %a : !x
}
