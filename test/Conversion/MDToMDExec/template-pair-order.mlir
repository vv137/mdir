// RUN: mdir-opt %s --test-md-exec-template-windows --verify-each | FileCheck %s
// Force pairs are added in an order where consecutive pairs reach different
// sites where one exists: pairs 0 and 1 both add to sites 0 and 3, pair 2
// to sites 1 and 4, so the guards go 0, 2, 1. A reduction column keeps the
// source order 0, 1, 2. The scale of each pair (2, 3, 5) names it.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
// CHECK-LABEL: func.func @window_forces
// CHECK: scf.if
// CHECK: arith.constant 2.000000e+00
// CHECK: } else {
// CHECK: scf.if
// CHECK: arith.constant 5.000000e+00
// CHECK: } else {
// CHECK: scf.if
// CHECK: arith.constant 3.000000e+00
// CHECK: } else {
// CHECK: return
// CHECK-LABEL: func.func @window_energy
// CHECK: scf.if
// CHECK: arith.constant 2.000000e+00
// CHECK: } else {
// CHECK: scf.if
// CHECK: arith.constant 3.000000e+00
// CHECK: } else {
// CHECK: scf.if
// CHECK: arith.constant 5.000000e+00
// CHECK: } else {
// CHECK: return
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell) -> (!x, f64) {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [3, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %d = arith.subf %base, %b1 : vector<3xf64>
    %e = arith.subf %base, %b2 : vector<3xf64>
    %g = arith.subf %a1, %b1 : vector<3xf64>
    %two = arith.constant 2.0 : f64
    %three = arith.constant 3.0 : f64
    %five = arith.constant 5.0 : f64
    %s0 = vector.broadcast %two : f64 to vector<3xf64>
    %s1 = vector.broadcast %three : f64 to vector<3xf64>
    %s2 = vector.broadcast %five : f64 to vector<3xf64>
    %t0 = arith.mulf %s0, %d : vector<3xf64>
    %t1 = arith.mulf %s1, %e : vector<3xf64>
    %t2 = arith.mulf %s2, %g : vector<3xf64>
    %n0 = arith.negf %t0 : vector<3xf64>
    %n1 = arith.negf %t1 : vector<3xf64>
    %n2 = arith.negf %t2 : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %t0, %z, %z, %n0, %z, %z,
                  %t1, %z, %z, %n1, %z, %z,
                  %z, %t2, %z, %z, %n2, %z,
                  %d, %e, %g
      : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>,
        vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>,
        vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>,
        vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%q0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.template_for %pairs, %x, %h reduce(%u0 : f64) pairs [3, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %d = arith.subf %base, %b1 : vector<3xf64>
    %e = arith.subf %base, %b2 : vector<3xf64>
    %g = arith.subf %a1, %b1 : vector<3xf64>
    %two = arith.constant 2.0 : f64
    %three = arith.constant 3.0 : f64
    %five = arith.constant 5.0 : f64
    %dd = arith.mulf %d, %d : vector<3xf64>
    %ee = arith.mulf %e, %e : vector<3xf64>
    %gg = arith.mulf %g, %g : vector<3xf64>
    %r0 = vector.reduction <add>, %dd : vector<3xf64> into f64
    %r1 = vector.reduction <add>, %ee : vector<3xf64> into f64
    %r2 = vector.reduction <add>, %gg : vector<3xf64> into f64
    %u0p = arith.mulf %two, %r0 : f64
    %u1p = arith.mulf %three, %r1 : f64
    %u2p = arith.mulf %five, %r2 : f64
    md_exec.yield %u0p, %u1p, %u2p, %d, %e, %g : f64, f64, f64, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%q0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> f64
  func.return %f, %u : !x, f64
}
