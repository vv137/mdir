// md-exec-assign-algebra (D[reduction-algebra]) on the loops over pairs and
// tuples: their destinations, then their sums (the storage form of these
// loops is covered by the tests of the lowerings, which refuse an op
// without an algebra).
//
// RUN: mdir-opt %s --md-exec-assign-algebra | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true \
// RUN: | FileCheck %s --check-prefix=DET

!vec = !md.field<@atoms, 3 x f32>
!nl = !mdrt.neighbors<@atoms>
!bonds = !mdrt.incidence<@atoms, 2, @bonds>
md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(ordered)

// A loop over pairs: its destination, then its sum. A loop over particles
// without a sum states nothing.
// CHECK-LABEL: func.func @pairs(
// CHECK: md_exec.particle_for
// CHECK-NOT: md_exec.algebra
// CHECK: } {md_exec.algebra = #md_exec.algebra<any_order, any_order>}
// DET-LABEL: func.func @pairs(
// DET: } {md_exec.algebra = #md_exec.algebra<fixed_order, fixed_order>}
func.func @pairs(%x: !vec, %cell: !md.cell, %nl: !nl) -> (!vec, f64) {
  %f0 = md_exec.zeros : !vec
  %y = md_exec.particle_for ins(%x : !vec) outs(%f0 : !vec) {
  ^bb0(%x_i: vector<3xf32>):
    md_exec.yield %x_i : vector<3xf32>
  } -> !vec
  %u0 = arith.constant 0.0 : f64
  %f, %u = md_exec.pair_for %nl, %y, %cell outs(%f0 : !vec) reduce(%u0 : f64)
      cutoff(1.0) policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_exec.yield %d, %e : vector<3xf32>, f64
  } : !nl, !vec -> !vec, f64
  return %f, %u : !vec, f64
}

// A loop over tuples.
// CHECK-LABEL: func.func @bond_forces(
// CHECK: } {md_exec.algebra = #md_exec.algebra<any_order>}
// DET-LABEL: func.func @bond_forces(
// DET: } {md_exec.algebra = #md_exec.algebra<fixed_order>}
func.func @bond_forces(%inc: !bonds, %x: !vec, %cell: !md.cell) -> !vec {
  %f0 = md_exec.zeros : !vec
  %f = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         outs(%f0 : !vec) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n : vector<3xf32>, vector<3xf32>
  } : !bonds, !vec -> !vec
  return %f : !vec
}
