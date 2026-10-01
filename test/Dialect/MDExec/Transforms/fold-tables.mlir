// RUN: mdir-opt %s --md-exec-fold-tables | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>
!t   = !md.table<2, f64, symmetric>

md.particle_set @atoms

// The force of Lennard-Jones from the tables of σ and ε: 48 ε σ¹² and
// 24 ε σ⁶ depend on the pair of types only, and become two tables, which
// the force reads as one table of vectors, with one lookup; the loop of the
// energy takes the table of 48 ε σ¹² alone.
//
// CHECK-LABEL: func.func @lennard_jones(
// CHECK:         %[[S:[0-9]+]] = mdrt.from_buffer %{{.*}} to !md.table<2, f64, symmetric>
// CHECK:         %[[E:[0-9]+]] = mdrt.from_buffer %{{.*}} to !md.table<2, f64, symmetric>
// CHECK:         %[[A:[0-9]+]] = md_exec.tabulate(%[[S]], %[[E]] :
// CHECK:           arith.constant 4.800000e+01
// CHECK:         } -> !md.table<2, f64, symmetric>
// CHECK:         %[[AB:[0-9]+]] = md_exec.tabulate(%[[S]], %[[E]] :
// CHECK:           arith.constant 4.800000e+01
// CHECK:           arith.constant 2.400000e+01
// CHECK:           vector.from_elements
// CHECK:         } -> !md.table<2, vector<2xf64>, symmetric>
// CHECK-NOT:     md_exec.tabulate
// CHECK:         md_exec.pair_for
// CHECK:           %[[V:[0-9]+]] = md.lookup %[[AB]][%{{.*}}, %{{.*}}] {{.*}} -> vector<2xf64>
// CHECK:           vector.extract %[[V]][0]
// CHECK:           vector.extract %[[V]][1]
// CHECK-NOT:       md.lookup
// CHECK:         md_exec.pair_for
// CHECK:           md.lookup %[[A]]
func.func @lennard_jones(%x: !vec, %cell: !md.cell, %nl: !nl,
                         %types: !md.field<@atoms, i32>,
                         %sb: memref<?x?xf64>, %eb: memref<?x?xf64>)
    -> (!vec, f64) {
  %sigma = mdrt.from_buffer %sb : memref<?x?xf64> to !t
  %epsilon = mdrt.from_buffer %eb : memref<?x?xf64> to !t
  %f0 = md_exec.zeros : !vec
  %f = md_exec.pair_for %nl, %x, %cell ins(%types : !md.field<@atoms, i32>)
      outs(%f0 : !vec) cutoff(2.5) exchange [antisymmetric]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>, %ti: i32, %tj: i32):
    %s = md.lookup %sigma[%ti, %tj] : !t, i32, i32 -> f64
    %e = md.lookup %epsilon[%ti, %tj] : !t, i32, i32 -> f64
    %c48 = arith.constant 48.0 : f64
    %c24 = arith.constant 24.0 : f64
    %s2 = arith.mulf %s, %s : f64
    %s6a = arith.mulf %s2, %s2 : f64
    %s6 = arith.mulf %s6a, %s2 : f64
    %s12 = arith.mulf %s6, %s6 : f64
    %e48 = arith.mulf %e, %c48 : f64
    %a = arith.mulf %e48, %s12 : f64
    %e24 = arith.mulf %e, %c24 : f64
    %b = arith.mulf %e24, %s6 : f64
    %i2 = arith.divf %c24, %r2 : f64
    %i6a = arith.mulf %i2, %i2 : f64
    %i6 = arith.mulf %i6a, %i2 : f64
    %ta = arith.mulf %a, %i6 : f64
    %tb = arith.subf %ta, %b : f64
    %g = arith.mulf %tb, %i6 : f64
    %gv = vector.broadcast %g : f64 to vector<3xf64>
    %k = arith.mulf %gv, %d : vector<3xf64>
    md_exec.yield %k : vector<3xf64>
  } : !nl, !vec -> !vec
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell ins(%types : !md.field<@atoms, i32>)
      reduce(%u0 : f64) cutoff(2.5) weights [0.5] exchange [symmetric]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>, %ti: i32, %tj: i32):
    %s = md.lookup %sigma[%ti, %tj] : !t, i32, i32 -> f64
    %e = md.lookup %epsilon[%ti, %tj] : !t, i32, i32 -> f64
    %c48 = arith.constant 48.0 : f64
    %s2 = arith.mulf %s, %s : f64
    %s6a = arith.mulf %s2, %s2 : f64
    %s6 = arith.mulf %s6a, %s2 : f64
    %s12 = arith.mulf %s6, %s6 : f64
    %e48 = arith.mulf %e, %c48 : f64
    %a = arith.mulf %e48, %s12 : f64
    %u = arith.divf %a, %r2 : f64
    md_exec.yield %u : f64
  } : !nl, !vec -> f64
  return %f, %u : !vec, f64
}
