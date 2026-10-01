// RUN: mdir-opt %s --md-exec-expand-radial | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

func.func private @md_radial(%s: f64) -> f64 {
  %r = math.sqrt %s : f64
  %beta = arith.constant 4.3236384215993748 : f64
  %br = arith.mulf %beta, %r : f64
  %e = math.erfc %br : f64
  %q = arith.divf %e, %r : f64
  return %q : f64
}

// In f64 the function is inlined, and no table is made.
//
// CHECK-NOT:   memref.global
// CHECK-LABEL: func.func @double(
// CHECK:         ^bb0(%[[R2:[a-z0-9]+]]: f64,
// CHECK:           %[[R:[0-9]+]] = math.sqrt %[[R2]] : f64
// CHECK:           math.erfc
// CHECK-NOT:       md_exec.radial
func.func @double(%x: !vec, %cell: !md.cell, %nl: !nl) -> f64 {
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(0.8)
      weights [0.5] exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %g = md_exec.radial @md_radial(%r2) : f64 -> f64
    md_exec.yield %g : f64
  } : !nl, !vec -> f64
  return %u : f64
}
