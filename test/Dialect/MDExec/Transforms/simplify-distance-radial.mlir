// RUN: mdir-opt %s --md-exec-simplify-distance=radial=true --canonicalize | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// The force of the direct sum of Ewald, q_i q_j (c1 exp(-β² r²) / r² +
// c2 erfc(β r) / r³): its two terms share the charges, and the rest is a
// function of r² alone, which becomes md_exec.radial of a function of the
// module. A term of Lennard-Jones, a power of r, stays as it is.
//
// CHECK-LABEL: func.func @ewald(
// CHECK:         ^bb0(%[[R2:[a-z0-9]+]]: f64, %{{[a-z0-9]+}}: vector<3xf64>, %[[QI:[a-z0-9]+]]: f64, %[[QJ:[a-z0-9]+]]: f64):
// CHECK:           %[[G:[0-9]+]] = md_exec.radial @md_radial(%[[R2]]) : f64 -> f64
// CHECK-NOT:       math.erfc
// CHECK-NOT:       math.exp
// CHECK:           md_exec.yield
// CHECK:       func.func private @md_radial(%{{[a-z0-9]+}}: f64) -> f64
// CHECK:         math.erfc
// CHECK:         math.exp
func.func @ewald(%x: !vec, %cell: !md.cell, %nl: !nl, %q: !real) -> !vec {
  %f0 = md_exec.zeros : !vec
  %f = md_exec.pair_for %nl, %x, %cell ins(%q : !real) outs(%f0 : !vec)
      cutoff(0.8) exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
    %r = math.sqrt %r2 : f64
    %beta = arith.constant 4.32 : f64
    %mb2 = arith.constant -18.66 : f64
    %c1 = arith.constant 677.8 : f64
    %c2 = arith.constant 138.9 : f64
    %qq = arith.mulf %qi, %qj : f64
    %br = arith.mulf %beta, %r : f64
    %erfc = math.erfc %br : f64
    %k = arith.mulf %mb2, %r2 : f64
    %ex = math.exp %k : f64
    %a = arith.mulf %qq, %ex : f64
    %a1 = arith.mulf %a, %c1 : f64
    %a2 = arith.divf %a1, %r2 : f64
    %b = arith.mulf %qq, %erfc : f64
    %b1 = arith.mulf %b, %c2 : f64
    %r3 = arith.mulf %r2, %r : f64
    %b2 = arith.divf %b1, %r3 : f64
    %c7 = arith.constant 7 : i32
    %lj = arith.constant 2.0e-3 : f64
    %i2 = arith.divf %lj, %r2 : f64
    %i14 = math.fpowi %i2, %c7 : f64, i32
    %s0 = arith.addf %a2, %b2 : f64
    %s = arith.addf %s0, %i14 : f64
    %sv = vector.broadcast %s : f64 to vector<3xf64>
    %force = arith.mulf %sv, %d : vector<3xf64>
    md_exec.yield %force : vector<3xf64>
  } : !nl, !vec -> !vec
  return %f : !vec
}
