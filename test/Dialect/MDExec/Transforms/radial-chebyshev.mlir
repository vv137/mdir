// A Chebyshev series of the distance (D[prolate-pair-table]): the direct
// sum of the prolate kernel, f q_i q_j (1 - x) P(t) / r with x = r / r_c and
// t = 2x - 1, as the builder writes it (the first terms of P at 1e-5).
//
// md-exec-simplify-distance counts the series as costly and takes the
// product into one function of r^2:
// RUN: mdir-opt %s --md-exec-simplify-distance=radial=true --canonicalize | FileCheck %s --check-prefix=SIMPLIFY
//
// md-exec-expand-radial makes such a function a table in a kernel of f32
// (@single, as precision leaves it), and inlines it in one of f64, the
// series as Clenshaw's recurrence:
// RUN: mdir-opt %s --md-exec-simplify-distance=radial=true --canonicalize \
// RUN:   --md-exec-expand-radial | FileCheck %s --check-prefix=EXPAND

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// SIMPLIFY-LABEL: func.func @prolate(
// SIMPLIFY:         ^bb0(%[[R2:[a-z0-9]+]]: f64, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64):
// SIMPLIFY:           md_exec.radial @md_radial(%[[R2]]) : f64 -> f64
// SIMPLIFY-NOT:       md.chebyshev
// SIMPLIFY-NOT:       math.sqrt
// SIMPLIFY:           md_exec.yield
// SIMPLIFY:       func.func private @md_radial(%{{[a-z0-9]+}}: f64) -> f64
// SIMPLIFY:         math.sqrt
// SIMPLIFY:         md.chebyshev %{{.*}} [0.318476{{[0-9]*}}, -0.487631{{[0-9]*}}, 0.2427165] : f64

// EXPAND:       memref.global "private" constant @md_radial_series_table : memref<{{[0-9]+}}x4xf32>
// EXPAND-NOT:   md.chebyshev
// EXPAND-LABEL: func.func @prolate(
// EXPAND:         md_exec.pair_for
// EXPAND-NOT:       md_exec.radial
// EXPAND:           arith.addf
// EXPAND:           arith.mulf
// EXPAND:           arith.subf
// EXPAND:           md_exec.yield
// EXPAND-LABEL: func.func @single(
// EXPAND:         md_exec.pair_for
// EXPAND:           vector.load %{{.*}} : memref<{{[0-9]+}}x4xf32>, vector<4xf32>
// EXPAND-NOT:       md_exec.radial
func.func @prolate(%x: !vec, %cell: !md.cell, %nl: !nl, %q: !real) -> f64 {
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell ins(%q : !real) reduce(%u0 : f64)
      cutoff(0.9) weights [0.5] exchange [symmetric]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
    %r = math.sqrt %r2 : f64
    %f = arith.constant 138.935 : f64
    %qq = arith.mulf %qi, %qj : f64
    %fqq = arith.mulf %f, %qq : f64
    %rc_inverse = arith.constant 1.1111111111111112 : f64
    %xr = arith.mulf %r, %rc_inverse : f64
    %two = arith.constant 2.0 : f64
    %one = arith.constant 1.0 : f64
    %twice = arith.mulf %two, %xr : f64
    %t = arith.subf %twice, %one : f64
    %mean = md.chebyshev %t [0.3184762, -0.4876314, 0.2427165] : f64
    %rest = arith.subf %one, %xr : f64
    %far = arith.mulf %rest, %mean : f64
    %screened = arith.divf %far, %r : f64
    %coulomb = arith.mulf %fqq, %screened : f64
    md_exec.yield %coulomb : f64
  } : !nl, !vec -> f64
  return %u : f64
}

// The function of @prolate after md-exec-simplify-distance, read by a
// kernel of f32.
func.func private @md_radial_series(%s: f64) -> f64 {
  %f = arith.constant 138.935 : f64
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %k = arith.constant 1.1111111111111112 : f64
  %r = math.sqrt %s : f64
  %x = arith.mulf %r, %k : f64
  %twice = arith.mulf %two, %x : f64
  %t = arith.subf %twice, %one : f64
  %mean = md.chebyshev %t [0.3184762, -0.4876314, 0.2427165] : f64
  %rest = arith.subf %one, %x : f64
  %far = arith.mulf %rest, %mean : f64
  %fr = arith.mulf %f, %far : f64
  %g = arith.divf %fr, %r : f64
  return %g : f64
}

func.func @single(%x: !md.field<@atoms, 3 x f32>, %cell: !md.cell, %nl: !nl)
    -> f64 {
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(0.9)
      weights [0.5] exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %g = md_exec.radial @md_radial_series(%r2) : f32 -> f32
    %w = arith.extf %g : f32 to f64
    md_exec.yield %w : f64
  } : !nl, !md.field<@atoms, 3 x f32> -> f64
  return %u : f64
}
