// RUN: mdir-opt %s --md-differentiate-grid --verify-each | FileCheck %s --check-prefix=AD --implicit-check-not=md.grid_jvp --implicit-check-not=md.grid_vjp
// RUN: mdir-opt %s --md-differentiate-grid --md-expand-grid-kernels --canonicalize --verify-each | FileCheck %s --check-prefix=LOWER --implicit-check-not=md.map_grid --implicit-check-not=md.sum_grid
// RUN: mdir-opt %s --canonicalize | FileCheck %s --check-prefix=EFFECT
!g = tensor<2x3x4xf64>
!s = tensor<2x3x3xcomplex<f64>>

// AD-LABEL: func.func @real
// AD: md.map_grid
// AD: md.map_grid
// AD: md.map_grid
// AD: md.sum_grid
// LOWER-LABEL: func.func @real
// LOWER: tensor.generate
// LOWER: scf.for
func.func @real(%input: !g, %dx: !g, %bar: !g, %scale: f64) -> (!g, !g, !g, f64) {
  %v = md.map_grid (%input, %scale) grid([2, 3, 4]) pairing("euclidean") : (!g, f64) -> !g {
  ^bb0(%index: vector<3xindex>, %x: f64, %a: f64):
    %s = arith.mulf %x, %x : f64
    %y = arith.mulf %a, %s : f64
    md.yield %y : f64
  }
  %j = md.grid_jvp %v wrt(0) %dx : !g, !g -> !g
  %b = md.grid_vjp %v wrt(0) %bar : !g, !g -> !g
  %ab = md.grid_vjp %v wrt(1) %bar : !g, !g -> f64
  return %v, %j, %b, %ab : !g, !g, !g, f64
}

// Complex arguments are real components. The reduction uses the Hermitian
// real pairing, not a complex bilinear transpose.
// AD-LABEL: func.func @half
// AD: md.sum_grid
// AD: md.sum_grid
// AD: md.map_grid
// LOWER-LABEL: func.func @half
// LOWER: scf.for
// LOWER: arith.select
func.func @half(%input: !s, %dz: !s, %seed: f64) -> (f64, f64, !s) {
  %u = md.sum_grid (%input) grid([2, 3, 4]) pairing("real_hermitian_half") : (!s) -> f64 {
  ^bb0(%index: vector<3xindex>, %z: vector<2xf64>):
    %square = arith.mulf %z, %z : vector<2xf64>
    %norm = vector.reduction <add>, %square : vector<2xf64> into f64
    %half = arith.constant 0.5 : f64
    %u = arith.mulf %half, %norm : f64
    md.yield %u : f64
  }
  %j = md.grid_jvp %u wrt(0) %dz : f64, !s -> f64
  %b = md.grid_vjp %u wrt(0) %seed : f64, f64 -> !s
  return %u, %j, %b : f64, f64, !s
}

// EFFECT-LABEL: func.func @effects
// EFFECT-NOT: md.map_grid
// EFFECT-NOT: md.sum_grid
// EFFECT: return
func.func @effects(%input: !g) {
  %u = md.sum_grid (%input) grid([2, 3, 4]) pairing("euclidean") : (!g) -> f64 {
  ^bb0(%index: vector<3xindex>, %x: f64):
    md.yield %x : f64
  }
  return
}
