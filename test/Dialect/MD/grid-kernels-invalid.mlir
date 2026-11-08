// RUN: mdir-opt %s --split-input-file --verify-diagnostics
func.func @shape(%input: tensor<2x3x4xf64>) {
  // expected-error @+1 {{tensor input must have the complete declared domain shape}}
  %u = md.sum_grid (%input) grid([2, 3, 4]) pairing("real_hermitian_half") : (tensor<2x3x4xf64>) -> f64 {
  ^bb0(%i: vector<3xindex>, %x: f64):
    md.yield %x : f64
  }
  return
}
// -----
func.func @bounds(%input: f64) {
  // expected-error @+1 {{grid extents must fit i32 and their product signed i64}}
  %u = md.sum_grid (%input) grid([2147483647, 2147483647, 2147483647]) pairing("euclidean") : (f64) -> f64 {
  ^bb0(%i: vector<3xindex>, %x: f64):
    md.yield %x : f64
  }
  return
}
// -----
func.func @capture(%input: f64, %a: f64) {
  // expected-error @+1 {{grid kernel cannot capture dynamic values}}
  %u = md.sum_grid (%input) grid([2, 3, 4]) pairing("euclidean") : (f64) -> f64 {
  ^bb0(%i: vector<3xindex>, %x: f64):
    %y = arith.mulf %x, %a : f64
    md.yield %y : f64
  }
  return
}
// -----
func.func private @opaque(f64) -> f64
func.func @effect(%input: f64) {
  // expected-error @+1 {{grid kernel must be straight-line and effect-free}}
  %u = md.sum_grid (%input) grid([2, 3, 4]) pairing("euclidean") : (f64) -> f64 {
  ^bb0(%i: vector<3xindex>, %x: f64):
    %y = func.call @opaque(%x) : (f64) -> f64
    md.yield %y : f64
  }
  return
}
// -----
func.func @coverage(%input: f64) {
  // expected-error @+1 {{full coverage and ideal ordered sum cannot be overridden}}
  %u = md.sum_grid (%input) grid([2, 3, 4]) pairing("euclidean") {coverage = "partial"} : (f64) -> f64 {
  ^bb0(%i: vector<3xindex>, %x: f64):
    md.yield %x : f64
  }
  return
}
// -----
func.func @seed(%input: tensor<2x3x4xf64>, %bad: f64) {
  %v = md.map_grid (%input) grid([2, 3, 4]) pairing("euclidean") : (tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  ^bb0(%i: vector<3xindex>, %x: f64):
    md.yield %x : f64
  }
  // expected-error @+1 {{grid derivative seed or result has the wrong operand shape}}
  %j = md.grid_jvp %v wrt(0) %bad : tensor<2x3x4xf64>, f64 -> tensor<2x3x4xf64>
  return
}
