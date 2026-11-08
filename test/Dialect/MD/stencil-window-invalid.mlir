// RUN: mdir-opt %s --split-input-file --md-differentiate-stencils -verify-diagnostics -o /dev/null
func.func @order_jvp(%u: vector<3xf64>, %v: vector<3xf64>) -> f64 {
  // expected-error@+1 {{window derivatives currently require order-four B-splines}}
  %w = md.bspline_window %u order(3) : vector<3xf64>
  %j = md.stencil_jvp %w wrt(0) %v : f64, vector<3xf64> -> f64
  return %j : f64
}
// -----
func.func @order_vjp(%u: vector<3xf64>, %v: f64) -> vector<3xf64> {
  // expected-error@+1 {{window derivatives currently require order-four B-splines}}
  %w = md.bspline_window %u order(3) : vector<3xf64>
  %j = md.stencil_vjp %w wrt(0) %v : f64, f64 -> vector<3xf64>
  return %j : vector<3xf64>
}
