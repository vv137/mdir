// RUN: mdir-opt %s --md-analyze-activity=argument=0 -verify-diagnostics -o /dev/null
// RUN: mdir-opt %s --md-analyze-activity=argument=1 -verify-diagnostics -o /dev/null
// RUN: mdir-opt %s --md-analyze-activity=argument=2 -verify-diagnostics -o /dev/null
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
md.potential @active(%x: !x, %h: !md.cell, %q: !q) -> f64 {
  // The indices are structural; the explicit consumer inputs stay active.
  // expected-remark@+1 {{activity: proven inactive}}
  %s = md.grid_stencil %x, %h : !x -> !s
  // expected-remark@+1 {{activity: active}}
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  // expected-remark@+1 {{activity: proven inactive}}
  %zero = arith.constant 0.0 : f64
  md.return %zero : f64
}
