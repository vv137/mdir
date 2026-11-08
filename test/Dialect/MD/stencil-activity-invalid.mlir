// RUN: mdir-opt %s --split-input-file --md-differentiate-stencils -verify-diagnostics -o /dev/null
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @active_jvp_positions(%x: !x, %h: !md.cell, %q: !q, %g: !g, %dh: vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  // expected-error@+1 {{unsupported stencil JVP for active positions operand}}
  %d = md.stencil_jvp %r wrt(1) %x : !g, !x -> !g
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @active_jvp_cell(%x: !x, %h: !md.cell, %q: !q, %g: !g, %dh: vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  // expected-error@+1 {{unsupported stencil JVP for active cell operand}}
  %d = md.stencil_jvp %r wrt(2) %dh : !g, vector<9xf64> -> !g
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @active_vjp_positions(%x: !x, %h: !md.cell, %q: !q, %g: !g, %dh: vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  // expected-error@+1 {{unsupported stencil VJP for active positions operand}}
  %d = md.stencil_vjp %r wrt(1) %g : !g, !g -> !x
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @active_vjp_cell(%x: !x, %h: !md.cell, %q: !q, %g: !g, %dh: vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  // expected-error@+1 {{unsupported stencil VJP for active cell operand}}
  %d = md.stencil_vjp %r wrt(2) %g : !g, !g -> vector<9xf64>
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @gather_active_jvp_positions(%x: !x, %h: !md.cell, %q: !q, %g: !g, %dh: vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.gather_stencil %s, %x, %h, %g {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !g -> !q
  // expected-error@+1 {{unsupported stencil JVP for active positions operand}}
  %d = md.stencil_jvp %r wrt(1) %x : !q, !x -> !q
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @gather_active_jvp_cell(%x: !x, %h: !md.cell, %q: !q, %g: !g, %dh: vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.gather_stencil %s, %x, %h, %g {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !g -> !q
  // expected-error@+1 {{unsupported stencil JVP for active cell operand}}
  %d = md.stencil_jvp %r wrt(2) %dh : !q, vector<9xf64> -> !q
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @gather_active_vjp_positions(%x: !x, %h: !md.cell, %q: !q, %g: !g, %dh: vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.gather_stencil %s, %x, %h, %g {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !g -> !q
  // expected-error@+1 {{unsupported stencil VJP for active positions operand}}
  %d = md.stencil_vjp %r wrt(1) %q : !q, !q -> !x
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @gather_active_vjp_cell(%x: !x, %h: !md.cell, %q: !q, %g: !g, %dh: vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.gather_stencil %s, %x, %h, %g {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !g -> !q
  // expected-error@+1 {{unsupported stencil VJP for active cell operand}}
  %d = md.stencil_vjp %r wrt(2) %q : !q, !q -> vector<9xf64>
  return
}
// -----
