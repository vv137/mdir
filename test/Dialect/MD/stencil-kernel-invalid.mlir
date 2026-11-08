// RUN: mdir-opt %s --split-input-file -verify-diagnostics -o /dev/null
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @shape(%x: !x, %other: !x, %h: !md.cell, %q: !q, %outside: f64, %buffer: memref<1xf64>, %zero: index) {
  %s = md.grid_stencil %x, %h : !x -> !s
  // expected-error@+1 {{expected a scalar f64 particle field and the full stencil-shaped f64 grid}}
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> tensor<4x5x5xf64>
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @order(%x: !x, %other: !x, %h: !md.cell, %q: !q, %outside: f64, %buffer: memref<1xf64>, %zero: index) {
  %s = md.grid_stencil %x, %h : !x -> !s
  // expected-error@+1 {{window and stencil support orders must agree}}
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(3) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @capture(%x: !x, %other: !x, %h: !md.cell, %q: !q, %outside: f64, %buffer: memref<1xf64>, %zero: index) {
  %s = md.grid_stencil %x, %h : !x -> !s
  // expected-error@+1 {{window kernel cannot capture values}}
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    %captured = arith.mulf %w, %outside : f64
    md.yield %captured : f64
  } : !s, !x, !q -> !g
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @effect(%x: !x, %other: !x, %h: !md.cell, %q: !q, %outside: f64, %buffer: memref<1xf64>, %zero: index) {
  %s = md.grid_stencil %x, %h : !x -> !s
  // expected-error@+1 {{window kernel must be straight-line and effect-free}}
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    memref.store %w, %buffer[%zero] : memref<1xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @coverage(%x: !x, %other: !x, %h: !md.cell, %q: !q, %outside: f64, %buffer: memref<1xf64>, %zero: index) {
  %s = md.grid_stencil %x, %h : !x -> !s
  // expected-error@+1 {{full coverage and ideal additive algebra cannot be overridden}}
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } {coverage = "partial"} : !s, !x, !q -> !g
  return
}
// -----
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @provenance(%x: !x, %other: !x, %h: !md.cell, %q: !q, %outside: f64, %buffer: memref<1xf64>, %zero: index) {
  %s = md.grid_stencil %x, %h : !x -> !s
  // expected-error@+1 {{expected the relation constructor's exact positions and cell operands}}
  %r = md.scatter_stencil %s, %other, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  return
}
// -----
