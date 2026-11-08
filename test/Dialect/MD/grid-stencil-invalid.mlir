// RUN: mdir-opt %s --split-input-file -verify-diagnostics -o /dev/null
md.particle_set @atoms
md.particle_set @other
func.func @wrong_set(%x: !md.field<@other, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{stencil and positions must name the same particle set}}
  %s = md.grid_stencil %x, %cell : !md.field<@other, 3 x f64> -> !md.grid_stencil<@atoms, [4, 4, 4], order(4), window("bspline")>
  return
}
// -----
md.particle_set @atoms
func.func @wrong_components(%x: !md.field<@atoms, f64>, %cell: !md.cell) {
  // expected-error@+1 {{expected positions with three f64 components}}
  %s = md.grid_stencil %x, %cell : !md.field<@atoms, f64> -> !md.grid_stencil<@atoms, [4, 4, 4], order(4), window("bspline")>
  return
}
// -----
func.func @unknown_set(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{expected a declared particle set for the stencil}}
  %s = md.grid_stencil %x, %cell : !md.field<@atoms, 3 x f64> -> !md.grid_stencil<@atoms, [4, 4, 4], order(4), window("bspline")>
  return
}
// -----
// expected-error@+1 {{expected three grid stencil extents}}
func.func @rank(%s: !md.grid_stencil<@atoms, [4, 4], order(4), window("bspline")>) { return }
// -----
// expected-error@+1 {{expected grid stencil window 'bspline'}}
func.func @window(%s: !md.grid_stencil<@atoms, [4, 4, 4], order(4), window("unknown")>) { return }
// -----
// expected-error@+1 {{expected grid stencil order from 2 through 8}}
func.func @order(%s: !md.grid_stencil<@atoms, [16, 16, 16], order(9), window("bspline")>) { return }
// -----
// expected-error@+1 {{expected each grid stencil extent from its order through the 32-bit limit}}
func.func @extent(%s: !md.grid_stencil<@atoms, [2, 4, 4], order(4), window("bspline")>) { return }
// -----
// expected-error@+1 {{grid stencil point count overflows i64}}
func.func @overflow(%s: !md.grid_stencil<@atoms, [2147483647, 2147483647, 2147483647], order(4), window("bspline")>) { return }
