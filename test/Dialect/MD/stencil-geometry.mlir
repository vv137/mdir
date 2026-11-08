// RUN: mdir-opt %s --md-differentiate-stencils=geometry=true --verify-each | FileCheck %s --implicit-check-not=md.stencil_jvp --implicit-check-not=md.stencil_vjp
// CHECK-COUNT-8: md.reduce_stencil
// The opt-in geometry rules retain the same common weight body and relation.
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @geometry(%x: !x, %h: !md.cell, %q: !q, %dx: !x, %dh: vector<9xf64>, %grid: !g, %seed: !q) -> (!g, !g, !x, vector<9xf64>, !q, !q, !x, vector<9xf64>) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %rho = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %weight = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %weight : f64
  } : !s, !x, !q -> !g
  %p = md.gather_stencil %s, %x, %h, %grid {
  ^bb0(%u: vector<3xf64>):
    %weight = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %weight : f64
  } : !s, !x, !g -> !q
  %sjx = md.stencil_jvp %rho wrt(1) %dx : !g, !x -> !g
  %sjh = md.stencil_jvp %rho wrt(2) %dh : !g, vector<9xf64> -> !g
  %svx = md.stencil_vjp %rho wrt(1) %grid : !g, !g -> !x
  %svh = md.stencil_vjp %rho wrt(2) %grid : !g, !g -> vector<9xf64>
  %gjx = md.stencil_jvp %p wrt(1) %dx : !q, !x -> !q
  %gjh = md.stencil_jvp %p wrt(2) %dh : !q, vector<9xf64> -> !q
  %gvx = md.stencil_vjp %p wrt(1) %seed : !q, !q -> !x
  %gvh = md.stencil_vjp %p wrt(2) %seed : !q, !q -> vector<9xf64>
  return %sjx, %sjh, %svx, %svh, %gjx, %gjh, %gvx, %gvh : !g, !g, !x, vector<9xf64>, !q, !q, !x, vector<9xf64>
}
