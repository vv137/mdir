// RUN: mdir-opt %s --md-differentiate-stencils --verify-each -o - | FileCheck %s --implicit-check-not=md.stencil_jvp --implicit-check-not=md.stencil_vjp
// RUN: mdir-opt %s --test-md-transpose --verify-each -o - 2>&1 | FileCheck %s --check-prefix=TRANSPOSE
// TRANSPOSE: remark: transpose in operand 3: 'md.gather_stencil'
// TRANSPOSE: remark: transpose in operand 3: 'md.scatter_stencil'
// Both linear directions clone the same weight kernel; no request remains.
// CHECK-NOT: md.stencil_jvp
// CHECK-NOT: md.stencil_vjp
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
func.func @linear(%x: !x, %h: !md.cell, %q: !q, %dq: !q, %g: !g) -> !q {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } {test.transpose = 3 : i32} : !s, !x, !q -> !g
  // CHECK: md.scatter_stencil
  %j = md.stencil_jvp %r wrt(3) %dq : !g, !q -> !g
  // CHECK: md.gather_stencil
  %v = md.stencil_vjp %r wrt(3) %g : !g, !g -> !q
  %p = md.gather_stencil %s, %x, %h, %g {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } {test.transpose = 3 : i32} : !s, !x, !g -> !q
  %pj = md.stencil_jvp %p wrt(3) %g : !q, !g -> !q
  %pv = md.stencil_vjp %p wrt(3) %q : !q, !q -> !g
  return %v : !q
}

// The weight body has no effects; an unused fresh reduction is removable.
// RUN: mdir-opt %s --md-differentiate-stencils --canonicalize -o - | FileCheck %s --check-prefix=EFFECT
// EFFECT-LABEL: func.func @effects
// EFFECT-NOT: md.scatter_stencil
// EFFECT-NOT: md.grid_stencil
// EFFECT: return
func.func @effects(%x: !x, %h: !md.cell, %q: !q) {
  %s = md.grid_stencil %x, %h : !x -> !s
  %r = md.scatter_stencil %s, %x, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  return
}
