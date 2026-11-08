// RUN: mdir-opt %s --split-input-file --md-check-transposes --md-check-derivative-coverage --md-differentiate -verify-diagnostics -o - | FileCheck %s
// RUN: mdir-opt %s --split-input-file --md-expand-stencil-windows --canonicalize --md-check-transposes --md-check-derivative-coverage --md-differentiate -verify-diagnostics -o - | FileCheck %s
// No active cell dependency may become a zero by omission. The first
// chunk uses a different position field, so the primary-position guard
// alone cannot protect the ordinary virial request.
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
md.potential @mesh(%x: !x, %h: !md.cell, %y: !x, %q: !q) -> f64 {
  %s = md.grid_stencil %y, %h : !x -> !s
  // expected-error@+1 {{unsupported stencil VJP for active cell operand in ordinary virial composition}}
  %r = md.scatter_stencil %s, %y, %h, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  %p = md.gather_stencil %s, %y, %h, %r {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !g -> !q
  %u = md.sum_particles gather(%p : !q) {
  ^bb0(%pi: f64):
    md.yield %pi : f64
  } : f64
  md.return %u : f64
}
md.function @caller(%x: !x, %h: !md.cell, %y: !x, %q: !q) -> vector<9xf64> {
  %w = md.evaluate @mesh(%x, %h, %y, %q) request [virial]
       : (!x, !md.cell, !x, !q) -> vector<9xf64>
  md.return %w : vector<9xf64>
}

// -----
// A genuinely cell-independent composition retains the proven zero.
// CHECK-LABEL: md.function @mesh_frozen.virial
// CHECK: arith.constant dense<0.000000e+00> : vector<9xf64>
// CHECK: md.return
!s = !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!g = tensor<4x5x6xf64>
md.particle_set @atoms
md.potential @mesh_frozen(%x: !x, %h: !md.cell, %y: !x, %q: !q) -> f64 {
  %length = arith.constant 10.0 : f64
  %fixed = md.orthorhombic_cell %length, %length, %length
  %s = md.grid_stencil %y, %fixed : !x -> !s
  %r = md.scatter_stencil %s, %y, %fixed, %q {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !q -> !g
  %p = md.gather_stencil %s, %y, %fixed, %r {
  ^bb0(%u: vector<3xf64>):
    %w = md.bspline_window %u order(4) : vector<3xf64>
    md.yield %w : f64
  } : !s, !x, !g -> !q
  %u = md.sum_particles gather(%p : !q) {
  ^bb0(%pi: f64):
    md.yield %pi : f64
  } : f64
  md.return %u : f64
}
md.function @caller_frozen(%x: !x, %h: !md.cell, %y: !x, %q: !q) -> vector<9xf64> {
  %w = md.evaluate @mesh_frozen(%x, %h, %y, %q) request [virial]
       : (!x, !md.cell, !x, !q) -> vector<9xf64>
  md.return %w : vector<9xf64>
}

// -----
// The common scalar window must not bypass the ordinary cell refusal either.
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
md.particle_set @atoms
md.potential @scalar_window(%x: !x, %h: !md.cell, %q: !q) -> f64 {
  %edges = md_exec.cell_edges %h : vector<3xf64>
  %u = md.sum_particles gather(%q : !q) {
  ^bb0(%qi: f64):
    // expected-error@+1 {{unsupported stencil VJP for active cell operand in ordinary virial composition}}
    %w = md.bspline_window %edges order(4) : vector<3xf64>
    %weighted = arith.mulf %qi, %w : f64
    md.yield %weighted : f64
  } : f64
  md.return %u : f64
}
md.function @scalar_caller(%x: !x, %h: !md.cell, %q: !q) -> vector<9xf64> {
  %w = md.evaluate @scalar_window(%x, %h, %q) request [virial]
      : (!x, !md.cell, !q) -> vector<9xf64>
  md.return %w : vector<9xf64>
}
