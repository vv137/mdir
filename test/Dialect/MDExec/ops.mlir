// RUN: mdir-opt %s | mdir-opt | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>

md.particle_set @atoms

// Forces and energy of a pair potential in one loop.
//
// CHECK-LABEL: md.function @forces(
md.function @forces(%x: !vec, %cell: !md.cell, %a: f64) -> (!vec, f64) {
  // CHECK: %[[CELLS:[0-9]+]] = md_exec.build_cells %{{[a-z0-9]+}}, %{{[a-z0-9]+}} width(2.800000e+00)
  // CHECK-SAME: : !md.field<@atoms, 3 x f64> -> !mdrt.cells<@atoms>
  %cells = md_exec.build_cells %x, %cell width(2.8) : !vec -> !mdrt.cells<@atoms>

  // CHECK: %[[ORDER:[0-9]+]] = md_exec.spatial_order %[[CELLS]]
  // CHECK-SAME: : !mdrt.cells<@atoms> -> !mdrt.permutation<@atoms>
  %order = md_exec.spatial_order %cells
      : !mdrt.cells<@atoms> -> !mdrt.permutation<@atoms>

  // CHECK: %[[XS:[0-9]+]] = md_exec.permute %{{[a-z0-9]+}}, %[[ORDER]]
  // CHECK-SAME: : !md.field<@atoms, 3 x f64>, !mdrt.permutation<@atoms>
  %xs = md_exec.permute %x, %order : !vec, !mdrt.permutation<@atoms>

  // CHECK: %[[NL:[0-9]+]] = md_exec.build_neighbors %[[CELLS]], %[[XS]], %{{[a-z0-9]+}}
  // CHECK-SAME: cutoff(2.500000e+00) skin(3.000000e-01) kind(matrix) width(96)
  // CHECK-SAME: : !mdrt.cells<@atoms>, !md.field<@atoms, 3 x f64> -> !mdrt.neighbors<@atoms>
  %nl = md_exec.build_neighbors %cells, %xs, %cell
      cutoff(2.5) skin(0.3) kind(matrix) width(96)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>

  // CHECK: %[[F0:[0-9]+]] = md_exec.zeros : !md.field<@atoms, 3 x f64>
  %f0 = md_exec.zeros : !vec
  %u0 = arith.constant 0.0 : f64

  // CHECK: %[[RESULT:[0-9]+]]:2 = md_exec.pair_for %[[NL]], %[[XS]], %{{[a-z0-9]+}}
  // CHECK-SAME: outs(%[[F0]] : !md.field<@atoms, 3 x f64>)
  // CHECK-SAME: reduce(%{{[a-z0-9_]+}} : f64)
  // CHECK-SAME: cutoff(2.500000e+00) weights [5.000000e-01]
  // CHECK-SAME: policy(directed, owner_only) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>):
  %f, %u = md_exec.pair_for %nl, %xs, %cell
      outs(%f0 : !vec) reduce(%u0 : f64) cutoff(2.5) weights [0.5]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %g  = arith.divf %a, %r2 : f64
    %gv = vector.broadcast %g : f64 to vector<3xf64>
    %kf = arith.mulf %gv, %d : vector<3xf64>
    // CHECK: md_exec.yield %{{[0-9]+}}, %{{[0-9]+}} : vector<3xf64>, f64
    md_exec.yield %kf, %g : vector<3xf64>, f64
  // CHECK: } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f64>, f64
  } : !mdrt.neighbors<@atoms>, !vec -> !vec, f64

  md.return %f, %u : !vec, f64
}

// A loop over pairs with per-particle parameters.
//
// CHECK-LABEL: md.function @charges(
md.function @charges(%x: !vec, %cell: !md.cell, %q: !real,
                     %nl: !mdrt.neighbors<@atoms>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // CHECK: md_exec.pair_for %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}}
  // CHECK-SAME: ins(%{{[a-z0-9]+}} : !md.field<@atoms, f64>)
  // CHECK-SAME: reduce(
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64):
  %u = md_exec.pair_for %nl, %x, %cell
      ins(%q : !real) reduce(%u0 : f64) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>, %q1: f64, %q2: f64):
    %qq = arith.mulf %q1, %q2 : f64
    %r  = math.sqrt %r2 : f64
    %k  = arith.divf %qq, %r : f64
    md_exec.yield %k : f64
  } : !mdrt.neighbors<@atoms>, !vec -> f64
  md.return %u : f64
}

// A kick, and the kinetic energy.
//
// CHECK-LABEL: md.function @kick(
md.function @kick(%v: !vec, %f: !vec, %m: !real, %dt: f64) -> (!vec, f64) {
  // CHECK: %[[V0:[0-9]+]] = md_exec.empty : !md.field<@atoms, 3 x f64>
  %v0 = md_exec.empty : !vec

  // CHECK: %[[V1:[0-9]+]] = md_exec.particle_for
  // CHECK-SAME: ins(%{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>)
  // CHECK-SAME: outs(%[[V0]] : !md.field<@atoms, 3 x f64>) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64):
  %v1 = md_exec.particle_for ins(%v, %f, %m : !vec, !vec, !real)
      outs(%v0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
    %s  = arith.divf %dt, %m_i : f64
    %sv = vector.broadcast %s : f64 to vector<3xf64>
    %dv = arith.mulf %sv, %f_i : vector<3xf64>
    %vn = arith.addf %v_i, %dv : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  // CHECK: } -> !md.field<@atoms, 3 x f64>
  } -> !vec

  %k0 = arith.constant 0.0 : f64
  // CHECK: %[[KE:[0-9]+]] = md_exec.particle_for
  // CHECK-SAME: reduce(%{{[a-z0-9_]+}} : f64) {
  %ke = md_exec.particle_for ins(%v1, %m : !vec, !real) reduce(%k0 : f64) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %half = arith.constant 0.5 : f64
    %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
    %mv2  = arith.mulf %m_i, %v2 : f64
    %k    = arith.mulf %half, %mv2 : f64
    md_exec.yield %k : f64
  // CHECK: } -> f64
  } -> f64

  // CHECK: md.return %[[V1]], %[[KE]]
  md.return %v1, %ke : !vec, f64
}
