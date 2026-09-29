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

// A neighbor structure that is refreshed.
//
// CHECK-LABEL: md.function @refresh(
md.function @refresh(%x: !vec, %cell: !md.cell) -> i64 {
  // CHECK: %[[NL0:[0-9]+]] = md_exec.empty_neighbors kind(matrix) width(96) : !mdrt.neighbors<@atoms>
  %nl0 = md_exec.empty_neighbors kind(matrix) width(96)
      : !mdrt.neighbors<@atoms>

  // CHECK: %[[NL1:[0-9]+]] = md_exec.refresh_neighbors %[[NL0]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}}
  // CHECK-SAME: cutoff(2.500000e+00) skin(3.000000e-01) cell_width(2.800000e+00) policy(check)
  // CHECK-SAME: : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  %nl1 = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
      : !mdrt.neighbors<@atoms>, !vec

  // CHECK: %[[BUILDS:[0-9]+]] = md_exec.rebuild_count %[[NL1]] : !mdrt.neighbors<@atoms>
  %builds = md_exec.rebuild_count %nl1 : !mdrt.neighbors<@atoms>
  md.return %builds : i64
}

// The test of validity as a loop over particles, which hands its result to
// the refresh.
//
// CHECK-LABEL: func.func @moved(
func.func @moved(%x: !vec, %cell: !md.cell, %nl: !mdrt.neighbors<@atoms>)
    -> !mdrt.neighbors<@atoms> {
  // CHECK: %[[REF:[0-9]+]] = md_exec.reference_positions %{{[a-z0-9]+}} : !mdrt.neighbors<@atoms> -> !md.field<@atoms, 3 x f64>
  %ref = md_exec.reference_positions %nl
      : !mdrt.neighbors<@atoms> -> !vec

  %no = arith.constant false
  // CHECK: %[[MOVED:[0-9]+]] = md_exec.particle_for
  // CHECK-SAME: ins(%{{[a-z0-9]+}}, %[[REF]] : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>)
  // CHECK-SAME: reduce(%{{[a-z0-9_]+}} : i1) {
  %moved = md_exec.particle_for ins(%x, %ref : !vec, !vec)
      reduce(%no : i1) {
  ^bb0(%x_i: vector<3xf64>, %ref_i: vector<3xf64>):
    %limit = arith.constant 0.0225 : f64
    %d = arith.subf %x_i, %ref_i : vector<3xf64>
    %sq = arith.mulf %d, %d : vector<3xf64>
    %d2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %far = arith.cmpf ugt, %d2, %limit : f64
    md_exec.yield %far : i1
  // CHECK: } -> i1
  } -> i1

  // CHECK: md_exec.refresh_neighbors %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} moved(%[[MOVED]])
  // CHECK-SAME: cutoff(2.500000e+00) skin(3.000000e-01) cell_width(2.800000e+00) policy(check)
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved)
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
      : !mdrt.neighbors<@atoms>, !vec
  return %nl1 : !mdrt.neighbors<@atoms>
}

// The same in the storage form.
//
// CHECK-LABEL: func.func @moved_storage(
func.func @moved_storage(%x: memref<?x3xf32>, %cell: !md.cell, %n: index) {
  %nl = md_exec.empty_neighbors size(%n) positions(memref<?x3xf32>)
      kind(matrix) width(96) : !mdrt.neighbors<@atoms>
  // CHECK: %[[REF:[0-9]+]] = md_exec.reference_positions %{{[0-9]+}} : !mdrt.neighbors<@atoms> -> memref<?x3xf32>
  %ref = md_exec.reference_positions %nl
      : !mdrt.neighbors<@atoms> -> memref<?x3xf32>

  %no = arith.constant false
  // CHECK: %[[MOVED:[0-9]+]] = md_exec.particle_for
  // CHECK-SAME: ins(%{{[a-z0-9]+}}, %[[REF]] : memref<?x3xf32>, memref<?x3xf32>)
  // CHECK-SAME: reduce(%{{[a-z0-9_]+}} : i1) {
  %moved = md_exec.particle_for
      ins(%x, %ref : memref<?x3xf32>, memref<?x3xf32>) reduce(%no : i1) {
  ^bb0(%x_i: vector<3xf32>, %ref_i: vector<3xf32>):
    %limit = arith.constant 0.0225 : f32
    %d = arith.subf %x_i, %ref_i : vector<3xf32>
    %sq = arith.mulf %d, %d : vector<3xf32>
    %d2 = vector.reduction <add>, %sq : vector<3xf32> into f32
    %far = arith.cmpf ugt, %d2, %limit : f32
    md_exec.yield %far : i1
  } -> i1

  // CHECK: md_exec.refresh_neighbors %{{[0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} moved(%[[MOVED]])
  // CHECK-SAME: policy(check) : !mdrt.neighbors<@atoms>, memref<?x3xf32>
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved)
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
      : !mdrt.neighbors<@atoms>, memref<?x3xf32>
  return
}

// Fields and kernels in f32, as the precision policy assigns them. A buffer
// states the type that a field is stored in.
//
// CHECK-LABEL: func.func @single(
func.func @single(%buffer: memref<?x3xf32>, %cell: !md.cell)
    -> (!md.field<@atoms, 3 x f32>, f64) {
  // CHECK: %[[X:[0-9]+]] = mdrt.from_buffer %{{[a-z0-9]+}} : memref<?x3xf32> to !md.field<@atoms, 3 x f32>
  %x = mdrt.from_buffer %buffer
      : memref<?x3xf32> to !md.field<@atoms, 3 x f32>
  %nl0 = md_exec.empty_neighbors kind(matrix) width(96)
      : !mdrt.neighbors<@atoms>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f32>
  %f0 = md_exec.zeros : !md.field<@atoms, 3 x f32>
  %u0 = arith.constant 0.0 : f64

  // CHECK: md_exec.pair_for %{{[0-9]+}}, %[[X]], %{{[a-z0-9]+}}
  // CHECK-SAME: outs(%{{[0-9]+}} : !md.field<@atoms, 3 x f32>) reduce(%{{[a-z0-9_]+}} : f64)
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f32, %{{[a-z0-9]+}}: vector<3xf32>):
  // CHECK: } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f32> -> !md.field<@atoms, 3 x f32>, f64
  %f, %u = md_exec.pair_for %nl, %x, %cell
      outs(%f0 : !md.field<@atoms, 3 x f32>) reduce(%u0 : f64)
      cutoff(2.5) weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_exec.yield %d, %e : vector<3xf32>, f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f32>
      -> !md.field<@atoms, 3 x f32>, f64
  return %f, %u : !md.field<@atoms, 3 x f32>, f64
}

// At the semantic level the field has the reference precision, whatever
// the buffer holds.
//
// CHECK-LABEL: func.func @stored_in_single(
func.func @stored_in_single(%buffer: memref<?x3xf32>) -> memref<?x3xf32> {
  // CHECK: %[[X:[0-9]+]] = mdrt.from_buffer %{{[a-z0-9]+}} : memref<?x3xf32> to !md.field<@atoms, 3 x f64>
  // CHECK: mdrt.to_buffer %[[X]] : !md.field<@atoms, 3 x f64> to memref<?x3xf32>
  %x = mdrt.from_buffer %buffer
      : memref<?x3xf32> to !md.field<@atoms, 3 x f64>
  %out = mdrt.to_buffer %x
      : !md.field<@atoms, 3 x f64> to memref<?x3xf32>
  return %out : memref<?x3xf32>
}

// The storage form: the loops take buffers and update them where they are.
//
// CHECK-LABEL: func.func @storage(
func.func @storage(%x: memref<?x3xf32>, %v: memref<?x3xf32>,
                   %f: memref<?x3xf32>, %cell: !md.cell, %n: index) -> f64 {
  // CHECK: %[[NL0:[0-9]+]] = md_exec.empty_neighbors size(%{{[a-z0-9]+}}) positions(memref<?x3xf32>) kind(matrix) width(96) : !mdrt.neighbors<@atoms>
  %nl0 = md_exec.empty_neighbors size(%n) positions(memref<?x3xf32>)
      kind(matrix) width(96) : !mdrt.neighbors<@atoms>

  // CHECK: %[[NL:[0-9]+]] = md_exec.refresh_neighbors %[[NL0]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}}
  // CHECK-SAME: policy(always) : !mdrt.neighbors<@atoms>, memref<?x3xf32>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(always)
      : !mdrt.neighbors<@atoms>, memref<?x3xf32>

  // CHECK: %[[U:[0-9]+]] = md_exec.pair_for %[[NL]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}}
  // CHECK-SAME: outs(%{{[a-z0-9]+}} : memref<?x3xf32>) reduce(%{{[a-z0-9_]+}} : f64)
  // CHECK-SAME: cutoff(2.500000e+00) weights [5.000000e-01] overwrite [true] policy(directed, owner_only) {
  // CHECK: } : !mdrt.neighbors<@atoms>, memref<?x3xf32> -> f64
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell outs(%f : memref<?x3xf32>)
      reduce(%u0 : f64) cutoff(2.5) weights [0.5] overwrite [true]
      policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_exec.yield %d, %e : vector<3xf32>, f64
  } : !mdrt.neighbors<@atoms>, memref<?x3xf32> -> f64

  // CHECK: md_exec.particle_for ins(%{{[a-z0-9]+}}, %{{[a-z0-9]+}} : memref<?x3xf32>, memref<?x3xf32>) outs(%{{[a-z0-9]+}} : memref<?x3xf32>) {
  md_exec.particle_for ins(%v, %f : memref<?x3xf32>, memref<?x3xf32>)
      outs(%v : memref<?x3xf32>) {
  ^bb0(%v_i: vector<3xf32>, %f_i: vector<3xf32>):
    %new = arith.addf %v_i, %f_i : vector<3xf32>
    md_exec.yield %new : vector<3xf32>
  }
  return %u : f64
}

// A function of the host that reads a field.
//
// CHECK-LABEL: func.func @host(
func.func private @write_frame(i64, memref<?x3xf32>)

func.func @host(%x: !md.field<@atoms, 3 x f64>, %step: i64) {
  // CHECK: mdrt.host_call @write_frame(%{{[a-z0-9]+}}, %{{[a-z0-9]+}}) : (i64, !md.field<@atoms, 3 x f64>)
  mdrt.host_call @write_frame(%step, %x)
      : (i64, !md.field<@atoms, 3 x f64>)
  return
}
