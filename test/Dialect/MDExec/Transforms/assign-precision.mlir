// RUN: mdir-opt %s --md-exec-assign-precision="mode=mixed" \
// RUN: | FileCheck %s --check-prefixes=CHECK,MIXED
// RUN: mdir-opt %s --md-exec-assign-precision="mode=single" \
// RUN: | FileCheck %s --check-prefixes=CHECK,SINGLE
// RUN: mdir-opt %s --md-exec-assign-precision="mode=double" \
// RUN: | FileCheck %s --check-prefixes=CHECK,DOUBLE
// RUN: mdir-opt %s \
// RUN:     --md-exec-assign-precision="mode=mixed accumulator=f32" \
// RUN: | FileCheck %s --check-prefixes=ACCUMULATOR

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!kinds = !md.field<@atoms, i32>

md.particle_set @atoms

// A loop over pairs computes in the type of the kernel and writes fields of
// the type of forces. The buffer states the type of the positions, whatever
// the mode. Values from outside the kernel are converted before the loop,
// and the contribution to a global sum where the kernel ends.

// CHECK-LABEL: func.func @forces
// MIXED-SAME:    -> (f64, !md.field<@atoms, 3 x f32>)
// SINGLE-SAME:   -> (f64, !md.field<@atoms, 3 x f32>)
// DOUBLE-SAME:   -> (f64, !md.field<@atoms, 3 x f64>)
// CHECK:         %[[X:.*]] = mdrt.from_buffer %{{.*}} : memref<?x3xf64> to !md.field<@atoms, 3 x f64>
// MIXED:         %[[F0:.*]] = md_exec.zeros : !md.field<@atoms, 3 x f32>
// MIXED:         %[[EPS:.*]] = arith.truncf %{{.*}} : f64 to f32
// MIXED:         md_exec.pair_for %{{.*}}, %[[X]], %{{.*}} outs(%[[F0]] : !md.field<@atoms, 3 x f32>) reduce(%{{.*}} : f64)
// MIXED-NEXT:    ^bb0(%[[R2:.*]]: f32, %[[D:.*]]: vector<3xf32>):
// MIXED-NEXT:      %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// MIXED-NEXT:      %[[A:.*]] = arith.mulf %[[EPS]], %[[R2]] : f32
// MIXED-NEXT:      %[[B:.*]] = arith.mulf %[[A]], %[[TWO]] : f32
// MIXED-NEXT:      %[[S:.*]] = vector.broadcast %[[B]] : f32 to vector<3xf32>
// MIXED-NEXT:      %[[K:.*]] = arith.mulf %[[S]], %[[D]] : vector<3xf32>
// MIXED-NEXT:      %[[WIDE:.*]] = arith.extf %[[A]] : f32 to f64
// MIXED-NEXT:      md_exec.yield %[[K]], %[[WIDE]] : vector<3xf32>, f64
// MIXED-NEXT:    } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f32>, f64
// DOUBLE:        md_exec.pair_for
// DOUBLE-NEXT:   ^bb0(%{{.*}}: f64, %{{.*}}: vector<3xf64>):
// DOUBLE-NOT:    arith.truncf
// DOUBLE-NOT:    arith.extf
// DOUBLE:        return

// A global sum that is accumulated in f32 is converted after the loop.

// ACCUMULATOR-LABEL: func.func @forces
// ACCUMULATOR:         %[[U0:.*]] = arith.truncf %{{.*}} : f64 to f32
// ACCUMULATOR:         %[[LOOP:.*]]:2 = md_exec.pair_for {{.*}} reduce(%[[U0]] : f32)
// ACCUMULATOR:           md_exec.yield %{{.*}}, %{{.*}} : vector<3xf32>, f32
// ACCUMULATOR:         %[[U:.*]] = arith.extf %[[LOOP]]#1 : f32 to f64
// ACCUMULATOR:         return %[[U]], %[[LOOP]]#0
func.func @forces(%positions: memref<?x3xf64>, %cell: !md.cell, %eps: f64)
    -> (f64, !vec) {
  %x = mdrt.from_buffer %positions : memref<?x3xf64> to !vec
  %cells = md_exec.build_cells %x, %cell width(2.8)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(2.5) skin(0.3) kind(matrix) width(96)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %f0 = md_exec.zeros : !vec
  %u0 = arith.constant 0.0 : f64
  %f, %u = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec)
      reduce(%u0 : f64) cutoff(2.5) weights [0.5]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %two = arith.constant 2.0 : f64
    %a = arith.mulf %eps, %r2 : f64
    %b = arith.mulf %a, %two : f64
    %s = vector.broadcast %b : f64 to vector<3xf64>
    %k = arith.mulf %s, %d : vector<3xf64>
    md_exec.yield %k, %a : vector<3xf64>, f64
  } : !mdrt.neighbors<@atoms>, !vec -> !vec, f64
  return %u, %f : f64, !vec
}

// A loop over particles computes in the type of the integrator. The kernel
// receives the values of a field in the type that the field is stored in
// and converts them.

// CHECK-LABEL: func.func @kick
// MIXED-SAME:    %{{.*}}: !md.field<@atoms, 3 x f32>
// MIXED:         md_exec.particle_for ins(%{{.*}}, %{{.*}}, %{{.*}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f32>, !md.field<@atoms, f64>) outs(%{{.*}} : !md.field<@atoms, 3 x f64>)
// MIXED-NEXT:    ^bb0(%[[V:.*]]: vector<3xf64>, %[[F:.*]]: vector<3xf32>, %[[M:.*]]: f64):
// MIXED-NEXT:      %[[WIDE:.*]] = arith.extf %[[F]] : vector<3xf32> to vector<3xf64>
// MIXED-NEXT:      %[[RATIO:.*]] = arith.divf %{{.*}}, %[[M]] : f64
// MIXED-NEXT:      %[[S:.*]] = vector.broadcast %[[RATIO]] : f64 to vector<3xf64>
// MIXED-NEXT:      %[[CHANGE:.*]] = arith.mulf %[[S]], %[[WIDE]] : vector<3xf64>
// MIXED-NEXT:      %[[NEW:.*]] = arith.addf %[[V]], %[[CHANGE]] : vector<3xf64>
// MIXED-NEXT:      md_exec.yield %[[NEW]] : vector<3xf64>
//
// The buffers hold f64 and the kernel computes in f32.
//
// SINGLE:        md_exec.particle_for ins(%{{.*}}, %{{.*}}, %{{.*}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f32>, !md.field<@atoms, f64>) outs(%{{.*}} : !md.field<@atoms, 3 x f64>)
// SINGLE-NEXT:   ^bb0(%[[V:.*]]: vector<3xf64>, %[[F:.*]]: vector<3xf32>, %[[M:.*]]: f64):
// SINGLE-NEXT:     %[[VN:.*]] = arith.truncf %[[V]] : vector<3xf64> to vector<3xf32>
// SINGLE-NEXT:     %[[MN:.*]] = arith.truncf %[[M]] : f64 to f32
// SINGLE:          %[[NEW:.*]] = arith.addf %[[VN]], %{{.*}} : vector<3xf32>
// SINGLE-NEXT:     %[[WIDE:.*]] = arith.extf %[[NEW]] : vector<3xf32> to vector<3xf64>
// SINGLE-NEXT:     md_exec.yield %[[WIDE]] : vector<3xf64>
func.func @kick(%velocities: memref<?x3xf64>, %f: !vec,
                %masses: memref<?xf64>, %cell: !md.cell, %dt: f64)
    -> memref<?x3xf64> {
  %v = mdrt.from_buffer %velocities : memref<?x3xf64> to !vec
  %m = mdrt.from_buffer %masses : memref<?xf64> to !real

  // The field is one that a loop over pairs accumulates into.
  %cells = md_exec.build_cells %v, %cell width(2.8)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %v, %cell
      cutoff(2.5) skin(0.3) kind(matrix) width(96)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %f1 = md_exec.pair_for %nl, %v, %cell outs(%f : !vec)
      cutoff(2.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms>, !vec -> !vec

  %v0 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f1, %m : !vec, !vec, !real)
      outs(%v0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
    %ratio = arith.divf %dt, %m_i : f64
    %s = vector.broadcast %ratio : f64 to vector<3xf64>
    %change = arith.mulf %s, %f_i : vector<3xf64>
    %new = arith.addf %v_i, %change : vector<3xf64>
    md_exec.yield %new : vector<3xf64>
  } -> !vec
  %result = mdrt.to_buffer %v1 : !vec to memref<?x3xf64>
  return %result : memref<?x3xf64>
}

// Fields that no buffer holds. A loop-carried field is stored with the
// fields that it is initialized and updated with: positions in the type of
// positions, and what a loop over particles writes in the type of the
// integrator. Fields of integers keep their type.

// CHECK-LABEL: func.func @carried
// MIXED-SAME:    (%{{.*}}: !md.field<@atoms, 3 x f64>, %{{.*}}: !md.field<@atoms, 3 x f64>, %{{.*}}: !md.field<@atoms, i32>,
// MIXED-SAME:    -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>)
// SINGLE-SAME:   (%{{.*}}: !md.field<@atoms, 3 x f32>, %{{.*}}: !md.field<@atoms, 3 x f32>, %{{.*}}: !md.field<@atoms, i32>,
// SINGLE-SAME:   -> (!md.field<@atoms, 3 x f32>, !md.field<@atoms, 3 x f32>)
// SINGLE:        scf.for {{.*}} -> (!md.field<@atoms, 3 x f32>, !md.field<@atoms, 3 x f32>, !mdrt.neighbors<@atoms>)
// SINGLE:          md_exec.empty : !md.field<@atoms, 3 x f32>
// SINGLE:          %[[DT:.*]] = arith.truncf %{{.*}} : f64 to f32
// SINGLE:          md_exec.particle_for ins(%{{.*}}, %{{.*}}, %{{.*}} : !md.field<@atoms, 3 x f32>, !md.field<@atoms, 3 x f32>, !md.field<@atoms, i32>)
// SINGLE-NEXT:     ^bb0(%{{.*}}: vector<3xf32>, %{{.*}}: vector<3xf32>, %{{.*}}: i32):
// SINGLE:            vector.broadcast %[[DT]] : f32 to vector<3xf32>
// SINGLE:          md_exec.refresh_neighbors %{{.*}}, %{{.*}}, %{{.*}} cutoff({{.*}}) skin({{.*}}) cell_width({{.*}}) policy(check) : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f32>
func.func @carried(%x0: !vec, %v0: !vec, %kinds: !kinds, %cell: !md.cell,
                   %dt: f64, %steps: index) -> (!vec, !vec) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %nl0 = md_exec.empty_neighbors kind(matrix) width(96)
      : !mdrt.neighbors<@atoms>
  %x, %v, %nl = scf.for %step = %c0 to %steps step %c1
      iter_args(%xa = %x0, %va = %v0, %nla = %nl0)
      -> (!vec, !vec, !mdrt.neighbors<@atoms>) {
    %e = md_exec.empty : !vec
    %xb = md_exec.particle_for ins(%xa, %va, %kinds : !vec, !vec, !kinds)
        outs(%e : !vec) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>, %kind: i32):
      %s = vector.broadcast %dt : f64 to vector<3xf64>
      %change = arith.mulf %s, %v_i : vector<3xf64>
      %new = arith.addf %x_i, %change : vector<3xf64>
      md_exec.yield %new : vector<3xf64>
    } -> !vec
    %nlb = md_exec.refresh_neighbors %nla, %xb, %cell
        cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
        : !mdrt.neighbors<@atoms>, !vec
    %vb = md_exec.particle_for ins(%va : !vec) outs(%e : !vec) {
    ^bb0(%v_i: vector<3xf64>):
      md_exec.yield %v_i : vector<3xf64>
    } -> !vec
    scf.yield %xb, %vb, %nlb : !vec, !vec, !mdrt.neighbors<@atoms>
  }
  return %x, %v : !vec, !vec
}

// A neighbor structure holds the positions that it was built at as the
// positions are stored: here in f32, in every mode, because the buffer
// says so. The result of the test has no floating-point type.
//
// CHECK-LABEL: func.func @validity(
// CHECK:         %[[REF:[0-9]+]] = md_exec.reference_positions %{{[a-z0-9]+}} : !mdrt.neighbors<@atoms> -> !md.field<@atoms, 3 x f32>
// CHECK:         md_exec.particle_for ins(%{{[a-z0-9]+}}, %{{[0-9]+}}, %[[REF]] :
// CHECK-SAME:      reduce(%{{[a-z0-9_]+}} : i1)
// CHECK-NEXT:    ^bb0(%{{[a-z0-9]+}}: vector<3xf32>, %{{[a-z0-9]+}}: vector<3xf32>, %{{[a-z0-9]+}}: vector<3xf32>):
// SINGLE:          arith.cmpf ugt, %{{[0-9]+}}, %{{[a-z0-9_]+}} : f32
// MIXED:           arith.cmpf ugt, %{{[0-9]+}}, %{{[a-z0-9_]+}} : f64
// CHECK:         } -> !md.field<@atoms, 3 x f32>, i1
// CHECK:         md_exec.refresh_neighbors %{{[a-z0-9]+}}, %{{[0-9]+}}#0, %{{[a-z0-9]+}} moved(%{{[0-9]+}}#1)
// ACCUMULATOR-LABEL: func.func @validity(
// ACCUMULATOR:   reduce(%{{[a-z0-9_]+}} : i1)

func.func @validity(%xb: memref<?x3xf32>, %vb: memref<?x3xf32>,
                  %cell: !md.cell, %n: index) -> memref<?x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %x = mdrt.from_buffer %xb : memref<?x3xf32> to !vec
  %v = mdrt.from_buffer %vb : memref<?x3xf32> to !vec
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  %xe, %nle = scf.for %step = %c0 to %n step %c1
      iter_args(%xa = %x, %na = %nl0) -> (!vec, !mdrt.neighbors<@atoms>) {
    %ref = md_exec.reference_positions %na
        : !mdrt.neighbors<@atoms> -> !vec
    %e = md_exec.empty : !vec
    %no = arith.constant false
    %x1, %moved = md_exec.particle_for ins(%xa, %v, %ref : !vec, !vec, !vec)
        outs(%e : !vec) reduce(%no : i1) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>, %r_i: vector<3xf64>):
      %xn = arith.addf %x_i, %v_i : vector<3xf64>
      %d = arith.subf %xn, %r_i : vector<3xf64>
      %sq = arith.mulf %d, %d : vector<3xf64>
      %d2 = vector.reduction <add>, %sq : vector<3xf64> into f64
      %limit = arith.constant 0.015625 : f64
      %far = arith.cmpf ugt, %d2, %limit : f64
      md_exec.yield %xn, %far : vector<3xf64>, i1
    } -> !vec, i1
    %nb = md_exec.refresh_neighbors %na, %x1, %cell moved(%moved)
        cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
        : !mdrt.neighbors<@atoms>, !vec
    scf.yield %x1, %nb : !vec, !mdrt.neighbors<@atoms>
  }
  %out = mdrt.to_buffer %xe : !vec to memref<?x3xf32>
  return %out : memref<?x3xf32>
}

// A field in another order is stored as the field is. The numbers of the
// particles have no floating-point type.
//
// CHECK-LABEL: func.func @ordered(
// CHECK:         %[[ORDER:[0-9]+]] = md_exec.spatial_order %[[XA:[a-z0-9]+]], %{{[a-z0-9]+}}, %[[IDA:[a-z0-9]+]] width(1.100000e+00)
// CHECK-SAME:      : !md.field<@atoms, 3 x f32>, !md.field<@atoms, i32> -> !mdrt.permutation<@atoms>
// CHECK:         md_exec.permute %[[XA]], %[[ORDER]] : !md.field<@atoms, 3 x f32>, !mdrt.permutation<@atoms> -> !md.field<@atoms, 3 x f32>
// CHECK:         md_exec.permute %{{[a-z0-9]+}}, %[[ORDER]] : !md.field<@atoms, f64>, !mdrt.permutation<@atoms> -> !md.field<@atoms, f64>
// CHECK:         md_exec.permute %[[IDA]], %[[ORDER]] : !md.field<@atoms, i32>, !mdrt.permutation<@atoms> -> !md.field<@atoms, i32>
// ACCUMULATOR-LABEL: func.func @ordered(

func.func private @write_ordered(i64, memref<?x3xf32>, memref<?xi32>)

func.func @ordered(%xb: memref<?x3xf32>, %mb: memref<?xf64>,
                   %ib: memref<?xi32>, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %step = arith.constant 0 : i64
  %x0 = mdrt.from_buffer %xb : memref<?x3xf32> to !vec
  %m0 = mdrt.from_buffer %mb : memref<?xf64> to !real
  %id0 = mdrt.from_buffer %ib
      : memref<?xi32> to !md.field<@atoms, i32>
  %xe, %me, %ide = scf.for %i = %c0 to %n step %c1
      iter_args(%xa = %x0, %ma = %m0, %ida = %id0)
      -> (!vec, !real, !md.field<@atoms, i32>) {
    %order = md_exec.spatial_order %xa, %cell, %ida width(1.1)
        : !vec, !md.field<@atoms, i32> -> !mdrt.permutation<@atoms>
    %xs = md_exec.permute %xa, %order
        : !vec, !mdrt.permutation<@atoms> -> !vec
    %ms = md_exec.permute %ma, %order
        : !real, !mdrt.permutation<@atoms> -> !real
    %ids = md_exec.permute %ida, %order
        : !md.field<@atoms, i32>, !mdrt.permutation<@atoms>
        -> !md.field<@atoms, i32>
    %e = md_exec.empty : !vec
    %x1 = md_exec.particle_for ins(%xs, %ms : !vec, !real) outs(%e : !vec) {
    ^bb0(%x_i: vector<3xf64>, %m_i: f64):
      %s = vector.broadcast %m_i : f64 to vector<3xf64>
      %xn = arith.addf %x_i, %s : vector<3xf64>
      md_exec.yield %xn : vector<3xf64>
    } -> !vec
    mdrt.host_call @write_ordered(%step, %x1, %ids)
        : (i64, !vec, !md.field<@atoms, i32>)
    scf.yield %x1, %ms, %ids : !vec, !real, !md.field<@atoms, i32>
  } {mdrt.segment}
  return
}
