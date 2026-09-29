// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>

md.particle_set @atoms

// A field that a buffer of the host holds is copied to the device, and a
// field that the host asks for is copied back, into a buffer of the host
// that the program was given. The buffers of the device have a memory
// space.
//
// CHECK-LABEL: func.func @kick(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64>, %[[F:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[DT:[a-z0-9]+]]: f64)
// CHECK-SAME:    -> memref<?x3xf64>
func.func @kick(%velocities: memref<?x3xf64>, %f: !vec, %dt: f64)
    -> memref<?x3xf64> {
  // CHECK:      %[[N:[a-z0-9_]+]] = memref.dim %[[F]],
  // CHECK:      %[[DEVICE:[a-z0-9_]+]] = gpu.alloc (%{{[a-z0-9_]+}}) : memref<?x3xf64, 1>
  // CHECK:      %[[T0:[0-9]+]] = gpu.wait async
  // CHECK:      %[[T1:[0-9]+]] = gpu.memcpy async [%[[T0]]] %[[DEVICE]], %[[V]] : memref<?x3xf64, 1>, memref<?x3xf64>
  // CHECK:      gpu.wait [%[[T1]]]
  %v = mdrt.from_buffer %velocities : memref<?x3xf64> to !vec

  // CHECK:      md_exec.particle_for ins(%[[DEVICE]], %[[F]] : memref<?x3xf64, 1>, memref<?x3xf64, 1>) outs(%[[DEVICE]] : memref<?x3xf64, 1>) {
  %v0 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f : !vec, !vec) outs(%v0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>):
    %s = vector.broadcast %dt : f64 to vector<3xf64>
    %change = arith.mulf %s, %f_i : vector<3xf64>
    %new = arith.addf %v_i, %change : vector<3xf64>
    md_exec.yield %new : vector<3xf64>
  } -> !vec

  // CHECK:      %[[T2:[0-9]+]] = gpu.wait async
  // CHECK:      %[[T3:[0-9]+]] = gpu.memcpy async [%[[T2]]] %[[V]], %[[DEVICE]] : memref<?x3xf64>, memref<?x3xf64, 1>
  // CHECK:      gpu.wait [%[[T3]]]
  // CHECK:      return %[[V]]
  %result = mdrt.to_buffer %v1 : !vec to memref<?x3xf64>
  return %result : memref<?x3xf64>
}

// A loop with a global sum gets two buffers for itself, with one value of
// the type of the sum per particle. They come from the pool, so a loop
// around them carries them, and nothing is allocated inside it.
//
// CHECK-LABEL: func.func @energy(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf32, 1>, %{{[a-z0-9]+}}: index)
func.func @energy(%v: !md.field<@atoms, 3 x f32>, %steps: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  // CHECK-DAG:  %[[A:[a-z0-9_]+]] = gpu.alloc (%{{[a-z0-9_]+}}) : memref<?xf64, 1>
  // CHECK-DAG:  %[[B:[a-z0-9_]+]] = gpu.alloc (%{{[a-z0-9_]+}}) : memref<?xf64, 1>
  // CHECK:      scf.for {{.*}} iter_args(%{{[a-z0-9]+}} = %{{[a-z0-9_]+}}, %[[AA:[a-z0-9]+]] = %{{[a-z0-9_]+}}, %[[BA:[a-z0-9]+]] = %{{[a-z0-9_]+}}) -> (f64, memref<?xf64, 1>, memref<?xf64, 1>) {
  // CHECK-NOT:    gpu.alloc
  // CHECK:        md_exec.particle_for ins(%[[V]] : memref<?x3xf32, 1>) reduce(%{{[a-z0-9_]+}} : f64) scratch(%{{[a-z0-9]+}}, %{{[a-z0-9]+}} : memref<?xf64, 1>, memref<?xf64, 1>) {
  %total = scf.for %step = %c0 to %steps step %c1
      iter_args(%sum = %zero) -> (f64) {
    %k = md_exec.particle_for ins(%v : !md.field<@atoms, 3 x f32>)
        reduce(%zero : f64) {
    ^bb0(%v_i: vector<3xf32>):
      %sq = arith.mulf %v_i, %v_i : vector<3xf32>
      %v2 = vector.reduction <add>, %sq : vector<3xf32> into f32
      %wide = arith.extf %v2 : f32 to f64
      md_exec.yield %wide : f64
    } -> f64
    %next = arith.addf %sum, %k : f64
    scf.yield %next : f64
  }
  return %total : f64
}

// The storage of a neighbor structure is on the device, and the test of
// validity gets two buffers of the type of the positions.
//
// CHECK-LABEL: func.func @refresh(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf32, 1>, %[[CELL:[a-z0-9]+]]: !md.cell)
func.func @refresh(%x: !md.field<@atoms, 3 x f32>, %cell: !md.cell) -> i64 {
  // CHECK:      %[[STORAGE:[0-9]+]] = md_exec.empty_neighbors size(%{{[a-z0-9_]+}}) positions(memref<?x3xf32, 1>) kind(matrix) width(48)
  // CHECK:      md_exec.refresh_neighbors %[[STORAGE]], %[[X]], %[[CELL]] scratch(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<?xf32, 1>, memref<?xf32, 1>)
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f32>
  %builds = md_exec.rebuild_count %nl : !mdrt.neighbors<@atoms>
  return %builds : i64
}

// The host reads a copy of a field that is on the device. One buffer of the
// host takes the copies.
//
// CHECK-LABEL: func.func @frames(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[STEP:[a-z0-9]+]]: i64, %{{[a-z0-9]+}}: index)
func.func private @write_frame(i64, memref<?x3xf64>)

func.func @frames(%x: !vec, %step: i64, %frames: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:      %[[HOST:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
  // CHECK:      scf.for
  // CHECK-NOT:    memref.alloc
  // CHECK:        gpu.memcpy async [%{{[0-9]+}}] %[[HOST]], %[[X]] : memref<?x3xf64>, memref<?x3xf64, 1>
  // CHECK:        call @write_frame(%[[STEP]], %[[HOST]])
  scf.for %frame = %c0 to %frames step %c1 {
    mdrt.host_call @write_frame(%step, %x) : (i64, !vec)
  }
  return
}

// A loop that tells whether the kernel yields true for any particle needs
// no buffers in `scratch`, and neither does the refresh that takes the
// result.
//
// CHECK-LABEL: func.func @validity(
// CHECK:         %[[NL:[0-9]+]] = md_exec.empty_neighbors size(%{{[a-z0-9_]+}}) positions(memref<?x3xf64, 1>)
// CHECK:         scf.for
// CHECK:           %[[REF:[0-9]+]] = md_exec.reference_positions %[[NL]] : !mdrt.neighbors<@atoms> -> memref<?x3xf64, 1>
// CHECK:           %[[MOVED:[0-9]+]] = md_exec.particle_for ins(%[[XA:[a-z0-9]+]], %{{[a-z0-9_]+}}, %[[REF]] :
// CHECK-SAME:        outs(%[[XA]] : memref<?x3xf64, 1>) reduce(%{{[a-z0-9_]+}} : i1) {
// CHECK:           md_exec.refresh_neighbors %[[NL]], %[[XA]], %{{[a-z0-9]+}} moved(%[[MOVED]]) cutoff

func.func @validity(%xb: memref<?x3xf64>, %vb: memref<?x3xf64>,
                  %cell: !md.cell, %n: index) -> memref<?x3xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %x = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
  %v = mdrt.from_buffer %vb : memref<?x3xf64> to !vec
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
  %out = mdrt.to_buffer %xe : !vec to memref<?x3xf64>
  return %out : memref<?x3xf64>
}
