// The storage form, lowered to loops.
//
// RUN: mdir-opt %s --convert-md-exec-to-loops | FileCheck %s

md.particle_set @atoms

// A buffer that the loop reads and writes. The kernel of a particle reads
// the values of that particle before it writes them.
//
// CHECK-LABEL: func.func @kick(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64>, %[[F:[a-z0-9]+]]: memref<?x3xf64>, %[[DT:[a-z0-9]+]]: f64)
func.func @kick(%v: memref<?x3xf64>, %f: memref<?x3xf64>, %dt: f64) {
  // CHECK:      %[[N:[a-z0-9_]+]] = memref.dim %[[V]],
  // CHECK:      scf.parallel (%[[I:[a-z0-9]+]]) = (%{{[a-z0-9_]+}}) to (%[[N]])
  // CHECK:        memref.load %[[V]][%[[I]],
  // CHECK:        memref.load %[[F]][%[[I]],
  // CHECK:        vector.broadcast %[[DT]]
  // CHECK:        memref.store %{{[0-9]+}}, %[[V]][%[[I]],
  // CHECK-NOT:  md_exec
  md_exec.particle_for ins(%v, %f : memref<?x3xf64>, memref<?x3xf64>)
      outs(%v : memref<?x3xf64>) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>):
    %s = vector.broadcast %dt : f64 to vector<3xf64>
    %change = arith.mulf %s, %f_i : vector<3xf64>
    %new = arith.addf %v_i, %change : vector<3xf64>
    md_exec.yield %new : vector<3xf64>
  }
  return
}

// The storage of a neighbor structure, a refresh, and two loops over its
// pairs: one that overwrites its destination and one that adds to it.
//
// CHECK-LABEL: func.func @forces(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[F:[a-z0-9]+]]: memref<?x3xf64>, %[[BOX:[a-z0-9]+]]: vector<3xf64>, %[[N:[a-z0-9]+]]: index)
// CHECK-SAME:    -> (f64, i64)
func.func @forces(%x: memref<?x3xf64>, %f: memref<?x3xf64>, %cell: !md.cell,
                  %n: index) -> (f64, i64) {
  // CHECK:      %[[COUNTS:[a-z0-9_]+]] = memref.alloc(%[[N]]) : memref<?xi32>
  // CHECK:      %[[ROWS:[a-z0-9_]+]] = call @mdrtHostMatrixCreate(
  // CHECK:      %[[REFERENCE:[a-z0-9_]+]] = memref.alloc(%[[N]]) : memref<?x3xf64>
  // CHECK:      %[[BUILDS:[a-z0-9_]+]] = memref.alloc() : memref<i64>
  %nl0 = md_exec.empty_neighbors size(%n) positions(memref<?x3xf64>)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>

  // CHECK:      memref.load %[[REFERENCE]][
  // CHECK:      scf.if
  // CHECK:        scf.while
  // CHECK:          %[[INDEX:[a-z0-9_]+]] = func.call @mdrtHostMatrixEntries(%[[ROWS]])
  // CHECK:          call @mdrt.build_neighbors_matrix(%[[X]], %[[BOX]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[COUNTS]], %[[INDEX]])
  // CHECK:          call @mdrtHostMatrixGrow(%[[ROWS]],
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, memref<?x3xf64>

  // CHECK:      %[[U:[0-9]+]] = scf.parallel (%[[I:[a-z0-9]+]]) =
  // CHECK:        memref.load %[[COUNTS]][%[[I]]]
  // CHECK:        scf.for
  // CHECK-NOT:    memref.load %[[F]]
  // CHECK:        memref.store %{{[0-9]+}}, %[[F]][%[[I]],
  // CHECK:        scf.reduce
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell outs(%f : memref<?x3xf64>)
      reduce(%u0 : f64) cutoff(1.5) weights [0.5] overwrite [true]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d, %r2 : vector<3xf64>, f64
  } : !mdrt.neighbors<@atoms>, memref<?x3xf64> -> f64

  // CHECK:      scf.parallel (%[[J:[a-z0-9]+]]) =
  // CHECK:        scf.for
  // CHECK:        memref.load %[[F]][%[[J]],
  // CHECK:        memref.store %{{[0-9]+}}, %[[F]][%[[J]],
  md_exec.pair_for %nl, %x, %cell outs(%f : memref<?x3xf64>)
      cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms>, memref<?x3xf64>

  // CHECK:      %[[COUNT:[0-9]+]] = memref.load %[[BUILDS]][]
  // CHECK:      return %[[U]], %[[COUNT]]
  %builds = md_exec.rebuild_count %nl : !mdrt.neighbors<@atoms>
  return %u, %builds : f64, i64
}

// A loop that tests the validity of a neighbor structure reads the
// positions that the structure was built at from the buffer of the
// structure. Threads combine what they find as integers of 32 bits. The
// refresh makes no test of its own.
//
// CHECK-LABEL: func.func @validity(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[V:[a-z0-9]+]]: memref<?x3xf64>,
// CHECK:         %[[REFERENCE:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
// CHECK:         %[[BOX:[a-z0-9_]+]] = memref.alloc() : memref<3xf64>
// CHECK:         %[[START:[0-9]+]] = arith.extui %{{[a-z0-9_]+}} : i1 to i32
// CHECK:         %[[ANY:[0-9]+]] = scf.parallel (%[[I:[a-z0-9]+]]) = (%{{[a-z0-9_]+}}) to (%{{[a-z0-9_]+}}) step (%{{[a-z0-9_]+}}) init (%[[START]]) -> i32 {
// CHECK:           memref.load %[[REFERENCE]][%[[I]],
// CHECK:           %[[FAR:[0-9]+]] = arith.cmpf ugt,
// CHECK:           memref.store %{{[0-9]+}}, %[[X]][%[[I]],
// CHECK:           %[[WIDE:[0-9]+]] = arith.extui %[[FAR]] : i1 to i32
// CHECK:           scf.reduce(%[[WIDE]] : i32) {
// CHECK:             arith.ori
// CHECK:         %[[MOVED:[0-9]+]] = arith.cmpi ne, %[[ANY]], %{{[a-z0-9_]+}} : i32
// CHECK-NOT:     scf.parallel
// CHECK:         %[[NEAR:[0-9]+]] = arith.xori %[[MOVED]], %{{[a-z0-9_]+}} : i1
// CHECK:         %[[VALID:[0-9]+]] = arith.andi %{{[0-9]+}}, %[[NEAR]]
// CHECK:         %[[STALE:[0-9]+]] = arith.xori %[[VALID]],
// CHECK:         scf.if %[[STALE]] {
// CHECK:           call @mdrt.build_neighbors_matrix(

func.func @validity(%x: memref<?x3xf64>, %v: memref<?x3xf64>,
                    %cell: !md.cell, %n: index) {
  %nl = md_exec.empty_neighbors size(%n) positions(memref<?x3xf64>)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  %ref = md_exec.reference_positions %nl
      : !mdrt.neighbors<@atoms> -> memref<?x3xf64>
  %no = arith.constant false
  %moved = md_exec.particle_for
      ins(%x, %v, %ref
          : memref<?x3xf64>, memref<?x3xf64>, memref<?x3xf64>)
      outs(%x : memref<?x3xf64>) reduce(%no : i1) {
  ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>, %r_i: vector<3xf64>):
    %xn = arith.addf %x_i, %v_i : vector<3xf64>
    %d = arith.subf %xn, %r_i : vector<3xf64>
    %sq = arith.mulf %d, %d : vector<3xf64>
    %d2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %limit = arith.constant 0.015625 : f64
    %far = arith.cmpf ugt, %d2, %limit : f64
    md_exec.yield %xn, %far : vector<3xf64>, i1
  } -> i1
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved)
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, memref<?x3xf64>
  return
}

// The order of the particles is a call to the template, which is in the
// module. A field in that order is a loop that reads place `order[k]` and
// writes place `k`.
//
// CHECK-LABEL: func.func @ordered(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[IDS:[a-z0-9]+]]: memref<?xi32>, %[[ORDER:[a-z0-9]+]]: memref<?xi32>, %[[XS:[a-z0-9]+]]: memref<?x3xf64>, %[[BOX:[a-z0-9]+]]: vector<3xf64>)
// CHECK:         %[[WIDTH:[a-z0-9_]+]] = arith.constant 1.100000e+00 : f64
// CHECK:         call @mdrt.spatial_order(%[[X]], %[[BOX]], %[[WIDTH]], %[[IDS]], %[[ORDER]])
// CHECK:         scf.parallel (%[[K:[a-z0-9]+]]) =
// CHECK:           %[[FROM:[0-9]+]] = memref.load %[[ORDER]][%[[K]]]
// CHECK:           %[[J:[0-9]+]] = arith.index_cast %[[FROM]] : i32 to index
// CHECK:           memref.load %[[X]][%[[J]],
// CHECK:           memref.store %{{[0-9]+}}, %[[XS]][%[[K]],
// CHECK-NOT:     md_exec

func.func @ordered(%x: memref<?x3xf64>, %ids: memref<?xi32>,
                   %order: memref<?xi32>, %xs: memref<?x3xf64>,
                   %cell: !md.cell) {
  md_exec.spatial_order %x, %cell, %ids outs(%order : memref<?xi32>)
      width(1.1) : memref<?x3xf64>, memref<?xi32>
  md_exec.permute %x, %order outs(%xs : memref<?x3xf64>)
      : memref<?x3xf64>, memref<?xi32>
  return
}

// CHECK-NOT: md.particle_set
