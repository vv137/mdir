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
  // CHECK:      %[[INDEX:[a-z0-9_]+]] = memref.alloc(%[[N]], %{{[a-z0-9_]+}}) : memref<?x?xi32>
  // CHECK:      %[[REFERENCE:[a-z0-9_]+]] = memref.alloc(%[[N]]) : memref<?x3xf64>
  // CHECK:      %[[BUILDS:[a-z0-9_]+]] = memref.alloc() : memref<i64>
  %nl0 = md_exec.empty_neighbors size(%n) element(f64)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>

  // CHECK:      memref.load %[[REFERENCE]][
  // CHECK:      scf.if
  // CHECK:        call @mdrt.build_neighbors_matrix(%[[X]], %[[BOX]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[COUNTS]], %[[INDEX]])
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

// CHECK-NOT: md.particle_set
