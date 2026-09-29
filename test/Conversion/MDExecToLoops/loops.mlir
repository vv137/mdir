// The two passes that take the value form to loops over buffers, together.
// The storage form between them is tested in
// test/Dialect/MDExec/Transforms/assign-storage.mlir.
//
// RUN: mdir-opt %s --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>

md.particle_set @atoms

// A loop over particles writes to the buffer of a field that it reads and
// that is dead afterward.
//
// CHECK-LABEL: func.func @kick(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64>, %[[F:[a-z0-9]+]]: memref<?x3xf64>, %[[M:[a-z0-9]+]]: memref<?xf64>, %[[DT:[a-z0-9]+]]: f64)
// CHECK-SAME:    -> memref<?x3xf64>
func.func @kick(%v: !vec, %f: !vec, %m: !real, %dt: f64) -> !vec {
  // CHECK:      %[[N:[a-z0-9_]+]] = memref.dim %[[V]],
  // CHECK-NOT:  memref.alloc
  // CHECK:      scf.parallel (%[[I:[a-z0-9]+]]) = (%{{[a-z0-9_]+}}) to (%{{[a-z0-9_]+}})
  // CHECK:        memref.load %[[V]][%[[I]],
  // CHECK:        memref.load %[[F]][%[[I]],
  // CHECK:        memref.load %[[M]][%[[I]]]
  // CHECK:        arith.divf %[[DT]],
  // CHECK:        memref.store %{{[0-9]+}}, %[[V]][%[[I]],
  %v0 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f, %m : !vec, !vec, !real)
      outs(%v0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
    %s  = arith.divf %dt, %m_i : f64
    %sv = vector.broadcast %s : f64 to vector<3xf64>
    %dv = arith.mulf %sv, %f_i : vector<3xf64>
    %vn = arith.addf %v_i, %dv : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  // CHECK:      return %[[V]] : memref<?x3xf64>
  return %v1 : !vec
}

// A field that is used afterward keeps its buffer. The result goes to a
// buffer of its own.
//
// CHECK-LABEL: func.func @keep(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64>)
func.func @keep(%v: !vec) -> (!vec, !vec) {
  // CHECK:      %[[NEW:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
  // CHECK:      scf.parallel
  // CHECK:        memref.load %[[V]][
  // CHECK:        memref.store %{{[0-9]+}}, %[[NEW]][
  %w0 = md_exec.empty : !vec
  %w = md_exec.particle_for ins(%v : !vec) outs(%w0 : !vec) {
  ^bb0(%v_i: vector<3xf64>):
    %s = arith.addf %v_i, %v_i : vector<3xf64>
    md_exec.yield %s : vector<3xf64>
  } -> !vec
  // CHECK:      return %[[V]], %[[NEW]]
  return %v, %w : !vec, !vec
}

// A sum over particles is a reduction.
//
// CHECK-LABEL: func.func @sum(
func.func @sum(%v: !vec) -> f64 {
  // CHECK:      %[[ZERO:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f64
  // CHECK:      %[[S:[0-9]+]] = scf.parallel (%{{[a-z0-9]+}}) = (%{{[a-z0-9_]+}}) to (%{{[a-z0-9_]+}}) step (%{{[a-z0-9_]+}}) init (%[[ZERO]]) -> f64 {
  // CHECK:        scf.reduce(%{{[0-9]+}} : f64) {
  // CHECK:          arith.addf
  // CHECK:          scf.reduce.return
  %s0 = arith.constant 0.0 : f64
  %s = md_exec.particle_for ins(%v : !vec) reduce(%s0 : f64) {
  ^bb0(%v_i: vector<3xf64>):
    %sq = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    md_exec.yield %v2 : f64
  } -> f64
  // CHECK:      return %[[S]] : f64
  return %s : f64
}

// A loop over pairs: the minimum image, the cutoff predicate, and the sum
// over the neighbors of each particle.
//
// CHECK-LABEL: func.func @forces(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[BOX:[a-z0-9]+]]: vector<3xf64>)
func.func @forces(%x: !vec, %cell: !md.cell) -> !vec {
  // CHECK:      %[[COUNTS:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?xi32>
  // CHECK:      %[[INDEX:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) : memref<?x?xi32>
  // CHECK:      %[[LARGEST:[0-9]+]] = call @mdrt.build_neighbors_matrix(%[[X]], %[[BOX]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[COUNTS]], %[[INDEX]])
  // CHECK:      arith.cmpi ugt, %[[LARGEST]],
  // CHECK:      scf.if
  // CHECK:        call @mdrtReportNeighborOverflow(
  %cells = md_exec.build_cells %x, %cell width(1.75)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.5) skin(0.25) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>

  // The destination is a field of zeros, so the loop stores the sum and
  // does not read the buffer.
  //
  // The minimum image takes one over the edge lengths, computed once.
  //
  // CHECK:      %[[OUT:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
  // CHECK:      %[[INVERSE:[0-9]+]] = arith.divf %{{[a-z0-9_]+}}, %[[BOX]] : vector<3xf64>
  // CHECK:      scf.parallel (%[[I:[a-z0-9]+]]) =
  // CHECK:        %[[CUTOFF2:[a-z0-9_]+]] = arith.constant 2.250000e+00 : f64
  // CHECK:        memref.load %[[COUNTS]][%[[I]]]
  // CHECK:        scf.for %[[K:[a-z0-9]+]] =
  // CHECK:          memref.load %[[INDEX]][%[[I]], %[[K]]]
  // CHECK:          %[[RAW:[0-9]+]] = arith.subf
  // CHECK:          %[[IMAGES:[0-9]+]] = arith.mulf %[[RAW]], %[[INVERSE]]
  // CHECK:          %[[NEAREST:[0-9]+]] = math.roundeven %[[IMAGES]]
  // CHECK:          %[[SHIFT:[0-9]+]] = arith.mulf %[[NEAREST]], %[[BOX]]
  // CHECK:          %[[D:[0-9]+]] = arith.subf %[[RAW]], %[[SHIFT]]
  // CHECK:          %[[R2:[0-9]+]] = vector.reduction <add>,
  // CHECK:          %[[WITHIN:[0-9]+]] = arith.cmpf olt, %[[R2]], %[[CUTOFF2]]
  // CHECK:          arith.select %[[WITHIN]], %[[D]],
  // CHECK-NOT:    memref.load %[[OUT]]
  // CHECK:        memref.store %{{[0-9]+}}, %[[OUT]][%[[I]],
  %f0 = md_exec.zeros : !vec
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms>, !vec -> !vec
  // CHECK:      return %[[OUT]]
  return %f : !vec
}

// A loop that carries fields carries their buffers. The buffer that the
// body needs beyond them is carried too, so the body does not allocate.
//
// CHECK-LABEL: func.func @steps(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[V:[a-z0-9]+]]: memref<?x3xf64>, %[[DT:[a-z0-9]+]]: f64, %[[N:[a-z0-9]+]]: index)
func.func @steps(%x: !vec, %v: !vec, %dt: f64, %n: index) -> (!vec, !vec) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:      %[[SPARE:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
  // CHECK:      %[[LOOP:[0-9]+]]:3 = scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[N]] step %{{[a-z0-9_]+}}
  // CHECK-SAME:   iter_args(%[[XA:[a-z0-9]+]] = %[[X]], %[[VA:[a-z0-9]+]] = %[[V]], %[[SA:[a-z0-9]+]] = %[[SPARE]])
  // CHECK-SAME:   -> (memref<?x3xf64>, memref<?x3xf64>, memref<?x3xf64>) {
  // CHECK-NOT:    memref.alloc
  %xe, %ve = scf.for %step = %c0 to %n step %c1
      iter_args(%xa = %x, %va = %v) -> (!vec, !vec) {
    // The scaled velocities are needed only within the step.
    //
    // CHECK:      memref.load %[[VA]][
    // CHECK:      memref.store %{{[0-9]+}}, %[[SA]][
    %w0 = md_exec.empty : !vec
    %w = md_exec.particle_for ins(%va : !vec) outs(%w0 : !vec) {
    ^bb0(%v_i: vector<3xf64>):
      %dtv = vector.broadcast %dt : f64 to vector<3xf64>
      %s = arith.mulf %dtv, %v_i : vector<3xf64>
      md_exec.yield %s : vector<3xf64>
    } -> !vec

    // CHECK:      memref.load %[[XA]][
    // CHECK:      memref.load %[[SA]][
    // CHECK:      memref.store %{{[0-9]+}}, %[[XA]][
    %x0 = md_exec.empty : !vec
    %xb = md_exec.particle_for ins(%xa, %w : !vec, !vec) outs(%x0 : !vec) {
    ^bb0(%x_i: vector<3xf64>, %w_i: vector<3xf64>):
      %s = arith.addf %x_i, %w_i : vector<3xf64>
      md_exec.yield %s : vector<3xf64>
    } -> !vec

    // CHECK:      scf.yield %[[XA]], %[[VA]], %[[SA]]
    scf.yield %xb, %va : !vec, !vec
  }
  // CHECK:      return %[[LOOP]]#0, %[[LOOP]]#1
  return %xe, %ve : !vec, !vec
}

// CHECK-NOT: md.particle_set
