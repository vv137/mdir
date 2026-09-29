// RUN: mdir-opt %s --md-exec-assign-storage | FileCheck %s

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
  // CHECK-NOT:  memref.alloc
  // CHECK:      md_exec.particle_for ins(%[[V]], %[[F]], %[[M]] : memref<?x3xf64>, memref<?x3xf64>, memref<?xf64>) outs(%[[V]] : memref<?x3xf64>) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: vector<3xf64>, %[[MI:[a-z0-9]+]]: f64):
  // CHECK-NEXT:   arith.divf %[[DT]], %[[MI]]
  // CHECK:        md_exec.yield %{{[0-9]+}} : vector<3xf64>
  // CHECK-NEXT: }
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
  // CHECK:      md_exec.particle_for ins(%[[V]] : memref<?x3xf64>) outs(%[[NEW]] : memref<?x3xf64>)
  %w0 = md_exec.empty : !vec
  %w = md_exec.particle_for ins(%v : !vec) outs(%w0 : !vec) {
  ^bb0(%v_i: vector<3xf64>):
    %s = arith.addf %v_i, %v_i : vector<3xf64>
    md_exec.yield %s : vector<3xf64>
  } -> !vec
  // CHECK:      return %[[V]], %[[NEW]]
  return %v, %w : !vec, !vec
}

// A global sum stays a result.
//
// CHECK-LABEL: func.func @sum(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64>)
func.func @sum(%v: !vec) -> f64 {
  // CHECK:      %[[S:[0-9]+]] = md_exec.particle_for ins(%[[V]] : memref<?x3xf64>) reduce(%{{[a-z0-9_]+}} : f64) {
  // CHECK:      } -> f64
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

// A neighbor structure that is built gets storage, which a refresh with the
// policy `always` builds. A loop that accumulates from zero overwrites its
// destination. The cell stays a value.
//
// CHECK-LABEL: func.func @forces(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[CELL:[a-z0-9]+]]: !md.cell)
func.func @forces(%x: !vec, %cell: !md.cell) -> (!vec, f64) {
  // CHECK:      %[[N:[a-z0-9_]+]] = memref.dim %[[X]],
  // CHECK:      %[[STORAGE:[0-9]+]] = md_exec.empty_neighbors size(%[[N]]) positions(memref<?x3xf64>) kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  // CHECK:      %[[NL:[0-9]+]] = md_exec.refresh_neighbors %[[STORAGE]], %[[X]], %[[CELL]]
  // CHECK-SAME:   cutoff(1.500000e+00) skin(2.500000e-01) cell_width(1.750000e+00) policy(always)
  // CHECK-SAME:   : !mdrt.neighbors<@atoms>, memref<?x3xf64>
  // CHECK-NOT:  md_exec.build_cells
  %cells = md_exec.build_cells %x, %cell width(1.75)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.5) skin(0.25) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>

  // CHECK:      %[[OUT:[a-z0-9_]+]] = memref.alloc(%[[N]]) : memref<?x3xf64>
  // CHECK:      %[[U:[0-9]+]] = md_exec.pair_for %[[NL]], %[[X]], %[[CELL]]
  // CHECK-SAME:   outs(%[[OUT]] : memref<?x3xf64>) reduce(%{{[a-z0-9_]+}} : f64)
  // CHECK-SAME:   cutoff(1.500000e+00) weights [5.000000e-01] overwrite [true]
  // CHECK-SAME:   policy(directed, owner_only) {
  // CHECK:      } : !mdrt.neighbors<@atoms>, memref<?x3xf64> -> f64
  %f0 = md_exec.zeros : !vec
  %u0 = arith.constant 0.0 : f64
  %f, %u = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec)
      reduce(%u0 : f64) cutoff(1.5) weights [0.5]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d, %r2 : vector<3xf64>, f64
  } : !mdrt.neighbors<@atoms>, !vec -> !vec, f64

  // A second loop continues in the buffer of the first: it adds to what
  // the buffer holds.
  //
  // CHECK:      md_exec.pair_for %[[NL]], %[[X]], %[[CELL]]
  // CHECK-SAME:   outs(%[[OUT]] : memref<?x3xf64>) cutoff(1.500000e+00)
  // CHECK-NOT:    overwrite
  // CHECK-SAME:   policy(directed, owner_only) {
  %g = md_exec.pair_for %nl, %x, %cell outs(%f : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms>, !vec -> !vec
  // CHECK:      return %[[OUT]], %[[U]]
  return %g, %u : !vec, f64
}

// A loop that carries fields carries their buffers. The buffer that the
// body needs beyond them is carried too, so the body does not allocate. A
// neighbor structure is refreshed where it is, so the loop does not carry
// it; its storage is allocated before the loop.
//
// CHECK-LABEL: func.func @steps(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[V:[a-z0-9]+]]: memref<?x3xf64>, %[[CELL:[a-z0-9]+]]: !md.cell, %[[DT:[a-z0-9]+]]: f64, %[[N:[a-z0-9]+]]: index)
func.func @steps(%x: !vec, %v: !vec, %cell: !md.cell, %dt: f64, %n: index)
    -> (!vec, !vec, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  // CHECK-DAG:  %[[STORAGE:[0-9]+]] = md_exec.empty_neighbors size(%{{[a-z0-9_]+}}) positions(memref<?x3xf64>) kind(matrix) width(48)
  // CHECK-DAG:  %[[SPARE:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
  // CHECK:      %[[LOOP:[0-9]+]]:3 = scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[N]] step %{{[a-z0-9_]+}}
  // CHECK-SAME:   iter_args(%[[XA:[a-z0-9]+]] = %[[X]], %[[VA:[a-z0-9]+]] = %[[V]], %[[SA:[a-z0-9]+]] = %[[SPARE]])
  // CHECK-SAME:   -> (memref<?x3xf64>, memref<?x3xf64>, memref<?x3xf64>) {
  // CHECK-NOT:    memref.alloc
  %xe, %ve, %nle = scf.for %step = %c0 to %n step %c1
      iter_args(%xa = %x, %va = %v, %nla = %nl0)
      -> (!vec, !vec, !mdrt.neighbors<@atoms>) {
    // The scaled velocities are needed only within the step.
    //
    // CHECK:      md_exec.particle_for ins(%[[VA]] : memref<?x3xf64>) outs(%[[SA]] : memref<?x3xf64>)
    %w0 = md_exec.empty : !vec
    %w = md_exec.particle_for ins(%va : !vec) outs(%w0 : !vec) {
    ^bb0(%v_i: vector<3xf64>):
      %dtv = vector.broadcast %dt : f64 to vector<3xf64>
      %s = arith.mulf %dtv, %v_i : vector<3xf64>
      md_exec.yield %s : vector<3xf64>
    } -> !vec

    // CHECK:      md_exec.particle_for ins(%[[XA]], %[[SA]] : memref<?x3xf64>, memref<?x3xf64>) outs(%[[XA]] : memref<?x3xf64>)
    %x0 = md_exec.empty : !vec
    %xb = md_exec.particle_for ins(%xa, %w : !vec, !vec) outs(%x0 : !vec) {
    ^bb0(%x_i: vector<3xf64>, %w_i: vector<3xf64>):
      %s = arith.addf %x_i, %w_i : vector<3xf64>
      md_exec.yield %s : vector<3xf64>
    } -> !vec

    // CHECK:      md_exec.refresh_neighbors %[[STORAGE]], %[[XA]], %[[CELL]]
    // CHECK-SAME:   policy(check) : !mdrt.neighbors<@atoms>, memref<?x3xf64>
    %nlb = md_exec.refresh_neighbors %nla, %xb, %cell
        cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
        : !mdrt.neighbors<@atoms>, !vec

    // CHECK:      scf.yield %[[XA]], %[[VA]], %[[SA]]
    scf.yield %xb, %va, %nlb : !vec, !vec, !mdrt.neighbors<@atoms>
  }
  // CHECK:      %[[BUILDS:[0-9]+]] = md_exec.rebuild_count %[[STORAGE]]
  %builds = md_exec.rebuild_count %nle : !mdrt.neighbors<@atoms>
  // CHECK:      return %[[LOOP]]#0, %[[LOOP]]#1, %[[BUILDS]]
  return %xe, %ve, %builds : !vec, !vec, i64
}

// A destination of zeros that is read like any other field is filled.
//
// CHECK-LABEL: func.func @zeros(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64>)
func.func @zeros(%v: !vec) -> !vec {
  // CHECK:      %[[Z:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
  // CHECK:      md_exec.particle_for outs(%[[Z]] : memref<?x3xf64>) {
  // CHECK-NEXT:   %[[ZERO:[a-z0-9_]+]] = arith.constant dense<0.000000e+00> : vector<3xf64>
  // CHECK-NEXT:   md_exec.yield %[[ZERO]] : vector<3xf64>
  // CHECK:      md_exec.particle_for ins(%[[V]], %[[Z]] : memref<?x3xf64>, memref<?x3xf64>) outs(%[[V]] : memref<?x3xf64>)
  %z = md_exec.zeros : !vec
  %e = md_exec.empty : !vec
  %w = md_exec.particle_for ins(%v, %z : !vec, !vec) outs(%e : !vec) {
  ^bb0(%v_i: vector<3xf64>, %z_i: vector<3xf64>):
    %s = arith.addf %v_i, %z_i : vector<3xf64>
    md_exec.yield %s : vector<3xf64>
  } -> !vec
  return %w : !vec
}

// A function without fields is left as it is.
//
// CHECK-LABEL: func.func @plain(
// CHECK-NEXT:    %[[S:[a-z0-9]+]] = arith.addf
// CHECK-NEXT:    return %[[S]]
func.func @plain(%a: f64) -> f64 {
  %s = arith.addf %a, %a : f64
  return %s : f64
}

// An empty structure inside a loop: the storage is allocated once, before
// the loops, and reset where the empty structure is used, so that every
// iteration of the outer loop begins with a structure that is built.
//
// CHECK-LABEL: func.func @segments(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[CELL:[a-z0-9]+]]: !md.cell,
func.func @segments(%x: !vec, %cell: !md.cell, %segments: index,
                    %steps: index) -> !vec {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:      %[[STORAGE:[0-9]+]] = md_exec.empty_neighbors size(%{{[a-z0-9_]+}}) positions(memref<?x3xf64>)
  // CHECK:      scf.for
  // CHECK-NEXT:   md_exec.reset_neighbors %[[STORAGE]] : !mdrt.neighbors<@atoms>
  // CHECK-NEXT:   scf.for
  // CHECK:          md_exec.refresh_neighbors %[[STORAGE]],
  %xe = scf.for %segment = %c0 to %segments step %c1
      iter_args(%xs = %x) -> (!vec) {
    %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
        : !mdrt.neighbors<@atoms>
    %xi, %nli = scf.for %step = %c0 to %steps step %c1
        iter_args(%xa = %xs, %nla = %nl0)
        -> (!vec, !mdrt.neighbors<@atoms>) {
      %nlb = md_exec.refresh_neighbors %nla, %xa, %cell
          cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
          : !mdrt.neighbors<@atoms>, !vec
      %f0 = md_exec.zeros : !vec
      %f = md_exec.pair_for %nlb, %xa, %cell outs(%f0 : !vec) cutoff(1.5)
          policy(directed, owner_only) {
      ^bb0(%r2: f64, %d: vector<3xf64>):
        md_exec.yield %d : vector<3xf64>
      } : !mdrt.neighbors<@atoms>, !vec -> !vec
      %e = md_exec.empty : !vec
      %xb = md_exec.particle_for ins(%xa, %f : !vec, !vec) outs(%e : !vec) {
      ^bb0(%x_i: vector<3xf64>, %f_i: vector<3xf64>):
        %s = arith.addf %x_i, %f_i : vector<3xf64>
        md_exec.yield %s : vector<3xf64>
      } -> !vec
      scf.yield %xb, %nlb : !vec, !mdrt.neighbors<@atoms>
    }
    scf.yield %xi : !vec
  }
  return %xe : !vec
}

// The host reads a field where it is. The program goes on using the
// buffer.
//
// CHECK-LABEL: func.func @frames(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[STEP:[a-z0-9]+]]: i64)
func.func private @write_frame(i64, memref<?x3xf64>)

func.func @frames(%x: !vec, %step: i64) -> !vec {
  // CHECK:      call @write_frame(%[[STEP]], %[[X]])
  // CHECK:      md_exec.particle_for ins(%[[X]] : memref<?x3xf64>) outs(%[[X]] : memref<?x3xf64>)
  mdrt.host_call @write_frame(%step, %x) : (i64, !vec)
  %e = md_exec.empty : !vec
  %y = md_exec.particle_for ins(%x : !vec) outs(%e : !vec) {
  ^bb0(%x_i: vector<3xf64>):
    %s = arith.addf %x_i, %x_i : vector<3xf64>
    md_exec.yield %s : vector<3xf64>
  } -> !vec
  return %y : !vec
}
