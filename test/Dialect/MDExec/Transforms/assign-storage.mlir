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

// A carried field that the body never reads holds nothing from the start
// of the body, so the new value is written into its buffer and the loop
// borrows none (#19: this asserted that the buffers of a loop are
// conserved).
//
// CHECK-LABEL: func.func @unread(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[V:[a-z0-9]+]]: memref<?x3xf64>, %[[N:[a-z0-9]+]]: index)
func.func @unread(%x: !vec, %v: !vec, %n: index) -> (!vec, !vec) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT:  memref.alloc
  // CHECK:      scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[N]] step %{{[a-z0-9_]+}}
  // CHECK-SAME:   iter_args(%[[XA:[a-z0-9]+]] = %[[X]], %[[VA:[a-z0-9]+]] = %[[V]])
  // CHECK-SAME:   -> (memref<?x3xf64>, memref<?x3xf64>) {
  %xe, %ve = scf.for %step = %c0 to %n step %c1
      iter_args(%xa = %x, %va = %v) -> (!vec, !vec) {
    // CHECK:      md_exec.particle_for ins(%[[XA]] : memref<?x3xf64>) outs(%[[VA]] : memref<?x3xf64>)
    %w0 = md_exec.empty : !vec
    %vb = md_exec.particle_for ins(%xa : !vec) outs(%w0 : !vec) {
    ^bb0(%x_i: vector<3xf64>):
      md_exec.yield %x_i : vector<3xf64>
    } -> !vec
    // CHECK:      md_exec.particle_for ins(%[[XA]], %[[VA]] : memref<?x3xf64>, memref<?x3xf64>) outs(%[[XA]] : memref<?x3xf64>)
    %x0 = md_exec.empty : !vec
    %xb = md_exec.particle_for ins(%xa, %vb : !vec, !vec) outs(%x0 : !vec) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
      %s = arith.addf %x_i, %v_i : vector<3xf64>
      md_exec.yield %s : vector<3xf64>
    } -> !vec
    // CHECK:      scf.yield %[[XA]], %[[VA]]
    scf.yield %xb, %vb : !vec, !vec
  }
  return %xe, %ve : !vec, !vec
}

// A result that nothing uses, such as the velocities of a fused step whose
// loop does not carry them, holds nothing once its op has run, so the
// buffer it took is free again at the yield (#19).
//
// CHECK-LABEL: func.func @unused_result(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[N:[a-z0-9]+]]: index)
func.func @unused_result(%x: !vec, %n: index) -> !vec {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:      %[[SPARE:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
  // CHECK:      scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[N]] step %{{[a-z0-9_]+}}
  // CHECK-SAME:   iter_args(%[[XA:[a-z0-9]+]] = %[[X]], %[[SA:[a-z0-9]+]] = %[[SPARE]])
  %xe = scf.for %step = %c0 to %n step %c1 iter_args(%xa = %x) -> (!vec) {
    // CHECK:      md_exec.particle_for ins(%[[XA]] : memref<?x3xf64>) outs(%[[XA]], %[[SA]] : memref<?x3xf64>, memref<?x3xf64>)
    %w0 = md_exec.empty : !vec
    %x0 = md_exec.empty : !vec
    %w, %xb = md_exec.particle_for ins(%xa : !vec) outs(%w0, %x0 : !vec, !vec) {
    ^bb0(%x_i: vector<3xf64>):
      %s = arith.addf %x_i, %x_i : vector<3xf64>
      md_exec.yield %x_i, %s : vector<3xf64>, vector<3xf64>
    } -> !vec, !vec
    // CHECK:      scf.yield %[[SA]], %[[XA]]
    scf.yield %xb : !vec
  }
  return %xe : !vec
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

// The positions that a neighbor structure was built at are a buffer of the
// structure. The loop that tests them writes the new positions where the
// old ones were, not there.
//
// CHECK-LABEL: func.func @validity(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[V:[a-z0-9]+]]: memref<?x3xf64>,
// CHECK:         %[[NL:[0-9]+]] = md_exec.empty_neighbors size(%{{[a-z0-9_]+}}) positions(memref<?x3xf64>)
// CHECK:         scf.for
// CHECK:           %[[REF:[0-9]+]] = md_exec.reference_positions %[[NL]] : !mdrt.neighbors<@atoms> -> memref<?x3xf64>
// CHECK:           %[[MOVED:[0-9]+]] = md_exec.particle_for ins(%[[XA:[a-z0-9]+]], %[[V]], %[[REF]] :
// CHECK-SAME:        outs(%[[XA]] : memref<?x3xf64>) reduce(%{{[a-z0-9_]+}} : i1) {
// CHECK:           } -> i1
// CHECK:           md_exec.refresh_neighbors %[[NL]], %[[XA]], %{{[a-z0-9]+}} moved(%[[MOVED]])
// CHECK-SAME:        policy(check)

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

// A field in another order takes a buffer of its own, and the order takes
// one. The loop borrows them and hands back as many: the buffers of the
// fields as they were before.
//
// CHECK-LABEL: func.func @ordered(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[M:[a-z0-9]+]]: memref<?xf64>, %[[ID:[a-z0-9]+]]: memref<?xi32>,
// CHECK-DAG:     %[[B0:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?xi32>
// CHECK-DAG:     %[[B1:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
// CHECK-DAG:     %[[B2:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?xf64>
// CHECK-DAG:     %[[B3:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?xi32>
// CHECK:         scf.for
// CHECK-SAME:      iter_args(%[[XA:[a-z0-9]+]] = %[[X]], %[[MA:[a-z0-9]+]] = %[[M]], %[[IDA:[a-z0-9]+]] = %[[ID]], %[[ORDER:[a-z0-9]+]] = %[[B0]], %[[XS:[a-z0-9]+]] = %[[B1]], %[[MS:[a-z0-9]+]] = %[[B2]], %[[IDS:[a-z0-9]+]] = %[[B3]])
// CHECK:           md_exec.spatial_order %[[XA]], %{{[a-z0-9]+}}, %[[IDA]] outs(%[[ORDER]] : memref<?xi32>) width(1.100000e+00)
// CHECK:           md_exec.permute %[[XA]], %[[ORDER]] outs(%[[XS]] : memref<?x3xf64>)
// CHECK:           md_exec.permute %[[MA]], %[[ORDER]] outs(%[[MS]] : memref<?xf64>)
// CHECK:           md_exec.permute %[[IDA]], %[[ORDER]] outs(%[[IDS]] : memref<?xi32>)
// CHECK:           md_exec.particle_for ins(%[[XS]], %[[MS]] :
// CHECK-SAME:        outs(%[[XS]] : memref<?x3xf64>)
// CHECK:           call @write_ordered(%{{[a-z0-9_]+}}, %[[XS]], %[[IDS]])
// CHECK:           scf.yield %[[XS]], %[[MS]], %[[IDS]], %[[ORDER]], %[[XA]], %[[MA]], %[[IDA]]

func.func private @write_ordered(i64, memref<?x3xf64>, memref<?xi32>)

func.func @ordered(%xb: memref<?x3xf64>, %mb: memref<?xf64>,
                   %ib: memref<?xi32>, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %step = arith.constant 0 : i64
  %x0 = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
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
