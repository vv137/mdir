// RUN: mdir-opt %s --md-exec-reuse-neighbors | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// The structure becomes a value that the loop carries. It is empty when the
// loop begins.
//
// CHECK-LABEL: func.func @steps(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell, %[[N:[a-z0-9]+]]: index)
func.func @steps(%x: !vec, %cell: !md.cell, %n: index) -> (!vec, f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  // CHECK:      %[[NL0:[0-9]+]] = md_exec.empty_neighbors kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  // CHECK:      %[[LOOP:[0-9]+]]:3 = scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[N]] step %{{[a-z0-9_]+}}
  // CHECK-SAME:   iter_args(%[[XA:[a-z0-9]+]] = %[[X]], %[[SUM:[a-z0-9]+]] = %{{[a-z0-9_]+}}, %[[NA:[a-z0-9]+]] = %[[NL0]])
  // CHECK-SAME:   -> (!md.field<@atoms, 3 x f64>, f64, !mdrt.neighbors<@atoms>) {
  %xe, %total = scf.for %step = %c0 to %n step %c1
      iter_args(%xa = %x, %sum = %zero) -> (!vec, f64) {
    // CHECK-NOT:  md_exec.build_cells
    // CHECK-NOT:  md_exec.build_neighbors
    // CHECK:      %[[NB:[0-9]+]] = md_exec.refresh_neighbors %[[NA]], %[[XA]], %[[CELL]]
    // CHECK-SAME:   cutoff(1.500000e+00) skin(2.500000e-01) cell_width(1.750000e+00) policy(check)
    %cells = md_exec.build_cells %xa, %cell width(1.75)
        : !vec -> !mdrt.cells<@atoms>
    %nl = md_exec.build_neighbors %cells, %xa, %cell
        cutoff(1.5) skin(0.25) kind(matrix) width(48)
        : !mdrt.cells<@atoms>, !vec -> !nl

    // CHECK:      md_exec.pair_for %[[NB]], %[[XA]], %[[CELL]]
    %none = arith.constant 0.0 : f64
    %pairs = md_exec.pair_for %nl, %xa, %cell reduce(%none : f64)
        cutoff(1.5) weights [0.5] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %next = arith.addf %sum, %pairs : f64

    // CHECK:      scf.yield %[[XA]], %{{[0-9]+}}, %[[NB]]
    scf.yield %xa, %next : !vec, f64
  }
  // CHECK:      return %[[LOOP]]#0, %[[LOOP]]#1
  return %xe, %total : !vec, f64
}

// A build outside a loop stays.
//
// CHECK-LABEL: func.func @once(
func.func @once(%x: !vec, %cell: !md.cell) -> f64 {
  // CHECK:      md_exec.build_cells
  // CHECK:      md_exec.build_neighbors
  // CHECK-NOT:  md_exec.refresh_neighbors
  %cells = md_exec.build_cells %x, %cell width(1.75)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.5) skin(0.25) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %none = arith.constant 0.0 : f64
  %pairs = md_exec.pair_for %nl, %x, %cell reduce(%none : f64)
      cutoff(1.5) weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %pairs : f64
}

// A structure moves outward through the loops around it, so that it lives
// on from one output to the next.
//
// CHECK-LABEL: func.func @nested(
func.func @nested(%x0: !vec, %cell: !md.cell, %frames: index, %steps: index)
    -> !vec {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:      %[[EMPTY:[0-9]+]] = md_exec.empty_neighbors kind(matrix) width(96)
  // CHECK:      scf.for {{.*}} iter_args(%{{[a-z0-9]+}} = %{{[a-z0-9]+}}, %[[OUTER:[a-z0-9]+]] = %[[EMPTY]])
  // CHECK-NOT:    md_exec.empty_neighbors
  // CHECK:        %[[INNER:[0-9]+]]:2 = scf.for {{.*}} iter_args(%{{[a-z0-9]+}} = %{{[a-z0-9]+}}, %[[NL:[a-z0-9]+]] = %[[OUTER]])
  // CHECK:          md_exec.refresh_neighbors %[[NL]],
  // CHECK:        scf.yield %[[INNER]]#0, %[[INNER]]#1
  %x = scf.for %frame = %c0 to %frames step %c1
      iter_args(%xf = %x0) -> (!vec) {
    %xs = scf.for %step = %c0 to %steps step %c1
        iter_args(%xa = %xf) -> (!vec) {
      %cells = md_exec.build_cells %xa, %cell width(2.8)
          : !vec -> !mdrt.cells<@atoms>
      %nl = md_exec.build_neighbors %cells, %xa, %cell
          cutoff(2.5) skin(0.3) kind(matrix) width(96)
          : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
      %f0 = md_exec.zeros : !vec
      %f = md_exec.pair_for %nl, %xa, %cell outs(%f0 : !vec) cutoff(2.5)
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
      scf.yield %xb : !vec
    }
    scf.yield %xs : !vec
  }
  return %x : !vec
}

// The iterations of a loop that is marked are segments of the run: a
// structure starts empty in each, so it does not move beyond the body of
// that loop.
//
// CHECK-LABEL: func.func @segments(
func.func @segments(%x0: !vec, %cell: !md.cell, %segments: index,
                    %steps: index) -> !vec {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT:  md_exec.empty_neighbors
  // CHECK:      scf.for {{.*}} iter_args(%{{[a-z0-9]+}} = %{{[a-z0-9]+}}) -> (!md.field<@atoms, 3 x f64>) {
  // CHECK-NEXT:   %[[EMPTY:[0-9]+]] = md_exec.empty_neighbors kind(matrix) width(96)
  // CHECK-NEXT:   scf.for {{.*}} iter_args(%{{[a-z0-9]+}} = %{{[a-z0-9]+}}, %{{[a-z0-9]+}} = %[[EMPTY]])
  // CHECK:      } {mdrt.segment}
  %x = scf.for %segment = %c0 to %segments step %c1
      iter_args(%xf = %x0) -> (!vec) {
    %xs = scf.for %step = %c0 to %steps step %c1
        iter_args(%xa = %xf) -> (!vec) {
      %cells = md_exec.build_cells %xa, %cell width(2.8)
          : !vec -> !mdrt.cells<@atoms>
      %nl = md_exec.build_neighbors %cells, %xa, %cell
          cutoff(2.5) skin(0.3) kind(matrix) width(96)
          : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
      %f0 = md_exec.zeros : !vec
      %f = md_exec.pair_for %nl, %xa, %cell outs(%f0 : !vec) cutoff(2.5)
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
      scf.yield %xb : !vec
    }
    scf.yield %xs : !vec
  } {mdrt.segment}
  return %x : !vec
}

// Structures that are built with the same parameters, one after the other,
// share their storage: the second build refreshes what the first left. A
// structure that is built with another cutoff has storage of its own.
//
// CHECK-LABEL: func.func @shared(
func.func @shared(%x0: !vec, %cell: !md.cell, %steps: index) -> !vec {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:      %[[A:[0-9]+]] = md_exec.empty_neighbors kind(matrix) width(96)
  // CHECK:      %[[B:[0-9]+]] = md_exec.empty_neighbors kind(matrix) width(96)
  // CHECK:      scf.for {{.*}} iter_args(%{{[a-z0-9]+}} = %{{[a-z0-9]+}}, %[[NA:[a-z0-9]+]] = %[[A]], %[[NB:[a-z0-9]+]] = %[[B]])
  // CHECK:        %[[FIRST:[0-9]+]] = md_exec.refresh_neighbors %[[NA]], {{.*}} cutoff(2.500000e+00)
  // CHECK:        %[[OTHER:[0-9]+]] = md_exec.refresh_neighbors %[[NB]], {{.*}} cutoff(2.000000e+00)
  // CHECK:        %[[SECOND:[0-9]+]] = md_exec.refresh_neighbors %[[FIRST]], {{.*}} cutoff(2.500000e+00)
  // CHECK:        scf.yield %{{[0-9]+}}, %[[SECOND]], %[[OTHER]]
  %x = scf.for %step = %c0 to %steps step %c1
      iter_args(%xa = %x0) -> (!vec) {
    %cells1 = md_exec.build_cells %xa, %cell width(2.8)
        : !vec -> !mdrt.cells<@atoms>
    %nl1 = md_exec.build_neighbors %cells1, %xa, %cell
        cutoff(2.5) skin(0.3) kind(matrix) width(96)
        : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
    %f0 = md_exec.zeros : !vec
    %f = md_exec.pair_for %nl1, %xa, %cell outs(%f0 : !vec) cutoff(2.5)
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

    %cells2 = md_exec.build_cells %xb, %cell width(2.8)
        : !vec -> !mdrt.cells<@atoms>
    %nl2 = md_exec.build_neighbors %cells2, %xb, %cell
        cutoff(2.0) skin(0.3) kind(matrix) width(96)
        : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
    %g0 = md_exec.zeros : !vec
    %g = md_exec.pair_for %nl2, %xb, %cell outs(%g0 : !vec) cutoff(2.0)
        policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %d : vector<3xf64>
    } : !mdrt.neighbors<@atoms>, !vec -> !vec

    %cells3 = md_exec.build_cells %xb, %cell width(2.8)
        : !vec -> !mdrt.cells<@atoms>
    %nl3 = md_exec.build_neighbors %cells3, %xb, %cell
        cutoff(2.5) skin(0.3) kind(matrix) width(96)
        : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
    %h = md_exec.pair_for %nl3, %xb, %cell outs(%g : !vec) cutoff(2.5)
        policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %d : vector<3xf64>
    } : !mdrt.neighbors<@atoms>, !vec -> !vec
    %e2 = md_exec.empty : !vec
    %xc = md_exec.particle_for ins(%xb, %h : !vec, !vec) outs(%e2 : !vec) {
    ^bb0(%x_i: vector<3xf64>, %f_i: vector<3xf64>):
      %s = arith.addf %x_i, %f_i : vector<3xf64>
      md_exec.yield %s : vector<3xf64>
    } -> !vec
    scf.yield %xc : !vec
  }
  return %x : !vec
}

// A build at the positions and in the cell of the last refresh of its
// structure, with its parameters, repeats that refresh and is removed: the
// potential of an output at the state of a step reads the structure of the
// step, in a loop over states as well, and does not act on it (#233). In
// another cell it is a refresh.
//
// CHECK-LABEL: func.func @repeated(
// CHECK-SAME:    %{{[a-z0-9]+}}: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell, %[[OTHER:[a-z0-9]+]]: !md.cell,
func.func @repeated(%x0: !vec, %cell: !md.cell, %other: !md.cell,
                    %steps: index, %states: index) -> (!vec, f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  // CHECK:      scf.for {{.*}} iter_args(%[[XA:[a-z0-9]+]] = %{{[a-z0-9]+}}, %{{[a-z0-9]+}} = %{{[a-z0-9_]+}}, %[[NA:[a-z0-9]+]] = %{{[0-9]+}})
  %x, %total = scf.for %step = %c0 to %steps step %c1
      iter_args(%xa = %x0, %sum = %zero) -> (!vec, f64) {
    // The step.
    // CHECK:        %[[STEP:[0-9]+]] = md_exec.refresh_neighbors %[[NA]], %[[XA]], %[[CELL]]
    // CHECK:        md_exec.pair_for %[[STEP]], %[[XA]], %[[CELL]]
    %cells1 = md_exec.build_cells %xa, %cell width(1.75)
        : !vec -> !mdrt.cells<@atoms>
    %nl1 = md_exec.build_neighbors %cells1, %xa, %cell
        cutoff(1.5) skin(0.25) kind(matrix) width(48)
        : !mdrt.cells<@atoms>, !vec -> !nl
    %none = arith.constant 0.0 : f64
    %a = md_exec.pair_for %nl1, %xa, %cell reduce(%none : f64)
        cutoff(1.5) weights [0.5] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64

    // An output at the state of the step.
    // CHECK-NOT:    md_exec.refresh_neighbors
    // CHECK:        md_exec.pair_for %[[STEP]], %[[XA]], %[[CELL]]
    %cells2 = md_exec.build_cells %xa, %cell width(1.75)
        : !vec -> !mdrt.cells<@atoms>
    %nl2 = md_exec.build_neighbors %cells2, %xa, %cell
        cutoff(1.5) skin(0.25) kind(matrix) width(48)
        : !mdrt.cells<@atoms>, !vec -> !nl
    %b = md_exec.pair_for %nl2, %xa, %cell reduce(%none : f64)
        cutoff(1.5) weights [0.5] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64

    // Its loop over states.
    // CHECK:        %[[STATES:[0-9]+]]:2 = scf.for {{.*}} iter_args(%{{[a-z0-9]+}} = %{{[0-9]+}}, %[[NI:[a-z0-9]+]] = %[[STEP]])
    // CHECK-NOT:      md_exec.refresh_neighbors
    // CHECK:          md_exec.pair_for %[[NI]], %[[XA]], %[[CELL]]
    // CHECK:          scf.yield %{{[0-9]+}}, %[[NI]]
    %c = scf.for %state = %c0 to %states step %c1
        iter_args(%acc = %b) -> (f64) {
      %cells3 = md_exec.build_cells %xa, %cell width(1.75)
          : !vec -> !mdrt.cells<@atoms>
      %nl3 = md_exec.build_neighbors %cells3, %xa, %cell
          cutoff(1.5) skin(0.25) kind(matrix) width(48)
          : !mdrt.cells<@atoms>, !vec -> !nl
      %p = md_exec.pair_for %nl3, %xa, %cell reduce(%none : f64)
          cutoff(1.5) weights [0.5] policy(directed, owner_only) {
      ^bb0(%r2: f64, %d: vector<3xf64>):
        md_exec.yield %r2 : f64
      } : !nl, !vec -> f64
      %more = arith.addf %acc, %p : f64
      scf.yield %more : f64
    }

    // The same positions in another cell.
    // CHECK:        %[[ELSEWHERE:[0-9]+]] = md_exec.refresh_neighbors %[[STATES]]#1, %[[XA]], %[[OTHER]]
    // CHECK:        md_exec.pair_for %[[ELSEWHERE]], %[[XA]], %[[OTHER]]
    %cells4 = md_exec.build_cells %xa, %other width(1.75)
        : !vec -> !mdrt.cells<@atoms>
    %nl4 = md_exec.build_neighbors %cells4, %xa, %other
        cutoff(1.5) skin(0.25) kind(matrix) width(48)
        : !mdrt.cells<@atoms>, !vec -> !nl
    %e = md_exec.pair_for %nl4, %xa, %other reduce(%none : f64)
        cutoff(1.5) weights [0.5] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64

    %ab = arith.addf %a, %c : f64
    %abe = arith.addf %ab, %e : f64
    %next = arith.addf %sum, %abe : f64
    // CHECK:        scf.yield %[[XA]], %{{[0-9]+}}, %[[ELSEWHERE]]
    scf.yield %xa, %next : !vec, f64
  }
  return %x, %total : !vec, f64
}
