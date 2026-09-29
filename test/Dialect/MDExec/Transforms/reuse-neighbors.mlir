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
