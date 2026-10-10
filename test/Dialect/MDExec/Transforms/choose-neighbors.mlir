// RUN: mdir-opt %s --md-exec-choose-neighbors | FileCheck %s
// RUN: mdir-opt %s --md-exec-choose-neighbors=kind=matrix \
// RUN: | FileCheck %s --check-prefix=MATRIX
// RUN: not mdir-opt %s --md-exec-choose-neighbors=kind=tiles 2>&1 \
// RUN: | FileCheck %s --check-prefix=KIND
// RUN: mdir-opt %s --md-exec-expose-validity --md-exec-choose-neighbors \
// RUN: | FileCheck %s --check-prefix=DUAL

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// A structure carried by a loop and refreshed in it becomes groups when
// every loop over it has a contract for each value: an antisymmetric
// destination and a symmetric sum. Its loops take each pair once (D89).
//
// CHECK-LABEL: func.func @groups(
// CHECK:         md_exec.empty_neighbors kind(groups)
// CHECK:         scf.for
// CHECK:           md_exec.refresh_neighbors
// CHECK:           md_exec.pair_for {{.*}} policy(unique, atomic)
// MATRIX-LABEL: func.func @groups(
// MATRIX:         md_exec.empty_neighbors kind(matrix)
// MATRIX:         md_exec.pair_for {{.*}} policy(directed, owner_only)
// KIND: expected the kind 'groups' or 'matrix', got 'tiles'
func.func @groups(%x: !vec, %cell: !md.cell, %n: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %u0 = arith.constant 0.0 : f64
  %nl0 = md_exec.empty_neighbors kind(matrix) width(96) : !nl
  %nl, %e = scf.for %i = %c0 to %n step %c1 iter_args(%s = %nl0, %t = %u0)
      -> (!nl, f64) {
    %s1 = md_exec.refresh_neighbors %s, %x, %cell
        cutoff(2.5) skin(0.3) cell_width(2.8) policy(check) : !nl, !vec
    %f0 = md_exec.zeros : !vec
    %f, %u = md_exec.pair_for %s1, %x, %cell outs(%f0 : !vec)
        reduce(%u0 : f64) cutoff(2.5) weights [0.5]
        exchange [antisymmetric, symmetric] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      %g = vector.broadcast %r2 : f64 to vector<3xf64>
      %k = arith.mulf %g, %d : vector<3xf64>
      md_exec.yield %k, %r2 : vector<3xf64>, f64
    } : !nl, !vec -> !vec, f64
    %t1 = arith.addf %t, %u : f64
    scf.yield %s1, %t1 : !nl, f64
  }
  return %e : f64
}

// A loop without a contract for its sum keeps the matrix, and so does the
// other loop over the same structure.
//
// CHECK-LABEL: func.func @matrix(
// CHECK:         md_exec.build_neighbors {{.*}} kind(matrix)
// CHECK:         md_exec.pair_for {{.*}} policy(directed, owner_only)
// CHECK:         md_exec.pair_for {{.*}} policy(directed, owner_only)
func.func @matrix(%x: !vec, %cell: !md.cell) -> (f64, f64) {
  %u0 = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !md.field<@atoms, 3 x f64> -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(8)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %a = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      weights [0.5] exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  %b = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %a, %b : f64, f64
}

// Under a dual list (D114) every refresh of a run has the reach and the
// test of an inner list. A structure that becomes groups keeps them; one
// that stays a matrix, here for a sum without a contract, keeps one list:
// its refresh loses them, and what the test read of the pruning reads the
// build (D[python-groups]).
//
// DUAL-LABEL: func.func @dual(
// DUAL-DAG:     %[[G0:[a-z0-9]+]] = md_exec.empty_neighbors kind(groups)
// DUAL-DAG:     %[[M0:[a-z0-9]+]] = md_exec.empty_neighbors kind(matrix)
// DUAL:         scf.for {{.*}} iter_args(%[[G:[a-z0-9]+]] = %[[G0]], %[[M:[a-z0-9]+]] = %[[M0]]
// DUAL:           md_exec.reference_positions %[[G]] pruned
// DUAL:           md_exec.reference_cell %[[G]] pruned
// DUAL:           md_exec.refresh_neighbors %[[G]], {{.*}} stale({{.*}}) cutoff({{.*}} prune_skin(1.000000e-01)
// DUAL-NOT:       pruned
// DUAL:           md_exec.refresh_neighbors %[[M]], {{.*}} moved(%{{[0-9]+}}) cutoff(
// DUAL-NOT:       prune_skin
// DUAL:           md_exec.pair_for {{.*}} policy(unique, atomic)
// DUAL:           md_exec.pair_for {{.*}} policy(directed, owner_only)
func.func @dual(%x: !vec, %cell: !md.cell, %n: index) -> (f64, f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %u0 = arith.constant 0.0 : f64
  %g0 = md_exec.empty_neighbors kind(matrix) width(96) : !nl
  %m0 = md_exec.empty_neighbors kind(matrix) width(96) : !nl
  %g, %m, %a, %b = scf.for %i = %c0 to %n step %c1
      iter_args(%s = %g0, %t = %m0, %p = %u0, %q = %u0)
      -> (!nl, !nl, f64, f64) {
    %s1 = md_exec.refresh_neighbors %s, %x, %cell
        cutoff(2.5) skin(0.3) prune_skin(0.1) cell_width(2.8) policy(check)
        : !nl, !vec
    %t1 = md_exec.refresh_neighbors %t, %x, %cell
        cutoff(2.5) skin(0.3) prune_skin(0.1) cell_width(2.8) policy(check)
        : !nl, !vec
    %u = md_exec.pair_for %s1, %x, %cell reduce(%u0 : f64) cutoff(2.5)
        weights [0.5] exchange [symmetric] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %w = md_exec.pair_for %t1, %x, %cell reduce(%u0 : f64) cutoff(2.5)
        weights [0.5] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %p1 = arith.addf %p, %u : f64
    %q1 = arith.addf %q, %w : f64
    scf.yield %s1, %t1, %p1, %q1 : !nl, !nl, f64, f64
  }
  return %a, %b : f64, f64
}
