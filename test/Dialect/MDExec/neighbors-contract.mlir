// The layout contract of a neighbor structure (D[neighbors-contract], #881):
// the type goes through a refresh, the values that a loop carries, and the
// results of an scf.if unchanged; each consumer whose requirement it meets
// is accepted; the cast that weakens it folds away where it changes nothing.
//
// RUN: mdir-opt %s | FileCheck %s
// RUN: mdir-opt %s --canonicalize | FileCheck %s --check-prefix=FOLD

!vec = !md.field<@atoms, 3 x f64>
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
md.particle_set @atoms

// A structure of groups, refreshed in a loop that carries it and chosen by
// an scf.if: every value keeps the type.
// CHECK-LABEL: func.func @carried(
// CHECK:         md_exec.empty_neighbors kind(groups) width(64) : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
// CHECK:         scf.for {{.*}} -> (!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>, f64)
// CHECK:           md_exec.refresh_neighbors {{.*}} : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>, !md.field<@atoms, 3 x f64>
// CHECK:           md_exec.pair_for {{.*}} policy(unique, atomic)
// CHECK:           scf.yield {{.*}} : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>, f64
// CHECK:         scf.if {{.*}} -> (!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>)
// CHECK:         md_exec.rebuild_count {{.*}} : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
func.func @carried(%x: !vec, %cell: !md.cell, %n: index, %c: i1) -> (f64, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %u0 = arith.constant 0.0 : f64
  %nl0 = md_exec.empty_neighbors kind(groups) width(64) : !groups
  %nl, %e = scf.for %i = %c0 to %n step %c1 iter_args(%s = %nl0, %t = %u0)
      -> (!groups, f64) {
    %s1 = md_exec.refresh_neighbors %s, %x, %cell cutoff(1.0) skin(0.1)
        cell_width(1.1) policy(check) : !groups, !vec
    %u = md_exec.pair_for %s1, %x, %cell reduce(%u0 : f64) cutoff(1.0)
        exchange [symmetric] policy(unique, atomic) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !groups, !vec -> f64
    scf.yield %s1, %u : !groups, f64
  }
  %chosen = scf.if %c -> !groups {
    scf.yield %nl : !groups
  } else {
    scf.yield %nl0 : !groups
  }
  %builds = md_exec.rebuild_count %chosen : !groups
  return %e, %builds : f64, i64
}

// The matrix meets every consumer of its rows: owner_only, a fixed_order sum
// (the deterministic mode, D301), the two-ended loop of D290, and the
// triplets (D160).
// CHECK-LABEL: func.func @matrix(
// CHECK:         md_exec.build_triplets
func.func @matrix(%x: !vec, %cell: !md.cell, %f0: !vec) -> (!vec, f64, !vec) {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec -> !vec
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } {md_exec.algebra = #md_exec.algebra<fixed_order>}
    : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec -> f64
  %g = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      exchange [antisymmetric] policy(directed, atomic) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec -> !vec
  %t = md_exec.build_triplets %nl, %x, %cell cutoff(1.0)
      : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec -> !md.relation<@atoms, 3, reversal>
  return %f, %u, %g : !vec, f64, !vec
}

// A weaker type meets what needs less: owner_only over a matrix weakened to
// its grouping, and the two-ended loop over one weakened to no order. The
// loop over groups with a fixed_order sum passes the type (its order is
// stable); its lowering refuses it (#877, item 4).
// CHECK-LABEL: func.func @weakened(
// CHECK:         md_exec.weaken_neighbors %{{.*}} : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)> to !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>
// CHECK:         md_exec.weaken_neighbors %{{.*}} : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)> to !mdrt.neighbors<@atoms, directed, minimum, masked, none>
// FOLD-LABEL: func.func @weakened(
// FOLD:         md_exec.weaken_neighbors
// FOLD:         md_exec.weaken_neighbors
// FOLD-NOT:     md_exec.weaken_neighbors
// FOLD:         return
func.func @weakened(%x: !vec, %cell: !md.cell, %f0: !vec) -> (!vec, !vec, f64) {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  %grouped = md_exec.weaken_neighbors %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      to !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>
  %f = md_exec.pair_for %grouped, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>, !vec -> !vec
  %none = md_exec.weaken_neighbors %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      to !mdrt.neighbors<@atoms, directed, minimum, masked, none>
  %g = md_exec.pair_for %none, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      exchange [antisymmetric] policy(directed, atomic) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, none>, !vec -> !vec
  %groups = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(groups) width(64) : !mdrt.cells<@atoms>, !vec -> !groups
  %same = md_exec.weaken_neighbors %groups : !groups to !groups
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %same, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      exchange [symmetric] policy(unique, atomic) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } {md_exec.algebra = #md_exec.algebra<fixed_order>} : !groups, !vec -> f64
  return %f, %g, %u : !vec, !vec, f64
}

// The rows of a domain (D290): explicit images, through a refresh.
// CHECK-LABEL: func.func @domain(
// CHECK:         md_exec.refresh_neighbors {{.*}} images(explicit) policy(always) : !mdrt.neighbors<@atoms, directed, explicit, masked, stable(owner, search)>
func.func @domain(%x: !vec, %cell: !md.cell, %n: index) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl0 = md_exec.build_neighbors %cells, %x, %cell centers(%n) cutoff(1.0)
      skin(0.0) kind(matrix) width(64) images(explicit) : !mdrt.cells<@atoms>, !vec
      -> !mdrt.neighbors<@atoms, directed, explicit, masked, stable(owner, search)>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell centers(%n) cutoff(1.0)
      skin(0.0) cell_width(1.0) images(explicit) policy(always)
      : !mdrt.neighbors<@atoms, directed, explicit, masked, stable(owner, search)>, !vec
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell centers(%n) reduce(%u0 : f64) cutoff(1.0)
      images(explicit) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms, directed, explicit, masked, stable(owner, search)>, !vec -> f64
  return %u : f64
}
