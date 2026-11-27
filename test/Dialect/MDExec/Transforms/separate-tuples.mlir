// RUN: mdir-opt %s --md-exec-accumulate-destinations=separate-tuples=true | FileCheck %s
// RUN: mdir-opt %s --md-exec-accumulate-destinations | FileCheck %s --check-prefix=ONE

// D449: the loops over tuples after the loops over pairs
// accumulate into a destination of their own, so that the two chains
// write disjoint memory.

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
!inc = !mdrt.incidence<@atoms, 2, @bonds>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// A pair and two tuples: the second tuple accumulates onto the first from
// zeros of its own, not onto the pairs, and the kick adds the two chains,
// pairs + (tuples), then the velocity.
//
// CHECK-LABEL: func.func @two_chains(
// CHECK:         %[[Z1:[0-9]+]] = md_exec.zeros
// CHECK:         %[[F1:[0-9]+]] = md_exec.pair_for {{.*}} outs(%[[Z1]] :
// CHECK:         %[[Z2:[0-9]+]] = md_exec.zeros
// CHECK:         %[[F2:[0-9]+]] = md_exec.tuple_for {{.*}} outs(%[[Z2]] :
// CHECK-NOT:     md_exec.zeros
// CHECK:         %[[F3:[0-9]+]] = md_exec.tuple_for {{.*}} outs(%[[F2]] :
// CHECK:         md_exec.particle_for ins(%[[F1]], %[[F3]], %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>)
// CHECK-NEXT:    ^bb0(%[[P:[a-z0-9]+]]: vector<3xf64>, %[[T:[a-z0-9]+]]: vector<3xf64>, %[[V:[a-z0-9]+]]: vector<3xf64>):
// CHECK-NEXT:      %[[SUM:[0-9]+]] = arith.addf %[[P]], %[[T]]
// CHECK-NEXT:      %[[KICKED:[0-9]+]] = arith.addf %[[V]], %[[SUM]]
// CHECK-NEXT:      md_exec.yield %[[KICKED]]
//
// Without the option the three accumulate into one destination.
// ONE-LABEL: func.func @two_chains(
// ONE:         %[[Z1:[0-9]+]] = md_exec.zeros
// ONE:         %[[F1:[0-9]+]] = md_exec.pair_for {{.*}} outs(%[[Z1]] :
// ONE-NOT:     md_exec.zeros
// ONE:         %[[F2:[0-9]+]] = md_exec.tuple_for {{.*}} outs(%[[F1]] :
// ONE:         %[[F3:[0-9]+]] = md_exec.tuple_for {{.*}} outs(%[[F2]] :
// ONE:         md_exec.particle_for ins(%[[F3]], %{{[a-z0-9]+}} :
func.func @two_chains(%x: !vec, %cell: !md.cell, %bonds: !inc, %v: !vec)
    -> !vec {
  %z1 = md_exec.zeros : !vec
  %coverage_cells = md_exec.build_cells %x, %cell width(1.5)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %coverage_cells, %x, %cell
      cutoff(1.5) skin(0.0) kind(matrix) width(64)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %f1 = md_exec.pair_for %nl, %x, %cell outs(%z1 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !nl, !vec -> !vec
  %z2 = md_exec.zeros : !vec
  %f2 = md_exec.tuple_for %bonds, %x, %cell coordinates(displacement(0, 1))
      outs(%z2 : !vec) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %n = arith.negf %d : vector<3xf64>
    md_exec.yield %d, %n : vector<3xf64>, vector<3xf64>
  } : !inc, !vec -> !vec
  %z3 = md_exec.zeros : !vec
  %f3 = md_exec.tuple_for %bonds, %x, %cell coordinates(displacement(0, 1))
      outs(%z3 : !vec) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %n = arith.negf %d : vector<3xf64>
    md_exec.yield %n, %d : vector<3xf64>, vector<3xf64>
  } : !inc, !vec -> !vec
  %e = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%f1, %f2, %f3, %v : !vec, !vec, !vec, !vec)
      outs(%e : !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>, %c: vector<3xf64>,
       %w: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    %t = arith.addf %s, %c : vector<3xf64>
    %k = arith.addf %w, %t : vector<3xf64>
    md_exec.yield %k : vector<3xf64>
  } -> !vec
  return %v1 : !vec
}

// The sum adds the tuple first and the pairs after it: no loop over pairs
// comes before the tuples, so there is one chain, as without the option.
//
// CHECK-LABEL: func.func @tuples_first(
// CHECK:         %[[Z:[0-9]+]] = md_exec.zeros
// CHECK:         %[[F2:[0-9]+]] = md_exec.tuple_for {{.*}} outs(%[[Z]] :
// CHECK-NOT:     md_exec.zeros
// CHECK:         %[[F1:[0-9]+]] = md_exec.pair_for {{.*}} outs(%[[F2]] :
// CHECK:         md_exec.particle_for ins(%[[F1]] : !md.field<@atoms, 3 x f64>)
func.func @tuples_first(%x: !vec, %cell: !md.cell, %bonds: !inc) -> !vec {
  %z1 = md_exec.zeros : !vec
  %coverage_cells = md_exec.build_cells %x, %cell width(1.5)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %coverage_cells, %x, %cell
      cutoff(1.5) skin(0.0) kind(matrix) width(64)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %z2 = md_exec.zeros : !vec
  %f2 = md_exec.tuple_for %bonds, %x, %cell coordinates(displacement(0, 1))
      outs(%z2 : !vec) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %n = arith.negf %d : vector<3xf64>
    md_exec.yield %d, %n : vector<3xf64>, vector<3xf64>
  } : !inc, !vec -> !vec
  %f1 = md_exec.pair_for %nl, %x, %cell outs(%z1 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !nl, !vec -> !vec
  %e = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%f2, %f1 : !vec, !vec) outs(%e : !vec) {
  ^bb0(%b: vector<3xf64>, %a: vector<3xf64>):
    %s = arith.addf %b, %a : vector<3xf64>
    md_exec.yield %s : vector<3xf64>
  } -> !vec
  return %v1 : !vec
}

// After the split of one chain, the other chains of the same loop over
// particles are rewritten as before: here two loops over pairs of a second
// field accumulate into one destination.
//
// CHECK-LABEL: func.func @two_fields(
// CHECK:         %[[P:[0-9]+]] = md_exec.pair_for {{.*}} outs(%{{[0-9]+}} :
// CHECK:         %[[Q1:[0-9]+]] = md_exec.pair_for {{.*}} outs(%{{[0-9]+}} :
// CHECK:         %[[Q2:[0-9]+]] = md_exec.pair_for {{.*}} outs(%[[Q1]] :
// CHECK:         %[[T:[0-9]+]] = md_exec.tuple_for {{.*}} outs(%{{[0-9]+}} :
// CHECK:         md_exec.particle_for ins(%[[P]], %[[T]], %[[Q2]] :
// CHECK-NOT:     md_exec.split_sum
// CHECK:         return
func.func @two_fields(%x: !vec, %cell: !md.cell, %bonds: !inc)
    -> (!vec, !vec) {
  %coverage_cells = md_exec.build_cells %x, %cell width(1.5)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %coverage_cells, %x, %cell
      cutoff(1.5) skin(0.0) kind(matrix) width(64)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %z1 = md_exec.zeros : !vec
  %p = md_exec.pair_for %nl, %x, %cell outs(%z1 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !nl, !vec -> !vec
  %z3 = md_exec.zeros : !vec
  %q1 = md_exec.pair_for %nl, %x, %cell outs(%z3 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !nl, !vec -> !vec
  %z4 = md_exec.zeros : !vec
  %q2 = md_exec.pair_for %nl, %x, %cell outs(%z4 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %n = arith.negf %d : vector<3xf64>
    md_exec.yield %n : vector<3xf64>
  } : !nl, !vec -> !vec
  %z2 = md_exec.zeros : !vec
  %t = md_exec.tuple_for %bonds, %x, %cell coordinates(displacement(0, 1))
      outs(%z2 : !vec) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %n = arith.negf %d : vector<3xf64>
    md_exec.yield %d, %n : vector<3xf64>, vector<3xf64>
  } : !inc, !vec -> !vec
  %e1 = md_exec.empty : !vec
  %e2 = md_exec.empty : !vec
  %v:2 = md_exec.particle_for ins(%p, %t, %q1, %q2 : !vec, !vec, !vec, !vec)
      outs(%e1, %e2 : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>, %c: vector<3xf64>,
       %d: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    %u = arith.addf %c, %d : vector<3xf64>
    md_exec.yield %s, %u : vector<3xf64>, vector<3xf64>
  } -> !vec, !vec
  return %v#0, %v#1 : !vec, !vec
}
