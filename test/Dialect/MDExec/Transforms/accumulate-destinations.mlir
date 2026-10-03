// RUN: mdir-opt %s --md-exec-accumulate-destinations | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>
!inc = !mdrt.incidence<@atoms, 2, @bonds>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// The forces of two terms, added by a loop over particles that kicks: the
// second loop accumulates onto the first, and the kick reads the last.
//
// CHECK-LABEL: func.func @terms(
// CHECK:         %[[Z:[0-9]+]] = md_exec.zeros
// CHECK:         %[[F1:[0-9]+]] = md_exec.pair_for {{.*}} outs(%[[Z]] :
// CHECK-NOT:     md_exec.zeros
// CHECK:         %[[F2:[0-9]+]] = md_exec.tuple_for {{.*}} outs(%[[F1]] :
// CHECK:         md_exec.particle_for ins(%[[F2]], %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>)
// CHECK-NEXT:    ^bb0(%[[SUM:[a-z0-9]+]]: vector<3xf64>, %[[V:[a-z0-9]+]]: vector<3xf64>):
// CHECK-NEXT:      %[[KICKED:[0-9]+]] = arith.addf %[[V]], %[[SUM]]
// CHECK-NEXT:      md_exec.yield %[[KICKED]]
func.func @terms(%x: !vec, %cell: !md.cell, %nl: !nl, %bonds: !inc, %v: !vec)
    -> !vec {
  %z1 = md_exec.zeros : !vec
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
  %e = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%f1, %f2, %v : !vec, !vec, !vec)
      outs(%e : !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>, %w: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    %k = arith.addf %w, %s : vector<3xf64>
    md_exec.yield %k : vector<3xf64>
  } -> !vec
  return %v1 : !vec
}

// The loops come in the other order than the sum: they keep their
// destinations.
//
// CHECK-LABEL: func.func @reversed(
// CHECK:         md_exec.particle_for ins(%{{[0-9]+}}, %{{[0-9]+}} :
func.func @reversed(%x: !vec, %cell: !md.cell, %nl: !nl, %bonds: !inc) -> !vec {
  %z1 = md_exec.zeros : !vec
  %f1 = md_exec.pair_for %nl, %x, %cell outs(%z1 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !nl, !vec -> !vec
  %z2 = md_exec.zeros : !vec
  %f2 = md_exec.tuple_for %bonds, %x, %cell coordinates(displacement(0, 1))
      outs(%z2 : !vec) arity(2) {
  ^bb0(%d: vector<3xf64>):
    md_exec.yield %d, %d : vector<3xf64>, vector<3xf64>
  } : !inc, !vec -> !vec
  %e = md_exec.empty : !vec
  %s = md_exec.particle_for ins(%f2, %f1 : !vec, !vec) outs(%e : !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %t = arith.addf %a, %b : vector<3xf64>
    md_exec.yield %t : vector<3xf64>
  } -> !vec
  return %s : !vec
}

// One loop gives two destinations that the kick adds one after the other,
// as the forces of the sums over the centers of groups (D139): the loop
// yields the sum of the two contributions of each member into the first,
// and then accumulates onto the loop before it.
//
// CHECK-LABEL: func.func @merged(
// CHECK:         %[[F1:[0-9]+]] = md_exec.pair_for
// CHECK:         %[[F2:[0-9]+]] = md_exec.tuple_for {{.*}} outs(%[[F1]] : !md.field<@atoms, 3 x f64>) arity(2) {
// CHECK:           %[[N:[0-9]+]] = arith.negf %[[D:[a-z0-9]+]]
// CHECK:           %[[M:[0-9]+]] = arith.mulf %[[D]], %[[D]]
// CHECK:           %[[P:[0-9]+]] = arith.addf %[[D]], %[[M]]
// CHECK:           %[[Q:[0-9]+]] = arith.addf %[[N]], %[[M]]
// CHECK:           md_exec.yield %[[P]], %[[Q]] : vector<3xf64>, vector<3xf64>
// CHECK:         } : !mdrt.incidence<@atoms, 2, @bonds>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f64>
// CHECK-NOT:     md_exec.zeros
// CHECK:         md_exec.particle_for ins(%[[F2]], %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>)
func.func @merged(%x: !vec, %cell: !md.cell, %nl: !nl, %bonds: !inc, %v: !vec)
    -> !vec {
  %z1 = md_exec.zeros : !vec
  %f1 = md_exec.pair_for %nl, %x, %cell outs(%z1 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !nl, !vec -> !vec
  %z2 = md_exec.zeros : !vec
  %z3 = md_exec.zeros : !vec
  %f2:2 = md_exec.tuple_for %bonds, %x, %cell coordinates(displacement(0, 1))
      outs(%z2, %z3 : !vec, !vec) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %n = arith.negf %d : vector<3xf64>
    %m = arith.mulf %d, %d : vector<3xf64>
    md_exec.yield %d, %n, %m, %m
        : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !inc, !vec -> !vec, !vec
  %e = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%f1, %f2#0, %f2#1, %v : !vec, !vec, !vec, !vec)
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
