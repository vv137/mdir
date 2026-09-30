// RUN: mdir-opt %s --md-bypass-updates | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered, @pairs>
!trios = !md.relation<@atoms, 3, ordered, @trios>
!other = !md.relation<@atoms, 2, unordered, @other>

md.particle_set @atoms
md.tuple_set @pairs on(@atoms) arity(2) orientation(unordered) disjoint
md.tuple_set @trios on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @other on(@atoms) arity(2) orientation(unordered) disjoint
md.disjoint_union @groups on(@atoms) of [@pairs, @trios]

// Two updates in a chain, as the constraints of a step have them: the loop
// over the trios reads the positions before the update of the pairs, which
// changes none of its members, both as its positions and as a gathered
// field. The sum over the trios reads what the update of its own set
// changed, and keeps it.
//
// CHECK-LABEL: md.function @chain(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>
// CHECK:         %[[C1:[a-z0-9]+]] = md.gather_tuples %{{[a-z0-9]+}}, %[[X]], %{{[a-z0-9]+}}
// CHECK:         %[[Y1:[a-z0-9]+]] = md.map_particles gather(%[[X]], %[[C1]] :
// CHECK:         %[[C2:[a-z0-9]+]] = md.gather_tuples %{{[a-z0-9]+}}, %[[X]], %{{[a-z0-9]+}}
// CHECK-SAME:      gather(%[[X]] :
// CHECK:         %[[Y2:[a-z0-9]+]] = md.map_particles gather(%[[Y1]], %[[C2]] :
// CHECK:         md.sum_tuples %{{[a-z0-9]+}}, %[[Y2]], %{{[a-z0-9]+}}
// CHECK-SAME:      gather(%[[Y2]] :
md.function @chain(%x: !vec, %cell: !md.cell, %pairs: !pairs, %trios: !trios)
    -> (!vec, f64) {
  %c1 = md.gather_tuples %pairs, %x, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    %n = arith.negf %d : vector<3xf64>
    md.yield %n, %d : vector<3xf64>, vector<3xf64>
  } : !pairs, !vec -> !vec
  %y1 = md.map_particles gather(%x, %c1 : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    md.yield %s : vector<3xf64>
  } : !vec
  %c2 = md.gather_tuples %trios, %y1, %cell coordinates(displacement(0, 1))
          gather(%y1 : !vec) {
  ^bb0(%d: vector<3xf64>, %p0: vector<3xf64>, %p1: vector<3xf64>,
       %p2: vector<3xf64>):
    md.yield %p0, %p1, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !trios, !vec -> !vec
  // The change first, the field second.
  %y2 = md.map_particles gather(%y1, %c2 : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %s = arith.addf %b, %a : vector<3xf64>
    md.yield %s : vector<3xf64>
  } : !vec
  %u = md.sum_tuples %trios, %y2, %cell coordinates(distance(0, 1))
         gather(%y2 : !vec) {
  ^bb0(%r: f64, %p0: vector<3xf64>, %p1: vector<3xf64>, %p2: vector<3xf64>):
    md.yield %r : f64
  } : !trios, !vec -> f64
  md.return %y2, %u : !vec, f64
}

// The loop over the trios reads what the loop over its own set changed, and
// a set outside the union reads what the sets of the union changed: both
// keep the fields they read.
//
// CHECK-LABEL: md.function @kept(
// CHECK:         %[[Y:[a-z0-9]+]] = md.map_particles
// CHECK:         md.gather_tuples %{{[a-z0-9]+}}, %[[Y]], %{{[a-z0-9]+}}
// CHECK:         md.gather_tuples %{{[a-z0-9]+}}, %[[Y]], %{{[a-z0-9]+}}
md.function @kept(%x: !vec, %cell: !md.cell, %trios: !trios, %other: !other)
    -> (!vec, !vec) {
  %c = md.gather_tuples %trios, %x, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    md.yield %d, %d, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !trios, !vec -> !vec
  %y = md.map_particles gather(%x, %c : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    md.yield %s : vector<3xf64>
  } : !vec
  %again = md.gather_tuples %trios, %y, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    md.yield %d, %d, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !trios, !vec -> !vec
  %outside = md.gather_tuples %other, %y, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    md.yield %d, %d : vector<3xf64>, vector<3xf64>
  } : !other, !vec -> !vec
  md.return %again, %outside : !vec, !vec
}

// A map that does more than add the change is not bypassed.
//
// CHECK-LABEL: md.function @scaled(
// CHECK:         %[[Y:[a-z0-9]+]] = md.map_particles
// CHECK:         md.gather_tuples %{{[a-z0-9]+}}, %[[Y]], %{{[a-z0-9]+}}
md.function @scaled(%x: !vec, %cell: !md.cell, %pairs: !pairs, %trios: !trios)
    -> !vec {
  %c = md.gather_tuples %pairs, %x, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    md.yield %d, %d : vector<3xf64>, vector<3xf64>
  } : !pairs, !vec -> !vec
  %y = md.map_particles gather(%x, %c : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    %t = arith.addf %s, %a : vector<3xf64>
    md.yield %t : vector<3xf64>
  } : !vec
  %z = md.gather_tuples %trios, %y, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    md.yield %d, %d, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !trios, !vec -> !vec
  md.return %z : !vec
}

// A loop over the trios that is no link of the chain, as the loop over the
// velocities that reads the positions a chain gave: nothing adds what it
// gathers to the field it reads, so it keeps the field, and the earlier
// field need not be kept alive for it.
//
// CHECK-LABEL: md.function @no_link(
// CHECK:         %[[Y:[a-z0-9]+]] = md.map_particles
// CHECK:         md.gather_tuples %{{[a-z0-9]+}}, %[[Y]], %{{[a-z0-9]+}}
md.function @no_link(%x: !vec, %cell: !md.cell, %pairs: !pairs, %trios: !trios)
    -> !vec {
  %c = md.gather_tuples %pairs, %x, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    md.yield %d, %d : vector<3xf64>, vector<3xf64>
  } : !pairs, !vec -> !vec
  %y = md.map_particles gather(%x, %c : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    md.yield %s : vector<3xf64>
  } : !vec
  %z = md.gather_tuples %trios, %y, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    md.yield %d, %d, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !trios, !vec -> !vec
  md.return %z : !vec
}
