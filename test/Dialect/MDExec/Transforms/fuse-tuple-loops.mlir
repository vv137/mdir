// RUN: mdir-opt %s --md-exec-fuse-loops | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!inc = !mdrt.incidence<@atoms, @links, 2>
!of_link = !md.field<@links, f64>

md.particle_set @atoms
md.tuple_set @links on(@atoms) arity(2) orientation(ordered)

// Two sums over the tuples of one set and the forces of each: the sums
// become one loop that takes the displacement once and the parameters of
// both, and so do the forces, which depend on the sums through numbers
// outside the loops and stay after them.
//
// CHECK-LABEL: func.func @components(
// CHECK:         %[[S:[0-9]+]]:2 = md_exec.tuple_for %{{.*}} coordinates(displacement(0, 1)) tuple(%{{.*}}, %{{.*}} : !md.field<@links, f64>, !md.field<@links, f64>) reduce(%{{.*}}, %{{.*}} : f64, f64) arity(2) {
// CHECK-NEXT:    ^bb0(%[[D:[a-z0-9]+]]: vector<3xf64>, %[[W1:[a-z0-9]+]]: f64, %[[W2:[a-z0-9]+]]: f64):
// CHECK-NEXT:      %[[X:[0-9]+]] = vector.extract %[[D]][0]
// CHECK-NEXT:      %[[E1:[0-9]+]] = arith.mulf %[[W1]], %[[X]]
// CHECK-NEXT:      %[[Y:[0-9]+]] = vector.extract %[[D]][1]
// CHECK-NEXT:      %[[E2:[0-9]+]] = arith.mulf %[[W2]], %[[Y]]
// CHECK-NEXT:      md_exec.yield %[[E1]], %[[E2]] : f64, f64
// CHECK:         %[[P:[0-9]+]] = arith.mulf %[[S]]#0, %[[S]]#1
// CHECK:         %[[F:[0-9]+]]:2 = md_exec.tuple_for {{.*}} outs(%{{.*}}, %{{.*}} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>) arity(2)
// CHECK:           md_exec.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
// CHECK-NOT:     md_exec.tuple_for
func.func @components(%x: !vec, %cell: !md.cell, %links: !inc,
                      %w1: !of_link, %w2: !of_link) -> (f64, !vec, !vec) {
  %zero = arith.constant 0.0 : f64
  %sx = md_exec.tuple_for %links, %x, %cell coordinates(displacement(0, 1))
      tuple(%w1 : !of_link) reduce(%zero : f64) arity(2) {
  ^bb0(%d: vector<3xf64>, %w: f64):
    %dx = vector.extract %d[0] : f64 from vector<3xf64>
    %e = arith.mulf %w, %dx : f64
    md_exec.yield %e : f64
  } : !inc, !vec -> f64
  %sy = md_exec.tuple_for %links, %x, %cell coordinates(displacement(0, 1))
      tuple(%w2 : !of_link) reduce(%zero : f64) arity(2) {
  ^bb0(%d: vector<3xf64>, %w: f64):
    %dy = vector.extract %d[1] : f64 from vector<3xf64>
    %e = arith.mulf %w, %dy : f64
    md_exec.yield %e : f64
  } : !inc, !vec -> f64
  %p = arith.mulf %sx, %sy : f64
  %z1 = md_exec.zeros : !vec
  %f1 = md_exec.tuple_for %links, %x, %cell coordinates(displacement(0, 1))
      tuple(%w1 : !of_link) outs(%z1 : !vec) arity(2) {
  ^bb0(%d: vector<3xf64>, %w: f64):
    %k = arith.mulf %w, %p : f64
    %b = vector.broadcast %k : f64 to vector<3xf64>
    %n = arith.negf %b : vector<3xf64>
    md_exec.yield %b, %n : vector<3xf64>, vector<3xf64>
  } : !inc, !vec -> !vec
  %z2 = md_exec.zeros : !vec
  %f2 = md_exec.tuple_for %links, %x, %cell coordinates(displacement(0, 1))
      tuple(%w2 : !of_link) outs(%z2 : !vec) arity(2) {
  ^bb0(%d: vector<3xf64>, %w: f64):
    %k = arith.mulf %w, %p : f64
    %b = vector.broadcast %k : f64 to vector<3xf64>
    %n = arith.negf %b : vector<3xf64>
    md_exec.yield %n, %b : vector<3xf64>, vector<3xf64>
  } : !inc, !vec -> !vec
  return %p, %f1, %f2 : f64, !vec, !vec
}
