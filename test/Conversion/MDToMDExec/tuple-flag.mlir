// `md.sum_tuples` with a result of the type i1 tells whether its kernel
// yields true for some tuple (D278): it becomes a loop
// over the tuples with an i1 in `reduce`, which takes no buffers of scratch
// in storage on a device, and which `md-exec-fuse-loops` does not join with
// a loop over the same tuples.
//
// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.2 width=48" \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.2 width=48" \
// RUN:     --md-exec-fuse-loops \
// RUN: | FileCheck %s --check-prefix=FUSE
// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.2 width=48" \
// RUN:     --md-exec-assign-storage="memory=device" \
// RUN: | FileCheck %s --check-prefix=DEVICE

!vec     = !md.field<@atoms, 3 x f64>
!rel     = !md.relation<@atoms, 2, ordered, @links>
!of_link = !md.field<@links, f64>

md.particle_set @atoms
md.tuple_set @links on(@atoms) arity(2) orientation(ordered) disjoint

// CHECK-LABEL: md.function @stretched(
// CHECK:         %[[NO:[a-z0-9_]+]] = arith.constant false
// CHECK:         md_exec.tuple_for {{.*}} coordinates(displacement(1, 0)) tuple(%{{.*}} : !md.field<@links, f64>) reduce(%[[NO]] : i1) arity(2) disjoint {
// CHECK:           md_exec.yield %{{[0-9]+}} : i1
// CHECK:         } : !mdrt.incidence<@atoms, 2, @links>, !md.field<@atoms, 3 x f64> -> i1
//
// FUSE-COUNT-4: md_exec.tuple_for

md.function @stretched(%x: !vec, %cell: !md.cell, %links: !rel,
                       %d: !of_link) -> (i1, !vec) {
  %any = md.sum_tuples %links, %x, %cell coordinates(displacement(1, 0))
           tuple(%d : !of_link) {
  ^bb0(%r: vector<3xf64>, %d_t: f64):
    %sq = arith.mulf %r, %r : vector<3xf64>
    %r2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %d2 = arith.mulf %d_t, %d_t : f64
    %q = arith.divf %r2, %d2 : f64
    %one = arith.constant 1.0 : f64
    %off = arith.subf %q, %one : f64
    %abs = math.absf %off : f64
    %limit = arith.constant 1.0e-4 : f64
    %bad = arith.cmpf ugt, %abs, %limit : f64
    md.yield %bad : i1
  } : !rel, !vec -> i1
  %f = md.gather_tuples %links, %x, %cell coordinates(displacement(1, 0)) {
  ^bb0(%r: vector<3xf64>):
    %n = arith.negf %r : vector<3xf64>
    md.yield %n, %r : vector<3xf64>, vector<3xf64>
  } : !rel, !vec -> !vec
  md.return %any, %f : i1, !vec
}

// The loop in storage on a device: a sum takes two buffers of scratch, the
// flag none.
//
// DEVICE-LABEL: func.func @stored(
// DEVICE:         md_exec.tuple_for {{.*}} reduce(%{{[a-z0-9_]+}} : i1) arity(2) disjoint {
// DEVICE:         } : memref<?xi32, 1>, memref<?x3xf64, 1> -> i1
// DEVICE:         md_exec.tuple_for {{.*}} reduce(%{{[a-z0-9_]+}} : f64) scratch(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<?xf64, 1>, memref<?xf64, 1>) arity(2) disjoint {
func.func @stored(%x: !vec, %cell: !md.cell, %links: !rel) -> (i1, f64) {
  %any = md.sum_tuples %links, %x, %cell coordinates(displacement(1, 0)) {
  ^bb0(%r: vector<3xf64>):
    %sq = arith.mulf %r, %r : vector<3xf64>
    %r2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %limit = arith.constant 1.0 : f64
    %bad = arith.cmpf ugt, %r2, %limit : f64
    md.yield %bad : i1
  } : !rel, !vec -> i1
  %sum = md.sum_tuples %links, %x, %cell coordinates(displacement(1, 0)) {
  ^bb0(%r: vector<3xf64>):
    %sq = arith.mulf %r, %r : vector<3xf64>
    %r2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    md.yield %r2 : f64
  } : !rel, !vec -> f64
  return %any, %sum : i1, f64
}
