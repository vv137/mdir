// RUN: mdir-opt %s %md_passes --convert-md-to-md-exec="skin=0.2 width=48" \
// RUN: | FileCheck %s
// RUN: mdir-opt %s %md_passes --convert-md-to-md-exec="skin=0.2 width=48" \
// RUN:     --md-exec-choose-neighbors=kind=groups \
// RUN: | FileCheck %s --check-prefix=CHOOSE
// RUN: mdir-opt %s %md_passes --convert-md-to-md-exec="skin=0.2 width=48" \
// RUN:     --md-exec-assign-storage \
// RUN: | FileCheck %s --check-prefix=STORAGE
// RUN: not mdir-opt %s %md_passes --convert-md-to-md-exec="skin=0.2 width=48" \
// RUN:     --md-exec-assign-storage="memory=device" 2>&1 \
// RUN: | FileCheck %s --check-prefix=DEVICE

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
!trip  = !md.relation<@atoms, 3, reversal>

md.particle_set @atoms

md.potential @three_body(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  %t = md.triplets %n cutoff(1.2) : !pairs -> !trip
  %u = md.sum_tuples %t, %x, %cell
         coordinates(distance(0, 1), distance(2, 1), cosine(0, 1, 2)) {
  ^bb0(%r1: f64, %r2: f64, %c: f64):
    %p = arith.mulf %r1, %r2 : f64
    %e = arith.mulf %p, %c : f64
    md.yield %e : f64
  } : !trip, !vec -> f64
  // A pair term over the same neighborhood, whose loop alone could take
  // each pair once over groups.
  %v = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !pairs, !vec -> f64
  %s = arith.addf %u, %v : f64
  md.return %s : f64
}

// The triplets are found in the neighbor structure of the neighborhood at
// its positions, and the loops over them take the incidence structure of
// a relation without a tuple set (D160).
//
// CHECK-LABEL: func.func @forces(
// CHECK:       %[[NL:[0-9]+]] = md_exec.build_neighbors
// CHECK:       %[[T:[0-9]+]] = md_exec.build_triplets %[[NL]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}} cutoff(1.200000e+00)
// CHECK-SAME:    : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 3, reversal>
// CHECK-NEXT:  %[[INC:[0-9]+]] = md_exec.build_incidence %[[T]]
// CHECK-SAME:    -> !mdrt.incidence<@atoms, 3>
// CHECK:       md_exec.tuple_for %[[INC]]
// CHECK-SAME:    arity(3)
// CHECK-NOT:   md.triplets
//
// The triplets need the rows of a matrix: the structure stays one, though
// the pair loop alone would take groups.
//
// CHOOSE-LABEL: func.func @forces(
// CHOOSE:      md_exec.build_neighbors
// CHOOSE-SAME:   kind(matrix)
//
// The members are a buffer of the host with a row for each triplet.
//
// STORAGE-LABEL: func.func @forces(
// STORAGE:       %[[NL:[0-9]+]] = md_exec.refresh_neighbors
// STORAGE:       %[[M:[0-9]+]] = md_exec.build_triplets %[[NL]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}} cutoff(1.200000e+00)
// STORAGE-SAME:    : !mdrt.neighbors<@atoms>, memref<?x3xf64> -> memref<?x3xi32>
// STORAGE-NEXT:  md_exec.build_incidence %[[M]] size(%{{[a-z0-9]+}}) : memref<?x3xi32> -> memref<?x?xi32>
//
// DEVICE: error: 'md_exec.build_triplets' op finds triplets on the host only: terms over triplets do not run on a device yet (D160)
func.func @forces(%x: !vec, %cell: !md.cell) -> (f64, !vec) {
  %u, %f = md.evaluate @three_body(%x, %cell) request [energy, forces]
      : (!vec, !md.cell) -> (f64, !vec)
  return %u, %f : f64, !vec
}
