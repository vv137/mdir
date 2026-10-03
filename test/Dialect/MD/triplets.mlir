// RUN: mdir-opt %s | mdir-opt | FileCheck %s
// RUN: mdir-opt %s --md-differentiate | FileCheck %s --check-prefix=GRAD

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
!trip  = !md.relation<@atoms, 3, reversal>

md.particle_set @atoms

// The triplets of a neighborhood, summed by md.sum_tuples with a gathered
// field and no fields of tuples (D160).
//
// CHECK-LABEL: md.potential @three_body(
// CHECK:       %[[N:[0-9]+]] = md.neighborhood
// CHECK-NEXT:  %[[T:[0-9]+]] = md.triplets %[[N]] cutoff(1.200000e+00)
// CHECK-SAME:    : !md.relation<@atoms, 2, unordered> -> !md.relation<@atoms, 3, reversal>
// CHECK-NEXT:  md.sum_tuples %[[T]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}}
// CHECK-SAME:    coordinates(distance(0, 1), distance(2, 1), cosine(0, 1, 2))
// CHECK-SAME:    gather(%{{[a-z0-9]+}} : !md.field<@atoms, f64>)
md.potential @three_body(%x: !vec, %cell: !md.cell,
                         %q: !md.field<@atoms, f64>) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  %t = md.triplets %n cutoff(1.2) : !pairs -> !trip
  %u = md.sum_tuples %t, %x, %cell
         coordinates(distance(0, 1), distance(2, 1), cosine(0, 1, 2))
         gather(%q : !md.field<@atoms, f64>) {
  ^bb0(%r1: f64, %r2: f64, %c: f64, %q0: f64, %q1: f64, %q2: f64):
    %p = arith.mulf %r1, %r2 : f64
    %e = arith.mulf %p, %c : f64
    %s = arith.mulf %e, %q1 : f64
    md.yield %s : f64
  } : !trip, !vec -> f64
  md.return %u : f64
}

// The forces of a sum over triplets are a gather over the same triplets,
// with the displacements of the two legs from the center.
//
// GRAD:       %[[T:[0-9]+]] = md.triplets
// GRAD:       md.gather_tuples %[[T]]
// GRAD-SAME:    displacement(0, 1), displacement(2, 1)
func.func @forces(%x: !vec, %cell: !md.cell, %q: !md.field<@atoms, f64>)
    -> (f64, !vec) {
  %u, %f = md.evaluate @three_body(%x, %cell, %q) request [energy, forces]
      : (!vec, !md.cell, !md.field<@atoms, f64>) -> (f64, !vec)
  return %u, %f : f64, !vec
}
