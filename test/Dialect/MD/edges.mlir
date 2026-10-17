// RUN: mdir-opt %s | mdir-opt | FileCheck %s

// The pairs of images within a cutoff that a learned potential takes,
// derived from a neighborhood (D277).

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

// CHECK-LABEL: md.function @model_pairs(
// CHECK:       %[[N:[0-9]+]] = md.neighborhood %{{[a-z0-9]+}}, %{{[a-z0-9]+}} cutoff(0.69999999999999996)
// CHECK-NEXT:  %{{[0-9]+}} = md.edges %[[N]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}} cutoff(0.69999999999999996)
// CHECK-SAME:    : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> !md.edges<@atoms>
md.function @model_pairs(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(0.7) : !vec -> !pairs
  %e = md.edges %n, %x, %cell cutoff(0.7) : !pairs, !vec -> !md.edges<@atoms>
  md.return
}
