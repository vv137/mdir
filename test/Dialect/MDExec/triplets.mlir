// RUN: mdir-opt %s -split-input-file | mdir-opt -split-input-file | FileCheck %s

// The value form: a relation without a tuple set, and its incidence
// structure (D160).
//
// CHECK-LABEL: func.func @value_form(
// CHECK:       %[[T:[0-9]+]] = md_exec.build_triplets %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} cutoff(1.200000e+00)
// CHECK-SAME:    : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 3, reversal>
// CHECK-NEXT:  md_exec.build_incidence %[[T]]
// CHECK-SAME:    : !md.relation<@atoms, 3, reversal> -> !mdrt.incidence<@atoms, 3>
func.func @value_form(%nl: !mdrt.neighbors<@atoms>,
                      %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> !mdrt.incidence<@atoms, 3> {
  %t = md_exec.build_triplets %nl, %x, %cell cutoff(1.2)
         : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
           -> !md.relation<@atoms, 3, reversal>
  %inc = md_exec.build_incidence %t
           : !md.relation<@atoms, 3, reversal> -> !mdrt.incidence<@atoms, 3>
  return %inc : !mdrt.incidence<@atoms, 3>
}

// -----

// The storage form: the members of the triplets on the host.
//
// CHECK-LABEL: func.func @storage_form(
// CHECK:       md_exec.build_triplets %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} cutoff(1.200000e+00)
// CHECK-SAME:    : !mdrt.neighbors<@atoms>, memref<?x3xf32> -> memref<?x3xi32>
func.func @storage_form(%nl: !mdrt.neighbors<@atoms>, %x: memref<?x3xf32>,
                        %cell: !md.cell) -> memref<?x3xi32> {
  %m = md_exec.build_triplets %nl, %x, %cell cutoff(1.2)
         : !mdrt.neighbors<@atoms>, memref<?x3xf32> -> memref<?x3xi32>
  return %m : memref<?x3xi32>
}
