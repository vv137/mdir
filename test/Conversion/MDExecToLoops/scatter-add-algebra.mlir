// md_exec.scatter_add on the host by its stated algebra
// (D[reduction-algebra]): an `any_order` sum adds in a parallel loop with
// atomics (`atomic-grid`, the default), a `fixed_order` one in a loop on one
// thread in the order of the values, whatever the option.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-loops \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true \
// RUN:     --convert-md-exec-to-loops | FileCheck %s --check-prefix=ORDERED
//
// CHECK-LABEL: func.func @add(
// CHECK:         scf.parallel
// CHECK:           memref.atomic_rmw addf
// CHECK:         return
// ORDERED-LABEL: func.func @add(
// ORDERED-NOT:   scf.parallel
// ORDERED-NOT:   memref.atomic_rmw
// ORDERED:       scf.for
// ORDERED:         arith.addf
// ORDERED:       return
func.func @add(%v: memref<?x3xf64>, %i: memref<?xi32>, %a: memref<?x3xf64>) {
  md_exec.scatter_add %v at %i into %a overwrite(false)
      : memref<?x3xf64>, memref<?xi32>, memref<?x3xf64>
  return
}
