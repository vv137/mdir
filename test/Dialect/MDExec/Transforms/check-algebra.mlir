// --md-exec-check-algebra (D[reduction-algebra]): every op that reduces
// values states their algebra; the lowerings refuse one that does not, on
// the CPU and on a device.
//
// RUN: mdir-opt %s --md-exec-check-algebra -verify-diagnostics -o /dev/null
// RUN: not mdir-opt %s --convert-md-exec-to-loops 2>&1 | FileCheck %s
// RUN: not mdir-opt %s --convert-md-exec-to-gpu 2>&1 | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-check-algebra -o /dev/null
//
// CHECK: error: 'md_exec.scatter_add' op states no algebra of the values it reduces; run 'md-exec-assign-algebra'

func.func @add(%v: memref<?x3xf32, 1>, %i: memref<?xi32, 1>,
               %a: memref<?x3xf32, 1>) {
  // expected-error@+1 {{'md_exec.scatter_add' op states no algebra of the values it reduces; run 'md-exec-assign-algebra'}}
  md_exec.scatter_add %v at %i into %a overwrite(false)
      : memref<?x3xf32, 1>, memref<?xi32, 1>, memref<?x3xf32, 1>
  return
}

// Stated: no error.
func.func @stated(%v: memref<?x3xf32, 1>, %i: memref<?xi32, 1>,
                  %a: memref<?x3xf32, 1>) {
  md_exec.scatter_add %v at %i into %a overwrite(false)
      {md_exec.algebra = #md_exec.algebra<any_order>}
      : memref<?x3xf32, 1>, memref<?xi32, 1>, memref<?x3xf32, 1>
  return
}
