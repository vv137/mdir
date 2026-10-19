// RUN: mdir-opt %s | FileCheck %s
// RUN: mdir-opt %s | mdir-opt | FileCheck %s

// The scatter-accumulate of md_exec (D[scatter-op]): the value form, and
// the storage form with `overwrite` and `scatter`.

!val = !md.field<@ghosts, 3 x f64>
!idx = !md.field<@ghosts, i32>
!dst = !md.field<@atoms, 3 x f64>
md.particle_set @atoms
md.particle_set @ghosts

// CHECK-LABEL: func.func @value(
// CHECK: %[[A:[0-9]+]] = md_exec.scatter_add %arg0 at %arg1 into %arg2 : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32> -> !md.field<@atoms, 3 x f64>
// CHECK: md_exec.scatter_add %arg0 at %arg1 into %[[A]] scatter(hash(128, 8, 100000)) : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32> -> !md.field<@atoms, 3 x f64>
func.func @value(%v: !val, %i: !idx, %a: !dst) -> !dst {
  %a1 = md_exec.scatter_add %v at %i into %a : !val, !idx -> !dst
  %a2 = md_exec.scatter_add %v at %i into %a1 scatter(hash(128, 8, 100000))
      : !val, !idx -> !dst
  return %a2 : !dst
}

// CHECK-LABEL: func.func @storage(
// CHECK: md_exec.scatter_add %arg0 at %arg1 into %arg2 overwrite(false) : memref<?x3xf32, 1>, memref<?xi32, 1>, memref<?x3xf32, 1>
// CHECK: md_exec.scatter_add %arg3 at %arg1 into %arg4 overwrite(true) scatter(atomic) : memref<?xf64, 1>, memref<?xi32, 1>, memref<?xf64, 1>
func.func @storage(%v: memref<?x3xf32, 1>, %i: memref<?xi32, 1>,
                   %a: memref<?x3xf32, 1>, %w: memref<?xf64, 1>,
                   %b: memref<?xf64, 1>) {
  md_exec.scatter_add %v at %i into %a overwrite(false)
      : memref<?x3xf32, 1>, memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.scatter_add %w at %i into %b overwrite(true) scatter(atomic)
      : memref<?xf64, 1>, memref<?xi32, 1>, memref<?xf64, 1>
  return
}
