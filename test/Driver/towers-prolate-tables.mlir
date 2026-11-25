// RUN: mdir-towers-trial-inputs-test %s --table-types | FileCheck %s
// Storage assignment keeps the host table dimensions when it narrows or
// folds elements. The fixture supplies a square and a rectangular table.
// CHECK: memref<9x9xvector<2xf32>, 1>
// CHECK-NEXT: memref<128x4xf32, 1>
// CHECK-NEXT: memref<32x4xf64, 1>
func.func @tables(%pos: index, %vel: index, %box: index,
                  %lj: memref<?x?xf64>, %poly: memref<?x?xf64>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %rows = memref.dim %lj, %zero : memref<?x?xf64>
  %cols = memref.dim %lj, %one : memref<?x?xf64>
  %a = gpu.alloc(%rows, %cols) : memref<?x?xvector<2xf32>, 1>
  %cells = memref.dim %poly, %zero : memref<?x?xf64>
  %terms = memref.dim %poly, %one : memref<?x?xf64>
  %b = gpu.alloc(%cells, %terms) : memref<?x?xf32, 1>
  %n = arith.constant 32 : index
  %c = gpu.alloc(%n) : memref<?x4xf64, 1>
  return
}
