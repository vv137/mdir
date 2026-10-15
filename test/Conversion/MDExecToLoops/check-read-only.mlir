// RUN: mdir-opt %s --check-read-only-arguments | FileCheck %s

// What `check-read-only-arguments` accepts of a buffer that a function
// declares read-only (D[start-arguments]): loads, the shape, the
// source of a copy to the host or to a device, a view of it, a call of a
// function of the module that reads it, and a call of a declared function
// that declares its parameter read-only. The module passes as it is.

func.func private @mdrtTake(memref<?x?xi32> {mdrt.read_only}, i64) -> i64

func.func @reads(%table: memref<?x?xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %v = memref.load %table[%c0, %c0] : memref<?x?xf64>
  return %v : f64
}

// CHECK-LABEL: func.func @entry(
// CHECK-SAME:    %{{[a-z0-9]+}}: memref<?x?xf64> {mdrt.read_only}
func.func @entry(%table: memref<?x?xf64> {mdrt.read_only},
                 %members: memref<?x2xi32> {mdrt.read_only},
                 %field: memref<?xf32> {mdrt.read_only},
                 %written: memref<?xf32>) -> f64 {
  %c0 = arith.constant 0 : index
  %n = memref.dim %field, %c0 : memref<?xf32>
  %x = memref.load %field[%c0] : memref<?xf32>
  memref.store %x, %written[%c0] : memref<?xf32>
  memref.copy %field, %written : memref<?xf32> to memref<?xf32>
  %device = gpu.alloc (%n) : memref<?xf32, 1>
  gpu.memcpy %device, %field : memref<?xf32, 1>, memref<?xf32>
  %wide = memref.cast %members : memref<?x2xi32> to memref<?x?xi32>
  %count = arith.constant 4 : i64
  %handle = func.call @mdrtTake(%wide, %count) : (memref<?x?xi32>, i64) -> i64
  %v = func.call @reads(%table) : (memref<?x?xf64>) -> f64
  return %v : f64
}
