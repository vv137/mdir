// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s --implicit-check-not=gpu.memcpy --implicit-check-not=mdir.readback
// Only a device batch replaces a reciprocal stream's result readback with
// an event join and direct device loads in the consuming kernel.
// CHECK-LABEL: func.func @device_sums(
// CHECK-SAME: %[[SUM:[a-zA-Z0-9_]+]]: memref<1xf64, 1>
// CHECK: call @mdrtSideBegin
// CHECK: gpu.launch
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: memref.load
// CHECK: call @mdrtSideEnd
// CHECK: call @mdrtSideJoin
// CHECK: gpu.launch
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = memref.load %[[SUM]]
// CHECK: arith.mulf %[[VALUE]]
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: memref.load
// CHECK: return
func.func private @mdrtSideBegin()
func.func private @mdrtSideEnd()
func.func private @mdrtSideJoin()
func.func @device_sums(%sums: memref<1xf64, 1>, %host: memref<1xf64>,
                      %output: memref<1xf64, 1>) attributes {mdir.frame_batch} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  func.call @mdrtSideBegin() : () -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %one, %sums[%c0] : memref<1xf64, 1>
    gpu.terminator
  }
  gpu.memcpy %host, %sums : memref<1xf64>, memref<1xf64, 1> {mdir.readback = 1 : i64}
  %value = memref.load %host[%c0] {mdir.readback = 1 : i64} : memref<1xf64>
  %scaled = arith.mulf %value, %two : f64
  func.call @mdrtSideEnd() : () -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %scaled, %output[%c0] : memref<1xf64, 1>
    gpu.terminator
  }
  return
}
