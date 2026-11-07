// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s --implicit-check-not=mdir.frame_parameter
// A shared f64 expression is reconstructed in each consuming kernel,
// including a side-stream kernel, without a host load or readback.
// CHECK-LABEL: func.func @parameters(
// CHECK-SAME: %[[THETA:[a-zA-Z0-9_]+]]: memref<?xf64, 1>
// CHECK-NOT: memref.load
// CHECK: gpu.launch
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = memref.load %[[THETA]]
// CHECK: %[[SCALE:[a-zA-Z0-9_]+]] = arith.mulf %[[VALUE]]
// CHECK: arith.addf %[[SCALE]], %[[SCALE]]
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: memref.load
// CHECK: call @mdrtSideBegin
// CHECK-NOT: memref.load
// CHECK: gpu.launch
// CHECK: %[[SIDEVALUE:[a-zA-Z0-9_]+]] = memref.load %[[THETA]]
// CHECK: %[[SIDESCALE:[a-zA-Z0-9_]+]] = arith.mulf %[[SIDEVALUE]]
// CHECK: arith.addf %[[SIDESCALE]], %[[SIDESCALE]]
// CHECK: arith.negf
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: memref.load
// CHECK: call @mdrtSideEnd
// CHECK: return
func.func private @mdrtSideBegin()
func.func private @mdrtSideEnd()
func.func @parameters(%theta: memref<?xf64, 1>, %output: memref<?xf64, 1>, %index: index)
    attributes {mdir.frame_batch} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %two = arith.constant 2.0 : f64
  %value = memref.load %theta[%index] {mdir.frame_parameter} : memref<?xf64, 1>
  %scale = arith.mulf %value, %two : f64
  %sum = arith.addf %scale, %scale : f64
  %negative = arith.negf %sum : f64
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %sum, %output[%c0] : memref<?xf64, 1>
    gpu.terminator
  }
  func.call @mdrtSideBegin() : () -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %negative, %output[%c1] : memref<?xf64, 1>
    gpu.terminator
  }
  func.call @mdrtSideEnd() : () -> ()
  return
}
