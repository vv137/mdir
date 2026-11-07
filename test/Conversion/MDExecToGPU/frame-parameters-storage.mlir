// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu | FileCheck %s --implicit-check-not=mdir.frame_parameter --implicit-check-not=gpu.memcpy
// Storage assignment must retain the batch marker for GPU parameter
// lowering when the function also imports physical fields.
// CHECK-LABEL: func.func @parameters(
// CHECK-SAME: attributes {mdir.frame_batch}
// CHECK-NOT: memref.load
// CHECK: gpu.launch
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.mulf
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: memref.load
// CHECK: return
md.particle_set @atoms
func.func @parameters(%positions: memref<?x3xf64, 1>, %theta: memref<?xf64, 1>,
                      %output: memref<?xf64, 1>) attributes {mdir.frame_batch} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %field = mdrt.from_buffer %positions : memref<?x3xf64, 1> to !md.field<@atoms, 3 x f64>
  %view = mdrt.to_buffer %field : !md.field<@atoms, 3 x f64> to memref<?x3xf64, 1>
  %parameter = memref.load %theta[%c0] {mdir.frame_parameter} : memref<?xf64, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %x = memref.load %view[%c0, %c0] : memref<?x3xf64, 1>
    %value = arith.mulf %x, %parameter : f64
    memref.store %value, %output[%c0] : memref<?xf64, 1>
    gpu.terminator
  }
  return
}
