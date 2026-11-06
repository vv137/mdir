// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" | FileCheck %s --implicit-check-not=gpu.memcpy
// RUN: not mdir-opt %s --md-exec-assign-storage="memory=host" 2>&1 | FileCheck %s --check-prefix=HOST
// HOST: a device field result requires memory=device
//
// Direct device reads keep iteration storage alive until the loop yield.
// Allocation stays outside the loop; neither input nor output is staged.
// CHECK-LABEL: func.func @frames(
// CHECK: %[[STORAGE:[a-zA-Z0-9_]+]] = gpu.alloc
// CHECK: scf.for {{.*}} iter_args(%[[BUFFER:[a-zA-Z0-9_]+]] = %[[STORAGE]])
// CHECK-NOT: gpu.alloc
// CHECK: md_exec.particle_for
// CHECK-SAME: outs(%[[BUFFER]]
// CHECK: %[[VIEW:[a-zA-Z0-9_]+]] = memref.cast %[[BUFFER]]
// CHECK: gpu.launch
// CHECK: memref.load %[[VIEW]]
// CHECK: memref.store
// CHECK: scf.yield %[[BUFFER]]
// CHECK: return

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms
func.func @frames(%input: !vec, %frames: index, %output: memref<?x3xf64, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %frame = %c0 to %frames step %c1 {
    %empty = md_exec.empty : !vec
    %field = md_exec.particle_for ins(%input : !vec) outs(%empty : !vec) {
    ^bb0(%x: vector<3xf64>):
      %one = arith.constant dense<1.0> : vector<3xf64>
      %y = arith.addf %x, %one : vector<3xf64>
      md_exec.yield %y : vector<3xf64>
    } -> !vec
    %view = mdrt.to_buffer %field : !vec to memref<?x3xf64, 1>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
        threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %value = memref.load %view[%c0, %c0] : memref<?x3xf64, 1>
      memref.store %value, %output[%c0, %c0] : memref<?x3xf64, 1>
      gpu.terminator
    }
  }
  return
}

// A root view exposes the existing field storage directly.
// CHECK-LABEL: func.func @result(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: memref<?x3xf64{{.*}}, 1>)
// CHECK-NOT: gpu.alloc
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = memref.cast %[[INPUT]]
// CHECK-NEXT: return %[[RESULT]] : memref<?x3xf64, 1>
func.func @result(%field: !vec) -> memref<?x3xf64, 1> {
  %buffer = mdrt.to_buffer %field : !vec to memref<?x3xf64, 1>
  return %buffer : memref<?x3xf64, 1>
}
