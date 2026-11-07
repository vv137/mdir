// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s
// A helper's numerical return is snapshotted on the device for a frame
// consumer. The helper's ordinary ABI remains available unchanged.
module {
  func.func private @numbers(%rows: memref<7xf64, 1>) -> f64 {
    %zero = arith.constant 0 : index
    %host = memref.alloc() : memref<7xf64>
    %start = gpu.wait async
    %copied = gpu.memcpy async [%start] %host, %rows : memref<7xf64>, memref<7xf64, 1>
    gpu.wait [%copied]
    %energy = memref.load %host[%zero] : memref<7xf64>
    memref.dealloc %host : memref<7xf64>
    return %energy : f64
  }
  // CHECK-LABEL: func.func @frame(
  // CHECK: gpu.alloc
  // CHECK-NOT: gpu.memcpy
  // CHECK-NOT: call @numbers
  // CHECK: gpu.launch
  // CHECK: memref.load {{.*}} : memref<7xf64, 1>
  // CHECK: memref.store
  // CHECK: gpu.launch
  // CHECK: memref.load {{.*}} : memref<7xf64, 1>
  // CHECK: memref.store
  // CHECK-NOT: gpu.memcpy
  // CHECK: return
  func.func @frame(%rows: memref<7xf64, 1>, %out: memref<1xf64, 1>) attributes {mdir.frame_batch} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %energy = func.call @numbers(%rows) : (memref<7xf64, 1>) -> f64
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
        threads(%tx, %ty, %tz) in (%dx = %one, %dy = %one, %dz = %one) {
      memref.store %energy, %out[%zero] : memref<1xf64, 1>
      gpu.terminator
    }
    return
  }
}
