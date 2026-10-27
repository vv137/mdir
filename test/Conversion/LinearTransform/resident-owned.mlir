// RUN: mdir-opt %s --md-exec-place-linear-transform-device | FileCheck %s
// A cast preserves a proved allocation. Even an unused result completes
// its actual producer queue before the device storage is released.
// CHECK-LABEL: func.func @unused_result
// CHECK: md_exec.linear_transform
// CHECK: call @mdrtCudaWaitScoped
// CHECK-NEXT: memref.memory_space_cast
// CHECK-NEXT: gpu.dealloc
func.func @unused_result() {
  %input = memref.alloc() : memref<24xf64>
  %output = memref.alloc() : memref<32xf64>
  %i = memref.cast %input : memref<24xf64> to memref<?xf64>
  %o = memref.cast %output : memref<32xf64> to memref<?xf64>
  md_exec.linear_transform %i outs(%o : memref<?xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<?xf64>
  memref.dealloc %input : memref<24xf64>
  memref.dealloc %output : memref<32xf64>
  return
}
