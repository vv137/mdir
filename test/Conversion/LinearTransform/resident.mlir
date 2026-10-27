// RUN: mdir-opt %s '--convert-linear-transform-values=memory=device' --one-shot-bufferize --buffer-deallocation-pipeline --convert-linalg-to-loops --md-exec-place-linear-transform-device | FileCheck %s
// RUN: mdir-opt %s '--convert-linear-transform-values=memory=device' --one-shot-bufferize --buffer-deallocation-pipeline --convert-linalg-to-loops --md-exec-place-linear-transform-device | FileCheck %s --check-prefix=COUNT
// Explicit scalar packing proves the layout without sharing native complex bytes.
// The half-spectrum has no host reader: only the private device inverse copy.
// COUNT-LABEL: func.func @composition
// COUNT-COUNT-3: call @mdrtCudaMemcpyScoped
// COUNT-NOT: call @mdrtCudaMemcpyScoped
// CHECK-LABEL: func.func @composition
// CHECK: call @mdrtCudaMemcpyScoped
// CHECK: md_exec.linear_transform {{.*}} direction("forward")
// CHECK-NOT: call @mdrtCudaMemcpyScoped
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: call @mdrtCudaMemcpyScoped
// CHECK: md_exec.linear_transform {{.*}} direction("backward")
// CHECK: call @mdrtCudaMemcpyScoped
// CHECK: gpu.dealloc
// CHECK-NOT: gpu.memcpy
func.func private @consume(f64)
func.func @composition() {
  %input = arith.constant dense<1.0> : tensor<2x2x6xf64>
  %forward = md.linear_transform %input kind("fft3d") direction("forward")
    adjoint("backward") pairing("real_hermitian_half") : tensor<2x2x6xf64> -> tensor<2x2x4xcomplex<f64>>
  %backward = md.linear_transform %forward kind("fft3d") direction("backward")
    adjoint("forward") pairing("real_hermitian_half") : tensor<2x2x4xcomplex<f64>> -> tensor<2x2x6xf64>
  %c0 = arith.constant 0 : index
  %value = tensor.extract %backward[%c0, %c0, %c0] : tensor<2x2x6xf64>
  func.call @consume(%value) : (f64) -> ()
  return
}

