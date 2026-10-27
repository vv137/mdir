// RUN: mdir-opt %s --md-exec-place-linear-transform-device 2>&1 | FileCheck %s
// Unknown facts do not select the scoped adapter; the existing bridge remains.
// CHECK: fallback to host bridge: unproved allocation extent
// CHECK: fallback to host bridge: unknown alias use or release
// CHECK: fallback to host bridge: backward input lacks a private allocation
// CHECK: fallback to host bridge: unknown producer queue at call boundary
// CHECK-NOT: call @mdrtCudaMemcpyScoped
// CHECK: gpu.memcpy
func.func private @escape(memref<?xf64>)
func.func private @queue_boundary()
func.func @dynamic_extent(%n: index, %m: index) {
  %input = memref.alloc(%n) : memref<?xf64>
  %output = memref.alloc(%m) : memref<?xf64>
  md_exec.linear_transform %input outs(%output : memref<?xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<?xf64>
  memref.dealloc %input : memref<?xf64>
  memref.dealloc %output : memref<?xf64>
  return
}
func.func @escaped_alias() {
  %input = memref.alloc() : memref<24xf64>
  %output = memref.alloc() : memref<32xf64>
  %alias = memref.cast %input : memref<24xf64> to memref<?xf64>
  func.call @escape(%alias) : (memref<?xf64>) -> ()
  md_exec.linear_transform %input outs(%output : memref<32xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<24xf64>
  memref.dealloc %input : memref<24xf64>
  memref.dealloc %output : memref<32xf64>
  return
}
func.func @shared_inverse_input() {
  %input = memref.alloc() : memref<24xf64>
  %half = memref.alloc() : memref<32xf64>
  %output = memref.alloc() : memref<24xf64>
  md_exec.linear_transform %input outs(%half : memref<32xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<24xf64>
  md_exec.linear_transform %half outs(%output : memref<24xf64>) grid([2, 2, 6]) kind("fft3d") direction("backward") adjoint("forward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<32xf64>
  memref.dealloc %input : memref<24xf64>
  memref.dealloc %half : memref<32xf64>
  memref.dealloc %output : memref<24xf64>
  return
}
func.func @unknown_queue() {
  %a = memref.alloc() : memref<24xf64>
  %b = memref.alloc() : memref<32xf64>
  %c = memref.alloc() : memref<24xf64>
  %d = memref.alloc() : memref<32xf64>
  md_exec.linear_transform %a outs(%b : memref<32xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<24xf64>
  func.call @queue_boundary() : () -> ()
  md_exec.linear_transform %c outs(%d : memref<32xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<24xf64>
  memref.dealloc %a : memref<24xf64>
  memref.dealloc %b : memref<32xf64>
  memref.dealloc %c : memref<24xf64>
  memref.dealloc %d : memref<32xf64>
  return
}
