// RUN: mdir-opt %s --split-input-file --md-exec-place-linear-transform-device --verify-diagnostics
// A marker cannot prove ownership, extents, aliases, or release.
func.func @unknown(%input: memref<?xf64>, %output: memref<?xf64>) {
  // expected-remark@+2 {{fallback to host bridge: unknown allocation provenance}}
  // expected-error@+1 {{expected distinct owned host allocations}}
  md_exec.linear_transform %input outs(%output : memref<?xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<?xf64>
  return
}
// -----
func.func @released() {
  %input = memref.alloc() : memref<24xf64>
  %output = memref.alloc() : memref<32xf64>
  // expected-error@+1 {{resident storage is used after release}}
  memref.dealloc %input : memref<24xf64>
  md_exec.linear_transform %input outs(%output : memref<32xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<24xf64>
  memref.dealloc %output : memref<32xf64>
  return
}
// -----
func.func @extent() {
  %wrong = arith.constant 23 : index
  %right = arith.constant 32 : index
  %input = memref.alloc(%wrong) : memref<?xf64>
  %output = memref.alloc(%right) : memref<?xf64>
  // expected-error@+1 {{resident allocation extent contradicts transform grid}}
  md_exec.linear_transform %input outs(%output : memref<?xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<?xf64>
  memref.dealloc %input : memref<?xf64>
  memref.dealloc %output : memref<?xf64>
  return
}
// -----
func.func @layout() {
  %n = arith.constant 24 : index
  %m = arith.constant 32 : index
  %input = memref.alloc(%n) : memref<?xf64, strided<[2], offset: 0>>
  %output = memref.alloc(%m) : memref<?xf64>
  // expected-error@+1 {{resident scalar layout must be contiguous with zero offset}}
  md_exec.linear_transform %input outs(%output : memref<?xf64>) grid([2, 2, 6]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") {fft.value_bridge, fft.resident} : memref<?xf64, strided<[2], offset: 0>>
  memref.dealloc %input : memref<?xf64, strided<[2], offset: 0>>
  memref.dealloc %output : memref<?xf64>
  return
}
