// The declared adjoint, pairing, grid, and static buffer sizes are part of
// the linear transform's IR contract (D306).
// RUN: mdir-opt %s -verify-diagnostics

func.func @wrong_adjoint(%x: memref<20xf64>, %y: memref<24xf64>) {
  // expected-error@+1 {{expected declared adjoint 'backward', got 'forward'}}
  md_exec.linear_transform %x outs(%y : memref<24xf64>) grid([2, 2, 5]) kind("fft3d") direction("forward") adjoint("forward") pairing("real_hermitian_half") : memref<20xf64>
  return
}

func.func @wrong_pairing(%x: memref<20xf64>, %y: memref<24xf64>) {
  // expected-error@+1 {{expected pairing 'real_hermitian_half', got 'euclidean'}}
  md_exec.linear_transform %x outs(%y : memref<24xf64>) grid([2, 2, 5]) kind("fft3d") direction("forward") adjoint("backward") pairing("euclidean") : memref<20xf64>
  return
}

func.func @wrong_odd_size(%x: memref<20xf64>, %y: memref<20xf64>) {
  // expected-error@+1 {{expected 24 output values, got 20}}
  md_exec.linear_transform %x outs(%y : memref<20xf64>) grid([2, 2, 5]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : memref<20xf64>
  return
}

func.func @wrong_even_size(%x: memref<20xf32>, %y: memref<24xf32>) {
  // expected-error@+1 {{expected 16 input values, got 20}}
  md_exec.linear_transform %x outs(%y : memref<24xf32>) grid([2, 2, 4]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : memref<20xf32>
  return
}

func.func @product_overflow(%x: memref<?xf64>, %y: memref<?xf64>) {
  // expected-error@+1 {{the grid dimensions or their product overflow}}
  md_exec.linear_transform %x outs(%y : memref<?xf64>) grid([2147483647, 2147483647, 3]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : memref<?xf64>
  return
}

func.func @spectrum_overflow(%x: memref<?xf64>, %y: memref<?xf64>) {
  // expected-error@+1 {{the packed half-spectrum size overflows}}
  md_exec.linear_transform %x outs(%y : memref<?xf64>) grid([2147483647, 2147483647, 2]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : memref<?xf64>
  return
}
