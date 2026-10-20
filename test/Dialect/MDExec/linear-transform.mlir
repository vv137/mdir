// The execution-level linear transform (D306) has a direction
// and an adjoint in the same primitive family. Its buffers expose the data
// dependencies of the transform to later scheduling passes.
// RUN: mdir-opt %s | FileCheck %s

func.func @forward(%real: memref<?xf64>, %half: memref<?xf64>) {
  // CHECK: md_exec.linear_transform %{{.*}} outs(%{{.*}} : memref<?xf64>) grid([4, 3, 5]) kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : memref<?xf64>
  md_exec.linear_transform %real outs(%half : memref<?xf64>)
    grid([4, 3, 5]) kind("fft3d") direction("forward")
    adjoint("backward") pairing("real_hermitian_half") : memref<?xf64>
  return
}

func.func @backward(%half: memref<?xf32>, %real: memref<?xf32>) {
  // CHECK: md_exec.linear_transform %{{.*}} outs(%{{.*}} : memref<?xf32>) grid([4, 3, 6]) kind("fft3d") direction("backward") adjoint("forward") pairing("real_hermitian_half") : memref<?xf32>
  md_exec.linear_transform %half outs(%real : memref<?xf32>)
    grid([4, 3, 6]) kind("fft3d") direction("backward")
    adjoint("forward") pairing("real_hermitian_half") : memref<?xf32>
  return
}
