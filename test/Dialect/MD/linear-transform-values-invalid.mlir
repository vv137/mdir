// Device placement accepts only distinct allocations made by the semantic
// bridge. A caller's buffer has no ownership proof. D[fft-value-buffer].
// RUN: not mdir-opt %s --md-exec-place-linear-transform-device -o /dev/null 2>&1 | FileCheck %s
// CHECK: expected distinct owned host allocations from the value bridge
func.func @unknown_ownership(%input: memref<?xf64>, %output: memref<?xf64>) {
  md_exec.linear_transform %input outs(%output : memref<?xf64>)
    grid([3, 2, 5]) kind("fft3d") direction("forward")
    adjoint("backward") pairing("real_hermitian_half")
    {fft.value_bridge} : memref<?xf64>
  return
}
