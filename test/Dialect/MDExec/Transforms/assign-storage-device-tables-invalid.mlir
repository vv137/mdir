// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" -split-input-file -verify-diagnostics
func.func @wrong_width(%input: memref<?x?xf32, 1>) -> memref<?x?xf64, 1> {
  %table = mdrt.from_buffer %input : memref<?x?xf32, 1> to !md.table<2, f64>
  // expected-error@+1 {{the table storage has type}}
  %view = mdrt.to_buffer %table : !md.table<2, f64> to memref<?x?xf64, 1>
  return %view : memref<?x?xf64, 1>
}
// -----
func.func @no_host_copy(%input: memref<?x?xf64, 1>) -> memref<?x?xf64> {
  %table = mdrt.from_buffer %input : memref<?x?xf64, 1> to !md.table<2, f64>
  // expected-error@+1 {{the table has no buffer in the requested memory space}}
  %view = mdrt.to_buffer %table : !md.table<2, f64> to memref<?x?xf64>
  return %view : memref<?x?xf64>
}
