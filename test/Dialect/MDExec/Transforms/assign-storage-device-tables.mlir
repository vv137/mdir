// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" | FileCheck %s --implicit-check-not=gpu.memcpy --implicit-check-not=gpu.alloc
// RUN: not mdir-opt %s --md-exec-assign-storage="memory=host" 2>&1 | FileCheck %s --check-prefix=HOST
// HOST: a borrowed device table requires memory=device
// Device tables retain their explicit storage width without a copy.
// CHECK-LABEL: func.func @narrow(
// CHECK-SAME: %[[NARROW:[a-zA-Z0-9_]+]]: memref<?x?xf32, 1>)
// CHECK-NEXT: return %[[NARROW]] : memref<?x?xf32, 1>
func.func @narrow(%input: memref<?x?xf32, 1>) -> memref<?x?xf32, 1> {
  %table = mdrt.from_buffer %input : memref<?x?xf32, 1> to !md.table<2, f64, symmetric>
  %view = mdrt.to_buffer %table : !md.table<2, f64, symmetric> to memref<?x?xf32, 1>
  return %view : memref<?x?xf32, 1>
}
// CHECK-LABEL: func.func @wide(
// CHECK-SAME: %[[WIDE:[a-zA-Z0-9_]+]]: memref<?x?xf64, 1>)
// CHECK-NEXT: return %[[WIDE]] : memref<?x?xf64, 1>
func.func @wide(%input: memref<?x?xf64, 1>) -> memref<?x?xf64, 1> {
  %table = mdrt.from_buffer %input : memref<?x?xf64, 1> to !md.table<2, f64>
  %view = mdrt.to_buffer %table : !md.table<2, f64> to memref<?x?xf64, 1>
  return %view : memref<?x?xf64, 1>
}
