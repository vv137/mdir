// RUN: not mdir-opt %s --convert-md-exec-to-gpu 2>&1 | FileCheck %s
// A frame batch refuses an unconverted host consumer of a result.
// CHECK: device frame result still requires a host readback
func.func @escape_sum(%sums: memref<1xf64, 1>, %host: memref<1xf64>) -> f64
    attributes {mdir.frame_batch} {
  %c0 = arith.constant 0 : index
  gpu.memcpy %host, %sums : memref<1xf64>, memref<1xf64, 1> {mdir.readback = 1 : i64}
  %value = memref.load %host[%c0] {mdir.readback = 1 : i64} : memref<1xf64>
  return %value : f64
}
