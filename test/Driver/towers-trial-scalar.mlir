// RUN: mdir-towers-trial-inputs-test %s | FileCheck %s
// The nearest f32 to 2.4975943159567202 is 2.497594356536865234375.
// Subtracting half an f32 ulp then rounds back to that same even significand.
// Skipping the first rounding would instead yield the preceding f32.
// CHECK: 0x401fd896
func.func @table(%device: memref<1x1xf32, 1>) {
  %host = memref.alloc() : memref<1x1xf32>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %value = arith.constant 2.4975943159567202 : f64
  %half_ulp = arith.constant 1.1920928955078125e-7 : f64
  scf.for %i = %zero to %one step %one {
    scf.for %j = %zero to %one step %one {
      %rounded = arith.truncf %value : f64 to f32
      %wide = arith.extf %rounded : f32 to f64
      %midpoint = arith.subf %wide, %half_ulp : f64
      %even = arith.truncf %midpoint : f64 to f32
      memref.store %even, %host[%i, %j] : memref<1x1xf32>
    }
  }
  gpu.memcpy %device, %host : memref<1x1xf32, 1>, memref<1x1xf32>
  return
}
