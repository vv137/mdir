// An op of no registered dialect, which --allow-unregistered-dialect
// admits, in the module and in a function of the storage form on a device
// (#395). The lowering took its dialect with `isa<...Dialect>`, which
// asserts on the null dialect of such an op; it now takes it as an op of
// another dialect and leaves it where it is.
//
// RUN: mdir-opt --allow-unregistered-dialect --md-exec-assign-algebra --convert-md-exec-to-gpu %s \
// RUN:   | FileCheck %s

md.particle_set @atoms

// CHECK: "test.global"() : () -> ()
"test.global"() : () -> ()

// CHECK-LABEL: func.func @kick(
// CHECK:         "test.unknown"() : () -> ()
// CHECK:         gpu.launch
func.func @kick(%v: memref<?x3xf64, 1>, %f: memref<?x3xf32, 1>, %dt: f64) {
  "test.unknown"() : () -> ()
  md_exec.particle_for ins(%v, %f : memref<?x3xf64, 1>, memref<?x3xf32, 1>)
      outs(%v : memref<?x3xf64, 1>) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf32>):
    %fe = arith.extf %f_i : vector<3xf32> to vector<3xf64>
    %s = vector.broadcast %dt : f64 to vector<3xf64>
    %dv = arith.mulf %s, %fe : vector<3xf64>
    %vn = arith.addf %v_i, %dv : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  }
  return
}
