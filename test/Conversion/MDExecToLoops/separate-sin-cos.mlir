// RUN: mdir-opt --separate-sin-cos %s | FileCheck %s
// RUN: mdir-opt --separate-sin-cos %s --convert-vector-to-llvm \
// RUN:   --convert-math-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir | llc -O2 | FileCheck %s --check-prefix=ASM
// RUN: mdir-opt %s --convert-vector-to-llvm --convert-math-to-llvm \
// RUN:   --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir | llc -O2 | FileCheck %s --check-prefix=MERGED

// A sine and a cosine become calls of their own to the C library, so that
// the code generator does not compute the two of one argument by one call
// of `sincos`, which does not round as `sin` and `cos` do (#243): the
// rounding of the sine of a force does not depend on whether the kernel
// also computes the cosine of the energy.

// CHECK-DAG: func.func private @sin(f64) -> f64
// CHECK-DAG: func.func private @cos(f64) -> f64
// CHECK-DAG: func.func private @sinf(f32) -> f32
// CHECK-DAG: func.func private @cosf(f32) -> f32

// CHECK-LABEL: func.func @both
// CHECK-SAME: (%[[X:.*]]: f64)
// CHECK: %[[S:.*]] = call @sin(%[[X]]) : (f64) -> f64
// CHECK: %[[C:.*]] = call @cos(%[[X]]) : (f64) -> f64
// CHECK: return %[[S]], %[[C]]
// ASM-LABEL: both:
// ASM-NOT: sincos
// ASM-LABEL: single:
// Without the pass, the two are one call (with the C library of GNU).
// MERGED-LABEL: both:
// MERGED: sincos
func.func @both(%x: f64) -> (f64, f64) {
  %s = math.sin %x : f64
  %c = math.cos %x : f64
  return %s, %c : f64, f64
}

// CHECK-LABEL: func.func @single
// CHECK: call @sinf(%{{.*}}) : (f32) -> f32
// CHECK: call @cosf(%{{.*}}) : (f32) -> f32
// CHECK-NOT: math.
// ASM-NOT: sincosf
// MERGED-LABEL: single:
// MERGED: sincosf
func.func @single(%x: f32) -> (f32, f32) {
  %s = math.sin %x : f32
  %c = math.cos %x : f32
  return %s, %c : f32, f32
}

// A vector of a fixed length, element by element.
// CHECK-LABEL: func.func @vectors
// CHECK-SAME: (%[[V:.*]]: vector<3xf64>)
// CHECK: %[[E0:.*]] = vector.extract %[[V]][0]
// CHECK: %[[S0:.*]] = call @sin(%[[E0]])
// CHECK: %[[E1:.*]] = vector.extract %[[V]][1]
// CHECK: %[[S1:.*]] = call @sin(%[[E1]])
// CHECK: %[[E2:.*]] = vector.extract %[[V]][2]
// CHECK: %[[S2:.*]] = call @sin(%[[E2]])
// CHECK: %[[R:.*]] = vector.from_elements %[[S0]], %[[S1]], %[[S2]] : vector<3xf64>
// CHECK: return %[[R]]
func.func @vectors(%v: vector<3xf64>) -> vector<3xf64> {
  %s = math.sin %v : vector<3xf64>
  return %s : vector<3xf64>
}

// Other functions stay.
// CHECK-LABEL: func.func @others
// CHECK: math.tan
// CHECK: math.exp
func.func @others(%x: f64) -> (f64, f64) {
  %t = math.tan %x : f64
  %e = math.exp %x : f64
  return %t, %e : f64, f64
}
