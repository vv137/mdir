// The checked words of the deterministic distributed sums (D[distribute-driver]).
// RUN: mlir-opt %s %lower_loops_to_llvm > %t
// RUN: mlir-runner %t -e recover --entry-point-result=i64 --shared-libs=%mdrt | FileCheck %s
// RUN: not --crash mlir-runner %t -e overflow --entry-point-result=i64 --shared-libs=%mdrt 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not --crash mlir-runner %t -e wide --entry-point-result=i64 --shared-libs=%mdrt 2>&1 | FileCheck %s --check-prefix=WIDE
// RUN: not --crash mlir-runner %t -e term --entry-point-result=i64 --shared-libs=%mdrt 2>&1 | FileCheck %s --check-prefix=TERM
// RUN: mlir-runner %t -e count --entry-point-result=i64 --shared-libs=%mdrt | FileCheck %s --check-prefix=COUNT
// RUN: not --crash mlir-runner %t -e many --entry-point-result=i64 --shared-libs=%mdrt 2>&1 | FileCheck %s --check-prefix=MANY
// A cancellation beyond 2^63 within a sum of 128 bits.
// CHECK: 9223372036854775807
// The completed quantity, narrowed with the name of its quantity (5: the
// sum of F^2/m).
// RANGE: deterministic fixed-point sum of F^2/m (the kinetic correction) exceeds signed 64-bit range
// A sum of two words beyond 2^127, which the bound on the count excludes.
// WIDE: deterministic fixed-point sum exceeds signed 128-bit range
// One term of 2^63 in Q32 (0: a pair energy).
// TERM: deterministic fixed-point pair energy or virial term outside signed 64-bit range
// COUNT: 4294967296
// MANY: deterministic pair count exceeds the signed 128-bit accumulation bound

func.func private @mdrtDistAddWide(i128, i128) -> i128
func.func private @mdrtDistNarrowWord(i128, i64) -> i64
func.func private @mdrtDistWord(f64, f64, i64) -> i64
func.func private @mdrtDistCheckWordCount(i64)
func.func @recover() -> i64 {
  %max = arith.constant 9223372036854775807 : i128
  %one = arith.constant 1 : i128
  %minus = arith.constant -1 : i128
  %code = arith.constant 0 : i64
  %partial = func.call @mdrtDistAddWide(%max, %one) : (i128, i128) -> i128
  %sum = func.call @mdrtDistAddWide(%partial, %minus) : (i128, i128) -> i128
  %word = func.call @mdrtDistNarrowWord(%sum, %code) : (i128, i64) -> i64
  return %word : i64
}
func.func @overflow() -> i64 {
  %outside = arith.constant 9223372036854775808 : i128
  %code = arith.constant 5 : i64
  %word = func.call @mdrtDistNarrowWord(%outside, %code) : (i128, i64) -> i64
  return %word : i64
}
func.func @wide() -> i64 {
  // 2^126 + 2^126 = 2^127.
  %half = arith.constant 85070591730234615865843651857942052864 : i128
  %sum = func.call @mdrtDistAddWide(%half, %half) : (i128, i128) -> i128
  %low = arith.trunci %sum : i128 to i64
  return %low : i64
}
func.func @term() -> i64 {
  // 2^31 kJ/mol at the scale 2^32.
  %x = arith.constant 2147483648.0 : f64
  %scale = arith.constant 4294967296.0 : f64
  %code = arith.constant 0 : i64
  %word = func.call @mdrtDistWord(%x, %scale, %code) : (f64, f64, i64) -> i64
  return %word : i64
}
func.func @count() -> i64 {
  %limit = arith.constant 4294967296 : i64
  func.call @mdrtDistCheckWordCount(%limit) : (i64) -> ()
  return %limit : i64
}
func.func @many() -> i64 {
  %over = arith.constant 4294967297 : i64
  func.call @mdrtDistCheckWordCount(%over) : (i64) -> ()
  return %over : i64
}
