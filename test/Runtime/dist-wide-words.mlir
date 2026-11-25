// RUN: mlir-opt %s %lower_loops_to_llvm > %t
// RUN: mlir-runner %t -e recover --entry-point-result=i64 --shared-libs=%mdrt | FileCheck %s
// RUN: not --crash mlir-runner %t -e overflow --entry-point-result=i64 --shared-libs=%mdrt 2>&1 | FileCheck %s --check-prefix=RANGE
// CHECK: 9223372036854775807
// RANGE: deterministic fixed-point result exceeds signed 64-bit range

func.func private @mdrtDistAddWide(i128, i128) -> i128
func.func private @mdrtDistNarrowWord(i128) -> i64
func.func @recover() -> i64 {
  %max = arith.constant 9223372036854775807 : i128
  %one = arith.constant 1 : i128
  %minus = arith.constant -1 : i128
  %partial = func.call @mdrtDistAddWide(%max, %one) : (i128, i128) -> i128
  %sum = func.call @mdrtDistAddWide(%partial, %minus) : (i128, i128) -> i128
  %word = func.call @mdrtDistNarrowWord(%sum) : (i128) -> i64
  return %word : i64
}
func.func @overflow() -> i64 {
  %outside = arith.constant 9223372036854775808 : i128
  %word = func.call @mdrtDistNarrowWord(%outside) : (i128) -> i64
  return %word : i64
}
