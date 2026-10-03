// RUN: mdir-opt --fixed-order-reductions="chunks=4" %s | FileCheck %s
// RUN: mdir-opt --fixed-order-reductions="chunks=4" %s \
// RUN:   --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_c_runner_utils | FileCheck %s --check-prefix=OUT

// A reduction of a parallel loop is summed over a fixed number of chunks of
// contiguous iterations, each folded in order into a buffer at the entry of
// the function, and the initial value then takes the chunks in their order
// (D171). Neither the threads nor their number decide
// the order of the additions.

// CHECK-LABEL: func.func @sum
// CHECK:         %[[PARTS:.*]] = memref.alloca() : memref<4xf64>
// CHECK:         %[[COUNT:.*]] = arith.minsi
// CHECK:         scf.parallel (%[[CHUNK:.*]]) = (%{{.*}}) to (%[[COUNT]])
// CHECK:           %[[FOLD:.*]]:2 = scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}, %[[FIRST:.*]] = %true)
// CHECK:             %[[ADD:.*]] = arith.addf %[[ACC]]
// CHECK:             arith.select %[[FIRST]], %{{.*}}, %[[ADD]]
// CHECK:           memref.store %[[FOLD]]#0, %[[PARTS]][%[[CHUNK]]]
// CHECK:         %[[TOTAL:.*]] = scf.for {{.*}} iter_args(%[[RUNNING:.*]] = %{{.*}}) -> (f64)
// CHECK:           memref.load %[[PARTS]]
// CHECK:         return %[[TOTAL]]
// CHECK-NOT:     scf.reduce(

func.func @sum(%values: memref<?xf64>, %init: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %values, %c0 : memref<?xf64>
  %sum = scf.parallel (%i) = (%c0) to (%n) step (%c1) init (%init) -> f64 {
    %v = memref.load %values[%i] : memref<?xf64>
    scf.reduce(%v : f64) {
    ^bb0(%lhs: f64, %rhs: f64):
      %s = arith.addf %lhs, %rhs : f64
      scf.reduce.return %s : f64
    }
  }
  return %sum : f64
}

// A loop without results, and an empty loop, keep their meaning: the first
// is left alone, the second gives its initial value.

// CHECK-LABEL: func.func @none
// CHECK:         scf.parallel
// CHECK-NOT:     memref.alloca
func.func @none(%values: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %values, %c0 : memref<?xf64>
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %v = memref.load %values[%i] : memref<?xf64>
    memref.store %v, %values[%i] : memref<?xf64>
    scf.reduce
  }
  return
}

// Ten values 1, 2, ..., 10 in 4 chunks, starting from 0.5: 55.5; then
// three values, fewer than the chunks, and none at all.
// OUT: 55.5
// OUT-NEXT: 6.5
// OUT-NEXT: 0.5
func.func private @printF64(f64)
func.func private @printNewline()
func.func @fill(%count: index) -> memref<?xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %buffer = memref.alloc(%count) : memref<?xf64>
  scf.for %i = %c0 to %count step %c1 {
    %next = arith.addi %i, %c1 : index
    %int = arith.index_cast %next : index to i64
    %value = arith.sitofp %int : i64 to f64
    memref.store %value, %buffer[%i] : memref<?xf64>
  }
  return %buffer : memref<?xf64>
}
func.func @main() {
  %half = arith.constant 0.5 : f64
  %c10 = arith.constant 10 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %ten = func.call @fill(%c10) : (index) -> memref<?xf64>
  %a = func.call @sum(%ten, %half) : (memref<?xf64>, f64) -> f64
  func.call @printF64(%a) : (f64) -> ()
  func.call @printNewline() : () -> ()
  %seven = func.call @fill(%c3) : (index) -> memref<?xf64>
  %b = func.call @sum(%seven, %half) : (memref<?xf64>, f64) -> f64
  func.call @printF64(%b) : (f64) -> ()
  func.call @printNewline() : () -> ()
  %none = func.call @fill(%c0) : (index) -> memref<?xf64>
  %c = func.call @sum(%none, %half) : (memref<?xf64>, f64) -> f64
  func.call @printF64(%c) : (f64) -> ()
  func.call @printNewline() : () -> ()
  return
}
