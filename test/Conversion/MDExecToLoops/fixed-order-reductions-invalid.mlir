// RUN: mdir-opt --fixed-order-reductions %s -split-input-file -verify-diagnostics

// A reduction over two dimensions is refused rather than left in an order
// that the threads decide (D[threaded-determinism]).

func.func @two(%values: memref<?x?xf64>, %init: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %values, %c0 : memref<?x?xf64>
  // expected-error @below {{has a reduction over more than one dimension}}
  %sum = scf.parallel (%i, %j) = (%c0, %c0) to (%n, %n) step (%c1, %c1) init (%init) -> f64 {
    %v = memref.load %values[%i, %j] : memref<?x?xf64>
    scf.reduce(%v : f64) {
    ^bb0(%lhs: f64, %rhs: f64):
      %s = arith.addf %lhs, %rhs : f64
      scf.reduce.return %s : f64
    }
  }
  return %sum : f64
}
