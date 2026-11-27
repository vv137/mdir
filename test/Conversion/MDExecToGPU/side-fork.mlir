// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu -split-input-file -verify-diagnostics | FileCheck %s

// D[concurrent-terms]: a fork lets a loop over tuples on the third stream
// start from the first stream before a loop that is issued before it. The
// lowering records the fork's event where the fork stands, and the loop's
// work after the other loop's.
//
// CHECK-LABEL: func.func @forked(
// CHECK:         call @mdrtSide2Fork()
// CHECK:         gpu.launch
// CHECK:         call @mdrtSide2Begin()
// CHECK:         gpu.launch
// CHECK:         call @mdrtSide2End()
// CHECK:         call @mdrtSideJoin()
func.func @forked(%members: memref<?x2xi32>, %n: index,
                  %x: memref<?x3xf32, 1>, %cell: !md.cell) {
  %f = gpu.alloc (%n) : memref<?x3xf32, 1>
  %g = gpu.alloc (%n) : memref<?x3xf32, 1>
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?xi32, 1>
  md_exec.fork 2
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%g : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %n0, %d : vector<3xf32>, vector<3xf32>
  } {md_exec.side, md_exec.lane = 2 : i64} : memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.join
  return
}

// -----

// The loop between the fork and the marked loop writes what the marked
// loop writes: from the fork the two would run at once.

func.func @forked_dependent(%members: memref<?x2xi32>, %n: index,
                            %x: memref<?x3xf32, 1>, %cell: !md.cell) {
  %g = gpu.alloc (%n) : memref<?x3xf32, 1>
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?xi32, 1>
  md_exec.fork 2
  // expected-note @+1 {{the op}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%g : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, memref<?x3xf32, 1>
  // expected-error @+1 {{runs on a side stream from a fork before an op that is not independent of it}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%g : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %n0, %d : vector<3xf32>, vector<3xf32>
  } {md_exec.side, md_exec.lane = 2 : i64} : memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.join
  return
}
