// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu | FileCheck %s

// D[concurrent-terms]: two loops over tuples marked for the second stream,
// with a destination of their own, are issued there (one kernel, as one
// run of loops), beside a loop over tuples on the first stream that writes
// another buffer; the first stream waits at the join, before the loop over
// particles that reads both.
//
// CHECK-LABEL: func.func @beside(
// CHECK:         call @mdrtSideBegin()
// CHECK:         gpu.launch
// CHECK-NOT:     gpu.launch
// CHECK:         call @mdrtSideEnd()
// CHECK-NOT:     call @mdrtSide
// CHECK:         gpu.launch
// CHECK-NOT:     call @mdrtSide
// CHECK:         call @mdrtSideJoin()
// CHECK:         gpu.launch
func.func @beside(%members: memref<?x2xi32>, %n: index,
                  %x: memref<?x3xf32, 1>, %cell: !md.cell,
                  %v: memref<?x3xf32, 1>) {
  // Distinct allocations: the two destinations are disjoint memory.
  %f = gpu.alloc (%n) : memref<?x3xf32, 1>
  %g = gpu.alloc (%n) : memref<?x3xf32, 1>
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?xi32, 1>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%g : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } {md_exec.side} : memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%g : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %n0, %d : vector<3xf32>, vector<3xf32>
  } {md_exec.side} : memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.join
  md_exec.particle_for ins(%f, %g : memref<?x3xf32, 1>, memref<?x3xf32, 1>)
      outs(%v : memref<?x3xf32, 1>) {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>):
    %s = arith.addf %a, %b : vector<3xf32>
    md_exec.yield %s : vector<3xf32>
  }
  return
}
