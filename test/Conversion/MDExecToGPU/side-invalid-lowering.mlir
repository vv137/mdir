// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu -split-input-file -verify-diagnostics

// D[concurrent-terms]: the refusals of the lowering of a loop over tuples
// marked for a side stream. A loop names a stream that does not exist; a
// loop whose kernel calls a function that allocates device memory, which
// on a side stream would let the pool hand a block to the next allocation
// without ordering it (D87, premise P4). The third refusal, a marked loop
// whose lowering launches nothing between the begin and the end of its
// stream, has no valid input: every lowering of a loop over tuples launches
// a kernel there. It guards a future lowering that would issue its work
// elsewhere.

func.func @lane_three(%members: memref<?x2xi32>, %n: index,
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
  } {md_exec.side, md_exec.lane = 3 : i64} : memref<?xi32, 1>, memref<?x3xf32, 1>
  // The two loops are one run, lowered at its last loop.
  // expected-error @+1 {{names side stream 3; there are streams 1 and 2}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%g : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %n0, %d : vector<3xf32>, vector<3xf32>
  } {md_exec.side, md_exec.lane = 3 : i64} : memref<?xi32, 1>, memref<?x3xf32, 1>
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

// -----

func.func @helper(%n: index) {
  %b = gpu.alloc (%n) : memref<?xf32, 1>
  gpu.dealloc %b : memref<?xf32, 1>
  return
}
func.func @allocates(%members: memref<?x2xi32>, %n: index,
                     %x: memref<?x3xf32, 1>, %cell: !md.cell) {
  %g = gpu.alloc (%n) : memref<?x3xf32, 1>
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?xi32, 1>
  // expected-error @+1 {{allocates or frees memory on a second stream}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%g : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    func.call @helper(%n) : (index) -> ()
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } {md_exec.side, md_exec.lane = 2 : i64} : memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.join
  return
}
