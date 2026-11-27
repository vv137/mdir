// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu -split-input-file -verify-diagnostics

// The lowering checks again that the ops before a join are independent of
// the op that runs on a second stream (D87), so that it holds of any
// program that it lowers.

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!ord = memref<?xi32, 1>
!mod = memref<?x?xf64, 1>

func.func @writes_positions(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %x0 = gpu.alloc (%n) : !pos
  %q = gpu.alloc (%n) : !chg
  %o = gpu.alloc (%n) : !ord
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  // expected-error @+1 {{runs on a second stream beside an op that is not independent of it}}
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      {md_exec.side} : !pos, !chg, !mod -> f64, vector<9xf64>
  // expected-note @+1 {{the op}}
  md_exec.permute %x0, %o outs(%x : !pos) : !pos, !ord
  md_exec.join
  return
}

// -----

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!mod = memref<?x?xf64, 1>

func.func @no_join(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %q = gpu.alloc (%n) : !chg
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  // expected-error @+1 {{runs on a second stream, but no join follows it in its block}}
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      {md_exec.side} : !pos, !chg, !mod -> f64, vector<9xf64>
  return
}

// -----

!pos = memref<?x3xf64, 1>
!ord = memref<?xi32, 1>

func.func @not_reciprocal(%n: index) {
  %x = gpu.alloc (%n) : !pos
  %y = gpu.alloc (%n) : !pos
  %o = gpu.alloc (%n) : !ord
  // expected-error @+1 {{cannot run on a second stream; only the reciprocal sums of PME and loops over tuples can}}
  md_exec.permute %x, %o outs(%y : !pos) {md_exec.side} : !pos, !ord
  md_exec.join
  return
}

// -----

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!ord = memref<?xi32, 1>
!mod = memref<?x?xf64, 1>

func.func @writes_forces(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %other = gpu.alloc (%n) : !frc
  %q = gpu.alloc (%n) : !chg
  %o = gpu.alloc (%n) : !ord
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  // expected-error @+1 {{runs on a second stream beside an op that is not independent of it}}
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      {md_exec.side} : !pos, !chg, !mod -> f64, vector<9xf64>
  // expected-note @+1 {{the op}}
  md_exec.permute %other, %o outs(%f : !frc) : !frc, !ord
  md_exec.join
  return
}

// -----

// D[concurrent-terms]: a loop over tuples on the second stream and a loop
// on the first that adds to the same forces before the join. The two
// would add to one buffer at once, in an order that the streams decide.

func.func @tuples_share_forces(%members: memref<?x2xi32>, %n: index,
                               %x: memref<?x3xf32, 1>, %cell: !md.cell,
                               %f: memref<?x3xf32, 1>) {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?xi32, 1>
  // expected-error @+1 {{runs on a second stream beside an op that is not independent of it}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } {md_exec.side} : memref<?xi32, 1>, memref<?x3xf32, 1>
  // expected-note @+1 {{the op}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %n0, %d : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, memref<?x3xf32, 1>
  md_exec.join
  return
}

// -----

// A loop over tuples that reduces values does not run on the second stream:
// the host may read its sums.

func.func @tuples_reduce(%members: memref<?x2xi32>, %n: index,
                         %x: memref<?x3xf32, 1>, %cell: !md.cell,
                         %f: memref<?x3xf32, 1>, %a: memref<?xf64, 1>,
                         %b: memref<?xf64, 1>) -> f64 {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?xi32, 1>
  %u0 = arith.constant 0.0 : f64
  // expected-error @+1 {{cannot run on a second stream: it reduces values, which the host may read}}
  %u = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      scratch(%a, %b : memref<?xf64, 1>, memref<?xf64, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    %s = vector.reduction <add>, %d : vector<3xf32> into f32
    %e = arith.extf %s : f32 to f64
    md_exec.yield %d, %n0, %e : vector<3xf32>, vector<3xf32>, f64
  } {md_exec.side} : memref<?xi32, 1>, memref<?x3xf32, 1> -> f64
  md_exec.join
  return %u : f64
}
