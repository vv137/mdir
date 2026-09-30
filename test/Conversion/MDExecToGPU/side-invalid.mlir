// RUN: mdir-opt %s --convert-md-exec-to-gpu -split-input-file -verify-diagnostics

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
  // expected-error @+1 {{cannot run on a second stream; only the reciprocal sums of PME can}}
  md_exec.permute %x, %o outs(%y : !pos) {md_exec.side} : !pos, !ord
  md_exec.join
  return
}
