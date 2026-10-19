// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu -split-input-file -verify-diagnostics
// RUN: not mdir-opt %s --md-exec-assign-algebra=deterministic=true --convert-md-exec-to-gpu=deterministic=true -split-input-file 2>&1 \
// RUN: | FileCheck %s --check-prefix=DET

// The lowering checks the preconditions of the strategy of a scatter
// (D285) with the functions that md-exec-choose-scatter
// uses (Scatter.h), so that they hold of any program that it lowers.

!pos = memref<?x3xf32, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!mod = memref<?x?xf32, 1>

func.func @short(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %q = gpu.alloc (%n) : !chg
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf32, 1>
  %bins = gpu.alloc (%k) : memref<?xi32, 1>
  // DET: error: 'md_exec.reciprocal' op takes the strategy 'tile', whose precondition does not hold: the sum's algebra is 'fixed_order': the deterministic mode adds floating-point values in a fixed order
  // expected-error @+1 {{takes the strategy 'tile', whose precondition does not hold: an edge of the grid of 8 points is shorter than a tile of 11}}
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3, %bins : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf32, 1>, memref<?xi32, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      scatter(tile(8)) : !pos, !chg, !mod -> f64, vector<9xf64>
  return
}

// -----

// Two forces in f64 at 4,096 slots take 4 x 4,096 + 2 x 98,304 bytes.
func.func @wide(%members: memref<?x2xi32>, %n: index,
                %x: memref<?x3xf32, 1>, %cell: !md.cell,
                %f: memref<?x3xf64, 1>, %g: memref<?x3xf64, 1>) {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  // expected-error @+1 {{takes the strategy 'hash', whose precondition does not hold: a table of 4096 slots takes 212992 bytes of block memory, more than 49152}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f, %g : memref<?x3xf64, 1>, memref<?x3xf64, 1>) arity(2)
      scatter(hash(4096, 8, 0)) {
  ^bb0(%d: vector<3xf32>):
    %w = arith.extf %d : vector<3xf32> to vector<3xf64>
    md_exec.yield %w, %w, %w, %w
        : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1>
  return
}
