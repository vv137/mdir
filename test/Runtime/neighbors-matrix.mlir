// The template that builds a neighbor matrix, run on its own and compared
// with a search over all pairs.
//
// RUN: cat %S/../../lib/Runtime/Templates/NeighborsMatrix.mlir %s \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// RUN: cat %S/../../lib/Runtime/Templates/NeighborsMatrix.mlir %s \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%openmp \
// RUN: | FileCheck %s

// Each case places 200 particles in a periodic cube from a linear
// congruential generator and prints three numbers:
//
//   - the number of entries in the matrix;
//   - the sum over the entries (i, j) of (i + 1) (j + 1), which depends on
//     which pairs were found but not on their order;
//   - the largest number of neighbors of one particle.
//
// The search takes the pairs within the reach and those within its margin
// beyond. The reference values were found by testing all pairs, with
// Inputs/neighbors_reference.py. In the first case, 2664 of the entries are
// within the reach.

func.func private @printI64(i64)
func.func private @printNewline()

func.func @fill(%x: memref<?x3xf64>, %length: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 42 : i64
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %v = arith.mulf %f, %length : f64
      memref.store %v, %x[%i, %k] : memref<?x3xf64>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

func.func @run(%length: f64, %reach: f64, %width: f64, %row: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = arith.constant 200 : index
  %x = memref.alloc(%count) : memref<?x3xf64>
  %counts = memref.alloc(%count) : memref<?xi32>
  %index = memref.alloc(%count, %row) : memref<?x?xi32>
  call @fill(%x, %length) : (memref<?x3xf64>, f64) -> ()
  %box = vector.broadcast %length : f64 to vector<3xf64>
  %largest = call @mdrt.build_neighbors_matrix(%x, %box, %reach, %width,
                                               %counts, %index)
      : (memref<?x3xf64>, vector<3xf64>, f64, f64, memref<?xi32>,
         memref<?x?xi32>) -> index

  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %entries, %sum = scf.for %i = %c0 to %count step %c1
      iter_args(%p = %zero, %s = %zero) -> (i64, i64) {
    %ci = memref.load %counts[%i] : memref<?xi32>
    %cn = arith.index_cast %ci : i32 to index
    %cw = arith.extsi %ci : i32 to i64
    %p1 = arith.addi %p, %cw : i64
    %iw = arith.index_cast %i : index to i64
    %i1 = arith.addi %iw, %one : i64
    %s1 = scf.for %k = %c0 to %cn step %c1 iter_args(%t = %s) -> (i64) {
      %j = memref.load %index[%i, %k] : memref<?x?xi32>
      %jw = arith.extsi %j : i32 to i64
      %j1 = arith.addi %jw, %one : i64
      %prod = arith.muli %i1, %j1 : i64
      %t1 = arith.addi %t, %prod : i64
      scf.yield %t1 : i64
    }
    scf.yield %p1, %s1 : i64, i64
  }
  %lw = arith.index_cast %largest : index to i64
  call @printI64(%entries) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%sum) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%lw) : (i64) -> ()
  call @printNewline() : () -> ()
  memref.dealloc %x : memref<?x3xf64>
  memref.dealloc %counts : memref<?xi32>
  memref.dealloc %index : memref<?x?xi32>
  return
}

func.func @main() {
  %wide = arith.constant 128 : index
  %narrow = arith.constant 8 : index
  %reach = arith.constant 1.5 : f64

  // Three cells along each direction, as wide as the reach and more.
  // CHECK:      2666
  // CHECK-NEXT: 27305838
  // CHECK-NEXT: 22
  %l0 = arith.constant 6.0 : f64
  call @run(%l0, %reach, %reach, %wide) : (f64, f64, f64, index) -> ()

  // Cells of half the reach: seven along each direction, of which five are
  // searched. The pairs are the same.
  // CHECK-NEXT: 2666
  // CHECK-NEXT: 27305838
  // CHECK-NEXT: 22
  %half = arith.constant 0.75 : f64
  call @run(%l0, %reach, %half, %wide) : (f64, f64, f64, index) -> ()

  // Cells of a third of the reach: eleven along each direction, of which
  // seven are searched.
  // CHECK-NEXT: 2666
  // CHECK-NEXT: 27305838
  // CHECK-NEXT: 22
  %third = arith.constant 0.5 : f64
  call @run(%l0, %reach, %third, %wide) : (f64, f64, f64, index) -> ()

  // Cells of half the reach in a cell that has four of them along each
  // direction: all are searched, each once.
  // CHECK-NEXT: 13056
  // CHECK-NEXT: 132165308
  // CHECK-NEXT: 80
  %small = arith.constant 3.5 : f64
  call @run(%small, %reach, %half, %wide) : (f64, f64, f64, index) -> ()

  // Two cells: the cell before and the cell after a cell are the same.
  // CHECK-NEXT: 13056
  // CHECK-NEXT: 132165308
  // CHECK-NEXT: 80
  %l1 = arith.constant 3.5 : f64
  call @run(%l1, %reach, %reach, %wide) : (f64, f64, f64, index) -> ()

  // One cell.
  // CHECK-NEXT: 18476
  // CHECK-NEXT: 187655900
  // CHECK-NEXT: 107
  %l2 = arith.constant 2.9 : f64
  %r2 = arith.constant 1.4 : f64
  call @run(%l2, %r2, %reach, %wide) : (f64, f64, f64, index) -> ()

  // Rows that are too narrow: every row is full, and the largest count
  // tells the caller so.
  // CHECK-NEXT: 1600
  // CHECK-NEXT: {{[0-9]+}}
  // CHECK-NEXT: 33
  %l3 = arith.constant 5.0 : f64
  call @run(%l3, %reach, %reach, %narrow) : (f64, f64, f64, index) -> ()
  return
}
