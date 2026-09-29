// The template that orders the particles by cell, run on its own.
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

// 2000 particles in a periodic cube of the edge 12, from a linear
// congruential generator. Every fifth particle is moved by a multiple of
// the edge, so that it is outside of the cell. The numbers of the
// particles are a permutation of their indices.
//
// The test prints
//
//   - the number of particles that the order does not hold once;
//   - the number of places whose particle is in a cell before that of the
//     place before, or in the same cell with a smaller number;
//   - the sum over the places k of (k + 1) (order[k] + 1), which was found
//     with Inputs/neighbors_reference.py.

func.func private @printI64(i64)
func.func private @printNewline()

func.func @fill(%x: memref<?x3xf64>, %length: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 42 : i64
  %two = arith.constant 2.0 : f64
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    // The cells that the particle is away from the cell: -2 to 2.
    %image = arith.remui %i, %c5 : index
    %wide = arith.index_cast %image : index to i64
    %real = arith.sitofp %wide : i64 to f64
    %away = arith.subf %real, %two : f64
    %shift = arith.mulf %away, %length : f64
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %v = arith.mulf %f, %length : f64
      %moved = arith.addf %v, %shift : f64
      memref.store %moved, %x[%i, %k] : memref<?x3xf64>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

// The cell of particle `i`.
func.func @cell_of(%x: memref<?x3xf64>, %i: index, %length: f64,
                   %width: f64) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %unit = arith.constant 1.0 : f64
  %inverse = arith.divf %unit, %length : f64
  %count = call @mdrt.cell_count(%length, %width) : (f64, f64) -> index
  %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
  %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
  %zi = memref.load %x[%i, %c2] : memref<?x3xf64>
  %wx = call @mdrt.wrap(%xi, %length, %inverse) : (f64, f64, f64) -> f64
  %wy = call @mdrt.wrap(%yi, %length, %inverse) : (f64, f64, f64) -> f64
  %wz = call @mdrt.wrap(%zi, %length, %inverse) : (f64, f64, f64) -> f64
  %cx = call @mdrt.cell_coordinate(%wx, %inverse, %count)
      : (f64, f64, index) -> index
  %cy = call @mdrt.cell_coordinate(%wy, %inverse, %count)
      : (f64, f64, index) -> index
  %cz = call @mdrt.cell_coordinate(%wz, %inverse, %count)
      : (f64, f64, index) -> index
  %zy = arith.muli %cz, %count : index
  %row = arith.addi %zy, %cy : index
  %rows = arith.muli %row, %count : index
  %cell = arith.addi %rows, %cx : index
  return %cell : index
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = arith.constant 2000 : index
  %prime = arith.constant 7919 : index
  %length = arith.constant 12.0 : f64
  %width = arith.constant 1.4 : f64
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %none = arith.constant 0 : i32

  %x = memref.alloc(%count) : memref<?x3xf64>
  call @fill(%x, %length) : (memref<?x3xf64>, f64) -> ()
  %box = vector.broadcast %length : f64 to vector<3xf64>

  %ids = memref.alloc(%count) : memref<?xi32>
  scf.for %i = %c0 to %count step %c1 {
    %scattered = arith.muli %i, %prime : index
    %id = arith.remui %scattered, %count : index
    %narrow = arith.index_cast %id : index to i32
    memref.store %narrow, %ids[%i] : memref<?xi32>
  }

  %order = memref.alloc(%count) : memref<?xi32>
  call @mdrt.spatial_order(%x, %box, %width, %ids, %order)
      : (memref<?x3xf64>, vector<3xf64>, f64, memref<?xi32>, memref<?xi32>)
        -> ()

  // Every particle once.
  %seen = memref.alloc(%count) : memref<?xi32>
  scf.for %i = %c0 to %count step %c1 {
    memref.store %none, %seen[%i] : memref<?xi32>
  }
  scf.for %k = %c0 to %count step %c1 {
    %j32 = memref.load %order[%k] : memref<?xi32>
    %j = arith.index_cast %j32 : i32 to index
    %old = memref.load %seen[%j] : memref<?xi32>
    %more = arith.constant 1 : i32
    %new = arith.addi %old, %more : i32
    memref.store %new, %seen[%j] : memref<?xi32>
  }
  %missing = scf.for %i = %c0 to %count step %c1
      iter_args(%wrong = %zero) -> (i64) {
    %times = memref.load %seen[%i] : memref<?xi32>
    %once = arith.constant 1 : i32
    %differs = arith.cmpi ne, %times, %once : i32
    %d = arith.extui %differs : i1 to i64
    %next = arith.addi %wrong, %d : i64
    scf.yield %next : i64
  }

  // By cell, and within a cell by number.
  %misplaced, %sum = scf.for %k = %c1 to %count step %c1
      iter_args(%wrong = %zero, %s = %one) -> (i64, i64) {
    %before = arith.subi %k, %c1 : index
    %i32 = memref.load %order[%before] : memref<?xi32>
    %j32 = memref.load %order[%k] : memref<?xi32>
    %i = arith.index_cast %i32 : i32 to index
    %j = arith.index_cast %j32 : i32 to index
    %ci = func.call @cell_of(%x, %i, %length, %width)
        : (memref<?x3xf64>, index, f64, f64) -> index
    %cj = func.call @cell_of(%x, %j, %length, %width)
        : (memref<?x3xf64>, index, f64, f64) -> index
    %idi = memref.load %ids[%i] : memref<?xi32>
    %idj = memref.load %ids[%j] : memref<?xi32>
    %back = arith.cmpi ugt, %ci, %cj : index
    %same = arith.cmpi eq, %ci, %cj : index
    %smaller = arith.cmpi sge, %idi, %idj : i32
    %within = arith.andi %same, %smaller : i1
    %bad = arith.ori %back, %within : i1
    %b = arith.extui %bad : i1 to i64
    %next = arith.addi %wrong, %b : i64

    %kw = arith.index_cast %k : index to i64
    %k1 = arith.addi %kw, %one : i64
    %jw = arith.extsi %j32 : i32 to i64
    %j1 = arith.addi %jw, %one : i64
    %product = arith.muli %k1, %j1 : i64
    %s1 = arith.addi %s, %product : i64
    scf.yield %next, %s1 : i64, i64
  }
  // Place 0 is not in the loop: (0 + 1) (order[0] + 1), less the 1 that
  // the sum began with.
  %first32 = memref.load %order[%c0] : memref<?xi32>
  %first = arith.extsi %first32 : i32 to i64
  %total = arith.addi %sum, %first : i64

  // CHECK:      {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 2001267935
  call @printI64(%missing) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%misplaced) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%total) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}
