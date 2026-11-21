// The emitter that builds a neighbor matrix on a device, compared with
// the template for the host, entry by entry.
//
// REQUIRES: cuda
//
// RUN: cat %S/../../lib/Runtime/Templates/NeighborsMatrix.mlir %s \
// RUN: | mdir-opt --add-matrix-build --convert-md-alg-to-gpu \
// RUN: | mlir-opt --gpu-lower-to-nvvm-pipeline="cubin-format=isa" \
// RUN:     --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt_cuda \
// RUN: | FileCheck %s

// Each case places 2000 particles in a periodic cube from a linear
// congruential generator and prints three numbers:
//
//   - the number of entries in the matrix of the device;
//   - the number of counts and entries in which the two matrices differ;
//   - the largest number of neighbors of one particle, from the device.
//
// The rows agree in their order as well: the particles of a cell are
// sorted by index on the device, which is the order of the host. The
// device keeps the matrix in the order of the cells (D86); the test takes
// it back to the particles.
//
// The numbers of entries were found by testing all pairs, with
// Inputs/neighbors_reference.py.

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
  %count = arith.constant 2000 : index
  %x = memref.alloc(%count) : memref<?x3xf64>
  call @fill(%x, %length) : (memref<?x3xf64>, f64) -> ()
  %box = vector.broadcast %length : f64 to vector<3xf64>

  // On the host.
  %counts = memref.alloc(%count) : memref<?xi32>
  %oracle_width = arith.constant 128 : index
  %index = memref.alloc(%count, %oracle_width) : memref<?x?xi32>
  %expected = call @mdrt.build_neighbors_matrix(%x, %box, %reach, %width,
                                                %counts, %index)
      : (memref<?x3xf64>, vector<3xf64>, f64, f64, memref<?xi32>,
         memref<?x?xi32>) -> index

  // On the device.
  %xd = gpu.alloc (%count) : memref<?x3xf64, 1>
  %countsd = gpu.alloc (%count) : memref<?xi32, 1>
  %indexd = gpu.alloc (%count, %row) : memref<?x?xi32, 1>
  %spill_columns = arith.constant 6 : index
  %spill_rows = arith.constant 65536 : index
  %spill = gpu.alloc (%spill_rows, %spill_columns) : memref<?x?xi32, 1>
  %orderd = gpu.alloc (%count) : memref<?xi32, 1>
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %xd, %x : memref<?x3xf64, 1>, memref<?x3xf64>
  gpu.wait [%t1]
  // No excluded pairs: a buffer of no elements, which has no rows
  // (ragged rows, include/mdir/Conversion/RaggedRows.h).
  %none = gpu.alloc (%c0) : memref<?xi32, 1>
  call @mdrt_gpu_build_neighbors_matrix(
      %xd, %box, %reach, %width, %none, %countsd, %indexd, %orderd, %spill)
      : (memref<?x3xf64, 1>, vector<3xf64>, f64, f64, memref<?xi32, 1>,
         memref<?xi32, 1>, memref<?x?xi32, 1>, memref<?xi32, 1>, memref<?x?xi32, 1>) -> ()
  %counts2 = memref.alloc(%count) : memref<?xi32>
  %index2 = memref.alloc(%count, %row) : memref<?x?xi32>
  %t2 = gpu.wait async
  %t3 = gpu.memcpy async [%t2] %counts2, %countsd
      : memref<?xi32>, memref<?xi32, 1>
  %t4 = gpu.memcpy async [%t3] %index2, %indexd
      : memref<?x?xi32>, memref<?x?xi32, 1>
  %order2 = memref.alloc(%count) : memref<?xi32>
  %t5 = gpu.memcpy async [%t4] %order2, %orderd
      : memref<?xi32>, memref<?xi32, 1>
  %tail = memref.alloc(%spill_rows, %spill_columns) : memref<?x?xi32>
  %t6 = gpu.memcpy async [%t5] %tail, %spill
      : memref<?x?xi32>, memref<?x?xi32, 1>
  gpu.wait [%t6]

  %zero = arith.constant 0 : i64
  // Row `i` of the device is the row of the particle `order[i]` of the
  // host, and an entry `q` of the device is the entry `order[q]` (D86).
  %entries, %wrong = scf.for %i = %c0 to %count step %c1
      iter_args(%p = %zero, %w = %zero) -> (i64, i64) {
    %hi32 = memref.load %order2[%i] : memref<?xi32>
    %hi = arith.index_cast %hi32 : i32 to index
    %ci = memref.load %counts[%hi] : memref<?xi32>
    %di = memref.load %counts2[%i] : memref<?xi32>
    %cn = arith.index_cast %di : i32 to index
    %cw = arith.extsi %di : i32 to i64
    %p1 = arith.addi %p, %cw : i64
    %limit = arith.index_cast %row : index to i32
    %prefix = arith.minui %ci, %limit : i32
    %differs = arith.cmpi ne, %prefix, %di : i32
    %d = arith.extui %differs : i1 to i64
    %w0 = arith.addi %w, %d : i64
    %w1 = scf.for %k = %c0 to %cn step %c1 iter_args(%t = %w0) -> (i64) {
      %j = memref.load %index[%hi, %k] : memref<?x?xi32>
      %q2 = memref.load %index2[%i, %k] : memref<?x?xi32>
      %q = arith.index_cast %q2 : i32 to index
      %j2 = memref.load %order2[%q] : memref<?xi32>
      %other = arith.cmpi ne, %j, %j2 : i32
      %o = arith.extui %other : i1 to i64
      %next = arith.addi %t, %o : i64
      scf.yield %next : i64
    }
    scf.yield %p1, %w1 : i64, i64
  }
  %status_col = arith.constant 2 : index
  %largest = memref.load %tail[%c0, %status_col] : memref<?x?xi32>
  %lw = arith.extui %largest : i32 to i64
  %ew = arith.index_cast %expected : index to i64
  %same = arith.cmpi ne, %lw, %ew : i64
  %s = arith.extui %same : i1 to i64
  // Match each tail record once, independent of atomic reservation order.
  %length32 = memref.load %tail[%c0, %c0] : memref<?x?xi32>
  %length_idx = arith.index_cast %length32 : i32 to index
  %stop = arith.addi %length_idx, %c1 : index
  %minus_one = arith.constant -1 : i32
  %bad = scf.for %r = %c1 to %stop step %c1 iter_args(%w = %wrong) -> i64 {
    %p32 = memref.load %tail[%r, %c0] : memref<?x?xi32>
    %q32 = memref.load %tail[%r, %c1] : memref<?x?xi32>
    %p = arith.index_cast %p32 : i32 to index
    %q = arith.index_cast %q32 : i32 to index
    %id = memref.load %order2[%p] : memref<?xi32>
    %i = arith.index_cast %id : i32 to index
    %j = memref.load %order2[%q] : memref<?xi32>
    %n32 = memref.load %counts[%i] : memref<?xi32>
    %n = arith.index_cast %n32 : i32 to index
    %two = arith.constant 2 : index
    %rank32 = memref.load %tail[%r, %two] : memref<?x?xi32>
    %rank = arith.index_castui %rank32 : i32 to index
    %k = arith.addi %row, %rank : index
    %valid = arith.cmpi ult, %k, %n : index
    %hit = scf.if %valid -> i1 {
      %value = memref.load %index[%i, %k] : memref<?x?xi32>
      %three = arith.constant 3 : index
      %four = arith.constant 4 : index
      %off32 = memref.load %tail[%p, %four] : memref<?x?xi32>
      %off = arith.index_castui %off32 : i32 to index
      %at = arith.addi %off, %rank : index
      %sorted = memref.load %tail[%at, %three] : memref<?x?xi32>
      %ordered = arith.cmpi eq, %sorted, %q32 : i32
      %matches = arith.cmpi eq, %value, %j : i32
      %eq = arith.andi %matches, %ordered : i1
      memref.store %minus_one, %index[%i, %k] : memref<?x?xi32>
      scf.yield %eq : i1
    } else {
      %no = arith.constant false
      scf.yield %no : i1
    }
    %one = arith.constant 1 : i64
    %miss = arith.select %hit, %zero, %one : i64
    %next = arith.addi %w, %miss : i64
    scf.yield %next : i64
  }
  %total = scf.for %i = %c0 to %count step %c1 iter_args(%sum = %zero) -> i64 {
    %n = memref.load %counts[%i] : memref<?xi32>
    %v = arith.extui %n : i32 to i64
    %next = arith.addi %sum, %v : i64
    scf.yield %next : i64
  }
  %tail_count = arith.extui %length32 : i32 to i64
  %observed = arith.addi %entries, %tail_count : i64
  %missing = arith.cmpi ne, %total, %observed : i64
  %m = arith.extui %missing : i1 to i64
  %errors = arith.addi %bad, %m : i64
  %all = arith.addi %errors, %s : i64
  call @printI64(%entries) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%all) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%lw) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @main() {
  %wide = arith.constant 128 : index
  %narrow = arith.constant 8 : index
  %reach = arith.constant 1.5 : f64
  %half = arith.constant 0.75 : f64

  // Seven cells along each direction, as wide as the reach and more.
  // CHECK:      32704
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 31
  %l0 = arith.constant 12.0 : f64
  call @run(%l0, %reach, %reach, %wide)
      : (f64, f64, f64, index) -> ()

  // Cells of half the reach: fifteen along each direction, of which five
  // are searched.
  // CHECK-NEXT: 32704
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 31
  call @run(%l0, %reach, %half, %wide)
      : (f64, f64, f64, index) -> ()

  // Two cells: the cell before and the cell after a cell are the same.
  // CHECK-NEXT: 48778
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 42
  %l1 = arith.constant 7.0 : f64
  %r1 = arith.constant 1.0 : f64
  %w1 = arith.constant 3.0 : f64
  call @run(%l1, %r1, %w1, %wide)
      : (f64, f64, f64, index) -> ()

  // One cell.
  // CHECK-NEXT: 43958
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 40
  %l2 = arith.constant 5.8 : f64
  %r2 = arith.constant 0.8 : f64
  call @run(%l2, %r2, %w1, %wide)
      : (f64, f64, f64, index) -> ()

  // Rows that are too narrow: every row is full, and the largest count
  // tells the caller so.
  // CHECK-NEXT: 16000
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 44
  %l3 = arith.constant 10.0 : f64
  call @run(%l3, %reach, %reach, %narrow)
      : (f64, f64, f64, index) -> ()
  return
}

func.func private @mdrt_gpu_build_neighbors_matrix(
    %x: memref<?x3xf64, 1>, %box: vector<3xf64>, %reach: f64,
    %cell_width: f64, %excluded: memref<?xi32, 1>,
    %counts: memref<?xi32, 1>, %index: memref<?x?xi32, 1>,
    %order: memref<?xi32, 1>, %spill: memref<?x?xi32, 1>) -> ()
