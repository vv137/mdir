// The template that builds a neighbor matrix on a device in a triclinic
// cell (docs/triclinic-m2.md), against the matrix of the host, which
// neighbors-matrix-triclinic.mlir compares with every pair and every image.
//
// REQUIRES: cuda
//
// RUN: cat %S/../../lib/Runtime/Templates/NeighborsMatrix.mlir \
// RUN:     %S/../../lib/Runtime/Templates/NeighborsMatrixGPU.mlir %s \
// RUN: | mlir-opt --gpu-lower-to-nvvm-pipeline="cubin-format=isa" \
// RUN:     --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt_cuda \
// RUN: | FileCheck %s

// Each case places 400 particles in a triclinic cell H, of the diagonal
// a_x, b_y, c_z and the tilts b_x, c_x, c_y, as neighbors-matrix-triclinic.mlir
// does, with the reach at half of the least of a_x, b_y, c_z or a little
// under it, and prints three numbers:
//
//   - the number of entries in the matrix of the device;
//   - the number of counts and entries in which the two matrices differ;
//   - the largest number of neighbors of one particle, from the device.
//
// The device keeps the matrix in the order of the cells (D86); the test
// takes it back to the particles.

func.func private @printI64(i64)
func.func private @printNewline()

// The cell as a vector of six: a_x, b_y, c_z, b_x, c_x, c_y.
func.func @fill(%x: memref<?x3xf64>, %h: vector<6xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 42 : i64
  %three_f = arith.constant 3.0 : f64
  %ax = vector.extract %h[0] : f64 from vector<6xf64>
  %by = vector.extract %h[1] : f64 from vector<6xf64>
  %cz = vector.extract %h[2] : f64 from vector<6xf64>
  %bx = vector.extract %h[3] : f64 from vector<6xf64>
  %cx = vector.extract %h[4] : f64 from vector<6xf64>
  %cy = vector.extract %h[5] : f64 from vector<6xf64>
  %s = memref.alloca() : memref<3xf64>
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%sk = %s0) -> (i64) {
      %t0 = arith.muli %sk, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %ik = arith.addi %i, %k : index
      %ik3 = arith.addi %ik, %k : index
      %ik33 = arith.addi %ik3, %k : index
      %r7 = arith.remui %ik33, %c7 : index
      %r7i = arith.index_cast %r7 : index to i64
      %r7f = arith.sitofp %r7i : i64 to f64
      %cells = arith.subf %r7f, %three_f : f64
      %v = arith.addf %f, %cells : f64
      memref.store %v, %s[%k] : memref<3xf64>
      scf.yield %t2 : i64
    }
    // x = s H.
    %sa = memref.load %s[%c0] : memref<3xf64>
    %sb = memref.load %s[%c1] : memref<3xf64>
    %sc = memref.load %s[%c2] : memref<3xf64>
    %xa = arith.mulf %sa, %ax : f64
    %xb = arith.mulf %sb, %bx : f64
    %xc = arith.mulf %sc, %cx : f64
    %xab = arith.addf %xa, %xb : f64
    %xx = arith.addf %xab, %xc : f64
    %yb = arith.mulf %sb, %by : f64
    %yc = arith.mulf %sc, %cy : f64
    %yy = arith.addf %yb, %yc : f64
    %zz = arith.mulf %sc, %cz : f64
    memref.store %xx, %x[%i, %c0] : memref<?x3xf64>
    memref.store %yy, %x[%i, %c1] : memref<?x3xf64>
    memref.store %zz, %x[%i, %c2] : memref<?x3xf64>
    scf.yield %s3 : i64
  }
  return
}

func.func @run(%h: vector<6xf64>, %reach: f64, %width: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = arith.constant 400 : index
  %row = arith.constant 400 : index
  %x = memref.alloc(%count) : memref<?x3xf64>
  call @fill(%x, %h) : (memref<?x3xf64>, vector<6xf64>) -> ()

  // The widths of the cell between its faces: the volume over the areas
  // of the faces, |b x c|, |c x a|, and |a x b|.
  %ax = vector.extract %h[0] : f64 from vector<6xf64>
  %by = vector.extract %h[1] : f64 from vector<6xf64>
  %cz = vector.extract %h[2] : f64 from vector<6xf64>
  %bx = vector.extract %h[3] : f64 from vector<6xf64>
  %cx = vector.extract %h[4] : f64 from vector<6xf64>
  %cy = vector.extract %h[5] : f64 from vector<6xf64>
  %n0 = arith.mulf %by, %cz : f64
  %n1 = arith.mulf %bx, %cz : f64
  %n2a = arith.mulf %bx, %cy : f64
  %n2b = arith.mulf %by, %cx : f64
  %n2 = arith.subf %n2a, %n2b : f64
  %n00 = arith.mulf %n0, %n0 : f64
  %n11 = arith.mulf %n1, %n1 : f64
  %n22 = arith.mulf %n2, %n2 : f64
  %n01 = arith.addf %n00, %n11 : f64
  %n012 = arith.addf %n01, %n22 : f64
  %area_a = math.sqrt %n012 : f64
  %ab = arith.mulf %ax, %by : f64
  %volume = arith.mulf %ab, %cz : f64
  %wa = arith.divf %volume, %area_a : f64
  %cz2 = arith.mulf %cz, %cz : f64
  %cy2 = arith.mulf %cy, %cy : f64
  %czy = arith.addf %cz2, %cy2 : f64
  %root = math.sqrt %czy : f64
  %bcz = arith.mulf %by, %cz : f64
  %wb = arith.divf %bcz, %root : f64
  %widths = vector.from_elements %wa, %wb, %cz : vector<3xf64>


  // On the host.
  %counts = memref.alloc(%count) : memref<?xi32>
  %index = memref.alloc(%count, %row) : memref<?x?xi32>
  %expected = call @mdrt.build_neighbors_matrix_triclinic(%x, %h, %widths,
      %reach, %width, %counts, %index)
      : (memref<?x3xf64>, vector<6xf64>, vector<3xf64>, f64, f64,
         memref<?xi32>, memref<?x?xi32>) -> index

  // On the device.
  %xd = gpu.alloc (%count) : memref<?x3xf64, 1>
  %countsd = gpu.alloc (%count) : memref<?xi32, 1>
  %indexd = gpu.alloc (%count, %row) : memref<?x?xi32, 1>
  %orderd = gpu.alloc (%count) : memref<?xi32, 1>
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %xd, %x : memref<?x3xf64, 1>, memref<?x3xf64>
  gpu.wait [%t1]
  // No excluded pairs: a buffer with no rows.
  %none = gpu.alloc (%c0, %c1) : memref<?x?xi32, 1>
  %largest, %not_numbers = call @mdrt_gpu_build_neighbors_matrix_triclinic(
      %xd, %h, %widths, %reach, %width, %none, %countsd, %indexd, %orderd)
      : (memref<?x3xf64, 1>, vector<6xf64>, vector<3xf64>, f64, f64,
         memref<?x?xi32, 1>, memref<?xi32, 1>, memref<?x?xi32, 1>,
         memref<?xi32, 1>) -> (index, index)
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
  gpu.wait [%t5]

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
    %differs = arith.cmpi ne, %ci, %di : i32
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
  %lw = arith.index_cast %largest : index to i64
  %ew = arith.index_cast %expected : index to i64
  %same = arith.cmpi ne, %lw, %ew : i64
  %s = arith.extui %same : i1 to i64
  %all = arith.addi %wrong, %s : i64
  call @printI64(%entries) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%all) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%lw) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @main() {
  %reach = arith.constant 1.5 : f64
  %half = arith.constant 0.75 : f64
  %third = arith.constant 0.5 : f64

  // A rhombic dodecahedron of GROMACS, c_x = c_y = a_x/2 on the bounds of
  // the reduced form, with c_z twice the reach: cells of the reach, of half
  // of it, and of a third.
  // CHECK:      41250
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 122
  %dodec = arith.constant dense<[4.242640687119286, 4.242640687119286, 3.0, 0.0, 2.121320343559643, 2.121320343559643]> : vector<6xf64>
  call @run(%dodec, %reach, %reach) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 41250
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 122
  call @run(%dodec, %reach, %half) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 41250
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 122
  call @run(%dodec, %reach, %third) : (vector<6xf64>, f64, f64) -> ()

  // A truncated octahedron in Amber's frame, c_y = -b_y/2 on a bound, with
  // c_z twice the reach.
  // CHECK-NEXT: 58676
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 177
  %oct = arith.constant dense<[3.6742346141747673, 3.4641016151377544, 3.0, -1.2247448713915890, -1.2247448713915890, -1.7320508075688772]> : vector<6xf64>
  call @run(%oct, %reach, %half) : (vector<6xf64>, f64, f64) -> ()

  // A hexagonal cell, b_x = -a_x/2 on a bound, with b_y twice the reach.
  // CHECK-NEXT: 67616
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 188
  %hexa = arith.constant dense<[3.4641016151377544, 3.0, 3.2, -1.7320508075688772, 0.0, 0.0]> : vector<6xf64>
  call @run(%hexa, %reach, %half) : (vector<6xf64>, f64, f64) -> ()

  // Every tilt on its bound, b_x = a_x/2, c_x = -a_x/2, c_y = b_y/2, in a
  // cell of equal a_x, b_y, c_z, twice the reach.
  // CHECK-NEXT: 83424
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 238
  %bounds = arith.constant dense<[3.0, 3.0, 3.0, 1.5, -1.5, 1.5]> : vector<6xf64>
  call @run(%bounds, %reach, %half) : (vector<6xf64>, f64, f64) -> ()

  // Tilts of both signs within their bounds, a reach a little under half
  // of c_z, the least of the diagonal.
  // CHECK-NEXT: 69684
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 201
  %small = arith.constant dense<[4.2, 4.0, 3.9, 1.0, -1.5, 1.2]> : vector<6xf64>
  %r1 = arith.constant 1.9 : f64
  %w1 = arith.constant 0.95 : f64
  call @run(%small, %r1, %w1) : (vector<6xf64>, f64, f64) -> ()
  return
}
