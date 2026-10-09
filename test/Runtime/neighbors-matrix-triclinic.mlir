// The template that builds a neighbor matrix in a triclinic cell
// (docs/triclinic-m2.md), against every pair and every image: each pair
// whose nearest image is within the reach is in the row of each of its
// particles once, and no pair beyond the reach is.
//
// RUN: cat %S/../../lib/Runtime/Templates/NeighborsMatrix.mlir %s \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// Each case places 400 particles in a triclinic cell H, of the diagonal
// a_x, b_y, c_z and the tilts b_x, c_x, c_y, at fractional coordinates from
// a linear congruential generator, each moved by -3 to 3 lattice vectors
// along each of a, b, and c as unwrapped positions are, and prints:
//
//   - the number of ordered pairs (i, j) whose nearest image, the least in
//     f64 of the images up to three lattice vectors along each of a, b, and
//     c from the one of the pass of Section 2, is within the reach;
//   - the number of those that row i does not hold;
//   - the number of pairs that a row holds more than once;
//   - the number of pairs that a row holds whose nearest image is farther
//     than the reach and a thousandth of it (the search takes a margin for
//     the rounding of f32).
//
// The cells are narrow. In the first cases the reach is at half of the
// least of a_x, b_y, c_z or a little under it, where the image of the one
// pass along c, b, and a is the nearest one; in the others it is beyond,
// up to more than the least of the diagonal, where the build tests the
// other images within the reach (#258, docs/triclinic-m2.md, Section 2).

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

// The square of the length of the nearest image of the displacement d: the
// image of the pass along c, b, and a, then the least of the 343 up to
// three lattice vectors along each of a, b, and c from it.
func.func @nearest2(%dx: f64, %dy: f64, %dz: f64, %h: vector<6xf64>) -> f64 {
  %ax = vector.extract %h[0] : f64 from vector<6xf64>
  %by = vector.extract %h[1] : f64 from vector<6xf64>
  %cz = vector.extract %h[2] : f64 from vector<6xf64>
  %bx = vector.extract %h[3] : f64 from vector<6xf64>
  %cx = vector.extract %h[4] : f64 from vector<6xf64>
  %cy = vector.extract %h[5] : f64 from vector<6xf64>
  %nc0 = arith.divf %dz, %cz : f64
  %nc = math.roundeven %nc0 : f64
  %x1a = arith.mulf %nc, %cx : f64
  %y1a = arith.mulf %nc, %cy : f64
  %z1a = arith.mulf %nc, %cz : f64
  %x1 = arith.subf %dx, %x1a : f64
  %y1 = arith.subf %dy, %y1a : f64
  %z1 = arith.subf %dz, %z1a : f64
  %nb0 = arith.divf %y1, %by : f64
  %nb = math.roundeven %nb0 : f64
  %x2a = arith.mulf %nb, %bx : f64
  %y2a = arith.mulf %nb, %by : f64
  %x2 = arith.subf %x1, %x2a : f64
  %y2 = arith.subf %y1, %y2a : f64
  %na0 = arith.divf %x2, %ax : f64
  %na = math.roundeven %na0 : f64
  %x3a = arith.mulf %na, %ax : f64
  %x3 = arith.subf %x2, %x3a : f64

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %three = arith.constant 3.0 : f64
  %far = arith.constant 1.0e30 : f64
  %least = scf.for %ic = %c0 to %c7 step %c1 iter_args(%mc = %far) -> (f64) {
    %ic_i = arith.index_cast %ic : index to i64
    %ic_f = arith.sitofp %ic_i : i64 to f64
    %kc = arith.subf %ic_f, %three : f64
    %after_b = scf.for %ib = %c0 to %c7 step %c1 iter_args(%mb = %mc) -> (f64) {
      %ib_i = arith.index_cast %ib : index to i64
      %ib_f = arith.sitofp %ib_i : i64 to f64
      %kb = arith.subf %ib_f, %three : f64
      %after_a = scf.for %ia = %c0 to %c7 step %c1 iter_args(%ma = %mb) -> (f64) {
        %ia_i = arith.index_cast %ia : index to i64
        %ia_f = arith.sitofp %ia_i : i64 to f64
        %ka = arith.subf %ia_f, %three : f64
        // d - k H.
        %sxa = arith.mulf %ka, %ax : f64
        %sxb = arith.mulf %kb, %bx : f64
        %sxc = arith.mulf %kc, %cx : f64
        %sxab = arith.addf %sxa, %sxb : f64
        %sx = arith.addf %sxab, %sxc : f64
        %syb = arith.mulf %kb, %by : f64
        %syc = arith.mulf %kc, %cy : f64
        %sy = arith.addf %syb, %syc : f64
        %sz = arith.mulf %kc, %cz : f64
        %ex = arith.subf %x3, %sx : f64
        %ey = arith.subf %y2, %sy : f64
        %ez = arith.subf %z1, %sz : f64
        %ex2 = arith.mulf %ex, %ex : f64
        %ey2 = arith.mulf %ey, %ey : f64
        %ez2 = arith.mulf %ez, %ez : f64
        %exy = arith.addf %ex2, %ey2 : f64
        %e2 = arith.addf %exy, %ez2 : f64
        %next = arith.minimumf %ma, %e2 : f64
        scf.yield %next : f64
      }
      scf.yield %after_a : f64
    }
    scf.yield %after_b : f64
  }
  return %least : f64
}

func.func @run(%h: vector<6xf64>, %reach: f64, %width: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %count = arith.constant 400 : index
  %x = memref.alloc(%count) : memref<?x3xf64>
  %counts = memref.alloc(%count) : memref<?xi32>
  %index = memref.alloc(%count, %count) : memref<?x?xi32>
  %held = memref.alloc(%count, %count) : memref<?x?xi32>
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

  %largest = call @mdrt.build_neighbors_matrix_triclinic(%x, %h, %widths,
      %reach, %width, %counts, %index)
      : (memref<?x3xf64>, vector<6xf64>, vector<3xf64>, f64, f64,
         memref<?xi32>, memref<?x?xi32>) -> index

  // How often row i holds j.
  %zero32 = arith.constant 0 : i32
  %one32 = arith.constant 1 : i32
  scf.for %i = %c0 to %count step %c1 {
    scf.for %j = %c0 to %count step %c1 {
      memref.store %zero32, %held[%i, %j] : memref<?x?xi32>
    }
  }
  scf.for %i = %c0 to %count step %c1 {
    %ci = memref.load %counts[%i] : memref<?xi32>
    %cn = arith.index_cast %ci : i32 to index
    scf.for %k = %c0 to %cn step %c1 {
      %j32 = memref.load %index[%i, %k] : memref<?x?xi32>
      %j = arith.index_cast %j32 : i32 to index
      %old = memref.load %held[%i, %j] : memref<?x?xi32>
      %new = arith.addi %old, %one32 : i32
      memref.store %new, %held[%i, %j] : memref<?x?xi32>
    }
  }

  %reach2 = arith.mulf %reach, %reach : f64
  %slack = arith.constant 1.001 : f64
  %beyond = arith.mulf %reach, %slack : f64
  %beyond2 = arith.mulf %beyond, %beyond : f64
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %pairs, %missing, %twice, %far = scf.for %i = %c0 to %count step %c1
      iter_args(%p0 = %zero, %m0 = %zero, %t0 = %zero, %f0 = %zero)
      -> (i64, i64, i64, i64) {
    %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
    %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
    %zi = memref.load %x[%i, %c2] : memref<?x3xf64>
    %p2, %m2, %t2, %f2 = scf.for %j = %c0 to %count step %c1
        iter_args(%p1 = %p0, %m1 = %m0, %t1 = %t0, %f1 = %f0)
        -> (i64, i64, i64, i64) {
      %xj = memref.load %x[%j, %c0] : memref<?x3xf64>
      %yj = memref.load %x[%j, %c1] : memref<?x3xf64>
      %zj = memref.load %x[%j, %c2] : memref<?x3xf64>
      %dx = arith.subf %xi, %xj : f64
      %dy = arith.subf %yi, %yj : f64
      %dz = arith.subf %zi, %zj : f64
      %r2 = func.call @nearest2(%dx, %dy, %dz, %h)
          : (f64, f64, f64, vector<6xf64>) -> f64
      %times = memref.load %held[%i, %j] : memref<?x?xi32>
      %other = arith.cmpi ne, %i, %j : index
      %near0 = arith.cmpf olt, %r2, %reach2 : f64
      %near = arith.andi %near0, %other : i1
      %absent = arith.cmpi eq, %times, %zero32 : i32
      %lost = arith.andi %near, %absent : i1
      %repeated = arith.cmpi sgt, %times, %one32 : i32
      %out0 = arith.cmpf ogt, %r2, %beyond2 : f64
      %present = arith.cmpi sgt, %times, %zero32 : i32
      %out = arith.andi %out0, %present : i1
      %near_w = arith.extui %near : i1 to i64
      %lost_w = arith.extui %lost : i1 to i64
      %repeated_w = arith.extui %repeated : i1 to i64
      %out_w = arith.extui %out : i1 to i64
      %p3 = arith.addi %p1, %near_w : i64
      %m3 = arith.addi %m1, %lost_w : i64
      %t3 = arith.addi %t1, %repeated_w : i64
      %f3 = arith.addi %f1, %out_w : i64
      scf.yield %p3, %m3, %t3, %f3 : i64, i64, i64, i64
    }
    scf.yield %p2, %m2, %t2, %f2 : i64, i64, i64, i64
  }

  call @printI64(%pairs) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%missing) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%twice) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%far) : (i64) -> ()
  call @printNewline() : () -> ()
  memref.dealloc %x : memref<?x3xf64>
  memref.dealloc %counts : memref<?xi32>
  memref.dealloc %index : memref<?x?xi32>
  memref.dealloc %held : memref<?x?xi32>
  return
}

func.func @main() {
  %reach = arith.constant 1.5 : f64
  %half = arith.constant 0.75 : f64
  %third = arith.constant 0.5 : f64

  // A rhombic dodecahedron of GROMACS, c_x = c_y = a_x/2 on the bounds of
  // the reduced form, with c_z twice the reach: cells of the reach, of half
  // of it, and of a third.
  // CHECK:      41238
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %dodec = arith.constant dense<[4.242640687119286, 4.242640687119286, 3.0, 0.0, 2.121320343559643, 2.121320343559643]> : vector<6xf64>
  call @run(%dodec, %reach, %reach) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 41238
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%dodec, %reach, %half) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 41238
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%dodec, %reach, %third) : (vector<6xf64>, f64, f64) -> ()

  // A truncated octahedron in Amber's frame, c_y = -b_y/2 on a bound, with
  // c_z twice the reach.
  // CHECK-NEXT: 58666
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %oct = arith.constant dense<[3.6742346141747673, 3.4641016151377544, 3.0, -1.2247448713915890, -1.2247448713915890, -1.7320508075688772]> : vector<6xf64>
  call @run(%oct, %reach, %half) : (vector<6xf64>, f64, f64) -> ()

  // A hexagonal cell, b_x = -a_x/2 on a bound, with b_y twice the reach.
  // CHECK-NEXT: 67604
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %hexa = arith.constant dense<[3.4641016151377544, 3.0, 3.2, -1.7320508075688772, 0.0, 0.0]> : vector<6xf64>
  call @run(%hexa, %reach, %half) : (vector<6xf64>, f64, f64) -> ()

  // Every tilt on its bound, b_x = a_x/2, c_x = -a_x/2, c_y = b_y/2, in a
  // cell of equal a_x, b_y, c_z, twice the reach.
  // CHECK-NEXT: 83416
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %bounds = arith.constant dense<[3.0, 3.0, 3.0, 1.5, -1.5, 1.5]> : vector<6xf64>
  call @run(%bounds, %reach, %half) : (vector<6xf64>, f64, f64) -> ()

  // Tilts of both signs within their bounds, a reach a little under half
  // of c_z, the least of the diagonal.
  // CHECK-NEXT: 69682
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %small = arith.constant dense<[4.2, 4.0, 3.9, 1.0, -1.5, 1.2]> : vector<6xf64>
  %r1 = arith.constant 1.9 : f64
  %w1 = arith.constant 0.95 : f64
  call @run(%small, %r1, %w1) : (vector<6xf64>, f64, f64) -> ()

  // Reaches beyond half of the least of a_x, b_y, c_z, where the image of
  // the one pass need not be the nearest (#258): a little beyond in the
  // dodecahedron, where the build that took that image alone left out 42
  // pairs; then 1.4 times the bound, just under c_z, and beyond c_z, where
  // a pair has several images within the reach and a row holds it once.
  // CHECK-NEXT: 43790
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %beyond0 = arith.constant 1.53 : f64
  call @run(%dodec, %beyond0, %half) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 114516
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %beyond1 = arith.constant 2.1 : f64
  %w2 = arith.constant 1.05 : f64
  call @run(%dodec, %beyond1, %w2) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 159598
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %beyond2 = arith.constant 2.95 : f64
  call @run(%dodec, %beyond2, %w2) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 159600
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %beyond3 = arith.constant 3.6 : f64
  call @run(%dodec, %beyond3, %reach) : (vector<6xf64>, f64, f64) -> ()

  // The same in the octahedron, in the hexagonal cell, with every tilt on
  // its bound, and with tilts of both signs.
  // CHECK-NEXT: 147368
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%oct, %beyond1, %w2) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 147636
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%hexa, %beyond1, %w2) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 158578
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%bounds, %beyond1, %w2) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 159600
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%bounds, %beyond2, %w2) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 159600
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %beyond4 = arith.constant 3.5 : f64
  call @run(%small, %beyond4, %w1) : (vector<6xf64>, f64, f64) -> ()

  // A flat cell, a_x and b_y six times c_z, with c_x and c_y a quarter of
  // them: the nearest image of some pairs is two lattice vectors c from the
  // image of the pass, as (0, 0, -1.5) is of (3, 3, 0.5).
  // CHECK-NEXT: 102430
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %flat = arith.constant dense<[6.0, 6.0, 1.0, 0.0, 1.5, 1.5]> : vector<6xf64>
  %beyond5 = arith.constant 1.9 : f64
  call @run(%flat, %beyond5, %w1) : (vector<6xf64>, f64, f64) -> ()
  // CHECK-NEXT: 159600
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %beyond6 = arith.constant 2.9 : f64
  call @run(%flat, %beyond6, %w1) : (vector<6xf64>, f64, f64) -> ()
  return
}
