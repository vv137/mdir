// The template that builds groups of 16 particles sharing a list of
// neighbors on a device in a triclinic cell (docs/triclinic-m2.md, D89),
// against every image of every pair: each image within the reach has a
// bit in the lists once, of its particle in the group, and an excluded
// pair has none (docs/groups-m1.md, Section 5.2).
//
// REQUIRES: cuda
//
// RUN: cat %S/../../lib/Runtime/Templates/NeighborsGroupsGPUTriclinic.mlir %s \
// RUN: | mlir-opt --gpu-lower-to-nvvm-pipeline="cubin-format=isa" \
// RUN:     --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt_cuda \
// RUN: | FileCheck %s

// Each case places 2000 particles, or as many as it says, in a triclinic
// cell H, of the diagonal a_x, b_y, c_z and the tilts b_x, c_x, c_y, at fractional coordinates from
// a linear congruential generator, each moved by -3 to 3 lattice vectors
// along each of a, b, and c as unwrapped positions are, and prints:
//
//   - the number of pairs that the lists hold (bits of the masks);
//   - the number of pairs whose bits in the lists are not as many as their
//     images within the reach (in f64 on the host, of the images up to two
//     lattice vectors along each of a, b, and c from the nearest), or whose
//     displacements in the frames of the groups (the place shifts and the
//     shift of the entry, lattice vectors applied as n H) are not those of
//     these images, by the sum of their squares; and of excluded pairs
//     that are in the lists;
//   - the number of bits of the lists whose image, in the frames of the
//     groups, is farther than the reach, with a margin for the rounding of
//     f32;
//   - whether a group lies across two places that are not of one chunk (0).
//   - the number of pairs with several images within the reach.

func.func private @printI64(i64)
func.func private @printNewline()

// The cell as a vector of six: a_x, b_y, c_z, b_x, c_x, c_y.
func.func @fill(%x: memref<?x3xf64>, %h: vector<6xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 42 : i64
  %c7 = arith.constant 7 : index
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
      %boxes = arith.subf %r7f, %three_f : f64
      %v = arith.addf %f, %boxes : f64
      memref.store %v, %s[%k] : memref<3xf64>
      scf.yield %t2 : i64
    }
    // x = s H.
    %sa = memref.load %s[%c0] : memref<3xf64>
    %sb = memref.load %s[%c1] : memref<3xf64>
    %c2 = arith.constant 2 : index
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

// The nearest image of the displacement d: the one-pass image, then the
// least of the 27 a lattice vector or less from it.
func.func @nearest(%dx: f64, %dy: f64, %dz: f64, %h: vector<6xf64>) -> (f64, f64, f64) {
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
  %c3 = arith.constant 3 : index
  %big = arith.constant 1.0e300 : f64
  %one = arith.constant 1.0 : f64
  %bd, %bxr, %byr, %bzr = scf.for %i = %c0 to %c3 step %c1 iter_args(%d0 = %big, %rx0 = %x3, %ry0 = %y2, %rz0 = %z1) -> (f64, f64, f64, f64) {
    %r1:4 = scf.for %j = %c0 to %c3 step %c1 iter_args(%d1 = %d0, %rx1 = %rx0, %ry1 = %ry0, %rz1 = %rz0) -> (f64, f64, f64, f64) {
      %r2:4 = scf.for %k = %c0 to %c3 step %c1 iter_args(%d2 = %d1, %rx2 = %rx1, %ry2 = %ry1, %rz2 = %rz1) -> (f64, f64, f64, f64) {
        %ii = arith.index_cast %i : index to i64
        %jj = arith.index_cast %j : index to i64
        %kk = arith.index_cast %k : index to i64
        %fi0 = arith.sitofp %ii : i64 to f64
        %fj0 = arith.sitofp %jj : i64 to f64
        %fk0 = arith.sitofp %kk : i64 to f64
        %fi = arith.subf %fi0, %one : f64
        %fj = arith.subf %fj0, %one : f64
        %fk = arith.subf %fk0, %one : f64
        // n = (fi, fj, fk) along (a, b, c).
        %ta = arith.mulf %fi, %ax : f64
        %tb = arith.mulf %fj, %bx : f64
        %tc = arith.mulf %fk, %cx : f64
        %tab = arith.addf %ta, %tb : f64
        %tx = arith.addf %tab, %tc : f64
        %ub = arith.mulf %fj, %by : f64
        %uc = arith.mulf %fk, %cy : f64
        %ty = arith.addf %ub, %uc : f64
        %tz = arith.mulf %fk, %cz : f64
        %ex = arith.subf %x3, %tx : f64
        %ey = arith.subf %y2, %ty : f64
        %ez = arith.subf %z1, %tz : f64
        %ex2 = arith.mulf %ex, %ex : f64
        %ey2 = arith.mulf %ey, %ey : f64
        %ez2 = arith.mulf %ez, %ez : f64
        %exy = arith.addf %ex2, %ey2 : f64
        %e2 = arith.addf %exy, %ez2 : f64
        %less = arith.cmpf olt, %e2, %d2 : f64
        %nd = arith.select %less, %e2, %d2 : f64
        %nx = arith.select %less, %ex, %rx2 : f64
        %ny = arith.select %less, %ey, %ry2 : f64
        %nz = arith.select %less, %ez, %rz2 : f64
        scf.yield %nd, %nx, %ny, %nz : f64, f64, f64, f64
      }
      scf.yield %r2#0, %r2#1, %r2#2, %r2#3 : f64, f64, f64, f64
    }
    scf.yield %r1#0, %r1#1, %r1#2, %r1#3 : f64, f64, f64, f64
  }
  return %bxr, %byr, %bzr : f64, f64, f64
}

// The square of the displacement of the pair (i at place p, j at place q) in
// the frames of the groups, with the place shifts of `shift` and the shift
// of the entry in its mask `m`: the square of the distance of the image
// that the entry takes.
func.func @framed2(%x: memref<?x3xf64>, %shift: memref<?xi32>,
                          %m: i32, %p: index, %q: index, %i: index,
                          %j: index, %h: vector<6xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %sp = memref.load %shift[%p] : memref<?xi32>
  %sq = memref.load %shift[%q] : memref<?xi32>
  %mask10 = arith.constant 1023 : i32
  %off = arith.constant 512 : i32
  %mask5 = arith.constant 31 : i32
  %offset5 = arith.constant 16 : i32
  %n = memref.alloca() : memref<3xf64>
  %c3 = arith.constant 3 : index
  scf.for %k = %c0 to %c3 step %c1 {
    %k32 = arith.index_cast %k : index to i32
    %ten = arith.constant 10 : i32
    %five = arith.constant 5 : i32
    %sixteen = arith.constant 16 : i32
    %bits10 = arith.muli %k32, %ten : i32
    %bits5a = arith.muli %k32, %five : i32
    %bits5 = arith.addi %bits5a, %sixteen : i32
    %pp0 = arith.shrui %sp, %bits10 : i32
    %pp1 = arith.andi %pp0, %mask10 : i32
    %pp = arith.subi %pp1, %off : i32
    %qq0 = arith.shrui %sq, %bits10 : i32
    %qq1 = arith.andi %qq0, %mask10 : i32
    %qq = arith.subi %qq1, %off : i32
    %ee0 = arith.shrui %m, %bits5 : i32
    %ee1 = arith.andi %ee0, %mask5 : i32
    %ee = arith.subi %ee1, %offset5 : i32
    // The lattice vectors between the frames: s_p - s_q - e.
    %d0 = arith.subi %pp, %qq : i32
    %d1 = arith.subi %d0, %ee : i32
    %df = arith.sitofp %d1 : i32 to f64
    memref.store %df, %n[%k] : memref<3xf64>
  }
  %na = memref.load %n[%c0] : memref<3xf64>
  %nb = memref.load %n[%c1] : memref<3xf64>
  %nc = memref.load %n[%c2] : memref<3xf64>
  %ax = vector.extract %h[0] : f64 from vector<6xf64>
  %by = vector.extract %h[1] : f64 from vector<6xf64>
  %cz = vector.extract %h[2] : f64 from vector<6xf64>
  %bx = vector.extract %h[3] : f64 from vector<6xf64>
  %cx = vector.extract %h[4] : f64 from vector<6xf64>
  %cy = vector.extract %h[5] : f64 from vector<6xf64>
  %ta = arith.mulf %na, %ax : f64
  %tb = arith.mulf %nb, %bx : f64
  %tc = arith.mulf %nc, %cx : f64
  %tab = arith.addf %ta, %tb : f64
  %tx = arith.addf %tab, %tc : f64
  %ub = arith.mulf %nb, %by : f64
  %uc = arith.mulf %nc, %cy : f64
  %ty = arith.addf %ub, %uc : f64
  %tz = arith.mulf %nc, %cz : f64
  %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
  %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
  %zi = memref.load %x[%i, %c2] : memref<?x3xf64>
  %xj = memref.load %x[%j, %c0] : memref<?x3xf64>
  %yj = memref.load %x[%j, %c1] : memref<?x3xf64>
  %zj = memref.load %x[%j, %c2] : memref<?x3xf64>
  %dx = arith.subf %xi, %xj : f64
  %dy = arith.subf %yi, %yj : f64
  %dz = arith.subf %zi, %zj : f64
  %fx = arith.addf %dx, %tx : f64
  %fy = arith.addf %dy, %ty : f64
  %fz = arith.addf %dz, %tz : f64
  %fx2 = arith.mulf %fx, %fx : f64
  %fy2 = arith.mulf %fy, %fy : f64
  %fz2 = arith.mulf %fz, %fz : f64
  %fxy = arith.addf %fx2, %fy2 : f64
  %f2 = arith.addf %fxy, %fz2 : f64
  return %f2 : f64
}

// The number of the images of the pair (i, j) within sqrt(limit2), of those
// up to two lattice vectors along each of a, b, and c from the nearest, and
// the sum of the squares of their lengths.
func.func @images(%x: memref<?x3xf64>, %i: index, %j: index, %h: vector<6xf64>, %limit2: f64) -> (i64, f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
  %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
  %zi = memref.load %x[%i, %c2] : memref<?x3xf64>
  %xj = memref.load %x[%j, %c0] : memref<?x3xf64>
  %yj = memref.load %x[%j, %c1] : memref<?x3xf64>
  %zj = memref.load %x[%j, %c2] : memref<?x3xf64>
  %dx = arith.subf %xi, %xj : f64
  %dy = arith.subf %yi, %yj : f64
  %dz = arith.subf %zi, %zj : f64
  %rx, %ry, %rz = func.call @nearest(%dx, %dy, %dz, %h) : (f64, f64, f64, vector<6xf64>) -> (f64, f64, f64)
  %ax = vector.extract %h[0] : f64 from vector<6xf64>
  %by = vector.extract %h[1] : f64 from vector<6xf64>
  %cz = vector.extract %h[2] : f64 from vector<6xf64>
  %bx = vector.extract %h[3] : f64 from vector<6xf64>
  %cx = vector.extract %h[4] : f64 from vector<6xf64>
  %cy = vector.extract %h[5] : f64 from vector<6xf64>
  %two = arith.constant 2.0 : f64
  %zero = arith.constant 0 : i64
  %zf = arith.constant 0.0 : f64
  %n, %s = scf.for %ic = %c0 to %c5 step %c1 iter_args(%n0 = %zero, %s0 = %zf) -> (i64, f64) {
    %ic_i = arith.index_cast %ic : index to i64
    %ic_f = arith.sitofp %ic_i : i64 to f64
    %kc = arith.subf %ic_f, %two : f64
    %n1, %s1 = scf.for %ib = %c0 to %c5 step %c1 iter_args(%n2 = %n0, %s2 = %s0) -> (i64, f64) {
      %ib_i = arith.index_cast %ib : index to i64
      %ib_f = arith.sitofp %ib_i : i64 to f64
      %kb = arith.subf %ib_f, %two : f64
      %n3, %s3 = scf.for %ia = %c0 to %c5 step %c1 iter_args(%n4 = %n2, %s4 = %s2) -> (i64, f64) {
        %ia_i = arith.index_cast %ia : index to i64
        %ia_f = arith.sitofp %ia_i : i64 to f64
        %ka = arith.subf %ia_f, %two : f64
        %sxa = arith.mulf %ka, %ax : f64
        %sxb = arith.mulf %kb, %bx : f64
        %sxc = arith.mulf %kc, %cx : f64
        %sxab = arith.addf %sxa, %sxb : f64
        %sx = arith.addf %sxab, %sxc : f64
        %syb = arith.mulf %kb, %by : f64
        %syc = arith.mulf %kc, %cy : f64
        %sy = arith.addf %syb, %syc : f64
        %sz = arith.mulf %kc, %cz : f64
        %ex = arith.subf %rx, %sx : f64
        %ey = arith.subf %ry, %sy : f64
        %ez = arith.subf %rz, %sz : f64
        %ex2 = arith.mulf %ex, %ex : f64
        %ey2 = arith.mulf %ey, %ey : f64
        %ez2 = arith.mulf %ez, %ez : f64
        %exy = arith.addf %ex2, %ey2 : f64
        %e2 = arith.addf %exy, %ez2 : f64
        %in = arith.cmpf ole, %e2, %limit2 : f64
        %inc = arith.extui %in : i1 to i64
        %add = arith.select %in, %e2, %zf : f64
        %n5 = arith.addi %n4, %inc : i64
        %s5 = arith.addf %s4, %add : f64
        scf.yield %n5, %s5 : i64, f64
      }
      scf.yield %n3, %s3 : i64, f64
    }
    scf.yield %n1, %s1 : i64, f64
  }
  return %n, %s : i64, f64
}

// Excluded pairs (i, i + d), d = 1 .. k, around the ring of the particles,
// as an incidence structure: a row for each particle with its number of
// pairs, 2k, then for each the number of the pair, the place of the
// particle in it, and its two members.
func.func @exclusions(%n: index, %k: index) -> memref<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %k8 = arith.muli %k, %c8 : index
  %width = arith.addi %k8, %c1 : index
  %e = memref.alloc(%n, %width) : memref<?x?xi32>
  %k2 = arith.muli %k, %c2 : index
  %k2_32 = arith.index_cast %k2 : index to i32
  scf.for %i = %c0 to %n step %c1 {
    memref.store %k2_32, %e[%i, %c0] : memref<?x?xi32>
    scf.for %d1 = %c0 to %k step %c1 {
      %d = arith.addi %d1, %c1 : index
      // The pair (i, i + d), the particle first.
      %id = arith.addi %i, %d : index
      %j = arith.remui %id, %n : index
      %num0 = arith.muli %i, %k : index
      %num = arith.addi %num0, %d1 : index
      %col0 = arith.muli %d1, %c4 : index
      %col = arith.addi %col0, %c1 : index
      %col1 = arith.addi %col, %c1 : index
      %col2 = arith.addi %col, %c2 : index
      %col3 = arith.addi %col, %c3 : index
      %num32 = arith.index_cast %num : index to i32
      %zero32 = arith.constant 0 : i32
      %i32 = arith.index_cast %i : index to i32
      %j32 = arith.index_cast %j : index to i32
      memref.store %num32, %e[%i, %col] : memref<?x?xi32>
      memref.store %zero32, %e[%i, %col1] : memref<?x?xi32>
      memref.store %i32, %e[%i, %col2] : memref<?x?xi32>
      memref.store %j32, %e[%i, %col3] : memref<?x?xi32>
      // The pair (i - d, i), the particle second.
      %in = arith.addi %i, %n : index
      %ind = arith.subi %in, %d : index
      %h = arith.remui %ind, %n : index
      %hnum0 = arith.muli %h, %k : index
      %hnum = arith.addi %hnum0, %d1 : index
      %hd = arith.addi %d1, %k : index
      %hcol0 = arith.muli %hd, %c4 : index
      %hcol = arith.addi %hcol0, %c1 : index
      %hcol1 = arith.addi %hcol, %c1 : index
      %hcol2 = arith.addi %hcol, %c2 : index
      %hcol3 = arith.addi %hcol, %c3 : index
      %hnum32 = arith.index_cast %hnum : index to i32
      %one32 = arith.constant 1 : i32
      %h32 = arith.index_cast %h : index to i32
      memref.store %hnum32, %e[%i, %hcol] : memref<?x?xi32>
      memref.store %one32, %e[%i, %hcol1] : memref<?x?xi32>
      memref.store %h32, %e[%i, %hcol2] : memref<?x?xi32>
      memref.store %i32, %e[%i, %hcol3] : memref<?x?xi32>
    }
  }
  return %e : memref<?x?xi32>
}

func.func @run(%h: vector<6xf64>, %reach: f64, %degree: index, %poison: i1,
               %count: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %exclude = arith.cmpi ne, %degree, %c0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %x = memref.alloc(%count) : memref<?x3xf64>
  call @fill(%x, %h) : (memref<?x3xf64>, vector<6xf64>) -> ()
  // Poisoned, particle 7 has no position: the build leaves it out (D107).
  scf.if %poison {
    %c7 = arith.constant 7 : index
    %nan = arith.constant 0x7FF8000000000000 : f64
    memref.store %nan, %x[%c7, %c0] : memref<?x3xf64>
  }

  %capacity = arith.muli %count, %c4 : index
  %groups = arith.divui %capacity, %c16 : index
  %unit_capacity = arith.muli %groups, %c16 : index
  %entry_capacity = arith.muli %unit_capacity, %c64 : index
  %xd = gpu.alloc (%count) : memref<?x3xf64, 1>
  %orderd = gpu.alloc (%capacity) : memref<?xi32, 1>
  %placed = gpu.alloc (%count) : memref<?xi32, 1>
  %entriesd = gpu.alloc (%entry_capacity) : memref<?xi32, 1>
  %masksd = gpu.alloc (%entry_capacity) : memref<?xi32, 1>
  %countsd = gpu.alloc (%groups) : memref<?xi32, 1>
  %unitsd = gpu.alloc (%unit_capacity) : memref<?xi32, 1>
  %ordinalsd = gpu.alloc (%unit_capacity) : memref<?xi32, 1>
  %shiftd = gpu.alloc (%capacity) : memref<?xi32, 1>
  %sizes = memref.alloc() : memref<5xi32>
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %xd, %x : memref<?x3xf64, 1>, memref<?x3xf64>
  gpu.wait [%t1]
  %excluded_host = call @exclusions(%count, %degree) : (index, index) -> memref<?x?xi32>
  %rows = arith.select %exclude, %count, %c0 : index
  %c8 = arith.constant 8 : index
  %width0 = arith.muli %degree, %c8 : index
  %width = arith.addi %width0, %c1 : index
  %excludedd = gpu.alloc (%rows, %width) : memref<?x?xi32, 1>
  scf.if %exclude {
    %t2 = gpu.wait async
    %t3 = gpu.memcpy async [%t2] %excludedd, %excluded_host : memref<?x?xi32, 1>, memref<?x?xi32>
    gpu.wait [%t3]
  }
  func.call @mdrt_gpu_build_neighbors_groups_triclinic(%xd, %h, %reach, %excludedd,
      %orderd, %placed, %entriesd, %masksd, %countsd, %unitsd, %ordinalsd,
      %shiftd, %sizes)
      : (memref<?x3xf64, 1>, vector<6xf64>, f64, memref<?x?xi32, 1>,
         memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>,
         memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>,
         memref<?xi32, 1>, memref<?xi32, 1>, memref<5xi32>) -> ()
  %order = memref.alloc(%capacity) : memref<?xi32>
  %entries = memref.alloc(%entry_capacity) : memref<?xi32>
  %masks = memref.alloc(%entry_capacity) : memref<?xi32>
  %counts = memref.alloc(%groups) : memref<?xi32>
  %units = memref.alloc(%unit_capacity) : memref<?xi32>
  %ordinals = memref.alloc(%unit_capacity) : memref<?xi32>
  %shift = memref.alloc(%capacity) : memref<?xi32>
  %t4 = gpu.wait async
  %t5 = gpu.memcpy async [%t4] %order, %orderd : memref<?xi32>, memref<?xi32, 1>
  %t6 = gpu.memcpy async [%t5] %entries, %entriesd : memref<?xi32>, memref<?xi32, 1>
  %t7 = gpu.memcpy async [%t6] %masks, %masksd : memref<?xi32>, memref<?xi32, 1>
  %t8 = gpu.memcpy async [%t7] %counts, %countsd : memref<?xi32>, memref<?xi32, 1>
  %t9 = gpu.memcpy async [%t8] %units, %unitsd : memref<?xi32>, memref<?xi32, 1>
  %t10 = gpu.memcpy async [%t9] %ordinals, %ordinalsd : memref<?xi32>, memref<?xi32, 1>
  %t11 = gpu.memcpy async [%t10] %shift, %shiftd : memref<?xi32>, memref<?xi32, 1>
  gpu.wait [%t11]
  %blocks32 = memref.load %sizes[%c2] : memref<5xi32>
  %blocks = arith.index_cast %blocks32 : i32 to index

  %places32 = memref.load %sizes[%c0] : memref<5xi32>
  %places = arith.index_cast %places32 : i32 to index
  %used_groups = arith.divui %places, %c16 : index

  // How often each pair is in the lists, and the pairs too far apart.
  %seen = memref.alloc(%count, %count) : memref<?x?xi8>
  %sums = memref.alloc(%count, %count) : memref<?x?xf64>
  %zsum = arith.constant 0.0 : f64
  %z8 = arith.constant 0 : i8
  %one8 = arith.constant 1 : i8
  scf.for %i = %c0 to %count step %c1 {
    scf.for %j = %c0 to %count step %c1 {
      memref.store %z8, %seen[%i, %j] : memref<?x?xi8>
      memref.store %zsum, %sums[%i, %j] : memref<?x?xf64>
    }
  }
  %zero = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  // The build widens the reach by 3e-6 of the sum of the diagonal and the
  // magnitudes of the tilts; a pair within twice that may be there.
  %hm = math.absf %h : vector<6xf64>
  %sum = vector.reduction <add>, %hm : vector<6xf64> into f64
  %tiny = arith.constant 6.0e-6 : f64
  %margin = arith.mulf %sum, %tiny : f64
  %far = arith.addf %reach, %margin : f64
  %far2 = arith.mulf %far, %far : f64
  %zero32 = arith.constant 0 : i32
  %one32 = arith.constant 1 : i32
  // The lists, block by block: block b is block ordinals[b] of the list of
  // group units[b].
  %bits, %too_far = scf.for %blk = %c0 to %blocks step %c1
      iter_args(%b = %zero, %f = %zero) -> (i64, i64) {
    %g32 = memref.load %units[%blk] : memref<?xi32>
    %k32 = memref.load %ordinals[%blk] : memref<?xi32>
    %g = arith.index_cast %g32 : i32 to index
    %k = arith.index_cast %k32 : i32 to index
    %n32 = memref.load %counts[%g] : memref<?xi32>
    %n = arith.index_cast %n32 : i32 to index
    %k64 = arith.muli %k, %c64 : index
    %left = arith.subi %n, %k64 : index
    %ne = arith.minui %left, %c64 : index
    %blk64 = arith.muli %blk, %c64 : index
    %b1, %f1 = scf.for %e = %c0 to %ne step %c1
        iter_args(%bb = %b, %ff = %f) -> (i64, i64) {
      %at = arith.addi %blk64, %e : index
      %q32 = memref.load %entries[%at] : memref<?xi32>
      %m = memref.load %masks[%at] : memref<?xi32>
      %q = arith.index_cast %q32 : i32 to index
      %j32 = memref.load %order[%q] : memref<?xi32>
      %j = arith.index_cast %j32 : i32 to index
      %b2, %f2 = scf.for %u = %c0 to %c16 step %c1
          iter_args(%bbb = %bb, %fff = %ff) -> (i64, i64) {
        %u32 = arith.index_cast %u : index to i32
        %sh = arith.shrui %m, %u32 : i32
        %bit = arith.andi %sh, %one32 : i32
        %set = arith.cmpi ne, %bit, %zero32 : i32
        %r:2 = scf.if %set -> (i64, i64) {
          %g16 = arith.muli %g, %c16 : index
          %p = arith.addi %g16, %u : index
          %i32v = memref.load %order[%p] : memref<?xi32>
          %i = arith.index_cast %i32v : i32 to index
          %lo = arith.minui %i, %j : index
          %hi = arith.maxui %i, %j : index
          %old = memref.load %seen[%lo, %hi] : memref<?x?xi8>
          %new = arith.addi %old, %one8 : i8
          memref.store %new, %seen[%lo, %hi] : memref<?x?xi8>
          %d2 = func.call @framed2(%x, %shift, %m, %p, %q, %i, %j, %h) : (memref<?x3xf64>, memref<?xi32>, i32, index, index, index, index, vector<6xf64>) -> f64
          %out = arith.cmpf ogt, %d2, %far2 : f64
          %olds = memref.load %sums[%lo, %hi] : memref<?x?xf64>
          %news = arith.addf %olds, %d2 : f64
          memref.store %news, %sums[%lo, %hi] : memref<?x?xf64>
          %o = arith.extui %out : i1 to i64
          %nb = arith.addi %bbb, %one64 : i64
          %nf = arith.addi %fff, %o : i64
          scf.yield %nb, %nf : i64, i64
        } else {
          scf.yield %bbb, %fff : i64, i64
        }
        scf.yield %r#0, %r#1 : i64, i64
      }
      scf.yield %b2, %f2 : i64, i64
    }
    scf.yield %b1, %f1 : i64, i64
  }

  // Every image of every pair within the reach once, but excluded pairs:
  // the number of the bits of a pair in the lists is that of its images
  // within the reach (a few more may be there within the rounding of the
  // reach), and, where the two numbers of images agree, the sum of the
  // squares of the displacements in the frames is that of those images.
  %reach2 = arith.mulf %reach, %reach : f64
  %tol = arith.constant 1.0e-5 : f64
  %wrong, %several = scf.for %i = %c0 to %count step %c1 iter_args(%w = %zero, %sv = %zero) -> (i64, i64) {
    %i1 = arith.addi %i, %c1 : index
    %wi, %svi = scf.for %j = %i1 to %count step %c1 iter_args(%ww = %w, %svv = %sv) -> (i64, i64) {
      %within, %within_sum = func.call @images(%x, %i, %j, %h, %reach2) : (memref<?x3xf64>, index, index, vector<6xf64>, f64) -> (i64, f64)
      %near, %near_sum = func.call @images(%x, %i, %j, %h, %far2) : (memref<?x3xf64>, index, index, vector<6xf64>, f64) -> (i64, f64)
      // Excluded: within `degree` of each other around the ring.
      %dij = arith.subi %j, %i : index
      %dji = arith.subi %count, %dij : index
      %dring = arith.minui %dij, %dji : index
      %pair = arith.cmpi ule, %dring, %degree : index
      %is_excluded = arith.andi %pair, %exclude : i1
      %want = arith.select %is_excluded, %zero, %within : i64
      %most = arith.select %is_excluded, %zero, %near : i64
      %got8 = memref.load %seen[%i, %j] : memref<?x?xi8>
      %got = arith.extui %got8 : i8 to i64
      %got_sum = memref.load %sums[%i, %j] : memref<?x?xf64>
      %few = arith.cmpi slt, %got, %want : i64
      %many = arith.cmpi sgt, %got, %most : i64
      %sharp0 = arith.cmpi eq, %within, %near : i64
      %true = arith.constant true
      %kept = arith.xori %is_excluded, %true : i1
      %sharp = arith.andi %sharp0, %kept : i1
      %ds = arith.subf %got_sum, %within_sum : f64
      %ads = math.absf %ds : f64
      %off = arith.cmpf ogt, %ads, %tol : f64
      %other = arith.andi %sharp, %off : i1
      %bad0 = arith.ori %few, %many : i1
      %bad = arith.ori %bad0, %other : i1
      %b = arith.extui %bad : i1 to i64
      %next = arith.addi %ww, %b : i64
      %two64 = arith.constant 2 : i64
      %is_several = arith.cmpi sge, %want, %two64 : i64
      %sn0 = arith.extui %is_several : i1 to i64
      %sn = arith.addi %svv, %sn0 : i64
      scf.yield %next, %sn : i64, i64
    }
    scf.yield %wi, %svi : i64, i64
  }

  // The places of a group are in one chunk of 64.
  %spans = scf.for %g = %c0 to %used_groups step %c1 iter_args(%s = %zero) -> (i64) {
    %g16 = arith.muli %g, %c16 : index
    %g16e = arith.addi %g16, %c16 : index
    %last = arith.subi %g16e, %c1 : index
    %ca = arith.divui %g16, %c64 : index
    %cb = arith.divui %last, %c64 : index
    %apart = arith.cmpi ne, %ca, %cb : index
    %a = arith.extui %apart : i1 to i64
    %next = arith.addi %s, %a : i64
    scf.yield %next : i64
  }

  call @printI64(%bits) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%wrong) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%too_far) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%spans) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%several) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @main() {
  %n2000 = arith.constant 2000 : index
  %reach = arith.constant 1.5 : f64
  %none = arith.constant 0 : index
  %one = arith.constant 1 : index
  %nine = arith.constant 9 : index
  %no = arith.constant false
  %yes = arith.constant true

  // A truncated octahedron of side 12 in Amber's frame, c_y = -b_y/2 on a
  // bound of the reduced form.
  // CHECK:      21149
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %oct = arith.constant dense<[12.0, 11.313708498984761, 9.797958971132712, -4.0, -4.0, -5.656854249492381]> : vector<6xf64>
  call @run(%oct, %reach, %none, %no, %n2000) : (vector<6xf64>, f64, index, i1, index) -> ()

  // With the excluded pairs (i, i + 1).
  // CHECK-NEXT: 21125
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%oct, %reach, %one, %no, %n2000) : (vector<6xf64>, f64, index, i1, index) -> ()

  // Nine excluded pairs a particle each way (D106).
  // CHECK-NEXT: 20953
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%oct, %reach, %nine, %no, %n2000) : (vector<6xf64>, f64, index, i1, index) -> ()

  // A position that is not a number (D107).
  // CHECK-NEXT: 21132
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%oct, %reach, %none, %yes, %n2000) : (vector<6xf64>, f64, index, i1, index) -> ()

  // A rhombic dodecahedron of GROMACS, c_x = c_y = a_x/2 on the bounds:
  // the windows of x move by more than the cell is wide.
  // CHECK-NEXT: 23156
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %dodec = arith.constant dense<[12.0, 12.0, 8.485281374238571, 0.0, 6.0, 6.0]> : vector<6xf64>
  call @run(%dodec, %reach, %none, %no, %n2000) : (vector<6xf64>, f64, index, i1, index) -> ()

  // A cell narrower than twice the reach and the extent of a group, with
  // tilts of both signs: some pairs need images a lattice vector away from
  // the one nearest to the center of the group (D115).
  // CHECK-NEXT: 875982
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %small = arith.constant dense<[4.2, 4.0, 3.9, 1.0, -1.5, 1.2]> : vector<6xf64>
  %r3 = arith.constant 1.9 : f64
  call @run(%small, %r3, %none, %no, %n2000) : (vector<6xf64>, f64, index, i1, index) -> ()

  // A hexagonal cell of CHARMM-GUI, b_x = -a_x/2 on a bound.
  // CHECK-NEXT: 25088
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %hexa = arith.constant dense<[12.0, 10.392304845413264, 9.0, -6.0, 0.0, 0.0]> : vector<6xf64>
  call @run(%hexa, %reach, %none, %no, %n2000) : (vector<6xf64>, f64, index, i1, index) -> ()
  // Reaches of more than half of the least of a_x, b_y, c_z (#258,
  // D242): a pair then has several images within the reach, each
  // with an entry bit of its own. Narrow cells with 320 particles, whose
  // lists fit the buffers of the test: a dodecahedron, an octahedron, a
  // hexagonal cell, and a cell with every tilt on its bound, at 0.6, 0.85,
  // and 0.975 of the least of the diagonal.
  %n320 = arith.constant 320 : index
  %q0 = arith.constant 1.8 : f64
  %q1 = arith.constant 2.55 : f64
  %q2 = arith.constant 2.925 : f64
  %ndodec = arith.constant dense<[4.242640687119286, 4.242640687119286, 3.0, 0.0, 2.121320343559643, 2.121320343559643]> : vector<6xf64>
  // CHECK-NEXT: 22998
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 0
  call @run(%ndodec, %q0, %none, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  // CHECK-NEXT: 65313
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 15162
  call @run(%ndodec, %q1, %one, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  // CHECK-NEXT: 99408
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 35791
  call @run(%ndodec, %q2, %none, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  %noct = arith.constant dense<[3.6742346141747673, 3.4641016151377544, 3.0, -1.224744871391589, -1.224744871391589, -1.7320508075688772]> : vector<6xf64>
  // CHECK-NEXT: 32533
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 0
  call @run(%noct, %q0, %none, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  // CHECK-NEXT: 92527
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 33437
  call @run(%noct, %q1, %one, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  // CHECK-NEXT: 140360
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 46186
  call @run(%noct, %q2, %none, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  %nhexa = arith.constant dense<[3.4641016151377544, 3.0, 3.2, -1.7320508075688772, 0.0, 0.0]> : vector<6xf64>
  // CHECK-NEXT: 37351
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 889
  call @run(%nhexa, %q0, %none, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  // CHECK-NEXT: 105866
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 42846
  call @run(%nhexa, %q1, %one, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  // CHECK-NEXT: 161158
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 50026
  call @run(%nhexa, %q2, %none, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  %nbounds = arith.constant dense<[3.0, 3.0, 3.0, 1.5, -1.5, 1.5]> : vector<6xf64>
  // CHECK-NEXT: 46107
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 2703
  call @run(%nbounds, %q0, %none, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  // CHECK-NEXT: 130675
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 45014
  call @run(%nbounds, %q1, %one, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  // CHECK-NEXT: 197964
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 50644
  call @run(%nbounds, %q2, %none, %no, %n320) : (vector<6xf64>, f64, index, i1, index) -> ()
  return
}
