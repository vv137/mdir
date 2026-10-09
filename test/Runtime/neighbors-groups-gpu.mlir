// The template that builds groups of 16 particles sharing a list of
// neighbors on a device (D89), against every pair: each pair within the
// reach is in the lists once, with the bit of its particle in the group,
// and an excluded pair is not.
//
// REQUIRES: cuda
//
// RUN: cat %S/../../lib/Runtime/Templates/NeighborsGroupsGPU.mlir %s \
// RUN: | mlir-opt --gpu-lower-to-nvvm-pipeline="cubin-format=isa" \
// RUN:     --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt_cuda \
// RUN: | FileCheck %s

// Each case places 2000 particles, or as many as it says, in a periodic cube from a linear
// congruential generator (the positions of neighbors-matrix-gpu.mlir),
// each moved by -3 to 3 edges of the cube along each axis as unwrapped
// positions are, and prints:
//
//   - the number of pairs that the lists hold (bits of the masks);
//   - the number of pairs whose bits in the lists are not as many as their
//     images within the reach (in f64 on the host, of the images up to two
//     edges away along each axis), or whose displacements in the frames of
//     the groups (the place shifts and the shift of the entry) are not
//     those of these images, by the sum of their squares; and of excluded
//     pairs that are in the lists;
//   - the number of bits of the lists whose image, in the frames of the
//     groups, is farther than the reach, with a margin for the rounding of
//     f32;
//   - whether a group lies across two places that are not of one chunk (0).
//   - the number of pairs with several images within the reach.
//
// The build widens the reach by 3e-6 of the sum of the edges of the cell,
// more than the rounding of the positions in f32 can move a distance, as
// the build of the matrix does: the lists hold the pairs of the matrices of
// neighbors-matrix-gpu.mlir (32704 entries are 16352 pairs), a few more
// than those within the reach in f64 (16349).

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
  %c7 = arith.constant 7 : index
  %three_f = arith.constant 3.0 : f64
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %v0 = arith.mulf %f, %length : f64
      %ik = arith.addi %i, %k : index
      %ik3 = arith.addi %ik, %k : index
      %ik33 = arith.addi %ik3, %k : index
      %r7 = arith.remui %ik33, %c7 : index
      %r7i = arith.index_cast %r7 : index to i64
      %r7f = arith.sitofp %r7i : i64 to f64
      %boxes = arith.subf %r7f, %three_f : f64
      %away = arith.mulf %boxes, %length : f64
      %v = arith.addf %v0, %away : f64
      memref.store %v, %x[%i, %k] : memref<?x3xf64>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

// The square of the displacement of the pair (i at place p, j at place q) in
// the frames of the groups, with the place shifts of `shift` and the shift
// of the entry in its mask `m`: the square of the distance of the image
// that the entry takes.
func.func @framed2(%x: memref<?x3xf64>, %shift: memref<?xi32>,
                          %m: i32, %p: index, %q: index, %i: index,
                          %j: index, %length: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %sp = memref.load %shift[%p] : memref<?xi32>
  %sq = memref.load %shift[%q] : memref<?xi32>
  %ten = arith.constant 10 : i32
  %width4 = arith.constant 4 : i32
  %sixteen = arith.constant 16 : i32
  %mask10 = arith.constant 1023 : i32
  %mask4 = arith.constant 15 : i32
  %off = arith.constant 512 : i32
  %offset4 = arith.constant 4 : i32
  %none = arith.constant 0.0 : f64
  %sum = scf.for %k = %c0 to %c3 step %c1 iter_args(%b = %none) -> (f64) {
    %k32 = arith.index_cast %k : index to i32
    %bits10 = arith.muli %k32, %ten : i32
    %bits4a = arith.muli %k32, %width4 : i32
    %bits4 = arith.addi %bits4a, %sixteen : i32
    %pp0 = arith.shrui %sp, %bits10 : i32
    %pp1 = arith.andi %pp0, %mask10 : i32
    %pp = arith.subi %pp1, %off : i32
    %qq0 = arith.shrui %sq, %bits10 : i32
    %qq1 = arith.andi %qq0, %mask10 : i32
    %qq = arith.subi %qq1, %off : i32
    %ee0 = arith.shrui %m, %bits4 : i32
    %ee1 = arith.andi %ee0, %mask4 : i32
    %ee = arith.subi %ee1, %offset4 : i32
    %ppf = arith.sitofp %pp : i32 to f64
    %qqf = arith.sitofp %qq : i32 to f64
    %eef = arith.sitofp %ee : i32 to f64
    %xi = memref.load %x[%i, %k] : memref<?x3xf64>
    %xj = memref.load %x[%j, %k] : memref<?x3xf64>
    %ai0 = arith.mulf %ppf, %length : f64
    %ai = arith.addf %xi, %ai0 : f64
    %qe = arith.addf %qqf, %eef : f64
    %aj0 = arith.mulf %qe, %length : f64
    %aj = arith.addf %xj, %aj0 : f64
    %df = arith.subf %ai, %aj : f64
    %df2 = arith.mulf %df, %df : f64
    %nb = arith.addf %b, %df2 : f64
    scf.yield %nb : f64
  }
  return %sum : f64
}

// The number of the images of the pair (i, j) within sqrt(limit2), of those
// up to two edges of the cube along each axis from the nearest, and the sum
// of the squares of their lengths.
func.func @images(%x: memref<?x3xf64>, %i: index, %j: index, %length: f64, %limit2: f64) -> (i64, f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %r = memref.alloca() : memref<3xf64>
  scf.for %k = %c0 to %c3 step %c1 {
    %xi = memref.load %x[%i, %k] : memref<?x3xf64>
    %xj = memref.load %x[%j, %k] : memref<?x3xf64>
    %d = arith.subf %xi, %xj : f64
    %q = arith.divf %d, %length : f64
    %n = math.roundeven %q : f64
    %nl = arith.mulf %n, %length : f64
    %e = arith.subf %d, %nl : f64
    memref.store %e, %r[%k] : memref<3xf64>
  }
  %rx = memref.load %r[%c0] : memref<3xf64>
  %ry = memref.load %r[%c1] : memref<3xf64>
  %rz = memref.load %r[%c2] : memref<3xf64>
  %two = arith.constant 2.0 : f64
  %zero = arith.constant 0 : i64
  %zf = arith.constant 0.0 : f64
  %n, %s = scf.for %ic = %c0 to %c5 step %c1 iter_args(%n0 = %zero, %s0 = %zf) -> (i64, f64) {
    %ic_i = arith.index_cast %ic : index to i64
    %ic_f = arith.sitofp %ic_i : i64 to f64
    %kc = arith.subf %ic_f, %two : f64
    %sz = arith.mulf %kc, %length : f64
    %ez = arith.subf %rz, %sz : f64
    %ez2 = arith.mulf %ez, %ez : f64
    %n1, %s1 = scf.for %ib = %c0 to %c5 step %c1 iter_args(%n2 = %n0, %s2 = %s0) -> (i64, f64) {
      %ib_i = arith.index_cast %ib : index to i64
      %ib_f = arith.sitofp %ib_i : i64 to f64
      %kb = arith.subf %ib_f, %two : f64
      %sy = arith.mulf %kb, %length : f64
      %ey = arith.subf %ry, %sy : f64
      %ey2 = arith.mulf %ey, %ey : f64
      %ezy = arith.addf %ez2, %ey2 : f64
      %n3, %s3 = scf.for %ia = %c0 to %c5 step %c1 iter_args(%n4 = %n2, %s4 = %s2) -> (i64, f64) {
        %ia_i = arith.index_cast %ia : index to i64
        %ia_f = arith.sitofp %ia_i : i64 to f64
        %ka = arith.subf %ia_f, %two : f64
        %sx = arith.mulf %ka, %length : f64
        %ex = arith.subf %rx, %sx : f64
        %ex2 = arith.mulf %ex, %ex : f64
        %e2 = arith.addf %ezy, %ex2 : f64
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

func.func @run(%length: f64, %reach: f64, %degree: index, %poison: i1,
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
  call @fill(%x, %length) : (memref<?x3xf64>, f64) -> ()
  // Poisoned, particle 7 has no position: the build leaves it out (D107).
  scf.if %poison {
    %c7 = arith.constant 7 : index
    %nan = arith.constant 0x7FF8000000000000 : f64
    memref.store %nan, %x[%c7, %c0] : memref<?x3xf64>
  }
  %box = vector.broadcast %length : f64 to vector<3xf64>

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
  func.call @mdrt_gpu_build_neighbors_groups(%xd, %box, %reach, %excludedd,
      %orderd, %placed, %entriesd, %masksd, %countsd, %unitsd, %ordinalsd,
      %shiftd, %sizes)
      : (memref<?x3xf64, 1>, vector<3xf64>, f64, memref<?x?xi32, 1>,
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
  // The build widens the reach by 3e-6 of the sum of the edges; a pair
  // within twice that may be there.
  %edges = arith.constant 3.0 : f64
  %sum = arith.mulf %length, %edges : f64
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
          %d2 = func.call @framed2(%x, %shift, %m, %p, %q, %i, %j, %length) : (memref<?x3xf64>, memref<?xi32>, i32, index, index, index, index, f64) -> f64
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
      %within, %within_sum = func.call @images(%x, %i, %j, %length, %reach2) : (memref<?x3xf64>, index, index, f64, f64) -> (i64, f64)
      %near, %near_sum = func.call @images(%x, %i, %j, %length, %far2) : (memref<?x3xf64>, index, index, f64, f64) -> (i64, f64)
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

  // Seven cells of the reach along each direction.
  // CHECK:      16352
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %l0 = arith.constant 12.0 : f64
  call @run(%l0, %reach, %none, %no, %n2000) : (f64, f64, index, i1, index) -> ()

  // With the excluded pairs (i, i + 1): 15 of them are within the reach.
  // CHECK-NEXT: 16337
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%l0, %reach, %one, %no, %n2000) : (f64, f64, index, i1, index) -> ()

  // Nine excluded pairs a particle each way: 288 partners a group, more
  // than the memory of a warp holds, so the lists take the excluded pairs
  // from their rows (D106).
  // CHECK-NEXT: 16200
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%l0, %reach, %nine, %no, %n2000) : (f64, f64, index, i1, index) -> ()

  // A position that is not a number: its particle is left out of the
  // lists and the build completes (D107); the 15 pairs of particle 7
  // within the reach are missing.
  // CHECK-NEXT: 16337
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%l0, %reach, %none, %yes, %n2000) : (f64, f64, index, i1, index) -> ()

  // A denser cube: the grid of the candidates is 14 cells a side.
  // CHECK-NEXT: 24389
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %l1 = arith.constant 7.0 : f64
  %r1 = arith.constant 1.0 : f64
  call @run(%l1, %r1, %none, %no, %n2000) : (f64, f64, index, i1, index) -> ()

  // A smaller reach in a smaller cube.
  // CHECK-NEXT: 21979
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %l2 = arith.constant 5.8 : f64
  %r2 = arith.constant 0.8 : f64
  call @run(%l2, %r2, %none, %no, %n2000) : (f64, f64, index, i1, index) -> ()

  // A cell narrower than twice the reach and the extent of a group: the
  // reach 1.9 and a group about 0.8 wide are more than half the cell of
  // 4.0, so some pairs need the image on the other side of the boundary
  // from the one nearest to the center of the group (D115).
  // CHECK-NEXT: 896816
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %l3 = arith.constant 4.0 : f64
  %r3 = arith.constant 1.9 : f64
  call @run(%l3, %r3, %none, %no, %n2000) : (f64, f64, index, i1, index) -> ()
  // Reaches of more than half the cell (#263, D242): a pair then
  // has several images within the reach, each with an entry bit of its own,
  // and a box and the reach span more than the cell, so that an image a
  // cell away on either side of the nearest can be within the reach. 0.55
  // of the edge with 2000 particles; then 320 particles, whose lists fit
  // the buffers of the test, at 0.6, 0.75, 0.9, and 0.975 of the edge.
  // CHECK-NEXT: 1392935
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 50582
  %l4 = arith.constant 5.8 : f64
  %r4 = arith.constant 3.19 : f64
  call @run(%l4, %r4, %none, %no, %n2000) : (f64, f64, index, i1, index) -> ()
  %n320 = arith.constant 320 : index
  // CHECK-NEXT: 46145
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 5503
  %r5 = arith.constant 2.4 : f64
  call @run(%l3, %r5, %none, %no, %n320) : (f64, f64, index, i1, index) -> ()
  // CHECK-NEXT: 90433
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 37377
  %r6 = arith.constant 3.0 : f64
  call @run(%l3, %r6, %none, %no, %n320) : (f64, f64, index, i1, index) -> ()
  // CHECK-NEXT: 154718
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 50266
  %r7 = arith.constant 3.6 : f64
  call @run(%l3, %r7, %one, %no, %n320) : (f64, f64, index, i1, index) -> ()
  // CHECK-NEXT: 198183
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: 51035
  %r8 = arith.constant 3.9 : f64
  call @run(%l3, %r8, %none, %no, %n320) : (f64, f64, index, i1, index) -> ()
  return
}
