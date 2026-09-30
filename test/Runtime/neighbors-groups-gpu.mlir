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

// Each case places 2000 particles in a periodic cube from a linear
// congruential generator (the positions of neighbors-matrix-gpu.mlir) and
// prints:
//
//   - the number of pairs that the lists hold (bits of the masks);
//   - the number of pairs within the reach (tested in f64 on the host)
//     that are not in the lists exactly once, or excluded pairs that are;
//   - the number of pairs in the lists farther apart than the reach, with a
//     margin for the rounding of f32;
//   - whether a group lies across two places that are not of one chunk (0).
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

// The squared distance of i and j in the minimum image.
func.func @distance2(%x: memref<?x3xf64>, %i: index, %j: index,
                     %length: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %s = scf.for %k = %c0 to %c3 step %c1 iter_args(%a = %zero) -> (f64) {
    %xi = memref.load %x[%i, %k] : memref<?x3xf64>
    %xj = memref.load %x[%j, %k] : memref<?x3xf64>
    %d = arith.subf %xi, %xj : f64
    %q = arith.divf %d, %length : f64
    %r = math.roundeven %q : f64
    %rl = arith.mulf %r, %length : f64
    %e = arith.subf %d, %rl : f64
    %e2 = arith.mulf %e, %e : f64
    %t = arith.addf %a, %e2 : f64
    scf.yield %t : f64
  }
  return %s : f64
}

// Excluded pairs (2k, 2k + 1), as an incidence structure: a row for each
// particle with its number of pairs, then for each the number of the pair,
// the place of the particle in it, and its two members.
func.func @exclusions(%n: index) -> memref<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %e = memref.alloc(%n, %c5) : memref<?x?xi32>
  %one = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 {
    %half = arith.divui %i, %c2 : index
    %s = arith.remui %i, %c2 : index
    %first = arith.muli %half, %c2 : index
    %second = arith.addi %first, %c1 : index
    %h32 = arith.index_cast %half : index to i32
    %s32 = arith.index_cast %s : index to i32
    %f32 = arith.index_cast %first : index to i32
    %g32 = arith.index_cast %second : index to i32
    memref.store %one, %e[%i, %c0] : memref<?x?xi32>
    memref.store %h32, %e[%i, %c1] : memref<?x?xi32>
    memref.store %s32, %e[%i, %c2] : memref<?x?xi32>
    memref.store %f32, %e[%i, %c3] : memref<?x?xi32>
    memref.store %g32, %e[%i, %c4] : memref<?x?xi32>
  }
  return %e : memref<?x?xi32>
}

func.func @run(%length: f64, %reach: f64, %exclude: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %count = arith.constant 2000 : index
  %x = memref.alloc(%count) : memref<?x3xf64>
  call @fill(%x, %length) : (memref<?x3xf64>, f64) -> ()
  %box = vector.broadcast %length : f64 to vector<3xf64>

  %capacity = arith.muli %count, %c4 : index
  %groups = arith.divui %capacity, %c16 : index
  %width = arith.constant 1024 : index
  %unit_capacity = arith.muli %groups, %c16 : index
  %xd = gpu.alloc (%count) : memref<?x3xf64, 1>
  %orderd = gpu.alloc (%capacity) : memref<?xi32, 1>
  %placed = gpu.alloc (%count) : memref<?xi32, 1>
  %entriesd = gpu.alloc (%groups, %width) : memref<?x?xi32, 1>
  %masksd = gpu.alloc (%groups, %width) : memref<?x?xi32, 1>
  %countsd = gpu.alloc (%groups) : memref<?xi32, 1>
  %unitsd = gpu.alloc (%unit_capacity) : memref<?xi32, 1>
  %sizes = memref.alloc() : memref<3xi32>
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %xd, %x : memref<?x3xf64, 1>, memref<?x3xf64>
  gpu.wait [%t1]
  %excluded_host = call @exclusions(%count) : (index) -> memref<?x?xi32>
  %rows = arith.select %exclude, %count, %c0 : index
  %c5 = arith.constant 5 : index
  %excludedd = gpu.alloc (%rows, %c5) : memref<?x?xi32, 1>
  scf.if %exclude {
    %t2 = gpu.wait async
    %t3 = gpu.memcpy async [%t2] %excludedd, %excluded_host : memref<?x?xi32, 1>, memref<?x?xi32>
    gpu.wait [%t3]
  }
  func.call @mdrt_gpu_build_neighbors_groups(%xd, %box, %reach, %excludedd,
      %orderd, %placed, %entriesd, %masksd, %countsd, %unitsd, %sizes)
      : (memref<?x3xf64, 1>, vector<3xf64>, f64, memref<?x?xi32, 1>,
         memref<?xi32, 1>, memref<?xi32, 1>, memref<?x?xi32, 1>,
         memref<?x?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>,
         memref<3xi32>) -> ()
  %order = memref.alloc(%capacity) : memref<?xi32>
  %entries = memref.alloc(%groups, %width) : memref<?x?xi32>
  %masks = memref.alloc(%groups, %width) : memref<?x?xi32>
  %counts = memref.alloc(%groups) : memref<?xi32>
  %t4 = gpu.wait async
  %t5 = gpu.memcpy async [%t4] %order, %orderd : memref<?xi32>, memref<?xi32, 1>
  %t6 = gpu.memcpy async [%t5] %entries, %entriesd : memref<?x?xi32>, memref<?x?xi32, 1>
  %t7 = gpu.memcpy async [%t6] %masks, %masksd : memref<?x?xi32>, memref<?x?xi32, 1>
  %t8 = gpu.memcpy async [%t7] %counts, %countsd : memref<?xi32>, memref<?xi32, 1>
  gpu.wait [%t8]

  %places32 = memref.load %sizes[%c0] : memref<3xi32>
  %places = arith.index_cast %places32 : i32 to index
  %used_groups = arith.divui %places, %c16 : index

  // How often each pair is in the lists, and the pairs too far apart.
  %seen = memref.alloc(%count, %count) : memref<?x?xi8>
  %z8 = arith.constant 0 : i8
  %one8 = arith.constant 1 : i8
  scf.for %i = %c0 to %count step %c1 {
    scf.for %j = %c0 to %count step %c1 {
      memref.store %z8, %seen[%i, %j] : memref<?x?xi8>
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
  %bits, %too_far = scf.for %g = %c0 to %used_groups step %c1
      iter_args(%b = %zero, %f = %zero) -> (i64, i64) {
    %n32 = memref.load %counts[%g] : memref<?xi32>
    %ne = arith.index_cast %n32 : i32 to index
    %b1, %f1 = scf.for %e = %c0 to %ne step %c1
        iter_args(%bb = %b, %ff = %f) -> (i64, i64) {
      %q32 = memref.load %entries[%g, %e] : memref<?x?xi32>
      %m = memref.load %masks[%g, %e] : memref<?x?xi32>
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
          %d2 = func.call @distance2(%x, %i, %j, %length) : (memref<?x3xf64>, index, index, f64) -> f64
          %out = arith.cmpf ogt, %d2, %far2 : f64
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

  // Every pair within the reach once, but excluded pairs.
  %reach2 = arith.mulf %reach, %reach : f64
  %wrong = scf.for %i = %c0 to %count step %c1 iter_args(%w = %zero) -> (i64) {
    %i1 = arith.addi %i, %c1 : index
    %wi = scf.for %j = %i1 to %count step %c1 iter_args(%ww = %w) -> (i64) {
      %d2 = func.call @distance2(%x, %i, %j, %length) : (memref<?x3xf64>, index, index, f64) -> f64
      %within = arith.cmpf ole, %d2, %reach2 : f64
      %ih = arith.divui %i, %c2 : index
      %jh = arith.divui %j, %c2 : index
      %pair = arith.cmpi eq, %ih, %jh : index
      %is_excluded = arith.andi %pair, %exclude : i1
      %true = arith.constant true
      %kept = arith.xori %is_excluded, %true : i1
      %want = arith.andi %within, %kept : i1
      %want8 = arith.extui %want : i1 to i8
      %got = memref.load %seen[%i, %j] : memref<?x?xi8>
      // A pair beyond the reach but within its rounding may be there.
      %near_edge0 = arith.cmpf ole, %d2, %far2 : f64
      %edge = arith.xori %within, %near_edge0 : i1
      %allowed_extra = arith.andi %edge, %kept : i1
      %differs = arith.cmpi ne, %got, %want8 : i8
      %extra_ok = arith.cmpi eq, %got, %one8 : i8
      %excused = arith.andi %allowed_extra, %extra_ok : i1
      %true2 = arith.constant true
      %not_excused = arith.xori %excused, %true2 : i1
      %bad = arith.andi %differs, %not_excused : i1
      %b = arith.extui %bad : i1 to i64
      %next = arith.addi %ww, %b : i64
      scf.yield %next : i64
    }
    scf.yield %wi : i64
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
  return
}

func.func @main() {
  %reach = arith.constant 1.5 : f64
  %no = arith.constant false
  %yes = arith.constant true

  // Seven cells of the reach along each direction.
  // CHECK:      16352
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %l0 = arith.constant 12.0 : f64
  call @run(%l0, %reach, %no) : (f64, f64, i1) -> ()

  // With the excluded pairs (2k, 2k + 1): 4 of them are within the reach.
  // CHECK-NEXT: 16348
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  call @run(%l0, %reach, %yes) : (f64, f64, i1) -> ()

  // A denser cube: the grid of the candidates is 14 cells a side.
  // CHECK-NEXT: 24389
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %l1 = arith.constant 7.0 : f64
  %r1 = arith.constant 1.0 : f64
  call @run(%l1, %r1, %no) : (f64, f64, i1) -> ()

  // A smaller reach in a smaller cube.
  // CHECK-NEXT: 21979
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  // CHECK-NEXT: {{^0$}}
  %l2 = arith.constant 5.8 : f64
  %r2 = arith.constant 0.8 : f64
  call @run(%l2, %r2, %no) : (f64, f64, i1) -> ()
  return
}
