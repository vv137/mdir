// The emitters of compaction, scans, and counting sorts
// (CompactionEmitter.h, D296) called by name in kernels
// written for the test, expanded by `expand-device-primitives`, and run on
// a device by device-primitives-gpu.test. Each case compares with a result
// written out here (the small ones) or computed on the host in this file
// by the definition (the others, which print 1 where every value agrees).
//
// The expansion leaves no call and no declaration of a primitive:
// RUN: mdir-opt %s --expand-device-primitives | FileCheck %s --check-prefix=EXPAND
// EXPAND-NOT: mdir.primitive
// EXPAND: gpu.ballot
// EXPAND: math.ctpop
// EXPAND: gpu.shuffle up
// EXPAND: llvm.atomicrmw add
// EXPAND-NOT: mdir.primitive
//
// Warp compaction: the threads t of 2 warps with 7 t mod 5 < 2 packed in
// the order of the lanes, at the front of their warp's 32 places (-1 where
// none), and the count of each warp.
// CHECK:      ( 0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 33, 35, 38, 40, 43, 45, 48, 50, 53, 55, 58, 60, 63, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 )
// CHECK-NEXT: ( 13, 13 )
// Warp scans of t mod 7 + 1: over the 32 lanes, and over segments of 16
// lanes with the lane compared as an unsigned number.
// CHECK-NEXT: ( 1, 3, 6, 10, 15, 21, 28, 29, 31, 34, 38, 43, 49, 56, 57, 59, 62, 66, 71, 77, 84, 85, 87, 90, 94, 99, 105, 112, 113, 115, 118, 122 )
// CHECK-NEXT: ( 1, 3, 6, 10, 15, 21, 28, 29, 31, 34, 38, 43, 49, 56, 57, 59, 3, 7, 12, 18, 25, 26, 28, 31, 35, 40, 46, 53, 54, 56, 59, 63 )
// The block scan of 256 threads: the sums before each thread, the total.
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// The device scan of 5000 counts, with and without the cursor, and the
// chunked scan of 1000.
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// The single-pass scan with 129 tiles and with no counts.
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// The counting sort of 3000 items by 50 keys, a tenth of them with none:
// the scatter and the insertion sort; the scatter and the stable rank;
// every item a key, sorted by the reverse of the item.
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NOT: {{.}}

!h = memref<?xi32>
!d = memref<?xi32, 1>
!wide = memref<?xi64, 1>
!wg = memref<9xi32, #gpu.address_space<workgroup>>

func.func private @mdir.primitive.warp_lanes_below(i32) -> i32
func.func private @mdir.primitive.warp_ballot(i1) -> i32
func.func private @mdir.primitive.warp_rank(i32, i32) -> i32
func.func private @mdir.primitive.warp_count(i32) -> i32
func.func private @mdir.primitive.warp_scan(i32, i32, i32, i32) -> i32
func.func private @mdir.primitive.block_scan(i32, !wg, index) -> (i32, i32)
func.func private @mdir.primitive.device_scan(!d, !d, index, !d, index, index, index)
func.func private @mdir.primitive.device_scan_cursor(!d, !d, index, !d, index, index, index, !d)
func.func private @mdir.primitive.single_pass_scan(!d, !d, index, !wide, index, index, index)
func.func private @mdir.primitive.single_pass_scan_cursor(!d, !d, index, !wide, index, index, index, !d)
func.func private @mdir.primitive.chunked_scan(!d, !d, !d, !d, index, index, index, index, index, index)
func.func private @mdir.primitive.cursor_scatter(!d, !d, !d, index, index, index, index)
func.func private @mdir.primitive.sort_bins(!d, !d, index, index, index, index)
func.func private @mdir.primitive.sort_bins_by(!d, !d, index, index, index, index, !d)
func.func private @mdir.primitive.rank_in_bins(!d, !d, !d, !d, index, index, index, index)

func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printComma()
func.func private @printOpen()
func.func private @printClose()

//===----------------------------------------------------------------------===//
// Helpers on the host
//===----------------------------------------------------------------------===//

func.func @to_device(%h: !h) -> !d {
  %c0 = arith.constant 0 : index
  %n = memref.dim %h, %c0 : !h
  %d = gpu.alloc (%n) : !d
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %d, %h : !d, !h
  gpu.wait [%t1]
  return %d : !d
}

func.func @to_host(%d: !d) -> !h {
  %c0 = arith.constant 0 : index
  %n = memref.dim %d, %c0 : !d
  %h = memref.alloc (%n) : !h
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %h, %d : !h, !d
  gpu.wait [%t1]
  return %h : !h
}

// A buffer of `n` copies of `v` on the device.
func.func @filled(%n: index, %v: i32) -> !d {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %h = memref.alloc (%n) : !h
  scf.for %i = %c0 to %n step %c1 {
    memref.store %v, %h[%i] : !h
  }
  %d = func.call @to_device(%h) : (!h) -> !d
  return %d : !d
}

func.func @print_row(%h: !h) {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %h, %c0 : !h
  call @printOpen() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    %first = arith.cmpi eq, %i, %c0 : index
    %more = arith.xori %first, %true : i1
    scf.if %more {
      func.call @printComma() : () -> ()
    }
    %v = memref.load %h[%i] : !h
    %w = arith.extsi %v : i32 to i64
    func.call @printI64(%w) : (i64) -> ()
  }
  call @printClose() : () -> ()
  call @printNewline() : () -> ()
  return
}

// Prints 1 if the first `n` values of `d` (on the device) are those of
// `h`, 0 otherwise.
func.func @same(%d: !d, %h: !h, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %got = func.call @to_host(%d) : (!d) -> !h
  %all = scf.for %i = %c0 to %n step %c1 iter_args(%s = %true) -> (i1) {
    %a = memref.load %got[%i] : !h
    %b = memref.load %h[%i] : !h
    %e = arith.cmpi eq, %a, %b : i32
    %u = arith.andi %s, %e : i1
    scf.yield %u : i1
  }
  %flag = arith.extui %all : i1 to i64
  call @printI64(%flag) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}

// The exclusive sums of the `n` values of `h`, the sum of all at n.
func.func @prefix(%h: !h, %n: index) -> !h {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %n1 = arith.addi %n, %c1 : index
  %out = memref.alloc (%n1) : !h
  %total = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> (i32) {
    memref.store %s, %out[%i] : !h
    %v = memref.load %h[%i] : !h
    %t = arith.addi %s, %v : i32
    scf.yield %t : i32
  }
  memref.store %total, %out[%n] : !h
  return %out : !h
}

//===----------------------------------------------------------------------===//
// The cases
//===----------------------------------------------------------------------===//

func.func @warps() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %none = arith.constant -1 : i32
  %zero = arith.constant 0 : i32
  %out = func.call @filled(%c64, %none) : (index, i32) -> !d
  %counts = func.call @filled(%c2, %none) : (index, i32) -> !d
  %scan32 = func.call @filled(%c32, %none) : (index, i32) -> !d
  %scan16 = func.call @filled(%c32, %none) : (index, i32) -> !d
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c64, %sy = %c1, %sz = %c1) {
    %i0 = arith.constant 0 : i32
    %i2 = arith.constant 2 : i32
    %i5 = arith.constant 5 : i32
    %i7 = arith.constant 7 : i32
    %i15 = arith.constant 15 : i32
    %i32 = arith.constant 32 : i32
    %t = arith.index_cast %tx : index to i32
    %lane = arith.remui %t, %i32 : i32
    %warp = arith.divui %tx, %c32 : index
    %t7 = arith.muli %t, %i7 : i32
    %m = arith.remui %t7, %i5 : i32
    %keep = arith.cmpi ult, %m, %i2 : i32
    %below = func.call @mdir.primitive.warp_lanes_below(%lane) : (i32) -> i32
    %bits = func.call @mdir.primitive.warp_ballot(%keep) : (i1) -> i32
    %rank = func.call @mdir.primitive.warp_rank(%bits, %below) : (i32, i32) -> i32
    %count = func.call @mdir.primitive.warp_count(%bits) : (i32) -> i32
    scf.if %keep {
      %base = arith.muli %warp, %c32 : index
      %r = arith.index_cast %rank : i32 to index
      %at = arith.addi %base, %r : index
      memref.store %t, %out[%at] : !d
    }
    %first = arith.cmpi eq, %lane, %i0 : i32
    scf.if %first {
      memref.store %count, %counts[%warp] : !d
    }
    // The scans, in the first warp: t mod 7 + 1.
    %i1 = arith.constant 1 : i32
    %t_7 = arith.remui %t, %i7 : i32
    %v = arith.addi %t_7, %i1 : i32
    %u16 = arith.andi %lane, %i15 : i32
    %s32 = func.call @mdir.primitive.warp_scan(%v, %lane, %i0, %i32) : (i32, i32, i32, i32) -> i32
    %s16 = func.call @mdir.primitive.warp_scan(%v, %u16, %i0, %i32) {lanes = 16 : i64, unsigned} : (i32, i32, i32, i32) -> i32
    %in_first = arith.cmpi ult, %tx, %c32 : index
    scf.if %in_first {
      memref.store %s32, %scan32[%tx] : !d
      memref.store %s16, %scan16[%tx] : !d
    }
    gpu.terminator
  }
  %h_out = func.call @to_host(%out) : (!d) -> !h
  call @print_row(%h_out) : (!h) -> ()
  %h_counts = func.call @to_host(%counts) : (!d) -> !h
  call @print_row(%h_counts) : (!h) -> ()
  %h32 = func.call @to_host(%scan32) : (!d) -> !h
  call @print_row(%h32) : (!h) -> ()
  %h16 = func.call @to_host(%scan16) : (!d) -> !h
  call @print_row(%h16) : (!h) -> ()
  return
}

// The block scan of (13 t) mod 11 over 256 threads.
func.func @block() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %none = arith.constant -1 : i32
  %values = memref.alloc (%c256) : !h
  scf.for %i = %c0 to %c256 step %c1 {
    %w = arith.index_cast %i : index to i32
    %c13 = arith.constant 13 : i32
    %c11 = arith.constant 11 : i32
    %p = arith.muli %w, %c13 : i32
    %v = arith.remui %p, %c11 : i32
    memref.store %v, %values[%i] : !h
  }
  %expect = func.call @prefix(%values, %c256) : (!h, index) -> !h
  %before = func.call @filled(%c256, %none) : (index, i32) -> !d
  %totals = func.call @filled(%c256, %none) : (index, i32) -> !d
  %dv = func.call @to_device(%values) : (!h) -> !d
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c256, %sy = %c1, %sz = %c1)
             workgroup(%wg : !wg) {
    %v = memref.load %dv[%tx] : !d
    %b, %total = func.call @mdir.primitive.block_scan(%v, %wg, %tx) : (i32, !wg, index) -> (i32, i32)
    memref.store %b, %before[%tx] : !d
    memref.store %total, %totals[%tx] : !d
    gpu.terminator
  }
  call @same(%before, %expect, %c256) : (!d, !h, index) -> ()
  // The total in every thread.
  %all = memref.load %expect[%c256] : !h
  %h_all = memref.alloc (%c256) : !h
  scf.for %i = %c0 to %c256 step %c1 {
    memref.store %all, %h_all[%i] : !h
  }
  call @same(%totals, %h_all, %c256) : (!d, !h, index) -> ()
  return
}

// The device scan of (7 i) mod 5 over 5000 counts, with and without the
// cursor, and the chunked scan of the first 1000.
func.func @scans() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 5000 : index
  %n1 = arith.constant 5001 : index
  %m = arith.constant 1000 : index
  %m1 = arith.constant 1001 : index
  %none = arith.constant -1 : i32
  %counts = memref.alloc (%n) : !h
  scf.for %i = %c0 to %n step %c1 {
    %w = arith.index_cast %i : index to i32
    %c7 = arith.constant 7 : i32
    %c5 = arith.constant 5 : i32
    %p = arith.muli %w, %c7 : i32
    %v = arith.remui %p, %c5 : i32
    memref.store %v, %counts[%i] : !h
  }
  %expect = func.call @prefix(%counts, %n) : (!h, index) -> !h
  %expect_m = func.call @prefix(%counts, %m) : (!h, index) -> !h
  %dc = func.call @to_device(%counts) : (!h) -> !d
  // 5 blocks of 1024 counts.
  %blocks = arith.constant 5 : index
  %threads = arith.constant 256 : index
  %sums = gpu.alloc (%blocks) : !d
  %start = func.call @filled(%n1, %none) : (index, i32) -> !d
  %cursor = func.call @filled(%n, %none) : (index, i32) -> !d
  func.call @mdir.primitive.device_scan_cursor(%dc, %start, %n, %sums, %blocks, %threads, %c1, %cursor) : (!d, !d, index, !d, index, index, index, !d) -> ()
  call @same(%start, %expect, %n1) : (!d, !h, index) -> ()
  call @same(%cursor, %expect, %n) : (!d, !h, index) -> ()
  %start2 = func.call @filled(%n1, %none) : (index, i32) -> !d
  func.call @mdir.primitive.device_scan(%dc, %start2, %n, %sums, %blocks, %threads, %c1) : (!d, !d, index, !d, index, index, index) -> ()
  call @same(%start2, %expect, %n1) : (!d, !h, index) -> ()
  %state_size = arith.addi %blocks, %c1 : index
  %state = gpu.alloc (%state_size) : !wide
  %start4 = func.call @filled(%n1, %none) : (index, i32) -> !d
  %cursor4 = func.call @filled(%n, %none) : (index, i32) -> !d
  func.call @mdir.primitive.single_pass_scan_cursor(%dc, %start4, %n, %state, %blocks, %threads, %c1, %cursor4) : (!d, !d, index, !wide, index, index, index, !d) -> ()
  call @same(%start4, %expect, %n1) : (!d, !h, index) -> ()
  call @same(%cursor4, %expect, %n) : (!d, !h, index) -> ()
  %start5 = func.call @filled(%n1, %none) : (index, i32) -> !d
  func.call @mdir.primitive.single_pass_scan(%dc, %start5, %n, %state, %blocks, %threads, %c1) : (!d, !d, index, !wide, index, index, index) -> ()
  call @same(%start5, %expect, %n1) : (!d, !h, index) -> ()
  // 4 chunks of 256 counts, a thread a chunk in one block of 128.
  %chunks = arith.constant 4 : index
  %chunk = arith.constant 256 : index
  %block = arith.constant 128 : index
  %chunk_sums = gpu.alloc (%chunks) : !d
  %start3 = func.call @filled(%m1, %none) : (index, i32) -> !d
  %cursor3 = func.call @filled(%m, %none) : (index, i32) -> !d
  func.call @mdir.primitive.chunked_scan(%dc, %start3, %cursor3, %chunk_sums, %m, %chunks, %chunk, %c1, %block, %c1) : (!d, !d, !d, !d, index, index, index, index, index, index) -> ()
  call @same(%start3, %expect_m, %m1) : (!d, !h, index) -> ()
  return
}

func.func @scans_long_and_empty() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 131083 : index
  %n1 = arith.constant 131084 : index
  %blocks = arith.constant 129 : index
  %state_size = arith.constant 130 : index
  %threads = arith.constant 256 : index
  %zero = arith.constant 0 : i32
  %minus_one = arith.constant -1 : i32
  %h = memref.alloc (%n) : !h
  scf.for %i = %c0 to %n step %c1 {
    %v = arith.index_cast %i : index to i32
    %c7 = arith.constant 7 : i32
    %c5 = arith.constant 5 : i32
    %a = arith.muli %v, %c7 : i32
    %count = arith.remui %a, %c5 : i32
    memref.store %count, %h[%i] : !h
  }
  %expected = func.call @prefix(%h, %n) : (!h, index) -> !h
  %counts = func.call @to_device(%h) : (!h) -> !d
  %state = gpu.alloc (%state_size) : !wide
  %start = func.call @filled(%n1, %minus_one) : (index, i32) -> !d
  func.call @mdir.primitive.single_pass_scan(%counts, %start, %n, %state, %blocks, %threads, %c1) : (!d, !d, index, !wide, index, index, index) -> ()
  call @same(%start, %expected, %n1) : (!d, !h, index) -> ()

  %empty_host = memref.alloc (%c1) : !h
  memref.store %zero, %empty_host[%c0] : !h
  %empty_device = func.call @to_device(%empty_host) : (!h) -> !d
  %empty_start = func.call @filled(%c1, %minus_one) : (index, i32) -> !d
  %empty_state_size = arith.constant 2 : index
  %empty_state = gpu.alloc (%empty_state_size) : !wide
  func.call @mdir.primitive.single_pass_scan(%empty_device, %empty_start, %c0, %empty_state, %c1, %threads, %c1) : (!d, !d, index, !wide, index, index, index) -> ()
  call @same(%empty_start, %empty_host, %c1) : (!d, !h, index) -> ()
  return
}

// The counting sort of 3000 items by the keys (37 i) mod 50, where every
// tenth item has none (-1), and then with every item a key.
func.func @sorts() {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 3000 : index
  %bins = arith.constant 50 : index
  %bins1 = arith.constant 51 : index
  %block = arith.constant 128 : index
  %grid_n = arith.constant 24 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %none = arith.constant -1 : i32
  %keys = memref.alloc (%n) : !h
  %all_keys = memref.alloc (%n) : !h
  %by = memref.alloc (%n) : !h
  %counts = memref.alloc (%bins) : !h
  %all_counts = memref.alloc (%bins) : !h
  scf.for %k = %c0 to %bins step %c1 {
    memref.store %zero, %counts[%k] : !h
    memref.store %zero, %all_counts[%k] : !h
  }
  // The keys, and their histogram.
  scf.for %i = %c0 to %n step %c1 {
    %w = arith.index_cast %i : index to i32
    %c37 = arith.constant 37 : i32
    %c50 = arith.constant 50 : i32
    %c10 = arith.constant 10 : i32
    %c9 = arith.constant 9 : i32
    %p = arith.muli %w, %c37 : i32
    %k32 = arith.remui %p, %c50 : i32
    %r = arith.remui %w, %c10 : i32
    %skip = arith.cmpi eq, %r, %c9 : i32
    %key = arith.select %skip, %none, %k32 : i32
    memref.store %key, %keys[%i] : !h
    memref.store %k32, %all_keys[%i] : !h
    %n32 = arith.index_cast %n : index to i32
    %rev0 = arith.subi %n32, %w : i32
    %rev = arith.subi %rev0, %one : i32
    memref.store %rev, %by[%i] : !h
    %k = arith.index_cast %k32 : i32 to index
    %a = memref.load %all_counts[%k] : !h
    %a1 = arith.addi %a, %one : i32
    memref.store %a1, %all_counts[%k] : !h
    %has = arith.xori %skip, %true : i1
    scf.if %has {
      %c = memref.load %counts[%k] : !h
      %c1_ = arith.addi %c, %one : i32
      memref.store %c1_, %counts[%k] : !h
    }
  }
  // The expected orders, by the definition: the items of each key in
  // ascending order (keyed), and in descending order (every item a key).
  %expect = memref.alloc (%n) : !h
  %expect_all = memref.alloc (%n) : !h
  %placed = scf.for %k = %c0 to %bins step %c1 iter_args(%at = %c0) -> (index) {
    %k32 = arith.index_cast %k : index to i32
    %next = scf.for %i = %c0 to %n step %c1 iter_args(%p = %at) -> (index) {
      %key = memref.load %keys[%i] : !h
      %same = arith.cmpi eq, %key, %k32 : i32
      %q = scf.if %same -> (index) {
        %w = arith.index_cast %i : index to i32
        memref.store %w, %expect[%p] : !h
        %p1 = arith.addi %p, %c1 : index
        scf.yield %p1 : index
      } else {
        scf.yield %p : index
      }
      scf.yield %q : index
    }
    scf.yield %next : index
  }
  %placed_all = scf.for %k = %c0 to %bins step %c1 iter_args(%at = %c0) -> (index) {
    %k32 = arith.index_cast %k : index to i32
    %next = scf.for %j = %c0 to %n step %c1 iter_args(%p = %at) -> (index) {
      %nm1 = arith.subi %n, %c1 : index
      %i = arith.subi %nm1, %j : index
      %key = memref.load %all_keys[%i] : !h
      %same = arith.cmpi eq, %key, %k32 : i32
      %q = scf.if %same -> (index) {
        %w = arith.index_cast %i : index to i32
        memref.store %w, %expect_all[%p] : !h
        %p1 = arith.addi %p, %c1 : index
        scf.yield %p1 : index
      } else {
        scf.yield %p : index
      }
      scf.yield %q : index
    }
    scf.yield %next : index
  }
  %dk = func.call @to_device(%keys) : (!h) -> !d
  %dka = func.call @to_device(%all_keys) : (!h) -> !d
  %dby = func.call @to_device(%by) : (!h) -> !d
  %dc = func.call @to_device(%counts) : (!h) -> !d
  %dca = func.call @to_device(%all_counts) : (!h) -> !d
  %threads = arith.constant 256 : index
  %sums = gpu.alloc (%c1) : !d

  // The device scan, the scatter, and the insertion sort.
  %start = func.call @filled(%bins1, %none) : (index, i32) -> !d
  %cursor = func.call @filled(%bins, %none) : (index, i32) -> !d
  %sorted = func.call @filled(%n, %none) : (index, i32) -> !d
  func.call @mdir.primitive.device_scan_cursor(%dc, %start, %bins, %sums, %c1, %threads, %c1, %cursor) : (!d, !d, index, !d, index, index, index, !d) -> ()
  func.call @mdir.primitive.cursor_scatter(%dk, %cursor, %sorted, %n, %grid_n, %block, %c1) : (!d, !d, !d, index, index, index, index) -> ()
  func.call @mdir.primitive.sort_bins(%start, %sorted, %bins, %c1, %block, %c1) : (!d, !d, index, index, index, index) -> ()
  call @same(%sorted, %expect, %placed) : (!d, !h, index) -> ()

  // The chunked scan, the scatter, and the stable rank.
  %start2 = func.call @filled(%bins1, %none) : (index, i32) -> !d
  %cursor2 = func.call @filled(%bins, %none) : (index, i32) -> !d
  %sorted2 = func.call @filled(%n, %none) : (index, i32) -> !d
  %ranked = func.call @filled(%n, %none) : (index, i32) -> !d
  %chunk = arith.constant 256 : index
  %chunk_sums = gpu.alloc (%c1) : !d
  func.call @mdir.primitive.chunked_scan(%dc, %start2, %cursor2, %chunk_sums, %bins, %c1, %chunk, %c1, %block, %c1) : (!d, !d, !d, !d, index, index, index, index, index, index) -> ()
  func.call @mdir.primitive.cursor_scatter(%dk, %cursor2, %sorted2, %n, %grid_n, %block, %c1) : (!d, !d, !d, index, index, index, index) -> ()
  func.call @mdir.primitive.rank_in_bins(%sorted2, %dk, %start2, %ranked, %bins, %grid_n, %block, %c1) : (!d, !d, !d, !d, index, index, index, index) -> ()
  call @same(%ranked, %expect, %placed) : (!d, !h, index) -> ()

  // Every item a key, each bin sorted by the reverse of the item.
  %start3 = func.call @filled(%bins1, %none) : (index, i32) -> !d
  %cursor3 = func.call @filled(%bins, %none) : (index, i32) -> !d
  %sorted3 = func.call @filled(%n, %none) : (index, i32) -> !d
  func.call @mdir.primitive.device_scan_cursor(%dca, %start3, %bins, %sums, %c1, %threads, %c1, %cursor3) : (!d, !d, index, !d, index, index, index, !d) -> ()
  func.call @mdir.primitive.cursor_scatter(%dka, %cursor3, %sorted3, %n, %grid_n, %block, %c1) {every_key} : (!d, !d, !d, index, index, index, index) -> ()
  func.call @mdir.primitive.sort_bins_by(%start3, %sorted3, %bins, %c1, %block, %c1, %dby) : (!d, !d, index, index, index, index, !d) -> ()
  call @same(%sorted3, %expect_all, %placed_all) : (!d, !h, index) -> ()
  return
}

func.func @main() {
  call @warps() : () -> ()
  call @block() : () -> ()
  call @scans() : () -> ()
  call @scans_long_and_empty() : () -> ()
  call @sorts() : () -> ()
  return
}
