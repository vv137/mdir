// The cases of the contract test of md_alg (D[md-alg]): each collection
// op on random and boundary inputs. A prelude defines `!buf`, `!wide`,
// `@alloc`, `@alloc_wide`, `@upload`, and `@download`: in the memory of
// the host for the serial run (host.mlir) or of a device for the run
// through the emitters (device.mlir). md-alg-contract-gpu.test runs both
// and compares their outputs; md-alg-serial.test checks the serial one
// against values written out.
//
// The output is, for each buffer, its length and a hash of all its
// elements, and the elements themselves where there are at most 40. Every
// buffer that an op writes has elements past the ones it may write, set to
// -7, so that a write out of place shows. The scatter's order within a bin
// is not fixed (its contract is unordered): its bins are sorted on the
// host before they are printed, and its cursor and the elements outside
// the bins are printed as they are.

func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printComma()
func.func private @printOpen()
func.func private @printClose()

//===----------------------------------------------------------------------===//
// Helpers on the host
//===----------------------------------------------------------------------===//

// max(1, ceil(n / per)).
func.func @blocks_of(%n: index, %per: index) -> index {
  %c1 = arith.constant 1 : index
  %last = arith.subi %per, %c1 : index
  %padded = arith.addi %n, %last : index
  %q = arith.divui %padded, %per : index
  %r = arith.maxui %q, %c1 : index
  return %r : index
}

func.func @fill(%n: index, %v: i32) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %h = memref.alloc (%n) : memref<?xi32>
  scf.for %i = %c0 to %n step %c1 {
    memref.store %v, %h[%i] : memref<?xi32>
  }
  return %h : memref<?xi32>
}

// `n` values in [0, m) from a linear congruential generator of `seed`, in a
// buffer of n + 1 (at least one element; the last is 0).
func.func @random(%n: index, %m: i32, %seed: i32) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %a = arith.constant 1103515245 : i32
  %b = arith.constant 12345 : i32
  %c8 = arith.constant 8 : i32
  %size = arith.addi %n, %c1 : index
  %h = func.call @fill(%size, %zero) : (index, i32) -> memref<?xi32>
  %end = scf.for %i = %c0 to %n step %c1 iter_args(%x = %seed) -> (i32) {
    %ax = arith.muli %x, %a : i32
    %y = arith.addi %ax, %b : i32
    %hi = arith.shrui %y, %c8 : i32
    %v = arith.remui %hi, %m : i32
    memref.store %v, %h[%i] : memref<?xi32>
    scf.yield %y : i32
  }
  return %h : memref<?xi32>
}

// The keys of `n` items in [0, bins), with -1 (no bin) where a random
// number in [0, gap) is 0 (none for a gap of 0).
func.func @keys(%n: index, %bins: i32, %seed: i32, %gap: i32) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %none = arith.constant -1 : i32
  %keys = func.call @random(%n, %bins, %seed) : (index, i32, i32) -> memref<?xi32>
  %with = arith.cmpi ne, %gap, %zero : i32
  scf.if %with {
    %salt = arith.constant 7919 : i32
    %seed2 = arith.addi %seed, %salt : i32
    %holes = func.call @random(%n, %gap, %seed2) : (index, i32, i32) -> memref<?xi32>
    scf.for %i = %c0 to %n step %c1 {
      %r = memref.load %holes[%i] : memref<?xi32>
      %hole = arith.cmpi eq, %r, %zero : i32
      scf.if %hole {
        memref.store %none, %keys[%i] : memref<?xi32>
      }
    }
  }
  return %keys : memref<?xi32>
}

// The count of each of the `bins` keys among `n`, in a buffer of bins + 1.
func.func @histogram(%keys: memref<?xi32>, %n: index, %bins: index) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %size = arith.addi %bins, %c1 : index
  %counts = func.call @fill(%size, %zero) : (index, i32) -> memref<?xi32>
  scf.for %i = %c0 to %n step %c1 {
    %k = memref.load %keys[%i] : memref<?xi32>
    %has = arith.cmpi sge, %k, %zero : i32
    scf.if %has {
      %at = arith.index_cast %k : i32 to index
      %c = memref.load %counts[%at] : memref<?xi32>
      %c1_ = arith.addi %c, %one : i32
      memref.store %c1_, %counts[%at] : memref<?xi32>
    }
  }
  return %counts : memref<?xi32>
}

// The exclusive sums of the first `n` counts, n + 1 of them.
func.func @prefix(%counts: memref<?xi32>, %n: index) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %size = arith.addi %n, %c1 : index
  %start = func.call @fill(%size, %zero) : (index, i32) -> memref<?xi32>
  %total = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> (i32) {
    memref.store %s, %start[%i] : memref<?xi32>
    %v = memref.load %counts[%i] : memref<?xi32>
    %t = arith.addi %s, %v : i32
    scf.yield %t : i32
  }
  memref.store %total, %start[%n] : memref<?xi32>
  return %start : memref<?xi32>
}

// The items of each of the `bins` bins of `start` in `items`, ascending,
// by insertion on the host: the canonical form of a scatter's output.
func.func @canonical(%items: memref<?xi32>, %start: memref<?xi32>, %bins: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %b = %c0 to %bins step %c1 {
    %b1 = arith.addi %b, %c1 : index
    %s32 = memref.load %start[%b] : memref<?xi32>
    %e32 = memref.load %start[%b1] : memref<?xi32>
    %s = arith.index_cast %s32 : i32 to index
    %e = arith.index_cast %e32 : i32 to index
    scf.for %p = %s to %e step %c1 {
      scf.for %q0 = %s to %p step %c1 {
        // q from p - 1 down to s: a bubble of the item at p.
        %d = arith.subi %q0, %s : index
        %pm1 = arith.subi %p, %c1 : index
        %q = arith.subi %pm1, %d : index
        %q1 = arith.addi %q, %c1 : index
        %l = memref.load %items[%q] : memref<?xi32>
        %r = memref.load %items[%q1] : memref<?xi32>
        %swap = arith.cmpi sgt, %l, %r : i32
        scf.if %swap {
          memref.store %r, %items[%q] : memref<?xi32>
          memref.store %l, %items[%q1] : memref<?xi32>
        }
      }
    }
  }
  return
}

// The length of `h`, a hash of its elements, and, for at most 40, the
// elements.
func.func @emit(%h: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c40 = arith.constant 40 : index
  %true = arith.constant true
  %mul = arith.constant 1000003 : i64
  %seed = arith.constant 14695981039 : i64
  %n = memref.dim %h, %c0 : memref<?xi32>
  %hash = scf.for %i = %c0 to %n step %c1 iter_args(%x = %seed) -> (i64) {
    %v = memref.load %h[%i] : memref<?xi32>
    %w = arith.extui %v : i32 to i64
    %m = arith.muli %x, %mul : i64
    %y = arith.xori %m, %w : i64
    scf.yield %y : i64
  }
  %n64 = arith.index_cast %n : index to i64
  func.call @printOpen() : () -> ()
  func.call @printI64(%n64) : (i64) -> ()
  func.call @printComma() : () -> ()
  func.call @printI64(%hash) : (i64) -> ()
  func.call @printClose() : () -> ()
  %small = arith.cmpi ule, %n, %c40 : index
  scf.if %small {
    func.call @printOpen() : () -> ()
    scf.for %i = %c0 to %n step %c1 {
      %first = arith.cmpi eq, %i, %c0 : index
      %more = arith.xori %first, %true : i1
      scf.if %more {
        func.call @printComma() : () -> ()
      }
      %v = memref.load %h[%i] : memref<?xi32>
      %w = arith.extsi %v : i32 to i64
      func.call @printI64(%w) : (i64) -> ()
    }
    func.call @printClose() : () -> ()
  }
  func.call @printNewline() : () -> ()
  return
}

func.func @emit_buf(%b: !buf) {
  %h = func.call @download(%b) : (!buf) -> memref<?xi32>
  func.call @emit(%h) : (memref<?xi32>) -> ()
  return
}

// A buffer of `n` copies of -7.
func.func @marked(%n: index) -> !buf {
  %mark = arith.constant -7 : i32
  %h = func.call @fill(%n, %mark) : (index, i32) -> memref<?xi32>
  %b = func.call @upload(%h) : (memref<?xi32>) -> !buf
  return %b : !buf
}

//===----------------------------------------------------------------------===//
// Scans
//===----------------------------------------------------------------------===//

// Each strategy, with and without the cursor, on the n counts of `h`:
// start (n + 1 and one more) and cursor (n and one more).
func.func @scan_all(%h: memref<?xi32>, %n: index) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
  %counts = func.call @upload(%h) : (memref<?xi32>) -> !buf
  %n1 = arith.addi %n, %c1 : index
  %n2 = arith.addi %n, %c2 : index
  %blocks = func.call @blocks_of(%n, %c1024) : (index, index) -> index
  %sums = func.call @alloc(%blocks) : (index) -> !buf
  %words = arith.addi %blocks, %c1 : index
  %state = func.call @alloc_wide(%words) : (index) -> !wide

  %s1 = func.call @marked(%n2) : (index) -> !buf
  %u1 = func.call @marked(%n1) : (index) -> !buf
  md_alg.scan %counts[%n] into %s1 cursor %u1
      strategy reduce_then_scan scratch %sums : !buf
      launch(%blocks, %c256, %c1) : !buf, !buf, !buf
  func.call @emit_buf(%s1) : (!buf) -> ()
  func.call @emit_buf(%u1) : (!buf) -> ()

  %s2 = func.call @marked(%n2) : (index) -> !buf
  md_alg.scan %counts[%n] into %s2
      strategy reduce_then_scan scratch %sums : !buf
      launch(%blocks, %c256, %c1) : !buf, !buf
  func.call @emit_buf(%s2) : (!buf) -> ()

  %s3 = func.call @marked(%n2) : (index) -> !buf
  %u3 = func.call @marked(%n1) : (index) -> !buf
  md_alg.scan %counts[%n] into %s3 cursor %u3
      strategy single_pass scratch %state : !wide
      launch(%blocks, %c256, %c1) : !buf, !buf, !buf
  func.call @emit_buf(%s3) : (!buf) -> ()
  func.call @emit_buf(%u3) : (!buf) -> ()

  %s4 = func.call @marked(%n2) : (index) -> !buf
  md_alg.scan %counts[%n] into %s4
      strategy single_pass scratch %state : !wide
      launch(%blocks, %c256, %c1) : !buf, !buf
  func.call @emit_buf(%s4) : (!buf) -> ()

  %chunks = func.call @blocks_of(%n, %c256) : (index, index) -> index
  %grid = func.call @blocks_of(%chunks, %c128) : (index, index) -> index
  %chunk_sums = func.call @alloc(%chunks) : (index) -> !buf
  %s5 = func.call @marked(%n2) : (index) -> !buf
  %u5 = func.call @marked(%n1) : (index) -> !buf
  md_alg.scan %counts[%n] into %s5 cursor %u5
      strategy chunked scratch %chunk_sums : !buf
      launch(%chunks, %c256, %grid, %c128, %c1) : !buf, !buf, !buf
  func.call @emit_buf(%s5) : (!buf) -> ()
  func.call @emit_buf(%u5) : (!buf) -> ()
  return
}

func.func @scans() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %eight = arith.constant 8 : i32
  %big = arith.constant 1048576 : i32
  // The counts 3, 1, 4, 1, 5.
  %h = func.call @fill(%c5, %eight) : (index, i32) -> memref<?xi32>
  %v3 = arith.constant 3 : i32
  %v1 = arith.constant 1 : i32
  %v4 = arith.constant 4 : i32
  %v5 = arith.constant 5 : i32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  memref.store %v3, %h[%c0] : memref<?xi32>
  memref.store %v1, %h[%c1] : memref<?xi32>
  memref.store %v4, %h[%c2] : memref<?xi32>
  memref.store %v1, %h[%c3] : memref<?xi32>
  memref.store %v5, %h[%c4] : memref<?xi32>
  func.call @scan_all(%h, %c5) : (memref<?xi32>, index) -> ()
  // No counts, one count, and sizes about a block of 1,024 counts, of a
  // chunk of 256, and of many blocks; 131,083 counts make 129 tiles.
  func.call @scan_random(%c0, %eight) : (index, i32) -> ()
  func.call @scan_random(%c1, %eight) : (index, i32) -> ()
  %c255 = arith.constant 255 : index
  func.call @scan_random(%c255, %eight) : (index, i32) -> ()
  %c257 = arith.constant 257 : index
  func.call @scan_random(%c257, %eight) : (index, i32) -> ()
  %c1023 = arith.constant 1023 : index
  func.call @scan_random(%c1023, %eight) : (index, i32) -> ()
  %c1024 = arith.constant 1024 : index
  func.call @scan_random(%c1024, %eight) : (index, i32) -> ()
  %c1025 = arith.constant 1025 : index
  func.call @scan_random(%c1025, %eight) : (index, i32) -> ()
  %c5000 = arith.constant 5000 : index
  func.call @scan_random(%c5000, %eight) : (index, i32) -> ()
  %c131083 = arith.constant 131083 : index
  func.call @scan_random(%c131083, %eight) : (index, i32) -> ()
  // Counts whose sum passes 2^31: the sums wrap.
  func.call @scan_random(%c5000, %big) : (index, i32) -> ()
  return
}

// The scans of n random counts in [0, m).
func.func @scan_random(%n: index, %m: i32) {
  %seed = arith.index_cast %n : index to i32
  %r = func.call @random(%n, %m, %seed) : (index, i32, i32) -> memref<?xi32>
  func.call @scan_all(%r, %n) : (memref<?xi32>, index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Scatters and sorts within bins
//===----------------------------------------------------------------------===//

// n items with random keys in [0, bins) (-1 where a number in [0, gap) is
// 0): their counting sort by the scan, the scatter, and the stable rank
// into another buffer or the insertion sort in place; then the scatter of
// keys without gaps with `every_key`.
func.func @counting_sort(%n: index, %bins: index, %seed: i32, %gap: i32) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
  %bins32 = arith.index_cast %bins : index to i32
  %keys_h = func.call @keys(%n, %bins32, %seed, %gap) : (index, i32, i32, i32) -> memref<?xi32>
  %counts_h = func.call @histogram(%keys_h, %n, %bins) : (memref<?xi32>, index, index) -> memref<?xi32>
  %keys = func.call @upload(%keys_h) : (memref<?xi32>) -> !buf
  %counts = func.call @upload(%counts_h) : (memref<?xi32>) -> !buf
  %bins1 = arith.addi %bins, %c1 : index
  %n1 = arith.addi %n, %c1 : index
  %blocks = func.call @blocks_of(%bins, %c1024) : (index, index) -> index
  %sums = func.call @alloc(%blocks) : (index) -> !buf
  %grid_n = func.call @blocks_of(%n, %c128) : (index, index) -> index
  %grid_bins = func.call @blocks_of(%bins, %c128) : (index, index) -> index

  %start = func.call @marked(%bins1) : (index) -> !buf
  %cursor = func.call @marked(%bins1) : (index) -> !buf
  md_alg.scan %counts[%bins] into %start cursor %cursor
      strategy reduce_then_scan scratch %sums : !buf
      launch(%blocks, %c256, %c1) : !buf, !buf, !buf
  %out = func.call @marked(%n1) : (index) -> !buf
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out
      launch(%grid_n, %c128, %c1) : !buf, !buf, !buf
  // The scatter: its cursor, then its items, each bin canonical.
  func.call @emit_buf(%cursor) : (!buf) -> ()
  %start_h = func.call @download(%start) : (!buf) -> memref<?xi32>
  %out_h = func.call @download(%out) : (!buf) -> memref<?xi32>
  func.call @canonical(%out_h, %start_h, %bins) : (memref<?xi32>, memref<?xi32>, index) -> ()
  func.call @emit(%out_h) : (memref<?xi32>) -> ()
  // The stable rank into another buffer, from the scatter's order.
  %ranked = func.call @marked(%n1) : (index) -> !buf
  md_alg.sort_bins %start[%bins] items %out into %ranked keys %keys
      strategy stable_rank launch(%grid_n, %c128, %c1)
      : !buf, !buf into !buf keys !buf
  func.call @emit_buf(%ranked) : (!buf) -> ()
  // The insertion sort in place.
  md_alg.sort_bins %start[%bins] items %out
      strategy insertion launch(%grid_bins, %c128, %c1) : !buf, !buf
  func.call @emit_buf(%out) : (!buf) -> ()

  // Every key a bin.
  %all_keys_h = func.call @random(%n, %bins32, %seed) : (index, i32, i32) -> memref<?xi32>
  %all_counts_h = func.call @histogram(%all_keys_h, %n, %bins) : (memref<?xi32>, index, index) -> memref<?xi32>
  %all_keys = func.call @upload(%all_keys_h) : (memref<?xi32>) -> !buf
  %all_counts = func.call @upload(%all_counts_h) : (memref<?xi32>) -> !buf
  %start2 = func.call @marked(%bins1) : (index) -> !buf
  %cursor2 = func.call @marked(%bins1) : (index) -> !buf
  md_alg.scan %all_counts[%bins] into %start2 cursor %cursor2
      strategy reduce_then_scan scratch %sums : !buf
      launch(%blocks, %c256, %c1) : !buf, !buf, !buf
  %out2 = func.call @marked(%n1) : (index) -> !buf
  md_alg.scatter_to_bins %all_keys[%n] cursor %cursor2 into %out2
      launch(%grid_n, %c128, %c1) {every_key} : !buf, !buf, !buf
  func.call @emit_buf(%cursor2) : (!buf) -> ()
  %start2_h = func.call @download(%start2) : (!buf) -> memref<?xi32>
  %out2_h = func.call @download(%out2) : (!buf) -> memref<?xi32>
  func.call @canonical(%out2_h, %start2_h, %bins) : (memref<?xi32>, memref<?xi32>, index) -> ()
  func.call @emit(%out2_h) : (memref<?xi32>) -> ()
  return
}

// The insertion sort of bins of random items with ties, in place, by the
// items and by `by`, whose many equal values show whether the sort is
// stable: the items are given in a fixed order.
func.func @stable(%n: index, %bins: index, %seed: i32) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bins32 = arith.index_cast %bins : index to i32
  %n32 = arith.index_cast %n : index to i32
  %four = arith.constant 4 : i32
  %keys_h = func.call @random(%n, %bins32, %seed) : (index, i32, i32) -> memref<?xi32>
  %counts_h = func.call @histogram(%keys_h, %n, %bins) : (memref<?xi32>, index, index) -> memref<?xi32>
  %start_h = func.call @prefix(%counts_h, %bins) : (memref<?xi32>, index) -> memref<?xi32>
  %salt = arith.constant 31 : i32
  %seed2 = arith.addi %seed, %salt : i32
  %items_h = func.call @random(%n, %n32, %seed2) : (index, i32, i32) -> memref<?xi32>
  %seed3 = arith.addi %seed2, %salt : i32
  // The order of item v is by[v], in [0, 4): ties everywhere.
  %size = arith.addi %n, %c1 : index
  %by_h = func.call @random(%size, %four, %seed3) : (index, i32, i32) -> memref<?xi32>
  %start = func.call @upload(%start_h) : (memref<?xi32>) -> !buf
  %by = func.call @upload(%by_h) : (memref<?xi32>) -> !buf
  %grid = func.call @blocks_of(%bins, %c128) : (index, index) -> index
  %items = func.call @upload(%items_h) : (memref<?xi32>) -> !buf
  md_alg.sort_bins %start[%bins] items %items
      strategy insertion launch(%grid, %c128, %c1) : !buf, !buf
  func.call @emit_buf(%items) : (!buf) -> ()
  %items2 = func.call @upload(%items_h) : (memref<?xi32>) -> !buf
  md_alg.sort_bins %start[%bins] items %items2 by %by
      strategy insertion launch(%grid, %c128, %c1) : !buf, !buf by !buf
  func.call @emit_buf(%items2) : (!buf) -> ()
  return
}

func.func @sorts() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // (items, bins): none; one; few items in many bins; 3,000 in 50 bins;
  // all 3,000 in one bin (the largest bin); counts not a multiple of the
  // block of 128.
  %c7 = arith.constant 7 : index
  %c20 = arith.constant 20 : index
  %c50 = arith.constant 50 : index
  %c129 = arith.constant 129 : index
  %c1000 = arith.constant 1000 : index
  %c1025 = arith.constant 1025 : index
  %c3000 = arith.constant 3000 : index
  func.call @sort_case(%c0, %c1) : (index, index) -> ()
  func.call @sort_case(%c1, %c1) : (index, index) -> ()
  func.call @sort_case(%c20, %c7) : (index, index) -> ()
  func.call @sort_case(%c1025, %c1000) : (index, index) -> ()
  func.call @sort_case(%c3000, %c50) : (index, index) -> ()
  func.call @sort_case(%c3000, %c1) : (index, index) -> ()
  func.call @sort_case(%c129, %c129) : (index, index) -> ()
  return
}

func.func @sort_case(%n: index, %bins: index) {
  %ten = arith.constant 10 : i32
  %n32 = arith.index_cast %n : index to i32
  %b32 = arith.index_cast %bins : index to i32
  %s = arith.addi %n32, %b32 : i32
  func.call @counting_sort(%n, %bins, %s, %ten) : (index, index, i32, i32) -> ()
  func.call @stable(%n, %bins, %s) : (index, index, i32) -> ()
  return
}

func.func @main() {
  func.call @scans() : () -> ()
  func.call @sorts() : () -> ()
  return
}
