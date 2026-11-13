// The helpers of the contract tests of md_alg (D406, D[md-alg-stages]),
// on the host: the printers, buffers filled or random, the histogram, the
// exclusive sums, the canonical form of a scatter, and the output of a
// buffer (its length, a hash of its elements, and the elements where there
// are at most 40). A prelude (host.mlir or device.mlir) defines `!buf`,
// `@upload`, and `@download` before it.

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

