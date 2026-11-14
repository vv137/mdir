// The cases of md_alg.bin_members (D[md-alg-analysis]) on the host, after
// host.mlir and helpers.mlir: a counting sort of `n` items into `bins`
// bins (a tenth without a bin), then, for each bin, a visit of its items
// in any order that sums them, takes the exclusive or of three times each,
// and their maximum and minimum (integer combiners), writes the bin into
// the slot of each item, and counts the items in a counter (an addition);
// then a visit of every bin at once, and a visit of each bin in order by
// the item that sums them as f64 and keeps the last (an ordered visit may
// combine floats). md-alg-members.test runs it in order and with the
// unordered visits reversed, and compares.

func.func private @printF64(f64)

func.func @members_case(%n: index, %bins: index, %seed: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : i32
  %zero = arith.constant 0 : i32
  %low = arith.constant -2147483648 : i32
  %high = arith.constant 2147483647 : i32
  %three = arith.constant 3 : i32
  %one = arith.constant 1 : i32
  %fzero = arith.constant 0.0 : f64
  %bins32 = arith.index_cast %bins : index to i32
  %keys_h = func.call @keys(%n, %bins32, %seed, %c10) : (index, i32, i32, i32) -> memref<?xi32>
  %keys = func.call @upload(%keys_h) : (memref<?xi32>) -> !buf
  %n1 = arith.addi %n, %c1 : index
  %b1 = arith.addi %bins, %c1 : index
  %counts = func.call @marked(%b1) : (index) -> !buf
  %start = func.call @marked(%b1) : (index) -> !buf
  %cursor = func.call @marked(%b1) : (index) -> !buf
  %order = func.call @marked(%n1) : (index) -> !buf
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %order order stable : !buf
  func.call @emit_buf(%keys) : (!buf) -> ()
  func.call @emit_buf(%order) : (!buf) -> ()

  // Each bin, in any order.
  %r4 = arith.muli %bins, %c4 : index
  %results = func.call @marked(%r4) : (index) -> !buf
  %slots = func.call @marked(%n) : (index) -> !buf
  %counter = func.call @marked(%c1) : (index) -> !buf
  memref.store %zero, %counter[%c0] : !buf
  scf.for %b = %c0 to %bins step %c1 {
    %next = arith.addi %b, %c1 : index
    %bin = arith.index_cast %b : index to i32
    %sum, %xor, %max, %min = md_alg.bin_members %start[%b, %next]
        items %order visit unordered writes(%slots : !buf)
        adds(%counter : !buf)
        init(%zero, %zero, %low, %high : i32, i32, i32, i32)
        combine ["addi", "xori", "maxsi", "minsi"] : !buf, !buf {
    ^bb0(%item: index):
      %v = arith.index_cast %item : index to i32
      %t = arith.muli %v, %three : i32
      memref.store %bin, %slots[%item] : !buf
      %old = memref.atomic_rmw addi %one, %counter[%c0] : (i32, !buf) -> i32
      md_alg.yield %v, %t, %v, %v : i32, i32, i32, i32
    } -> i32, i32, i32, i32
    %at = arith.muli %b, %c4 : index
    %at1 = arith.addi %at, %c1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %at2 = arith.addi %at, %c2 : index
    %at3 = arith.addi %at, %c3 : index
    memref.store %sum, %results[%at] : !buf
    memref.store %xor, %results[%at1] : !buf
    memref.store %max, %results[%at2] : !buf
    memref.store %min, %results[%at3] : !buf
  }
  func.call @emit_buf(%results) : (!buf) -> ()
  func.call @emit_buf(%slots) : (!buf) -> ()
  func.call @emit_buf(%counter) : (!buf) -> ()

  // Every bin at once.
  %total = md_alg.bin_members %start[%c0, %bins] items %order
      visit unordered init(%zero : i32) combine ["addi"] : !buf, !buf {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    md_alg.yield %v : i32
  } -> i32
  %total64 = arith.extsi %total : i32 to i64
  func.call @printI64(%total64) : (i64) -> ()
  func.call @printNewline() : () -> ()

  // Each bin in order by the item: a sum of floats.
  scf.for %b = %c0 to %bins step %c1 {
    %next = arith.addi %b, %c1 : index
    %f = md_alg.bin_members %start[%b, %next] items %order visit stable
        init(%fzero : f64) combine ["addf"] : !buf, !buf {
    ^bb0(%item: index):
      %v = arith.index_cast %item : index to i32
      %w = arith.sitofp %v : i32 to f64
      %c = arith.constant 0.1 : f64
      %x = arith.mulf %w, %c : f64
      md_alg.yield %x : f64
    } -> f64
    func.call @printF64(%f) : (f64) -> ()
    func.call @printNewline() : () -> ()
  }
  return
}

func.func @main() {
  %c7 = arith.constant 7 : index
  %c20 = arith.constant 20 : index
  %c1000 = arith.constant 1000 : index
  %c1025 = arith.constant 1025 : index
  %c50 = arith.constant 50 : index
  %c3000 = arith.constant 3000 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %s1 = arith.constant 1 : i32
  %s2 = arith.constant 2 : i32
  %s3 = arith.constant 3 : i32
  %s4 = arith.constant 4 : i32
  func.call @members_case(%c20, %c7, %s1) : (index, index, i32) -> ()
  func.call @members_case(%c1025, %c1000, %s2) : (index, index, i32) -> ()
  func.call @members_case(%c3000, %c50, %s3) : (index, index, i32) -> ()
  func.call @members_case(%c0, %c1, %s4) : (index, index, i32) -> ()
  return
}
