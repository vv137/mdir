// The cases of md-alg-members-contract-gpu.test (D409), after a prelude
// (host.mlir or device.mlir), helpers.mlir, and the visits of the host or
// of a device: a counting sort of `n` items into `bins` bins (a tenth
// without a bin, or none), then @visit_bins, whose outputs are printed:
// for each bin the sum, the exclusive or of three times each item, the
// maximum, and the minimum of its items (an unordered visit, integer
// combiners); the bin in the slot of each item; the count of the items
// visited; the bits of each bin's sum of a tenth of its items, in order by
// the item (a stable visit, floats); and the sum of every item at once.

func.func @members_contract_case(%n: index, %bins: index, %seed: i32,
                                 %gap: i32) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %zero = arith.constant 0 : i32
  %bins32 = arith.index_cast %bins : index to i32
  %keys_h = func.call @keys(%n, %bins32, %seed, %gap) : (index, i32, i32, i32) -> memref<?xi32>
  %keys = func.call @upload(%keys_h) : (memref<?xi32>) -> !buf
  %n1 = arith.addi %n, %c1 : index
  %b1 = arith.addi %bins, %c1 : index
  %counts = func.call @marked(%b1) : (index) -> !buf
  %start = func.call @marked(%b1) : (index) -> !buf
  %cursor = func.call @marked(%b1) : (index) -> !buf
  %order = func.call @marked(%n1) : (index) -> !buf
  %grid_n = func.call @blocks_of(%n, %c128) : (index, index) -> index
  %grid_bins = func.call @blocks_of(%bins, %c128) : (index, index) -> index
  %blocks = func.call @blocks_of(%bins, %c256) : (index, index) -> index
  %blocks1 = arith.addi %blocks, %c1 : index
  %sums = func.call @alloc(%blocks1) : (index) -> !buf
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %order order stable : !buf
      launch(%grid_n, %c128, %c1)
      scan reduce_then_scan scratch %sums : !buf launch(%blocks, %c256, %c1)
      sort insertion launch(%grid_bins, %c128, %c1)
  func.call @emit_buf(%order) : (!buf) -> ()

  %r4 = arith.muli %bins, %c4 : index
  %r2 = arith.muli %bins, %c2 : index
  %results = func.call @marked(%r4) : (index) -> !buf
  %slots = func.call @marked(%n1) : (index) -> !buf
  %floats = func.call @marked(%r2) : (index) -> !buf
  %total = func.call @marked(%c2) : (index) -> !buf
  %zeros = func.call @fill(%c1, %zero) : (index, i32) -> memref<?xi32>
  %counter0 = func.call @upload(%zeros) : (memref<?xi32>) -> !buf
  func.call @visit_bins(%start, %order, %bins, %results, %slots, %counter0,
                        %floats, %total)
      : (!buf, !buf, index, !buf, !buf, !buf, !buf, !buf) -> ()
  func.call @emit_buf(%results) : (!buf) -> ()
  func.call @emit_buf(%slots) : (!buf) -> ()
  func.call @emit_buf(%counter0) : (!buf) -> ()
  func.call @emit_buf(%floats) : (!buf) -> ()
  func.call @emit_buf(%total) : (!buf) -> ()
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %c20 = arith.constant 20 : index
  %c50 = arith.constant 50 : index
  %c129 = arith.constant 129 : index
  %c1000 = arith.constant 1000 : index
  %c1025 = arith.constant 1025 : index
  %c3000 = arith.constant 3000 : index
  %c5000 = arith.constant 5000 : index
  %c4000 = arith.constant 4000 : index
  %c40000 = arith.constant 40000 : index
  %gap = arith.constant 10 : i32
  %none = arith.constant 0 : i32
  %s1 = arith.constant 1 : i32
  %s2 = arith.constant 2 : i32
  %s3 = arith.constant 3 : i32
  %s4 = arith.constant 4 : i32
  %s5 = arith.constant 5 : i32
  %s6 = arith.constant 6 : i32
  %s7 = arith.constant 7 : i32
  %s8 = arith.constant 8 : i32
  func.call @members_contract_case(%c0, %c1, %s1, %gap) : (index, index, i32, i32) -> ()
  func.call @members_contract_case(%c1, %c1, %s2, %none) : (index, index, i32, i32) -> ()
  func.call @members_contract_case(%c20, %c7, %s1, %gap) : (index, index, i32, i32) -> ()
  func.call @members_contract_case(%c129, %c129, %s4, %gap) : (index, index, i32, i32) -> ()
  func.call @members_contract_case(%c1025, %c1000, %s5, %gap) : (index, index, i32, i32) -> ()
  func.call @members_contract_case(%c3000, %c50, %s6, %gap) : (index, index, i32, i32) -> ()
  func.call @members_contract_case(%c3000, %c1, %s7, %none) : (index, index, i32, i32) -> ()
  func.call @members_contract_case(%c5000, %c3, %s8, %gap) : (index, index, i32, i32) -> ()
  func.call @members_contract_case(%c40000, %c4000, %s3, %gap) : (index, index, i32, i32) -> ()
  return
}
