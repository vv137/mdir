// The visits of md-alg-members-contract-gpu.test on a device, for the
// SIMT lowering (D409), which has not run on a device before: a thread a
// bin, in blocks of 128, concurrently (the counter and the slots written
// by threads at once), then every bin at once by one thread.

func.func @visit_bins(%start: !buf, %order: !buf, %bins: index, %results: !buf,
    %slots: !buf, %counter: !buf, %floats: !buf, %total: !buf) {
  %cb1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %grid = func.call @blocks_of(%bins, %c128) : (index, index) -> index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid, %gy = %cb1, %gz = %cb1)
             threads(%tx, %ty, %tz) in (%sx = %c128, %sy = %cb1, %sz = %cb1) {
    %first = arith.muli %bx, %c128 : index
    %b = arith.addi %first, %tx : index
    %inside = arith.cmpi ult, %b, %bins : index
    scf.if %inside {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c32_64 = arith.constant 32 : i64
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %three = arith.constant 3 : i32
    %low = arith.constant -2147483648 : i32
    %high = arith.constant 2147483647 : i32
    %fzero = arith.constant 0.0 : f64
    %tenth = arith.constant 0.1 : f64

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
    %at2 = arith.addi %at, %c2 : index
    %at3 = arith.addi %at, %c3 : index
    memref.store %sum, %results[%at] : !buf
    memref.store %xor, %results[%at1] : !buf
    memref.store %max, %results[%at2] : !buf
    memref.store %min, %results[%at3] : !buf
    %f = md_alg.bin_members %start[%b, %next] items %order visit stable
        init(%fzero : f64) combine ["addf"] : !buf, !buf {
    ^bb0(%item: index):
      %v = arith.index_cast %item : index to i32
      %w = arith.sitofp %v : i32 to f64
      %x = arith.mulf %w, %tenth : f64
      md_alg.yield %x : f64
    } -> f64
    %bits = arith.bitcast %f : f64 to i64
    %lo = arith.trunci %bits : i64 to i32
    %hi64 = arith.shrui %bits, %c32_64 : i64
    %hi = arith.trunci %hi64 : i64 to i32
    %fa = arith.muli %b, %c2 : index
    %fb = arith.addi %fa, %c1 : index
    memref.store %lo, %floats[%fa] : !buf
    memref.store %hi, %floats[%fb] : !buf
    }
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %cb1, %gy = %cb1, %gz = %cb1)
             threads(%tx, %ty, %tz) in (%sx = %cb1, %sy = %cb1, %sz = %cb1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c32_64 = arith.constant 32 : i64
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %three = arith.constant 3 : i32
    %low = arith.constant -2147483648 : i32
    %high = arith.constant 2147483647 : i32
    %fzero = arith.constant 0.0 : f64
    %tenth = arith.constant 0.1 : f64

    %all = md_alg.bin_members %start[%c0, %bins] items %order
        visit unordered init(%zero : i32) combine ["addi"] : !buf, !buf {
    ^bb0(%item: index):
      %v = arith.index_cast %item : index to i32
      md_alg.yield %v : i32
    } -> i32
    memref.store %all, %total[%c0] : !buf
    gpu.terminator
  }
  return
}
