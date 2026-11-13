// The cases of the contract test of the stage ops of md_alg
// (D407), after helpers.mlir and a driver of appends
// (append-host.mlir or append-device.mlir), whose prelude (host.mlir or
// device.mlir) defines `!buf`, `!wide`, `@alloc`, `@alloc_wide`,
// `@upload`, and `@download`. md-alg-stages-contract-gpu.test runs both
// and compares their outputs; md-alg-stages-serial.test checks the serial
// one against values written out.
//
// The counting sort is ordered and stable: every buffer it writes, its
// side outputs among them, is printed as it is, to the bit. Every buffer
// has elements past those the op may write, set to -7. The appends are
// ordered within a list, but the numbers of their blocks are not fixed:
// the output is the counts and the counter, to the bit, and three checks
// against the lists that the host forms itself from the items: the blocks
// named, whether every named block holds the entries of its list at its
// ordinal (and every other element its mark), and whether every list is
// whole.

//===----------------------------------------------------------------------===//
// Counting sorts
//===----------------------------------------------------------------------===//

// n items with random keys in [0, bins), -1 (no bin) where a number in
// [0, gap) is 0, sorted four ways:
// A. by a key region of i32 bins that writes the slot 2 i + 1 of its item
//    (three times its key) and adds the items without a bin to a counter;
//    the op clears its counts; the three-launch scan; by insertion.
// B. by a key region of index bins; the counts cleared by the caller
//    (`cleared`); in the order of `by` (descending numbers); the chunked
//    scan; by insertion.
// C. by the keys given, some without a bin; the single-pass scan; by the
//    stable rank.
// D. by the keys given, every key a bin (`every_key`); by insertion.
// E. by the keys given, some without a bin; unordered (no sort within the
//    bins): its bins after a canonical sort.
func.func @sort_by_key_case(%n: index, %bins: index, %seed: i32, %gap: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
  %zero = arith.constant 0 : i32
  %mark = arith.constant -7 : i32
  %bins32 = arith.index_cast %bins : index to i32
  %keys_h = func.call @keys(%n, %bins32, %seed, %gap) : (index, i32, i32, i32) -> memref<?xi32>
  %x = func.call @upload(%keys_h) : (memref<?xi32>) -> !buf
  %grid_n = func.call @blocks_of(%n, %c128) : (index, index) -> index
  %grid_bins = func.call @blocks_of(%bins, %c128) : (index, index) -> index
  %blocks = func.call @blocks_of(%bins, %c1024) : (index, index) -> index
  %sums = func.call @alloc(%blocks) : (index) -> !buf
  %words = arith.addi %blocks, %c1 : index
  %state = func.call @alloc_wide(%words) : (index) -> !wide
  %chunks = func.call @blocks_of(%bins, %c256) : (index, index) -> index
  %grid_chunks = func.call @blocks_of(%chunks, %c128) : (index, index) -> index
  %chunk_sums = func.call @alloc(%chunks) : (index) -> !buf
  %n1 = arith.addi %n, %c1 : index
  %b1 = arith.addi %bins, %c1 : index
  %b2 = arith.addi %bins, %c2 : index
  %n2 = arith.addi %n, %n : index
  %slots_n = arith.addi %n2, %c2 : index

  // A.
  %ka = func.call @marked(%n1) : (index) -> !buf
  %ca = func.call @marked(%b1) : (index) -> !buf
  %sa = func.call @marked(%b2) : (index) -> !buf
  %ua = func.call @marked(%b1) : (index) -> !buf
  %oa = func.call @marked(%n1) : (index) -> !buf
  %slots = func.call @marked(%slots_n) : (index) -> !buf
  %bad_h = func.call @fill(%c3, %mark) : (index, i32) -> memref<?xi32>
  memref.store %zero, %bad_h[%c1] : memref<?xi32>
  %bad = func.call @upload(%bad_h) : (memref<?xi32>) -> !buf
  md_alg.sort_by_key %n bins %bins keys %ka counts %ca start %sa cursor %ua
      into %oa order stable writes(%slots : !buf) adds(%bad : !buf) : !buf
      launch(%grid_n, %c128, %c1)
      scan reduce_then_scan scratch %sums : !buf launch(%blocks, %c256, %c1)
      sort insertion launch(%grid_bins, %c128, %c1) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !buf
    %three = arith.constant 3 : i32
    %t = arith.muli %v, %three : i32
    %r2 = arith.constant 2 : index
    %r1 = arith.constant 1 : index
    %i2 = arith.muli %i, %r2 : index
    %at = arith.addi %i2, %r1 : index
    memref.store %t, %slots[%at] : !buf
    %z = arith.constant 0 : i32
    %none = arith.cmpi slt, %v, %z : i32
    scf.if %none {
      md_alg.no_bin
      %one = arith.constant 1 : i32
      %old = memref.atomic_rmw addi %one, %bad[%r1] : (i32, !buf) -> i32
    } else {
      md_alg.bin %v : i32
    }
  }
  func.call @emit_buf(%ka) : (!buf) -> ()
  func.call @emit_buf(%ca) : (!buf) -> ()
  func.call @emit_buf(%sa) : (!buf) -> ()
  func.call @emit_buf(%ua) : (!buf) -> ()
  func.call @emit_buf(%oa) : (!buf) -> ()
  func.call @emit_buf(%slots) : (!buf) -> ()
  func.call @emit_buf(%bad) : (!buf) -> ()

  // B.
  %kb = func.call @marked(%n1) : (index) -> !buf
  %cb_h = func.call @fill(%b1, %zero) : (index, i32) -> memref<?xi32>
  memref.store %mark, %cb_h[%bins] : memref<?xi32>
  %cb = func.call @upload(%cb_h) : (memref<?xi32>) -> !buf
  %sb = func.call @marked(%b2) : (index) -> !buf
  %ub = func.call @marked(%b1) : (index) -> !buf
  %ob = func.call @marked(%n1) : (index) -> !buf
  %by_h = func.call @fill(%n1, %zero) : (index, i32) -> memref<?xi32>
  scf.for %i = %c0 to %n step %c1 {
    %down = arith.subi %n, %i : index
    %d = arith.subi %down, %c1 : index
    %d32 = arith.index_cast %d : index to i32
    memref.store %d32, %by_h[%i] : memref<?xi32>
  }
  %by = func.call @upload(%by_h) : (memref<?xi32>) -> !buf
  md_alg.sort_by_key %n bins %bins keys %kb counts %cb start %sb cursor %ub
      into %ob order stable by %by : !buf cleared : !buf launch(%grid_n, %c128, %c1)
      scan chunked scratch %chunk_sums : !buf
      launch(%chunks, %c256, %grid_chunks, %c128, %c1)
      sort insertion launch(%grid_bins, %c128, %c1) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !buf
    %z = arith.constant 0 : i32
    %none = arith.cmpi slt, %v, %z : i32
    scf.if %none {
      md_alg.no_bin
    } else {
      %k = arith.index_cast %v : i32 to index
      md_alg.bin %k : index
    }
  }
  func.call @emit_buf(%kb) : (!buf) -> ()
  func.call @emit_buf(%cb) : (!buf) -> ()
  func.call @emit_buf(%sb) : (!buf) -> ()
  func.call @emit_buf(%ub) : (!buf) -> ()
  func.call @emit_buf(%ob) : (!buf) -> ()

  // C.
  %cc = func.call @marked(%b1) : (index) -> !buf
  %sc = func.call @marked(%b2) : (index) -> !buf
  %uc = func.call @marked(%b1) : (index) -> !buf
  %oc = func.call @marked(%n1) : (index) -> !buf
  %scattered = func.call @marked(%n1) : (index) -> !buf
  md_alg.sort_by_key %n bins %bins keys %x counts %cc start %sc cursor %uc
      into %oc order stable : !buf launch(%grid_n, %c128, %c1)
      scan single_pass scratch %state : !wide launch(%blocks, %c256, %c1)
      sort stable_rank buffer %scattered : !buf launch(%grid_n, %c128, %c1)
  func.call @emit_buf(%x) : (!buf) -> ()
  func.call @emit_buf(%cc) : (!buf) -> ()
  func.call @emit_buf(%sc) : (!buf) -> ()
  func.call @emit_buf(%uc) : (!buf) -> ()
  func.call @emit_buf(%oc) : (!buf) -> ()

  // D.
  %all_h = func.call @random(%n, %bins32, %seed) : (index, i32, i32) -> memref<?xi32>
  %all = func.call @upload(%all_h) : (memref<?xi32>) -> !buf
  %cd = func.call @marked(%b1) : (index) -> !buf
  %sd = func.call @marked(%b2) : (index) -> !buf
  %ud = func.call @marked(%b1) : (index) -> !buf
  %od = func.call @marked(%n1) : (index) -> !buf
  md_alg.sort_by_key %n bins %bins keys %all counts %cd start %sd cursor %ud
      into %od order stable every_key : !buf launch(%grid_n, %c128, %c1)
      scan reduce_then_scan scratch %sums : !buf launch(%blocks, %c256, %c1)
      sort insertion launch(%grid_bins, %c128, %c1)
  func.call @emit_buf(%cd) : (!buf) -> ()
  func.call @emit_buf(%sd) : (!buf) -> ()
  func.call @emit_buf(%ud) : (!buf) -> ()
  func.call @emit_buf(%od) : (!buf) -> ()

  // E. By the keys given, some without a bin, unordered: no sort within
  // the bins, whose items are compared after a canonical sort.
  %ce = func.call @marked(%b1) : (index) -> !buf
  %se = func.call @marked(%b2) : (index) -> !buf
  %ue = func.call @marked(%b1) : (index) -> !buf
  %oe = func.call @marked(%n1) : (index) -> !buf
  md_alg.sort_by_key %n bins %bins keys %x counts %ce start %se cursor %ue
      into %oe order unordered : !buf launch(%grid_n, %c128, %c1)
      scan reduce_then_scan scratch %sums : !buf launch(%blocks, %c256, %c1)
  func.call @emit_buf(%ce) : (!buf) -> ()
  func.call @emit_buf(%se) : (!buf) -> ()
  func.call @emit_buf(%ue) : (!buf) -> ()
  %se_h = func.call @download(%se) : (!buf) -> memref<?xi32>
  %oe_h = func.call @download(%oe) : (!buf) -> memref<?xi32>
  func.call @canonical(%oe_h, %se_h, %bins) : (memref<?xi32>, memref<?xi32>, index) -> ()
  func.call @emit(%oe_h) : (memref<?xi32>) -> ()
  return
}

func.func @sorts_by_key() {
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
  %gap = arith.constant 10 : i32
  %none = arith.constant 0 : i32
  %s1 = arith.constant 11 : i32
  %s2 = arith.constant 23 : i32
  %s3 = arith.constant 37 : i32
  %s4 = arith.constant 41 : i32
  %s5 = arith.constant 53 : i32
  %s6 = arith.constant 67 : i32
  %s7 = arith.constant 71 : i32
  %s8 = arith.constant 83 : i32
  // (items, bins): none; one; few items in few bins; about one item a bin
  // past a block of 1,024 counts; 3,000 in 50 bins; all 3,000 in one bin,
  // some without a bin and then none without; as many bins as items, past
  // a block of 128; 5,000 items in 3 bins.
  func.call @sort_by_key_case(%c0, %c1, %s1, %gap) : (index, index, i32, i32) -> ()
  func.call @sort_by_key_case(%c1, %c1, %s2, %gap) : (index, index, i32, i32) -> ()
  func.call @sort_by_key_case(%c20, %c7, %s3, %gap) : (index, index, i32, i32) -> ()
  func.call @sort_by_key_case(%c1025, %c1000, %s4, %gap) : (index, index, i32, i32) -> ()
  func.call @sort_by_key_case(%c3000, %c50, %s5, %gap) : (index, index, i32, i32) -> ()
  func.call @sort_by_key_case(%c3000, %c1, %s6, %gap) : (index, index, i32, i32) -> ()
  func.call @sort_by_key_case(%c3000, %c1, %s6, %none) : (index, index, i32, i32) -> ()
  func.call @sort_by_key_case(%c129, %c129, %s7, %gap) : (index, index, i32, i32) -> ()
  func.call @sort_by_key_case(%c5000, %c3, %s8, %gap) : (index, index, i32, i32) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Appends
//===----------------------------------------------------------------------===//

// `lists` lists of `rounds` rounds of 32 items each: item
// (g rounds + r) 32 + l of list g is kept where keep[item] = 1, with the
// value vals[item]. The appends into blocks of 64 with a capacity of `cap`
// blocks (the driver: append-host.mlir, append-device.mlir), then the
// output: the counts (lists, and one mark), the counter (at place 1 of 3),
// and the checks [named blocks, consistent, whole].
func.func @append_run(%keep_h: memref<?xi32>, %lists: index, %rounds: index,
                      %seed: i32, %cap: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %mark = arith.constant -7 : i32
  %big = arith.constant 1000000 : i32
  %lr = arith.muli %lists, %rounds : index
  %items = arith.muli %lr, %c32 : index
  %salt = arith.constant 101 : i32
  %seed2 = arith.addi %seed, %salt : i32
  %vals_h = func.call @random(%items, %big, %seed2) : (index, i32, i32) -> memref<?xi32>
  %keep = func.call @upload(%keep_h) : (memref<?xi32>) -> !buf
  %vals = func.call @upload(%vals_h) : (memref<?xi32>) -> !buf
  %cap1 = arith.addi %cap, %c1 : index
  %slots = arith.muli %cap1, %c64 : index
  %entries = func.call @marked(%slots) : (index) -> !buf
  %units = func.call @marked(%cap1) : (index) -> !buf
  %ordinals = func.call @marked(%cap1) : (index) -> !buf
  %pool_h = func.call @fill(%c3, %mark) : (index, i32) -> memref<?xi32>
  memref.store %zero, %pool_h[%c1] : memref<?xi32>
  %pool = func.call @upload(%pool_h) : (memref<?xi32>) -> !buf
  %lists1 = arith.addi %lists, %c1 : index
  %counts = func.call @marked(%lists1) : (index) -> !buf
  func.call @append_lists(%keep, %vals, %lists, %rounds, %entries, %units,
                          %ordinals, %pool, %counts, %cap)
      : (!buf, !buf, index, index, !buf, !buf, !buf, !buf, !buf, index) -> ()
  func.call @emit_buf(%counts) : (!buf) -> ()
  func.call @emit_buf(%pool) : (!buf) -> ()

  // The lists, from the items on the host: list g's entries at
  // g rounds 32 + idx, and their number.
  %per = arith.muli %rounds, %c32 : index
  %expected = func.call @fill(%items, %mark) : (index, i32) -> memref<?xi32>
  %expcount = func.call @fill(%lists1, %zero) : (index, i32) -> memref<?xi32>
  scf.for %g = %c0 to %lists step %c1 {
    %base = arith.muli %g, %per : index
    %end = arith.addi %base, %per : index
    %total = scf.for %item = %base to %end step %c1 iter_args(%m = %c0) -> (index) {
      %k = memref.load %keep_h[%item] : memref<?xi32>
      %kept = arith.cmpi eq, %k, %one : i32
      %next = scf.if %kept -> (index) {
        %v = memref.load %vals_h[%item] : memref<?xi32>
        %at = arith.addi %base, %m : index
        memref.store %v, %expected[%at] : memref<?xi32>
        %m1 = arith.addi %m, %c1 : index
        scf.yield %m1 : index
      } else {
        scf.yield %m : index
      }
      scf.yield %next : index
    }
    %total32 = arith.index_cast %total : index to i32
    memref.store %total32, %expcount[%g] : memref<?xi32>
  }

  %entries_h = func.call @download(%entries) : (!buf) -> memref<?xi32>
  %units_h = func.call @download(%units) : (!buf) -> memref<?xi32>
  %ordinals_h = func.call @download(%ordinals) : (!buf) -> memref<?xi32>
  %counts_h = func.call @download(%counts) : (!buf) -> memref<?xi32>
  // The blocks that each list holds named.
  %held = func.call @fill(%lists1, %zero) : (index, i32) -> memref<?xi32>
  %true = arith.constant true
  %named, %consistent = scf.for %b = %c0 to %cap step %c1
      iter_args(%nm = %zero, %ok = %true) -> (i32, i1) {
    %u = memref.load %units_h[%b] : memref<?xi32>
    %is = arith.cmpi ne, %u, %mark : i32
    %first = arith.muli %b, %c64 : index
    %r:2 = scf.if %is -> (i32, i1) {
      %nm1 = arith.addi %nm, %one : i32
      %g = arith.index_cast %u : i32 to index
      %h = memref.load %held[%g] : memref<?xi32>
      %h1 = arith.addi %h, %one : i32
      memref.store %h1, %held[%g] : memref<?xi32>
      %k32 = memref.load %ordinals_h[%b] : memref<?xi32>
      %k = arith.index_cast %k32 : i32 to index
      %ec32 = memref.load %expcount[%g] : memref<?xi32>
      %ec = arith.index_cast %ec32 : i32 to index
      %gbase = arith.muli %g, %per : index
      %kbase = arith.muli %k, %c64 : index
      %ok1 = scf.for %j = %c0 to %c64 step %c1 iter_args(%o = %ok) -> (i1) {
        %idx = arith.addi %kbase, %j : index
        %at = arith.addi %first, %j : index
        %e = memref.load %entries_h[%at] : memref<?xi32>
        %in = arith.cmpi ult, %idx, %ec : index
        %want = scf.if %in -> (i32) {
          %p = arith.addi %gbase, %idx : index
          %w = memref.load %expected[%p] : memref<?xi32>
          scf.yield %w : i32
        } else {
          scf.yield %mark : i32
        }
        %same = arith.cmpi eq, %e, %want : i32
        %o1 = arith.andi %o, %same : i1
        scf.yield %o1 : i1
      }
      scf.yield %nm1, %ok1 : i32, i1
    } else {
      %ok1 = scf.for %j = %c0 to %c64 step %c1 iter_args(%o = %ok) -> (i1) {
        %at = arith.addi %first, %j : index
        %e = memref.load %entries_h[%at] : memref<?xi32>
        %same = arith.cmpi eq, %e, %mark : i32
        %o1 = arith.andi %o, %same : i1
        scf.yield %o1 : i1
      }
      scf.yield %nm, %ok1 : i32, i1
    }
    scf.yield %r#0, %r#1 : i32, i1
  }
  // Nothing at or past the capacity.
  %past0 = arith.muli %cap, %c64 : index
  %past_ok = scf.for %at = %past0 to %slots step %c1 iter_args(%o = %consistent) -> (i1) {
    %e = memref.load %entries_h[%at] : memref<?xi32>
    %same = arith.cmpi eq, %e, %mark : i32
    %o1 = arith.andi %o, %same : i1
    scf.yield %o1 : i1
  }
  %ucap = memref.load %units_h[%cap] : memref<?xi32>
  %ocap = memref.load %ordinals_h[%cap] : memref<?xi32>
  %ucap_ok = arith.cmpi eq, %ucap, %mark : i32
  %ocap_ok = arith.cmpi eq, %ocap, %mark : i32
  %caps_ok = arith.andi %ucap_ok, %ocap_ok : i1
  %all_ok = arith.andi %past_ok, %caps_ok : i1
  // Every list whole: its count, and its blocks all named.
  %whole = scf.for %g = %c0 to %lists step %c1 iter_args(%w = %true) -> (i1) {
    %ec = memref.load %expcount[%g] : memref<?xi32>
    %got = memref.load %counts_h[%g] : memref<?xi32>
    %h = memref.load %held[%g] : memref<?xi32>
    %c63 = arith.constant 63 : i32
    %six = arith.constant 6 : i32
    %ec63 = arith.addi %ec, %c63 : i32
    %need = arith.shrui %ec63, %six : i32
    %a = arith.cmpi eq, %ec, %got : i32
    %b = arith.cmpi eq, %h, %need : i32
    %ab = arith.andi %a, %b : i1
    %w1 = arith.andi %w, %ab : i1
    scf.yield %w1 : i1
  }
  %checks = func.call @fill(%c3, %zero) : (index, i32) -> memref<?xi32>
  memref.store %named, %checks[%c0] : memref<?xi32>
  %ok32 = arith.extui %all_ok : i1 to i32
  %c2 = arith.constant 2 : index
  memref.store %ok32, %checks[%c1] : memref<?xi32>
  %whole32 = arith.extui %whole : i1 to i32
  memref.store %whole32, %checks[%c2] : memref<?xi32>
  func.call @emit(%checks) : (memref<?xi32>) -> ()
  return
}

// The items kept where a random number in [0, mod) is 0.
func.func @kept(%items: index, %mod: i32, %seed: i32) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %r = func.call @random(%items, %mod, %seed) : (index, i32, i32) -> memref<?xi32>
  scf.for %i = %c0 to %items step %c1 {
    %v = memref.load %r[%i] : memref<?xi32>
    %is = arith.cmpi eq, %v, %zero : i32
    %k = arith.select %is, %one, %zero : i32
    memref.store %k, %r[%i] : memref<?xi32>
  }
  return %r : memref<?xi32>
}

func.func @append_case(%lists: index, %rounds: index, %mod: i32, %seed: i32,
                       %cap: index) {
  %c32 = arith.constant 32 : index
  %lr = arith.muli %lists, %rounds : index
  %items = arith.muli %lr, %c32 : index
  %keep = func.call @kept(%items, %mod, %seed) : (index, i32, i32) -> memref<?xi32>
  func.call @append_run(%keep, %lists, %rounds, %seed, %cap)
      : (memref<?xi32>, index, index, i32, index) -> ()
  return
}

func.func @appends() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %c64 = arith.constant 64 : index
  %c200 = arith.constant 200 : index
  %all = arith.constant 1 : i32
  %half = arith.constant 2 : i32
  %third = arith.constant 3 : i32
  %seed = arith.constant 5 : i32
  %seed2 = arith.constant 17 : i32
  %one = arith.constant 1 : i32
  %zero = arith.constant 0 : i32
  // No item kept: no block, a counter of 0.
  %none = func.call @fill(%c64, %zero) : (index, i32) -> memref<?xi32>
  func.call @append_run(%none, %c2, %c1, %seed, %c3)
      : (memref<?xi32>, index, index, i32, index) -> ()
  // One item, lane 5 of the only round: one block.
  %single = func.call @fill(%c64, %zero) : (index, i32) -> memref<?xi32>
  memref.store %one, %single[%c5] : memref<?xi32>
  func.call @append_run(%single, %c2, %c1, %seed, %c3)
      : (memref<?xi32>, index, index, i32, index) -> ()
  // Every item of one list of 20 rounds: 640 entries, 10 blocks, with
  // room for exactly 10, for 9 (one short), and for 200.
  func.call @append_case(%c1, %c20, %all, %seed, %c10) : (index, index, i32, i32, index) -> ()
  func.call @append_case(%c1, %c20, %all, %seed, %c9) : (index, index, i32, i32, index) -> ()
  func.call @append_case(%c1, %c20, %all, %seed, %c200) : (index, index, i32, i32, index) -> ()
  // Half the items of 8 lists of 10 rounds, some blocks begun in a
  // round's middle: room for 200 blocks, and for 3 (far too few).
  func.call @append_case(%c8, %c10, %half, %seed2, %c200) : (index, index, i32, i32, index) -> ()
  func.call @append_case(%c8, %c10, %half, %seed2, %c3) : (index, index, i32, i32, index) -> ()
  // A third of the items of 40 lists of 7 rounds, all running at once on
  // a device; room for 200, and for none.
  func.call @append_case(%c40, %c7, %third, %seed2, %c200) : (index, index, i32, i32, index) -> ()
  func.call @append_case(%c40, %c7, %third, %seed2, %c0) : (index, index, i32, i32, index) -> ()
  return
}

func.func @main() {
  func.call @sorts_by_key() : () -> ()
  func.call @appends() : () -> ()
  return
}
