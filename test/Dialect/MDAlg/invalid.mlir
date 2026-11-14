// The refusals of the verifiers of md_alg (D406): one case for each
// rule, in place of the hand checks of the named device primitives.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics

!d = memref<?xi32, 1>

// Every buffer of a collection op is a 1-D memref of i32.
func.func @buffer_type(%counts: memref<?xf32, 1>, %start: !d, %n: index) {
  // expected-error @+1 {{operand #0 must be 1D memref of 32-bit signless integer values}}
  md_alg.scan %counts[%n] into %start : memref<?xf32, 1>, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @scan_together(%counts: !d, %start: !d, %sums: !d, %n: index) {
  // expected-error @+1 {{takes its strategy, its scratch, and its launch together}}
  md_alg.scan %counts[%n] into %start strategy reduce_then_scan scratch %sums : !d
      : !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @single_pass_i64(%counts: !d, %start: !d, %sums: !d, %n: index) {
  // expected-error @+1 {{with the strategy 'single_pass' takes a scratch of i64, not 'i32'}}
  md_alg.scan %counts[%n] into %start strategy single_pass scratch %sums : !d
      launch(%n, %n, %n) : !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @blocks_i32(%counts: !d, %start: !d, %state: memref<?xi64, 1>,
                      %n: index) {
  // expected-error @+1 {{with the strategy 'reduce_then_scan' takes a scratch of i32, not 'i64'}}
  md_alg.scan %counts[%n] into %start strategy reduce_then_scan
      scratch %state : memref<?xi64, 1> launch(%n, %n, %n) : !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @scan_launch(%counts: !d, %start: !d, %sums: !d, %n: index) {
  // expected-error @+1 {{takes 3 launch values (blocks, threads, one), not 2}}
  md_alg.scan %counts[%n] into %start strategy reduce_then_scan scratch %sums : !d
      launch(%n, %n) : !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @chunked_launch(%counts: !d, %start: !d, %cursor: !d, %sums: !d,
                          %n: index) {
  // expected-error @+1 {{takes 5 launch values (chunks, chunk, grid, block, one), not 3}}
  md_alg.scan %counts[%n] into %start cursor %cursor strategy chunked
      scratch %sums : !d launch(%n, %n, %n) : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @chunked_cursor(%counts: !d, %start: !d, %sums: !d, %n: index) {
  // expected-error @+1 {{with the strategy 'chunked' requires a cursor, which it writes}}
  md_alg.scan %counts[%n] into %start strategy chunked scratch %sums : !d
      launch(%n, %n, %n, %n, %n) : !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @scatter_launch(%keys: !d, %cursor: !d, %out: !d, %n: index) {
  // expected-error @+1 {{takes 3 launch values (grid, block, one), not 2}}
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out launch(%n, %n)
      : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @scatter_into_keys(%keys: !d, %cursor: !d, %n: index) {
  // expected-error @+1 {{writes its items into its keys or its cursor, which it reads}}
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %keys : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @scatter_into_cursor(%keys: !d, %cursor: !d, %n: index) {
  // expected-error @+1 {{writes its items into its keys or its cursor, which it reads}}
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %cursor : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @sort_together(%start: !d, %items: !d, %n: index) {
  // expected-error @+1 {{takes its strategy and its launch together}}
  md_alg.sort_bins %start[%n] items %items strategy insertion : !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @insertion_in_place(%start: !d, %items: !d, %out: !d, %n: index) {
  // expected-error @+1 {{with the strategy 'insertion' sorts in place; it takes no 'into'}}
  md_alg.sort_bins %start[%n] items %items into %out strategy insertion
      launch(%n, %n, %n) : !d, !d into !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @rank_keys(%start: !d, %items: !d, %out: !d, %n: index) {
  // expected-error @+1 {{with the strategy 'stable_rank' requires 'into' and the keys of the items}}
  md_alg.sort_bins %start[%n] items %items into %out strategy stable_rank
      launch(%n, %n, %n) : !d, !d into !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @rank_by(%start: !d, %items: !d, %out: !d, %keys: !d, %by: !d,
                   %n: index) {
  // expected-error @+1 {{with the strategy 'stable_rank' orders by the items; it takes no 'by'}}
  md_alg.sort_bins %start[%n] items %items by %by into %out keys %keys
      strategy stable_rank launch(%n, %n, %n) : !d, !d by !d into !d keys !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @keys_into(%start: !d, %items: !d, %keys: !d, %n: index) {
  // expected-error @+1 {{takes the keys of the items only with 'into'}}
  md_alg.sort_bins %start[%n] items %items keys %keys : !d, !d keys !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @into_items(%start: !d, %items: !d, %n: index) {
  // expected-error @+1 {{sorts into its items, its offsets, or its order, which it reads}}
  md_alg.sort_bins %start[%n] items %items into %items : !d, !d into !d
  return
}

// -----

func.func @signed_index(%a: index, %b: index, %c: index) -> index {
  // expected-error @+1 {{compares signed only on i32}}
  %w = md_alg.wrap %a, %b, %c signed : index
  return %w : index
}

// -----

func.func @wrap_types(%a: i32, %b: index, %c: i32) -> i32 {
  // expected-error @+1 {{all of {start, offset, size, result} have same type}}
  %w = "md_alg.wrap"(%a, %b, %c) : (i32, index, i32) -> i32
  return %w : i32
}

// -----

func.func @read_coordinates(%grid: memref<?xf32, 1>, %base: i32,
                            %z: index) -> f32 {
  // expected-error @+1 {{all of {base, z} have same type}}
  %p = "md_alg.grid_read"(%grid, %base, %z) : (memref<?xf32, 1>, i32, index) -> f32
  return %p : f32
}

// -----

func.func @read_result(%grid: memref<?xf32, 1>, %base: i32, %z: i32) -> f64 {
  // expected-error @+1 {{the result is an element of the grid}}
  %p = "md_alg.grid_read"(%grid, %base, %z) : (memref<?xf32, 1>, i32, i32) -> f64
  return %p : f64
}

// -----

func.func @read_grid(%grid: memref<?xi32, 1>, %base: i32, %z: i32) -> i32 {
  // expected-error @+1 {{operand #0 must be 1D memref of 32-bit float or 64-bit float values}}
  %p = "md_alg.grid_read"(%grid, %base, %z) : (memref<?xi32, 1>, i32, i32) -> i32
  return %p : i32
}

// -----

!d = memref<?xi32, 1>
func.func @sort_launch(%start: !d, %items: !d, %n: index) {
  // expected-error @+1 {{takes 3 launch values (grid, block, one), not 4}}
  md_alg.sort_bins %start[%n] items %items strategy insertion
      launch(%n, %n, %n, %n) : !d, !d
  return
}

// The stage ops (D407): one case for each rule.

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A sort's execution group is whole or absent.
func.func @stage_0(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes its launch, its scan's strategy, scratch, and launch, and its sort's strategy and launch together}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Three launch values for the key and the scatter.
func.func @stage_1(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes 3 launch values (grid, block, one), not 2}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The scratch of a single-pass scan is i64.
func.func @stage_2(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{with the scan 'single_pass' takes a scratch of i64, not 'i32'}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan single_pass scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A chunked scan takes five launch values.
func.func @stage_3(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes 5 launch values (chunks, chunk, grid, block, one), not 3}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan chunked scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The other scans take three.
func.func @stage_4(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes 3 launch values (blocks, threads, one), not 2}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The sort takes three launch values.
func.func @stage_5(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes 3 launch values (grid, block, one), not 1}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The stable rank takes the buffer of its scatter.
func.func @stage_6(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes the buffer of its scatter with the sort 'stable_rank' alone}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort stable_rank launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The insertion sort takes none.
func.func @stage_7(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes the buffer of its scatter with the sort 'stable_rank' alone}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion buffer %by : !d launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The stable rank orders by the items.
func.func @stage_8(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{with the sort 'stable_rank' orders by the items; it takes no 'by'}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable by %by : !d : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort stable_rank buffer %x : !d launch(%g, %b, %one)
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// `every_key` only with keys given.
func.func @stage_9(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes 'every_key' only with keys given; with a region the bins say it}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable every_key : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Its own buffers are distinct.
func.func @stage_10(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes one buffer for two of its keys, counts, start, cursor, order, and scatter}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %k order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one)
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Nor are two of them views of one buffer: a cast of the keys as the order (#874).
func.func @stage_10a(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  %view = memref.cast %k : !d to !d
  // expected-error @+1 {{takes one buffer for two of its keys, counts, start, cursor, order, and scatter}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %view order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one)
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// It does not sort into its order 'by'.
func.func @stage_11(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{sorts into its order 'by', which it reads}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable by %o : !d : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one)
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Nor by a view of the order (#874).
func.func @stage_11a(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  %view = memref.reinterpret_cast %o to offset: [0], sizes: [%n], strides: [1] : !d to memref<?xi32, strided<[1]>, 1>
  // expected-error @+1 {{sorts into its order 'by', which it reads}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable by %view : memref<?xi32, strided<[1]>, 1> : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one)
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A side output is none of its own buffers.
func.func @stage_12(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{declares as a side output one of its own buffers or its 'by', or another side output, or a view of one}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable writes(%c : !d) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A side output is written or added to, not both.
func.func @stage_13(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{declares a buffer both written and added to}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable writes(%x : !d) adds(%x : !d) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A side output is no view of its own buffers: a subview of the keys (#871).
func.func @stage_13a(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  %view = memref.subview %k[0] [4] [1] : !d to memref<4xi32, strided<[1]>, 1>
  // expected-error @+1 {{declares as a side output one of its own buffers or its 'by', or another side output, or a view of one}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable writes(%view : memref<4xi32, strided<[1]>, 1>) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Nor a cast of the order, which it fills (#871).
func.func @stage_13b(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  %view = memref.cast %o : !d to !d
  // expected-error @+1 {{declares as a side output one of its own buffers or its 'by', or another side output, or a view of one}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable writes(%view : !d) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Nor a view of another side output (#871).
func.func @stage_13c(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  %view = memref.memory_space_cast %x : !d to memref<?xi32>
  // expected-error @+1 {{declares as a side output one of its own buffers or its 'by', or another side output, or a view of one}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable writes(%x : !d) adds(%view : memref<?xi32>) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The key region takes the item.
func.func @stage_14(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{takes a key region of one index argument, the item}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: i32):
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// One bin on every path: not none on a branch.
func.func @stage_15(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{runs exactly one 'md_alg.bin' or 'md_alg.no_bin' on every path through its key region, in the branches of 'scf.if' alone}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %t = arith.constant true
    scf.if %t {
      md_alg.no_bin
    }
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Not two.
func.func @stage_16(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{runs exactly one 'md_alg.bin' or 'md_alg.no_bin' on every path through its key region, in the branches of 'scf.if' alone}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    md_alg.no_bin
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Not in a loop.
func.func @stage_17(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %j = %c0 to %c1 step %c1 {
    // expected-error @+1 {{stands in the key region of an 'md_alg.sort_by_key', directly or in the branches of 'scf.if'}}
      md_alg.no_bin
    }
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The key region does not use the op's own buffers.
func.func @stage_18(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    // expected-error @+1 {{uses a buffer of the 'md_alg.sort_by_key' of its key region}}
    %v = memref.load %c[%i] : !d
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Nor through a view made outside it: a subview of the counts (#874).
func.func @stage_18a(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  %view = memref.subview %c[0] [4] [1] : !d to memref<4xi32, strided<[1]>, 1>
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{uses a buffer of the 'md_alg.sort_by_key' of its key region}}
    %v = memref.load %view[%c0] : memref<4xi32, strided<[1]>, 1>
    md_alg.bin %v : i32
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// It writes only declared side outputs.
func.func @stage_19(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %z = arith.constant 0 : i32
    // expected-error @+1 {{writes a buffer that the key region does not declare in 'writes'}}
    memref.store %z, %x[%i] : !d
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// At a slot of its item: not a place that two items share.
func.func @stage_20(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable writes(%x : !d) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %z = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{writes a side output at a place that is not a slot of the item}}
    memref.store %z, %x[%c0] : !d
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Nor the item times zero.
func.func @stage_21(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable writes(%x : !d) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %z = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %at = arith.muli %i, %c0 : index
    // expected-error @+1 {{writes a side output at a place that is not a slot of the item}}
    memref.store %z, %x[%at] : !d
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// The other indices of a slot are constants.
func.func @stage_22(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable writes(%w3 : memref<?x3xf32, 1>) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %f = arith.constant 0.0 : f32
    // expected-error @+1 {{writes a side output at a place that is not a slot of the item}}
    memref.store %f, %w3[%i, %i] : memref<?x3xf32, 1>
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A reduction is an addition: not a maximum.
func.func @stage_23(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable adds(%x : !d) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %one32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{is not an addition to a counter that the key region declares in 'adds'}}
    %r = memref.atomic_rmw maxs %one32, %x[%c0] : (i32, !d) -> i32
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Through an address: of a declared counter.
func.func @stage_24(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable adds(%x : !d) : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %one32 = arith.constant 1 : i32
    %a = arith.index_cast %i : index to i64
    %ptr = llvm.inttoptr %a : i64 to !llvm.ptr<1>
    // expected-error @+1 {{is not an addition at the address of a counter that the key region declares in 'adds', plus a constant}}
    %r = llvm.atomicrmw add %ptr, %one32 monotonic : !llvm.ptr<1>, i32
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// An address is taken of a declared counter alone.
func.func @stage_25(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    // expected-error @+1 {{takes the address of a buffer that the key region does not declare in 'adds'}}
    %a = memref.extract_aligned_pointer_as_index %x : !d -> index
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// No op whose effects it cannot state.
func.func @stage_26(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    // expected-error @+1 {{has effects that the key region of 'md_alg.sort_by_key' cannot state}}
    func.call @opaque() : () -> ()
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// No other write.
func.func @stage_27(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u into %o order stable : !d
      launch(%g, %b, %one) scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one) sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    // expected-error @+1 {{writes memory in the key region of 'md_alg.sort_by_key' other than a declared side output}}
    memref.copy %x, %by : !d to !d
    md_alg.no_bin
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A bin stands in a key region.
func.func @stage_28(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{stands in the key region of an 'md_alg.sort_by_key', directly or in the branches of 'scf.if'}}
  md_alg.bin %n : index
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// So does an item without one.
func.func @stage_29(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>, %p: memref<5xi32, 1>, %sums: !d, %wide: memref<?xi64, 1>, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{stands in the key region of an 'md_alg.sort_by_key', directly or in the branches of 'scf.if'}}
  md_alg.no_bin
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// An append writes one value into each order stable buffer of `into`.
func.func @stage_30(%keep: i1, %v: i32, %e: !d, %m: !d, %cs: i32, %csb: i32, %own: i32, %un: !d, %ord: !d, %pool: !d, %at: index, %cap: index, %lane: i32) {
  // expected-error @+1 {{takes one buffer of 'into' for each of its values, and at least one}}
  %r:2 = md_alg.append %keep values(%v, %v) into(%e : !d) list(%cs, %csb) owner %own names(%un, %ord) pool %pool[%at] capacity %cap block 64 : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Its blocks hold a power of two of entries.
func.func @stage_31(%keep: i1, %v: i32, %e: !d, %m: !d, %cs: i32, %csb: i32, %own: i32, %un: !d, %ord: !d, %pool: !d, %at: index, %cap: index, %lane: i32) {
  // expected-error @+1 {{takes blocks of a power of two of entries, not 48}}
  %r:2 = md_alg.append %keep values(%v) into(%e : !d) list(%cs, %csb) owner %own names(%un, %ord) pool %pool[%at] capacity %cap block 48 : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Its lane and the mask of the lanes below it together.
func.func @stage_32(%keep: i1, %v: i32, %e: !d, %m: !d, %cs: i32, %csb: i32, %own: i32, %un: !d, %ord: !d, %pool: !d, %at: index, %cap: index, %lane: i32) {
  // expected-error @+1 {{takes its lane and the mask of the lanes below it together}}
  %r:2 = "md_alg.append"(%keep, %v, %e, %cs, %csb, %own, %un, %ord, %pool, %at, %cap, %lane) <{block = 64 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0>}> : (i1, i32, !d, i32, i32, i32, !d, !d, !d, index, index, i32) -> (i32, i32)
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Its buffers are distinct.
func.func @stage_33(%keep: i1, %v: i32, %e: !d, %m: !d, %cs: i32, %csb: i32, %own: i32, %un: !d, %ord: !d, %pool: !d, %at: index, %cap: index, %lane: i32) {
  // expected-error @+1 {{takes one buffer for two of its entries, names, and pool}}
  %r:2 = md_alg.append %keep values(%v) into(%e : !d) list(%cs, %csb) owner %own names(%e, %ord) pool %pool[%at] capacity %cap block 64 : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A grid plan has an edge or more, and a count for each.
func.func @stage_34(%lx: f64, %ly: f64, %w: f64, %v: f64, %n: index, %h: f32) {
  // expected-error @+1 {{plans along one edge or more, a count for each}}
  %r = md_alg.grid_plan %lx, %ly : f64, f64 width %w : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Its edges and width are of one type.
func.func @stage_35(%lx: f64, %ly: f64, %w: f64, %v: f64, %n: index, %h: f32) {
  // expected-error @+1 {{takes its edges and width of one type}}
  %r:2 = md_alg.grid_plan %lx, %h : f64, f32 width %w : f64 bounds [1.0 : f64, 256.0 : f64] -> i64, i64
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Its bounds are of their type.
func.func @stage_36(%lx: f64, %ly: f64, %w: f64, %v: f64, %n: index, %h: f32) {
  // expected-error @+1 {{takes its bounds and occupancy of the type of its edges}}
  %r = md_alg.grid_plan %lx : f64 width %w : f64 bounds [1.0 : f32, 256.0 : f64] -> i64
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// Whole, with 1 <= least <= most.
func.func @stage_37(%lx: f64, %ly: f64, %w: f64, %v: f64, %n: index, %h: f32) {
  // expected-error @+1 {{takes whole bounds with 1 <= least <= most}}
  %r = md_alg.grid_plan %lx : f64 width %w : f64 bounds [3.0 : f64, 2.5 : f64] -> i64
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// A width or an occupancy.
func.func @stage_38(%lx: f64, %ly: f64, %w: f64, %v: f64, %n: index, %h: f32) {
  // expected-error @+1 {{takes a width or an occupancy, one of them}}
  %r = md_alg.grid_plan %lx : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// An occupancy above zero.
func.func @stage_39(%lx: f64, %ly: f64, %w: f64, %v: f64, %n: index, %h: f32) {
  // expected-error @+1 {{takes an occupancy greater than zero}}
  %r = md_alg.grid_plan %lx : f64 volume %v : f64 occupancy 0.0 : f64 of %n bounds [1.0 : f64, 256.0 : f64] -> i64
  return
}

// -----

!d = memref<?xi32, 1>
func.func private @opaque()
// An occupancy with its volume and population.
func.func @stage_40(%lx: f64, %ly: f64, %w: f64, %v: f64, %n: index, %h: f32) {
  // expected-error @+1 {{takes an occupancy with its volume and its population}}
  %r = "md_alg.grid_plan"(%lx, %v) <{least = 1.0 : f64, most = 256.0 : f64, occupancy = 64.0 : f64, operandSegmentSizes = array<i32: 1, 0, 1, 0>}> : (f64, f64) -> i64
  return
}

// -----

// The order within the bins is a contract of its own, which no strategy
// implies: an unordered sort takes no `by`.
!d = memref<?xi32, 1>
func.func @unordered_by(%n: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d, %by: !d) {
  // expected-error @+1 {{orders its bins by 'by' only with the order 'stable'}}
  md_alg.sort_by_key %n bins %n keys %k counts %c start %s cursor %u
      into %o order unordered by %by : !d every_key : !d
  return
}

// -----

// Nor a sort within its bins.
!d = memref<?xi32, 1>
func.func @unordered_sort(%n: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d,
                          %sums: !d, %g: index) {
  // expected-error @+1 {{with the order 'unordered' takes no sort within its bins}}
  md_alg.sort_by_key %n bins %n keys %k counts %c start %s cursor %u
      into %o order unordered every_key : !d launch(%g, %g, %g)
      scan reduce_then_scan scratch %sums : !d launch(%g, %g, %g)
      sort insertion launch(%g, %g, %g)
  return
}

// -----

// A stable sort with an execution group takes the sort within its bins.
!d = memref<?xi32, 1>
func.func @stable_sort(%n: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d,
                       %sums: !d, %g: index) {
  // expected-error @+1 {{takes its launch, its scan's strategy, scratch, and launch, and its sort's strategy and launch together}}
  md_alg.sort_by_key %n bins %n keys %k counts %c start %s cursor %u
      into %o order stable every_key : !d launch(%g, %g, %g)
      scan reduce_then_scan scratch %sums : !d launch(%g, %g, %g)
  return
}
