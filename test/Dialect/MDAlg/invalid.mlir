// The refusals of the verifiers of md_alg (D[md-alg]): one case for each
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
