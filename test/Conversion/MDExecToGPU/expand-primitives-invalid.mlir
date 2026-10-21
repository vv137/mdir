// The refusals of the expansion of the device primitives
// (D296).
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics \
// RUN:     --expand-device-primitives

func.func private @mdir.primitive.warp_sort(i32) -> i32
func.func @unknown(%v: i32) -> i32 {
  // expected-error @+1 {{calls an unknown device primitive 'warp_sort'}}
  %s = func.call @mdir.primitive.warp_sort(%v) : (i32) -> i32
  return %s : i32
}

// -----

func.func private @mdir.primitive.warp_rank(i32) -> i32
func.func @operands(%v: i32) -> i32 {
  // expected-error @+1 {{gives the device primitive 'warp_rank' 1 operands; it takes 2}}
  %s = func.call @mdir.primitive.warp_rank(%v) : (i32) -> i32
  return %s : i32
}

// -----

func.func private @mdir.primitive.warp_ballot(i32) -> i32
func.func @types(%v: i32) -> i32 {
  // expected-error @+1 {{gives the device primitive 'warp_ballot' an operand 0 of the type 'i32', which it does not take}}
  %s = func.call @mdir.primitive.warp_ballot(%v) : (i32) -> i32
  return %s : i32
}

// -----

func.func private @mdir.primitive.warp_count(i32) -> i64
func.func @results(%v: i32) -> i64 {
  // expected-error @+1 {{declares results of the device primitive 'warp_count' that it does not have}}
  %s = func.call @mdir.primitive.warp_count(%v) : (i32) -> i64
  return %s : i64
}

// -----

// expected-error @+1 {{defines a device primitive, which the emitters give}}
func.func private @mdir.primitive.warp_count(%v: i32) -> i32 {
  return %v : i32
}

// -----

func.func private @mdir.primitive.grid_gather_i32(memref<?xf32, 1>, i32, i32) -> f64
func.func @grid_result(%grid: memref<?xf32, 1>, %row: i32, %z: i32) -> f64 {
  // expected-error @+1 {{declares a grid gather result unlike the grid element type}}
  %v = func.call @mdir.primitive.grid_gather_i32(%grid, %row, %z)
      : (memref<?xf32, 1>, i32, i32) -> f64
  return %v : f64
}
