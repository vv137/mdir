// The accessors of md_alg (D406) and the grid plan (D[md-alg-stages]) by
// themselves, on the host (convert-md-alg-to-scf): their code is that of
// their emitters in both lowerings, so these values, written out in
// md-alg-accessors.test, are their test.

func.func private @printI64(i64)
func.func private @printF64(f64)
func.func private @printNewline()

func.func @print(%v: index) {
  %w = arith.index_cast %v : index to i64
  func.call @printI64(%w) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}

func.func @print32(%v: i32) {
  %w = arith.extsi %v : i32 to i64
  func.call @printI64(%w) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}

func.func @print64(%v: i64) {
  func.call @printI64(%v) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}

func.func @printf(%v: f64) {
  func.call @printF64(%v) : (f64) -> ()
  func.call @printNewline() : () -> ()
  return
}

// Ragged rows of 3 rows, entries of 2 values: offsets 0, 2, 2, 5 (rows of
// 2, 0, and 3 entries), entry value j at 5 + j is 100 + j, and the entry of
// zeros at the end: 3 + 2 + 2 (5 + 1) = 17 elements.
func.func @ragged() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c15 = arith.constant 15 : index
  %c17 = arith.constant 17 : index
  %rows = memref.alloc(%c17) : memref<?xi32>
  %zero = arith.constant 0 : i32
  scf.for %i = %c0 to %c17 step %c1 {
    memref.store %zero, %rows[%i] : memref<?xi32>
  }
  %three = arith.constant 3 : i32
  %two = arith.constant 2 : i32
  %five = arith.constant 5 : i32
  memref.store %three, %rows[%c0] : memref<?xi32>
  memref.store %zero, %rows[%c1] : memref<?xi32>
  memref.store %two, %rows[%c2] : memref<?xi32>
  memref.store %two, %rows[%c3] : memref<?xi32>
  memref.store %five, %rows[%c4] : memref<?xi32>
  scf.for %i = %c5 to %c15 step %c1 {
    %j = arith.subi %i, %c5 : index
    %j32 = arith.index_cast %j : index to i32
    %h = arith.constant 100 : i32
    %v = arith.addi %h, %j32 : i32
    memref.store %v, %rows[%i] : memref<?xi32>
  }
  // Rows 0, 1, 2, unchecked: (5, 2), (9, 0), (9, 3).
  %f0, %n0 = md_alg.ragged_row %rows[%c0] entry %c2 : memref<?xi32>
  %f1, %n1 = md_alg.ragged_row %rows[%c1] entry %c2 : memref<?xi32>
  %f2, %n2 = md_alg.ragged_row %rows[%c2] entry %c2 : memref<?xi32>
  // Checked: row 2 as before, rows 3 and 7 past the rows: (0, 0).
  %f3, %n3 = md_alg.ragged_row %rows[%c2] entry %c2 checked : memref<?xi32>
  %f4, %n4 = md_alg.ragged_row %rows[%c3] entry %c2 checked : memref<?xi32>
  %f5, %n5 = md_alg.ragged_row %rows[%c7] entry %c2 checked : memref<?xi32>
  // A buffer of no elements, checked: (0, 0).
  %empty = memref.alloc(%c0) : memref<?xi32>
  %f6, %n6 = md_alg.ragged_row %empty[%c1] entry %c2 checked : memref<?xi32>
  func.call @print(%f0) : (index) -> ()
  func.call @print(%n0) : (index) -> ()
  func.call @print(%f1) : (index) -> ()
  func.call @print(%n1) : (index) -> ()
  func.call @print(%f2) : (index) -> ()
  func.call @print(%n2) : (index) -> ()
  func.call @print(%f3) : (index) -> ()
  func.call @print(%n3) : (index) -> ()
  func.call @print(%f4) : (index) -> ()
  func.call @print(%n4) : (index) -> ()
  func.call @print(%f5) : (index) -> ()
  func.call @print(%n5) : (index) -> ()
  func.call @print(%f6) : (index) -> ()
  func.call @print(%n6) : (index) -> ()
  // Entries: row 0 entry 1 value 0 (102), row 2 entry 1 value 1 (107),
  // row 2 entry 2 value 0 (108).
  %e0 = md_alg.ragged_entry %rows[%f0, %c1, %c0] entry %c2 : memref<?xi32>
  %e1 = md_alg.ragged_entry %rows[%f2, %c1, %c1] entry %c2 : memref<?xi32>
  %e2 = md_alg.ragged_entry %rows[%f2, %c2, %c0] entry %c2 : memref<?xi32>
  func.call @print32(%e0) : (i32) -> ()
  func.call @print32(%e1) : (i32) -> ()
  func.call @print32(%e2) : (i32) -> ()
  return
}

// Wrapped coordinates: 3 + 4 in 10 (7); 8 + 5 in 10 (3); 9 + 9 in 10,
// signed (8); 0 + 0 in 1 (0); 6 + 7 in 10 as index (3).
func.func @wraps() {
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32
  %i3 = arith.constant 3 : i32
  %i4 = arith.constant 4 : i32
  %i5 = arith.constant 5 : i32
  %i8 = arith.constant 8 : i32
  %i9 = arith.constant 9 : i32
  %i10 = arith.constant 10 : i32
  %a = md_alg.wrap %i3, %i4, %i10 : i32
  %b = md_alg.wrap %i8, %i5, %i10 : i32
  %c = md_alg.wrap %i9, %i9, %i10 signed : i32
  %d = md_alg.wrap %i0, %i0, %i1 : i32
  %x6 = arith.constant 6 : index
  %x7 = arith.constant 7 : index
  %x10 = arith.constant 10 : index
  %e = md_alg.wrap %x6, %x7, %x10 : index
  func.call @print32(%a) : (i32) -> ()
  func.call @print32(%b) : (i32) -> ()
  func.call @print32(%c) : (i32) -> ()
  func.call @print32(%d) : (i32) -> ()
  func.call @print(%e) : (index) -> ()
  return
}

// Reads of a grid of f32 (point k is k + 0.5) and of f64 (k + 0.25):
// base 2 + z 3 (5.5); the same, inactive (the base: 2.5); index
// coordinates 4 + 1 (5.25).
func.func @reads() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %g = memref.alloc(%c8) : memref<?xf32>
  %h = memref.alloc(%c8) : memref<?xf64>
  scf.for %k = %c0 to %c8 step %c1 {
    %k64 = arith.index_cast %k : index to i64
    %kf = arith.sitofp %k64 : i64 to f64
    %half = arith.constant 0.5 : f64
    %quarter = arith.constant 0.25 : f64
    %a = arith.addf %kf, %half : f64
    %b = arith.addf %kf, %quarter : f64
    %a32 = arith.truncf %a : f64 to f32
    memref.store %a32, %g[%k] : memref<?xf32>
    memref.store %b, %h[%k] : memref<?xf64>
  }
  %i2 = arith.constant 2 : i32
  %i3 = arith.constant 3 : i32
  %true = arith.constant true
  %false = arith.constant false
  %p = md_alg.grid_read %g[%i2 + %i3] : memref<?xf32>, i32
  %q = md_alg.grid_read %g[%i2 + %i3] if %false : memref<?xf32>, i32
  %r = md_alg.grid_read %g[%i2 + %i3] if %true : memref<?xf32>, i32
  %x4 = arith.constant 4 : index
  %s = md_alg.grid_read %h[%x4 + %c1] : memref<?xf64>, index
  %p64 = arith.extf %p : f32 to f64
  %q64 = arith.extf %q : f32 to f64
  %r64 = arith.extf %r : f32 to f64
  func.call @printf(%p64) : (f64) -> ()
  func.call @printf(%q64) : (f64) -> ()
  func.call @printf(%r64) : (f64) -> ()
  func.call @printf(%s) : (f64) -> ()
  return
}

// Grid plans, min(max(floor(L / w), least), most): 30 / 7 (4); 30 / 100
// (1, the least); 1e9 / 1 (256, the most); a length that is not a number
// (1); a width that is not a number (1); an infinite length (256); a
// width of 0 (256); a negative length (1); 10 / 5 with a least of 3 (3).
// By an occupancy of 64: 3,000 items in 30^3 (an edge of 8.32: 3, 3); no
// items (an infinite edge: 1, 1); 180,000 in 400 x 90 x 50 (46, 10). In
// f32: 30 / 7 (4).
func.func @plans() {
  %l30 = arith.constant 30.0 : f64
  %l1e9 = arith.constant 1.0e9 : f64
  %lm5 = arith.constant -5.0 : f64
  %l10 = arith.constant 10.0 : f64
  %w7 = arith.constant 7.0 : f64
  %w100 = arith.constant 100.0 : f64
  %w1 = arith.constant 1.0 : f64
  %w0 = arith.constant 0.0 : f64
  %w5 = arith.constant 5.0 : f64
  %zero = arith.constant 0.0 : f64
  %nan = arith.divf %zero, %zero : f64
  %one = arith.constant 1.0 : f64
  %inf = arith.divf %one, %zero : f64
  %a = md_alg.grid_plan %l30 : f64 width %w7 : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  %b = md_alg.grid_plan %l30 : f64 width %w100 : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  %c = md_alg.grid_plan %l1e9 : f64 width %w1 : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  %d = md_alg.grid_plan %nan : f64 width %w1 : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  %e = md_alg.grid_plan %l30 : f64 width %nan : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  %f = md_alg.grid_plan %inf : f64 width %w1 : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  %g = md_alg.grid_plan %l30 : f64 width %w0 : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  %h = md_alg.grid_plan %lm5 : f64 width %w1 : f64 bounds [1.0 : f64, 256.0 : f64] -> i64
  %i = md_alg.grid_plan %l10 : f64 width %w5 : f64 bounds [3.0 : f64, 256.0 : f64] -> i64
  func.call @print64(%a) : (i64) -> ()
  func.call @print64(%b) : (i64) -> ()
  func.call @print64(%c) : (i64) -> ()
  func.call @print64(%d) : (i64) -> ()
  func.call @print64(%e) : (i64) -> ()
  func.call @print64(%f) : (i64) -> ()
  func.call @print64(%g) : (i64) -> ()
  func.call @print64(%h) : (i64) -> ()
  func.call @print64(%i) : (i64) -> ()
  %v27 = arith.constant 27000.0 : f64
  %n3000 = arith.constant 3000 : index
  %n0 = arith.constant 0 : index
  %j:2 = md_alg.grid_plan %l30, %l30 : f64, f64 volume %v27 : f64
      occupancy 64.0 : f64 of %n3000 bounds [1.0 : f64, 256.0 : f64] -> i64, i64
  %k:2 = md_alg.grid_plan %l30, %l30 : f64, f64 volume %v27 : f64
      occupancy 64.0 : f64 of %n0 bounds [1.0 : f64, 256.0 : f64] -> i64, i64
  %l400 = arith.constant 400.0 : f64
  %l90 = arith.constant 90.0 : f64
  %v18 = arith.constant 1800000.0 : f64
  %n180 = arith.constant 180000 : index
  %m:2 = md_alg.grid_plan %l400, %l90 : f64, f64 volume %v18 : f64
      occupancy 64.0 : f64 of %n180 bounds [1.0 : f64, 256.0 : f64] -> i64, i64
  func.call @print64(%j#0) : (i64) -> ()
  func.call @print64(%j#1) : (i64) -> ()
  func.call @print64(%k#0) : (i64) -> ()
  func.call @print64(%k#1) : (i64) -> ()
  func.call @print64(%m#0) : (i64) -> ()
  func.call @print64(%m#1) : (i64) -> ()
  %s30 = arith.constant 30.0 : f32
  %s7 = arith.constant 7.0 : f32
  %n = md_alg.grid_plan %s30 : f32 width %s7 : f32 bounds [1.0 : f32, 256.0 : f32] -> i64
  func.call @print64(%n) : (i64) -> ()
  return
}

func.func @main() {
  func.call @ragged() : () -> ()
  func.call @wraps() : () -> ()
  func.call @reads() : () -> ()
  func.call @plans() : () -> ()
  return
}
