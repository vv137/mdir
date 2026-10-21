// The expansion of the calls of the device primitives
// (D296): each call becomes the code of its emitter where
// it stands, in the order of the sites that the emitters replaced, and the
// declarations go.
//
// RUN: mdir-opt %s --expand-device-primitives | FileCheck %s
// expand-primitives-invalid.mlir holds the refusals.

// CHECK-NOT: mdir.primitive
func.func private @mdir.primitive.warp_lanes_below(i32, i32) -> i32
func.func private @mdir.primitive.warp_ballot(i1) -> i32
func.func private @mdir.primitive.warp_rank(i32, i32) -> i32
func.func private @mdir.primitive.warp_count(i32) -> i32
func.func private @mdir.primitive.warp_scan(i32, i32, i32, i32) -> i32

// The warp compaction: the mask of the lanes below from the site's 1, the
// ballot, the rank under the mask, and the count.
// CHECK-LABEL: func.func @compaction(
// CHECK-SAME:    %[[KEEP:[a-z0-9]+]]: i1, %[[LANE:[a-z0-9]+]]: i32, %[[ONE:[a-z0-9]+]]: i32)
// CHECK:         %[[BIT:.+]] = arith.shli %[[ONE]], %[[LANE]] : i32
// CHECK-NEXT:    %[[BELOW:.+]] = arith.subi %[[BIT]], %[[ONE]] : i32
// CHECK-NEXT:    %[[BALLOT:.+]] = gpu.ballot %[[KEEP]] : i32
// CHECK-NEXT:    %[[AND:.+]] = arith.andi %[[BALLOT]], %[[BELOW]] : i32
// CHECK-NEXT:    %[[RANK:.+]] = math.ctpop %[[AND]] : i32
// CHECK-NEXT:    %[[COUNT:.+]] = math.ctpop %[[BALLOT]] : i32
// CHECK-NEXT:    return %[[RANK]], %[[COUNT]]
func.func @compaction(%keep: i1, %lane: i32, %one: i32) -> (i32, i32) {
  %below = func.call @mdir.primitive.warp_lanes_below(%lane, %one) : (i32, i32) -> i32
  %bits = func.call @mdir.primitive.warp_ballot(%keep) : (i1) -> i32
  %rank = func.call @mdir.primitive.warp_rank(%bits, %below) : (i32, i32) -> i32
  %count = func.call @mdir.primitive.warp_count(%bits) : (i32) -> i32
  return %rank, %count : i32, i32
}

// A warp scan over segments of 16 lanes, compared as unsigned numbers:
// four rounds of a shuffle up, a test of the lane, and an addition.
// CHECK-LABEL: func.func @scan16(
// CHECK-COUNT-4: gpu.shuffle up
// CHECK-NOT:     gpu.shuffle
// CHECK-NOT:     arith.cmpi sge
// CHECK:         return
// Over 32 lanes, five rounds, the lane compared as a signed number.
// CHECK-LABEL: func.func @scan32(
// CHECK-COUNT-5: gpu.shuffle up
// CHECK-NOT:     gpu.shuffle
// CHECK-NOT:     arith.cmpi uge
// CHECK:         return
func.func @scan16(%v: i32, %lane: i32, %zero: i32, %width: i32) -> i32 {
  %s = func.call @mdir.primitive.warp_scan(%v, %lane, %zero, %width) {lanes = 16 : i64, unsigned} : (i32, i32, i32, i32) -> i32
  return %s : i32
}
func.func @scan32(%v: i32, %lane: i32, %zero: i32, %width: i32) -> i32 {
  %s = func.call @mdir.primitive.warp_scan(%v, %lane, %zero, %width) : (i32, i32, i32, i32) -> i32
  return %s : i32
}

// The accessor of ragged rows (D314, RaggedRows.h): the row,
// checked against a buffer of no elements and against the number of rows
// at element 0, from the offsets at elements i + 1 and i + 2 past the
// header of N + 2 elements; then a value of an entry of the row.
func.func private @mdir.primitive.ragged_row(memref<?xi32, 1>, index, index) -> (index, index)
func.func private @mdir.primitive.ragged_entry(memref<?xi32, 1>, index, index, index, index) -> i32
// CHECK-LABEL: func.func @ragged(
// CHECK-SAME:    %[[ROWS:[a-z0-9]+]]: memref<?xi32, 1>, %[[I:[a-z0-9]+]]: index, %[[E:[a-z0-9]+]]: index, %[[K:[a-z0-9]+]]: index, %[[C:[a-z0-9]+]]: index)
// CHECK:         %[[SIZE:.+]] = memref.dim %[[ROWS]], %{{.+}} : memref<?xi32, 1>
// CHECK:         %[[SOME:.+]] = arith.cmpi ne, %[[SIZE]]
// CHECK:         %[[ROW:.+]]:2 = scf.if %[[SOME]] -> (index, index)
// CHECK:           memref.load %[[ROWS]][%{{.+}}] : memref<?xi32, 1>
// CHECK:           arith.cmpi ult, %[[I]]
// CHECK:           scf.if
// CHECK:             %[[AT:.+]] = arith.addi %[[I]], %{{.+}} : index
// CHECK:             memref.load %[[ROWS]][%[[AT]]] : memref<?xi32, 1>
// CHECK:             arith.muli %{{.+}}, %[[E]] : index
// CHECK:         %[[STEP:.+]] = arith.muli %[[K]], %[[E]] : index
// CHECK:         %[[ENTRY:.+]] = arith.addi %[[ROW]]#0, %[[STEP]] : index
// CHECK:         %[[VALUE:.+]] = arith.addi %[[ENTRY]], %[[C]] : index
// CHECK:         %[[V:.+]] = memref.load %[[ROWS]][%[[VALUE]]] : memref<?xi32, 1>
// CHECK:         return %[[ROW]]#1, %[[V]]
func.func @ragged(%rows: memref<?xi32, 1>, %i: index, %e: index, %k: index,
                  %c: index) -> (index, i32) {
  %first, %count = func.call @mdir.primitive.ragged_row(%rows, %i, %e)
      : (memref<?xi32, 1>, index, index) -> (index, index)
  %v = func.call @mdir.primitive.ragged_entry(%rows, %first, %k, %c, %e)
      : (memref<?xi32, 1>, index, index, index, index) -> i32
  return %count, %v : index, i32
}

// Both PME gather paths use one wrapped-axis emitter and one grid-read
// emitter. The order-4 stored-weight path keeps its i32 address arithmetic.
func.func private @mdir.primitive.wrapped_grid_axis_index(index, index, index) -> index
func.func private @mdir.primitive.wrapped_grid_axis_i32(i32, i32, i32) -> i32
func.func private @mdir.primitive.grid_gather_index(memref<?xf64, 1>, index, index) -> f64
func.func private @mdir.primitive.grid_gather_i32(memref<?xf32, 1>, i32, i32) -> f32

// CHECK-LABEL: func.func @grid_index(
// CHECK: arith.addi %{{.+}}, %{{.+}} : index
// CHECK: arith.subi %{{.+}}, %{{.+}} : index
// CHECK: arith.cmpi uge
// CHECK: arith.select
// CHECK: arith.addi %{{.+}}, %{{.+}} : index
// CHECK: memref.load %{{.+}}[%{{.+}}] : memref<?xf64, 1>
func.func @grid_index(%grid: memref<?xf64, 1>, %base: index,
                      %z: index, %lane: index, %size: index) -> f64 {
  %wrapped = func.call @mdir.primitive.wrapped_grid_axis_index(%z, %lane, %size)
      : (index, index, index) -> index
  %value = func.call @mdir.primitive.grid_gather_index(%grid, %base, %wrapped)
      : (memref<?xf64, 1>, index, index) -> f64
  return %value : f64
}

// CHECK-LABEL: func.func @grid_i32(
// CHECK: arith.addi %{{.+}}, %{{.+}} : i32
// CHECK: arith.subi %{{.+}}, %{{.+}} : i32
// CHECK: arith.cmpi uge
// CHECK: arith.select
// CHECK: arith.addi %{{.+}}, %{{.+}} : i32
// CHECK: arith.index_cast %{{.+}} : i32 to index
// CHECK: memref.load %{{.+}}[%{{.+}}] : memref<?xf32, 1>
func.func @grid_i32(%grid: memref<?xf32, 1>, %base: i32,
                    %z: i32, %lane: i32, %size: i32) -> f32 {
  %wrapped = func.call @mdir.primitive.wrapped_grid_axis_i32(%z, %lane, %size)
      : (i32, i32, i32) -> i32
  %value = func.call @mdir.primitive.grid_gather_i32(%grid, %base, %wrapped)
      : (memref<?xf32, 1>, i32, i32) -> f32
  return %value : f32
}

func.func private @mdir.primitive.grid_gather_masked_i32(memref<?xf64, 1>, i32, i32, i1) -> f64

// An inactive prolate lane reads the start of its row, retaining the
// existing in-bounds load and the signed coordinate comparison.
// CHECK-LABEL: func.func @grid_prolate(
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.addi %{{.+}}, %{{.+}} : i32
// CHECK: arith.select %{{.+}}, %{{.+}}, %{{.+}} : i32
// CHECK: arith.index_cast %{{.+}} : i32 to index
// CHECK: memref.load %{{.+}}[%{{.+}}] : memref<?xf64, 1>
func.func @grid_prolate(%grid: memref<?xf64, 1>, %base: i32,
                        %z: i32, %lane: i32, %size: i32,
                        %active: i1) -> f64 {
  %wrapped = func.call @mdir.primitive.wrapped_grid_axis_i32(%z, %lane, %size)
      {signed} : (i32, i32, i32) -> i32
  %value = func.call @mdir.primitive.grid_gather_masked_i32(
      %grid, %base, %wrapped, %active)
      : (memref<?xf64, 1>, i32, i32, i1) -> f64
  return %value : f64
}
