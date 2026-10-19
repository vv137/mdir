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
