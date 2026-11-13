// The warp helpers of CompactionEmitter.h, each by itself
// (test-md-alg-warp-helpers, D407), in one warp of a device:
// for each lane l, the mask of the lanes below it; the ballot of the lanes
// with l mod 3 = 0, the rank of each among them, and their count; the
// inclusive scan of l + 1 over 32 lanes and over segments of 16 (the
// shuffles over the whole warp, the lane its place in its segment, unsigned,
// as the groups build scans its excluded pairs); and the
// scan of 1 with the lane taken as l - 8, compared signed and unsigned.
// Each buffer has a mark (-7) past its 32 lanes.

func.func private @test.lanes_below(i32) -> i32
func.func private @test.warp_ballot(i1) -> i32
func.func private @test.warp_rank(i32, i32) -> i32
func.func private @test.warp_count(i32) -> i32
func.func private @test.warp_scan(i32, i32, i32, i32) -> i32

func.func @main() {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c33 = arith.constant 33 : index
  %below = func.call @marked(%c33) : (index) -> !buf
  %ballot = func.call @marked(%c33) : (index) -> !buf
  %rank = func.call @marked(%c33) : (index) -> !buf
  %count = func.call @marked(%c33) : (index) -> !buf
  %s32 = func.call @marked(%c33) : (index) -> !buf
  %s16 = func.call @marked(%c33) : (index) -> !buf
  %sneg = func.call @marked(%c33) : (index) -> !buf
  %uneg = func.call @marked(%c33) : (index) -> !buf
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %three = arith.constant 3 : i32
    %eight = arith.constant 8 : i32
    %sixteen = arith.constant 16 : i32
    %w32 = arith.constant 32 : i32
    %lane = arith.index_cast %tx : index to i32
    %b = func.call @test.lanes_below(%lane) : (i32) -> i32
    memref.store %b, %below[%tx] : !buf
    %m = arith.remui %lane, %three : i32
    %third = arith.cmpi eq, %m, %zero : i32
    %v = func.call @test.warp_ballot(%third) : (i1) -> i32
    memref.store %v, %ballot[%tx] : !buf
    %r = func.call @test.warp_rank(%v, %b) : (i32, i32) -> i32
    memref.store %r, %rank[%tx] : !buf
    %n = func.call @test.warp_count(%v) : (i32) -> i32
    memref.store %n, %count[%tx] : !buf
    %x = arith.addi %lane, %one : i32
    %a = func.call @test.warp_scan(%x, %lane, %zero, %w32) {lanes = 32 : i64}
        : (i32, i32, i32, i32) -> i32
    memref.store %a, %s32[%tx] : !buf
    %seg = arith.remui %lane, %sixteen : i32
    %h = func.call @test.warp_scan(%x, %seg, %zero, %w32) {lanes = 16 : i64, unsigned}
        : (i32, i32, i32, i32) -> i32
    memref.store %h, %s16[%tx] : !buf
    %shifted = arith.subi %lane, %eight : i32
    %sn = func.call @test.warp_scan(%one, %shifted, %zero, %w32) {lanes = 32 : i64}
        : (i32, i32, i32, i32) -> i32
    memref.store %sn, %sneg[%tx] : !buf
    %un = func.call @test.warp_scan(%one, %shifted, %zero, %w32) {lanes = 32 : i64, unsigned}
        : (i32, i32, i32, i32) -> i32
    memref.store %un, %uneg[%tx] : !buf
    gpu.terminator
  }
  func.call @emit_buf(%below) : (!buf) -> ()
  func.call @emit_buf(%ballot) : (!buf) -> ()
  func.call @emit_buf(%rank) : (!buf) -> ()
  func.call @emit_buf(%count) : (!buf) -> ()
  func.call @emit_buf(%s32) : (!buf) -> ()
  func.call @emit_buf(%s16) : (!buf) -> ()
  func.call @emit_buf(%sneg) : (!buf) -> ()
  func.call @emit_buf(%uneg) : (!buf) -> ()
  return
}
