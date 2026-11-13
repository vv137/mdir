// The driver of the appends of stages.mlir on a device, for the SIMT
// lowering (D407): a block of one warp for each list, all
// lists at once, which append round after round, the lanes of a round
// together, each with its lane and the mask of the lanes below it.

func.func @append_lists(%keep: !buf, %vals: !buf, %lists: index,
                        %rounds: index, %entries: !buf, %units: !buf,
                        %ordinals: !buf, %pool: !buf, %counts: !buf,
                        %cap: index) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %blocks = arith.maxui %lists, %c1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %blocks, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
    %inside = arith.cmpi ult, %bx, %lists : index
    scf.if %inside {
      %c0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i32 = arith.constant 32 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %none = arith.constant -1 : i32
      %lane = arith.index_cast %tx : index to i32
      %shifted = arith.shli %one, %lane : i32
      %below = arith.subi %shifted, %one : i32
      %g32 = arith.index_cast %bx : index to i32
      %base = arith.muli %bx, %rounds : index
      %end:2 = scf.for %r = %c0 to %rounds step %i1
          iter_args(%cs = %zero, %csb = %none) -> (i32, i32) {
        %row = arith.addi %base, %r : index
        %first = arith.muli %row, %i32 : index
        %item = arith.addi %first, %tx : index
        %k = memref.load %keep[%item] : !buf
        %kept = arith.cmpi eq, %k, %one : i32
        %v = memref.load %vals[%item] : !buf
        %n, %b = md_alg.append %kept values(%v) into(%entries : !buf)
            list(%cs, %csb) owner %g32 names(%units, %ordinals)
            pool %pool[%i1] capacity %cap block 64 lane(%lane, %below)
            : !buf, !buf, !buf
        scf.yield %n, %b : i32, i32
      }
      %lead = arith.cmpi eq, %tx, %c0 : index
      scf.if %lead {
        memref.store %end#0, %counts[%bx] : !buf
      }
    }
    gpu.terminator
  }
  return
}
