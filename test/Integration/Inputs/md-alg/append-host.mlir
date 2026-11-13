// The driver of the appends of stages.mlir on the host, for the serial
// definition (D407): list after list, round after round, each
// item of a round's 32 by itself, in order. The append takes no lanes.

func.func @append_lists(%keep: !buf, %vals: !buf, %lists: index,
                        %rounds: index, %entries: !buf, %units: !buf,
                        %ordinals: !buf, %pool: !buf, %counts: !buf,
                        %cap: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %none = arith.constant -1 : i32
  scf.for %g = %c0 to %lists step %c1 {
    %g32 = arith.index_cast %g : index to i32
    %base = arith.muli %g, %rounds : index
    %end:2 = scf.for %r = %c0 to %rounds step %c1
        iter_args(%cs = %zero, %csb = %none) -> (i32, i32) {
      %row = arith.addi %base, %r : index
      %first = arith.muli %row, %c32 : index
      %after:2 = scf.for %l = %c0 to %c32 step %c1
          iter_args(%ls = %cs, %lb = %csb) -> (i32, i32) {
        %item = arith.addi %first, %l : index
        %k = memref.load %keep[%item] : !buf
        %kept = arith.cmpi eq, %k, %one : i32
        %v = memref.load %vals[%item] : !buf
        %n, %b = md_alg.append %kept values(%v) into(%entries : !buf)
            list(%ls, %lb) owner %g32 names(%units, %ordinals)
            pool %pool[%c1] capacity %cap block 64 : !buf, !buf, !buf
        scf.yield %n, %b : i32, i32
      }
      scf.yield %after#0, %after#1 : i32, i32
    }
    memref.store %end#0, %counts[%g] : !buf
  }
  return
}
