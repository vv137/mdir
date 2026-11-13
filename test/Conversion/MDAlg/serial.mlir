// The serial lowering of md_alg (convert-md-alg-to-scf, D406): the
// definitions of the ops in loops of one thread, with no launch, atomic,
// or barrier, whatever execution parameters the ops carry.
// Integration/md-alg-serial.test runs them; md-alg-contract-gpu.test
// compares them with the emitters.
//
// RUN: mdir-opt %s --convert-md-alg-to-scf | FileCheck %s \
// RUN:     --implicit-check-not=md_alg --implicit-check-not=gpu. \
// RUN:     --implicit-check-not=llvm.atomicrmw

!d = memref<?xi32>

// The scan: one loop that stores the sum before each count, in the start
// and the cursor, and the sum of all after it.
// CHECK-LABEL: func.func @scan(
// CHECK-SAME:    %[[COUNTS:[a-z0-9]+]]: memref<?xi32>, %[[START:[a-z0-9]+]]: memref<?xi32>, %[[CURSOR:[a-z0-9]+]]: memref<?xi32>, %{{.*}}: memref<?xi32>, %[[N:[a-z0-9]+]]: index
// CHECK:         %[[TOTAL:.+]] = scf.for %[[I:.+]] = %{{.+}} to %[[N]] step %{{.+}} iter_args(%[[S:.+]] = %{{.+}}) -> (i32) {
// CHECK:           memref.store %[[S]], %[[START]][%[[I]]]
// CHECK:           memref.store %[[S]], %[[CURSOR]][%[[I]]]
// CHECK:           %[[V:.+]] = memref.load %[[COUNTS]][%[[I]]]
// CHECK:           %[[NEXT:.+]] = arith.addi %[[S]], %[[V]] : i32
// CHECK:           scf.yield %[[NEXT]]
// CHECK:         memref.store %[[TOTAL]], %[[START]][%[[N]]]
func.func @scan(%counts: !d, %start: !d, %cursor: !d, %sums: !d, %n: index,
                %b: index, %t: index, %one: index) {
  md_alg.scan %counts[%n] into %start cursor %cursor
      strategy reduce_then_scan scratch %sums : !d launch(%b, %t, %one)
      : !d, !d, !d
  return
}

// The scatter: the items in order, each with a bin to the slot of its
// cursor, which is incremented.
// CHECK-LABEL: func.func @scatter(
// CHECK:         scf.for
// CHECK:           arith.cmpi sge
// CHECK:           scf.if
// CHECK:             memref.load
// CHECK:             arith.addi
// CHECK:             memref.store
// CHECK:             memref.store
func.func @scatter(%keys: !d, %cursor: !d, %out: !d, %n: index) {
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out : !d, !d, !d
  return
}

// The sort within bins: the items copied first where it sorts into another
// buffer, then exchanges with the item before while it orders after.
// CHECK-LABEL: func.func @sort_into(
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:           scf.for
// CHECK:             scf.while
func.func @sort_into(%start: !d, %items: !d, %out: !d, %keys: !d, %n: index,
                     %g: index, %t: index, %one: index) {
  md_alg.sort_bins %start[%n] items %items into %out keys %keys
      strategy stable_rank launch(%g, %t, %one) : !d, !d into !d keys !d
  return
}

// The stage ops (D[md-alg-stages]): the counting sort a loop over the
// items running the key region (each bin a store of its key and an
// increment of its count), then the scan, the scatter, and the sort as
// above; the append the item's own: the block taken from the pool where
// its slot begins one.
// CHECK-LABEL: func.func @sort_by_key(
// CHECK:         scf.for
// CHECK:           memref.store %{{.*}}, %[[C:[a-z0-9]+]][
// CHECK:         scf.for %[[I:.+]] = %{{.+}} to %{{.+}} step
// CHECK:           memref.load %{{.*}}[%[[I]]]
// CHECK:           scf.if
// CHECK:             arith.constant -1 : i32
// CHECK:           } else {
// CHECK:             memref.load %[[C]][
// CHECK:             arith.addi
// CHECK:             memref.store %{{.*}}, %[[C]][
// CHECK:         scf.for
// CHECK:         scf.for
// CHECK:         scf.for
// CHECK:         return
func.func @sort_by_key(%n: index, %bins: index, %k: !d, %c: !d, %s: !d,
                       %u: !d, %o: !d, %x: !d, %sums: !d, %g: index,
                       %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u
      into %o order stable : !d launch(%g, %b, %one)
      scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one)
      sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %v = memref.load %x[%i] : !d
    %zero = arith.constant 0 : i32
    %no = arith.cmpi slt, %v, %zero : i32
    scf.if %no {
      md_alg.no_bin
    } else {
      md_alg.bin %v : i32
    }
  }
  return
}

// CHECK-LABEL: func.func @append(
// CHECK:         scf.if %{{.*}} -> (i32, i32)
// CHECK:           arith.remui
// CHECK:           arith.divui
// CHECK:           scf.if %{{.*}} -> (i32)
// CHECK:             memref.load %[[POOL:[a-z0-9]+]][
// CHECK:             memref.store %{{.*}}, %[[POOL]][
// CHECK:         return
func.func @append(%keep: i1, %v: i32, %e: !d, %cs: i32, %csb: i32,
                  %own: i32, %un: !d, %ord: !d, %pool: !d, %at: index,
                  %cap: index, %lane: i32, %below: i32) -> (i32, i32) {
  %count, %last = md_alg.append %keep values(%v) into(%e : !d)
      list(%cs, %csb) owner %own names(%un, %ord) pool %pool[%at]
      capacity %cap block 64 lane(%lane, %below) : !d, !d, !d
  return %count, %last : i32, i32
}
