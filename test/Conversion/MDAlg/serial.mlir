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
