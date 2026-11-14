// Both lowerings of md_alg.bin_members (D409): the loop over
// the places of its bins, from start[from] to start[to], the item of each
// place through the load of md_alg.bin_item, the region inlined with the
// item for its argument, and the values combined in order. The SIMT
// lowering writes the same loop where the op stands (here in a kernel) and
// records no policy; the serial lowering reverses an unordered visit under
// its option.
//
// RUN: mdir-opt %s --convert-md-alg-to-gpu | FileCheck %s
// RUN: mdir-opt %s --convert-md-alg-to-scf | FileCheck %s
// RUN: mdir-opt %s --convert-md-alg-to-scf=reverse-unordered-visits=true \
// RUN: | FileCheck %s --check-prefix=REVERSE

!d = memref<?xi32, 1>

// CHECK-NOT: mdir.alg_policy
// CHECK-LABEL: func.func @kernel
// CHECK: gpu.launch
// CHECK: %[[S0:.*]] = memref.load %[[START:.*]][%[[FROM:.*]]] : memref<?xi32, 1>
// CHECK: %[[S1:.*]] = memref.load %[[START]][%[[TO:.*]]] : memref<?xi32, 1>
// CHECK: %[[B:.*]] = arith.index_cast %[[S0]] : i32 to index
// CHECK: %[[E:.*]] = arith.index_cast %[[S1]] : i32 to index
// CHECK: %[[R:.*]] = scf.for %[[P:.*]] = %[[B]] to %[[E]] step %{{.*}} iter_args(%[[A:.*]] = %{{.*}}) -> (i32) {
// CHECK-NEXT: %[[I:.*]] = memref.load %{{.*}}[%[[P]]] : memref<?xi32, 1>
// CHECK-NEXT: %[[X:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK-NEXT: %[[V:.*]] = arith.index_cast %[[X]] : index to i32
// CHECK-NEXT: memref.store %[[V]], %{{.*}}[%[[X]]] : memref<?xi32, 1>
// CHECK-NEXT: %[[N:.*]] = arith.addi %[[A]], %[[V]] : i32
// CHECK-NEXT: scf.yield %[[N]] : i32
// CHECK-NOT: md_alg.
// REVERSE-LABEL: func.func @kernel
// REVERSE: scf.for %[[P:.*]] = %[[B:.*]] to %[[E:.*]] step %[[ONE:.*]] iter_args
// REVERSE-NEXT: %[[S:.*]] = arith.addi %[[E]], %[[B]] : index
// REVERSE-NEXT: %[[L:.*]] = arith.subi %[[S]], %[[ONE]] : index
// REVERSE-NEXT: %[[Q:.*]] = arith.subi %[[L]], %[[P]] : index
// REVERSE-NEXT: memref.load %{{.*}}[%[[Q]]] : memref<?xi32, 1>
func.func @kernel(%s: !d, %o: !d, %mark: !d, %from: index, %to: index,
                  %out: !d) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %z = arith.constant 0 : i32
    %sum = md_alg.bin_members %s[%from, %to] items %o visit unordered
        writes(%mark : !d) init(%z : i32) combine ["addi"] : !d, !d {
    ^bb0(%item: index):
      %v = arith.index_cast %item : index to i32
      memref.store %v, %mark[%item] : !d
      md_alg.yield %v : i32
    } -> i32
    %c0 = arith.constant 0 : index
    memref.store %sum, %out[%c0] : !d
    gpu.terminator
  }
  return
}
