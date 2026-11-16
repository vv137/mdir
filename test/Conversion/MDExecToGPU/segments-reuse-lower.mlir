// The form that md-exec-reuse-neighbors gives a loop over steps, a list of
// units carried through the loop's values, lowers on a device
// (D[md-alg-segments-build]). The build of a list of units in a loop over
// steps becomes an empty list before it and a refresh in it, carried; on
// the device's buffers md-exec-assign-storage keeps that shape, and
// convert-md-exec-to-gpu finds the positions through the loop, stops
// carrying the list, which is refreshed where it is, and builds it with the
// build over units.
//
// RUN: mdir-opt %s --md-exec-reuse-neighbors --md-exec-assign-storage="memory=device" | FileCheck %s --check-prefix=CARRIED
// RUN: mdir-opt %s --md-exec-reuse-neighbors --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu | FileCheck %s
//
// CARRIED: %[[E:.*]] = md_exec.empty_segmented_pairs
// CARRIED: scf.for {{.*}} iter_args(%[[L:.*]] = %[[E]]) -> (!mdrt.segmented_pairs
// CARRIED: md_exec.refresh_segmented_pairs %[[L]]
// CHECK-LABEL: func.func @steps
// CHECK-NOT: !mdrt.segmented_pairs
// CHECK: scf.for
// CHECK: call @mdrt_gpu_build_segmented_pairs_k1(
// CHECK-NOT: !mdrt.segmented_pairs
// CHECK: return

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @steps(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %l = md_exec.build_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) at(%x, %cell) cutoff(0.9) skin(0.1) : memref<?xi32, 1>, memref<?x3xf64, 1> -> !sp
  }
  return
}
