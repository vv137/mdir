// RUN: mdir-opt %s --md-exec-reuse-neighbors=prune-skin=0.05 --verify-each | FileCheck %s
// RUN: mdir-opt %s --md-exec-reuse-neighbors=prune-skin=0.05 --md-exec-assign-precision=mode=mixed --md-exec-assign-storage=memory=host --verify-each | FileCheck %s --check-prefix=STORAGE
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(3) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!water = !md.relation<@atoms, 3, ordered, @water>
!list = !mdrt.segmented_pairs<@atoms>

// One allocation lives through both loop levels. Repeated evaluations
// continue its state rather than initialize another allocation.
// CHECK-LABEL: func.func @nested
// CHECK: %[[EMPTY:.*]] = md_exec.empty_segmented_pairs
// CHECK: scf.for {{.*}} iter_args(%[[OUTER:.*]] = %[[EMPTY]])
// CHECK-NOT: md_exec.empty_segmented_pairs
// CHECK: scf.for {{.*}} iter_args(%[[INNER:.*]] = %[[OUTER]])
// CHECK-NOT: md_exec.empty_segmented_pairs
// CHECK: %[[ONE:.*]] = md_exec.refresh_segmented_pairs %[[INNER]]
// CHECK-SAME: prune_skin(5.000000e-02)
// CHECK: %[[TWO:.*]] = md_exec.refresh_segmented_pairs %[[ONE]]
// CHECK: scf.yield %[[TWO]] : !mdrt.segmented_pairs<@atoms>
// STORAGE-LABEL: func.func @nested
// STORAGE: md_exec.empty_segmented_pairs
// STORAGE-SAME: memref<?xi32
// STORAGE: scf.for
// STORAGE: scf.for
// STORAGE: md_exec.refresh_segmented_pairs
// STORAGE-SAME: memref<?x3xf64
func.func @nested(%x: !x, %h: !md.cell, %types: !i, %water: !water, %n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %frame = %zero to %n step %one {
    scf.for %step = %zero to %n step %one {
      %a = md_exec.build_segmented_pairs %types, %water at(%x, %h)
        cutoff(1.0) skin(0.2) extent(0.1) alpha(0.125)
        : !i, !water, !x -> !list
      %b = md_exec.build_segmented_pairs %types, %water at(%x, %h)
        cutoff(1.0) skin(0.2) extent(0.1) alpha(0.125)
        : !i, !water, !x -> !list
    }
  }
  return
}

// A segment resets its allocation, while steps inside it reuse it.
// CHECK-LABEL: func.func @segments
// CHECK-NOT: md_exec.empty_segmented_pairs
// CHECK: scf.for
// CHECK: %[[SEG:.*]] = md_exec.empty_segmented_pairs
// CHECK: scf.for {{.*}} iter_args(%[[STATE:.*]] = %[[SEG]])
// CHECK: md_exec.refresh_segmented_pairs %[[STATE]]
// CHECK: } {mdrt.segment}
func.func @segments(%x: !x, %h: !md.cell, %types: !i, %water: !water, %n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %segment = %zero to %n step %one {
    scf.for %step = %zero to %n step %one {
      %list = md_exec.build_segmented_pairs %types, %water at(%x, %h)
        cutoff(1.0) skin(0.2) extent(0.1) alpha(0.125)
        : !i, !water, !x -> !list
    }
  } {mdrt.segment}
  return
}

// Fresh step loops reset after the preceding initial evaluation. That
// reset also cannot move out of an enclosing loop and become shared.
// CHECK-LABEL: func.func @fresh
// CHECK-NOT: md_exec.empty_segmented_pairs
// CHECK: scf.for
// CHECK: md_exec.empty_segmented_pairs
// CHECK: md_exec.refresh_segmented_pairs
// CHECK: %[[FRESH:.*]] = md_exec.empty_segmented_pairs
// CHECK-SAME: {mdrt.fresh}
// CHECK: scf.for {{.*}} iter_args(%[[STATE:.*]] = %[[FRESH]])
// CHECK: md_exec.refresh_segmented_pairs %[[STATE]]
// CHECK: } {mdrt.fresh}
func.func @fresh(%x: !x, %h: !md.cell, %types: !i, %water: !water, %n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %segment = %zero to %n step %one {
    %initial = md_exec.build_segmented_pairs %types, %water at(%x, %h)
      cutoff(1.0) skin(0.2) extent(0.1) alpha(0.125)
      : !i, !water, !x -> !list
    scf.for %step = %zero to %n step %one {
      %list = md_exec.build_segmented_pairs %types, %water at(%x, %h)
        cutoff(1.0) skin(0.2) extent(0.1) alpha(0.125)
        : !i, !water, !x -> !list
    } {mdrt.fresh}
  } {mdrt.segment}
  return
}
