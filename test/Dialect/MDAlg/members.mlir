// md_alg.bin_members (D[md-alg-analysis]): a reader of the bins of a
// counting sort that states the order of its visit, in its three visits,
// with side outputs (a slot of the item, an addition to a counter),
// reductions, and a determinism requirement, in the custom and the generic
// form.
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s
// RUN: mdir-opt %s --mlir-print-op-generic | mdir-opt | FileCheck %s

!d = memref<?xi32, 1>

// CHECK-LABEL: func.func @visits
func.func @visits(%s: !d, %o: !d, %by: !d, %mark: !d, %count: !d,
                  %from: index, %to: index) -> (i32, i32, f32) {
  %z = arith.constant 0 : i32
  %f = arith.constant 0.0 : f32
  // CHECK: md_alg.bin_members %{{.*}}[%{{.*}}, %{{.*}}] items %{{.*}} visit unordered writes(%{{.*}} : memref<?xi32, 1>) adds(%{{.*}} : memref<?xi32, 1>) init(%{{.*}}, %{{.*}} : i32, i32) combine ["addi", "maxsi"] : memref<?xi32, 1>, memref<?xi32, 1> {
  // CHECK: md_alg.yield %{{.*}}, %{{.*}} : i32, i32
  // CHECK: } -> i32, i32
  %sum, %high = md_alg.bin_members %s[%from, %to] items %o visit unordered
      writes(%mark : !d) adds(%count : !d)
      init(%z, %z : i32, i32) combine ["addi", "maxsi"] : !d, !d {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    memref.store %v, %mark[%item] : !d
    %old = memref.atomic_rmw addi %c1, %count[%c0] : (i32, !d) -> i32
    md_alg.yield %v, %v : i32, i32
  } -> i32, i32
  // CHECK: md_alg.bin_members %{{.*}} visit sorted by %{{.*}} : memref<?xi32, 1> init(%{{.*}} : f32) combine ["addf"] determinism reproducible
  %t = md_alg.bin_members %s[%from, %to] items %o visit sorted by %by : !d
      init(%f : f32) combine ["addf"] determinism reproducible : !d, !d {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    %w = arith.sitofp %v : i32 to f32
    md_alg.yield %w : f32
  } -> f32
  // CHECK: md_alg.bin_members %{{.*}} visit stable : memref<?xi32, 1>, memref<?xi32, 1> {
  md_alg.bin_members %s[%from, %to] items %o visit stable : !d, !d {
  ^bb0(%item: index):
    md_alg.yield
  }
  return %sum, %high, %t : i32, i32, f32
}
