// The determinism requirements of md_alg (D[md-alg-ir]): each op with an
// ordering or a floating-point aspect states one, in its custom form, and
// the generic form parses back to it. Every tier up to an op's bound is
// accepted: a scan, a sort within bins, a stable counting sort, and a grid
// plan up to `bitwise`; a scatter, an unordered counting sort, and an
// append up to `reproducible`, which a schedule can fix though their
// contracts do not. Without the attribute nothing is printed.
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s
// RUN: mdir-opt %s --mlir-print-op-generic | mdir-opt | FileCheck %s

!d = memref<?xi32, 1>

// CHECK-LABEL: func.func @tiers(
func.func @tiers(%counts: !d, %start: !d, %cursor: !d, %keys: !d, %out: !d,
                 %n: index, %keep: i1, %v: i32, %cs: i32, %pool: !d,
                 %lx: f64, %w: f64) -> (i32, i64) {
  // CHECK: md_alg.scan %{{.*}}[%{{.*}}] into %{{.*}} : memref<?xi32, 1>, memref<?xi32, 1>
  md_alg.scan %counts[%n] into %start : !d, !d
  // CHECK: md_alg.scan %{{.*}}[%{{.*}}] into %{{.*}} determinism bitwise : memref<?xi32, 1>, memref<?xi32, 1>
  md_alg.scan %counts[%n] into %start determinism bitwise : !d, !d
  // CHECK: md_alg.scan %{{.*}}[%{{.*}}] into %{{.*}} determinism unordered :
  md_alg.scan %counts[%n] into %start determinism unordered : !d, !d
  // CHECK: md_alg.scatter_to_bins %{{.*}}[%{{.*}}] cursor %{{.*}} into %{{.*}} determinism reproducible :
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out
      determinism reproducible : !d, !d, !d
  // CHECK: md_alg.sort_bins %{{.*}}[%{{.*}}] items %{{.*}} determinism reproducible :
  md_alg.sort_bins %start[%n] items %out determinism reproducible : !d, !d
  // CHECK: md_alg.sort_by_key %{{.*}} order stable every_key : memref<?xi32, 1> determinism bitwise{{$}}
  md_alg.sort_by_key %n bins %n keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !d
      determinism bitwise
  // CHECK: md_alg.sort_by_key %{{.*}} order unordered every_key : memref<?xi32, 1> determinism reproducible{{$}}
  md_alg.sort_by_key %n bins %n keys %keys counts %counts start %start
      cursor %cursor into %out order unordered every_key : !d
      determinism reproducible
  // CHECK: md_alg.append {{.*}} block 64 determinism reproducible : memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>
  %count, %last = md_alg.append %keep values(%v) into(%out : !d)
      list(%cs, %cs) owner %cs names(%keys, %cursor) pool %pool[%n]
      capacity %n block 64 determinism reproducible : !d, !d, !d
  // CHECK: md_alg.grid_plan %{{.*}} : f64 width %{{.*}} : f64 bounds[1.000000e+00 : f64, 2.560000e+02 : f64] determinism bitwise -> i64
  %c = md_alg.grid_plan %lx : f64 width %w : f64
      bounds [1.0 : f64, 256.0 : f64] determinism bitwise -> i64
  return %count, %c : i32, i64
}
