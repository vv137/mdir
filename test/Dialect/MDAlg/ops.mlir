// The ops of md_alg (D[md-alg]) in their custom form: each prints as it
// parses, and the generic form parses back to it. The result types of the
// accessors are inferred: the grid read takes the element type of its
// grid.
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s
// RUN: mdir-opt %s --mlir-print-op-generic | mdir-opt | FileCheck %s

!d = memref<?xi32, 1>
!w = memref<?xi64, 1>

// CHECK-LABEL: func.func @collections(
func.func @collections(%counts: !d, %start: !d, %cursor: !d, %sums: !d,
                       %state: !w, %keys: !d, %out: !d, %by: !d,
                       %n: index, %b: index, %t: index, %one: index) {
  // CHECK: md_alg.scan %{{.*}}[%{{.*}}] into %{{.*}} cursor %{{.*}} strategy reduce_then_scan scratch %{{.*}} : memref<?xi32, 1> launch(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>
  md_alg.scan %counts[%n] into %start cursor %cursor
      strategy reduce_then_scan scratch %sums : !d launch(%b, %t, %one)
      : !d, !d, !d
  // CHECK: md_alg.scan %{{.*}}[%{{.*}}] into %{{.*}} strategy single_pass scratch %{{.*}} : memref<?xi64, 1> launch(
  md_alg.scan %counts[%n] into %start
      strategy single_pass scratch %state : !w launch(%b, %t, %one) : !d, !d
  // CHECK: md_alg.scan %{{.*}}[%{{.*}}] into %{{.*}} cursor %{{.*}} strategy chunked scratch %{{.*}} : memref<?xi32, 1> launch(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) :
  md_alg.scan %counts[%n] into %start cursor %cursor
      strategy chunked scratch %sums : !d launch(%b, %t, %b, %t, %one)
      : !d, !d, !d
  // The meaning alone, for the serial lowering and a policy to come.
  // CHECK: md_alg.scan %{{.*}}[%{{.*}}] into %{{.*}} : memref<?xi32, 1>, memref<?xi32, 1>
  md_alg.scan %counts[%n] into %start : !d, !d
  // CHECK: md_alg.scatter_to_bins %{{.*}}[%{{.*}}] cursor %{{.*}} into %{{.*}} launch(%{{.*}}, %{{.*}}, %{{.*}}) {every_key} : memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out
      launch(%b, %t, %one) {every_key} : !d, !d, !d
  // CHECK: md_alg.sort_bins %{{.*}}[%{{.*}}] items %{{.*}} by %{{.*}} strategy insertion launch(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?xi32, 1>, memref<?xi32, 1> by memref<?xi32, 1>
  md_alg.sort_bins %start[%n] items %out by %by
      strategy insertion launch(%b, %t, %one) : !d, !d by !d
  // CHECK: md_alg.sort_bins %{{.*}}[%{{.*}}] items %{{.*}} into %{{.*}} keys %{{.*}} strategy stable_rank launch(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?xi32, 1>, memref<?xi32, 1> into memref<?xi32, 1> keys memref<?xi32, 1>
  md_alg.sort_bins %start[%n] items %out into %by keys %keys
      strategy stable_rank launch(%b, %t, %one) : !d, !d into !d keys !d
  return
}

// CHECK-LABEL: func.func @accessors(
func.func @accessors(%rows: !d, %i: index, %e: index, %k: index,
                     %grid: memref<?xf32, 1>, %base: i32, %z: i32,
                     %size: i32, %use: i1) -> (index, i32, i32, f32, f32) {
  // CHECK: %{{.*}}, %{{.*}} = md_alg.ragged_row %{{.*}}[%{{.*}}] entry %{{.*}} checked : memref<?xi32, 1>
  %first, %count = md_alg.ragged_row %rows[%i] entry %e checked : !d
  // CHECK: md_alg.ragged_entry %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] entry %{{.*}} : memref<?xi32, 1>
  %v = md_alg.ragged_entry %rows[%first, %k, %i] entry %e : !d
  // CHECK: md_alg.wrap %{{.*}}, %{{.*}}, %{{.*}} signed : i32
  %w = md_alg.wrap %base, %z, %size signed : i32
  // CHECK: md_alg.grid_read %{{.*}}[%{{.*}} + %{{.*}}] : memref<?xf32, 1>, i32
  %p = md_alg.grid_read %grid[%base + %w] : memref<?xf32, 1>, i32
  // CHECK: md_alg.grid_read %{{.*}}[%{{.*}} + %{{.*}}] if %{{.*}} : memref<?xf32, 1>, i32
  %q = md_alg.grid_read %grid[%base + %w] if %use : memref<?xf32, 1>, i32
  return %count, %v, %w, %p, %q : index, i32, i32, f32, f32
}
