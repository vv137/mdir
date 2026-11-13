// The ops of md_alg (D406) in their custom form: each prints as it
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

// The stage ops (D[md-alg-stages]): a counting sort by a key region that
// writes a slot of its item and adds to a counter, one by keys given, one
// unordered (no sort within its bins), the readers of the bins, an append,
// and the grid plans by a width and by an occupancy.
// CHECK-LABEL: func.func @stages(
func.func @stages(%n: index, %bins: index, %k: !d, %c: !d, %s: !d, %u: !d,
                  %o: !d, %by: !d, %x: !d, %w3: memref<?x3xf32, 1>,
                  %sums: !d, %g: index, %b: index, %one: index,
                  %keep: i1, %v: i32, %cs: i32, %csb: i32, %own: i32,
                  %pool: memref<5xi32, 1>, %at: index, %lane: i32,
                  %below: i32, %lx: f64, %ly: f64, %w: f64, %vol: f64)
    -> (i32, i64) {
  // CHECK: md_alg.sort_by_key %{{.*}} bins %{{.*}} keys %{{.*}} counts %{{.*}} start %{{.*}} cursor %{{.*}} into %{{.*}} order stable by %{{.*}} : memref<?xi32, 1> writes(%{{.*}} : memref<?x3xf32, 1>) adds(%{{.*}} : memref<?xi32, 1>) cleared : memref<?xi32, 1> launch(%{{.*}}, %{{.*}}, %{{.*}}) scan chunked scratch %{{.*}} : memref<?xi32, 1> launch(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) sort insertion launch(%{{.*}}, %{{.*}}, %{{.*}}) key {
  // CHECK: ^bb0(%{{.*}}: index):
  // CHECK: md_alg.no_bin
  // CHECK: md_alg.bin %{{.*}} : i32
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u
      into %o order stable by %by : !d writes(%w3 : memref<?x3xf32, 1>) adds(%x : !d)
      cleared : !d launch(%g, %b, %one)
      scan chunked scratch %sums : !d launch(%g, %b, %g, %b, %one)
      sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %c0 = arith.constant 0 : index
    %f = arith.constant 1.0 : f32
    memref.store %f, %w3[%i, %c0] : memref<?x3xf32, 1>
    %key = memref.load %by[%i] : !d
    %zero = arith.constant 0 : i32
    %none = arith.cmpi slt, %key, %zero : i32
    scf.if %none {
      md_alg.no_bin
      %one32 = arith.constant 1 : i32
      %old = memref.atomic_rmw addi %one32, %x[%c0] : (i32, !d) -> i32
    } else {
      md_alg.bin %key : i32
    }
  }
  // CHECK: md_alg.sort_by_key %{{.*}} every_key : memref<?xi32, 1> launch(%{{.*}}) scan reduce_then_scan scratch %{{.*}} : memref<?xi32, 1> launch(%{{.*}}) sort stable_rank buffer %{{.*}} : memref<?xi32, 1> launch(
  // CHECK-NOT: key {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u
      into %o order stable every_key : !d launch(%g, %b, %one)
      scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one)
      sort stable_rank buffer %x : !d launch(%g, %b, %one)
  // CHECK: md_alg.sort_by_key %{{.*}} order unordered every_key : memref<?xi32, 1> launch(%{{.*}}) scan reduce_then_scan scratch %{{.*}} : memref<?xi32, 1> launch(%{{.*}}){{$}}
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u
      into %o order unordered every_key : !d launch(%g, %b, %one)
      scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one)
  // CHECK: md_alg.bin_span %{{.*}}[%{{.*}}, %{{.*}}] : memref<?xi32, 1>
  %begin, %end = md_alg.bin_span %s[%g, %b] : !d
  // CHECK: md_alg.bin_item %{{.*}}[%{{.*}}] : memref<?xi32, 1>
  %item = md_alg.bin_item %o[%g] : !d
  // CHECK: md_alg.append %{{.*}} values(%{{.*}}) into(%{{.*}} : memref<?xi32, 1>) list(%{{.*}}, %{{.*}}) owner %{{.*}} names(%{{.*}}, %{{.*}}) pool %{{.*}}[%{{.*}}] capacity %{{.*}} block 64 lane(%{{.*}}, %{{.*}}) : memref<?xi32, 1>, memref<?xi32, 1>, memref<5xi32, 1>
  %count, %last = md_alg.append %keep values(%v) into(%o : !d)
      list(%cs, %csb) owner %own names(%s, %u) pool %pool[%at]
      capacity %b block 64 lane(%lane, %below) : !d, !d, memref<5xi32, 1>
  // CHECK: md_alg.grid_plan %{{.*}}, %{{.*}} : f64, f64 width %{{.*}} : f64 bounds[3.000000e+00 : f64, 2.560000e+02 : f64] -> i64, i64
  %gx, %gy = md_alg.grid_plan %lx, %ly : f64, f64 width %w : f64
      bounds [3.0 : f64, 256.0 : f64] -> i64, i64
  // CHECK: md_alg.grid_plan %{{.*}} : f64 volume %{{.*}} : f64 occupancy 6.400000e+01 : f64 of %{{.*}} bounds[1.000000e+00 : f64, 2.560000e+02 : f64] -> i64
  %cx = md_alg.grid_plan %lx : f64 volume %vol : f64 occupancy 64.0 : f64
      of %n bounds [1.0 : f64, 256.0 : f64] -> i64
  return %count, %cx : i32, i64
}
