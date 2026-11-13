// The record of the policy (D408): for each op with a determinism
// aspect that the SIMT lowering lowers, an entry of the module's attribute
// `mdir.alg_policy`, in the order of the lowering, after the entries that
// were there (those of the ops that convert-md-exec-to-gpu lowers where it
// creates them): the function, the op, its requirement (stated or the
// default), its order, its strategies, and its launches, a constant as its
// number and any other value as `?`. An accessor has no entry, and the
// serial lowering, which has no policy, records none.
//
// RUN: mdir-opt %s --convert-md-alg-to-gpu | FileCheck %s
// RUN: mdir-opt %s --convert-md-alg-to-scf | FileCheck %s --check-prefix=SERIAL

// CHECK: module attributes {mdir.alg_policy = [
// CHECK-SAME: {earlier},
// CHECK-SAME: {determinism = "bitwise", function = "scan", launch = "(?,256,1)", op = "scan", strategy = "reduce_then_scan"},
// CHECK-SAME: {determinism = "reproducible", function = "scan", launch = "(?,256,1)", op = "scan", strategy = "single_pass"},
// CHECK-SAME: {determinism = "unordered", function = "scatter", launch = "(?,128,1)", op = "scatter_to_bins"},
// CHECK-SAME: {determinism = "bitwise", function = "sort", launch = "(?,?,1)", op = "sort_by_key", order = "stable", scan = "chunked", scan_launch = "(?,?,?,?,1)", sort = "insertion", sort_launch = "(?,?,1)"},
// CHECK-SAME: {determinism = "unordered", function = "sort", launch = "(?,?,1)", op = "sort_by_key", order = "unordered", scan = "reduce_then_scan", scan_launch = "(?,?,1)"},
// CHECK-SAME: {block = "64", determinism = "unordered", function = "append", op = "append"},
// CHECK-SAME: {determinism = "bitwise", function = "plan", op = "grid_plan"}]}
// SERIAL: module attributes {mdir.alg_policy = [{earlier}]} {
// SERIAL-NOT: md_alg
module attributes {mdir.alg_policy = [{earlier}]} {
  func.func @scan(%counts: memref<?xi32, 1>, %start: memref<?xi32, 1>,
                  %sums: memref<?xi32, 1>, %state: memref<?xi64, 1>,
                  %n: index, %blocks: index) {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    md_alg.scan %counts[%n] into %start strategy reduce_then_scan
        scratch %sums : memref<?xi32, 1> launch(%blocks, %c256, %c1)
        : memref<?xi32, 1>, memref<?xi32, 1>
    md_alg.scan %counts[%n] into %start strategy single_pass
        scratch %state : memref<?xi64, 1> launch(%blocks, %c256, %c1)
        determinism reproducible : memref<?xi32, 1>, memref<?xi32, 1>
    return
  }
  func.func @scatter(%keys: memref<?xi32, 1>, %cursor: memref<?xi32, 1>,
                     %out: memref<?xi32, 1>, %n: index, %g: index) {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out
        launch(%g, %c128, %c1)
        : memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>
    // An accessor: no entry.
    %first, %count = md_alg.ragged_row %keys[%n] entry %n
        : memref<?xi32, 1>
    return
  }
  func.func @sort(%n: index, %k: memref<?xi32, 1>, %c: memref<?xi32, 1>,
                  %s: memref<?xi32, 1>, %u: memref<?xi32, 1>,
                  %o: memref<?xi32, 1>, %sums: memref<?xi32, 1>, %g: index,
                  %b: index) {
    %c1 = arith.constant 1 : index
    md_alg.sort_by_key %n bins %n keys %k counts %c start %s cursor %u
        into %o order stable every_key : memref<?xi32, 1> launch(%g, %b, %c1)
        scan chunked scratch %sums : memref<?xi32, 1>
            launch(%g, %b, %g, %b, %c1)
        sort insertion launch(%g, %b, %c1)
    md_alg.sort_by_key %n bins %n keys %k counts %c start %s cursor %u
        into %o order unordered every_key : memref<?xi32, 1>
        launch(%g, %b, %c1)
        scan reduce_then_scan scratch %sums : memref<?xi32, 1>
            launch(%g, %b, %c1)
    return
  }
  func.func @append(%keep: i1, %v: i32, %e: memref<?xi32, 1>, %cs: i32,
                    %un: memref<?xi32, 1>, %ord: memref<?xi32, 1>,
                    %pool: memref<5xi32, 1>, %at: index) -> i32 {
    %r:2 = md_alg.append %keep values(%v) into(%e : memref<?xi32, 1>)
        list(%cs, %cs) owner %cs names(%un, %ord) pool %pool[%at]
        capacity %at block 64 lane(%cs, %cs)
        : memref<?xi32, 1>, memref<?xi32, 1>, memref<5xi32, 1>
    return %r#0 : i32
  }
  func.func @plan(%lx: f64, %w: f64) -> i64 {
    %c = md_alg.grid_plan %lx : f64 width %w : f64
        bounds [1.0 : f64, 256.0 : f64] -> i64
    return %c : i64
  }
}
