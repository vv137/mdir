// The SIMT lowering of md_alg (convert-md-alg-to-gpu, D406): each
// op becomes the code of the emitter of its device primitive where it
// stands, and no op of md_alg is left. simt-invalid.mlir holds the
// refusals; Integration/md-alg-contract-gpu.test runs the code.
//
// RUN: mdir-opt %s --convert-md-alg-to-gpu | FileCheck %s --implicit-check-not=md_alg

!d = memref<?xi32, 1>

// The scans: three launches of the reduce-then-scan, two of the
// single-pass (a clear of its state and the scan), three of the chunked
// scan, of which the middle one is a single thread.
// CHECK-LABEL: func.func @scans(
// CHECK-COUNT-3: gpu.launch blocks
// CHECK-NOT:     gpu.launch blocks
// CHECK:         return
// CHECK-LABEL: func.func @single_pass(
// CHECK-COUNT-2: gpu.launch blocks
// CHECK-NOT:     gpu.launch blocks
// CHECK:         return
// CHECK-LABEL: func.func @chunked(
// CHECK:         gpu.launch blocks
// CHECK:         gpu.launch blocks(%{{.*}}) in (%{{.*}} = %[[ONE:.*]], %{{.*}} = %[[ONE]], %{{.*}} = %[[ONE]]) threads(%{{.*}}) in (%{{.*}} = %[[ONE]],
// CHECK:         gpu.launch blocks
// CHECK-NOT:     gpu.launch blocks
// CHECK:         return
func.func @scans(%counts: !d, %start: !d, %cursor: !d, %sums: !d, %n: index,
                 %blocks: index, %threads: index, %one: index) {
  md_alg.scan %counts[%n] into %start cursor %cursor
      strategy reduce_then_scan scratch %sums : !d
      launch(%blocks, %threads, %one) : !d, !d, !d
  return
}
func.func @single_pass(%counts: !d, %start: !d, %state: memref<?xi64, 1>,
                       %n: index, %blocks: index, %threads: index,
                       %one: index) {
  md_alg.scan %counts[%n] into %start
      strategy single_pass scratch %state : memref<?xi64, 1>
      launch(%blocks, %threads, %one) : !d, !d
  return
}
func.func @chunked(%counts: !d, %start: !d, %cursor: !d, %sums: !d,
                   %n: index, %chunks: index, %chunk: index, %grid: index,
                   %block: index, %one: index) {
  md_alg.scan %counts[%n] into %start cursor %cursor
      strategy chunked scratch %sums : !d
      launch(%chunks, %chunk, %grid, %block, %one) : !d, !d, !d
  return
}

// The scatter: a launch, a thread an item, an atomic addition to the
// cursor of the item's key; without `every_key` only where the key is not
// negative.
// CHECK-LABEL: func.func @scatter(
// CHECK:         gpu.launch
// CHECK:           arith.cmpi sge
// CHECK:           scf.if
// CHECK:             llvm.atomicrmw add %{{.*}}, %{{.*}} syncscope("device") monotonic
// CHECK-LABEL: func.func @scatter_every_key(
// CHECK:         gpu.launch
// CHECK-NOT:       arith.cmpi sge
// CHECK:           llvm.atomicrmw add
func.func @scatter(%keys: !d, %cursor: !d, %out: !d, %n: index, %grid: index,
                   %block: index, %one: index) {
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out
      launch(%grid, %block, %one) : !d, !d, !d
  return
}
func.func @scatter_every_key(%keys: !d, %cursor: !d, %out: !d, %n: index,
                             %grid: index, %block: index, %one: index) {
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out
      launch(%grid, %block, %one) {every_key} : !d, !d, !d
  return
}

// The sorts within bins: by insertion, a thread a bin, the order read
// from `by` where it is given; by the stable rank, a thread an item, which
// counts the smaller items of its bin.
// CHECK-LABEL: func.func @insertion(
// CHECK:         gpu.launch
// CHECK:           scf.while
// CHECK:         return
// CHECK-LABEL: func.func @insertion_by(
// CHECK:         gpu.launch
// CHECK:           scf.while
// CHECK:             memref.load %arg2
// CHECK-LABEL: func.func @rank(
// CHECK:         gpu.launch
// CHECK:           arith.cmpi slt
// CHECK:           arith.extui
// CHECK-NOT:       scf.while
func.func @insertion(%start: !d, %items: !d, %n: index, %grid: index,
                     %block: index, %one: index) {
  md_alg.sort_bins %start[%n] items %items
      strategy insertion launch(%grid, %block, %one) : !d, !d
  return
}
func.func @insertion_by(%start: !d, %items: !d, %by: !d, %n: index,
                        %grid: index, %block: index, %one: index) {
  md_alg.sort_bins %start[%n] items %items by %by
      strategy insertion launch(%grid, %block, %one) : !d, !d by !d
  return
}
func.func @rank(%start: !d, %items: !d, %out: !d, %keys: !d, %n: index,
                %grid: index, %block: index, %one: index) {
  md_alg.sort_bins %start[%n] items %items into %out keys %keys
      strategy stable_rank launch(%grid, %block, %one) : !d, !d into !d keys !d
  return
}

// The accessors of ragged rows (D314, RaggedRows.h): the row,
// checked against a buffer of no elements and against the number of rows
// at element 0, from the offsets at elements i + 1 and i + 2 past the
// header of N + 2 elements; then a value of an entry of the row.
// CHECK-LABEL: func.func @ragged(
// CHECK-SAME:    %[[ROWS:[a-z0-9]+]]: memref<?xi32, 1>, %[[I:[a-z0-9]+]]: index, %[[E:[a-z0-9]+]]: index, %[[K:[a-z0-9]+]]: index, %[[C:[a-z0-9]+]]: index)
// CHECK:         %[[SIZE:.+]] = memref.dim %[[ROWS]], %{{.+}} : memref<?xi32, 1>
// CHECK:         %[[SOME:.+]] = arith.cmpi ne, %[[SIZE]]
// CHECK:         %[[ROW:.+]]:2 = scf.if %[[SOME]] -> (index, index)
// CHECK:           memref.load %[[ROWS]][%{{.+}}] : memref<?xi32, 1>
// CHECK:           arith.cmpi ult, %[[I]]
// CHECK:           scf.if
// CHECK:             %[[AT:.+]] = arith.addi %[[I]], %{{.+}} : index
// CHECK:             memref.load %[[ROWS]][%[[AT]]] : memref<?xi32, 1>
// CHECK:             arith.muli %{{.+}}, %[[E]] : index
// CHECK:         %[[STEP:.+]] = arith.muli %[[K]], %[[E]] : index
// CHECK:         %[[ENTRY:.+]] = arith.addi %[[ROW]]#0, %[[STEP]] : index
// CHECK:         %[[VALUE:.+]] = arith.addi %[[ENTRY]], %[[C]] : index
// CHECK:         %[[V:.+]] = memref.load %[[ROWS]][%[[VALUE]]] : memref<?xi32, 1>
// CHECK:         return %[[ROW]]#1, %[[V]]
func.func @ragged(%rows: memref<?xi32, 1>, %i: index, %e: index, %k: index,
                  %c: index) -> (index, i32) {
  %first, %count = md_alg.ragged_row %rows[%i] entry %e checked : memref<?xi32, 1>
  %v = md_alg.ragged_entry %rows[%first, %k, %c] entry %e : memref<?xi32, 1>
  return %count, %v : index, i32
}

// Both PME gather paths use one wrapped-axis emitter and one grid-read
// emitter. The order-4 stored-weight path keeps its i32 address arithmetic.
// CHECK-LABEL: func.func @grid_index(
// CHECK: arith.addi %{{.+}}, %{{.+}} : index
// CHECK: arith.subi %{{.+}}, %{{.+}} : index
// CHECK: arith.cmpi uge
// CHECK: arith.select
// CHECK: arith.addi %{{.+}}, %{{.+}} : index
// CHECK: memref.load %{{.+}}[%{{.+}}] : memref<?xf64, 1>
func.func @grid_index(%grid: memref<?xf64, 1>, %base: index,
                      %z: index, %lane: index, %size: index) -> f64 {
  %wrapped = md_alg.wrap %z, %lane, %size : index
  %value = md_alg.grid_read %grid[%base + %wrapped] : memref<?xf64, 1>, index
  return %value : f64
}

// CHECK-LABEL: func.func @grid_i32(
// CHECK: arith.addi %{{.+}}, %{{.+}} : i32
// CHECK: arith.subi %{{.+}}, %{{.+}} : i32
// CHECK: arith.cmpi uge
// CHECK: arith.select
// CHECK: arith.addi %{{.+}}, %{{.+}} : i32
// CHECK: arith.index_cast %{{.+}} : i32 to index
// CHECK: memref.load %{{.+}}[%{{.+}}] : memref<?xf32, 1>
func.func @grid_i32(%grid: memref<?xf32, 1>, %base: i32,
                    %z: i32, %lane: i32, %size: i32) -> f32 {
  %wrapped = md_alg.wrap %z, %lane, %size : i32
  %value = md_alg.grid_read %grid[%base + %wrapped] : memref<?xf32, 1>, i32
  return %value : f32
}

// An inactive prolate lane reads the start of its row, retaining the
// existing in-bounds load and the signed coordinate comparison.
// CHECK-LABEL: func.func @grid_prolate(
// CHECK: arith.cmpi sge
// CHECK: arith.select
// CHECK: arith.addi %{{.+}}, %{{.+}} : i32
// CHECK: arith.select %{{.+}}, %{{.+}}, %{{.+}} : i32
// CHECK: arith.index_cast %{{.+}} : i32 to index
// CHECK: memref.load %{{.+}}[%{{.+}}] : memref<?xf64, 1>
func.func @grid_prolate(%grid: memref<?xf64, 1>, %base: i32,
                        %z: i32, %lane: i32, %size: i32,
                        %active: i1) -> f64 {
  %wrapped = md_alg.wrap %z, %lane, %size signed : i32
  %value = md_alg.grid_read %grid[%base + %wrapped] if %active : memref<?xf64, 1>, i32
  return %value : f64
}

// The stage ops (D407).
//
// A counting sort by a key region: the kernel of the keys, a thread an
// item, with the region's code where its item is below n, each bin a store
// of the key and an atomic addition to its count, no bin a store of -1;
// then the scan's three launches, the scatter, which skips the items
// without a bin, and the sort. With `cleared` there is no clear.
// CHECK-LABEL: func.func @sort_by_key(
// CHECK:         gpu.launch blocks
// CHECK:           arith.cmpi ult
// CHECK:           scf.if
// CHECK:             memref.store %{{.*}}, %[[SIDE:.*]][%{{.*}}, %{{.*}}] : memref<?x3xf32, 1>
// CHECK:             scf.if
// CHECK:               %[[NONE:.*]] = arith.constant -1 : i32
// CHECK:               memref.store %[[NONE]], %[[KEYS:[a-z0-9]+]][
// CHECK:               memref.atomic_rmw addi
// CHECK:             } else {
// CHECK:               %[[K32:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK:               memref.store %[[K32]], %[[KEYS]][
// CHECK:               llvm.atomicrmw add %{{.*}} syncscope("device") monotonic
// CHECK-COUNT-3: gpu.launch blocks
// CHECK:         gpu.launch blocks
// CHECK:           arith.cmpi sge
// CHECK:         gpu.launch blocks
// CHECK-NOT:     gpu.launch blocks
// CHECK:         return
func.func @sort_by_key(%n: index, %bins: index, %k: !d, %c: !d, %s: !d,
                       %u: !d, %o: !d, %x: !d, %w3: memref<?x3xf32, 1>,
                       %sums: !d, %g: index, %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u
      into %o order stable writes(%w3 : memref<?x3xf32, 1>) adds(%x : !d)
      cleared : !d launch(%g, %b, %one)
      scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one)
      sort insertion launch(%g, %b, %one) key {
  ^bb0(%i: index):
    %c0 = arith.constant 0 : index
    %f = arith.constant 1.0 : f32
    memref.store %f, %w3[%i, %c0] : memref<?x3xf32, 1>
    %v = memref.load %x[%i] : !d
    %zero = arith.constant 0 : i32
    %no = arith.cmpi slt, %v, %zero : i32
    scf.if %no {
      md_alg.no_bin
      %one32 = arith.constant 1 : i32
      %old = memref.atomic_rmw addi %one32, %x[%c0] : (i32, !d) -> i32
    } else {
      %key = arith.index_cast %v : i32 to index
      md_alg.bin %key : index
    }
  }
  return
}

// By keys given, every key a bin, without `cleared`: the clear of the
// counts first, a thread a bin, then a count for each item with no test
// of its key, the scan, the scatter, and the stable rank.
// CHECK-LABEL: func.func @sort_by_keys(
// CHECK:         arith.maxui
// CHECK:         gpu.launch blocks
// CHECK:           memref.store %{{.*}}, %[[COUNTS:[a-z0-9]+]][
// CHECK:         gpu.launch blocks
// CHECK-NOT:       arith.cmpi sge
// CHECK:           llvm.atomicrmw add
// CHECK-COUNT-3: gpu.launch blocks
// CHECK:         gpu.launch blocks
// CHECK-NOT:       arith.cmpi sge
// CHECK:         gpu.launch blocks
// CHECK-NOT:     gpu.launch blocks
// CHECK:         return
func.func @sort_by_keys(%n: index, %bins: index, %k: !d, %c: !d, %s: !d,
                        %u: !d, %o: !d, %x: !d, %sums: !d, %g: index,
                        %b: index, %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u
      into %o order stable every_key : !d launch(%g, %b, %one)
      scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one)
      sort stable_rank buffer %x : !d launch(%g, %b, %one)
  return
}

// Unordered: the scatter ends it, with no sort within the bins.
// CHECK-LABEL: func.func @sort_unordered(
// CHECK:         gpu.launch blocks
// CHECK:         gpu.launch blocks
// CHECK-COUNT-3: gpu.launch blocks
// CHECK:         gpu.launch blocks
// CHECK-NOT:     gpu.launch blocks
// CHECK:         return
func.func @sort_unordered(%n: index, %bins: index, %k: !d, %c: !d, %s: !d,
                          %u: !d, %o: !d, %sums: !d, %g: index, %b: index,
                          %one: index) {
  md_alg.sort_by_key %n bins %bins keys %k counts %c start %s cursor %u
      into %o order unordered every_key : !d launch(%g, %b, %one)
      scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one)
  return
}

// An append by the lanes of a warp: the ballot of the lanes that keep an
// entry, their ranks, the blocks taken by the first lane with one atomic
// at the counter's place (a constant place, a constant offset of 4 bytes
// an element) and shared by a shuffle, the entries stored below the
// capacity, and the new blocks named.
// CHECK-LABEL: func.func @append(
// CHECK:         gpu.ballot
// CHECK:         math.ctpop
// CHECK:         math.ctpop
// CHECK:         scf.if %{{.*}} -> (i32)
// CHECK:           arith.constant 8 : i64
// CHECK:           llvm.atomicrmw add %{{.*}} syncscope("device") monotonic
// CHECK:         gpu.shuffle  idx
// CHECK:         scf.if
// CHECK:           memref.store
// CHECK:         scf.if
// CHECK:           memref.store
// CHECK:           memref.store
// CHECK:         scf.if
// CHECK:           memref.store
// CHECK:           memref.store
// CHECK:         arith.select
// CHECK:         return
func.func @append(%keep: i1, %v: i32, %e: !d, %cs: i32, %csb: i32,
                  %own: i32, %un: !d, %ord: !d, %pool: memref<5xi32, 1>,
                  %cap: index, %lane: i32, %below: i32) -> (i32, i32) {
  %c2 = arith.constant 2 : index
  %count, %last = md_alg.append %keep values(%v) into(%e : !d)
      list(%cs, %csb) owner %own names(%un, %ord) pool %pool[%c2]
      capacity %cap block 64 lane(%lane, %below)
      : !d, !d, memref<5xi32, 1>
  return %count, %last : i32, i32
}

// A grid plan, the stages across the edges: the divisions, the floors, the
// bounds, the conversions.
// CHECK-LABEL: func.func @grid_plan(
// CHECK-COUNT-2: arith.divf
// CHECK-COUNT-2: math.floor
// CHECK-COUNT-2: arith.maxnumf
// CHECK-COUNT-2: arith.minnumf
// CHECK-COUNT-2: arith.fptosi
func.func @grid_plan(%lx: f64, %ly: f64, %w: f64) -> (i64, i64) {
  %x, %y = md_alg.grid_plan %lx, %ly : f64, f64 width %w : f64
      bounds [3.0 : f64, 256.0 : f64] -> i64, i64
  return %x, %y : i64, i64
}
