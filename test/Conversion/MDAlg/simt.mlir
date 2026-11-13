// The SIMT lowering of md_alg (convert-md-alg-to-gpu, D[md-alg]): each
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
