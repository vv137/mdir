// The shared analysis of the collections of md_alg (D[md-alg-analysis],
// #866), printed by the test pass: the order that each consumer requires,
// joined over the consumers and compared with what each producer
// guarantees, and the verdict on each sort. A sort is redundant only when
// every consumer's requirement is met without it and every consumer
// combines exactly or requires only `unordered`; every use that the
// analysis cannot read as a declared md_alg reader requires the layout.
//
// RUN: mdir-opt %s --split-input-file --allow-unregistered-dialect \
// RUN:     --test-md-alg-print-collections -o /dev/null | FileCheck %s

!b = memref<?xi32>

// A counting sort whose one reader visits its bins in any order and sums
// integers: the order within the bins is not observed, and the sort is
// redundant. The facts of the order, its root, and its lifetime.
// CHECK-LABEL: md_alg collections of @accessor_only
// CHECK:      collection out of md_alg.sort_by_key at 5: %alloc_2
// CHECK-NEXT:   root: %alloc_2, allocated at 4, aliases 0
// CHECK-NEXT:   escapes: no
// CHECK-NEXT:   guarantees: stable(%alloc_0, item)
// CHECK-NEXT:   use: md_alg.bin_members at 7: read, requires grouped_by(%alloc_0), streamed (visits the bins in any order)
// CHECK-NEXT:   use: memref.dealloc at 10: release
// CHECK-NEXT:   requires: grouped_by(%alloc_0)
// CHECK-NEXT:   access: streamed
// CHECK-NEXT:   cardinality: run-time %arg0
// CHECK-NEXT:   lifetime: produced at 5, last use at 7, released at 10
// CHECK-NEXT:   determinism: bitwise
// CHECK-NEXT: sort md_alg.sort_by_key at 5: redundant: requires grouped_by(%alloc_0), without the sort grouped_by(%alloc_0), with it stable(%alloc_0, item){{$}}
func.func @accessor_only(%n: index, %bins: index, %keys: !b, %from: index,
                         %to: index) -> i32 {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  %zero = arith.constant 0 : i32
  %sum = md_alg.bin_members %start[%from, %to] items %out visit unordered
      init(%zero : i32) combine ["addi"] : !b, !b {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    md_alg.yield %v : i32
  } -> i32
  memref.dealloc %out : !b
  return %sum : i32
}

// -----

!b = memref<?xi32>

// The same sort read by index arithmetic (a pair kernel's loads): the
// layout, so the sort is not redundant. Rule 1.
// CHECK-LABEL: md_alg collections of @raw_consumer
// CHECK:      collection out of md_alg.sort_by_key
// CHECK:        use: memref.load at {{[0-9]+}}: read, requires layout, random (a read by index)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout, without the sort grouped_by(%alloc_0), with it stable(%alloc_0, item); the order is not met without it; a consumer does not combine exactly
func.func @raw_consumer(%n: index, %bins: index, %keys: !b, %i: index)
    -> i32 {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  %v = memref.load %out[%i] : !b
  return %v : i32
}

// -----

!b = memref<?xi32>

// A read through a followed view (a subview) is still a read by index:
// the layout. A read by a declared reader through a cast to another
// memory space (a followed view) keeps the reader's requirement.
// CHECK-LABEL: md_alg collections of @views
// CHECK:      collection out of md_alg.sort_by_key at {{[0-9]+}}: %alloc_2
// CHECK-NEXT:   root: %alloc_2, allocated at {{[0-9]+}}, aliases 2
// CHECK:        use: md_alg.bin_members at {{[0-9]+}}: read, requires grouped_by(%alloc_0), streamed
// CHECK-NEXT:   use: memref.load at {{[0-9]+}}: read, requires layout, random (a read by index)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @views(%n: index, %bins: index, %keys: !b, %from: index,
                 %to: index, %i: index) -> i32 {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  %far = memref.memory_space_cast %out : !b to memref<?xi32, 1>
  %zero = arith.constant 0 : i32
  %sum = md_alg.bin_members %start[%from, %to] items %far visit unordered
      init(%zero : i32) combine ["addi"] : !b, memref<?xi32, 1> {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    md_alg.yield %v : i32
  } -> i32
  %c1 = arith.constant 1 : index
  %part = memref.subview %out[%c1] [%c1] [%c1] : !b to memref<?xi32, strided<[?], offset: ?>>
  %w = memref.load %part[%i] : memref<?xi32, strided<[?], offset: ?>>
  %r = arith.addi %sum, %w : i32
  return %r : i32
}

// -----

!b = memref<?xi32>

// Escapes, each the layout (rule 3): the order written into an argument;
// passed to a call; returned; its address taken; forwarded by a view that
// the analysis does not follow (expand_shape); used by an op whose effects
// are unknown (rule 4).
// CHECK-LABEL: md_alg collections of @escape_argument
// CHECK:      collection out of md_alg.sort_by_key at {{[0-9]+}}: %arg5
// CHECK:        escapes: an argument of the function
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @escape_argument(%n: index, %bins: index, %keys: !b, %from: index,
                           %to: index, %out: !b) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  md_alg.bin_members %start[%from, %to] items %out visit unordered : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  return
}

// -----

!b = memref<?xi32>

func.func private @take(!b)

// CHECK-LABEL: md_alg collections of @escape_call
// CHECK:      collection out of md_alg.sort_by_key
// CHECK:        escapes: passed to a call
// CHECK:        use: func.call at {{[0-9]+}}: escape, requires layout (passed to a call)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @escape_call(%n: index, %bins: index, %keys: !b) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  func.call @take(%out) : (!b) -> ()
  return
}

// -----

!b = memref<?xi32>

// CHECK-LABEL: md_alg collections of @escape_return
// CHECK:      collection out of md_alg.sort_by_key
// CHECK:        escapes: returned
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @escape_return(%n: index, %bins: index, %keys: !b) -> !b {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  return %out : !b
}

// -----

!b = memref<?xi32>

// CHECK-LABEL: md_alg collections of @escape_address
// CHECK:      collection out of md_alg.sort_by_key
// CHECK:        escapes: its address taken
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @escape_address(%n: index, %bins: index, %keys: !b) -> index {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  %p = memref.extract_aligned_pointer_as_index %out : !b -> index
  return %p : index
}

// -----

!b = memref<?xi32>

// CHECK-LABEL: md_alg collections of @escape_unfollowed_view
// CHECK:      collection out of md_alg.sort_by_key
// CHECK:        escapes: forwarded by an op that is not a followed view
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @escape_unfollowed_view(%n: index, %bins: index, %keys: !b)
    -> memref<?x1xi32> {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  %e = memref.expand_shape %out [[0, 1]] output_shape [%n, 1]
      : !b into memref<?x1xi32>
  return %e : memref<?x1xi32>
}

// -----

!b = memref<?xi32>

// CHECK-LABEL: md_alg collections of @escape_unknown_effects
// CHECK:      collection out of md_alg.sort_by_key
// CHECK:        escapes: an op whose effects are unknown
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @escape_unknown_effects(%n: index, %bins: index, %keys: !b) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  "test.use"(%out) : (!b) -> ()
  return
}

// -----

!b = memref<?xi32>

// A write by an op that is not md_alg after the sort (rule 2): the layout.
// CHECK-LABEL: md_alg collections of @raw_writer
// CHECK:        use: memref.store at {{[0-9]+}}: write, requires layout (a write by index)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @raw_writer(%n: index, %bins: index, %keys: !b, %from: index,
                      %to: index, %i: index, %v: i32) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  memref.store %v, %out[%i] : !b
  md_alg.bin_members %start[%from, %to] items %out visit unordered : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  return
}

// -----

!b = memref<?xi32>

// The requirement kinds of the reader: sorted by a key, stable by a key,
// and sorted by the item, which is stable by the item. A sort by `by`
// meets the first two; without it, neither is met.
// CHECK-LABEL: md_alg collections of @kinds
// CHECK:      collection out of md_alg.sort_by_key at {{[0-9]+}}: %alloc_2
// CHECK:        guarantees: stable(%alloc_0, %arg3)
// CHECK-NEXT:   use: md_alg.bin_members at {{[0-9]+}}: read, requires sorted_within(%alloc_0, %arg3), streamed (visits the bins sorted)
// CHECK-NEXT:   use: md_alg.bin_members at {{[0-9]+}}: read, requires stable(%alloc_0, %arg3), streamed (visits the bins stably)
// CHECK-NEXT:   requires: stable(%alloc_0, %arg3)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires stable(%alloc_0, %arg3), without the sort grouped_by(%alloc_0), with it stable(%alloc_0, %arg3); the order is not met without it{{$}}
// CHECK:      collection out of md_alg.sort_by_key at {{[0-9]+}}: %alloc_6
// CHECK-NEXT:   root: %alloc_6
// CHECK-NEXT:   escapes: no
// CHECK-NEXT:   guarantees: stable(%alloc_4, item)
// CHECK-NEXT:   use: md_alg.bin_members at {{[0-9]+}}: read, requires stable(%alloc_4, item), streamed (visits the bins sorted)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires stable(%alloc_4, item), without the sort grouped_by(%alloc_4), with it stable(%alloc_4, item); the order is not met without it{{$}}
func.func @kinds(%n: index, %bins: index, %keys: !b, %by: !b, %from: index,
                 %to: index) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable by %by : !b every_key : !b
  md_alg.bin_members %start[%from, %to] items %out visit sorted by %by : !b
      : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  md_alg.bin_members %start[%from, %to] items %out visit stable by %by : !b
      : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  %counts2 = memref.alloc(%bins) : !b
  %start2 = memref.alloc(%bins) : !b
  %cursor2 = memref.alloc(%bins) : !b
  %out2 = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts2 start %start2
      cursor %cursor2 into %out2 order stable every_key : !b
  md_alg.bin_members %start2[%from, %to] items %out2 visit sorted : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  return
}

// -----

!b = memref<?xi32>

// Keys that do not match (rule 5): a reader sorted by another key than the
// sort's, and a reader of the groups of another sort's offsets. Neither is
// met with the sort or without it: not redundant, and not sufficient.
// CHECK-LABEL: md_alg collections of @key_mismatch
// CHECK:      collection out of md_alg.sort_by_key at {{[0-9]+}}: %alloc_2
// CHECK:        use: md_alg.bin_members at {{[0-9]+}}: read, requires sorted_within(%alloc_0, %arg4)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires sorted_within(%alloc_0, %arg4), without the sort grouped_by(%alloc_0), with it stable(%alloc_0, %arg3) (not sufficient); the order is not met without it{{$}}
// CHECK:      collection out of md_alg.sort_by_key at {{[0-9]+}}: %alloc_6
// CHECK:        use: md_alg.bin_members at {{[0-9]+}}: read, requires grouped_by(%alloc_0)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires grouped_by(%alloc_0), without the sort grouped_by(%alloc_4), with it stable(%alloc_4, item) (not sufficient); the order is not met without it{{$}}
func.func @key_mismatch(%n: index, %bins: index, %keys: !b, %by: !b,
                        %other: !b, %from: index, %to: index) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable by %by : !b every_key : !b
  md_alg.bin_members %start[%from, %to] items %out visit sorted by %other
      : !b : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  %counts2 = memref.alloc(%bins) : !b
  %start2 = memref.alloc(%bins) : !b
  %cursor2 = memref.alloc(%bins) : !b
  %out2 = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts2 start %start2
      cursor %cursor2 into %out2 order stable every_key : !b
  md_alg.bin_members %start[%from, %to] items %out2 visit unordered : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  return
}

// -----

!b = memref<?xi32>

// Two readers whose requirements join: grouped_by(G) and stable(G, item)
// give stable(G, item); grouped_by of two different offsets give the
// layout, since the analysis does not know whether one refines the other.
// CHECK-LABEL: md_alg collections of @join
// CHECK:      collection out of md_alg.sort_by_key at {{[0-9]+}}: %alloc_2
// CHECK:        requires: stable(%alloc_0, item)
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires stable(%alloc_0, item)
// CHECK:      collection out of md_alg.sort_by_key at {{[0-9]+}}: %alloc_6
// CHECK:        use: md_alg.bin_members at {{[0-9]+}}: read, requires grouped_by(%alloc_4)
// CHECK-NEXT:   use: md_alg.bin_members at {{[0-9]+}}: read, requires grouped_by(%alloc_0)
// CHECK-NEXT:   requires: layout
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @join(%n: index, %bins: index, %keys: !b, %from: index,
                %to: index) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  md_alg.bin_members %start[%from, %to] items %out visit unordered : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  md_alg.bin_members %start[%from, %to] items %out visit stable : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  %counts2 = memref.alloc(%bins) : !b
  %start2 = memref.alloc(%bins) : !b
  %cursor2 = memref.alloc(%bins) : !b
  %out2 = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts2 start %start2
      cursor %cursor2 into %out2 order stable every_key : !b
  md_alg.bin_members %start2[%from, %to] items %out2 visit unordered : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  md_alg.bin_members %start[%from, %to] items %out2 visit unordered : !b, !b {
  ^bb0(%item: index):
    md_alg.yield
  }
  return
}

// -----

!b = memref<?xi32>

// The determinism axis: a sort within bins by the item after a counting
// sort that already ordered each bin by the item. Its reader's order,
// stable(G, item), is met without it; but the reader sums floats under
// `bitwise`, so the sort is not redundant. Under `unordered` it is.
// CHECK-LABEL: md_alg collections of @float_combination
// CHECK:      collection items of md_alg.sort_bins at {{[0-9]+}}: %alloc_2
// CHECK:        use: md_alg.bin_members at {{[0-9]+}}: read, requires stable(%alloc_0, item), streamed (visits the bins stably, combines floats under 'bitwise')
// CHECK:      sort md_alg.sort_bins at {{[0-9]+}}: not redundant: requires stable(%alloc_0, item), without the sort stable(%alloc_0, item), with it stable(%alloc_0, item); a consumer does not combine exactly{{$}}
// CHECK:      collection items of md_alg.sort_bins at {{[0-9]+}}: %alloc_2
// CHECK:      sort md_alg.sort_bins at {{[0-9]+}}: redundant: requires stable(%alloc_0, item), without the sort stable(%alloc_0, item), with it stable(%alloc_0, item){{$}}
func.func @float_combination(%n: index, %bins: index, %keys: !b,
                             %from: index, %to: index) -> (f32, f32) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
      cursor %cursor into %out order stable every_key : !b
  md_alg.sort_bins %start[%bins] items %out : !b, !b
  %zero = arith.constant 0.0 : f32
  %s = md_alg.bin_members %start[%from, %to] items %out visit stable
      init(%zero : f32) combine ["addf"] : !b, !b {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    %f = arith.sitofp %v : i32 to f32
    md_alg.yield %f : f32
  } -> f32
  md_alg.sort_bins %start[%bins] items %out : !b, !b
  %t = md_alg.bin_members %start[%from, %to] items %out visit stable
      init(%zero : f32) combine ["addf"] determinism unordered : !b, !b {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    %f = arith.sitofp %v : i32 to f32
    md_alg.yield %f : f32
  } -> f32
  return %s, %t : f32, f32
}

// -----

!b = memref<?xi32>

// A producer in a loop: a use before it in the loop reads what the
// previous iteration wrote, so it counts.
// CHECK-LABEL: md_alg collections of @loop
// CHECK:      collection out of md_alg.sort_by_key
// CHECK:        use: memref.load at {{[0-9]+}}: read, requires layout
// CHECK:      sort md_alg.sort_by_key at {{[0-9]+}}: not redundant: requires layout
func.func @loop(%n: index, %bins: index, %keys: !b, %i: index) {
  %counts = memref.alloc(%bins) : !b
  %start = memref.alloc(%bins) : !b
  %cursor = memref.alloc(%bins) : !b
  %out = memref.alloc(%n) : !b
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %round = %c0 to %c4 step %c1 {
    %v = memref.load %out[%i] : !b
    memref.store %v, %keys[%i] : !b
    md_alg.sort_by_key %n bins %bins keys %keys counts %counts start %start
        cursor %cursor into %out order stable every_key : !b
  }
  return
}

// -----

!b = memref<?xi32>

// Membership: the guarantee of an append's lists, whose blocks are free.
// A scatter's order is grouped by the offsets of the scan whose cursor it
// takes; the cardinalities exact, bound, and run-time.
// CHECK-LABEL: md_alg collections of @others
// CHECK:      collection start of md_alg.scan at {{[0-9]+}}: %alloc
// CHECK:        cardinality: exact 9
// CHECK:      collection out of md_alg.scatter_to_bins at {{[0-9]+}}: %alloc_1
// CHECK-NEXT:   root: %alloc_1
// CHECK-NEXT:   escapes: no
// CHECK-NEXT:   guarantees: grouped_by(%alloc)
// CHECK:        cardinality: bound %arg0
// CHECK:        determinism: unordered
// CHECK:      collection units of md_alg.append at {{[0-9]+}}: %arg3
// CHECK:        guarantees: membership
// CHECK:        cardinality: bound %arg6
func.func @others(%n: index, %keys: !b, %counts: !b, %units: !b,
                  %ordinals: !b, %pool: !b, %capacity: index, %into: !b,
                  %keep: i1, %v: i32) {
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %start = memref.alloc(%c9) : !b
  %cursor = memref.alloc(%c8) : !b
  %order = memref.alloc(%n) : !b
  md_alg.scan %counts[%c8] into %start cursor %cursor : !b, !b, !b
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %order : !b, !b, !b
  %c0 = arith.constant 0 : index
  %z = arith.constant 0 : i32
  %count, %last = md_alg.append %keep values(%v) into(%into : !b)
      list(%z, %z) owner %z names(%units, %ordinals) pool %pool[%c0]
      capacity %capacity block 64 : !b, !b, !b
  return
}
