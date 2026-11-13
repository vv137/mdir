// The refusals of the SIMT lowering of md_alg (D406): a collection
// op without the execution parameters that its emitter takes, which no
// policy has filled in yet. The module is left as it was.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics \
// RUN:     --convert-md-alg-to-gpu

!d = memref<?xi32, 1>
func.func @scan(%counts: !d, %start: !d, %n: index) {
  // expected-error @+1 {{has no execution parameters for its SIMT lowering}}
  md_alg.scan %counts[%n] into %start : !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @scatter(%keys: !d, %cursor: !d, %out: !d, %n: index) {
  // expected-error @+1 {{has no execution parameters for its SIMT lowering}}
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @sort(%start: !d, %items: !d, %n: index) {
  // expected-error @+1 {{has no execution parameters for its SIMT lowering}}
  md_alg.sort_bins %start[%n] items %items : !d, !d
  return
}

// -----

// The stage ops (D[md-alg-stages]): a sort without its execution group,
// an append without its lane, and an append whose blocks are shorter than
// a warp's batch, which could begin more than two of them.
!d = memref<?xi32, 1>
func.func @sort_by_key(%n: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d) {
  // expected-error @+1 {{has no execution parameters for its SIMT lowering}}
  md_alg.sort_by_key %n bins %n keys %k counts %c start %s cursor %u
      into %o order stable every_key : !d
  return
}

// -----

!d = memref<?xi32, 1>
func.func @append(%keep: i1, %v: i32, %e: !d, %cs: i32, %un: !d, %ord: !d,
                  %pool: !d, %at: index) -> i32 {
  // expected-error @+1 {{has no execution parameters for its SIMT lowering}}
  %r:2 = md_alg.append %keep values(%v) into(%e : !d) list(%cs, %cs)
      owner %cs names(%un, %ord) pool %pool[%at] capacity %at block 64
      : !d, !d, !d
  return %r#0 : i32
}

// -----

!d = memref<?xi32, 1>
func.func @append_block(%keep: i1, %v: i32, %e: !d, %cs: i32, %un: !d,
                        %ord: !d, %pool: !d, %at: index) -> i32 {
  // expected-error @+1 {{takes blocks of at least 32 entries in its SIMT lowering, a warp's batch beginning at most two; not 16}}
  %r:2 = md_alg.append %keep values(%v) into(%e : !d) list(%cs, %cs)
      owner %cs names(%un, %ord) pool %pool[%at] capacity %at block 16
      lane(%cs, %cs) : !d, !d, !d
  return %r#0 : i32
}
