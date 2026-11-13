// The refusals of the determinism requirements of md_alg (D[md-alg-ir]):
// one case for each op whose contract leaves a result free, where
// `bitwise`, the results of the serial definition on every target, is
// above its bound.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics

// A scatter leaves the order within a bin to its implementation.
!d = memref<?xi32, 1>
func.func @scatter(%keys: !d, %cursor: !d, %out: !d, %n: index) {
  // expected-error @+1 {{cannot meet the determinism requirement 'bitwise': its contract leaves the order of the items within a bin free, so it admits at most 'reproducible'}}
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out
      determinism bitwise : !d, !d, !d
  return
}

// -----

// So does an unordered counting sort, whatever its strategies: the bound
// follows the contract `order`, not the execution group.
!d = memref<?xi32, 1>
func.func @sort_unordered(%n: index, %k: !d, %c: !d, %s: !d, %u: !d,
                          %o: !d) {
  // expected-error @+1 {{cannot meet the determinism requirement 'bitwise': its contract leaves the order of the items within a bin (order unordered) free, so it admits at most 'reproducible'}}
  md_alg.sort_by_key %n bins %n keys %k counts %c start %s cursor %u
      into %o order unordered every_key : !d determinism bitwise
  return
}

// -----

// An append leaves the numbers of its blocks to the order in which
// concurrent lists take them.
!d = memref<?xi32, 1>
func.func @append(%keep: i1, %v: i32, %e: !d, %cs: i32, %un: !d, %ord: !d,
                  %pool: !d, %at: index) -> i32 {
  // expected-error @+1 {{cannot meet the determinism requirement 'bitwise': its contract leaves the numbers of the blocks free, so it admits at most 'reproducible'}}
  %r:2 = md_alg.append %keep values(%v) into(%e : !d) list(%cs, %cs)
      owner %cs names(%un, %ord) pool %pool[%at] capacity %at block 64
      determinism bitwise : !d, !d, !d
  return %r#0 : i32
}
