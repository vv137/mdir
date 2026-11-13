// The determinism requirements that the SIMT lowering of md_alg refuses
// (D408): those that the verifier accepts but that its code, whose
// atomics fix no order, does not meet. The module is left as it was. The
// serial lowering, which runs the items in order, meets them.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics \
// RUN:     --convert-md-alg-to-gpu
// RUN: mdir-opt %s --split-input-file --convert-md-alg-to-scf \
// RUN: | FileCheck %s --check-prefix=SERIAL --implicit-check-not=md_alg.

!d = memref<?xi32, 1>
// SERIAL-LABEL: func.func @scatter(
func.func @scatter(%keys: !d, %cursor: !d, %out: !d, %n: index, %g: index,
                   %b: index, %one: index) {
  // expected-error @+1 {{cannot meet the determinism requirement 'reproducible' in its SIMT lowering, whose atomics fix no order; it meets 'unordered'}}
  md_alg.scatter_to_bins %keys[%n] cursor %cursor into %out
      launch(%g, %b, %one) determinism reproducible : !d, !d, !d
  return
}

// -----

!d = memref<?xi32, 1>
// SERIAL-LABEL: func.func @sort_unordered(
func.func @sort_unordered(%n: index, %k: !d, %c: !d, %s: !d, %u: !d, %o: !d,
                          %sums: !d, %g: index, %b: index, %one: index) {
  // expected-error @+1 {{cannot meet the determinism requirement 'reproducible' in its SIMT lowering, whose atomics fix no order; it meets 'unordered'}}
  md_alg.sort_by_key %n bins %n keys %k counts %c start %s cursor %u
      into %o order unordered every_key : !d launch(%g, %b, %one)
      scan reduce_then_scan scratch %sums : !d launch(%g, %b, %one)
      determinism reproducible
  return
}

// -----

!d = memref<?xi32, 1>
// SERIAL-LABEL: func.func @append(
func.func @append(%keep: i1, %v: i32, %e: !d, %cs: i32, %un: !d, %ord: !d,
                  %pool: !d, %at: index) -> i32 {
  // expected-error @+1 {{cannot meet the determinism requirement 'reproducible' in its SIMT lowering, whose atomics fix no order; it meets 'unordered'}}
  %r:2 = md_alg.append %keep values(%v) into(%e : !d) list(%cs, %cs)
      owner %cs names(%un, %ord) pool %pool[%at] capacity %at block 64
      lane(%cs, %cs) determinism reproducible : !d, !d, !d
  return %r#0 : i32
}
