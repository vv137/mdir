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
