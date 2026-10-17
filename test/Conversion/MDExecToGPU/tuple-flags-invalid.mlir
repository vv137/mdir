// A flag of a loop over tuples that may share particles in the
// deterministic mode (D278).
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu="deterministic=true" \
// RUN:     -verify-diagnostics

md.particle_set @atoms

// Over tuples that may share particles, the deterministic mode evaluates
// each tuple by each of its members, in rows that groups of threads share,
// which take no flag.
//
func.func @shared(%pairs: memref<?x?xi32, 1>, %x: memref<?x3xf64, 1>,
                  %cell: !md.cell) -> i1 {
  %no = arith.constant false
  // expected-error@+1 {{takes a flag in 'reduce' over tuples that may share particles}}
  %any = md_exec.tuple_for %pairs, %x, %cell coordinates(displacement(0, 1))
      reduce(%no : i1) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %sq = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %limit = arith.constant 1.0 : f64
    %far = arith.cmpf ugt, %r2, %limit : f64
    md_exec.yield %far : i1
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1> -> i1
  return %any : i1
}
