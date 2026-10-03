// RUN: mdir-opt %s --md-differentiate -split-input-file --allow-unregistered-dialect -verify-diagnostics

// The two errors of a derivative with respect to a parameter (D161): a
// value whose independence of the parameter cannot be proved, and a value
// that depends on it without a rule for its derivative, each named.

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// An op that the pass does not know may read anything: its independence
// cannot be proved, though no operand of it is %a.
md.potential @undetermined(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  // expected-error@+1 {{cannot prove that this value does not depend on argument 2 of 'undetermined': 'foo.read' is not an op whose dependences the pass knows}}
  %v = "foo.read"() : () -> f64
  %u = arith.mulf %v, %a : f64
  md.return %u : f64
}

md.function @third(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %d = md.evaluate @undetermined(%x, %cell, %a) request [derivative(2)]
      : (!vec, !md.cell, f64) -> f64
  md.return %d : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// An op that takes %a and has no rule for its derivative.
md.potential @norule(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  // expected-error@+1 {{'foo.opaque' depends on argument 2 of 'norule' and has no rule for its derivative}}
  %v = "foo.opaque"(%a) : (f64) -> f64
  md.return %v : f64
}

md.function @fourth(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %d = md.evaluate @norule(%x, %cell, %a) request [derivative(2)]
      : (!vec, !md.cell, f64) -> f64
  md.return %d : f64
}
