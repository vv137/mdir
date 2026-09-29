// RUN: mdir-opt %s --convert-md-to-md-exec -split-input-file -verify-diagnostics

md.particle_set @atoms

md.potential @u(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5)
      : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{cannot convert a sum with a truncation; run 'md-expand-truncation' first}}
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) truncation(shift) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.particle_set @atoms

md.potential @u(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  md.return %zero : f64
}

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  // expected-error@+1 {{cannot be converted; run 'md-differentiate' first}}
  %u = md.evaluate @u(%x, %cell) request [energy]
      : (!md.field<@atoms, 3 x f64>, !md.cell) -> f64
  md.return %u : f64
}

// -----

md.particle_set @atoms

// A relation that is passed in has no cutoff.
md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %bonds: !md.relation<@atoms, 2, unordered>) -> f64 {
  // expected-error@+1 {{cannot convert a loop over a relation that is not the result of 'md.neighborhood'}}
  %u = md.sum_relation %bonds, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}
