// RUN: mdir-opt %s -split-input-file -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>

md.function @f(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(1.5)
         : !vec -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected a cutoff of at most 1.500000e+00, that of the neighborhood, got 2.000000e+00}}
  %t = md.triplets %n cutoff(2.0)
         : !md.relation<@atoms, 2, unordered> -> !md.relation<@atoms, 3, reversal>
  md.return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

md.function @f(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(1.5)
         : !vec -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected a positive cutoff, got 0.000000e+00}}
  %t = md.triplets %n cutoff(0.0)
         : !md.relation<@atoms, 2, unordered> -> !md.relation<@atoms, 3, reversal>
  md.return
}

// -----

md.function @f(%b: !md.relation<@atoms, 2, unordered, @bonds>) {
  // expected-error@+1 {{expected the unordered relation of arity 2 of a neighborhood, got '!md.relation<@atoms, 2, unordered, @bonds>'}}
  %t = md.triplets %b cutoff(1.0)
         : !md.relation<@atoms, 2, unordered, @bonds> -> !md.relation<@atoms, 3, reversal>
  md.return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

md.function @f(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(1.5)
         : !vec -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected the result to have type '!md.relation<@atoms, 3, reversal>', got '!md.relation<@atoms, 3, ordered>'}}
  %t = md.triplets %n cutoff(1.0)
         : !md.relation<@atoms, 2, unordered> -> !md.relation<@atoms, 3, ordered>
  md.return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

// A relation without a tuple set that md.triplets does not give.
md.function @f(%x: !vec, %cell: !md.cell,
               %t: !md.relation<@atoms, 3, reversal>) -> f64 {
  // expected-error@+1 {{expected the relation of a tuple set or of 'md.triplets', got '!md.relation<@atoms, 3, reversal>'}}
  %u = md.sum_tuples %t, %x, %cell coordinates(cosine(0, 1, 2)) {
  ^bb0(%c: f64):
    md.yield %c : f64
  } : !md.relation<@atoms, 3, reversal>, !vec -> f64
  md.return %u : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>

// The triplets have no fields of their own.
md.function @f(%x: !vec, %cell: !md.cell, %k: !md.field<@angles, f64>) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5)
         : !vec -> !md.relation<@atoms, 2, unordered>
  %t = md.triplets %n cutoff(1.0)
         : !md.relation<@atoms, 2, unordered> -> !md.relation<@atoms, 3, reversal>
  // expected-error@+1 {{takes no fields in 'tuple': the triplets have no tuple set}}
  %u = md.sum_tuples %t, %x, %cell coordinates(cosine(0, 1, 2))
         tuple(%k : !md.field<@angles, f64>) {
  ^bb0(%c: f64, %k_t: f64):
    md.yield %c : f64
  } : !md.relation<@atoms, 3, reversal>, !vec -> f64
  md.return %u : f64
}
