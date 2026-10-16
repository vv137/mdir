// RUN: mdir-opt %s -split-input-file -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>

md.function @f(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(0.7)
         : !vec -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected a cutoff of at most 7.000000e-01, that of the neighborhood, got 8.000000e-01}}
  %e = md.edges %n, %x, %cell cutoff(0.8)
         : !md.relation<@atoms, 2, unordered>, !vec -> !md.edges<@atoms>
  md.return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

md.function @f(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(0.7)
         : !vec -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected a positive cutoff, got 0.000000e+00}}
  %e = md.edges %n, %x, %cell cutoff(0.0)
         : !md.relation<@atoms, 2, unordered>, !vec -> !md.edges<@atoms>
  md.return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

md.function @f(%x: !vec, %y: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(0.7)
         : !vec -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected the positions and the cell of the neighborhood}}
  %e = md.edges %n, %y, %cell cutoff(0.7)
         : !md.relation<@atoms, 2, unordered>, !vec -> !md.edges<@atoms>
  md.return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

md.function @f(%x: !vec, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>) {
  %n = md.neighborhood %x, %cell cutoff(0.7)
         exclude(%b : !md.relation<@atoms, 2, unordered, @bonds>)
         : !vec -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected a neighborhood without excluded pairs: the edges take every pair}}
  %e = md.edges %n, %x, %cell cutoff(0.7)
         : !md.relation<@atoms, 2, unordered>, !vec -> !md.edges<@atoms>
  md.return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

md.function @f(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(0.7)
         : !vec -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected the positions and the result on @atoms, the set of the pairs}}
  %e = md.edges %n, %x, %cell cutoff(0.7)
         : !md.relation<@atoms, 2, unordered>, !vec -> !md.edges<@water>
  md.return
}

// -----

md.function @f(%b: !md.relation<@atoms, 2, unordered, @bonds>,
               %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{expected the unordered relation of arity 2 of a neighborhood, got '!md.relation<@atoms, 2, unordered, @bonds>'}}
  %e = md.edges %b, %x, %cell cutoff(0.7)
         : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
           -> !md.edges<@atoms>
  md.return
}
