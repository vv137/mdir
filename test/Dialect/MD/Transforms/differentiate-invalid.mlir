// RUN: mdir-opt %s --md-differentiate -split-input-file -verify-diagnostics

md.particle_set @atoms

md.potential @directional(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{cannot differentiate a kernel that uses the displacement}}
  %u = md.sum_relation %n, %x, %cell exchange(symmetric, asserted) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %dx = vector.extract %d[0] : f64 from vector<3xf64>
    %k = arith.mulf %dx, %dx : f64
    md.yield %k : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

md.function @caller(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> !md.field<@atoms, 3 x f64> {
  %f = md.evaluate @directional(%x, %cell) request [forces]
      : (!md.field<@atoms, 3 x f64>, !md.cell) -> !md.field<@atoms, 3 x f64>
  md.return %f : !md.field<@atoms, 3 x f64>
}

// -----

md.particle_set @atoms

md.potential @rounded(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    // expected-error@+1 {{no derivative rule for 'math.round'}}
    %k = math.round %r : f64
    md.yield %k : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

md.function @caller(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> !md.field<@atoms, 3 x f64> {
  %f = md.evaluate @rounded(%x, %cell) request [forces]
      : (!md.field<@atoms, 3 x f64>, !md.cell) -> !md.field<@atoms, 3 x f64>
  md.return %f : !md.field<@atoms, 3 x f64>
}

// -----

md.particle_set @atoms

md.potential @constant(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
                       %a: f64) -> f64 {
  // expected-error@-2 {{cannot compute forces: the energy does not depend on a sum over a relation}}
  md.return %a : f64
}

md.function @caller(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %a: f64)
    -> !md.field<@atoms, 3 x f64> {
  %f = md.evaluate @constant(%x, %cell, %a) request [forces]
      : (!md.field<@atoms, 3 x f64>, !md.cell, f64)
      -> !md.field<@atoms, 3 x f64>
  md.return %f : !md.field<@atoms, 3 x f64>
}
