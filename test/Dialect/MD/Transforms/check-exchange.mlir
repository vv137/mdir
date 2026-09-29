// RUN: mdir-opt %s --md-check-exchange -split-input-file -verify-diagnostics

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

// Contracts that can be proven.
md.function @proven(%x: !vec, %cell: !md.cell, %eps: !real, %sigma: !real,
                    %a: f64) -> (f64, f64, !vec) {
  %n = md.neighborhood %x, %cell cutoff(1.0) : !vec -> !pairs

  // A function of the distance and of values from outside the kernel.
  %u0 = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.divf %a, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64

  // Combining rules: commutative ops are compared in either order.
  %u1 = md.sum_relation %n, %x, %cell gather(%eps, %sigma : !real, !real)
          exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>,
       %eps1: f64, %eps2: f64, %sigma1: f64, %sigma2: f64):
    %ssum = arith.addf %sigma1, %sigma2 : f64
    %eprd = arith.mulf %eps1, %eps2 : f64
    %e    = math.sqrt %eprd : f64
    %sr   = arith.divf %ssum, %r : f64
    %k    = arith.mulf %e, %sr : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64

  // A symmetric factor times the displacement is antisymmetric.
  %f = md.gather_relation %n, %x, %cell exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %g  = arith.divf %a, %r : f64
    %gv = vector.broadcast %g : f64 to vector<3xf64>
    %k  = arith.mulf %gv, %d : vector<3xf64>
    md.yield %k : vector<3xf64>
  } : !pairs, !vec -> !vec

  md.return %u0, %u1, %f : f64, f64, !vec
}

// -----

md.particle_set @atoms

// The difference of the two parameters changes sign.
md.function @difference(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
                        %q: !md.field<@atoms, f64>) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+2 {{cannot prove that the kernel is symmetric under exchange of the two particles}}
  // expected-note@+1 {{if the contract holds, state it as 'exchange(symmetric, asserted)'}}
  %u = md.sum_relation %n, %x, %cell gather(%q : !md.field<@atoms, f64>)
         exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %q1: f64, %q2: f64):
    %k = arith.subf %q1, %q2 : f64
    md.yield %k : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.particle_set @atoms

// The same kernel is accepted when the front end asserts the contract.
md.function @asserted(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
                      %q: !md.field<@atoms, f64>) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  %u = md.sum_relation %n, %x, %cell gather(%q : !md.field<@atoms, f64>)
         exchange(symmetric, asserted) {
  ^bb0(%r: f64, %d: vector<3xf64>, %q1: f64, %q2: f64):
    %k = arith.subf %q1, %q2 : f64
    md.yield %k : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.particle_set @atoms

// A force kernel that is symmetric cannot be antisymmetric.
md.function @wrong_sign(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> !md.field<@atoms, 3 x f64> {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+2 {{cannot prove that the kernel is antisymmetric under exchange of the two particles}}
  // expected-note@+1 {{if the contract holds, state it as 'exchange(antisymmetric, asserted)'}}
  %f = md.gather_relation %n, %x, %cell exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = vector.broadcast %r : f64 to vector<3xf64>
    md.yield %k : vector<3xf64>
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64>
      -> !md.field<@atoms, 3 x f64>
  md.return %f : !md.field<@atoms, 3 x f64>
}
