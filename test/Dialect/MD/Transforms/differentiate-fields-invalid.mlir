// RUN: mdir-opt %s --md-differentiate -split-input-file -verify-diagnostics

// The errors of a derivative in a field of the particles
// (D230): a use of the field without a rule, named, so that
// a derivative that is not computed is never a zero.

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!grid = !md.table<2, f64>
md.particle_set @atoms

// The reciprocal sum takes the charges as its own operand: the derivative
// in them is the potential of the grid at the particles, which the op does
// not yield.
md.potential @mesh(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> f64 {
  // expected-error@+1 {{'md.reciprocal' takes argument 2 of 'mesh', a field, and has no rule for the derivative in it}}
  %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  md.return %u : f64
}

md.function @first(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> !real {
  %g = md.evaluate @mesh(%x, %cell, %q, %moduli) request [derivative(2)]
      : (!vec, !md.cell, !real, !grid) -> !real
  md.return %g : !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
md.particle_set @atoms

// A field computed from the field: the map has no rule here.
md.potential @mapped(%x: !vec, %cell: !md.cell, %q: !real) -> f64 {
  // expected-error@+1 {{'md.map_particles' takes argument 2 of 'mapped', a field, and has no rule for the derivative in it}}
  %h = md.map_particles gather(%q : !real) {
  ^bb0(%q_i: f64):
    %k = arith.mulf %q_i, %q_i : f64
    md.yield %k : f64
  } : !real
  %s = md.sum_particles gather(%h : !real) {
  ^bb0(%h_i: f64):
    md.yield %h_i : f64
  } : f64
  md.return %s : f64
}

md.function @second(%x: !vec, %cell: !md.cell, %q: !real) -> !real {
  %g = md.evaluate @mapped(%x, %cell, %q) request [derivative(2)]
      : (!vec, !md.cell, !real) -> !real
  md.return %g : !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!ids = !md.field<@atoms, i32>
md.particle_set @atoms

// A field of integers has no derivative.
md.potential @types(%x: !vec, %cell: !md.cell, %t: !ids) -> f64 {
  %u = arith.constant 0.0 : f64
  md.return %u : f64
}

md.function @third(%x: !vec, %cell: !md.cell, %t: !ids) -> !ids {
  // expected-error@+1 {{expected 'derivative' to name an argument of type f64 or a field of f64 with one component, but argument 2 has type '!md.field<@atoms, i32>'}}
  %g = md.evaluate @types(%x, %cell, %t) request [derivative(2)]
      : (!vec, !md.cell, !ids) -> !ids
  md.return %g : !ids
}
