// RUN: mdir-opt %s --md-differentiate -split-input-file -verify-diagnostics

// What the tangent in the positions does not take yet (D376,
// D379), each refused by name, never a zero.

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!grid = !md.table<2, f64>
md.particle_set @atoms

// The derivative of the tangent of a reciprocal sum in the charge of each
// particle: the per-particle form, not formed (D379).
md.potential @mesh(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> f64 {
  // expected-error@+1 {{the derivative of the tangent of a reciprocal sum in the value of each particle of argument 2 of 'mesh.tangent' is not formed}}
  %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  md.return %u : f64
}

md.function @f(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid,
               %t: !vec) -> !real {
  %g = md.evaluate @mesh(%x, %cell, %q, %moduli, %t)
      request [tangent_derivative(2)]
      : (!vec, !md.cell, !real, !grid, !vec) -> !real
  md.return %g : !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!sca = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A field computed from the positions (a density of many-body terms).
md.potential @eam(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  // expected-error@+1 {{cannot form the tangent in the positions of a field computed from them}}
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !pairs, !vec -> !sca
  %e = md.sum_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    md.yield %a : f64
  } : f64
  md.return %e : f64
}

md.function @f(%x: !vec, %cell: !md.cell, %t: !vec) -> f64 {
  %p = md.evaluate @eam(%x, %cell, %t) request [tangent]
      : (!vec, !md.cell, !vec) -> f64
  md.return %p : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A pair kernel that reads the displacement, as for the forces.
md.potential @aniso(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  // expected-error@+1 {{cannot form the tangent of a kernel that uses the displacement}}
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %z = vector.extract %d[2] : f64 from vector<3xf64>
    %k = arith.mulf %z, %z : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

md.function @f(%x: !vec, %cell: !md.cell, %t: !vec) -> f64 {
  %p = md.evaluate @aniso(%x, %cell, %t) request [tangent]
      : (!vec, !md.cell, !vec) -> f64
  md.return %p : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!grid = !md.table<2, f64>
md.particle_set @atoms

// The forces of the tangent of a reciprocal sum: not formed yet
// (D[force-matching-hvp], #771).
md.potential @mesh(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> f64 {
  // expected-error@+1 {{the derivative in the positions of the tangent of a reciprocal sum is not formed yet}}
  %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  md.return %u : f64
}

md.function @f(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid,
               %t: !vec) -> !vec {
  %g = md.evaluate @mesh(%x, %cell, %q, %moduli, %t) request [tangent_forces]
      : (!vec, !md.cell, !real, !grid, !vec) -> !vec
  md.return %g : !vec
}
