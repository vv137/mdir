// RUN: mdir-opt %s --md-differentiate -split-input-file -verify-diagnostics

// What the derivative along directions of the arguments does not take
// (D389), each refused by name, never a zero.

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!list = !md.field<@atoms, i32>
!grid = !md.table<2, f64>
md.particle_set @atoms

// A reciprocal sum with the list of D262 whose charges are directed.
md.potential @mesh(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid,
                   %l: !list) -> f64 {
  // expected-error@+1 {{spreads a list of particles (D262), and its charges depend on the directed arguments}}
  %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli listed(%l)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid, !list -> f64, !vec, vector<9xf64>
  md.return %u : f64
}

md.function @f(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid, %l: !list,
               %dq: !real) -> f64 {
  %p = md.evaluate @mesh(%x, %cell, %q, %moduli, %l, %dq)
      request [direction] {directions = array<i64: 2>}
      : (!vec, !md.cell, !real, !grid, !list, !real) -> f64
  md.return %p : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!grid = !md.table<2, f64>
md.particle_set @atoms

// The forces of a reciprocal sum along the directions.
md.potential @mesh(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid) -> f64 {
  // expected-error@+1 {{'md.reciprocal' depends on the directed arguments and has no rule for its derivative along them}}
  %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  %s = md.sum_particles gather(%f : !vec) {
  ^bb0(%fi: vector<3xf64>):
    %e = vector.reduction <add>, %fi : vector<3xf64> into f64
    md.yield %e : f64
  } : f64
  %t = arith.addf %u, %s : f64
  md.return %t : f64
}

md.function @f(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid,
               %dq: !real) -> f64 {
  %p = md.evaluate @mesh(%x, %cell, %q, %moduli, %dq)
      request [direction] {directions = array<i64: 2>}
      : (!vec, !md.cell, !real, !grid, !real) -> f64
  md.return %p : f64
}
