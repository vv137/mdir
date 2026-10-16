// What md-differentiate does not differentiate on domains yet, refused with
// the reason (D[ml2-distribute]).
// RUN: mdir-opt %s --md-distribute="grid=2,1,1" --md-differentiate -split-input-file -verify-diagnostics

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A derivative in a parameter.
// expected-error @+1 {{cannot differentiate a potential on domains in a parameter or a field yet}}
md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.divf %eps, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

func.func @f(%x: !vec, %cell: !md.cell, %eps: f64) -> f64 {
  %g = md.evaluate @lj(%x, %cell, %eps) request [derivative(2)]
      : (!vec, !md.cell, f64) -> f64
  return %g : f64
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A pair sum that reads a field computed from the positions: the next part,
// D[ml2-eam].
md.potential @many_body(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  // expected-error @+1 {{cannot differentiate a sum over pairs on domains that gathers a field computed from the positions}}
  %u = md.sum_relation %n, %x, %cell gather(%rho : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

func.func @f(%x: !vec, %cell: !md.cell) -> !vec {
  %f = md.evaluate @many_body(%x, %cell) request [forces]
      : (!vec, !md.cell) -> !vec
  return %f : !vec
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// The embedding of a density: the next part, D[ml2-eam].
md.potential @embedding(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  // expected-error @+1 {{cannot differentiate through a field on domains computed from the positions}}
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md.yield %s : f64
  } : f64
  md.return %u : f64
}

func.func @f(%x: !vec, %cell: !md.cell) -> !vec {
  %f = md.evaluate @embedding(%x, %cell) request [forces]
      : (!vec, !md.cell) -> !vec
  return %f : !vec
}
