// What md-differentiate does not differentiate on domains yet, refused with
// the reason (D276).
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
