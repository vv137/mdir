// RUN: mdir-opt %s --md-check-derivative-coverage --allow-unregistered-dialect -verify-diagnostics -o /dev/null
!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
md.particle_set @atoms
md.potential @missing(%x: !vec, %cell: !md.cell, %q: !real) -> f64 {
  %sum = md.sum_particles gather(%q : !real) {
  ^bb0(%qi: f64):
    // expected-error@+1 {{no derivative rule or declared zero for 'foo.kernel' in potential 'missing'}}
    %unknown = "foo.kernel"(%qi) : (f64) -> f64
    md.yield %unknown : f64
  } : f64
  md.return %sum : f64
}
