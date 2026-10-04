// RUN: mdir-opt %s --md-analyze-activity=argument=2 --allow-unregistered-dialect -verify-diagnostics -o /dev/null
!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
md.particle_set @atoms
md.potential @activity(%x: !vec, %cell: !md.cell, %lambda: f64, %q: !real) -> f64 {
  // expected-remark@+1 {{activity: proven inactive}}
  %one = arith.constant 1.0 : f64
  // expected-remark@+1 {{activity: active}}
  %twice = arith.addf %lambda, %lambda : f64
  // expected-remark@+1 {{activity: proven inactive}}
  %independent = md.sum_particles gather(%q : !real) {
  ^bb0(%qi: f64):
    md.yield %qi : f64
  } : f64
  // The capture, rather than an operand, is the only path from lambda.
  // expected-remark@+1 {{activity: active}}
  %captured = md.map_particles gather(%q : !real) {
  ^bb0(%qi: f64):
    %scaled = arith.mulf %qi, %twice : f64
    md.yield %scaled : f64
  } : !real
  // expected-remark@+1 {{activity: active}}
  %sum = md.sum_particles gather(%captured : !real) {
  ^bb0(%qi: f64):
    md.yield %qi : f64
  } : f64
  // expected-remark@+1 {{activity: unknown: 'foo.read' is not an op whose dependences the pass knows}}
  %opaque = "foo.read"() : () -> f64
  // expected-remark@+1 {{activity: unknown}}
  %maybe = arith.addf %opaque, %one : f64
  // Activity alone does not imply that a derivative rule exists.
  // expected-remark@+1 {{activity: active}}
  %dependent = "foo.opaque"(%lambda) : (f64) -> f64
  md.return %sum : f64
}
