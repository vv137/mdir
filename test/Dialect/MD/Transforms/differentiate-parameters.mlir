// RUN: mdir-opt %s --md-differentiate -split-input-file | FileCheck %s
// RUN: mdir-opt %s --md-differentiate="remarks=true" -split-input-file -verify-diagnostics -o /dev/null

// The three outcomes of a derivative with respect to a parameter (D161): a
// value proved independent of it has the derivative zero; a dependent one
// takes its rule. A dependent one without a rule, or one whose
// independence cannot be proved, is an error that names it
// (differentiate-parameters-invalid.mlir).

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
md.particle_set @atoms

// The sum does not take %a: its derivative is zero, and only that of a²
// remains.
//
// CHECK-LABEL: md.function @independent.derivative3(
// CHECK-NOT: md.sum_particles
// CHECK: %[[D:.*]] = arith.addf %arg3, %arg3 : f64
// CHECK: md.return %[[D]] : f64
md.potential @independent(%x: !vec, %cell: !md.cell, %q: !real, %a: f64)
    -> f64 {
  // expected-remark@+1 {{independent of argument 3 of 'independent': its derivative is zero}}
  %s = md.sum_particles gather(%q : !real) {
  ^bb0(%q_i: f64):
    %k = arith.mulf %q_i, %q_i : f64
    md.yield %k : f64
  } : f64
  %aa = arith.mulf %a, %a : f64
  %u = arith.addf %s, %aa : f64
  md.return %u : f64
}

md.function @first(%x: !vec, %cell: !md.cell, %q: !real, %a: f64) -> f64 {
  %d = md.evaluate @independent(%x, %cell, %q, %a) request [derivative(3)]
      : (!vec, !md.cell, !real, f64) -> f64
  md.return %d : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
md.particle_set @atoms

// %a reaches the sum only through a value that its kernel takes from
// outside, %b = 2a: the sum depends on %a, and its derivative is the sum of
// the derivative of its kernel, q_i · 2.
//
// CHECK-LABEL: md.function @captured.derivative3(
// CHECK: md.sum_particles
// CHECK: ^bb0(%[[Q:.*]]: f64):
// CHECK: arith.mulf %[[Q]],
// CHECK: md.yield
md.potential @captured(%x: !vec, %cell: !md.cell, %q: !real, %a: f64)
    -> f64 {
  %b = arith.addf %a, %a : f64
  %s = md.sum_particles gather(%q : !real) {
  ^bb0(%q_i: f64):
    %k = arith.mulf %q_i, %b : f64
    md.yield %k : f64
  } : f64
  md.return %s : f64
}

md.function @second(%x: !vec, %cell: !md.cell, %q: !real, %a: f64) -> f64 {
  %d = md.evaluate @captured(%x, %cell, %q, %a) request [derivative(3)]
      : (!vec, !md.cell, !real, f64) -> f64
  md.return %d : f64
}
