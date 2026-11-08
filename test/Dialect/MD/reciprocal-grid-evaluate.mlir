// RUN: mdir-opt %s --md-expand-reciprocal=functions=energy --md-differentiate --verify-each | FileCheck %s --implicit-check-not=md.evaluate
// Only the proved spectral composition is admitted. The outer energy
// weight applies to forces, virial and the direct amplitude-field VJP.
// CHECK: md.function @energy.
// CHECK: md.reduce_stencil
// CHECK: md.gather_stencil
// CHECK: md.return
// CHECK: md.call @energy.
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!m = !md.table<2, f64>
md.particle_set @atoms
md.potential @energy(%x: !x, %h: !md.cell, %q: !q, %m: !m) -> f64 {
  %u, %f, %w = md.reciprocal %x, %q, %h, %m grid([4, 5, 6]) order(4) beta(1.7) coulomb(138.935) : !x, !q, !m -> f64, !x, vector<9xf64>
  %weight = arith.constant 2.5 : f64
  %weighted = arith.mulf %weight, %u : f64
  md.return %weighted : f64
}
md.function @caller(%x: !x, %h: !md.cell, %q: !q, %m: !m) -> (f64, !x, vector<9xf64>, !q) {
  %u, %f, %w, %dq = md.evaluate @energy(%x, %h, %q, %m) request [energy, forces, virial, derivative(2)]
      : (!x, !md.cell, !q, !m) -> (f64, !x, vector<9xf64>, !q)
  md.return %u, %f, %w, %dq : f64, !x, vector<9xf64>, !q
}
