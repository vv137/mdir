// RUN: not mdir-opt %s --md-expand-reciprocal=functions=energy --md-differentiate 2>&1 | FileCheck %s
// The explicit function chain is executable, but md.evaluate still retains
// D390's ordinary virial refusal. Shared evaluator integration needs a ruling.
// CHECK: unsupported stencil VJP for active cell operand in ordinary virial composition
!x = !md.field<@atoms, 3 x f64>
!q = !md.field<@atoms, f64>
!m = !md.table<2, f64>
md.particle_set @atoms
md.potential @energy(%x: !x, %h: !md.cell, %q: !q, %m: !m) -> f64 {
  %u, %f, %w = md.reciprocal %x, %q, %h, %m grid([4, 5, 6]) order(4) beta(1.7) coulomb(138.935) : !x, !q, !m -> f64, !x, vector<9xf64>
  md.return %u : f64
}
md.function @caller(%x: !x, %h: !md.cell, %q: !q, %m: !m) -> (f64, !x, vector<9xf64>) {
  %u, %f, %w = md.evaluate @energy(%x, %h, %q, %m) request [energy, forces, virial]
      : (!x, !md.cell, !q, !m) -> (f64, !x, vector<9xf64>)
  md.return %u, %f, %w : f64, !x, vector<9xf64>
}
