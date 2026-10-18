// Fields of features (D[ml4-feature-types], docs/ml4-ir.md, Section 3) and
// a gather of several fields from one kernel (D[ml4-neural-op]); printed and
// parsed again unchanged.
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s

// CHECK: func.func private @types(!md.field<@atoms, 16 x f64>, !md.field<@atoms, irreps<4x0e + 4x1o>, f64>, !md.field<@atoms, irreps<128x0e + 128x1o, ir_mul>, f32>, !md.field<@atoms, irreps<2x0e + 1x1o + 1x2e + 1x3o>, f64>)
func.func private @types(!md.field<@atoms, 16 x f64>,
                         !md.field<@atoms, irreps<4x0e + 4x1o>, f64>,
                         !md.field<@atoms, irreps<128x0e+128x1o, ir_mul>, f32>,
                         !md.field<@atoms, irreps<2x0e + 1x1o + 1x2e + 1x3o, mul_ir>, f64>)

!vec = !md.field<@atoms, 3 x f64>
!h = !md.field<@atoms, irreps<1x0e + 1x1o>, f64>
!pairs = !md.relation<@atoms, 2, unordered>

// CHECK-LABEL: md.function @two
// CHECK:       %{{.*}}:2 = md.gather_relation
// CHECK:         md.yield %{{.*}}, %{{.*}} : vector<4xf64>, vector<3xf64>
// CHECK:       } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, irreps<1x0e + 1x1o>, f64>, !md.field<@atoms, 3 x f64>
md.function @two(%x: !vec, %cell: !md.cell, %h: !h) -> (!h, !vec) {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  %a, %f = md.gather_relation %n, %x, %cell gather(%h : !h) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<4xf64>, %hj: vector<4xf64>):
    md.yield %hj, %d : vector<4xf64>, vector<3xf64>
  } : !pairs, !vec -> !h, !vec
  md.return %a, %f : !h, !vec
}
