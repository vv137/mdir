// md.chebyshev (D[prolate-pair-table]): its form, and its derivative, the
// series of the derivatives times the derivative of its argument:
// d/dy (1 - y/2 + T_2(y)/8) = -1/2 + y/2.
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s --check-prefix=FORM
// RUN: mdir-opt %s --md-check-derivative-coverage --md-differentiate | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// FORM-LABEL: md.potential @series(
// FORM:         md.chebyshev %{{.*}} [1.000000e+00, -5.000000e-01, 1.250000e-01] : f64
// FORM:         md.chebyshev %{{.*}} [2.500000e+00] : f64
md.potential @series(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %two = arith.constant 2.0 : f64
  %y = arith.mulf %two, %a : f64
  %s = md.chebyshev %y [1.0, -0.5, 0.125] : f64
  %t = md.chebyshev %a [2.5] : f64
  %u = arith.addf %s, %t : f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @series.derivative2(
// CHECK:         %[[TWO:.*]] = arith.constant 2.000000e+00 : f64
// CHECK:         %[[Y:.*]] = arith.mulf %[[TWO]], %arg2 : f64
// CHECK:         %[[D:.*]] = md.chebyshev %[[Y]] [-5.000000e-01, 5.000000e-01] : f64
// CHECK:         %[[DY:.*]] = arith.mulf %[[D]], %[[TWO]] : f64
// CHECK-NEXT:    md.return %[[DY]] : f64
md.function @slope(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %d = md.evaluate @series(%x, %cell, %a) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  md.return %d : f64
}
