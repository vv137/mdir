// RUN: mdir-opt %s --md-expand-truncation | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

// u(r) − u(r_c)
//
// CHECK-LABEL: md.potential @shift(
md.potential @shift(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  // CHECK: md.sum_relation
  // CHECK-NOT: truncation
  // CHECK: ^bb0(%[[R:[a-z0-9]+]]: f64,
  // CHECK: %[[U:[0-9]+]] = arith.mulf %[[R]], %[[R]]
  // CHECK: %[[RC:[a-z0-9_]+]] = arith.constant 1.500000e+00 : f64
  // CHECK: %[[UC:[0-9]+]] = arith.mulf %[[RC]], %[[RC]]
  // CHECK: %[[T:[0-9]+]] = arith.subf %[[U]], %[[UC]]
  // CHECK: md.yield %[[T]] : f64
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) truncation(shift) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// u(r) − u(r_c) − (r − r_c) · u'(r_c)
//
// CHECK-LABEL: md.potential @force_shift(
md.potential @force_shift(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  // CHECK: md.sum_relation
  // CHECK-NOT: truncation
  // CHECK: ^bb0(%[[R:[a-z0-9]+]]: f64,
  // CHECK: %[[U:[0-9]+]] = arith.mulf %[[R]], %[[R]]
  // CHECK: %[[RC:[a-z0-9_]+]] = arith.constant 1.500000e+00 : f64
  // CHECK: %[[UC:[0-9]+]] = arith.mulf %[[RC]], %[[RC]]
  // CHECK: %[[SLOPE:[0-9]+]] = arith.addf %[[RC]], %[[RC]]
  // CHECK: %[[T:[0-9]+]] = arith.subf %[[U]], %[[UC]]
  // CHECK: %[[DIST:[0-9]+]] = arith.subf %[[R]], %[[RC]]
  // CHECK: %[[LIN:[0-9]+]] = arith.mulf %[[DIST]], %[[SLOPE]]
  // CHECK: %[[FS:[0-9]+]] = arith.subf %[[T]], %[[LIN]]
  // CHECK: md.yield %[[FS]] : f64
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(force_shift) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// u(r) · S(r), with S = 1 up to the switching distance.
//
// CHECK-LABEL: md.potential @switch(
md.potential @switch(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  // CHECK: md.sum_relation
  // CHECK-NOT: truncation
  // CHECK: ^bb0(%[[R:[a-z0-9]+]]: f64,
  // CHECK: %[[U:[0-9]+]] = arith.mulf %[[R]], %[[R]]
  // CHECK: %[[FROM:[a-z0-9_]+]] = arith.constant 1.000000e+00 : f64
  // CHECK: %[[WIDTH:[a-z0-9_]+]] = arith.constant 5.000000e-01 : f64
  // CHECK: %[[OFFSET:[0-9]+]] = arith.subf %[[R]], %[[FROM]]
  // CHECK: %[[T:[0-9]+]] = arith.divf %[[OFFSET]], %[[WIDTH]]
  // CHECK: %[[BELOW:[0-9]+]] = arith.cmpf ole, %[[R]], %[[FROM]]
  // CHECK: %[[S:[0-9]+]] = arith.select %[[BELOW]], %{{[a-z0-9_]+}}, %{{[0-9]+}}
  // CHECK: %[[SW:[0-9]+]] = arith.mulf %[[U]], %[[S]]
  // CHECK: md.yield %[[SW]] : f64
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(switch, from = 1.0) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// u(r) − P(r) − C, with P = 0 up to the switching distance.
//
// CHECK-LABEL: md.potential @force_switch(
md.potential @force_switch(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  // CHECK: md.sum_relation
  // CHECK-NOT: truncation
  // CHECK: ^bb0(%[[R:[a-z0-9]+]]: f64,
  // CHECK: %[[U:[0-9]+]] = arith.mulf %[[R]], %[[R]]
  // CHECK: %[[RC:[a-z0-9_]+]] = arith.constant 1.500000e+00 : f64
  // CHECK: %[[UC:[0-9]+]] = arith.mulf %[[RC]], %[[RC]]
  // CHECK: %[[SLOPE:[0-9]+]] = arith.addf %[[RC]], %[[RC]]
  // CHECK: %[[BELOW:[0-9]+]] = arith.cmpf ole, %[[R]], %{{[a-z0-9_]+}}
  // CHECK: %[[P:[0-9]+]] = arith.select %[[BELOW]], %{{[a-z0-9_]+}}, %{{[0-9]+}}
  // CHECK: %[[REDUCED:[0-9]+]] = arith.subf %[[U]], %[[P]]
  // CHECK: %[[RESULT:[0-9]+]] = arith.subf %[[REDUCED]], %{{[0-9]+}}
  // CHECK: md.yield %[[RESULT]] : f64
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(force_switch, from = 1.0) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// Nothing to do.
//
// CHECK-LABEL: md.potential @none(
md.potential @none(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  // CHECK: ^bb0(%[[R:[a-z0-9]+]]: f64,
  // CHECK-NEXT: %[[U:[0-9]+]] = arith.mulf %[[R]], %[[R]]
  // CHECK-NEXT: md.yield %[[U]] : f64
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}
