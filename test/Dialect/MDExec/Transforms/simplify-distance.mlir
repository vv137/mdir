// RUN: mdir-opt %s --md-exec-simplify-distance --canonicalize --cse | FileCheck %s

!vec  = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!nl   = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// The Lennard-Jones energy has only even powers of the distance:
// 4 eps sigma^12 r^-12 - 4 eps sigma^6 r^-6. No square root is left. A
// term whose factors are 0 is 0, though the inverse power of r may have
// left the range of its type: eps = 0 does not make 0 * inf.
//
// CHECK-LABEL: func.func @lennard_jones(
// CHECK-SAME:    %[[EPS:[a-z0-9]+]]: f64, %[[SIGMA:[a-z0-9]+]]: f64)
func.func @lennard_jones(%x: !vec, %cell: !md.cell, %nl: !nl, %eps: f64,
                         %sigma: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  // CHECK:      md_exec.pair_for
  // CHECK-NEXT: ^bb0(%[[R2:[a-z0-9]+]]: f64,
  // CHECK-NEXT:   %[[INVERSE:[0-9]+]] = arith.divf %{{[a-z0-9_]+}}, %[[R2]]
  // CHECK-NOT:    math.sqrt
  // CHECK:        %[[S12:[0-9]+]] = math.fpowi %[[SIGMA]], %c12
  // CHECK-NEXT:   %[[F12:[0-9]+]] = arith.mulf %[[S12]], %[[EPS]]
  // CHECK-NEXT:   math.fpowi %[[INVERSE]], %c6
  // CHECK-NEXT:   arith.mulf %[[F12]],
  // CHECK-NEXT:   %[[T12:[0-9]+]] = arith.mulf
  // CHECK-NEXT:   %[[NONE12:[0-9]+]] = arith.cmpf oeq, %[[F12]], %[[ZERO:[a-z0-9_]+]]
  // CHECK-NEXT:   arith.select %[[NONE12]], %[[ZERO]], %[[T12]]
  // CHECK:        math.fpowi %[[SIGMA]], %c6
  // CHECK:        math.fpowi %[[INVERSE]], %c3
  // CHECK:        arith.cmpf oeq
  // CHECK-NEXT:   arith.select
  // CHECK-NOT:    math.sqrt
  // CHECK:        md_exec.yield
  %u = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(2.5)
      weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %r   = math.sqrt %r2 : f64
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md_exec.yield %k : f64
  } : !nl, !vec -> f64
  return %u : f64
}

// An odd power needs the distance: q1 q2 / r = q1 q2 (1 / r2) r.
//
// CHECK-LABEL: func.func @coulomb(
func.func @coulomb(%x: !vec, %cell: !md.cell, %nl: !nl, %q: !real) -> f64 {
  %zero = arith.constant 0.0 : f64
  // CHECK:      md_exec.pair_for
  // CHECK-NEXT: ^bb0(%[[R2:[a-z0-9]+]]: f64, %{{[a-z0-9]+}}: vector<3xf64>, %[[Q1:[a-z0-9]+]]: f64, %[[Q2:[a-z0-9]+]]: f64):
  // CHECK-DAG:    %[[R:[0-9]+]] = math.sqrt %[[R2]]
  // CHECK-DAG:    %[[INVERSE:[0-9]+]] = arith.divf %{{[a-z0-9_]+}}, %[[R2]]
  // CHECK:        md_exec.yield
  %u = md_exec.pair_for %nl, %x, %cell ins(%q : !real) reduce(%zero : f64)
      cutoff(2.5) weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>, %q1: f64, %q2: f64):
    %r  = math.sqrt %r2 : f64
    %qq = arith.mulf %q1, %q2 : f64
    %k  = arith.divf %qq, %r : f64
    md_exec.yield %k : f64
  } : !nl, !vec -> f64
  return %u : f64
}

// Terms that cancel are not computed: (a r) / r = a.
//
// CHECK-LABEL: func.func @cancel(
// CHECK-SAME:    %[[A:[a-z0-9]+]]: f64)
func.func @cancel(%x: !vec, %cell: !md.cell, %nl: !nl, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  // CHECK:      md_exec.pair_for
  // CHECK-NEXT: ^bb0(
  // CHECK-NEXT:   md_exec.yield %[[A]] : f64
  %u = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(2.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %r  = math.sqrt %r2 : f64
    %ar = arith.mulf %a, %r : f64
    %k  = arith.divf %ar, %r : f64
    md_exec.yield %k : f64
  } : !nl, !vec -> f64
  return %u : f64
}

// An op that forms are not carried through receives its operand as a
// value, and its result is a factor like any other.
//
// CHECK-LABEL: func.func @exponential(
func.func @exponential(%x: !vec, %cell: !md.cell, %nl: !nl) -> f64 {
  %zero = arith.constant 0.0 : f64
  // CHECK:      md_exec.pair_for
  // CHECK-NEXT: ^bb0(%[[R2:[a-z0-9]+]]: f64,
  // CHECK-NEXT:   %[[R:[0-9]+]] = math.sqrt %[[R2]]
  // CHECK:        %[[E:[0-9]+]] = math.exp %{{[0-9]+}}
  // CHECK:        %[[K:[0-9]+]] = arith.mulf %[[E]], %[[R2]]
  // CHECK:        md_exec.yield %[[K]] : f64
  %u = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(2.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %r  = math.sqrt %r2 : f64
    %m  = arith.negf %r : f64
    %e  = math.exp %m : f64
    %er = arith.mulf %e, %r : f64
    %k  = arith.mulf %er, %r : f64
    md_exec.yield %k : f64
  } : !nl, !vec -> f64
  return %u : f64
}

// A product of two sums is not multiplied out. Written in powers of the
// distance, a polynomial in `r - a` has terms that are much larger than its
// value, and their sum loses digits.
//
// CHECK-LABEL: func.func @polynomial(
func.func @polynomial(%x: !vec, %cell: !md.cell, %nl: !nl) -> f64 {
  %zero = arith.constant 0.0 : f64
  // CHECK:      md_exec.pair_for
  // CHECK-NEXT: ^bb0(%[[R2:[a-z0-9]+]]: f64,
  // CHECK-NEXT:   %[[R:[0-9]+]] = math.sqrt %[[R2]]
  // CHECK-NEXT:   %[[T:[0-9]+]] = arith.subf %[[R]], %{{[a-z0-9_]+}}
  // CHECK-NEXT:   %[[K:[0-9]+]] = math.fpowi %[[T]], %c3
  // CHECK-NEXT:   md_exec.yield %[[K]]
  %u = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(2.5)
      weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %r  = math.sqrt %r2 : f64
    %a  = arith.constant 2.0 : f64
    %t  = arith.subf %r, %a : f64
    %t2 = arith.mulf %t, %t : f64
    %k  = arith.mulf %t2, %t : f64
    md_exec.yield %k : f64
  } : !nl, !vec -> f64
  return %u : f64
}
