// RUN: mdir-opt %s -split-input-file --test-md-exec-template-windows --verify-each | FileCheck %s
// The next-to-last force pair of a water pair, when it reads a table that
// cannot fault and is otherwise pure, is computed unguarded and kept by a
// select on its cutoff test; the other pairs keep their guards. The three
// pairs reach disjoint sites, so their order is 0, 1, 2, and the scale of
// each (2, 3, 5) names it. In the second case the read of pair 1 is not
// clamped from above, so it cannot go before the guard, and pair 1 keeps
// its guard too.

// CHECK-LABEL: func.func @window_forces
// CHECK: scf.if
// CHECK: arith.constant 2.000000e+00
// CHECK: } else {
// CHECK-NOT: scf.if
// CHECK: %[[INSIDE:[0-9]+]] = arith.cmpf olt
// CHECK-NOT: scf.if
// CHECK: arith.constant 3.000000e+00
// CHECK-NOT: scf.if
// CHECK: %[[SUM:[0-9]+]] = arith.addf
// CHECK-NEXT: arith.select %[[INSIDE]], %[[SUM]], %
// CHECK-NOT: scf.if
// CHECK: %[[OTHER:[0-9]+]] = arith.addf
// CHECK-NEXT: arith.select %[[INSIDE]], %[[OTHER]], %
// CHECK: scf.if
// CHECK: arith.constant 5.000000e+00
// CHECK: } else {
// CHECK: return

// CHECK-LABEL: func.func @window_forces
// CHECK: scf.if
// CHECK: arith.constant 2.000000e+00
// CHECK: } else {
// CHECK: scf.if
// CHECK: memref.load
// CHECK: arith.constant 3.000000e+00
// CHECK: } else {
// CHECK: scf.if
// CHECK: arith.constant 5.000000e+00
// CHECK: } else {
// CHECK-NOT: arith.select
// CHECK: return
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [3, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %table = memref.get_global @table : memref<16xf64>
    %lo = arith.constant 0 : i32
    %hi = arith.constant 15 : i32
    %ten = arith.constant 10.0 : f64
    %d0 = arith.subf %base, %b1 : vector<3xf64>
    %q0 = arith.mulf %d0, %d0 : vector<3xf64>
    %r0 = vector.reduction <add>, %q0 : vector<3xf64> into f64
    %x0 = arith.mulf %r0, %ten : f64
    %k0 = arith.fptosi %x0 : f64 to i32
    %m0 = arith.maxsi %k0, %lo : i32
    %c0 = arith.minsi %m0, %hi : i32
    %i0 = arith.index_cast %c0 : i32 to index
    %v0 = memref.load %table[%i0] : memref<16xf64>
    %s0 = arith.constant 2.0 : f64
    %w0 = arith.mulf %v0, %s0 : f64
    %g0 = vector.broadcast %w0 : f64 to vector<3xf64>
    %t0 = arith.mulf %g0, %d0 : vector<3xf64>
    %n0 = arith.negf %t0 : vector<3xf64>
    %d1 = arith.subf %a1, %b1 : vector<3xf64>
    %q1 = arith.mulf %d1, %d1 : vector<3xf64>
    %r1 = vector.reduction <add>, %q1 : vector<3xf64> into f64
    %x1 = arith.mulf %r1, %ten : f64
    %k1 = arith.fptosi %x1 : f64 to i32
    %m1 = arith.maxsi %k1, %lo : i32
    %c1 = arith.minsi %m1, %hi : i32
    %i1 = arith.index_cast %c1 : i32 to index
    %v1 = memref.load %table[%i1] : memref<16xf64>
    %s1 = arith.constant 3.0 : f64
    %w1 = arith.mulf %v1, %s1 : f64
    %g1 = vector.broadcast %w1 : f64 to vector<3xf64>
    %t1 = arith.mulf %g1, %d1 : vector<3xf64>
    %n1 = arith.negf %t1 : vector<3xf64>
    %d2 = arith.subf %a2, %b2 : vector<3xf64>
    %q2 = arith.mulf %d2, %d2 : vector<3xf64>
    %r2 = vector.reduction <add>, %q2 : vector<3xf64> into f64
    %x2 = arith.mulf %r2, %ten : f64
    %k2 = arith.fptosi %x2 : f64 to i32
    %m2 = arith.maxsi %k2, %lo : i32
    %c2 = arith.minsi %m2, %hi : i32
    %i2 = arith.index_cast %c2 : i32 to index
    %v2 = memref.load %table[%i2] : memref<16xf64>
    %s2 = arith.constant 5.0 : f64
    %w2 = arith.mulf %v2, %s2 : f64
    %g2 = vector.broadcast %w2 : f64 to vector<3xf64>
    %t2 = arith.mulf %g2, %d2 : vector<3xf64>
    %n2 = arith.negf %t2 : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %t0, %z, %z, %n0, %z, %z,
                  %z, %t1, %z, %z, %n1, %z,
                  %z, %z, %t2, %z, %z, %n2,
                  %d0, %d1, %d2
      : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%q0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [3, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %table = memref.get_global @table : memref<16xf64>
    %lo = arith.constant 0 : i32
    %hi = arith.constant 15 : i32
    %ten = arith.constant 10.0 : f64
    %d0 = arith.subf %base, %b1 : vector<3xf64>
    %q0 = arith.mulf %d0, %d0 : vector<3xf64>
    %r0 = vector.reduction <add>, %q0 : vector<3xf64> into f64
    %x0 = arith.mulf %r0, %ten : f64
    %k0 = arith.fptosi %x0 : f64 to i32
    %m0 = arith.maxsi %k0, %lo : i32
    %c0 = arith.minsi %m0, %hi : i32
    %i0 = arith.index_cast %c0 : i32 to index
    %v0 = memref.load %table[%i0] : memref<16xf64>
    %s0 = arith.constant 2.0 : f64
    %w0 = arith.mulf %v0, %s0 : f64
    %g0 = vector.broadcast %w0 : f64 to vector<3xf64>
    %t0 = arith.mulf %g0, %d0 : vector<3xf64>
    %n0 = arith.negf %t0 : vector<3xf64>
    %d1 = arith.subf %a1, %b1 : vector<3xf64>
    %q1 = arith.mulf %d1, %d1 : vector<3xf64>
    %r1 = vector.reduction <add>, %q1 : vector<3xf64> into f64
    %x1 = arith.mulf %r1, %ten : f64
    %k1 = arith.fptosi %x1 : f64 to i32
    %c1 = arith.maxsi %k1, %lo : i32
    %i1 = arith.index_cast %c1 : i32 to index
    %v1 = memref.load %table[%i1] : memref<16xf64>
    %s1 = arith.constant 3.0 : f64
    %w1 = arith.mulf %v1, %s1 : f64
    %g1 = vector.broadcast %w1 : f64 to vector<3xf64>
    %t1 = arith.mulf %g1, %d1 : vector<3xf64>
    %n1 = arith.negf %t1 : vector<3xf64>
    %d2 = arith.subf %a2, %b2 : vector<3xf64>
    %q2 = arith.mulf %d2, %d2 : vector<3xf64>
    %r2 = vector.reduction <add>, %q2 : vector<3xf64> into f64
    %x2 = arith.mulf %r2, %ten : f64
    %k2 = arith.fptosi %x2 : f64 to i32
    %m2 = arith.maxsi %k2, %lo : i32
    %c2 = arith.minsi %m2, %hi : i32
    %i2 = arith.index_cast %c2 : i32 to index
    %v2 = memref.load %table[%i2] : memref<16xf64>
    %s2 = arith.constant 5.0 : f64
    %w2 = arith.mulf %v2, %s2 : f64
    %g2 = vector.broadcast %w2 : f64 to vector<3xf64>
    %t2 = arith.mulf %g2, %d2 : vector<3xf64>
    %n2 = arith.negf %t2 : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %t0, %z, %z, %n0, %z, %z,
                  %z, %t1, %z, %z, %n1, %z,
                  %z, %z, %t2, %z, %z, %n2,
                  %d0, %d1, %d2
      : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%q0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}
