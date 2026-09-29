// RUN: mdir-opt %s --md-exec-fuse-loops --cse | FileCheck %s

!vec  = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!nl   = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// Energy and forces become one loop. What both kernels compute is computed
// once.
//
// CHECK-LABEL: func.func @energy_forces(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell, %[[NL:[a-z0-9]+]]: !mdrt.neighbors<@atoms>, %[[A:[a-z0-9]+]]: f64)
func.func @energy_forces(%x: !vec, %cell: !md.cell, %nl: !nl, %a: f64)
    -> (f64, !vec) {
  // CHECK:      %[[U0:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f64
  // CHECK:      %[[F0:[0-9]+]] = md_exec.zeros
  // CHECK:      %[[LOOP:[0-9]+]]:2 = md_exec.pair_for %[[NL]], %[[X]], %[[CELL]]
  // CHECK-SAME:   outs(%[[F0]] : !md.field<@atoms, 3 x f64>) reduce(%[[U0]] : f64)
  // CHECK-SAME:   cutoff(1.500000e+00) weights [5.000000e-01]
  // CHECK-NEXT: ^bb0(%[[R2:[a-z0-9]+]]: f64, %[[D:[a-z0-9]+]]: vector<3xf64>):
  // CHECK-NEXT:   %[[R:[0-9]+]] = math.sqrt %[[R2]]
  // CHECK-NEXT:   %[[K:[0-9]+]] = arith.divf %[[A]], %[[R]]
  // CHECK-NOT:    math.sqrt
  // CHECK:        md_exec.yield %{{[0-9]+}}, %[[K]] : vector<3xf64>, f64
  // CHECK-NOT:  md_exec.pair_for
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.5)
      weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %r = math.sqrt %r2 : f64
    %k = arith.divf %a, %r : f64
    md_exec.yield %k : f64
  } : !nl, !vec -> f64

  %f0 = md_exec.zeros : !vec
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %r = math.sqrt %r2 : f64
    %k = arith.divf %a, %r : f64
    %g = arith.divf %k, %r2 : f64
    %gv = vector.broadcast %g : f64 to vector<3xf64>
    %kf = arith.mulf %gv, %d : vector<3xf64>
    md_exec.yield %kf : vector<3xf64>
  } : !nl, !vec -> !vec

  // CHECK:      return %[[LOOP]]#1, %[[LOOP]]#0
  return %u, %f : f64, !vec
}

// A field that both kernels read is read once.
//
// CHECK-LABEL: func.func @shared_field(
func.func @shared_field(%x: !vec, %cell: !md.cell, %nl: !nl, %q: !real,
                        %s: !real) -> (f64, f64) {
  // CHECK:      md_exec.pair_for
  // CHECK-SAME:   ins(%[[Q:[a-z0-9]+]], %[[S:[a-z0-9]+]] : !md.field<@atoms, f64>, !md.field<@atoms, f64>)
  // CHECK-SAME:   reduce(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : f64, f64)
  // CHECK-NOT:    weights
  // CHECK-SAME:   policy
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>, %[[Q1:[a-z0-9]+]]: f64, %[[Q2:[a-z0-9]+]]: f64, %[[S1:[a-z0-9]+]]: f64, %[[S2:[a-z0-9]+]]: f64):
  // CHECK-NEXT:   %[[QQ:[0-9]+]] = arith.mulf %[[Q1]], %[[Q2]]
  // CHECK-NEXT:   %[[SS:[0-9]+]] = arith.mulf %[[S1]], %[[S2]]
  // CHECK-NEXT:   %[[QS:[0-9]+]] = arith.mulf %[[QQ]], %[[SS]]
  // CHECK-NEXT:   md_exec.yield %[[QQ]], %[[QS]]
  // CHECK-NOT:  md_exec.pair_for
  %zero = arith.constant 0.0 : f64
  %a = md_exec.pair_for %nl, %x, %cell ins(%q : !real) reduce(%zero : f64)
      cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>, %q1: f64, %q2: f64):
    %qq = arith.mulf %q1, %q2 : f64
    md_exec.yield %qq : f64
  } : !nl, !vec -> f64
  %b = md_exec.pair_for %nl, %x, %cell ins(%s, %q : !real, !real)
      reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>, %s1: f64, %s2: f64, %q1: f64, %q2: f64):
    %qq = arith.mulf %q1, %q2 : f64
    %ss = arith.mulf %s1, %s2 : f64
    %qs = arith.mulf %qq, %ss : f64
    md_exec.yield %qs : f64
  } : !nl, !vec -> f64
  return %a, %b : f64, f64
}

// The second loop uses a result of the first, so the two stay apart.
//
// CHECK-LABEL: func.func @dependent(
func.func @dependent(%x: !vec, %cell: !md.cell, %nl: !nl) -> f64 {
  // CHECK:      md_exec.pair_for
  // CHECK:      md_exec.pair_for
  %zero = arith.constant 0.0 : f64
  %a = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  %b = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %k = arith.mulf %a, %r2 : f64
    md_exec.yield %k : f64
  } : !nl, !vec -> f64
  return %b : f64
}

// Different cutoffs: the loops run over different pairs.
//
// CHECK-LABEL: func.func @different_cutoffs(
func.func @different_cutoffs(%x: !vec, %cell: !md.cell, %nl: !nl)
    -> (f64, f64) {
  // CHECK:      md_exec.pair_for
  // CHECK:      md_exec.pair_for
  %zero = arith.constant 0.0 : f64
  %a = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  %b = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.2)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %a, %b : f64, f64
}

// Loops in the storage form are left alone.
//
// CHECK-LABEL: func.func @storage(
// CHECK:         md_exec.pair_for
// CHECK:         md_exec.pair_for
func.func @storage(%x: memref<?x3xf64>, %f: memref<?x3xf64>,
                   %g: memref<?x3xf64>, %cell: !md.cell,
                   %nl: !mdrt.neighbors<@atoms>) {
  md_exec.pair_for %nl, %x, %cell outs(%f : memref<?x3xf64>)
      cutoff(2.5) overwrite [true] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms>, memref<?x3xf64>
  md_exec.pair_for %nl, %x, %cell outs(%g : memref<?x3xf64>)
      cutoff(2.5) overwrite [true] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms>, memref<?x3xf64>
  return
}
