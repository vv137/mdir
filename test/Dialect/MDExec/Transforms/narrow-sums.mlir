// RUN: mdir-opt %s --md-exec-narrow-sums | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// The trace of a virial takes three of its nine elements: the loop sums
// those, in the type its kernel computes in, and the vector of nine that
// replaces its result has zeros elsewhere.
//
// CHECK-LABEL: func.func @trace(
// CHECK:         %[[S:[0-9]+]] = md_exec.pair_for
// CHECK-SAME:      reduce(%{{[0-9]+}} : vector<3xf64>)
// CHECK:           %[[N:[0-9]+]] = vector.from_elements %{{.*}} : vector<3xf32>
// CHECK:           %[[W:[0-9]+]] = arith.extf %[[N]] : vector<3xf32> to vector<3xf64>
// CHECK:           md_exec.yield %[[W]] : vector<3xf64>
// CHECK:         } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> vector<3xf64>
// CHECK:         %[[F:[0-9]+]] = vector.from_elements %{{.*}} : vector<9xf64>
// CHECK:         vector.extract %[[F]][0]
func.func @trace(%x: !vec, %cell: !md.cell, %nl: !nl) -> f64 {
  %w0 = arith.constant dense<0.0> : vector<9xf64>
  %w = md_exec.pair_for %nl, %x, %cell reduce(%w0 : vector<9xf64>)
      cutoff(2.5) weights [0.5] exchange [symmetric]
      policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %a = vector.extract %d[0] : f32 from vector<3xf32>
    %b = vector.extract %d[1] : f32 from vector<3xf32>
    %c = vector.extract %d[2] : f32 from vector<3xf32>
    %e = vector.from_elements %a, %b, %c, %a, %b, %c, %a, %b, %c : vector<9xf32>
    %wide = arith.extf %e : vector<9xf32> to vector<9xf64>
    md_exec.yield %wide : vector<9xf64>
  } : !nl, !vec -> vector<9xf64>
  %t0 = vector.extract %w[0] : f64 from vector<9xf64>
  %t4 = vector.extract %w[4] : f64 from vector<9xf64>
  %t8 = vector.extract %w[8] : f64 from vector<9xf64>
  %s = arith.addf %t0, %t4 : f64
  %t = arith.addf %s, %t8 : f64
  return %t : f64
}

// A user that the pass does not know takes every element.
//
// CHECK-LABEL: func.func @whole(
// CHECK:         md_exec.pair_for
// CHECK-SAME:      reduce(%{{.*}} : vector<9xf64>)
func.func @whole(%x: !vec, %cell: !md.cell, %nl: !nl) -> vector<9xf64> {
  %w0 = arith.constant dense<0.0> : vector<9xf64>
  %w = md_exec.pair_for %nl, %x, %cell reduce(%w0 : vector<9xf64>)
      cutoff(2.5) exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %e = vector.broadcast %r2 : f64 to vector<9xf64>
    md_exec.yield %e : vector<9xf64>
  } : !nl, !vec -> vector<9xf64>
  return %w : vector<9xf64>
}
