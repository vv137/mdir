// The list of units and the loops over its segments (D417):
// the type, the ops of md_exec, and md_alg.traverse_pairs over segments in
// its structure and buffer forms, printed and parsed again, in the custom
// and the generic form. The groups' form of D415 is unchanged
// (traverse.mlir).
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s
// RUN: mdir-opt %s --mlir-print-op-generic | mdir-opt | FileCheck %s

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint

// CHECK-LABEL: func.func @lists
// CHECK: md_exec.empty_segmented_pairs %{{.*}} kinds(%{{.*}} : !md.relation<@atoms, 4, ordered, @kind>) anchor(%{{.*}} : !md.relation<@atoms, 2, ordered>) radius(1.090000e-01) cutoff(9.000000e-01) skin(1.000000e-01) : !md.field<@atoms, i32> -> !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [5.000000e-01, 2.500000e-01, 2.500000e-01], extent = 1.000000e-01>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
// CHECK: md_exec.refresh_segmented_pairs %{{.*}} at(%{{.*}}, %{{.*}}) cutoff(9.000000e-01) skin(1.000000e-01) prune_skin(5.000000e-02)
// CHECK: md_exec.build_segmented_pairs %{{.*}} kinds(%{{.*}} : !md.relation<@atoms, 4, ordered, @kind>) at(%{{.*}}, %{{.*}}) cutoff(9.000000e-01) skin(1.000000e-01)
// CHECK: md_exec.segment_for %{{.*}}, %{{.*}}, %{{.*}} ins(%{{.*}} : !md.field<@atoms, f64>) outs(%{{.*}} : !md.field<@atoms, 3 x f64>) reduce(%{{.*}} : f64) cutoff(9.000000e-01) weights [5.000000e-01] exchange [antisymmetric, symmetric] segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
// CHECK:   ^bb0(%{{.*}}: f64, %{{.*}}: vector<3xf64>, %{{.*}}: f64, %{{.*}}: vector<3xf64>, %{{.*}}: f64, %{{.*}}: f64, %{{.*}}: f64, %{{.*}}: f64):
// CHECK: } segment(1, 1) sites [(0, 0) [1]] {
func.func @lists(%t: !i, %kind: !k, %a: !md.relation<@atoms, 2, ordered>, %x: !x, %cell: !md.cell, %q: !q, %f0: !x) -> (!x, f64) {
  %u0 = arith.constant 0.0 : f64
  %l0 = md_exec.empty_segmented_pairs %t kinds(%kind : !k) anchor(%a : !md.relation<@atoms, 2, ordered>) radius(0.109) cutoff(0.9) skin(0.1) : !i -> !sp
  %l1 = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  %l2 = md_exec.build_segmented_pairs %t kinds(%kind : !k) at(%x, %cell) cutoff(0.9) skin(0.1) : !i, !x -> !sp
  %f, %u = md_exec.segment_for %l1, %x, %cell ins(%q : !q) outs(%f0 : !x) reduce(%u0 : f64)
      cutoff(0.9) weights [0.5] exchange [antisymmetric, symmetric]
    segment(1, 0) sites [(0, 3) [0, 1], (0, 0) [1]] {
    ^bb0(%r0: f64, %d0: vector<3xf64>, %r1: f64, %d1: vector<3xf64>, %qi: f64, %qj0: f64, %qj1: f64, %qj2: f64):
      md_exec.yield %d0, %r0, %r1 : vector<3xf64>, f64, f64
    }
    segment(1, 1) sites [(0, 0) [1]] {
    ^bb0(%r2: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
      md_exec.yield %r2 : f64
    } : !sp, !x -> !x, f64
  return %f, %u : !x, f64
}

// CHECK-LABEL: func.func @structure
// CHECK: md_alg.traverse_pairs structure %{{.*}} : !mdrt.segmented_pairs<
// CHECK-SAME: unique visit unordered combine atomic algebra ["any_order", "any_order", "any_order"] exchange ["antisymmetric", "symmetric", "symmetric"] cutoff(9.000000e-01) weights [5.000000e-01, 5.000000e-01] rotate(16) segment(0, 0) sites [(1, 1) [0, 1], (3, 3) [2]] {
// CHECK: md_alg.yield
// CHECK: } segment(1, 0) sites [(0, 0) [1]] {
// CHECK: } -> f64, f64
func.func @structure(%l: !sp, %x: memref<?x3xf64, 1>, %cell: !md.cell, %f: memref<?x3xf32, 1>) -> (f64, f64) {
  %u0 = arith.constant 0.0 : f64
  %u:2 = md_alg.traverse_pairs structure %l : !sp
      positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
      outs(%f : memref<?x3xf32, 1>) reduce(%u0, %u0 : f64, f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order", "any_order"]
      exchange ["antisymmetric", "symmetric", "symmetric"]
      cutoff(0.9) weights [0.5, 0.5] rotate(16)
    segment(0, 0) sites [(1, 1) [0, 1], (3, 3) [2]] {
    ^bb0(%r0: f32, %d0: vector<3xf32>, %r1: f32, %d1: vector<3xf32>):
      %e0 = arith.extf %r0 : f32 to f64
      %e1 = arith.extf %r1 : f32 to f64
      md_alg.yield %d0, %e0, %e1 : vector<3xf32>, f64, f64
    }
    segment(1, 0) sites [(0, 0) [1]] {
    ^bb0(%r0: f32, %d0: vector<3xf32>):
      %e0 = arith.extf %r0 : f32 to f64
      md_alg.yield %e0 : f64
    } -> f64, f64
  return %u#0, %u#1 : f64, f64
}

// CHECK-LABEL: func.func @buffers
// CHECK: md_alg.traverse_pairs lists(%{{.*}} : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%{{.*}}, %{{.*}}) contract(!mdrt.segmented_pairs<
// CHECK-SAME: reach(1.000000e+00)
// CHECK-SAME: segment(1, 1) sites [(0, 0) [0]] {
func.func @buffers(%b: memref<?xi32>, %n: index, %x: memref<?x3xf64>, %cell: vector<3xf64>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  %u = md_alg.traverse_pairs lists(%b, %b, %b, %b, %b, %b, %b, %b, %b : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%n, %n)
      contract(!sp) reach(1.0)
      positions(%x : memref<?x3xf64>) cell(%cell : vector<3xf64>) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order"] exchange ["symmetric"]
      cutoff(0.9) weights [0.5]
    segment(1, 1) sites [(0, 0) [0]] {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_alg.yield %r2 : f64
    } -> f64
  return %u : f64
}
