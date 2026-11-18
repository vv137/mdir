// The lists of units of a real program lower on a device
// (D421, fixes of D419's lowering that the test route
// reached first):
//   - a loop over steps that carries other values besides the list;
//   - the members of the kinds, the excluded pairs, and the anchors on the
//     host, where a program keeps its tuples;
//   - a build in place, outside a loop over steps (the evaluation before the
//     first step), which md-exec-assign-storage gives storage of its own and
//     a refresh.
//
// RUN: mdir-opt %s --split-input-file --md-exec-reuse-neighbors --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu | FileCheck %s

// CHECK-LABEL: func.func @carried
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f64)
// CHECK: call @mdrt_gpu_build_segmented_pairs_k1(
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @carried(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %n: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %s = scf.for %i = %c0 to %n step %c1 iter_args(%a = %zero) -> (f64) {
    %l = md_exec.build_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) at(%x, %cell) cutoff(0.9) skin(0.1) : memref<?xi32, 1>, memref<?x3xf64, 1> -> !sp
    %b = arith.addf %a, %one : f64
    scf.yield %b : f64
  }
  return %s : f64
}

// -----

// CHECK-LABEL: func.func @host
// CHECK: call @mdrt_gpu_build_segmented_pairs_k1(
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @host(%t: memref<?xi32, 1>, %k: memref<?x3xi32>, %e: memref<?x2xi32>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %l = md_exec.build_segmented_pairs %t kinds(%k : memref<?x3xi32>) exclude(%e : memref<?x2xi32>) at(%x, %cell) cutoff(0.9) skin(0.1) : memref<?xi32, 1>, memref<?x3xf64, 1> -> !sp
  }
  return
}

// -----

// CHECK-LABEL: func.func @in_place
// CHECK: call @mdrt_gpu_build_segmented_pairs_k1(
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind_0 on(@atoms) arity(3) orientation(ordered) disjoint
func.func @in_place(%t: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 3, ordered, @kind_0>, %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  %l = md_exec.build_segmented_pairs %t kinds(%k : !md.relation<@atoms, 3, ordered, @kind_0>) at(%x, %cell) cutoff(0.9) skin(0.1) : !md.field<@atoms, i32>, !md.field<@atoms, 3 x f64> -> !sp
  return
}
