// A module with a list of units and a structure of groups, built in either
// order (D[md-alg-segments-build]): the build over units and the groups'
// build share their helpers, which the module defines once.
//
// RUN: mdir-opt %s --split-input-file --convert-md-exec-to-gpu | FileCheck %s
//
// CHECK-LABEL: func.func @units_first
// CHECK-COUNT-1: func.func private @mdrt_gpu_groups_grid(
// CHECK-NOT: func.func private @mdrt_gpu_groups_grid(
// CHECK-COUNT-1: func.func private @mdrt_gpu_groups_partner(
// CHECK-NOT: func.func private @mdrt_gpu_groups_partner(
// CHECK-LABEL: func.func @groups_first
// CHECK-COUNT-1: func.func private @mdrt_gpu_groups_grid(
// CHECK-NOT: func.func private @mdrt_gpu_groups_grid(

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!gr = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
md.particle_set @atoms
func.func @units_first(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %n: index) {
  %e = arith.constant 4.0 : f64
  %cell = md.orthorhombic_cell %e, %e, %e
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l1 = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  %g0 = md_exec.empty_neighbors size(%n) positions(memref<?x3xf64, 1>) kind(groups) width(64) : !gr
  %g1 = md_exec.refresh_neighbors %g0, %x, %cell cutoff(0.9) skin(0.1) cell_width(0.5) policy(always) : !gr, memref<?x3xf64, 1>
  return
}

// -----

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!gr = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
md.particle_set @atoms
func.func @groups_first(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %n: index) {
  %e = arith.constant 4.0 : f64
  %cell = md.orthorhombic_cell %e, %e, %e
  %g0 = md_exec.empty_neighbors size(%n) positions(memref<?x3xf64, 1>) kind(groups) width(64) : !gr
  %g1 = md_exec.refresh_neighbors %g0, %x, %cell cutoff(0.9) skin(0.1) cell_width(0.5) policy(always) : !gr, memref<?x3xf64, 1>
  %l0 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %l1 = md_exec.refresh_segmented_pairs %l0 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
  return
}
