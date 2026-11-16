// md-exec-reuse-neighbors carries a list of units through the loop over
// steps (D419): its build in the loop becomes an empty
// list before the loop, of the same kinds, excluded pairs, anchors, cutoff,
// and skin, and a refresh where the build was, dual with prune-skin below
// the skin (D114); an empty list that a loop within the body starts from
// moves out with it.
//
// RUN: mdir-opt %s --md-exec-reuse-neighbors | FileCheck %s
// RUN: mdir-opt %s --md-exec-reuse-neighbors="prune-skin=0.05" | FileCheck %s --check-prefix=DUAL

!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
!w = !md.relation<@atoms, 3, ordered, @kind_0>
!x = !md.relation<@atoms, 2, unordered, @excluded>
!a = !md.relation<@atoms, 2, ordered, @anchors>
md.particle_set @atoms
md.particle_set @kind_0
md.particle_set @excluded
md.particle_set @anchors
func.func private @use(!sp)

// CHECK-LABEL: func.func @steps
// CHECK:         %[[E:.*]] = md_exec.empty_segmented_pairs %{{.*}} kinds(%{{.*}} : !md.relation<@atoms, 3, ordered, @kind_0>) exclude(%{{.*}} : !md.relation<@atoms, 2, unordered, @excluded>) anchor(%{{.*}} : !md.relation<@atoms, 2, ordered, @anchors>) radius(1.090000e-01) cutoff(9.000000e-01) skin(1.000000e-01)
// CHECK:         scf.for {{.*}} iter_args(%[[L:.*]] = %[[E]])
// CHECK-NOT:       md_exec.build_segmented_pairs
// CHECK:           %[[R:.*]] = md_exec.refresh_segmented_pairs %[[L]] at(%{{.*}}, %{{.*}}) cutoff(9.000000e-01) skin(1.000000e-01) prune_skin(1.000000e-01)
// CHECK:           func.call @use(%[[R]])
// CHECK:           scf.yield %[[R]]
// DUAL:            md_exec.refresh_segmented_pairs {{.*}} prune_skin(5.000000e-02)
func.func @steps(%t: !md.field<@atoms, i32>, %k: !w, %ex: !x, %an: !a, %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %l = md_exec.build_segmented_pairs %t kinds(%k : !w) exclude(%ex : !x) anchor(%an : !a) radius(0.109) at(%x, %cell) cutoff(0.9) skin(0.1) : !md.field<@atoms, i32>, !md.field<@atoms, 3 x f64> -> !sp
    func.call @use(%l) : (!sp) -> ()
  }
  return
}
