// RUN: mdir-opt %s --md-distribute-group-observations --md-exec-assign-algebra | FileCheck %s
!vec = !md.field<@atoms, 3 x f64>
!mass = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered, @groups>
!owned = !md_dist.field<@dd, owned, 3 x f64>
!ghost = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 1.0>
!partial = !md_dist.partial<vector<3xf64>>
md.particle_set @atoms
md.tuple_set @groups on(@atoms) arity(2) orientation(unordered) disjoint
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @observe(%x: !vec, %other: !vec, %m: !mass, %rel: !pairs, %cell: !md.cell) -> (f64, vector<3xf64>) {
  %l = md_dist.partition %x, %cell groups(%rel : !pairs) : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !owned
  %h = md_dist.halo %l, %o, %cell : !owned -> !halo
  %g = md_dist.halo_copy %o, %h image(%cell) : !owned, !halo -> !ghost
  %rows = md_dist.tuple_rows %rel, %o, %h group : <@atoms, 2, unordered, @groups>, !owned, !halo -> memref<?x3xi64>
  %project = "md_dist.tuple_kernel"(%rows, %o, %g, %h) <{coordinate_kinds = array<i32: 1>, coordinate_members = array<i64: 1, 0, -1, -1>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>}> ({
  ^bb0(%dx: vector<3xf64>):
    md_dist.yield %dx : vector<3xf64>
  }) : (memref<?x3xi64>, !owned, !ghost, !halo) -> !partial
  %total = md_dist.all_reduce %project : !partial -> vector<3xf64>
  // A different position field is immaterial when the kernel reads only masses.
  // CHECK: md_dist.all_reduce
  // CHECK: md_dist.distribute
  // CHECK: "md_dist.tuple_kernel"
  // CHECK: md_dist.all_reduce
  %mass = md.sum_tuples %rel, %other, %cell coordinates(displacement(1, 0)) gather(%m : !mass) {
  ^bb0(%dx: vector<3xf64>, %a: f64, %b: f64):
    %v = arith.addf %a, %b : f64
    md.yield %v : f64
  } : !pairs, !vec -> f64
  // This observation needs its own geometry.
  // CHECK: md.sum_tuples
  %position = md.sum_tuples %rel, %other, %cell coordinates(displacement(1, 0)) {
  ^bb0(%dx: vector<3xf64>):
    md.yield %dx : vector<3xf64>
  } : !pairs, !vec -> vector<3xf64>
  return %mass, %position : f64, vector<3xf64>
}
