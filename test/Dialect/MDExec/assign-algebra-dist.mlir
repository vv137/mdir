// md-exec-assign-algebra alone on a module of md_dist ops: it states the
// algebra of the sums across domains, an attribute of md_exec, which no
// other pass of the pipeline has loaded. It aborted with "the dialect was
// likely not loaded" until the pass declared md_exec (#835).
// RUN: mdir-opt %s --md-exec-assign-algebra | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!gho = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.5>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// CHECK-LABEL: func.func @sums
func.func @sums(%x: !vec, %l: f64) -> (f64, !vec) {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %z = md_dist.zeros %lay : !md_dist.layout<@dd> -> !own
  // CHECK: md_dist.halo_sum {{.*}}md_exec.algebra = #md_exec.algebra<fixed_order>
  %t = md_dist.halo_sum %z, %xg, %h : !own, !gho, !halo -> !own
  %p = md_dist.sum_particles gather(%xg : !gho) {
  ^bb0(%v: vector<3xf64>):
    %s = vector.reduction <add>, %v : vector<3xf64> into f64
    md_dist.yield %s : f64
  } : !md_dist.partial<f64>
  // CHECK: md_dist.all_reduce {{.*}}md_exec.algebra = #md_exec.algebra<fixed_order>
  %s = md_dist.all_reduce %p : !md_dist.partial<f64> -> f64
  %c = md_dist.collect %t, %lay : !own -> !vec
  return %s, %c : f64, !vec
}
