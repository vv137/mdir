// D[distributed-pair-center-filter]: a shaped i32 field is not proof.
// RUN: mdir-opt %s -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>
!flag = !md.field<@atoms, i32>
!own = !md_dist.field<@dd, owned, 3 x f64>
!ghost = !md_dist.field<@dd, ghost, 3 x f64>
!mask = !md_dist.field<@dd, owned, i32>
!halo = !md_dist.halo<@dd, 2.0>
!relation = !md_dist.relation<@dd, directed>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

func.func @ordinary_field(%x: !vec, %flags: !flag,
                           %cell: !md.cell) -> f64 {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %f = md_dist.distribute %flags, %l : !flag -> !mask
  %h = md_dist.halo %l, %o, %cell : !own -> !halo
  %g = md_dist.halo_copy %o, %h image(%cell) : !own, !halo -> !ghost
  %n = md_dist.neighborhood %o, %g, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  // expected-error @+1 {{center_set must be a direct interior or boundary result}}
  %u = md_dist.sum_pairs %n, %o, %g center_set(%f : !mask)
      exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %zero = arith.constant 0.0 : f64
    md_dist.yield %zero : f64
  } : !relation, !own, !ghost -> !md_dist.partial<f64>
  %sum = md_dist.all_reduce %u : !md_dist.partial<f64> -> f64
  return %sum : f64
}

func.func @other_relation(%x: !vec, %cell: !md.cell) -> f64 {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o, %cell : !own -> !halo
  %g = md_dist.halo_copy %o, %h image(%cell) : !own, !halo -> !ghost
  %n0 = md_dist.neighborhood %o, %g, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  %n1 = md_dist.neighborhood %o, %g, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  %i, %b = md_dist.partition_centers %n0, %l, %h
      : !relation, !md_dist.layout<@dd>, !halo -> !mask, !mask
  // expected-error @+1 {{center_set must partition the exact traversed relation}}
  %u = md_dist.sum_pairs %n1, %o, %g center_set(%i : !mask)
      exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %zero = arith.constant 0.0 : f64
    md_dist.yield %zero : f64
  } : !relation, !own, !ghost -> !md_dist.partial<f64>
  %sum = md_dist.all_reduce %u : !md_dist.partial<f64> -> f64
  return %sum : f64
}
