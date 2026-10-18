// What the executor through md_exec refuses (D[ml2-exec]): a loop over the
// ghost rows, the inner product of the adjoint test, which the reference
// executor runs.
//
// RUN: mdir-opt %s --convert-md-dist-to-md-exec -verify-diagnostics

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

func.func @f(%x: !vec, %l: f64) -> f64 {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  // expected-error@+1 {{runs over ghost rows, which the executor through md_exec does not run; the reference executor does}}
  %sp = md_dist.sum_particles gather(%xg : !gho) {
  ^bb0(%v: vector<3xf64>):
    %s = vector.reduction <add>, %v : vector<3xf64> into f64
    md_dist.yield %s : f64
  } : !md_dist.partial<f64>
  %s = md_dist.all_reduce %sp : !md_dist.partial<f64> -> f64
  return %s : f64
}
