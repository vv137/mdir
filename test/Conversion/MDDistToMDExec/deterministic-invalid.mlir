// The deterministic mode on a device refuses a loop over the rows of a
// domain that adds at the neighbors with floating-point atomics
// (D290): the two-ended gather of the unique traversal.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-dist-to-md-exec \
// RUN:     --md-exec-assign-algebra=deterministic=true --md-exec-assign-storage=memory=device \
// RUN:     --convert-md-exec-to-gpu=deterministic=true -verify-diagnostics
// The ordered algebra refuses it whatever the option of the lowering
// (D301):
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-dist-to-md-exec \
// RUN:     --md-exec-assign-algebra=deterministic=true --md-exec-assign-storage=memory=device \
// RUN:     --convert-md-exec-to-gpu -verify-diagnostics

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([1, 1, 1])

func.func @g(%x: !vec, %l: f64) -> !vec {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  // expected-error@+1 {{adds at the neighbors with floating-point atomics, whose order depends on the threads; the deterministic mode (D84) takes the policy (directed, owner_only)}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho
      -> !md_dist.field<@dd, owned, partial, 3 x f64>,
         !md_dist.field<@dd, ghost, partial, 3 x f64>
  %f = md_dist.halo_sum %fo, %fg, %h
         : !md_dist.field<@dd, owned, partial, 3 x f64>,
           !md_dist.field<@dd, ghost, partial, 3 x f64>, !halo -> !own
  %fc = md_dist.collect %f, %lay : !own -> !vec
  return %fc : !vec
}
