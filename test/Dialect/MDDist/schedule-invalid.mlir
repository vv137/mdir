// The schedule proof rejects a transfer whose producer is captured from
// another block: no start edge inside the transfer's block is established.
// RUN: mdir-opt %s --md-dist-print-schedule -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!gho = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

func.func @cross_block(%x: !vec, %cell: !md.cell, %condition: i1) {
  %layout = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %owned = md_dist.distribute %x, %layout image(%cell) : !vec -> !own
  %halo = md_dist.halo %layout, %owned, %cell : !own -> !halo
  scf.if %condition {
    // expected-error @+1 {{cannot prove a start prerequisite in the same block}}
    %ghost = md_dist.halo_copy %owned, %halo image(%cell) : !own, !halo -> !gho
  }
  return
}
