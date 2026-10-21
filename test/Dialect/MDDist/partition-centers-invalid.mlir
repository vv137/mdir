// D317: no ownership-epoch inference from matching types.
// RUN: mdir-opt %s -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!ghost = !md_dist.field<@dd, ghost, 3 x f64>
!mask = !md_dist.field<@dd, owned, i32>
!wrong = !md_dist.field<@dd, owned, f64>
!halo = !md_dist.halo<@dd, 2.0>
!relation = !md_dist.relation<@dd, directed>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

func.func @wrong_halo(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h0 = md_dist.halo %l, %o, %cell : !own -> !halo
  %h1 = md_dist.halo %l, %o, %cell : !own -> !halo
  %g = md_dist.halo_copy %o, %h0 image(%cell) : !own, !halo -> !ghost
  %n = md_dist.neighborhood %o, %g, %h0 cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  // expected-error @+1 {{relation, halo, and layout must have the same ownership epoch}}
  %i, %b = md_dist.partition_centers %n, %l, %h1
      : !relation, !md_dist.layout<@dd>, !halo -> !mask, !mask
  return
}

func.func @wrong_layout(%x: !vec, %cell: !md.cell) {
  %l0 = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %l1 = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l0 image(%cell) : !vec -> !own
  %h = md_dist.halo %l0, %o, %cell : !own -> !halo
  %g = md_dist.halo_copy %o, %h image(%cell) : !own, !halo -> !ghost
  %n = md_dist.neighborhood %o, %g, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  // expected-error @+1 {{relation, halo, and layout must have the same ownership epoch}}
  %i, %b = md_dist.partition_centers %n, %l1, %h
      : !relation, !md_dist.layout<@dd>, !halo -> !mask, !mask
  return
}

func.func @wrong_indicator(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o, %cell : !own -> !halo
  %g = md_dist.halo_copy %o, %h image(%cell) : !own, !halo -> !ghost
  %n = md_dist.neighborhood %o, %g, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  // expected-error @+1 {{expected complete owned scalar i32 indicators}}
  %i, %b = md_dist.partition_centers %n, %l, %h
      : !relation, !md_dist.layout<@dd>, !halo -> !wrong, !mask
  return
}

func.func @cross_block(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o, %cell : !own -> !halo
  %g = md_dist.halo_copy %o, %h image(%cell) : !own, !halo -> !ghost
  %n = md_dist.neighborhood %o, %g, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  cf.br ^next
^next:
  // expected-error @+1 {{cannot prove relation in the same block before partition}}
  %i, %b = md_dist.partition_centers %n, %l, %h
      : !relation, !md_dist.layout<@dd>, !halo -> !mask, !mask
  return
}

func.func @cross_block_result(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o, %cell : !own -> !halo
  %g = md_dist.halo_copy %o, %h image(%cell) : !own, !halo -> !ghost
  %n = md_dist.neighborhood %o, %g, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  // expected-error @+1 {{cannot prove partition use in the same block}}
  %i, %b = md_dist.partition_centers %n, %l, %h
      : !relation, !md_dist.layout<@dd>, !halo -> !mask, !mask
  cf.br ^next
^next:
  %out = md_dist.collect %i, %l : !mask -> !md.field<@atoms, i32>
  return
}
