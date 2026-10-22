// D332: the collective predicate belongs to exactly the halo's
// reference version, cell, and ownership epoch.
// RUN: mdir-opt %s -split-input-file -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.2>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @wrong_reference(%x0: !vec, %x1: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x0, %cell : !vec -> !md_dist.layout<@dd>
  %o0 = md_dist.distribute %x0, %l image(%cell) : !vec -> !own
  %o1 = md_dist.distribute %x1, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o0, %cell : !own -> !halo
  // expected-error @+1 {{reference positions must be the halo build version}}
  %ok = md_dist.covered %h, %o1, %o1, %cell cutoff(2.0) : !halo, !own -> i1
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.2>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @wrong_cell(%x: !vec, %cell0: !md.cell, %cell1: !md.cell) {
  %l = md_dist.partition %x, %cell0 : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell0) : !vec -> !own
  %h = md_dist.halo %l, %o, %cell0 : !own -> !halo
  // expected-error @+1 {{cell must be the halo build cell}}
  %ok = md_dist.covered %h, %o, %o, %cell1 cutoff(2.0) : !halo, !own -> i1
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.2>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @stale_epoch(%x: !vec, %cell: !md.cell) {
  // expected-note @+1 {{the layout of the op}}
  %l0 = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o0 = md_dist.distribute %x, %l0 image(%cell) : !vec -> !own
  %h = md_dist.halo %l0, %o0, %cell : !own -> !halo
  // expected-note @+1 {{the layout of current positions}}
  %l1 = md_dist.migrate %l0, %o0, %cell : !md_dist.layout<@dd>, !own -> !md_dist.layout<@dd>
  %o1 = md_dist.redistribute %o0, %l0, %l1 image(%cell) : !own, !md_dist.layout<@dd>, !md_dist.layout<@dd> -> !own
  // expected-error @+1 {{current positions is of another layout (epoch of ownership)}}
  %ok = md_dist.covered %h, %o0, %o1, %cell cutoff(2.0) : !halo, !own -> i1
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.2>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @invalid_cutoff(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o, %cell : !own -> !halo
  // expected-error @+1 {{cutoff must be positive and smaller than the halo reach}}
  %ok = md_dist.covered %h, %o, %o, %cell cutoff(2.2) : !halo, !own -> i1
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.2>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @zero_cutoff(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o, %cell : !own -> !halo
  // expected-error @+1 {{cutoff must be positive and smaller than the halo reach}}
  %ok = md_dist.covered %h, %o, %o, %cell cutoff(0.0) : !halo, !own -> i1
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.2>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @negative_cutoff(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o, %cell : !own -> !halo
  // expected-error @+1 {{cutoff must be positive and smaller than the halo reach}}
  %ok = md_dist.covered %h, %o, %o, %cell cutoff(-0.1) : !halo, !own -> i1
  return
}
