// D399: a state is formed only under a witness of
// coverage, of one ownership epoch, with every field it carries, and it is
// linear; a neighborhood at other positions than its halo's needs the same
// witness (V10).
// RUN: mdir-opt %s -split-input-file -verify-diagnostics

!own = !md_dist.field<@dd, owned, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
// expected-error @+1 {{the cutoff must be positive and less than the reach}}
func.func private @cutoff(!md_dist.state<@dd, reach = 2.0, cutoff = 2.0, fields = [3 x f64]>)

// -----

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
// expected-error @+1 {{the first field is the positions, 3 x f64}}
func.func private @positions(!md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [f64]>)

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @no_witness(%x: !vec, %y: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %y0 = md_dist.distribute %y, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  // expected-error @+1 {{cannot prove that the halo covers the positions at the cutoff}}
  %s = md_dist.pack %l, %h, %cell fields(%y0 : !own) : !state
  %r:4 = md_dist.unpack %s, %cell : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!gho = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @neighborhood_no_witness(%x: !vec, %y: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %y0 = md_dist.distribute %y, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  %yg = md_dist.halo_copy %y0, %h image(%cell) : !own, !halo -> !gho
  // expected-error @+1 {{(V10)}}
  %n = md_dist.neighborhood %y0, %yg, %h cutoff(2.0)
      : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @else_region(%x: !vec, %y: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %y0 = md_dist.distribute %y, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  %ok = md_dist.covered %h, %x0, %y0, %cell cutoff(2.0) : !halo, !own -> i1
  %s = scf.if %ok -> (!state) {
    %a = md_dist.pack %l, %h, %cell fields(%y0 : !own) : !state
    scf.yield %a : !state
  } else {
    // The test failed: the halo does not cover these positions.
    // expected-error @+1 {{(V10)}}
    %b = md_dist.pack %l, %h, %cell fields(%y0 : !own) : !state
    scf.yield %b : !state
  }
  %r:4 = md_dist.unpack %s, %cell : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @smaller_cutoff(%x: !vec, %y: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %y0 = md_dist.distribute %y, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  // Covered at 1.5 does not cover the state's cutoff 2.0.
  %ok = md_dist.covered %h, %x0, %y0, %cell cutoff(1.5) : !halo, !own -> i1
  %s = scf.if %ok -> (!state) {
    // expected-error @+1 {{(V10)}}
    %a = md_dist.pack %l, %h, %cell fields(%y0 : !own) : !state
    scf.yield %a : !state
  } else {
    %b = md_dist.pack %l, %h, %cell fields(%x0 : !own) : !state
    scf.yield %b : !state
  }
  %r:4 = md_dist.unpack %s, %cell : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!lay = !md_dist.layout<@dd>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, 3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @field_left_behind(%x: !vec, %v: !vec, %cell: !md.cell) {
  // expected-note @+1 {{the layout of field #1}}
  %l = md_dist.partition %x, %cell : !vec -> !lay
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %v0 = md_dist.distribute %v, %l : !vec -> !own
  // expected-note @+1 {{the layout of the op}}
  %l2 = md_dist.migrate %l, %x0, %cell : !lay, !own -> !lay
  %x2 = md_dist.redistribute %x0, %l, %l2 image(%cell) : !own, !lay, !lay -> !own
  %h2 = md_dist.halo %l2, %x2, %cell : !own -> !halo
  // The velocities were not migrated with the positions.
  // expected-error @+1 {{field #1 is of another layout (epoch of ownership) than the op}}
  %s = md_dist.pack %l2, %h2, %cell fields(%x2, %v0 : !own, !own) : !state
  %r:5 = md_dist.unpack %s, %cell : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, 3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @missing_field(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  // expected-error @+1 {{the state carries 2 fields, given 1}}
  %s = md_dist.pack %l, %h, %cell fields(%x0 : !own) : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @twice(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  // expected-error @+1 {{a md_dist.state is linear}}
  %s = md_dist.pack %l, %h, %cell fields(%x0 : !own) : !state
  %a:4 = md_dist.unpack %s, %cell : !state
  %b:4 = md_dist.unpack %s, %cell : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @dropped(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  // expected-error @+1 {{a md_dist.state is linear}}
  %s = md_dist.pack %l, %h, %cell fields(%x0 : !own) : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @nested_unpack(%x: !vec, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  // expected-error @+1 {{a md_dist.state is linear}}
  %s = md_dist.pack %l, %h, %cell fields(%x0 : !own) : !state
  scf.for %i = %c0 to %n step %c1 {
    // An unpack in a loop would read the state at every iteration.
    %a:4 = md_dist.unpack %s, %cell : !state
  }
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @wrong_cell(%x: !vec, %cell: !md.cell, %other: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  // expected-error @+1 {{the cell must be the halo's}}
  %s = md_dist.pack %l, %h, %other fields(%x0 : !own) : !state
  %a:4 = md_dist.unpack %s, %cell : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @covered_unpacked(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  %s = md_dist.pack %l, %h, %cell fields(%x0 : !own) : !state
  %l1, %h1, %r1, %x1 = md_dist.unpack %s, %cell : !state
  // The reference of an unpacked halo is the unpack's, not a copy of it.
  // expected-error @+1 {{reference positions must be the halo build version}}
  %ok = md_dist.covered %h1, %x1, %x1, %cell cutoff(2.0) : !halo, !own -> i1
  return
}

// -----

// A step loop that rebuilds the halo at every step without migrating: the
// positions drift out of the cell, and a halo of the images within
// ceil(reach / L) cells of them misses pairs (V10).
!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!lay = !md_dist.layout<@dd>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, 3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @halo_only_rebuild(%x: !vec, %v: !vec, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %l0 = md_dist.partition %x, %cell : !vec -> !lay
  %x0 = md_dist.distribute %x, %l0 image(%cell) : !vec -> !own
  %v0 = md_dist.distribute %v, %l0 : !vec -> !own
  %h0 = md_dist.halo %l0, %x0, %cell : !own -> !halo
  %s0 = md_dist.pack %l0, %h0, %cell fields(%x0, %v0 : !own, !own) : !state
  %s = scf.for %i = %c0 to %n step %c1 iter_args(%sa = %s0) -> (!state) {
    %l, %h, %xr, %xa, %va = md_dist.unpack %sa, %cell : !state
    // expected-note @+1 {{the positions of the halo}}
    %x1 = md_dist.map_particles gather(%xa, %va : !own, !own) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
      %y = arith.addf %x_i, %v_i : vector<3xf64>
      md_dist.yield %y : vector<3xf64>
    } : !own
    %h1 = md_dist.halo %l, %x1, %cell : !own -> !halo
    // expected-error @+1 {{built at positions that are not shown to be wrapped into its cell}}
    %next = md_dist.pack %l, %h1, %cell fields(%x1, %va : !own, !own) : !state
    scf.yield %next : !state
  }
  %r:5 = md_dist.unpack %s, %cell : !state
  return
}

// -----

// The same mistake in the else-region of the test of coverage: the halo is
// rebuilt, the particles are not migrated.
!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!lay = !md_dist.layout<@dd>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, 3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @else_halo_only(%x: !vec, %v: !vec, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %l0 = md_dist.partition %x, %cell : !vec -> !lay
  %x0 = md_dist.distribute %x, %l0 image(%cell) : !vec -> !own
  %v0 = md_dist.distribute %v, %l0 : !vec -> !own
  %h0 = md_dist.halo %l0, %x0, %cell : !own -> !halo
  %s0 = md_dist.pack %l0, %h0, %cell fields(%x0, %v0 : !own, !own) : !state
  %s = scf.for %i = %c0 to %n step %c1 iter_args(%sa = %s0) -> (!state) {
    %l, %h, %xr, %xa, %va = md_dist.unpack %sa, %cell : !state
    // expected-note @+1 {{the positions of the halo}}
    %x1 = md_dist.map_particles gather(%xa, %va : !own, !own) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
      %y = arith.addf %x_i, %v_i : vector<3xf64>
      md_dist.yield %y : vector<3xf64>
    } : !own
    %ok = md_dist.covered %h, %xr, %x1, %cell cutoff(2.0) : !halo, !own -> i1
    %next = scf.if %ok -> (!state) {
      %kept = md_dist.pack %l, %h, %cell fields(%x1, %va : !own, !own) : !state
      scf.yield %kept : !state
    } else {
      %h2 = md_dist.halo %l, %x1, %cell : !own -> !halo
      // expected-error @+1 {{built at positions that are not shown to be wrapped into its cell}}
      %rebuilt = md_dist.pack %l, %h2, %cell fields(%x1, %va : !own, !own) : !state
      scf.yield %rebuilt : !state
    }
    scf.yield %next : !state
  }
  %r:5 = md_dist.unpack %s, %cell : !state
  return
}

// -----

// Positions distributed without 'image' are not shown to lie in the cell.
!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!gho = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @not_imaged(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  // expected-note @+1 {{the positions of the halo}}
  %x0 = md_dist.distribute %x, %l : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  %xg = md_dist.halo_copy %x0, %h image(%cell) : !own, !halo -> !gho
  // expected-error @+1 {{built at positions that are not shown to be wrapped into its cell}}
  %n = md_dist.neighborhood %x0, %xg, %h cutoff(2.0)
      : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  return
}

// -----

// A migration whose step forgets to rebuild the halo: the old epoch's halo
// with the new epoch's fields.
!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!lay = !md_dist.layout<@dd>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, 3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @halo_of_old_epoch(%x: !vec, %v: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !lay
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %v0 = md_dist.distribute %v, %l : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  %l2 = md_dist.migrate %l, %x0, %cell : !lay, !own -> !lay
  %x2 = md_dist.redistribute %x0, %l, %l2 image(%cell) : !own, !lay, !lay -> !own
  %v2 = md_dist.redistribute %v0, %l, %l2 : !own, !lay, !lay -> !own
  // expected-error @+1 {{the halo is of another layout (epoch of ownership) than the state}}
  %s = md_dist.pack %l2, %h, %cell fields(%x2, %v2 : !own, !own) : !state
  %r:5 = md_dist.unpack %s, %cell : !state
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @field_type(%x: !vec, %v: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %v0 = md_dist.distribute %v, %l : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  // expected-error @+1 {{field #1 is '!md_dist.field<@dd, owned, 3 x f64>', the state carries '!md_dist.field<@dd, owned, f64>'}}
  %s = md_dist.pack %l, %h, %cell fields(%x0, %v0 : !own, !own) : !state
  %r:5 = md_dist.unpack %s, %cell : !state
  return
}

// -----

// The unpack's witness holds at the state's cutoff, not above it.
!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!gho = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @unpack_cutoff(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %x0, %cell : !own -> !halo
  %s = md_dist.pack %l, %h, %cell fields(%x0 : !own) : !state
  %l1, %h1, %r1, %x1 = md_dist.unpack %s, %cell : !state
  %xg = md_dist.halo_copy %x1, %h1 image(%cell) : !own, !halo -> !gho
  // expected-error @+1 {{the state covers its positions at the cutoff 2, less than 2.2}}
  %n = md_dist.neighborhood %x1, %xg, %h1 cutoff(2.2)
      : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  return
}
