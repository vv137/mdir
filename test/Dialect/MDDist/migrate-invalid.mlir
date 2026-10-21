// RUN: mdir-opt %s -split-input-file -verify-diagnostics

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

func.func @stale_field(%x: !md.field<@atoms, 3 x f64>,
                       %cell: !md.cell) {
  // expected-note @+1 {{the layout of the field}}
  %l0 = md_dist.partition %x, %cell
      : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l0
      : !md.field<@atoms, 3 x f64> -> !md_dist.field<@dd, owned, 3 x f64>
  // expected-note @+1 {{the layout of the op}}
  %l1 = md_dist.migrate %l0, %xo, %cell
      : !md_dist.layout<@dd>, !md_dist.field<@dd, owned, 3 x f64>
      -> !md_dist.layout<@dd>
  // expected-error @+1 {{the field is of another layout (epoch of ownership)}}
  %bad = md_dist.collect %xo, %l1
      : !md_dist.field<@dd, owned, 3 x f64> -> !md.field<@atoms, 3 x f64>
  return
}

// -----

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

func.func @wrong_migration_source(%x: !md.field<@atoms, 3 x f64>,
                                  %cell: !md.cell) {
  %l0 = md_dist.partition %x, %cell
      : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
  %l_other = md_dist.partition %x, %cell
      : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l0
      : !md.field<@atoms, 3 x f64> -> !md_dist.field<@dd, owned, 3 x f64>
  %other = md_dist.distribute %x, %l_other
      : !md.field<@atoms, 3 x f64> -> !md_dist.field<@dd, owned, 3 x f64>
  %l1 = md_dist.migrate %l_other, %other, %cell
      : !md_dist.layout<@dd>, !md_dist.field<@dd, owned, 3 x f64>
      -> !md_dist.layout<@dd>
  // expected-error @+1 {{the new layout must come from migration of the field's old layout}}
  %bad = md_dist.redistribute %xo, %l0, %l1
      : !md_dist.field<@dd, owned, 3 x f64>, !md_dist.layout<@dd>,
        !md_dist.layout<@dd> -> !md_dist.field<@dd, owned, 3 x f64>
  return
}
