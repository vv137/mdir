// RUN: mdir-opt %s | FileCheck %s

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 2, 2])

func.func @epochs(%start: !md.field<@atoms, 3 x f64>,
                  %moved: !md.field<@atoms, 3 x f64>,
                  %velocity: !md.field<@atoms, 3 x f64>,
                  %cell: !md.cell) -> !md.field<@atoms, 3 x f64> {
  %l0 = md_dist.partition %start, %cell
      : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %moved, %l0
      : !md.field<@atoms, 3 x f64> -> !md_dist.field<@dd, owned, 3 x f64>
  %v0 = md_dist.distribute %velocity, %l0
      : !md.field<@atoms, 3 x f64> -> !md_dist.field<@dd, owned, 3 x f64>
  %l1 = md_dist.migrate %l0, %x0, %cell
      : !md_dist.layout<@dd>, !md_dist.field<@dd, owned, 3 x f64>
      -> !md_dist.layout<@dd>
  %x1 = md_dist.redistribute %x0, %l0, %l1 image(%cell)
      : !md_dist.field<@dd, owned, 3 x f64>, !md_dist.layout<@dd>,
        !md_dist.layout<@dd> -> !md_dist.field<@dd, owned, 3 x f64>
  %v1 = md_dist.redistribute %v0, %l0, %l1
      : !md_dist.field<@dd, owned, 3 x f64>, !md_dist.layout<@dd>,
        !md_dist.layout<@dd> -> !md_dist.field<@dd, owned, 3 x f64>
  %result = md_dist.collect %x1, %l1
      : !md_dist.field<@dd, owned, 3 x f64> -> !md.field<@atoms, 3 x f64>
  return %result : !md.field<@atoms, 3 x f64>
}

// CHECK: %[[NEW:.*]] = md_dist.migrate
// CHECK: md_dist.redistribute
// CHECK: md_dist.redistribute
// CHECK: md_dist.collect
