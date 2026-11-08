// RUN: mdir-opt %s --verify-each -o - | FileCheck %s
// The relation stores only structural metadata. Continuous operands remain
// explicit and the ordinary PME builder does not emit this operation.
md.particle_set @atoms

// CHECK-LABEL: func.func @support
func.func @support(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> !md.grid_stencil<@atoms, [4, 6, 8], order(4), window("bspline")> {
  // CHECK: md.grid_stencil %{{.*}}, %{{.*}} : !md.field<@atoms, 3 x f64> -> !md.grid_stencil<@atoms, [4, 6, 8], order(4), window("bspline")>
  %s = md.grid_stencil %x, %cell : !md.field<@atoms, 3 x f64>
    -> !md.grid_stencil<@atoms, [4, 6, 8], order(4), window("bspline")>
  return %s : !md.grid_stencil<@atoms, [4, 6, 8], order(4), window("bspline")>
}
