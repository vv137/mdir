// D[distributed-migration]: a state used with another cell than the one its
// halo was built in is stopped at run time; the images of the halo and the
// regions of the layout were made in that cell, and this slice has no change
// of the cell on domains.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | not --crash mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt 2>&1 \
// RUN: | FileCheck %s
// CHECK: mdrt_dist: a state is used with another cell than its halo's

!vec   = !md.field<@atoms, 3 x f64>
!own   = !md_dist.field<@dd, owned, 3 x f64>
!halo  = !md_dist.halo<@dd, 2.4>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64]>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 2 : index
  %one = arith.constant 1.0 : f64
  %edge = arith.constant 8.0 : f64
  %wider = arith.constant 8.5 : f64
  %buffer = memref.alloc(%n) : memref<?x3xf64>
  scf.for %i = %c0 to %n step %c1 {
    scf.for %k = %c0 to %n step %c1 {
      memref.store %one, %buffer[%i, %k] : memref<?x3xf64>
    }
    memref.store %one, %buffer[%i, %n] : memref<?x3xf64>
  }
  %x = mdrt.from_buffer %buffer : memref<?x3xf64> to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %other = md.orthorhombic_cell %wider, %edge, %edge
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %s = md_dist.pack %l, %h, %cell fields(%xo : !own) : !state
  %l1, %h1, %r1, %x1 = md_dist.unpack %s, %other : !state
  return
}
