// D332: single-mover x/y/z and strict binary-exact half-skin oracle.
// Reach 2.25 minus cutoff 2.0 gives half-skin 0.125 exactly.
// The 0.0625 move is covered; 0.25 in each axis and exactly 0.125 are not.
// RUN: mdir-opt --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.25 width=64" %md_exec_passes %s \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.25>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func private @printI64(i64)
func.func private @printNewline()
func.func @main() {
  %n = arith.constant 2 : index
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %a = arith.constant 0.01 : f64
  %b = arith.constant 6.5 : f64
  %four = arith.constant 4.0 : f64
  %edge = arith.constant 8.0 : f64
  %base = memref.alloc(%n) : memref<?x3xf64>
  memref.store %a, %base[%i0, %i0] : memref<?x3xf64>
  memref.store %b, %base[%i1, %i0] : memref<?x3xf64>
  scf.for %i = %i0 to %n step %i1 {
    memref.store %four, %base[%i, %i1] : memref<?x3xf64>
    memref.store %four, %base[%i, %i2] : memref<?x3xf64>
  }
  %small = memref.alloc(%n) : memref<?x3xf64>
  %x = memref.alloc(%n) : memref<?x3xf64>
  %y = memref.alloc(%n) : memref<?x3xf64>
  %z = memref.alloc(%n) : memref<?x3xf64>
  %exact = memref.alloc(%n) : memref<?x3xf64>
  scf.for %i = %i0 to %n step %i1 {
    scf.for %k = %i0 to %i2 step %i1 {
      %v = memref.load %base[%i, %k] : memref<?x3xf64>
      memref.store %v, %small[%i, %k] : memref<?x3xf64>
      memref.store %v, %x[%i, %k] : memref<?x3xf64>
      memref.store %v, %y[%i, %k] : memref<?x3xf64>
      memref.store %v, %z[%i, %k] : memref<?x3xf64>
      memref.store %v, %exact[%i, %k] : memref<?x3xf64>
    }
    %v2 = memref.load %base[%i, %i2] : memref<?x3xf64>
    memref.store %v2, %small[%i, %i2] : memref<?x3xf64>
    memref.store %v2, %x[%i, %i2] : memref<?x3xf64>
    memref.store %v2, %y[%i, %i2] : memref<?x3xf64>
    memref.store %v2, %z[%i, %i2] : memref<?x3xf64>
    memref.store %v2, %exact[%i, %i2] : memref<?x3xf64>
  }
  %small_value = arith.constant 6.5625 : f64
  memref.store %small_value, %small[%i1, %i0] : memref<?x3xf64>
  %x_value = arith.constant 6.75 : f64
  memref.store %x_value, %x[%i1, %i0] : memref<?x3xf64>
  %y_value = arith.constant 4.25 : f64
  memref.store %y_value, %y[%i1, %i1] : memref<?x3xf64>
  %z_value = arith.constant 4.25 : f64
  memref.store %z_value, %z[%i1, %i2] : memref<?x3xf64>
  %exact_value = arith.constant 6.625 : f64
  memref.store %exact_value, %exact[%i1, %i0] : memref<?x3xf64>
  %x0 = mdrt.from_buffer %base : memref<?x3xf64> to !vec
  %small_field = mdrt.from_buffer %small : memref<?x3xf64> to !vec
  %x_field = mdrt.from_buffer %x : memref<?x3xf64> to !vec
  %y_field = mdrt.from_buffer %y : memref<?x3xf64> to !vec
  %z_field = mdrt.from_buffer %z : memref<?x3xf64> to !vec
  %exact_field = mdrt.from_buffer %exact : memref<?x3xf64> to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %l = md_dist.partition %x0, %cell : !vec -> !md_dist.layout<@dd>
  %o0 = md_dist.distribute %x0, %l image(%cell) : !vec -> !own
  %small_owned = md_dist.distribute %small_field, %l image(%cell) : !vec -> !own
  %x_owned = md_dist.distribute %x_field, %l image(%cell) : !vec -> !own
  %y_owned = md_dist.distribute %y_field, %l image(%cell) : !vec -> !own
  %z_owned = md_dist.distribute %z_field, %l image(%cell) : !vec -> !own
  %exact_owned = md_dist.distribute %exact_field, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o0, %cell : !own -> !halo
  %small_ok = md_dist.covered %h, %o0, %small_owned, %cell cutoff(2.0) : !halo, !own -> i1
  %small_int = arith.extui %small_ok : i1 to i64
  func.call @printI64(%small_int) : (i64) -> ()
  %x_ok = md_dist.covered %h, %o0, %x_owned, %cell cutoff(2.0) : !halo, !own -> i1
  %x_int = arith.extui %x_ok : i1 to i64
  func.call @printI64(%x_int) : (i64) -> ()
  %y_ok = md_dist.covered %h, %o0, %y_owned, %cell cutoff(2.0) : !halo, !own -> i1
  %y_int = arith.extui %y_ok : i1 to i64
  func.call @printI64(%y_int) : (i64) -> ()
  %z_ok = md_dist.covered %h, %o0, %z_owned, %cell cutoff(2.0) : !halo, !own -> i1
  %z_int = arith.extui %z_ok : i1 to i64
  func.call @printI64(%z_int) : (i64) -> ()
  %exact_ok = md_dist.covered %h, %o0, %exact_owned, %cell cutoff(2.0) : !halo, !own -> i1
  %exact_int = arith.extui %exact_ok : i1 to i64
  func.call @printI64(%exact_int) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}
// CHECK: 10000
