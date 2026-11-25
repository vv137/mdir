// Both endpoints are ghosts in the midpoint domain [2,3). The reversed
// exclusion names the second pair; it has the same length as the first.
// RUN: mdir-opt %s --md-distribute="grid=8,1,1 traversal=midpoint" %md_passes --md-exec-assign-algebra --convert-md-dist-to-loops --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes | mlir-opt %lower_loops_to_llvm | mlir-runner -e main --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
!excluded = !md.relation<@atoms, 2, unordered, @ex>
md.particle_set @atoms
md.tuple_set @ex on(@atoms) arity(2) orientation(unordered)
md.potential @potential(%x: !vec, %cell: !md.cell, %ex: !excluded) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) exclude(%ex : !excluded) : !vec -> !pairs
  %e = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !pairs, !vec -> f64
  md.return %e : f64
}
memref.global "private" constant @coords : memref<4x3xf64> = dense<[[1.75, 1.0, 1.0], [3.25, 1.0, 1.0], [7.75, 5.0, 1.0], [1.25, 5.0, 1.0]]>
memref.global "private" constant @members : memref<1x2xi32> = dense<[[3, 2]]>
func.func private @printF64(f64)
func.func @main() {
  %a = memref.get_global @coords : memref<4x3xf64>
  %xb = memref.cast %a : memref<4x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
  %m = memref.get_global @members : memref<1x2xi32>
  %eb = memref.cast %m : memref<1x2xi32> to memref<?x2xi32>
  %ex = mdrt.from_buffer %eb : memref<?x2xi32> to !excluded
  %edge = arith.constant 8.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %energy = md.evaluate @potential(%x, %cell, %ex) request [energy] : (!vec, !md.cell, !excluded) -> f64
  func.call @printF64(%energy) : (f64) -> ()
  // CHECK: 1.5
  return
}
