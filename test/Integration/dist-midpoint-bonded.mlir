// Both evaluations use the same differentiated potential. Its two active
// sites are ghosts of the middle domain. The middle site has zero force.
// RUN: mdir-opt %s --md-differentiate --md-distribute="derivatives=true grid=8,1,1 traversal=midpoint reach=1.0" --md-inline --md-exec-assign-algebra --convert-md-dist-to-loops --convert-md-to-md-exec %md_exec_passes | mlir-opt %lower_loops_to_llvm | mlir-runner -e main --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s
// CHECK: 1.125
// CHECK-NEXT: 1.5
// CHECK-NEXT: -2.25
!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!layout = !md_dist.layout<@dd>
!halo = !md_dist.halo<@dd, 1.0>
!tuple = !md.relation<@atoms, 3, reversal, @angles>
md.particle_set @atoms
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
md_dist.decomposition @dd on(@atoms) grid([8, 1, 1])
memref.global "private" constant @coords : memref<3x3xf64> = dense<[[1.75, 1.0, 1.0], [2.5, 1.1, 1.0], [3.25, 1.0, 1.0]]>
memref.global "private" constant @members : memref<1x3xi32> = dense<[[2, 1, 0]]>
func.func private @printF64(f64)
func.func private @printNewline()
md.potential @harmonic(%x: !vec, %cell: !md.cell, %g: !tuple) -> f64 {
  %u = md.sum_tuples %g, %x, %cell coordinates(distance(0, 2)) {
  ^bb0(%r: f64):
    %half = arith.constant 0.5 : f64
    %sq = arith.mulf %r, %r : f64
    %e = arith.mulf %half, %sq : f64
    md.yield %e : f64
  } : !tuple, !vec -> f64
  md.return %u : f64
}
func.func @main() {
  %a = memref.get_global @coords : memref<3x3xf64>
  %xb = memref.cast %a : memref<3x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
  %m = memref.get_global @members : memref<1x3xi32>
  %gb = memref.cast %m : memref<1x3xi32> to memref<?x3xi32>
  %g = mdrt.from_buffer %gb : memref<?x3xi32> to !tuple
  %edge = arith.constant 8.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @harmonic(%x, %cell, %g) request [energy, forces, virial]
    : (!vec, !md.cell, !tuple) -> (f64, !vec, vector<9xf64>)
  %forces = mdrt.to_buffer %f : !vec to memref<?x3xf64>
  %zero = arith.constant 0 : index
  %fx = memref.load %forces[%zero, %zero] : memref<?x3xf64>
  %xx = vector.extract %w[0] : f64 from vector<9xf64>
  call @printF64(%u) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%fx) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%xx) : (f64) -> ()
  return
}
