// A tuple straddles three domains. Its center lies in domain 2; both
// extreme atoms are imported there. Reducing the reach removes atom 3.
// RUN: mdir-opt %s --convert-md-dist-to-loops %md_passes %md_exec_passes | mlir-opt %lower_loops_to_llvm | mlir-runner -e main --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s
// CHECK: 2
// RUN: sed 's/halo<@dd, 1.0>/halo<@dd, 0.1>/' %s | mdir-opt --convert-md-dist-to-loops %md_passes %md_exec_passes | mlir-opt %lower_loops_to_llvm > %t.bad
// RUN: not --crash mlir-runner %t.bad -e main --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt 2>&1 | FileCheck %s --check-prefix=MISSING
// MISSING: missing tuple owner coverage: tuple 1 domain 2 lacks atom 3
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
func.func private @printI64(i64)
func.func @main() {
  %a = memref.get_global @coords : memref<3x3xf64>
  %xb = memref.cast %a : memref<3x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
  %m = memref.get_global @members : memref<1x3xi32>
  %gb = memref.cast %m : memref<1x3xi32> to memref<?x3xi32>
  %g = mdrt.from_buffer %gb : memref<?x3xi32> to !tuple
  %edge = arith.constant 8.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %l = md_dist.partition %x, %cell : !vec -> !layout
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %rows = md_dist.tuple_rows %g, %xo, %h : !tuple, !own, !halo -> memref<?x4xi64>
  %c0 = arith.constant 0 : index
  %d = memref.load %rows[%c0, %c0] : memref<?x4xi64>
  call @printI64(%d) : (i64) -> ()
  return
}
