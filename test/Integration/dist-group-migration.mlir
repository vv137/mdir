// A periodic group crosses the box face as one unit. Before migration its
// sites at 7.7, 0.1, and 0.1 have a common unwrapped center near 7.97.
// RUN: mdir-opt %s --convert-md-dist-to-loops %md_passes %md_exec_passes | mlir-opt %lower_loops_to_llvm > %t
// RUN: env MDRT_DIST_TRACE=1 mlir-runner %t -e main --entry-point-result=void --shared-libs=%mdrt 2>&1 | FileCheck %s
// CHECK: migrate: 3 faces, 0 edges, 0 corners
// RUN: sed 's/groups(%g : !group)/groups(%g, %g2 : !group, !group)/' %s | mdir-opt --convert-md-dist-to-loops %md_passes %md_exec_passes | mlir-opt %lower_loops_to_llvm > %t.bad
// RUN: not --crash mlir-runner %t.bad -e main --entry-point-result=void --shared-libs=%mdrt 2>&1 | FileCheck %s --check-prefix=COUPLED
// COUPLED: coupled constraint groups: atom 2 occurs at centers 1 and 2

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!layout = !md_dist.layout<@dd>
!group = !md.relation<@atoms, 3, ordered, @water>
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(3) orientation(ordered) disjoint
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
memref.global "private" constant @before : memref<3x3xf64> = dense<[[7.7, 1.0, 1.0], [0.1, 1.1, 1.0], [0.1, 0.9, 1.0]]>
memref.global "private" constant @after : memref<3x3xf64> = dense<[[7.9, 1.0, 1.0], [0.3, 1.1, 1.0], [0.3, 0.9, 1.0]]>
memref.global "private" constant @members : memref<1x3xi32> = dense<[[0, 1, 2]]>
memref.global "private" constant @chain : memref<1x3xi32> = dense<[[1, 2, 0]]>
func.func @main() {
  %a = memref.get_global @before : memref<3x3xf64>
  %b = memref.get_global @after : memref<3x3xf64>
  %xb = memref.cast %a : memref<3x3xf64> to memref<?x3xf64>
  %yb = memref.cast %b : memref<3x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
  %y = mdrt.from_buffer %yb : memref<?x3xf64> to !vec
  %m = memref.get_global @members : memref<1x3xi32>
  %gb = memref.cast %m : memref<1x3xi32> to memref<?x3xi32>
  %g = mdrt.from_buffer %gb : memref<?x3xi32> to !group
  %m2 = memref.get_global @chain : memref<1x3xi32>
  %gb2 = memref.cast %m2 : memref<1x3xi32> to memref<?x3xi32>
  %g2 = mdrt.from_buffer %gb2 : memref<?x3xi32> to !group
  %edge = arith.constant 8.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %l = md_dist.partition %x, %cell groups(%g : !group) : !vec -> !layout
  %owned = md_dist.distribute %y, %l : !vec -> !own
  %next = md_dist.migrate %l, %owned, %cell : !layout, !own -> !layout
  return
}
