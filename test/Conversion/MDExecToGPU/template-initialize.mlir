// RUN: mdir-opt %s --convert-md-exec-to-gpu --verify-each | FileCheck %s --implicit-check-not=md_exec.empty_segmented_pairs --implicit-check-not='!mdrt.segmented_pairs'
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
// CHECK: mdrt.template_pairs_abi = 1 : i32
// CHECK-LABEL: func.func @initialize
// CHECK-SAME: -> i64
// CHECK: memref.dim
// CHECK: arith.subi
// CHECK: arith.maxui
// CHECK: %[[HANDLE:.*]] = call @mdrtTemplatePairsCreateMembers
// CHECK: return %[[HANDLE]] : i64
func.func @initialize(%types: memref<?xi32, 1>, %water: memref<?x4xi32>) -> !list {
  %list = md_exec.empty_segmented_pairs %types, %water
      cutoff(1.0) skin(0.2) extent(0.1) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
  return %list : !list
}
// CHECK: func.func private @mdrtTemplatePairsCreateMembers
// CHECK-SAME: llvm.emit_c_interface
