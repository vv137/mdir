// RUN: mdir-opt %s --convert-md-exec-to-gpu --verify-each | FileCheck %s --implicit-check-not=md_exec.empty_segmented_pairs --implicit-check-not='!mdrt.segmented_pairs'
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
// CHECK: mdrt.template_pairs_abi = 9 : i32
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
// Three- and four-site membership use one canonical descriptor ABI.
// CHECK-LABEL: func.func @initialize_three
// CHECK: memref.cast
// CHECK: call @mdrtTemplatePairsCreateMembers
// CHECK: return
func.func @initialize_three(%types: memref<?xi32, 1>, %water: memref<?x3xi32>) -> !list {
  %list = md_exec.empty_segmented_pairs %types, %water
      cutoff(1.0) skin(0.2) extent(0.1) alpha(0.0)
      : memref<?xi32, 1>, memref<?x3xi32> -> !list
  return %list : !list
}
// Already-buffered device metadata is copied once, before construction;
// the initializer's temporary host buffers are released after the call.
// CHECK-LABEL: func.func @initialize_device
// CHECK: gpu.memcpy
// CHECK: gpu.memcpy
// CHECK: call @mdrtTemplatePairsCreateMembers
// CHECK: memref.dealloc
// CHECK: memref.dealloc
// CHECK: return
func.func @initialize_device(%types: memref<?xi32, 1>, %water: memref<?x4xi32, 1>,
                             %excluded: memref<?x2xi32, 1>) -> !list {
  %list = md_exec.empty_segmented_pairs %types, %water
      exclude(%excluded : memref<?x2xi32, 1>)
      cutoff(1.0) skin(0.2) extent(0.1) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32, 1> -> !list
  return %list : !list
}
// Solute units: the (member, anchor) rows are copied like the exclusions.
// CHECK-LABEL: func.func @initialize_anchored
// CHECK: gpu.memcpy
// CHECK: gpu.memcpy
// CHECK: gpu.memcpy
// CHECK: call @mdrtTemplatePairsCreateMembers
// CHECK: return
func.func @initialize_anchored(%types: memref<?xi32, 1>, %water: memref<?x4xi32, 1>,
                               %excluded: memref<?x2xi32, 1>,
                               %anchored: memref<?x2xi32, 1>) -> !list {
  %list = md_exec.empty_segmented_pairs %types, %water
      exclude(%excluded : memref<?x2xi32, 1>)
      anchor(%anchored : memref<?x2xi32, 1>) radius(0.109)
      cutoff(1.0) skin(0.2) extent(0.1) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32, 1> -> !list
  return %list : !list
}
// CHECK: func.func private @mdrtTemplatePairsCreateMembers(memref<?x?xi32> {mdrt.read_only}, i64, i64, memref<?x?xi32> {mdrt.read_only}, memref<?x?xi32> {mdrt.read_only}) -> i64
// CHECK-SAME: llvm.emit_c_interface
