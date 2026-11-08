// RUN: mdir-opt %s --convert-md-exec-to-gpu --verify-each | FileCheck %s --implicit-check-not=md_exec.template_for --implicit-check-not='!mdrt.segmented_pairs'
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
// CHECK-LABEL: func.func @execute
// CHECK: call @mdrtTemplatePairsCreateMembers
// CHECK: call @mdrtTemplatePairsMetadata
// CHECK: call @mdrtTemplatePairsBuffer
// CHECK: call @mdrtTemplatePairsOrder
// CHECK: call @mdrtTemplatePairsFrames
// CHECK: call @mdrtTemplatePairsLayoutValue
// CHECK: gpu.launch
// CHECK: scf.for
// CHECK: scf.if
// CHECK: gpu.shuffle
// CHECK: gpu.terminator
// CHECK: return
func.func @execute(%types: memref<?xi32, 1>, %water: memref<?x4xi32>,
    %x: memref<?x3xf64, 1>, %h: !md.cell,
    %sums: memref<?xf64, 1>, %partials: memref<?xf64, 1>) -> f64 {
  %list = md_exec.empty_segmented_pairs %types, %water
      cutoff(1.0) skin(0.2) extent(0.2) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
  %zero = arith.constant 0.0 : f64
  %energy = md_exec.template_for %list, %x, %h
      reduce(%zero : f64)
      scratch(%sums, %partials : memref<?xf64, 1>, memref<?xf64, 1>)
      pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75)
      water {
      ^bb0(%d: vector<3xf32>, %a1: vector<3xf32>, %a2: vector<3xf32>,
           %b1: vector<3xf32>, %b2: vector<3xf32>):
        %square = arith.mulf %d, %d : vector<3xf32>
        %r2 = vector.reduction <add>, %square : vector<3xf32> into f32
        %wide = arith.extf %r2 : f32 to f64
        md_exec.yield %wide, %d : f64, vector<3xf32>
      } mixed {
      ^bb0(%d: vector<3xf32>, %h1: vector<3xf32>, %h2: vector<3xf32>):
        md_exec.yield
      } atoms {
      ^bb0(%d: vector<3xf32>):
        md_exec.yield
      } : !list, memref<?x3xf64, 1> -> f64
  return %energy : f64
}
