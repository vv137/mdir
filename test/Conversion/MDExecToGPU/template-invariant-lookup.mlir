// RUN: mdir-opt %s --convert-md-exec-to-gpu --verify-each | FileCheck %s
// A lookup at constant types and the arithmetic only it feeds are taken
// once per work unit, before the partner batches and their rotation. A
// lookup indexed by the loaded types, or by the geometry, stays in the
// guarded pair body.
// CHECK-LABEL: func.func @execute
// CHECK: %[[ROW:.*]] = arith.index_cast %c7_i32 : i32 to index
// CHECK: %[[FIXED:.*]] = memref.load %{{.*}}[%[[ROW]], %{{.*}}] : memref<?x?xf32, 1>
// CHECK: %[[SCALED:.*]] = arith.mulf %[[FIXED]]
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.if
// CHECK: scf.if
// CHECK-NOT: %c7_i32
// CHECK: arith.mulf %[[SCALED]]
// CHECK-NOT: %c7_i32
// CHECK: memref.load %{{.*}} : memref<?x?xf32, 1>
// CHECK-NOT: %c7_i32
// CHECK: arith.fptosi
// CHECK: memref.load %{{.*}} : memref<?x?xf32, 1>
// CHECK: else
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
func.func @execute(%types: memref<?xi32, 1>, %water: memref<?x4xi32>,
    %x: memref<?x3xf64, 1>, %h: !md.cell, %lj: memref<?x?xf32, 1>,
    %sums: memref<?xf64, 1>, %partials: memref<?xf64, 1>) -> f64 {
  %list = md_exec.empty_segmented_pairs %types, %water
      cutoff(1.0) skin(0.2) extent(0.2) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
  %zero = arith.constant 0.0 : f64
  %energy = md_exec.template_for %list, %x, %h ins(%types : memref<?xi32, 1>)
      reduce(%zero : f64)
      scratch(%sums, %partials : memref<?xf64, 1>, memref<?xf64, 1>)
      pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75)
      water {
      ^bb0(%d: vector<3xf32>, %a1: vector<3xf32>, %a2: vector<3xf32>,
           %b1: vector<3xf32>, %b2: vector<3xf32>, %t0: i32, %t1: i32, %t2: i32,
           %t3: i32, %t4: i32, %t5: i32):
        %c7 = arith.constant 7 : i32
        %two = arith.constant 2.0 : f32
        %square = arith.mulf %d, %d : vector<3xf32>
        %r2 = vector.reduction <add>, %square : vector<3xf32> into f32
        %fixed = md.lookup %lj[%c7, %c7] : memref<?x?xf32, 1>, i32, i32 -> f32
        %scaled = arith.mulf %fixed, %two : f32
        %typed = md.lookup %lj[%t0, %t3] : memref<?x?xf32, 1>, i32, i32 -> f32
        %bin = arith.fptosi %r2 : f32 to i32
        %near = md.lookup %lj[%bin, %bin] : memref<?x?xf32, 1>, i32, i32 -> f32
        %a = arith.mulf %scaled, %r2 : f32
        %b = arith.addf %a, %typed : f32
        %c = arith.addf %b, %near : f32
        %wide = arith.extf %c : f32 to f64
        md_exec.yield %wide, %d : f64, vector<3xf32>
      } mixed {
      ^bb0(%d: vector<3xf32>, %h1: vector<3xf32>, %h2: vector<3xf32>,
           %t0: i32, %t1: i32, %t2: i32, %t3: i32):
        md_exec.yield
      } atoms {
      ^bb0(%d: vector<3xf32>, %t0: i32, %t1: i32):
        md_exec.yield
      } : !list, memref<?x3xf64, 1> -> f64
  return %energy : f64
}
