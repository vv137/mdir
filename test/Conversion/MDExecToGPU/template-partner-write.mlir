// RUN: mdir-opt %s --convert-md-exec-to-gpu --verify-each | FileCheck %s
// The partner of a batch entry is read once, at the batch head, and its
// force is written after the sixteen turns from that value: the entry is
// not read again, and the three site rows of the partner are all read
// before the first atomic, so that their loads overlap instead of each
// waiting behind the reductions of the site before it.
// CHECK-LABEL: func.func @execute
// CHECK: scf.for %{{.*}} step %c32_i32
// CHECK: %[[PARTNER:[0-9]+]] = memref.load %[[PAIRS:[0-9]+]][%{{.*}}] : memref<?xi32, 1>
// CHECK: scf.for %{{.*}} to %c16_i32
// CHECK-NOT: memref.load %[[PAIRS]][
// CHECK: arith.muli %[[PARTNER]], %c3_i32
// CHECK-NOT: red.relaxed
// CHECK: memref.load %[[MEMBERS:[0-9]+]][
// CHECK-NOT: red.relaxed
// CHECK: memref.load %[[MEMBERS]][
// CHECK-NOT: red.relaxed
// CHECK: memref.load %[[MEMBERS]][
// CHECK-NOT: memref.load
// CHECK: red.relaxed.gpu.global.add.f32
// CHECK: red.relaxed.gpu.global.add.f32
// CHECK: red.relaxed.gpu.global.add.f32
// CHECK-NOT: memref.load
// CHECK: red.relaxed.gpu.global.add.f32
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
func.func @execute(%types: memref<?xi32, 1>, %water: memref<?x4xi32>,
    %x: memref<?x3xf64, 1>, %h: !md.cell, %f: memref<?x3xf32, 1>,
    %sums: memref<?xf64, 1>, %partials: memref<?xf64, 1>) {
  %list = md_exec.empty_segmented_pairs %types, %water
      cutoff(1.0) skin(0.2) extent(0.2) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
  md_exec.template_for %list, %x, %h outs(%f : memref<?x3xf32, 1>)
      pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75)
      water {
      ^bb0(%d: vector<3xf32>, %a1: vector<3xf32>, %a2: vector<3xf32>,
           %b1: vector<3xf32>, %b2: vector<3xf32>):
        %back = arith.negf %d : vector<3xf32>
        %z = arith.constant dense<0.0> : vector<3xf32>
        md_exec.yield %d, %z, %z, %back, %z, %z, %d : vector<3xf32>, vector<3xf32>, vector<3xf32>, vector<3xf32>, vector<3xf32>, vector<3xf32>, vector<3xf32>
      } mixed {
      ^bb0(%m: vector<3xf32>, %h1: vector<3xf32>, %h2: vector<3xf32>):
        md_exec.yield
      } atoms {
      ^bb0(%s: vector<3xf32>):
        md_exec.yield
      } : !list, memref<?x3xf64, 1>
  return
}
