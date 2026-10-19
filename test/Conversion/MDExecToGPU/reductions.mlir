// The reductions of a device through the shared emitters
// (D[reduction-algebra], ReductionEmitter.h): a global sum of a loop is a
// block reduction in each of the parts (each warp by a butterfly of
// `shuffle xor` at offsets 16, 8, 4, 2, 1, then the warps in their order
// through block memory after one barrier), and one block over the parts
// (D150). The order depends on the count alone, so the same code meets the
// algebran `any_order` and `fixed_order`.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true \
// RUN:     --convert-md-exec-to-gpu | FileCheck %s

md.particle_set @atoms

// A number of 64 bits travels as two of 32.
// CHECK-LABEL: func.func @sum(
// CHECK:         gpu.launch {{.*}} workgroup(%[[SHARED:[a-z0-9]+]] : memref<4xf64, #gpu.address_space<workgroup>>)
// CHECK:           arith.addf
// CHECK:           arith.bitcast %{{.*}} : f64 to i64
// CHECK:           gpu.shuffle xor %{{.*}}, %c16_i32, %c32_i32 : i32
// CHECK:           gpu.shuffle xor %{{.*}}, %c16_i32, %c32_i32 : i32
// CHECK:           arith.addf
// CHECK:           gpu.shuffle xor %{{.*}}, %c8_i32, %{{.*}} : i32
// CHECK:           gpu.shuffle xor %{{.*}}, %c4_i32, %{{.*}} : i32
// CHECK:           gpu.shuffle xor %{{.*}}, %c2_i32, %{{.*}} : i32
// CHECK:           gpu.shuffle xor %{{.*}}, %c1_i32, %{{.*}} : i32
// CHECK:           memref.store %{{.*}}, %[[SHARED]][
// CHECK:           gpu.barrier
// CHECK:           memref.load %[[SHARED]][
// CHECK:           memref.load %[[SHARED]][
// CHECK:           arith.addf
// CHECK:           memref.load %[[SHARED]][
// CHECK:           arith.addf
// CHECK:           memref.load %[[SHARED]][
// CHECK:           arith.addf
// CHECK:         gpu.launch {{.*}} workgroup(
// CHECK:           gpu.barrier
// CHECK:         return
func.func @sum(%v: memref<?x3xf64, 1>, %a: memref<?xf64, 1>,
               %b: memref<?xf64, 1>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %k = md_exec.particle_for ins(%v : memref<?x3xf64, 1>) reduce(%zero : f64)
      scratch(%a, %b : memref<?xf64, 1>, memref<?xf64, 1>) {
  ^bb0(%v_i: vector<3xf64>):
    %x = vector.extract %v_i[0] : f64 from vector<3xf64>
    md_exec.yield %x : f64
  } -> f64
  return %k : f64
}
