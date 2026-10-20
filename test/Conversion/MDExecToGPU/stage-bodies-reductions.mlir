// The sums of rows in the batched bodies of neural stages
// (D303, docs/ml4-ir.md, Section 8.3): a sum over the 64
// components of a product, in the form of stablehlo-legalize-to-linalg (a
// product, and a sum of one operand from a fill of zero), becomes a kernel
// of a warp for each of the 8 rows. Lane l adds the terms l and l + 32 in
// order, the warp combines its lanes by the butterfly of D301 (offsets 16,
// 8, 4, 2, 1), and the first lane stores zero plus the sum. A sum over 3
// components (fewer than a warp's lanes) can opt into a thread for each
// element, with its accumulator held in a register (D[ml4-short-reductions]).
//
// RUN: mdir-opt %s --md-exec-lower-stage-bodies='short-reductions=true' | FileCheck %s
// RUN: mdir-opt %s --md-exec-lower-stage-bodies | FileCheck %s --check-prefix=LOOPS
// RUN: mdir-opt %s --md-exec-lower-stage-bodies='device=false' | FileCheck %s --check-prefix=CPU

md.particle_set @atoms
md.stage @s forward(@s.f) : (vector<64xf64>, vector<64xf64>) -> (f64, vector<16xf64>)
md.stage @dependent forward(@dependent.f) : (vector<3xf64>) -> f64

#id = affine_map<(r, f) -> (r, f)>
#row = affine_map<(r, f) -> (r)>
#v = affine_map<(r, f, c) -> (r, c, f)>
#vo = affine_map<(r, f, c) -> (r, f)>

func.func @s.f(%x: tensor<8x64xf32>, %y: tensor<8x64xf32>)
    -> (tensor<8xf32>, tensor<8x16xf32>) {
  %zero = arith.constant 0.0 : f32
  %pe = tensor.empty() : tensor<8x64xf32>
  %p = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %y : tensor<8x64xf32>, tensor<8x64xf32>) outs(%pe : tensor<8x64xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32
  } -> tensor<8x64xf32>
  %e = tensor.empty() : tensor<8xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  %s = linalg.generic {indexing_maps = [#id, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%p : tensor<8x64xf32>) outs(%f : tensor<8xf32>) {
  ^bb0(%a: f32, %o: f32):
    %t = arith.addf %a, %o : f32
    linalg.yield %t : f32
  } -> tensor<8xf32>
  %xv = tensor.expand_shape %x [[0], [1, 2]] output_shape [8, 4, 16]
      : tensor<8x64xf32> into tensor<8x4x16xf32>
  %xs = tensor.extract_slice %xv[0, 0, 0] [8, 3, 16] [1, 1, 1]
      : tensor<8x4x16xf32> to tensor<8x3x16xf32>
  %e2 = tensor.empty() : tensor<8x16xf32>
  %f2 = linalg.fill ins(%zero : f32) outs(%e2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %n = linalg.generic {indexing_maps = [#v, #vo],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%xs : tensor<8x3x16xf32>) outs(%f2 : tensor<8x16xf32>) {
  ^bb0(%a: f32, %o: f32):
    %q = arith.mulf %a, %a : f32
    %t = arith.addf %o, %q : f32
    linalg.yield %t : f32
  } -> tensor<8x16xf32>
  return %s, %n : tensor<8xf32>, tensor<8x16xf32>
}

// The term depends on the accumulator, so it must keep the original loop.
func.func @dependent.f(%x: tensor<8x3xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %sum = linalg.generic {indexing_maps = [#id, #row],
                         iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<8x3xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%a: f32, %o: f32):
    %dependent = arith.mulf %a, %o : f32
    %next = arith.addf %o, %dependent : f32
    linalg.yield %next : f32
  } -> tensor<8xf32>
  return %sum : tensor<8xf32>
}


// CHECK-LABEL: func.func @md_stage_s_f(
// CHECK-SAME: %[[X:[^:]*]]: memref<8x64xf32>, %[[Y:[^:]*]]: memref<8x64xf32>, %[[S:[^:]*]]: memref<8xf32>, %[[N:[^:]*]]: memref<8x16xf32>)
// The sum of the rows: one block of 8 warps.
// CHECK: gpu.launch blocks({{.*}}) in ({{.*}} = %c1{{.*}}) threads({{.*}}) in ({{.*}} = %c256
// CHECK: %[[LANE:.*]] = arith.remui %{{.*}}, %c32
// CHECK: memref.load %[[X]]
// CHECK: memref.load %[[Y]]
// CHECK: arith.mulf
// CHECK: %[[NEXT:.*]] = arith.addi %[[LANE]], %c32
// CHECK: %[[SUM:.*]] = scf.for %[[K:.*]] = %[[NEXT]] to %c64 step %c32 iter_args
// CHECK: memref.load %[[X]][%{{.*}}, %[[K]]]
// CHECK: memref.load %[[Y]][%{{.*}}, %[[K]]]
// CHECK: scf.yield
// CHECK: gpu.shuffle xor %[[SUM]], %c16_i32, %c32_i32 : f32
// CHECK: gpu.shuffle xor {{.*}}, %c8_i32, %c32_i32 : f32
// CHECK: gpu.shuffle xor {{.*}}, %c4_i32, %c32_i32 : f32
// CHECK: gpu.shuffle xor {{.*}}, %c2_i32, %c32_i32 : f32
// CHECK: gpu.shuffle xor {{.*}}, %c1_i32, %c32_i32 : f32
// CHECK: %[[TOTAL:.*]] = arith.addf
// CHECK: %[[FIRST:.*]] = arith.cmpi eq, %[[LANE]], %c0
// CHECK: scf.if %[[FIRST]]
// CHECK: %[[OUT:.*]] = arith.addf {{.*}}%[[TOTAL]]{{.*}} : f32
// CHECK: memref.store %[[OUT]], %[[S]]
// The sum over 3: a thread for each element.
// CHECK: gpu.launch
// CHECK-NOT: gpu.shuffle
// CHECK-NOT: scf.for
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: memref.store {{.*}}, %[[N]]
// CHECK-NOT: gpu.launch
// CHECK: return
// CHECK-LABEL: func.func @md_stage_dependent_f(
// CHECK: gpu.launch
// CHECK: scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK: return

// LOOPS-LABEL: func.func @md_stage_s_f(
// LOOPS: scf.for %{{.*}} = %c0 to %c3 step %c1
// LOOPS-LABEL: func.func @md_stage_dependent_f(
// LOOPS: scf.for %{{.*}} = %c0 to %c3 step %c1

// CPU-LABEL: func.func @md_stage_s_f(
// CPU-NOT: gpu.launch
// CPU: scf.parallel
// CPU: scf.for %{{.*}} = %c0 to %c3 step %c1
// CPU-LABEL: func.func @md_stage_dependent_f(
// CPU: scf.parallel
// CPU: scf.for %{{.*}} = %c0 to %c3 step %c1
