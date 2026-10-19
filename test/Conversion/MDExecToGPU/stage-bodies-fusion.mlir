// The fusion of the batched bodies of neural stages (D299,
// docs/ml4-ir.md, Section 8.3): a producer of several uses, a pad of a
// slice, a reduction from a fill, and a result, in the forms that
// stablehlo-legalize-to-linalg gives them. With `fuse=false` (D293's
// lowering) each op is a kernel of its own, seven; by default the body is
// three: the producer, which the gather of the pad reads, the reduction,
// which starts from the fill's constant and evaluates the producer again,
// and the result, which evaluates the producer and the pad and is written
// in the caller's buffer.
//
// RUN: mdir-opt %s --md-exec-lower-stage-bodies="fuse=false" | FileCheck %s --check-prefix=SINGLE
// RUN: mdir-opt %s --md-exec-lower-stage-bodies | FileCheck %s

md.particle_set @atoms
md.stage @s forward(@s.f) : (vector<4xf64>) -> vector<4xf64>

#id = affine_map<(i, j) -> (i, j)>
#row = affine_map<(i, j) -> (i)>

func.func @s.f(%x: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %zero = arith.constant 0.0 : f32
  %two = arith.constant 2.0 : f32
  %e = tensor.empty() : tensor<8x4xf32>
  // Read by the reduction, the slice, and the result.
  %y = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<8x4xf32>) outs(%e : tensor<8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.mulf %in, %two : f32
    linalg.yield %m : f32
  } -> tensor<8x4xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e1 : tensor<8xf32>) -> tensor<8xf32>
  %r = linalg.generic {indexing_maps = [#id, #row],
                       iterator_types = ["parallel", "reduction"]}
      ins(%y : tensor<8x4xf32>) outs(%f : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<8xf32>
  %s = tensor.extract_slice %y[0, 0] [8, 2] [1, 1]
      : tensor<8x4xf32> to tensor<8x2xf32>
  %p = tensor.pad %s low[0, 0] high[0, 2] {
  ^bb0(%i: index, %j: index):
    tensor.yield %zero : f32
  } : tensor<8x2xf32> to tensor<8x4xf32>
  %z = linalg.generic {indexing_maps = [#id, #id, #row, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%y, %p, %r : tensor<8x4xf32>, tensor<8x4xf32>, tensor<8xf32>)
      outs(%e : tensor<8x4xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32, %out: f32):
    %ab = arith.addf %a, %b : f32
    %abc = arith.addf %ab, %c : f32
    linalg.yield %abc : f32
  } -> tensor<8x4xf32>
  return %z : tensor<8x4xf32>
}

// SINGLE-LABEL: func.func @md_stage_s_f(
// SINGLE-COUNT-7: gpu.launch blocks
// SINGLE-NOT: gpu.launch blocks
// SINGLE: return

// CHECK-LABEL: func.func @md_stage_s_f(
// CHECK-SAME: %[[X:.*]]: memref<8x4xf32>, %[[Z:.*]]: memref<8x4xf32>)
// CHECK: %[[Y:.*]] = gpu.alloc () : memref<8x4xf32>
// CHECK: gpu.launch blocks
// CHECK: memref.store {{.*}}, %[[Y]]
// CHECK: %[[R:.*]] = gpu.alloc () : memref<8xf32>
// CHECK: gpu.launch blocks
// CHECK: memref.load %[[X]]
// CHECK: arith.select
// CHECK: memref.store {{.*}}, %[[R]]
// CHECK: gpu.launch blocks
// CHECK: memref.load %[[X]]
// CHECK: memref.load %[[R]]
// CHECK: arith.select
// CHECK: memref.store {{.*}}, %[[Z]]
// CHECK-NOT: gpu.launch blocks
// CHECK-NOT: tensor.pad
// CHECK: gpu.dealloc %[[Y]]
// CHECK: gpu.dealloc %[[R]]
// CHECK-NEXT: return
