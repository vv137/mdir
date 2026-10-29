// A parameter-only all-unit generic has no row index even in an eight-row
// body. Folding its unit extent must not leave its accesses on the host.
// Resource storage keeps the parameter calculation from constant-folding
// away before this path is exercised; the single f32 weight is 2.0.
// RUN: mdir-opt %s --md-exec-lower-stage-bodies | FileCheck %s
// RUN: mdir-opt %s --md-exec-lower-stage-bodies='device=false' | FileCheck %s --check-prefix=CPU

md.particle_set @atoms
md.stage @scalar forward(@scalar.f) : (vector<4xf64>) -> vector<4xf64>

#unit = affine_map<(i) -> (i)>
#rows = affine_map<(r, f) -> (r, f)>
#parameter = affine_map<(r, f) -> (0)>

func.func @scalar.f(%x: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %weight = arith.constant dense_resource<weight> : tensor<1xf32>
  %empty = tensor.empty() : tensor<1xf32>
  %parameter = linalg.generic {indexing_maps = [#unit, #unit],
                               iterator_types = ["parallel"]}
      ins(%weight : tensor<1xf32>) outs(%empty : tensor<1xf32>) {
  ^bb0(%w: f32, %unused: f32):
    %value = math.exp %w : f32
    linalg.yield %value : f32
  } -> tensor<1xf32>
  %out = tensor.empty() : tensor<8x4xf32>
  %result = linalg.generic {indexing_maps = [#rows, #parameter, #rows],
                            iterator_types = ["parallel", "parallel"]}
      ins(%x, %parameter : tensor<8x4xf32>, tensor<1xf32>)
      outs(%out : tensor<8x4xf32>) {
  ^bb0(%xv: f32, %p: f32, %unused: f32):
    %value = arith.addf %xv, %p : f32
    linalg.yield %value : f32
  } -> tensor<8x4xf32>
  return %result : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @md_stage_scalar_f
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: gpu.launch blocks({{.*}}) in ({{.*}} = %[[ONE]], {{.*}} = %[[ONE]], {{.*}} = %[[ONE]]) threads({{.*}}) in ({{.*}} = %[[ONE]], {{.*}} = %[[ONE]], {{.*}} = %[[ONE]])
// CHECK: math.exp
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK: gpu.launch blocks
// CHECK: arith.addf
// CHECK: return
// CPU-LABEL: func.func @md_stage_scalar_f
// CPU-NOT: gpu.
// CPU: math.exp
// CPU: return

{-#
  dialect_resources: {
    builtin: {
      weight: "0x0400000000000040"
    }
  }
#-}
