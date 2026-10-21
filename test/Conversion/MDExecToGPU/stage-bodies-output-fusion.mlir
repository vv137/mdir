// An opt-in pointwise body with two identical results writes both caller
// buffers in its producer kernel when its inputs are local allocations.
// A producer reading a caller input keeps the copies: either output could
// alias that input at a call site.
//
// RUN: mdir-opt %s --md-exec-lower-stage-bodies='short-reductions=true' | FileCheck %s --check-prefix=FUSED
// RUN: mdir-opt %s --md-exec-lower-stage-bodies | FileCheck %s --check-prefix=DEFAULT
// RUN: mdir-opt %s --md-exec-lower-stage-bodies='fuse=false short-reductions=true' | FileCheck %s --check-prefix=NOFUSE
// RUN: mdir-opt %s --md-exec-lower-stage-bodies='device=false short-reductions=true' | FileCheck %s --check-prefix=CPU

md.particle_set @atoms
md.stage @local forward(@local.f) : (vector<4xf64>) -> (vector<4xf64>, vector<4xf64>)
md.stage @external forward(@external.f) : (vector<4xf64>) -> (vector<4xf64>, vector<4xf64>)
md.stage @unguarded forward(@unguarded.f) : (vector<4xf64>) -> (vector<4xf64>, vector<4xf64>)

#id = affine_map<(r, f) -> (r, f)>

func.func @local.f(%x: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %weight = arith.constant dense<1.0> : tensor<4x4xf32>
  %empty = tensor.empty() : tensor<8x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x4xf32>) -> tensor<8x4xf32>
  %sum = linalg.matmul ins(%x, %weight : tensor<8x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<8x4xf32>) -> tensor<8x4xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
      ins(%sum : tensor<8x4xf32>) outs(%empty : tensor<8x4xf32>) {
  ^bb0(%in: f32, %unused: f32):
    %next = arith.addf %in, %one : f32
    linalg.yield %next : f32
  } -> tensor<8x4xf32>
  return %out, %out : tensor<8x4xf32>, tensor<8x4xf32>
}

func.func @external.f(%x: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  %one = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<8x4xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<8x4xf32>) outs(%empty : tensor<8x4xf32>) {
  ^bb0(%in: f32, %unused: f32):
    %next = arith.addf %in, %one : f32
    linalg.yield %next : f32
  } -> tensor<8x4xf32>
  return %out, %out : tensor<8x4xf32>, tensor<8x4xf32>
}

// The extra symbol use may call with overlapping result views. It prevents
// the lowering from assuming the generated stage caller's disjoint buffers.
func.func @keeper() attributes {test.possible_alias = @unguarded.f} {
  return
}

func.func @unguarded.f(%x: tensor<8x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %weight = arith.constant dense<1.0> : tensor<4x4xf32>
  %empty = tensor.empty() : tensor<8x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x4xf32>) -> tensor<8x4xf32>
  %sum = linalg.matmul ins(%x, %weight : tensor<8x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<8x4xf32>) -> tensor<8x4xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
      ins(%sum : tensor<8x4xf32>) outs(%empty : tensor<8x4xf32>) {
  ^bb0(%in: f32, %unused: f32):
    %next = arith.addf %in, %one : f32
    linalg.yield %next : f32
  } -> tensor<8x4xf32>
  return %out, %out : tensor<8x4xf32>, tensor<8x4xf32>
}

// FUSED-LABEL: func.func @md_stage_local_f(
// FUSED: call @mdrtGemmF32
// FUSED-COUNT-1: gpu.launch
// FUSED-NOT: gpu.launch
// FUSED: return
// FUSED-LABEL: func.func @md_stage_external_f(
// FUSED-COUNT-3: gpu.launch
// FUSED-NOT: gpu.launch
// FUSED: return
// FUSED-LABEL: func.func @md_stage_unguarded_f(
// FUSED-COUNT-3: gpu.launch
// FUSED-NOT: gpu.launch
// FUSED: return

// DEFAULT-LABEL: func.func @md_stage_local_f(
// DEFAULT-COUNT-3: gpu.launch
// DEFAULT-NOT: gpu.launch
// DEFAULT: return

// NOFUSE-LABEL: func.func @md_stage_local_f(
// NOFUSE-COUNT-3: gpu.launch
// NOFUSE-NOT: gpu.launch
// NOFUSE: return

// CPU-LABEL: func.func @md_stage_local_f(
// CPU-NOT: gpu.launch
// CPU: return
