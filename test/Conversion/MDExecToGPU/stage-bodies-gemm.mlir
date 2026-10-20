// The products of the shape of a GEMM in the batched bodies of neural
// stages (D302, docs/ml4-ir.md, Section 8.3): a `linalg.matmul`
// and a generic contraction with the second operand transposed, each from a
// fill of zero, in the forms that stablehlo-legalize-to-linalg gives them.
// By default (`contractions=gemm`) each becomes a call of cuBLAS's f32 GEMM
// through the runtime, row-major (m, n, k, A, lda, strideA, transA, B, ldb,
// strideB, transB, C, ldc, strideC, beta, batch; the strides 0 and the batch
// 1 here), with beta 0 in place of the fill, and the weights (globals
// of the body) copied to the device by the runtime; one kernel is left, the
// elementwise op after them. With `contractions=loops` they are loops.
// cuBLAS against the loops on these values: the same results (measured on
// an RTX 4090 with a main of this body; the GPU tests of the stages run
// the default).
//
// RUN: mdir-opt %s --md-exec-lower-stage-bodies | FileCheck %s
// RUN: mdir-opt %s --md-exec-lower-stage-bodies="contractions=loops" | FileCheck %s --check-prefix=LOOPS
// RUN: not mdir-opt %s --md-exec-lower-stage-bodies="contractions=other" 2>&1 | FileCheck %s --check-prefix=WRONG

md.particle_set @atoms
md.stage @s forward(@s.f) : (vector<4xf64>) -> vector<6xf64>

#a = affine_map<(i, j, k) -> (i, k)>
#bt = affine_map<(i, j, k) -> (j, k)>
#c = affine_map<(i, j, k) -> (i, j)>
#id = affine_map<(i, j) -> (i, j)>

func.func @s.f(%x: tensor<8x4xf32>) -> tensor<8x6xf32> {
  %zero = arith.constant 0.0 : f32
  %w = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.5, 0.25, 0.125, 1.0, 2.0, 4.0], [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], [3.0, 0.0, 0.0, 0.0, 0.0, 7.0]]> : tensor<4x6xf32>
  %wt = arith.constant dense<[[1.0, 0.5, -1.0, 3.0], [2.0, 0.25, 1.0, 0.0], [3.0, 0.125, -1.0, 0.0], [4.0, 1.0, 1.0, 0.0], [5.0, 2.0, -1.0, 0.0], [6.0, 4.0, 1.0, 7.0]]> : tensor<6x4xf32>
  %e = tensor.empty() : tensor<8x6xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<8x6xf32>) -> tensor<8x6xf32>
  %p = linalg.matmul ins(%x, %w : tensor<8x4xf32>, tensor<4x6xf32>)
                     outs(%f : tensor<8x6xf32>) -> tensor<8x6xf32>
  %f2 = linalg.fill ins(%zero : f32) outs(%e : tensor<8x6xf32>) -> tensor<8x6xf32>
  %q = linalg.generic {indexing_maps = [#a, #bt, #c],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x, %wt : tensor<8x4xf32>, tensor<6x4xf32>) outs(%f2 : tensor<8x6xf32>) {
  ^bb0(%u: f32, %v: f32, %o: f32):
    %m = arith.mulf %u, %v : f32
    %s = arith.addf %o, %m : f32
    linalg.yield %s : f32
  } -> tensor<8x6xf32>
  %r = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%p, %q : tensor<8x6xf32>, tensor<8x6xf32>) outs(%e : tensor<8x6xf32>) {
  ^bb0(%u: f32, %v: f32, %o: f32):
    %s = arith.subf %u, %v : f32
    %t = arith.mulf %u, %u : f32
    %z = arith.addf %s, %t : f32
    linalg.yield %z : f32
  } -> tensor<8x6xf32>
  return %r : tensor<8x6xf32>
}

// CHECK-LABEL: func.func @md_stage_s_f(
// CHECK-SAME: %[[X:.*]]: memref<8x4xf32>, %[[R:.*]]: memref<8x6xf32>)
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[W:.*]] = memref.get_global @md_stage___constant_4x6xf32
// CHECK-DAG: %[[WT:.*]] = memref.get_global @md_stage___constant_6x4xf32
// CHECK-NOT: gpu.launch
// CHECK: memref.extract_aligned_pointer_as_index %[[W]]
// CHECK: call @mdrtDeviceConstant({{.*}}, %c96_i64)
// CHECK: call @mdrtGemmF32(%c8_i64, %c6_i64, %c4_i64, %{{.*}}, %c4_i64, %c0_i64, %c0_i32, %{{.*}}, %c6_i64, %c0_i64, %c0_i32, %{{.*}}, %c6_i64, %c0_i64, %[[ZERO]], %c1_i64)
// CHECK: memref.extract_aligned_pointer_as_index %[[WT]]
// CHECK: call @mdrtDeviceConstant({{.*}}, %c96_i64)
// CHECK: call @mdrtGemmF32(%c8_i64, %c6_i64, %c4_i64, %{{.*}}, %c4_i64, %c0_i64, %c0_i32, %{{.*}}, %c4_i64, %c0_i64, %c1_i32, %{{.*}}, %c6_i64, %c0_i64, %[[ZERO]], %c1_i64)
// CHECK: gpu.launch
// CHECK: memref.store {{.*}}, %[[R]]
// CHECK-NOT: gpu.launch
// CHECK: return
// CHECK: func.func private @mdrtDeviceConstant(i64, i64) -> i64
// CHECK: func.func private @mdrtGemmF32(i64, i64, i64, i64, i64, i64, i32, i64, i64, i64, i32, i64, i64, i64, f32, i64)

// LOOPS-LABEL: func.func @md_stage_s_f(
// LOOPS-NOT: mdrtGemmF32
// LOOPS: gpu.launch
// LOOPS: scf.for
// LOOPS-NOT: mdrtGemmF32

// WRONG: contractions is 'other'; expected 'gemm' or 'loops'
