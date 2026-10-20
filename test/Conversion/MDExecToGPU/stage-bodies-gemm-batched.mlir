// A batched product in a body of a neural stage (D302,
// docs/ml4-ir.md, Section 8.3): for each row r, y[r, c] = sum_f x[r, f]
// v[r, c, f], PaiNN's contraction of a vector's three components with a
// row of weights (`edge.b`), x and v slices of one operand. It becomes a
// call of cuBLAS's strided-batched f32 GEMM through the runtime, row-major
// (m, n, k, A, lda, strideA, transA, B, ldb, strideB, transB, C, ldc,
// strideC, beta, batch): for each of the 8 rows, C (3 x 1) = A (3 x 4, v)
// B (4 x 1, x), the rows of the operand 16 apart. Two forms: a contraction
// from a fill of zero, and the form of stablehlo-legalize-to-linalg, a
// product and a sum of one operand, which the fusion makes one generic
// that starts from zero at its first reduced iteration. Both call with beta
// 0 and leave no kernel; with `contractions=loops` they are loops.
//
// RUN: mdir-opt %s --md-exec-lower-stage-bodies | FileCheck %s
// RUN: mdir-opt %s --md-exec-lower-stage-bodies="contractions=loops" | FileCheck %s --check-prefix=LOOPS

md.particle_set @atoms
md.stage @t forward(@t.f) : (vector<16xf64>) -> (vector<3xf64>, vector<3xf64>)

#x = affine_map<(r, c, f) -> (r, f)>
#v = affine_map<(r, c, f) -> (r, c, f)>
#y = affine_map<(r, c, f) -> (r, c)>

func.func @t.f(%in: tensor<8x16xf32>) -> (tensor<8x3xf32>, tensor<8x3xf32>) {
  %zero = arith.constant 0.0 : f32
  %x = tensor.extract_slice %in[0, 0] [8, 4] [1, 1]
      : tensor<8x16xf32> to tensor<8x4xf32>
  %vf = tensor.extract_slice %in[0, 4] [8, 12] [1, 1]
      : tensor<8x16xf32> to tensor<8x12xf32>
  %v = tensor.expand_shape %vf [[0], [1, 2]] output_shape [8, 3, 4]
      : tensor<8x12xf32> into tensor<8x3x4xf32>
  %e = tensor.empty() : tensor<8x3xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<8x3xf32>) -> tensor<8x3xf32>
  %y = linalg.generic {indexing_maps = [#x, #v, #y],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x, %v : tensor<8x4xf32>, tensor<8x3x4xf32>) outs(%f : tensor<8x3xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %o, %m : f32
    linalg.yield %s : f32
  } -> tensor<8x3xf32>
  %pe = tensor.empty() : tensor<8x3x4xf32>
  %p = linalg.generic {indexing_maps = [#x, #v, #v],
                       iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x, %v : tensor<8x4xf32>, tensor<8x3x4xf32>) outs(%pe : tensor<8x3x4xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32
  } -> tensor<8x3x4xf32>
  %f2 = linalg.fill ins(%zero : f32) outs(%e : tensor<8x3xf32>) -> tensor<8x3xf32>
  %z = linalg.generic {indexing_maps = [#v, #y],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%p : tensor<8x3x4xf32>) outs(%f2 : tensor<8x3xf32>) {
  ^bb0(%a: f32, %o: f32):
    %s = arith.addf %a, %o : f32
    linalg.yield %s : f32
  } -> tensor<8x3xf32>
  return %y, %z : tensor<8x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func.func @md_stage_t_f(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NOT: gpu.launch
// CHECK: call @mdrtGemmF32(%c3_i64, %c1_i64, %c4_i64, %{{.*}}, %c4_i64, %c16_i64, %c0_i32, %{{.*}}, %c1_i64, %c16_i64, %c0_i32, %{{.*}}, %c1_i64, %c3_i64, %[[ZERO]], %c8_i64)
// CHECK-NOT: gpu.launch
// CHECK: call @mdrtGemmF32(%c3_i64, %c1_i64, %c4_i64, %{{.*}}, %c4_i64, %c16_i64, %c0_i32, %{{.*}}, %c1_i64, %c16_i64, %c0_i32, %{{.*}}, %c1_i64, %c3_i64, %[[ZERO]], %c8_i64)
// CHECK-NOT: gpu.launch
// CHECK: return
// CHECK: func.func private @mdrtGemmF32(i64, i64, i64, i64, i64, i64, i32, i64, i64, i64, i32, i64, i64, i64, f32, i64)

// LOOPS-LABEL: func.func @md_stage_t_f(
// LOOPS-NOT: mdrtGemmF32
// LOOPS: gpu.launch
// LOOPS: scf.for
// LOOPS-NOT: mdrtGemmF32
