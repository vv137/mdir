// The semantic transform is a pure value operation. Its declared transpose
// reverses direction while preserving the Hermitian pairing (D319).
// RUN: mdir-opt %s --test-md-transpose --verify-each -o - 2>&1 | FileCheck %s
// CHECK: remark: transpose in operand 0: 'md.linear_transform'
// CHECK: remark: transpose in operand 0: 'md.linear_transform'
// CHECK: md.linear_transform %{{.*}} kind("fft3d") direction("backward") adjoint("forward") pairing("real_hermitian_half") : tensor<3x2x3xcomplex<f64>> -> tensor<3x2x5xf64>
// CHECK: md.linear_transform %{{.*}} kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : tensor<3x2x5xf64> -> tensor<3x2x3xcomplex<f64>>

func.func @odd(%real: tensor<3x2x5xf64>,
               %half: tensor<3x2x3xcomplex<f64>>) {
  %forward = md.linear_transform %real kind("fft3d") direction("forward")
    adjoint("backward") pairing("real_hermitian_half") {test.transpose = 0 : i32} :
    tensor<3x2x5xf64> -> tensor<3x2x3xcomplex<f64>>
  %backward = md.linear_transform %half kind("fft3d") direction("backward")
    adjoint("forward") pairing("real_hermitian_half") {test.transpose = 0 : i32} :
    tensor<3x2x3xcomplex<f64>> -> tensor<3x2x5xf64>
  return
}

func.func @even(%real: tensor<3x2x6xf64>) -> tensor<3x2x4xcomplex<f64>> {
  %forward = md.linear_transform %real kind("fft3d") direction("forward")
    adjoint("backward") pairing("real_hermitian_half") :
    tensor<3x2x6xf64> -> tensor<3x2x4xcomplex<f64>>
  return %forward : tensor<3x2x4xcomplex<f64>>
}

// F32 values retain the same Hermitian adjoint declaration as f64.
func.func @mixed(%real: tensor<3x2x6xf32>,
                 %half: tensor<3x2x4xcomplex<f32>>) {
  // CHECK: md.linear_transform %{{.*}} kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : tensor<3x2x6xf32> -> tensor<3x2x4xcomplex<f32>>
  %forward = md.linear_transform %real kind("fft3d") direction("forward")
    adjoint("backward") pairing("real_hermitian_half") :
    tensor<3x2x6xf32> -> tensor<3x2x4xcomplex<f32>>
  // CHECK: md.linear_transform %{{.*}} kind("fft3d") direction("backward") adjoint("forward") pairing("real_hermitian_half") : tensor<3x2x4xcomplex<f32>> -> tensor<3x2x6xf32>
  %backward = md.linear_transform %half kind("fft3d") direction("backward")
    adjoint("forward") pairing("real_hermitian_half") :
    tensor<3x2x4xcomplex<f32>> -> tensor<3x2x6xf32>
  return
}
