// Verifier failures distinguish an invalid declaration from a valid zero or
// unused transform (D319).
// RUN: mdir-opt %s --split-input-file -verify-diagnostics -o /dev/null

func.func @direction(%x: tensor<3x2x5xf64>) {
  // expected-error@+1 {{expected direction 'forward' or 'backward'}}
  %s = md.linear_transform %x kind("fft3d") direction("sideways") adjoint("backward") pairing("real_hermitian_half") : tensor<3x2x5xf64> -> tensor<3x2x3xcomplex<f64>>
  return
}

// -----

func.func @adjoint(%x: tensor<3x2x5xf64>) {
  // expected-error@+1 {{expected declared adjoint 'backward'}}
  %s = md.linear_transform %x kind("fft3d") direction("forward") adjoint("forward") pairing("real_hermitian_half") : tensor<3x2x5xf64> -> tensor<3x2x3xcomplex<f64>>
  return
}

// -----

func.func @pairing(%x: tensor<3x2x5xf64>) {
  // expected-error@+1 {{expected pairing 'real_hermitian_half'}}
  %s = md.linear_transform %x kind("fft3d") direction("forward") adjoint("backward") pairing("euclidean") : tensor<3x2x5xf64> -> tensor<3x2x3xcomplex<f64>>
  return
}

// -----

func.func @half_shape(%x: tensor<3x2x5xf64>) {
  // expected-error@+1 {{expected the last dimension of the packed half-spectrum to be K2/2+1}}
  %s = md.linear_transform %x kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : tensor<3x2x5xf64> -> tensor<3x2x4xcomplex<f64>>
  return
}

// -----

func.func @wrong_type(%x: tensor<3x2x5xf64>) {
  // expected-error@+1 {{expected a real f32 or f64 grid and a matching complex half-spectrum}}
  %s = md.linear_transform %x kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : tensor<3x2x5xf64> -> tensor<3x2x3xf64>
  return
}

// -----

func.func @dynamic(%x: tensor<?x2x5xf64>) {
  // expected-error@+1 {{expected statically shaped rank-three grids}}
  %s = md.linear_transform %x kind("fft3d") direction("forward") adjoint("backward") pairing("real_hermitian_half") : tensor<?x2x5xf64> -> tensor<?x2x3xcomplex<f64>>
  return
}
