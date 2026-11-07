// RUN: mdir-opt %s --convert-md-exec-to-gpu -split-input-file -verify-diagnostics

func.func @not_a_batch(%theta: memref<?xf64, 1>, %i: index) -> f64 {
  // expected-error@+1 {{a device scalar input requires a frame batch entry}}
  %value = memref.load %theta[%i] {mdir.frame_parameter} : memref<?xf64, 1>
  return %value : f64
}

// -----

func.func @host_buffer(%theta: memref<?xf64>, %i: index) -> f64
    attributes {mdir.frame_batch} {
  // expected-error@+1 {{a frame parameter requires an f64 buffer in device memory space 1}}
  %value = memref.load %theta[%i] {mdir.frame_parameter} : memref<?xf64>
  return %value : f64
}

// -----

func.func @host_consumer(%theta: memref<?xf64, 1>, %i: index) -> f64
    attributes {mdir.frame_batch} {
  // expected-error@+1 {{a frame parameter cannot be consumed on the host}}
  %value = memref.load %theta[%i] {mdir.frame_parameter} : memref<?xf64, 1>
  return %value : f64
}
