// REQUIRES: pme-callback
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu | FileCheck %s

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!scalar = memref<?xf32, 1>
!mod = memref<?x?xf64, 1>
md.particle_set @atoms

// CHECK-LABEL: func.func @ordinary(
// CHECK-NOT: call @mdrt_gpu_pme_scale
// CHECK: call @mdrtCudaFFTGaussianBackwardF32
func.func @ordinary(%x: !pos, %q: !scalar, %cell: !md.cell, %m: !mod,
     %f: !frc, %p: !scalar, %s0: memref<?xi64, 1>, %s1: !scalar, %s2: !scalar,
     %s3: memref<?xf64, 1>, %s4: !scalar, %s5: !scalar) {
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
    scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, !scalar, !scalar, memref<?xf64, 1>, !scalar, !scalar)
    grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
    : !pos, !scalar, !mod -> f64, vector<9xf64>
  return
}

// CHECK-LABEL: func.func @prolate_spline(
// CHECK: call @mdrt_gpu_pme_radial_spline_tables_
// CHECK: call @mdrt_gpu_pme_prolate_scale_
// CHECK: call @mdrtCudaFFTBackward3DF32
func.func @prolate_spline(%x: !pos, %q: !scalar, %cell: !md.cell, %m: !mod,
     %f: !frc, %p: !scalar, %s0: memref<?xi64, 1>, %s1: !scalar, %s2: !scalar,
     %s3: memref<?xf64, 1>, %s4: !scalar, %s5: !scalar) {
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
    scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, !scalar, !scalar, memref<?xf64, 1>, !scalar, !scalar)
    grid([32, 32, 30]) order(4) beta(3.0) coulomb(138.9) prolate spline_assignment
    : !pos, !scalar, !mod -> f64, vector<9xf64>
  return
}

// CHECK-LABEL: func.func @energy(
// CHECK: call @mdrt_gpu_pme_convolve
// CHECK: call @mdrtCudaFFTBackward3DF32
func.func @energy(%x: !pos, %q: !scalar, %cell: !md.cell, %m: !mod,
     %f: !frc, %p: !scalar, %s0: memref<?xi64, 1>, %s1: !scalar, %s2: !scalar,
     %s3: memref<?xf64, 1>, %s4: !scalar, %s5: !scalar) -> f64 {
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
    scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, !scalar, !scalar, memref<?xf64, 1>, !scalar, !scalar)
    grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
    : !pos, !scalar, !mod -> f64, vector<9xf64>
  return %e : f64
}

// CHECK-LABEL: func.func @virial(
// CHECK: call @mdrt_gpu_pme_convolve
// CHECK: call @mdrtCudaFFTBackward3DF32
func.func @virial(%x: !pos, %q: !scalar, %cell: !md.cell, %m: !mod,
     %f: !frc, %p: !scalar, %s0: memref<?xi64, 1>, %s1: !scalar, %s2: !scalar,
     %s3: memref<?xf64, 1>, %s4: !scalar, %s5: !scalar) -> vector<9xf64> {
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
    scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, !scalar, !scalar, memref<?xf64, 1>, !scalar, !scalar)
    grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
    : !pos, !scalar, !mod -> f64, vector<9xf64>
  return %w : vector<9xf64>
}

// CHECK-LABEL: func.func @both(
// CHECK: call @mdrt_gpu_pme_scale
// CHECK: call @mdrtCudaFFTBackward3DF32
// CHECK: call @mdrt_gpu_pme_scale
// CHECK: call @mdrtCudaFFTBackward3DF32
func.func @both(%x: !pos, %q: !scalar, %cell: !md.cell, %m: !mod,
     %f: !frc, %p: !scalar, %s0: memref<?xi64, 1>, %s1: !scalar, %s2: !scalar,
     %s3: memref<?xf64, 1>, %s4: !scalar, %s5: !scalar) {
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
    scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, !scalar, !scalar, memref<?xf64, 1>, !scalar, !scalar)
    grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
    : !pos, !scalar, !mod -> f64, vector<9xf64>
  %ed, %wd = md_exec.reciprocal %x, %q, %cell, %m outs(%p : !scalar)
    scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, !scalar, !scalar, memref<?xf64, 1>, !scalar, !scalar)
    grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9) potential
    : !pos, !scalar, !mod -> f64, vector<9xf64>
  return
}

// CHECK-LABEL: func.func @unsupported_grid(
// CHECK: call @mdrt_gpu_pme_scale
// CHECK: call @mdrtCudaFFTBackward3DF32
func.func @unsupported_grid(%x: !pos, %q: !scalar, %cell: !md.cell, %m: !mod,
     %f: !frc, %p: !scalar, %s0: memref<?xi64, 1>, %s1: !scalar, %s2: !scalar,
     %s3: memref<?xf64, 1>, %s4: !scalar, %s5: !scalar) {
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
    scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, !scalar, !scalar, memref<?xf64, 1>, !scalar, !scalar)
    grid([127, 8, 8]) order(4) beta(3.0) coulomb(138.9)
    : !pos, !scalar, !mod -> f64, vector<9xf64>
  return
}

// CHECK-LABEL: func.func @wide_grid(
// CHECK: call @mdrt_gpu_pme_scale
// CHECK: call @mdrtCudaFFTBackward3DF32
func.func @wide_grid(%x: !pos, %q: !scalar, %cell: !md.cell, %m: !mod,
     %f: !frc, %p: !scalar, %s0: memref<?xi64, 1>, %s1: !scalar, %s2: !scalar,
     %s3: memref<?xf64, 1>, %s4: !scalar, %s5: !scalar) {
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
    scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, !scalar, !scalar, memref<?xf64, 1>, !scalar, !scalar)
    grid([65536, 65536, 8]) order(4) beta(3.0) coulomb(138.9)
    : !pos, !scalar, !mod -> f64, vector<9xf64>
  return
}
