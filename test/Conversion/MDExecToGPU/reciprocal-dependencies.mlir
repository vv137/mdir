// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf64, 1>
!mod = memref<?x?xf64, 1>
md.particle_set @atoms

// CHECK-LABEL: func.func @mapped_charges(
// CHECK: call @mdrt_gpu_build_neighbors_matrix
// CHECK: gpu.launch
// CHECK: call @mdrt_gpu_pme_spread_float
func.func @mapped_charges(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %q0 = gpu.alloc (%n) : !chg
  %q = gpu.alloc (%n) : !chg
  %other = gpu.alloc (%n) : !chg
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  %s4 = gpu.alloc (%k) : memref<?xf32, 1>
  %s5 = gpu.alloc (%k) : memref<?xf32, 1>
  %nl0 = md_exec.empty_neighbors size(%n) positions(!pos)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(always)
      : !mdrt.neighbors<@atoms>, !pos
  md_exec.particle_for ins(%q0 : !chg) outs(%q : !chg) {
  ^bb0(%qi: f64):
    %half = arith.constant 0.5 : f64
    %scaled = arith.mulf %qi, %half : f64
    md_exec.yield %scaled : f64
  }
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>, memref<?xf32, 1>, memref<?xf32, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !chg, !mod -> f64, vector<9xf64>
  return
}

// CHECK-LABEL: func.func @independent_field(
// CHECK: call @mdrt_gpu_pme_spread_float
// CHECK: call @mdrt_gpu_build_neighbors_matrix
// CHECK: gpu.launch
func.func @independent_field(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %q0 = gpu.alloc (%n) : !chg
  %q = gpu.alloc (%n) : !chg
  %other = gpu.alloc (%n) : !chg
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  %s4 = gpu.alloc (%k) : memref<?xf32, 1>
  %s5 = gpu.alloc (%k) : memref<?xf32, 1>
  %nl0 = md_exec.empty_neighbors size(%n) positions(!pos)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(always)
      : !mdrt.neighbors<@atoms>, !pos
  md_exec.particle_for ins(%q0 : !chg) outs(%other : !chg) {
  ^bb0(%qi: f64):
    %half = arith.constant 0.5 : f64
    %scaled = arith.mulf %qi, %half : f64
    md_exec.yield %scaled : f64
  }
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>, memref<?xf32, 1>, memref<?xf32, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !chg, !mod -> f64, vector<9xf64>
  return
}

// A producer through a view of the charge buffer is also a dependency.
// CHECK-LABEL: func.func @aliased_charges(
// CHECK: call @mdrt_gpu_build_neighbors_matrix
// CHECK: gpu.launch
// CHECK: call @mdrt_gpu_pme_spread_float
func.func @aliased_charges(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %q0 = gpu.alloc (%n) : !chg
  %q = gpu.alloc (%n) : !chg
  %other = gpu.alloc (%n) : !chg
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  %s4 = gpu.alloc (%k) : memref<?xf32, 1>
  %s5 = gpu.alloc (%k) : memref<?xf32, 1>
  %nl0 = md_exec.empty_neighbors size(%n) positions(!pos)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(always)
      : !mdrt.neighbors<@atoms>, !pos
  %view = memref.subview %q[0] [%n] [1] : !chg to memref<?xf64, strided<[1]>, 1>
  md_exec.particle_for ins(%q0 : !chg) outs(%view : memref<?xf64, strided<[1]>, 1>) {
  ^bb0(%qi: f64):
    %half = arith.constant 0.5 : f64
    %scaled = arith.mulf %qi, %half : f64
    md_exec.yield %scaled : f64
  }
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>, memref<?xf32, 1>, memref<?xf32, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !chg, !mod -> f64, vector<9xf64>
  return
}


func.func private @unknown()

// Effects that cannot be proved independent also keep the original order.
// CHECK-LABEL: func.func @unknown_effects(
// CHECK: call @mdrt_gpu_build_neighbors_matrix
// CHECK: call @unknown()
// CHECK: call @mdrt_gpu_pme_spread_float
func.func @unknown_effects(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %q0 = gpu.alloc (%n) : !chg
  %q = gpu.alloc (%n) : !chg
  %other = gpu.alloc (%n) : !chg
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  %s4 = gpu.alloc (%k) : memref<?xf32, 1>
  %s5 = gpu.alloc (%k) : memref<?xf32, 1>
  %nl0 = md_exec.empty_neighbors size(%n) positions(!pos)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(always)
      : !mdrt.neighbors<@atoms>, !pos
  func.call @unknown() : () -> ()
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3, %s4, %s5 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>, memref<?xf32, 1>, memref<?xf32, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !chg, !mod -> f64, vector<9xf64>
  return
}
