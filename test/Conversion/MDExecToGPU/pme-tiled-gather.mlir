// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu | FileCheck %s

// RUN: sed 's/^      scatter(tile(8))/      prolate spline_assignment scatter(tile(8))/' %s | mdir-opt --md-exec-assign-algebra --convert-md-exec-to-gpu | FileCheck %s --check-prefix=RADIAL
// RUN: sed 's/^      scatter(tile(8))/      prolate scatter(tile(8))/' %s | mdir-opt --md-exec-assign-algebra --convert-md-exec-to-gpu | FileCheck %s --check-prefix=LEGACY
//
// RADIAL-LABEL: func.func @tiled_gather(
// RADIAL: call @mdrt_gpu_pme_weights_
// RADIAL: call @mdrt_gpu_pme_spread_tiles8_
// RADIAL: call @mdrt_gpu_pme_radial_spline_tables_
// RADIAL: call @mdrt_gpu_pme_prolate_scale_
// RADIAL: call @mdrt_gpu_pme_gather_weights_
//
// LEGACY-LABEL: func.func @tiled_gather(
// LEGACY: call @mdrt_gpu_pme_prolate_coefficients_
// LEGACY: call @mdrt_gpu_pme_prolate_spread_tiles8_
// LEGACY: call @mdrt_gpu_pme_prolate_gather_
//
// The tiled spreading has bins and no bricks. Its f32 order-4 gathering
// still takes the particle weights, as the bricks path does.

!pos = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!mod = memref<?x?xf64, 1>
md.particle_set @atoms

// CHECK-LABEL: func.func @tiled_gather(
// CHECK: call @mdrt_gpu_pme_spread_tiles8
// CHECK: call @mdrt_gpu_pme_gather_weights
func.func @tiled_gather(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %q = gpu.alloc (%n) : !chg
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !pos
  %fixed = gpu.alloc (%k) : memref<?xi64, 1>
  %real = gpu.alloc (%k) : memref<?xf32, 1>
  %complex = gpu.alloc (%k) : memref<?xf32, 1>
  %rows = gpu.alloc (%k) : memref<?xf64, 1>
  %weights = gpu.alloc (%k) : memref<?xf32, 1>
  %tables = gpu.alloc (%k) : memref<?xf32, 1>
  %bins = gpu.alloc (%k) : memref<?xi32, 1>
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !pos)
      scratch(%fixed, %real, %complex, %rows, %weights, %tables, %bins :
              memref<?xi64, 1>, memref<?xf32, 1>, memref<?xf32, 1>,
              memref<?xf64, 1>, memref<?xf32, 1>, memref<?xf32, 1>,
              memref<?xi32, 1>)
      grid([64, 64, 64]) order(4) beta(3.0) coulomb(138.935457644)
      scatter(tile(8))
      : !pos, !chg, !mod -> f64, vector<9xf64>
  return
}
