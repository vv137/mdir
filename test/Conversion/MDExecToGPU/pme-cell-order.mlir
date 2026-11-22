// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu | FileCheck %s
// RUN: sed 's/order(4)/order(6)/' %s | not mdir-opt --md-exec-assign-algebra --convert-md-exec-to-gpu 2>&1 | FileCheck %s --check-prefix=ORDER
// RUN: sed 's/!record, !word, !word, !word, !word, !word, !word)/!record, !word, !word, !word, !word, !word)/; s/%cursor, %chunks :/%cursor :/' %s | not mdir-opt --md-exec-assign-algebra --convert-md-exec-to-gpu 2>&1 | FileCheck %s --check-prefix=WORK
// ORDER: cell-order storage requires a cubic Gaussian force sum
// WORK: cell-order storage requires two four-component mesh buffers and six i32 work buffers

!pos = memref<?x3xf64, 1>
!record = memref<?x4xf32, 1>
!real = memref<?xf32, 1>
!word = memref<?xi32, 1>
!mod = memref<?x?xf64, 1>
md.particle_set @atoms

// CHECK-LABEL: func.func @step(
// CHECK: call @mdrt_gpu_pme_cell_order
// CHECK: call @mdrt_gpu_pme_pack
// CHECK: call @mdrt_gpu_pme_spread_ordered
// CHECK: call @mdrt_gpu_pme_gather_ordered
// CHECK-NOT: call @mdrt_gpu_pme_force_xyz
// CHECK: return
func.func @step(%x: !pos, %q: !real, %cell: !md.cell, %m: !mod,
                %f: memref<?x3xf32, 1>, %fixed: memref<?xi64, 1>, %grid: !real,
                %complex: !real, %rows: memref<?xf64, 1>, %raw: !record,
                %tables: !real, %particles: !record,
                %keys: !word, %perm: !word, %counts: !word, %offsets: !word,
                %cursor: !word, %chunks: !word) {
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : memref<?x3xf32, 1>)
      scratch(%fixed, %grid, %complex, %rows, %raw, %tables, %particles,
              %keys, %perm, %counts, %offsets, %cursor, %chunks :
              memref<?xi64, 1>, !real, !real, memref<?xf64, 1>, !record, !real,
              !record, !word, !word, !word, !word, !word, !word)
      grid([144, 84, 80]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !real, !mod -> f64, vector<9xf64>
  return
}
