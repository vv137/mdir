// The parameters of the instance that the templates of particle mesh Ewald
// on a device call, `@mdir.template.*`, resolved in IR
// (D369, #739): for forces in f32 and the order 4,
// the tiles of edge 8 have sides of 8 + 4 - 1 = 11 points and 1331 in all,
// a value is added to a tile by an atomic of the block and to the grid by
// PTX's reduction of the device, and a load of the grid takes the
// alignment of 4 bytes. No call or declaration of a parameter is left,
// and no `mdrt.template_convert` (D[template-ir-pme-convert]).
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu \
// RUN: | FileCheck %s --implicit-check-not=mdir.template \
// RUN:     --implicit-check-not=mdrt.template_convert

!pos = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!mod = memref<?x?xf64, 1>
md.particle_set @atoms

func.func @parameters(%n: index, %k: index, %cell: !md.cell) {
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

// CHECK-LABEL: func.func @parameters(
// CHECK-DAG: func.func private @mdrt_gpu_pme_spread_tiles8_p32c32f32o4(
// CHECK-DAG: arith.constant 11 : i32
// CHECK-DAG: arith.constant 1331 : index
// CHECK-DAG: memref<1331xf32, #gpu.address_space<workgroup>>
// CHECK-DAG: llvm.atomicrmw fadd %{{[0-9]+}}, %{{[0-9]+}} syncscope("block") monotonic : !llvm.ptr<3>, f32
// CHECK-DAG: llvm.inline_asm has_side_effects "red.relaxed.gpu.global.add.f32 [$0], $1;", "l,f"
// CHECK-DAG: llvm.load %{{[0-9]+}} {alignment = 4 : i64} : !llvm.ptr<1> -> f32
