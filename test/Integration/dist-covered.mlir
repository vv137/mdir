// D[dist-covered]: independent analytic skin oracle. For reach 2.2 and
// cutoff 2.0, the half skin is 0.1. Two particles start at x=0.01 and
// x=3.00 in an 8-cube. A periodic move 0.01 -> 7.98 has displacement
// 0.03, and 3.00 -> 3.04 has displacement 0.04: covered = 1. Moves to
// 7.89 and 3.11 have displacements 0.12 and 0.11: covered = 0.
// A 16-domain grid leaves most domains empty but gives the same answer.
// RUN: sed 's/GRID/2, 1, 1/' %s \
// RUN: | mdir-opt --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: sed 's/GRID/16, 1, 1/' %s \
// RUN: | mdir-opt --convert-md-dist-to-md-exec="functions=main" \
// RUN:     --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.2>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([GRID])
func.func private @printI64(i64)
func.func private @printNewline()

func.func @main() {
  %n = arith.constant 2 : index
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %z = arith.constant 0.0 : f64
  %a = arith.constant 0.01 : f64
  %b = arith.constant 3.0 : f64
  %a1 = arith.constant 7.98 : f64
  %b1 = arith.constant 3.04 : f64
  %a2 = arith.constant 7.89 : f64
  %b2 = arith.constant 3.11 : f64
  %edge = arith.constant 8.0 : f64
  %base = memref.alloc(%n) : memref<?x3xf64>
  %near = memref.alloc(%n) : memref<?x3xf64>
  %far = memref.alloc(%n) : memref<?x3xf64>
  memref.store %a, %base[%i0, %i0] : memref<?x3xf64>
  memref.store %b, %base[%i1, %i0] : memref<?x3xf64>
  memref.store %a1, %near[%i0, %i0] : memref<?x3xf64>
  memref.store %b1, %near[%i1, %i0] : memref<?x3xf64>
  memref.store %a2, %far[%i0, %i0] : memref<?x3xf64>
  memref.store %b2, %far[%i1, %i0] : memref<?x3xf64>
  scf.for %i = %i0 to %n step %i1 {
    memref.store %z, %base[%i, %i1] : memref<?x3xf64>
    memref.store %z, %base[%i, %n] : memref<?x3xf64>
    memref.store %z, %near[%i, %i1] : memref<?x3xf64>
    memref.store %z, %near[%i, %n] : memref<?x3xf64>
    memref.store %z, %far[%i, %i1] : memref<?x3xf64>
    memref.store %z, %far[%i, %n] : memref<?x3xf64>
  }
  %x0 = mdrt.from_buffer %base : memref<?x3xf64> to !vec
  %x1 = mdrt.from_buffer %near : memref<?x3xf64> to !vec
  %x2 = mdrt.from_buffer %far : memref<?x3xf64> to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %l = md_dist.partition %x0, %cell : !vec -> !md_dist.layout<@dd>
  %o0 = md_dist.distribute %x0, %l image(%cell) : !vec -> !own
  %o1 = md_dist.distribute %x1, %l image(%cell) : !vec -> !own
  %o2 = md_dist.distribute %x2, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %o0, %cell : !own -> !halo
  %ok1 = md_dist.covered %h, %o0, %o1, %cell cutoff(2.0)
      : !halo, !own -> i1
  %ok2 = md_dist.covered %h, %o0, %o2, %cell cutoff(2.0)
      : !halo, !own -> i1
  %v1 = arith.extui %ok1 : i1 to i64
  %v2 = arith.extui %ok2 : i1 to i64
  func.call @printI64(%v1) : (i64) -> ()
  func.call @printI64(%v2) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}
// CHECK: 10
