// D[ml2-center-partition]: relation membership, not a domain-width bound.
// Four particles on a diagonal in a periodic 8-cube. Only particles 1 and
// 2 form a pair at cutoff 2; their displacement crosses a face, edge, or
// corner as the grid changes. The independent pair oracle is their distance
// sqrt(3), while every other minimum-image distance exceeds 2.
//
// RUN: sed 's/GRID/2, 1, 1/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=CHECK
// RUN: sed 's/GRID/1, 1, 1/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=ONE
// RUN: sed 's/GRID/2, 2, 2/; s/directed/unique/g' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=UNIQUE
// RUN: sed 's/GRID/2, 2, 1/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=CHECK
// RUN: sed 's/GRID/2, 2, 2/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=CHECK
// RUN: sed 's/GRID/8, 1, 1/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=CHECK

!vec = !md.field<@atoms, 3 x f64>
!mask = !md.field<@atoms, i32>
!own = !md_dist.field<@dd, owned, 3 x f64>
!ghost = !md_dist.field<@dd, ghost, 3 x f64>
!indicator = !md_dist.field<@dd, owned, i32>
!halo = !md_dist.halo<@dd, 2.0>
!relation = !md_dist.relation<@dd, directed>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([GRID])
func.func private @printI64(i64)
func.func private @printNewline()

func.func @main() {
  %n = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %a = arith.constant 1.0 : f64
  %b = arith.constant 3.5 : f64
  %c = arith.constant 4.5 : f64
  %d = arith.constant 7.0 : f64
  %box = arith.constant 8.0 : f64
  %x = memref.alloc(%n) : memref<?x3xf64>
  memref.store %a, %x[%c0, %c0] : memref<?x3xf64>
  memref.store %a, %x[%c0, %c1] : memref<?x3xf64>
  memref.store %a, %x[%c0, %c2] : memref<?x3xf64>
  memref.store %b, %x[%c1, %c0] : memref<?x3xf64>
  memref.store %b, %x[%c1, %c1] : memref<?x3xf64>
  memref.store %b, %x[%c1, %c2] : memref<?x3xf64>
  memref.store %c, %x[%c2, %c0] : memref<?x3xf64>
  memref.store %c, %x[%c2, %c1] : memref<?x3xf64>
  memref.store %c, %x[%c2, %c2] : memref<?x3xf64>
  memref.store %d, %x[%c3, %c0] : memref<?x3xf64>
  memref.store %d, %x[%c3, %c1] : memref<?x3xf64>
  memref.store %d, %x[%c3, %c2] : memref<?x3xf64>
  %field = mdrt.from_buffer %x : memref<?x3xf64> to !vec
  %cell = md.orthorhombic_cell %box, %box, %box
  %layout = md_dist.partition %field, %cell : !vec -> !md_dist.layout<@dd>
  %owned = md_dist.distribute %field, %layout image(%cell) : !vec -> !own
  %h = md_dist.halo %layout, %owned, %cell : !own -> !halo
  %ghost = md_dist.halo_copy %owned, %h image(%cell) : !own, !halo -> !ghost
  %relation = md_dist.neighborhood %owned, %ghost, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  %interior, %boundary = md_dist.partition_centers %relation, %layout, %h
      : !relation, !md_dist.layout<@dd>, !halo -> !indicator, !indicator
  %in = md_dist.collect %interior, %layout : !indicator -> !mask
  %bd = md_dist.collect %boundary, %layout : !indicator -> !mask
  %ib = mdrt.to_buffer %in : !mask to memref<?xi32>
  %bb = mdrt.to_buffer %bd : !mask to memref<?xi32>
  scf.for %i = %c0 to %n step %c1 {
    %v = memref.load %ib[%i] : memref<?xi32>
    %w = memref.load %bb[%i] : memref<?xi32>
    %v64 = arith.extsi %v : i32 to i64
    %w64 = arith.extsi %w : i32 to i64
    func.call @printI64(%v64) : (i64) -> ()
    func.call @printI64(%w64) : (i64) -> ()
    scf.yield
  }
  func.call @printNewline() : () -> ()
  return
}
// CHECK: 10010110
// ONE: 10101010
// UNIQUE: 10011010
