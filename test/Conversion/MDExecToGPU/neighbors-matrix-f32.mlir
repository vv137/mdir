// The instance for positions in f32 of the template that builds a neighbor
// matrix on a device (NeighborsMatrixGPU.mlir), made in IR from the template
// written for f64 (D[template-ir], #739). The functions that the template
// defines take the names of the instance; the declarations of the device
// primitives keep theirs, so that expandPrimitiveCalls resolves every call
// of them and removes the declarations: no `mdir.primitive` is left.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu \
// RUN: | FileCheck %s --implicit-check-not=mdir.primitive

md.particle_set @atoms

// CHECK-LABEL: func.func @build(
// CHECK:         call @mdrt_gpu_build_neighbors_matrix_f32(
func.func @build(%x: memref<?x3xf32, 1>, %cell: !md.cell, %n: index) {
  %nl0 = md_exec.empty_neighbors size(%n) positions(memref<?x3xf32, 1>)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(always)
      : !mdrt.neighbors<@atoms>, memref<?x3xf32, 1>
  return
}

// Every f64 of the template is f32 in the instance: the arguments, and the
// conversions from f64 to f32, now casts of the bits from f32 to f32.
// CHECK-DAG: func.func private @mdrt_gpu_cell_count_f32(%{{[a-z0-9]+}}: f32, %{{[a-z0-9]+}}: f32) -> index
// CHECK-DAG: func.func private @mdrt_gpu_cell_width_f32(%{{[a-z0-9]+}}: index, %{{[a-z0-9]+}}: vector<3xf32>,
// CHECK-DAG: func.func private @mdrt_gpu_build_neighbors_matrix_f32(%{{[a-z0-9]+}}: memref<?x3xf32, 1>, %{{[a-z0-9]+}}: vector<3xf32>,
// CHECK-DAG: func.func private @mdrt_gpu_build_neighbors_matrix_triclinic_f32(
// CHECK-DAG: func.func private @mdrt_gpu_matrix_sort_f32(
// CHECK-DAG: arith.bitcast %{{[a-z0-9_]+}} : f32 to f32
// The bound of a position far off the cell, 1.0e100, is +inf in f32, as
// the text of the template in f32 parsed to (#740).
// CHECK-DAG: arith.constant 0x7F800000 : f32
