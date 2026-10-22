// RUN: mdir-opt %s --split-input-file --md-exec-assign-storage="memory=device" | FileCheck %s
// RUN: mdir-opt %s --split-input-file --md-exec-assign-storage | FileCheck %s --check-prefix=HOST

// The storage of neighbor structures on a device (D330): a
// structure that begins where nothing uses an earlier one of the same
// shape any more takes its storage, and begins with a reset. Here the
// structure of the start, which a build makes, and that of the steps,
// which a loop carries from empty (D218): one storage, reset before the
// steps. On the host each has its own.
//
// CHECK-LABEL: func.func @start_and_steps(
// CHECK:         %[[NL:[0-9]+]] = md_exec.empty_neighbors size(%{{[a-z0-9_]+}}) positions(memref<?x3xf64, 1>) kind(matrix) width(48)
// CHECK-NOT:     md_exec.empty_neighbors
// CHECK:         md_exec.refresh_neighbors %[[NL]], {{.*}} policy(always)
// CHECK:         md_exec.rebuild_count
// CHECK:         md_exec.reset_neighbors %[[NL]] : !mdrt.neighbors<@atoms>
// CHECK:         scf.for
// CHECK:           md_exec.refresh_neighbors %[[NL]], {{.*}} policy(check)
// CHECK-NOT:     md_exec.empty_neighbors
// CHECK:         return
//
// HOST-LABEL: func.func @start_and_steps(
// HOST:         md_exec.empty_neighbors
// HOST:         md_exec.empty_neighbors
// HOST-NOT:     md_exec.reset_neighbors
// HOST:         return

!vec = !md.field<@atoms, 3 x f64>

md.particle_set @atoms

func.func @start_and_steps(%x: !vec, %cell: !md.cell, %n: index)
    -> (i64, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cells = md_exec.build_cells %x, %cell width(1.75)
      : !vec -> !mdrt.cells<@atoms>
  %start = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.5) skin(0.25) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %first = md_exec.rebuild_count %start : !mdrt.neighbors<@atoms>
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  %nle = scf.for %step = %c0 to %n step %c1
      iter_args(%na = %nl0) -> (!mdrt.neighbors<@atoms>) {
    %nb = md_exec.refresh_neighbors %na, %x, %cell
        cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
        : !mdrt.neighbors<@atoms>, !vec
    scf.yield %nb : !mdrt.neighbors<@atoms>
  }
  %last = md_exec.rebuild_count %nle : !mdrt.neighbors<@atoms>
  return %first, %last : i64, i64
}

// -----

// The structure of the start is used after the steps begin: two storages.
//
// CHECK-LABEL: func.func @overlap(
// CHECK:         md_exec.empty_neighbors
// CHECK:         md_exec.empty_neighbors
// CHECK-NOT:     md_exec.reset_neighbors
// CHECK:         return

!vec = !md.field<@atoms, 3 x f64>

md.particle_set @atoms

func.func @overlap(%x: !vec, %cell: !md.cell, %n: index) -> (i64, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cells = md_exec.build_cells %x, %cell width(1.75)
      : !vec -> !mdrt.cells<@atoms>
  %start = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.5) skin(0.25) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  %nle = scf.for %step = %c0 to %n step %c1
      iter_args(%na = %nl0) -> (!mdrt.neighbors<@atoms>) {
    %nb = md_exec.refresh_neighbors %na, %x, %cell
        cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
        : !mdrt.neighbors<@atoms>, !vec
    scf.yield %nb : !mdrt.neighbors<@atoms>
  }
  %first = md_exec.rebuild_count %start : !mdrt.neighbors<@atoms>
  %last = md_exec.rebuild_count %nle : !mdrt.neighbors<@atoms>
  return %first, %last : i64, i64
}

// -----

// Structures of another width, or of another kind, have storages of their
// own.
//
// CHECK-LABEL: func.func @shapes(
// CHECK:         md_exec.empty_neighbors {{.*}} kind(matrix) width(48)
// CHECK:         md_exec.empty_neighbors {{.*}} kind(matrix) width(64)
// CHECK:         md_exec.empty_neighbors {{.*}} kind(groups) width(64)
// CHECK-NOT:     md_exec.reset_neighbors
// CHECK:         return

!vec = !md.field<@atoms, 3 x f64>

md.particle_set @atoms

func.func @shapes(%x: !vec, %cell: !md.cell) -> (i64, i64, i64) {
  %cells = md_exec.build_cells %x, %cell width(1.75)
      : !vec -> !mdrt.cells<@atoms>
  %a = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.5) skin(0.25) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %na = md_exec.rebuild_count %a : !mdrt.neighbors<@atoms>
  %b = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.5) skin(0.25) kind(matrix) width(64)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %nb = md_exec.rebuild_count %b : !mdrt.neighbors<@atoms>
  %c = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.5) skin(0.25) kind(groups) width(64)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %nc = md_exec.rebuild_count %c : !mdrt.neighbors<@atoms>
  return %na, %nb, %nc : i64, i64, i64
}
