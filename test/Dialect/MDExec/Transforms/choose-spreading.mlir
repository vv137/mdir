// RUN: mdir-opt %s --md-exec-choose-spreading=min-points=100000 | FileCheck %s
// RUN: mdir-opt %s --md-exec-choose-spreading="layout=tiles edge=16" \
// RUN: | FileCheck %s --check-prefix=TILES
// RUN: mdir-opt %s --md-exec-choose-spreading=layout=global \
// RUN: | FileCheck %s --check-prefix=GLOBAL
// RUN: mdir-opt %s --md-exec-choose-spreading="layout=tiles deterministic=true" \
// RUN: | FileCheck %s --check-prefix=GLOBAL
// RUN: mdir-opt %s --md-exec-choose-spreading=min-points=100000 | mdir-opt \
// RUN: | FileCheck %s
// RUN: not mdir-opt %s --md-exec-choose-spreading=layout=bricks 2>&1 \
// RUN: | FileCheck %s --check-prefix=LAYOUT
// RUN: not mdir-opt %s --md-exec-choose-spreading=edge=6 2>&1 \
// RUN: | FileCheck %s --check-prefix=EDGE

// How each reciprocal sum spreads its charges on a device (D[pme-tiles]):
// `auto` tiles a grid of at least `min-points` points, with bins of 8
// points by default; `tiles` every sum that can be; `global` and the
// deterministic mode none. A sum cannot be tiled where an edge of the
// grid is shorter than a tile, E + P - 1 points, or where a tile of the
// type of the forces would not fit 48 KiB of the memory of a block.

!vec = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>
!vec64 = !md.field<@atoms, 3 x f64>
!real64 = !md.field<@atoms, f64>
!grid = !md.table<2, f32>
!grid64 = !md.table<2, f64>
md.particle_set @atoms

// CHECK-LABEL: func.func @large(
// CHECK: md_exec.reciprocal {{.*}} grid([64, 64, 64]) order(4) {{.*}} tiles(8)
// TILES-LABEL: func.func @large(
// TILES: md_exec.reciprocal {{.*}} tiles(16)
// GLOBAL-LABEL: func.func @large(
// GLOBAL-NOT: tiles(
func.func @large(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// Below `min-points`: as before, unless every sum is tiled.
// CHECK-LABEL: func.func @small(
// CHECK-NOT: tiles(
// CHECK: return
// TILES-LABEL: func.func @small(
// TILES: md_exec.reciprocal {{.*}} tiles(16)
func.func @small(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([40, 40, 40]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// An edge of 20 points is shorter than a tile of 16 + 8 - 1 = 23.
// TILES-LABEL: func.func @short(
// TILES-NOT: tiles(
// TILES: return
func.func @short(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 20]) order(8) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// A tile of 19^3 values of f64 takes 54,872 bytes; 11^3 of them fit.
// CHECK-LABEL: func.func @double(
// CHECK: md_exec.reciprocal {{.*}} tiles(8)
// TILES-LABEL: func.func @double(
// TILES-NOT: tiles(
// TILES: return
func.func @double(%x: !vec64, %cell: !md.cell, %q: !real64,
                    %moduli: !grid64) -> (f64, !vec64) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(4) beta(3.0) coulomb(138.9)
      : !vec64, !real64, !grid64 -> f64, vector<9xf64>, !vec64
  return %u, %f : f64, !vec64
}

// The prolate kernel (D266) is tiled as the B-splines are.
// CHECK-LABEL: func.func @prolate(
// CHECK: md_exec.reciprocal {{.*}} prolate {{.*}}tiles(8)
func.func @prolate(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(6) beta(0.0) coulomb(138.9) prolate
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// LAYOUT: expected the layout 'auto', 'global', or 'tiles', got 'bricks'
// EDGE: expected an edge of the tiles of 4, 8, or 16 points, got 6
