// RUN: mdir-opt %s --md-exec-choose-scatter=tile-min-points=100000 | FileCheck %s
// RUN: mdir-opt %s --md-exec-choose-scatter="tile=on tile-edge=16" \
// RUN: | FileCheck %s --check-prefix=TILES
// RUN: mdir-opt %s --md-exec-choose-scatter=tile=on \
// RUN: | FileCheck %s --check-prefix=T8
// RUN: mdir-opt %s --md-exec-choose-scatter="tile=off hash=off" \
// RUN: | FileCheck %s --check-prefix=GLOBAL
// RUN: mdir-opt %s --md-exec-choose-scatter="tile=on hash=on deterministic=true" \
// RUN: | FileCheck %s --check-prefix=GLOBAL
// RUN: mdir-opt %s --md-exec-choose-scatter="hash=on hash-probes=4" \
// RUN: | FileCheck %s --check-prefix=ON
// RUN: mdir-opt %s --md-exec-choose-scatter=tile-min-points=100000 | mdir-opt \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-exec-choose-scatter="tile-min-points=100000 explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=WHY
// RUN: mdir-opt %s --md-exec-choose-scatter="tile=on tile-edge=16 deterministic=true explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=DET
// RUN: mdir-opt %s --md-exec-choose-scatter="tile=on tile-edge=16 explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=E16
// RUN: not mdir-opt %s --md-exec-choose-scatter=tile=bricks 2>&1 \
// RUN: | FileCheck %s --check-prefix=SWITCH
// RUN: not mdir-opt %s --md-exec-choose-scatter=tile-edge=6 2>&1 \
// RUN: | FileCheck %s --check-prefix=EDGE

// The strategy of each scatter-accumulate on a device (D285,
// docs/device-primitives.md).
//
// The reciprocal sums of PME (D282): `tile=auto` tiles, with bins of 8
// points by default, every sum in f32 on a grid of at least
// `tile-min-points` points; `tile=on` every sum that can be; `tile=off` and
// the deterministic mode none. A sum cannot be tiled where an edge of the
// grid is shorter than a tile, E + P - 1 points, or where a tile of the
// type of the forces, with the particles of a round, would not fit 48 KiB
// of the memory of a block. The default of `tile-min-points` is checked in
// pme-tiles-gpu.test.
//
// The loops over tuples (D283): `hash(S, P, L)` on each loop that may share
// particles, has no global sums, and adds to destinations of f32 or f64
// whose table of S = 128 x arity slots, rounded up to a power of two,
// fits 48 KiB; L = 100,000 by default (`hash-least`), 0 with `hash=on`.

!vec = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>
!vec64 = !md.field<@atoms, 3 x f64>
!real64 = !md.field<@atoms, f64>
!grid = !md.table<2, f32>
!grid64 = !md.table<2, f64>
!bonds = !mdrt.incidence<@atoms, 2, @bonds>
!dihedrals = !mdrt.incidence<@atoms, 4, @dihedrals>
!waters = !mdrt.incidence<@atoms, 3, @waters>
md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(ordered)
md.tuple_set @dihedrals on(@atoms) arity(4) orientation(ordered)
md.tuple_set @waters on(@atoms) arity(3) orientation(ordered) disjoint

// CHECK-LABEL: func.func @large(
// CHECK: md_exec.reciprocal {{.*}} grid([64, 64, 64]) order(4) {{.*}} scatter(tile(8))
// TILES-LABEL: func.func @large(
// TILES: md_exec.reciprocal {{.*}} scatter(tile(16))
// GLOBAL-LABEL: func.func @large(
// GLOBAL-NOT: scatter(
// GLOBAL: return
// WHY: remark: scatter: tile(8)
// DET: remark: scatter: no tile: the deterministic mode adds floating-point values in a fixed order
func.func @large(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// Below `tile-min-points`: as before (the bricks of D108 here), unless every
// sum is tiled.
// CHECK-LABEL: func.func @small(
// CHECK-NOT: scatter(
// CHECK: return
// TILES-LABEL: func.func @small(
// TILES: md_exec.reciprocal {{.*}} scatter(tile(16))
// WHY: remark: scatter: no tile: the grid has 64000 points, fewer than tile-min-points
func.func @small(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([40, 40, 40]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// The same rule at order 6: tiled from `tile-min-points`.
// CHECK-LABEL: func.func @sixth(
// CHECK: md_exec.reciprocal {{.*}} order(6) {{.*}} scatter(tile(8))
// GLOBAL-LABEL: func.func @sixth(
// GLOBAL-NOT: scatter(
// GLOBAL: return
func.func @sixth(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([48, 48, 48]) order(6) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// An edge of 12 points is shorter than a tile of 8 + 8 - 1 = 15; a tile of
// 16 + 8 - 1 = 23 points an edge does not fit the block memory.
// TILES-LABEL: func.func @short(
// TILES-NOT: scatter(
// TILES: return
// WHY: remark: scatter: no tile: an edge of the grid of 12 points is shorter than a tile of 15, whose points would wrap around it more than once
// E16: remark: scatter: no tile: a tile with the particles of a round takes 63004 bytes of block memory, more than 49152
func.func @short(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 12]) order(8) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// A grid in f64 keeps the spreading of before with `auto`; with `tile=on`,
// a tile of 19^3 values of f64 (54,872 bytes) does not fit, one of 11^3
// does.
// CHECK-LABEL: func.func @double(
// CHECK-NOT: scatter(
// CHECK: return
// T8-LABEL: func.func @double(
// T8: md_exec.reciprocal {{.*}} scatter(tile(8))
// TILES-LABEL: func.func @double(
// TILES-NOT: scatter(
// TILES: return
// WHY: remark: scatter: no tile: the grid is in f64, which spreads faster without tiles (tile=auto)
// E16: remark: scatter: no tile: a tile with the particles of a round takes 77400 bytes of block memory, more than 49152
func.func @double(%x: !vec64, %cell: !md.cell, %q: !real64,
                    %moduli: !grid64) -> (f64, !vec64) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(4) beta(3.0) coulomb(138.9)
      : !vec64, !real64, !grid64 -> f64, vector<9xf64>, !vec64
  return %u, %f : f64, !vec64
}

// The prolate kernel (D266) is tiled as the B-splines are.
// CHECK-LABEL: func.func @prolate(
// CHECK: md_exec.reciprocal {{.*}} prolate {{.*}}scatter(tile(8))
func.func @prolate(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(6) beta(0.0) coulomb(138.9) prolate
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// Bonds: a table of 128 x 2 = 256 slots, from 100,000 tuples a launch; with
// `hash=on` at any count.
// CHECK-LABEL: func.func @bond_forces(
// CHECK: md_exec.tuple_for {{.*}} arity(2) scatter(hash(256, 8, 100000))
// ON-LABEL: func.func @bond_forces(
// ON: md_exec.tuple_for {{.*}} arity(2) scatter(hash(256, 4, 0))
// GLOBAL-LABEL: func.func @bond_forces(
// GLOBAL-NOT: scatter(
// GLOBAL: return
// WHY: remark: scatter: hash(256, 8, 100000)
// DET: remark: scatter: no hash: the deterministic mode adds floating-point values in a fixed order
func.func @bond_forces(%inc: !bonds, %x: !vec, %cell: !md.cell) -> !vec {
  %f0 = md_exec.zeros : !vec
  %f = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         outs(%f0 : !vec) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n : vector<3xf32>, vector<3xf32>
  } : !bonds, !vec -> !vec
  return %f : !vec
}

// Dihedrals: 128 x 4 = 512 slots; a force and an energy of each particle in
// f64 take 512 x (4 + 24 + 8) = 18,432 bytes.
// CHECK-LABEL: func.func @dihedral_forces(
// CHECK: md_exec.tuple_for {{.*}} arity(4) scatter(hash(512, 8, 100000))
func.func @dihedral_forces(%inc: !dihedrals, %x: !vec64, %cell: !md.cell)
    -> (!vec64, !real64) {
  %f0 = md_exec.zeros : !vec64
  %e0 = md_exec.zeros : !real64
  %f, %e = md_exec.tuple_for %inc, %x, %cell
         coordinates(displacement(0, 1), displacement(2, 1), displacement(3, 2))
         outs(%f0, %e0 : !vec64, !real64) arity(4) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>, %c: vector<3xf64>):
    %z = arith.constant 0.0 : f64
    md_exec.yield %a, %b, %c, %a, %z, %z, %z, %z
        : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>,
          f64, f64, f64, f64
  } : !dihedrals, !vec64 -> !vec64, !real64
  return %f, %e : !vec64, !real64
}

// Four forces in f64 at 512 slots take 4 x 12,288 + 2,048 = 51,200 bytes,
// more than 48 KiB.
// CHECK-LABEL: func.func @wide(
// CHECK-NOT: scatter(
// CHECK: return
// WHY: remark: scatter: no hash: a table of 512 slots takes 51200 bytes of block memory, more than 49152
func.func @wide(%inc: !dihedrals, %x: !vec64, %cell: !md.cell)
    -> (!vec64, !vec64, !vec64, !vec64) {
  %f0 = md_exec.zeros : !vec64
  %g0 = md_exec.zeros : !vec64
  %h0 = md_exec.zeros : !vec64
  %k0 = md_exec.zeros : !vec64
  %f, %g, %h, %k = md_exec.tuple_for %inc, %x, %cell
         coordinates(displacement(0, 1))
         outs(%f0, %g0, %h0, %k0 : !vec64, !vec64, !vec64, !vec64) arity(4) {
  ^bb0(%a: vector<3xf64>):
    md_exec.yield %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                  %a, %a
        : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>,
          vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>,
          vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>,
          vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !dihedrals, !vec64 -> !vec64, !vec64, !vec64, !vec64
  return %f, %g, %h, %k : !vec64, !vec64, !vec64, !vec64
}

// A loop with global sums keeps the atomics: its kernel takes several
// tuples a thread for the rows of its sums (D103).
// CHECK-LABEL: func.func @sums(
// CHECK-NOT: scatter(
// CHECK: return
// WHY: remark: scatter: no hash: the loop has global sums
func.func @sums(%inc: !bonds, %x: !vec, %cell: !md.cell) -> (!vec, f64) {
  %f0 = md_exec.zeros : !vec
  %u0 = arith.constant 0.0 : f64
  %f, %u = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         outs(%f0 : !vec) reduce(%u0 : f64) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n = arith.negf %d : vector<3xf32>
    %e = arith.constant 1.0 : f64
    md_exec.yield %d, %n, %e : vector<3xf32>, vector<3xf32>, f64
  } : !bonds, !vec -> !vec, f64
  return %f, %u : !vec, f64
}

// The tuples of a disjoint set share no particle: each is written by the
// thread of its member at place 0, without atomics (D83).
// CHECK-LABEL: func.func @water_forces(
// CHECK-NOT: scatter(
// CHECK: return
// WHY: remark: scatter: no hash: the tuples share no particle
func.func @water_forces(%inc: !waters, %x: !vec, %cell: !md.cell) -> !vec {
  %f0 = md_exec.zeros : !vec
  %f = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         outs(%f0 : !vec) arity(3) disjoint {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield %d, %d, %d : vector<3xf32>, vector<3xf32>, vector<3xf32>
  } : !waters, !vec -> !vec
  return %f : !vec
}

// The storage form: a loop that overwrites its destination writes the rows
// of its particles; one that adds to it takes the table.
// CHECK-LABEL: func.func @storage(
// CHECK: md_exec.tuple_for {{.*}} arity(2) overwrite [true] {
// CHECK: md_exec.tuple_for {{.*}} arity(2) scatter(hash(256, 8, 100000))
// WHY: remark: scatter: no hash: the loop overwrites a destination
// WHY: remark: scatter: hash(256, 8, 100000)
func.func @storage(%inc: memref<?x?xi32>, %x: memref<?x3xf32>,
                   %cell: !md.cell, %f: memref<?x3xf32>) {
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32>) arity(2) overwrite [true] {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield %d, %d : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32>, memref<?x3xf32>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield %d, %d : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32>, memref<?x3xf32>
  return
}

// SWITCH: expected 'auto', 'on', or 'off' for 'tile', 'hash', 'pair-hash', and 'scatter-hash', got 'bricks', 'auto', 'off', and 'auto'
// EDGE: expected an edge of the tiles of 4, 8, or 16 points, got 6
