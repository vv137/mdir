// RUN: mdir-opt %s --md-exec-assign-streams="reciprocal=true tuples=true" -split-input-file | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-streams -split-input-file | FileCheck %s --check-prefix=OFF

// D[concurrent-terms]: loops over tuples on the third stream
// (md_exec.lane = 2). Ops of one side stream are ordered by it; an op of
// another stream inside a window must be independent of the side op.

// A loop over tuples with a destination of its own moves up past the
// refresh and the loop over pairs; the join stands before the kick that
// reads both destinations.
// CHECK-LABEL: func.func @beside(
// CHECK:         md_exec.tuple_for
// CHECK:         } {md_exec.lane = 2 : i64, md_exec.side}
// CHECK:         md_exec.refresh_neighbors
// CHECK-NEXT:    md_exec.pair_for
// CHECK:         md_exec.join
// CHECK-NEXT:    md_exec.particle_for
// OFF-LABEL: func.func @beside(
// OFF-NOT:     md_exec.side
// OFF-NOT:     md_exec.join
!pos = memref<?x3xf32, 1>
!frc = memref<?x3xf32, 1>
!nl = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
md.particle_set @atoms
func.func @beside(%n: index, %members: memref<?x2xi32>, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %fp = gpu.alloc (%n) : !frc
  %ft = gpu.alloc (%n) : !frc
  %v = gpu.alloc (%n) : !frc
  %inc = md_exec.build_incidence %members size(%n) : memref<?x2xi32> -> memref<?xi32, 1>
  %nb = md_exec.empty_neighbors size(%n) positions(!pos) kind(matrix) width(64) : !nl
  %r = md_exec.refresh_neighbors %nb, %x, %cell cutoff(1.5) skin(0.2) cell_width(0.5) policy(always) : !nl, !pos
  md_exec.pair_for %r, %x, %cell outs(%fp : !frc) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !nl, !pos
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1)) outs(%ft : !frc) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %m = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %m : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, !pos
  md_exec.particle_for ins(%fp, %ft : !frc, !frc) outs(%v : !frc) {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>):
    %s = arith.addf %a, %b : vector<3xf32>
    md_exec.yield %s : vector<3xf32>
  }
  return
}

// -----

// The loop over tuples adds to the destination of the pairs: it cannot pass
// the loop over pairs, stays where it stood, and nothing is marked.
// CHECK-LABEL: func.func @shared(
// CHECK:         md_exec.refresh_neighbors
// CHECK-NEXT:    md_exec.pair_for
// CHECK:         md_exec.tuple_for
// CHECK-NOT:     md_exec.side
// CHECK-NOT:     md_exec.join
// CHECK:         return
!pos = memref<?x3xf32, 1>
!frc = memref<?x3xf32, 1>
!nl = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
md.particle_set @atoms
func.func @shared(%n: index, %members: memref<?x2xi32>, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %fp = gpu.alloc (%n) : !frc
  %ft = gpu.alloc (%n) : !frc
  %v = gpu.alloc (%n) : !frc
  %inc = md_exec.build_incidence %members size(%n) : memref<?x2xi32> -> memref<?xi32, 1>
  %nb = md_exec.empty_neighbors size(%n) positions(!pos) kind(matrix) width(64) : !nl
  %r = md_exec.refresh_neighbors %nb, %x, %cell cutoff(1.5) skin(0.2) cell_width(0.5) policy(always) : !nl, !pos
  md_exec.pair_for %r, %x, %cell outs(%fp : !frc) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !nl, !pos
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1)) outs(%fp : !frc) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %m = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %m : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, !pos
  md_exec.particle_for ins(%fp, %ft : !frc, !frc) outs(%v : !frc) {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>):
    %s = arith.addf %a, %b : vector<3xf32>
    md_exec.yield %s : vector<3xf32>
  }
  return
}

// -----

// A reciprocal sum (second stream) after the loop over tuples (third
// stream) writes what the tuples write. The two streams would run them at
// once: the tuples' window ends before the sum, with a join, though the
// sum is marked as well.
// CHECK-LABEL: func.func @cross_lane(
// CHECK:         md_exec.tuple_for
// CHECK:         } {md_exec.lane = 2 : i64, md_exec.side}
// CHECK:         md_exec.pair_for
// CHECK:         md_exec.join
// CHECK-NEXT:    md_exec.reciprocal
// CHECK:         md_exec.join
// CHECK-NEXT:    md_exec.particle_for
!pos = memref<?x3xf32, 1>
!frc = memref<?x3xf32, 1>
!nl = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
md.particle_set @atoms
func.func @cross_lane(%n: index, %k: index, %members: memref<?x2xi32>, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %fp = gpu.alloc (%n) : !frc
  %ft = gpu.alloc (%n) : !frc
  %v = gpu.alloc (%n) : !frc
  %inc = md_exec.build_incidence %members size(%n) : memref<?x2xi32> -> memref<?xi32, 1>
  %nb = md_exec.empty_neighbors size(%n) positions(!pos) kind(matrix) width(64) : !nl
  %r = md_exec.refresh_neighbors %nb, %x, %cell cutoff(1.5) skin(0.2) cell_width(0.5) policy(always) : !nl, !pos
  md_exec.pair_for %r, %x, %cell outs(%fp : !frc) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !nl, !pos
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1)) outs(%ft : !frc) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %m = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %m : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, !pos
  %q = gpu.alloc (%n) : memref<?xf32, 1>
  %m = gpu.alloc (%k, %k) : memref<?x?xf64, 1>
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%ft : !frc)
      scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(1.0)
      : !pos, memref<?xf32, 1>, memref<?x?xf64, 1> -> f64, vector<9xf64>
  md_exec.particle_for ins(%fp, %ft : !frc, !frc) outs(%v : !frc) {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>):
    %s = arith.addf %a, %b : vector<3xf32>
    md_exec.yield %s : vector<3xf32>
  }
  return
}

// -----

// Between two loops over tuples an op of the first stream reads what the
// first writes: the first runs beside the pairs and its window ends before
// that op; the second cannot pass it, and stays on the first stream.
// CHECK-LABEL: func.func @same_lane(
// CHECK:         md_exec.tuple_for
// CHECK:         } {md_exec.lane = 2 : i64, md_exec.side}
// CHECK:         md_exec.pair_for
// CHECK:         md_exec.join
// CHECK-NEXT:    md_exec.permute
// CHECK-NEXT:    md_exec.tuple_for
// CHECK-NOT:     md_exec.side
// CHECK:         md_exec.particle_for
!pos = memref<?x3xf32, 1>
!frc = memref<?x3xf32, 1>
!nl = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
md.particle_set @atoms
func.func @same_lane(%n: index, %members: memref<?x2xi32>, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %fp = gpu.alloc (%n) : !frc
  %ft = gpu.alloc (%n) : !frc
  %v = gpu.alloc (%n) : !frc
  %inc = md_exec.build_incidence %members size(%n) : memref<?x2xi32> -> memref<?xi32, 1>
  %nb = md_exec.empty_neighbors size(%n) positions(!pos) kind(matrix) width(64) : !nl
  %r = md_exec.refresh_neighbors %nb, %x, %cell cutoff(1.5) skin(0.2) cell_width(0.5) policy(always) : !nl, !pos
  md_exec.pair_for %r, %x, %cell outs(%fp : !frc) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !nl, !pos
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1)) outs(%ft : !frc) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %m = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %m : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, !pos
  %o = gpu.alloc (%n) : memref<?xi32, 1>
  %h = gpu.alloc (%n) : !frc
  md_exec.permute %ft, %o outs(%h : !frc) : !frc, memref<?xi32, 1>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1)) outs(%ft : !frc) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %m = arith.negf %d : vector<3xf32>
    md_exec.yield %m, %d : vector<3xf32>, vector<3xf32>
  } : memref<?xi32, 1>, !pos
  md_exec.particle_for ins(%fp, %ft : !frc, !frc) outs(%v : !frc) {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>):
    %s = arith.addf %a, %b : vector<3xf32>
    md_exec.yield %s : vector<3xf32>
  }
  return
}
