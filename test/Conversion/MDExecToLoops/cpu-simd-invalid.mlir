// RUN: not mdir-opt %s --convert-md-exec-to-loops='simd-width=4' 2>&1 | FileCheck %s
md.particle_set @atoms
func.func @unsupported(%counts: memref<?xi32>, %entries: memref<?x?xi32>,
                      %x: memref<?x3xf64>, %cell: !md.cell, %n: index) -> f64 {
  %nl = md_exec.neighbor_view %counts, %entries local_size(%n)
      : memref<?xi32>, memref<?x?xi32> -> !mdrt.neighbors<@atoms>
  %zero = arith.constant 0.0 : f64
  %e = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64)
      cutoff(2.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %r = math.sqrt %r2 : f64
    md_exec.yield %r : f64
  } : !mdrt.neighbors<@atoms>, memref<?x3xf64> -> f64
  return %e : f64
}
// CHECK: unsupported CPU SIMD kernel operation: math.sqrt
