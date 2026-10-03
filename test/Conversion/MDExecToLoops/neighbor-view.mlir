// RUN: mdir-opt %s --convert-md-exec-to-loops='simd-width=4' | FileCheck %s
// RUN: mdir-opt %s --convert-md-exec-to-loops='simd-width=8' | FileCheck %s --check-prefix=WIDE
// RUN: mdir-opt %s --convert-md-exec-to-loops | FileCheck %s --check-prefix=SCALAR
// RUN: not mdir-opt %s --convert-md-exec-to-loops='simd-width=3' 2>&1 | FileCheck %s --check-prefix=INVALID
md.particle_set @atoms
func.func @owned(%counts: memref<?xi32>, %entries: memref<?x?xi32>,
                 %x: memref<?x3xf64>, %f: memref<?x3xf64>,
                 %cell: !md.cell, %local: index) -> f64 {
  %nl = md_exec.neighbor_view %counts, %entries local_size(%local)
      : memref<?xi32>, memref<?x?xi32> -> !mdrt.neighbors<@atoms>
  %zero = arith.constant 0.0 : f64
  %e = md_exec.pair_for %nl, %x, %cell outs(%f : memref<?x3xf64>)
      reduce(%zero : f64) cutoff(2.5) weights [0.5] overwrite [true]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %one = arith.constant 1.0 : f64
    %u = arith.divf %one, %r2 : f64
    md_exec.yield %d, %u : vector<3xf64>, f64
  } : !mdrt.neighbors<@atoms>, memref<?x3xf64> -> f64
  return %e : f64
}
// CHECK: cf.assert {{.*}} "neighbor view
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: scf.if
// CHECK: arith.divf {{.*}} : vector<4xf64>
// CHECK: vector.reduction <add>
// CHECK-NOT: md_exec.
// WIDE: arith.divf {{.*}} : vector<8xf64>
// SCALAR: scf.parallel
// SCALAR: arith.divf {{.*}} : f64
// INVALID: simd-width must be 1, 4, or 8
