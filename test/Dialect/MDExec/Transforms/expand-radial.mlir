// RUN: mdir-opt %s --md-exec-expand-radial | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// erfc(β r) / r of the direct sum of Ewald, in nm, with a cutoff of 0.8.
func.func private @md_radial(%s: f64) -> f64 {
  %r = math.sqrt %s : f64
  %beta = arith.constant 4.3236384215993748 : f64
  %br = arith.mulf %beta, %r : f64
  %e = math.erfc %br : f64
  %q = arith.divf %e, %r : f64
  return %q : f64
}

// In f32 the function becomes a table: the interval from the bits of r²,
// clamped, its start, and a cubic in u = r² − start, from one load.
//
// CHECK:       memref.global "private" constant @md_radial_table : memref<{{[0-9]+}}x4xf32>
// CHECK-NOT:   func.func private @md_radial(
// CHECK-LABEL: func.func @single(
// CHECK:         md_exec.pair_for
// CHECK:           %[[BITS:[0-9]+]] = arith.bitcast %{{.*}} : f32 to i32
// CHECK:           arith.shrui %[[BITS]]
// CHECK:           arith.maxsi
// CHECK:           arith.minsi
// CHECK:           %[[START:[0-9]+]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK:           %[[U:[0-9]+]] = arith.subf %{{.*}}, %[[START]] : f32
// CHECK:           %[[T:[0-9]+]] = memref.get_global @md_radial_table
// CHECK:           vector.load %[[T]][%{{.*}}, %{{.*}}] : memref<{{[0-9]+}}x4xf32>, vector<4xf32>
// CHECK-COUNT-3:   math.fma %{{.*}}, %[[U]], %{{.*}} : f32
// CHECK-NOT:       md_exec.radial
func.func @single(%x: !md.field<@atoms, 3 x f32>, %cell: !md.cell, %nl: !nl)
    -> f64 {
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(0.8)
      weights [0.5] exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %g = md_exec.radial @md_radial(%r2) : f32 -> f32
    %w = arith.extf %g : f32 to f64
    md_exec.yield %w : f64
  } : !nl, !md.field<@atoms, 3 x f32> -> f64
  return %u : f64
}
