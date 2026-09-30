// RUN: mdir-opt %s --md-exec-approximate | FileCheck %s

!vec = !md.field<@atoms, 3 x f32>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// The force of the direct sum of Ewald in f32: erfc(β r) becomes
// exp(-β² r²) P(t), sharing the exponential of the derivative, and the
// divisions may be approximate.
//
// CHECK-LABEL: func.func @ewald(
// CHECK:         md_exec.pair_for
// CHECK:         ^bb0(%[[R2:[a-z0-9]+]]: f32,
// CHECK:           arith.divf {{.*}} fastmath<afn> : f32
// CHECK:           %[[K:[0-9]+]] = arith.mulf %[[R2]], %{{.*}} : f32
// CHECK:           %[[G:[0-9]+]] = math.exp %[[K]] : f32
// CHECK:           %[[T:[0-9]+]] = arith.divf %{{.*}}, %{{.*}} fastmath<afn> : f32
// CHECK-COUNT-9:   math.fma %{{.*}}, %[[T]], %{{.*}} : f32
// CHECK:           arith.mulf %[[G]], %{{.*}} : f32
// CHECK-NOT:       math.erfc
// CHECK-NOT:       math.exp
// CHECK:           md_exec.yield
func.func @ewald(%x: !vec, %cell: !md.cell, %nl: !nl) -> !vec {
  %f0 = md_exec.zeros : !vec
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(8.0)
      exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %one = arith.constant 1.0 : f32
    %beta = arith.constant 0.35 : f32
    %minus_beta2 = arith.constant -0.1225 : f32
    %inverse = arith.divf %one, %r2 : f32
    %r = math.sqrt %r2 : f32
    %br = arith.mulf %r, %beta : f32
    %erfc = math.erfc %br : f32
    %k = arith.mulf %r2, %minus_beta2 : f32
    %g = math.exp %k : f32
    %a = arith.mulf %erfc, %inverse : f32
    %b = arith.addf %a, %g : f32
    %s = vector.broadcast %b : f32 to vector<3xf32>
    %force = arith.mulf %s, %d : vector<3xf32>
    md_exec.yield %force : vector<3xf32>
  } : !nl, !vec -> !vec
  return %f : !vec
}

// Without an exponential to share, the rewrite makes one. An argument
// that is not a square root times a positive constant keeps erfc, and so
// does f64.
//
// CHECK-LABEL: func.func @alone(
// CHECK:         md_exec.particle_for
// CHECK:           math.exp
// CHECK-NOT:       math.erfc {{.*}} : f32
// CHECK:           math.erfc %{{.*}} : f32
// CHECK:           math.erfc %{{.*}} : f64
// CHECK:           arith.divf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:      md_exec.yield
func.func @alone(%y: !md.field<@atoms, f32>, %z: !md.field<@atoms, f64>)
    -> !md.field<@atoms, f64> {
  %d = md_exec.empty : !md.field<@atoms, f64>
  %e = md_exec.particle_for ins(%y, %z : !md.field<@atoms, f32>,
                                         !md.field<@atoms, f64>)
      outs(%d : !md.field<@atoms, f64>) {
  ^bb0(%yi: f32, %zi: f64):
    %root = math.sqrt %yi : f32
    %c = arith.constant 2.0 : f32
    %minus = arith.constant -2.0 : f32
    %x = arith.mulf %root, %c : f32
    %a = math.erfc %x : f32
    %xn = arith.mulf %root, %minus : f32
    %b = math.erfc %xn : f32
    %root64 = math.sqrt %zi : f64
    %c64 = arith.constant 2.0 : f64
    %x64 = arith.mulf %root64, %c64 : f64
    %e64 = math.erfc %x64 : f64
    %sum = arith.addf %a, %b : f32
    %wide = arith.extf %sum : f32 to f64
    %total = arith.addf %wide, %e64 : f64
    %quotient = arith.divf %total, %zi : f64
    md_exec.yield %quotient : f64
  } -> !md.field<@atoms, f64>
  return %e : !md.field<@atoms, f64>
}
