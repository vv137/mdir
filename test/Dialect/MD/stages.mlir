// Neural stages (D288, docs/ml4-ir.md): a declaration that binds
// a forward body and its vector-Jacobian product, their application in
// kernels, and fields of features; printed and parsed again unchanged.
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s

// CHECK-LABEL: md.stage @edge forward(@edge.f) vjp(@edge.b)
// CHECK-SAME:    : (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>, i32) -> vector<8xf64>
// CHECK:       md.stage @readout forward(@readout.f) : (vector<8xf64>) -> f64
// CHECK:       md.potential @model
// CHECK:         md.gather_relation
// CHECK:           md.apply_stage @edge(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>, i32) -> vector<8xf64>
// CHECK:         } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 8 x f64>
// CHECK:         md.map_particles
// CHECK:           md.apply_stage @edge vjp(%{{.*}}) : (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>, i32, vector<8xf64>) -> (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>)

!vec = !md.field<@atoms, 3 x f64>
!h = !md.field<@atoms, 8 x f64>
!z = !md.field<@atoms, i32>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.stage @edge forward(@edge.f) vjp(@edge.b)
    : (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>, i32) -> vector<8xf64>
md.stage @readout forward(@readout.f) : (vector<8xf64>) -> f64

func.func @edge.f(%r: f64, %d: vector<3xf64>, %hi: vector<8xf64>,
                  %hj: vector<8xf64>, %z: i32) -> vector<8xf64> {
  %b = vector.broadcast %r : f64 to vector<8xf64>
  %m = arith.mulf %b, %hj : vector<8xf64>
  return %m : vector<8xf64>
}

func.func @edge.b(%r: f64, %d: vector<3xf64>, %hi: vector<8xf64>,
                  %hj: vector<8xf64>, %z: i32, %y: vector<8xf64>)
    -> (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>) {
  %p = arith.mulf %y, %hj : vector<8xf64>
  %rb = vector.reduction <add>, %p : vector<8xf64> into f64
  %db = arith.constant dense<0.0> : vector<3xf64>
  %hib = arith.constant dense<0.0> : vector<8xf64>
  %b = vector.broadcast %r : f64 to vector<8xf64>
  %hjb = arith.mulf %b, %y : vector<8xf64>
  return %rb, %db, %hib, %hjb : f64, vector<3xf64>, vector<8xf64>, vector<8xf64>
}

func.func @readout.f(%h: vector<8xf64>) -> f64 {
  %s = vector.reduction <add>, %h : vector<8xf64> into f64
  return %s : f64
}

md.potential @model(%x: !vec, %cell: !md.cell, %h: !h, %z: !z) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  %a = md.gather_relation %n, %x, %cell gather(%h, %z : !h, !z)
      exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<8xf64>, %hj: vector<8xf64>,
       %zi: i32, %zj: i32):
    %m = md.apply_stage @edge(%r, %d, %hi, %hj, %zi)
        : (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>, i32)
          -> vector<8xf64>
    md.yield %m : vector<8xf64>
  } : !pairs, !vec -> !h
  // The product itself, as md-differentiate emits it.
  %g = md.map_particles gather(%a, %z : !h, !z) {
  ^bb0(%ai: vector<8xf64>, %zi: i32):
    %r = arith.constant 0.3 : f64
    %d = arith.constant dense<[0.3, 0.0, 0.0]> : vector<3xf64>
    %rb, %db, %hib, %hjb = md.apply_stage @edge vjp(%r, %d, %ai, %ai, %zi, %ai)
        : (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>, i32,
           vector<8xf64>)
          -> (f64, vector<3xf64>, vector<8xf64>, vector<8xf64>)
    md.yield %hjb : vector<8xf64>
  } : !h
  %u = md.sum_particles gather(%g : !h) {
  ^bb0(%gi: vector<8xf64>):
    %e = md.apply_stage @readout(%gi) : (vector<8xf64>) -> f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}
