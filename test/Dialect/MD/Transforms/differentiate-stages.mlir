// RUN: mdir-opt %s --md-differentiate | FileCheck %s
// RUN: mdir-opt %s --md-differentiate="separate-stages=true" \
// RUN: | FileCheck %s --check-prefix=SEPARATE

// The derivative through neural stages (D288, docs/ml4-ir.md,
// Section 4): the adjoint of the readout is a map of its product; for each
// edge stage, one gather over the same relation evaluates the product from
// both ends of each pair (the second end takes −d and the ends exchanged)
// and gives the adjoint of the features it reads and the forces,
// −[(r̄_ij + r̄_ji) d/r + d̄_ij − d̄_ji], together: two evaluations a
// directed pair. The separate form, the reference, gives each as an op of
// its own.

!vec = !md.field<@atoms, 3 x f64>
!h = !md.field<@atoms, 2 x f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.stage @edge forward(@edge.f) vjp(@edge.b)
    : (f64, vector<3xf64>, vector<2xf64>, vector<2xf64>) -> vector<2xf64>
md.stage @readout forward(@readout.f) vjp(@readout.b)
    : (vector<2xf64>) -> f64

func.func @edge.f(%r: f64, %d: vector<3xf64>, %hi: vector<2xf64>,
                  %hj: vector<2xf64>) -> vector<2xf64> {
  %b = vector.broadcast %r : f64 to vector<2xf64>
  %m = arith.mulf %b, %hj : vector<2xf64>
  return %m : vector<2xf64>
}
func.func @edge.b(%r: f64, %d: vector<3xf64>, %hi: vector<2xf64>,
                  %hj: vector<2xf64>, %y: vector<2xf64>)
    -> (f64, vector<3xf64>, vector<2xf64>, vector<2xf64>) {
  %p = arith.mulf %y, %hj : vector<2xf64>
  %rb = vector.reduction <add>, %p : vector<2xf64> into f64
  %db = arith.constant dense<0.0> : vector<3xf64>
  %hib = arith.constant dense<0.0> : vector<2xf64>
  %b = vector.broadcast %r : f64 to vector<2xf64>
  %hjb = arith.mulf %b, %y : vector<2xf64>
  return %rb, %db, %hib, %hjb
      : f64, vector<3xf64>, vector<2xf64>, vector<2xf64>
}
func.func @readout.f(%h: vector<2xf64>) -> f64 {
  %p = arith.mulf %h, %h : vector<2xf64>
  %s = vector.reduction <add>, %p : vector<2xf64> into f64
  return %s : f64
}
func.func @readout.b(%h: vector<2xf64>, %y: f64) -> vector<2xf64> {
  %two = arith.constant 2.0 : f64
  %t = arith.mulf %two, %y : f64
  %b = vector.broadcast %t : f64 to vector<2xf64>
  %g = arith.mulf %b, %h : vector<2xf64>
  return %g : vector<2xf64>
}

// Two layers of edges over the same features: the second reads the first's
// result, a field computed from the positions, at both ends.
md.potential @model(%x: !vec, %cell: !md.cell, %h: !h) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  %a = md.gather_relation %n, %x, %cell gather(%h : !h) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<2xf64>, %hj: vector<2xf64>):
    %m = md.apply_stage @edge(%r, %d, %hi, %hj)
        : (f64, vector<3xf64>, vector<2xf64>, vector<2xf64>) -> vector<2xf64>
    md.yield %m : vector<2xf64>
  } : !pairs, !vec -> !h
  %b = md.gather_relation %n, %x, %cell gather(%a : !h) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %ai: vector<2xf64>, %aj: vector<2xf64>):
    %m = md.apply_stage @edge(%r, %d, %ai, %aj)
        : (f64, vector<3xf64>, vector<2xf64>, vector<2xf64>) -> vector<2xf64>
    md.yield %m : vector<2xf64>
  } : !pairs, !vec -> !h
  %u = md.sum_particles gather(%b : !h) {
  ^bb0(%bi: vector<2xf64>):
    %e = md.apply_stage @readout(%bi) : (vector<2xf64>) -> f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

md.function @forces(%x: !vec, %cell: !md.cell, %h: !h) -> !vec {
  %f = md.evaluate @model(%x, %cell, %h) request [forces]
      : (!vec, !md.cell, !h) -> !vec
  md.return %f : !vec
}

// CHECK-LABEL: md.function @model.forces
// The adjoint of %b: the product of the readout, with the weight 1.
// CHECK:       %[[BBAR:.*]] = md.map_particles gather(%[[B:.*]] : !md.field<@atoms, 2 x f64>) {
// CHECK:         md.apply_stage @readout vjp
// CHECK:       } : !md.field<@atoms, 2 x f64>
// The second layer: the adjoint of %a and the forces in one gather.
// CHECK:       %[[SECOND:.*]]:2 = md.gather_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%[[A:.*]], %[[BBAR]] : !md.field<@atoms, 2 x f64>, !md.field<@atoms, 2 x f64>) exchange(none, derived) {
// CHECK-NEXT:  ^bb0(%[[R:.*]]: f64, %[[D:.*]]: vector<3xf64>, %[[AI:.*]]: vector<2xf64>, %[[AJ:.*]]: vector<2xf64>, %[[YI:.*]]: vector<2xf64>, %[[YJ:.*]]: vector<2xf64>):
// CHECK-NEXT:    %[[ND:.*]] = arith.negf %[[D]] : vector<3xf64>
// CHECK-NEXT:    %[[P:.*]]:4 = md.apply_stage @edge vjp(%[[R]], %[[D]], %[[AI]], %[[AJ]], %[[YI]])
// CHECK-NEXT:    %[[Q:.*]]:4 = md.apply_stage @edge vjp(%[[R]], %[[ND]], %[[AJ]], %[[AI]], %[[YJ]])
// CHECK-NEXT:    %[[ABAR:.*]] = arith.addf %[[P]]#2, %[[Q]]#3 : vector<2xf64>
// CHECK-NEXT:    %[[RB:.*]] = arith.addf %[[P]]#0, %[[Q]]#0 : f64
// CHECK-NEXT:    %[[S:.*]] = arith.divf %[[RB]], %[[R]] : f64
// CHECK-NEXT:    %[[SB:.*]] = vector.broadcast %[[S]] : f64 to vector<3xf64>
// CHECK-NEXT:    %[[DB:.*]] = arith.subf %[[P]]#1, %[[Q]]#1 : vector<3xf64>
// CHECK-NEXT:    %[[SD:.*]] = arith.mulf %[[SB]], %[[D]] : vector<3xf64>
// CHECK-NEXT:    %[[G:.*]] = arith.addf %[[SD]], %[[DB]] : vector<3xf64>
// CHECK-NEXT:    %[[K:.*]] = arith.negf %[[G]] : vector<3xf64>
// CHECK-NEXT:    md.yield %[[ABAR]], %[[K]] : vector<2xf64>, vector<3xf64>
// CHECK-NEXT:  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 2 x f64>, !md.field<@atoms, 3 x f64>
// The first layer: its features do not depend on the positions, so the
// forces alone.
// CHECK:       %[[FIRST:.*]] = md.gather_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%{{.*}}, %[[SECOND]]#0 : {{.*}}) exchange(none, derived) {
// CHECK-COUNT-2: md.apply_stage @edge vjp
// CHECK:       } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f64>
// CHECK:       md.map_particles gather(%[[FIRST]], %[[SECOND]]#1 :
// CHECK:       md.return

// SEPARATE-LABEL: md.function @model.forces
// SEPARATE:       %[[BBAR:.*]] = md.map_particles
// SEPARATE:         md.apply_stage @readout vjp
// The adjoint of %a: the product of the second edge stage from both ends.
// SEPARATE:       %[[ABAR:.*]] = md.gather_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%[[A:.*]], %[[BBAR]] : !md.field<@atoms, 2 x f64>, !md.field<@atoms, 2 x f64>) exchange(none, derived) {
// SEPARATE-COUNT-2: md.apply_stage @edge vjp
// SEPARATE:       } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 2 x f64>
// The forces of each layer, a gather of its own, antisymmetric.
// SEPARATE:       md.gather_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%{{.*}}, %[[ABAR]] : {{.*}}) exchange(antisymmetric, derived)
// SEPARATE-COUNT-2: md.apply_stage @edge vjp
// SEPARATE:       md.gather_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%[[A]], %[[BBAR]] : {{.*}}) exchange(antisymmetric, derived)
// SEPARATE-COUNT-2: md.apply_stage @edge vjp
// SEPARATE:       md.return
