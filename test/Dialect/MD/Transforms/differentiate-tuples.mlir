// RUN: mdir-opt %s --md-differentiate | FileCheck %s

!vec       = !md.field<@atoms, 3 x f64>
!real      = !md.field<@atoms, f64>
!pairs     = !md.relation<@atoms, 2, unordered>
!bonds     = !md.relation<@atoms, 2, unordered, @bonds>
!angles    = !md.relation<@atoms, 3, reversal, @angles>
!dihedrals = !md.relation<@atoms, 4, reversal, @dihedrals>
!of_bond   = !md.field<@bonds, f64>
!of_angle  = !md.field<@angles, f64>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
md.tuple_set @dihedrals on(@atoms) arity(4) orientation(reversal)

// The forces of a term over tuples are a gather over the tuples whose
// kernel yields one force for each member. It takes the displacements that
// the derivative of the coordinate needs, after the coordinates of the sum.
md.potential @bond(%x: !vec, %cell: !md.cell, %bonds: !bonds,
                   %k: !of_bond, %r0: !of_bond) -> f64 {
  %u = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1))
         tuple(%k, %r0 : !of_bond, !of_bond) {
  ^bb0(%r: f64, %k_t: f64, %r0_t: f64):
    %dr = arith.subf %r, %r0_t : f64
    %dr2 = arith.mulf %dr, %dr : f64
    %e = arith.mulf %k_t, %dr2 : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @bond.energy_forces_virial(
// CHECK-SAME:    -> (f64, !md.field<@atoms, 3 x f64>, vector<9xf64>)
// CHECK:         %[[U:[0-9]+]] = md.sum_tuples %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(distance(0, 1)) tuple(
// CHECK:         %[[F:[0-9]+]] = md.gather_tuples %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(distance(0, 1), displacement(0, 1)) tuple(%{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@bonds, f64>, !md.field<@bonds, f64>) {
// CHECK-NEXT:    ^bb0(%[[R:[a-z0-9]+]]: f64, %[[D:[a-z0-9]+]]: vector<3xf64>, %[[K:[a-z0-9]+]]: f64, %[[R0:[a-z0-9]+]]: f64):
// CHECK:           %[[DR:[0-9]+]] = arith.subf %[[R]], %[[R0]]
// CHECK:           %[[NORM:[0-9]+]] = math.sqrt
// CHECK:           %[[INVERSE:[0-9]+]] = arith.divf %{{[a-z0-9_]+}}, %[[NORM]]
// CHECK:           %[[B:[0-9]+]] = vector.broadcast %[[INVERSE]]
// CHECK:           %[[UNIT:[0-9]+]] = arith.mulf %[[B]], %[[D]]
// CHECK:           %[[OPPOSITE:[0-9]+]] = arith.negf %[[UNIT]]
// CHECK:           %[[FACTOR:[0-9]+]] = vector.broadcast
// CHECK:           %[[F0:[0-9]+]] = arith.mulf %[[FACTOR]], %[[UNIT]]
// CHECK:           %[[F1:[0-9]+]] = arith.mulf %[[FACTOR]], %[[OPPOSITE]]
// CHECK-NEXT:      md.yield %[[F0]], %[[F1]] : vector<3xf64>, vector<3xf64>
// CHECK-NEXT:    } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f64>
// CHECK:         %[[W:[0-9]+]] = md.sum_tuples %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(distance(0, 1), displacement(0, 1)) tuple(
// CHECK:           vector.from_elements
// CHECK:         } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64> -> vector<9xf64>
// CHECK:         md.return %[[U]], %[[F]], %[[W]]

md.function @caller_bond(%x: !vec, %cell: !md.cell, %bonds: !bonds,
                         %k: !of_bond, %r0: !of_bond)
    -> (f64, !vec, vector<9xf64>) {
  %u, %f, %w = md.evaluate @bond(%x, %cell, %bonds, %k, %r0)
      request [energy, forces, virial]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond)
        -> (f64, !vec, vector<9xf64>)
  md.return %u, %f, %w : f64, !vec, vector<9xf64>
}

// Two coordinates that need the same displacement receive it once. A
// displacement that the sum takes already is not taken again, and neither
// is a coordinate that the energy does not depend on.
md.potential @urey(%x: !vec, %cell: !md.cell, %angles: !angles,
                   %k: !of_angle) -> f64 {
  %u = md.sum_tuples %angles, %x, %cell
         coordinates(cosine(0, 1, 2), displacement(2, 1), distance(0, 1),
                     distance(0, 2)) tuple(%k : !of_angle) {
  ^bb0(%c: f64, %d: vector<3xf64>, %r: f64, %s: f64, %k_t: f64):
    %t = arith.mulf %c, %r : f64
    %e = arith.mulf %k_t, %t : f64
    md.yield %e : f64
  } : !angles, !vec -> f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @urey.forces(
// CHECK:         md.gather_tuples %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(cosine(0, 1, 2), displacement(2, 1), distance(0, 1), distance(0, 2), displacement(0, 1)) tuple(
// CHECK-NEXT:    ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64):
// CHECK:           md.yield %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : vector<3xf64>, vector<3xf64>, vector<3xf64>

md.function @caller_urey(%x: !vec, %cell: !md.cell, %angles: !angles,
                         %k: !of_angle) -> !vec {
  %f = md.evaluate @urey(%x, %cell, %angles, %k) request [forces]
      : (!vec, !md.cell, !angles, !of_angle) -> !vec
  md.return %f : !vec
}

// A dihedral needs three displacements. A member that no coordinate names
// receives no force.
md.potential @twist(%x: !vec, %cell: !md.cell, %dihedrals: !dihedrals)
    -> f64 {
  %u = md.sum_tuples %dihedrals, %x, %cell
         coordinates(dihedral(0, 1, 2, 3)) {
  ^bb0(%phi: f64):
    %e = math.cos %phi : f64
    md.yield %e : f64
  } : !dihedrals, !vec -> f64
  %v = md.sum_tuples %dihedrals, %x, %cell coordinates(distance(0, 3)) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !dihedrals, !vec -> f64
  %s = arith.addf %u, %v : f64
  md.return %s : f64
}

// CHECK-LABEL: md.function @twist.forces(
// CHECK:         %[[F1:[0-9]+]] = md.gather_tuples %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(dihedral(0, 1, 2, 3), displacement(0, 1), displacement(1, 2), displacement(3, 2)) {
// CHECK-NEXT:    ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: vector<3xf64>):
// CHECK:           md.yield %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} :
// CHECK:         %[[F2:[0-9]+]] = md.gather_tuples %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(distance(0, 3), displacement(0, 3)) {
// CHECK:           %[[ZERO:[a-z0-9_]+]] = arith.constant dense<0.000000e+00> : vector<3xf64>
// CHECK:           md.yield %{{[0-9]+}}, %[[ZERO]], %[[ZERO]], %{{[0-9]+}} :
// CHECK:         %[[F:[0-9]+]] = md.map_particles gather(%[[F1]], %[[F2]] :
// CHECK:         md.return %[[F]]

md.function @caller_twist(%x: !vec, %cell: !md.cell, %dihedrals: !dihedrals)
    -> !vec {
  %f = md.evaluate @twist(%x, %cell, %dihedrals) request [forces]
      : (!vec, !md.cell, !dihedrals) -> !vec
  md.return %f : !vec
}

// A sum over pairs and a sum over tuples in one potential. The weight of a
// sum is the derivative of the energy with respect to it.
md.potential @both(%x: !vec, %cell: !md.cell, %bonds: !bonds, %w: f64)
    -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %k = arith.mulf %r, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %v = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    %k = arith.mulf %w, %r : f64
    md.yield %k : f64
  } : !bonds, !vec -> f64
  %wv = arith.mulf %w, %v : f64
  %s = arith.addf %u, %wv : f64
  md.return %s : f64
}

// CHECK-LABEL: md.function @both.forces_derivative3(
// CHECK-SAME:    %[[WEIGHT:[a-z0-9]+]]: f64) -> (!md.field<@atoms, 3 x f64>, f64)
// CHECK:         %[[V:[0-9]+]] = md.sum_tuples
// CHECK:         %[[F1:[0-9]+]] = md.gather_relation
// CHECK:         %[[F2:[0-9]+]] = md.gather_tuples
// CHECK:           arith.mulf %[[WEIGHT]], %{{[a-z0-9_]+}}
// CHECK:           md.yield %{{[0-9]+}}, %{{[0-9]+}} : vector<3xf64>, vector<3xf64>
// CHECK:         %[[F:[0-9]+]] = md.map_particles gather(%[[F1]], %[[F2]] :
//
// d(w v)/dw = v + w dv/dw, and dv/dw is the sum of r over the tuples.
// CHECK:         %[[DV:[0-9]+]] = md.sum_tuples %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(distance(0, 1)) {
// CHECK-NEXT:    ^bb0(%[[R:[a-z0-9]+]]: f64):
// CHECK-NEXT:      md.yield %[[R]] : f64
// CHECK:         %[[WDV:[0-9]+]] = arith.mulf %[[WEIGHT]], %[[DV]]
// CHECK:         %[[G:[0-9]+]] = arith.addf %{{[0-9]+}}, %{{[0-9]+}}
// CHECK:         md.return %[[F]], %[[G]]

md.function @caller_both(%x: !vec, %cell: !md.cell, %bonds: !bonds, %w: f64)
    -> (!vec, f64) {
  %f, %g = md.evaluate @both(%x, %cell, %bonds, %w)
      request [forces, derivative(3)]
      : (!vec, !md.cell, !bonds, f64) -> (!vec, f64)
  md.return %f, %g : !vec, f64
}
