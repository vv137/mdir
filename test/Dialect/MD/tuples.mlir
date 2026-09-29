// RUN: mdir-opt %s | mdir-opt | FileCheck %s

!vec       = !md.field<@atoms, 3 x f64>
!real      = !md.field<@atoms, f64>
!bonds     = !md.relation<@atoms, 2, unordered, @bonds>
!angles    = !md.relation<@atoms, 3, reversal, @angles>
!dihedrals = !md.relation<@atoms, 4, reversal, @dihedrals>
!restraints = !md.relation<@atoms, 1, ordered, @restraints>
!of_bond   = !md.field<@bonds, f64>
!of_angle  = !md.field<@angles, f64>

// CHECK: md.particle_set @atoms
md.particle_set @atoms

// CHECK-NEXT: md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)
// CHECK-NEXT: md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
// CHECK-NEXT: md.tuple_set @dihedrals on(@atoms) arity(4) orientation(reversal)
// CHECK-NEXT: md.tuple_set @restraints on(@atoms) arity(1) orientation(ordered)
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
md.tuple_set @dihedrals on(@atoms) arity(4) orientation(reversal)
md.tuple_set @restraints on(@atoms) arity(1) orientation(ordered)

// CHECK-LABEL: md.potential @bonded(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell
// CHECK-SAME:    %[[BONDS:[a-z0-9]+]]: !md.relation<@atoms, 2, unordered, @bonds>
// CHECK-SAME:    %[[ANGLES:[a-z0-9]+]]: !md.relation<@atoms, 3, reversal, @angles>
// CHECK-SAME:    %[[DIHEDRALS:[a-z0-9]+]]: !md.relation<@atoms, 4, reversal, @dihedrals>
md.potential @bonded(%x: !vec, %cell: !md.cell,
                     %bonds: !bonds, %k: !of_bond, %r0: !of_bond,
                     %angles: !angles, %ka: !of_angle,
                     %dihedrals: !dihedrals, %q: !real) -> f64 {
  // CHECK: %[[UB:[a-z0-9]+]] = md.sum_tuples %[[BONDS]], %[[X]], %[[CELL]] coordinates(distance(0, 1))
  // CHECK-SAME: tuple(%{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@bonds, f64>, !md.field<@bonds, f64>) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64):
  // CHECK: } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64> -> f64
  %ub = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1))
          tuple(%k, %r0 : !of_bond, !of_bond) {
  ^bb0(%r: f64, %k_t: f64, %r0_t: f64):
    %half = arith.constant 0.5 : f64
    %dr = arith.subf %r, %r0_t : f64
    %dr2 = arith.mulf %dr, %dr : f64
    %kh = arith.mulf %half, %k_t : f64
    %e = arith.mulf %kh, %dr2 : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64

  // Two coordinates, and a field of the particles: one value for each
  // member.
  //
  // CHECK: %[[UA:[a-z0-9]+]] = md.sum_tuples %[[ANGLES]], %[[X]], %[[CELL]] coordinates(cosine(0, 1, 2), distance(0, 2))
  // CHECK-SAME: gather(%{{[a-z0-9]+}} : !md.field<@atoms, f64>)
  // CHECK-SAME: tuple(%{{[a-z0-9]+}} : !md.field<@angles, f64>) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64):
  %ua = md.sum_tuples %angles, %x, %cell
          coordinates(cosine(0, 1, 2), distance(0, 2))
          gather(%q : !real) tuple(%ka : !of_angle) {
  ^bb0(%c: f64, %r: f64, %q0: f64, %q1: f64, %q2: f64, %ka_t: f64):
    %qq = arith.mulf %q0, %q2 : f64
    %t = arith.mulf %c, %r : f64
    %s = arith.mulf %qq, %t : f64
    %e = arith.mulf %ka_t, %s : f64
    md.yield %e : f64
  } : !angles, !vec -> f64

  // CHECK: md.sum_tuples %[[DIHEDRALS]], %[[X]], %[[CELL]] coordinates(dihedral(0, 1, 2, 3), angle(1, 2, 3), displacement(3, 0)) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: vector<3xf64>):
  %ud = md.sum_tuples %dihedrals, %x, %cell
          coordinates(dihedral(0, 1, 2, 3), angle(1, 2, 3),
                      displacement(3, 0)) {
  ^bb0(%phi: f64, %theta: f64, %d: vector<3xf64>):
    %cos = math.cos %phi : f64
    %e = arith.mulf %cos, %theta : f64
    md.yield %e : f64
  } : !dihedrals, !vec -> f64

  %u0 = arith.addf %ub, %ua : f64
  %u = arith.addf %u0, %ud : f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @pull(
md.function @pull(%x: !vec, %cell: !md.cell, %bonds: !bonds, %k: !of_bond)
    -> !vec {
  // CHECK: md.gather_tuples %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(displacement(0, 1))
  // CHECK-SAME: tuple(%{{[a-z0-9]+}} : !md.field<@bonds, f64>) {
  // CHECK: md.yield %{{[a-z0-9]+}}, %{{[a-z0-9]+}} : vector<3xf64>, vector<3xf64>
  // CHECK: } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f64>
  %f = md.gather_tuples %bonds, %x, %cell coordinates(displacement(0, 1))
         tuple(%k : !of_bond) {
  ^bb0(%d: vector<3xf64>, %k_t: f64):
    %b = vector.broadcast %k_t : f64 to vector<3xf64>
    %f1 = arith.mulf %b, %d : vector<3xf64>
    %f0 = arith.negf %f1 : vector<3xf64>
    md.yield %f0, %f1 : vector<3xf64>, vector<3xf64>
  } : !bonds, !vec -> !vec
  md.return %f : !vec
}

// A neighborhood that leaves out pairs of a tuple set.
//
// CHECK-LABEL: md.function @excluding(
// CHECK: md.neighborhood %{{[a-z0-9]+}}, %{{[a-z0-9]+}} cutoff(1.000000e+00) exclude(%{{[a-z0-9]+}} : !md.relation<@atoms, 2, unordered, @bonds>)
md.function @excluding(%x: !vec, %cell: !md.cell, %bonds: !bonds)
    -> !md.relation<@atoms, 2, unordered> {
  %n = md.neighborhood %x, %cell cutoff(1.0) exclude(%bonds : !bonds)
         : !vec -> !md.relation<@atoms, 2, unordered>
  md.return %n : !md.relation<@atoms, 2, unordered>
}

// A tuple of one member, as a restraint of a position has. No coordinate
// takes fewer than two members, so that no op takes such a relation yet.
//
// CHECK-LABEL: md.function @declared(
// CHECK-SAME:    !md.relation<@atoms, 1, ordered, @restraints>
md.function @declared(%r: !restraints) {
  md.return
}
