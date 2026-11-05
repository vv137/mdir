// RUN: mdir-opt %s --md-differentiate -split-input-file | FileCheck %s

// The tangent of a potential in the positions, Φ = Σ_i t_i · ∂U/∂x_i, and
// its derivative in an argument at fixed t, the mixed derivative of force
// matching (D376, #726). md-differentiate builds Φ as a
// potential `<name>.tangent` that takes t last, asks it for its energy and
// derivatives by the rules of D161 and D230, and erases it: the functions
// hold copies of its body. The values are checked by central differences
// in test/Integration/tangent.mlir; the errors are in
// differentiate-tangent-invalid.mlir.

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// u = eps / r + q_i q_j: the kernel of the tangent gathers t at both ends
// and yields u'(r) d · (t_i − t_j) / r; the sum over particles of the
// charges alone reads no position, and has no tangent. The derivative of
// Φ in eps is a sum of the derivative of that kernel, and that in the
// charges a gather: the sum q_i q_j does not depend on r, so that only the
// sum over pairs whose kernel does takes part.
//
// CHECK-NOT: md.potential @pair.tangent
// CHECK-LABEL: md.function @pair.energy(
// CHECK-LABEL: md.function @pair.tangent.energy_derivative2_derivative3(
// CHECK-SAME: %[[T:[^:]*]]: !md.field<@atoms, 3 x f64>) -> (f64, f64, !md.field<@atoms, f64>)
// CHECK: %[[P:.*]] = md.sum_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%{{.*}}, %[[T]] : !md.field<@atoms, f64>, !md.field<@atoms, 3 x f64>) exchange(symmetric, derived)
// CHECK: ^bb0(%[[R:[^:]*]]: f64, %[[D:[^:]*]]: vector<3xf64>, %{{[^:]*}}: f64, %{{[^:]*}}: f64, %[[TI:[^:]*]]: vector<3xf64>, %[[TJ:[^:]*]]: vector<3xf64>):
// CHECK: %[[DT:.*]] = arith.subf %[[TI]], %[[TJ]] : vector<3xf64>
// CHECK: %[[M:.*]] = arith.mulf %[[D]], %[[DT]] : vector<3xf64>
// CHECK: %[[A:.*]] = vector.reduction <add>, %[[M]] : vector<3xf64> into f64
// CHECK: %[[S:.*]] = arith.divf %{{.*}}, %[[R]] : f64
// CHECK: %[[K:.*]] = arith.mulf %[[S]], %[[A]] : f64
// CHECK: md.yield %[[K]] : f64
// CHECK-NOT: md.sum_particles
// CHECK: md.return %[[P]],
// CHECK-NOT: md.potential @pair.tangent
md.potential @pair(%x: !vec, %cell: !md.cell, %eps: f64, %q: !real) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell gather(%q : !real) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
    %a = arith.divf %eps, %r : f64
    md.yield %a : f64
  } : !pairs, !vec -> f64
  %s = md.sum_particles gather(%q : !real) {
  ^bb0(%qi: f64):
    %k = arith.mulf %qi, %qi : f64
    md.yield %k : f64
  } : f64
  %e = arith.addf %u, %s : f64
  md.return %e : f64
}

md.function @requests(%x: !vec, %cell: !md.cell, %eps: f64, %q: !real,
                      %t: !vec) -> (f64, f64, f64, !real) {
  // The requests of U and those of Φ, each in a call of its own, in the
  // order of the results.
  // CHECK-LABEL: md.function @requests(
  // CHECK: %[[U:.*]] = md.call @pair.energy(
  // CHECK: %[[PHI:.*]]:3 = md.call @pair.tangent.energy_derivative2_derivative3(
  // CHECK: md.return %[[U]], %[[PHI]]#0, %[[PHI]]#1, %[[PHI]]#2
  %u, %p, %pe, %pq = md.evaluate @pair(%x, %cell, %eps, %q, %t)
      request [energy, tangent, tangent_derivative(2), tangent_derivative(3)]
      : (!vec, !md.cell, f64, !real, !vec) -> (f64, f64, f64, !real)
  md.return %u, %p, %pe, %pq : f64, f64, f64, !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!dihedrals = !md.relation<@atoms, 4, reversal, @dihedrals>
md.particle_set @atoms
md.tuple_set @dihedrals on(@atoms) arity(4) orientation(reversal)

// u = k (1 + cos 3φ): the tangent is (∂u/∂φ) Σ_m ∂φ/∂x_m · t_m, with the
// gradient of the dihedral from the three displacements that the op
// receives as coordinates, and the tangent at the four members after them.
//
// CHECK-LABEL: md.function @twist.tangent.energy_derivative2(
// CHECK: md.sum_tuples %{{.*}}, %{{.*}}, %{{.*}} coordinates(dihedral(0, 1, 2, 3), displacement(0, 1), displacement(1, 2), displacement(3, 2)) gather(%{{.*}} : !md.field<@atoms, 3 x f64>)
// CHECK: ^bb0(%{{[^:]*}}: f64, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: vector<3xf64>):
// CHECK-COUNT-4: vector.reduction <add>
// CHECK: md.yield
md.potential @twist(%x: !vec, %cell: !md.cell, %k: f64,
                    %dihedrals: !dihedrals) -> f64 {
  %u = md.sum_tuples %dihedrals, %x, %cell coordinates(dihedral(0, 1, 2, 3)) {
  ^bb0(%phi: f64):
    %one = arith.constant 1.0 : f64
    %three = arith.constant 3.0 : f64
    %p3 = arith.mulf %three, %phi : f64
    %c = math.cos %p3 : f64
    %s = arith.addf %one, %c : f64
    %e = arith.mulf %k, %s : f64
    md.yield %e : f64
  } : !dihedrals, !vec -> f64
  md.return %u : f64
}

md.function @requests(%x: !vec, %cell: !md.cell, %k: f64,
                      %dihedrals: !dihedrals, %t: !vec) -> (f64, f64) {
  %p, %pk = md.evaluate @twist(%x, %cell, %k, %dihedrals, %t)
      request [tangent, tangent_derivative(2)]
      : (!vec, !md.cell, f64, !dihedrals, !vec) -> (f64, f64)
  md.return %p, %pk : f64, f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A term of the absolute positions: the derivative of its kernel along t_i.
// A potential that does not read the positions has the tangent zero.
//
// CHECK-LABEL: md.function @well.tangent.energy(
// CHECK: md.sum_particles gather(%{{.*}}, %[[T:[^ ]*]] : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>)
// CHECK: ^bb0(%[[XI:[^:]*]]: vector<3xf64>, %[[TI:[^:]*]]: vector<3xf64>):
// CHECK: md.yield
// CHECK-LABEL: md.function @flat.tangent.energy(
// CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: md.return %[[Z]] : f64
md.potential @well(%x: !vec, %cell: !md.cell, %c: f64) -> f64 {
  %u = md.sum_particles gather(%x : !vec) {
  ^bb0(%xi: vector<3xf64>):
    %x2 = arith.mulf %xi, %xi : vector<3xf64>
    %s = vector.reduction <add>, %x2 : vector<3xf64> into f64
    %e = arith.mulf %c, %s : f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

md.potential @flat(%x: !vec, %cell: !md.cell, %c: f64) -> f64 {
  md.return %c : f64
}

md.function @requests(%x: !vec, %cell: !md.cell, %c: f64, %t: !vec)
    -> (f64, f64) {
  %p = md.evaluate @well(%x, %cell, %c, %t) request [tangent]
      : (!vec, !md.cell, f64, !vec) -> f64
  %z = md.evaluate @flat(%x, %cell, %c, %t) request [tangent]
      : (!vec, !md.cell, f64, !vec) -> f64
  md.return %p, %z : f64, f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!bonds = !md.relation<@atoms, 2, unordered, @bonds>
!of_bond = !md.field<@bonds, f64>
md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// The tangent at the members comes after the other gathered fields and
// before the fields of the tuples; the derivative of Φ in the charges is a
// gather over the same tuples, with the coordinates and the fields of Φ.
//
// CHECK-LABEL: md.function @bond.tangent.derivative5(
// CHECK: md.gather_tuples %{{.*}}, %{{.*}}, %{{.*}} coordinates(distance(0, 1), displacement(0, 1)) gather(%{{.*}}, %{{.*}} : !md.field<@atoms, f64>, !md.field<@atoms, 3 x f64>) tuple(%{{.*}}, %{{.*}} : !md.field<@bonds, f64>, !md.field<@bonds, f64>)
// CHECK: ^bb0(%{{[^:]*}}: f64, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: f64, %{{[^:]*}}: f64, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: f64, %{{[^:]*}}: f64):
md.potential @bond(%x: !vec, %cell: !md.cell, %bonds: !bonds, %k: !of_bond,
                   %r0: !of_bond, %q: !real) -> f64 {
  %u = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1))
         gather(%q : !real) tuple(%k, %r0 : !of_bond, !of_bond) {
  ^bb0(%r: f64, %qa: f64, %qb: f64, %k_t: f64, %r0_t: f64):
    %dr = arith.subf %r, %r0_t : f64
    %dr2 = arith.mulf %dr, %dr : f64
    %e0 = arith.mulf %k_t, %dr2 : f64
    %e = arith.mulf %e0, %qa : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64
  md.return %u : f64
}

md.function @requests(%x: !vec, %cell: !md.cell, %bonds: !bonds,
                      %k: !of_bond, %r0: !of_bond, %q: !real, %t: !vec)
    -> !real {
  %g = md.evaluate @bond(%x, %cell, %bonds, %k, %r0, %q, %t)
      request [tangent_derivative(5)]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !real, !vec) -> !real
  md.return %g : !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!grid = !md.table<2, f64>
md.particle_set @atoms

// The tangent of a reciprocal sum is −Σ_i t_i · F_i from the forces that
// it yields; its derivative in a number that the charges depend on is
// (Φ(q + Δ) − Φ(q − Δ)) / 2 with Δ = ∂q/∂λ, from two more reciprocal sums
// at the shifted charges (D379); the mark of the pass is
// gone from the result.
//
// CHECK-LABEL: md.function @mesh.tangent.energy_derivative4(
// CHECK-SAME: %[[T:[^:]*]]: !md.field<@atoms, 3 x f64>) -> (f64, f64)
// CHECK-NOT: md.reciprocal_tangent
// CHECK: %{{.*}}, %[[F:.*]], %{{.*}} = md.reciprocal
// CHECK: %[[P:.*]] = md.sum_particles gather(%[[F]], %[[T]] :
// CHECK: ^bb0(%[[FI:[^:]*]]: vector<3xf64>, %[[TI:[^:]*]]: vector<3xf64>):
// CHECK: %[[M:.*]] = arith.mulf %[[FI]], %[[TI]] : vector<3xf64>
// CHECK: %[[S:.*]] = vector.reduction <add>, %[[M]]
// CHECK: %[[N:.*]] = arith.negf %[[S]] : f64
// CHECK: md.yield %[[N]] : f64
// CHECK: %[[QP:.*]] = md.map_particles
// CHECK: %{{.*}}, %[[FP:.*]], %{{.*}} = md.reciprocal %{{.*}}, %[[QP]],
// CHECK: %[[PP:.*]] = md.sum_particles gather(%[[FP]], %[[T]] :
// CHECK: %[[QM:.*]] = md.map_particles
// CHECK: %{{.*}}, %[[FM:.*]], %{{.*}} = md.reciprocal %{{.*}}, %[[QM]],
// CHECK: %[[PM:.*]] = md.sum_particles gather(%[[FM]], %[[T]] :
// CHECK: %[[D:.*]] = arith.subf %[[PP]], %[[PM]] : f64
// CHECK: %[[H:.*]] = arith.mulf %{{.*}}, %[[D]] : f64
// CHECK: md.return %[[P]], %[[H]] : f64, f64
// CHECK-NOT: md.reciprocal_tangent
md.potential @mesh(%x: !vec, %cell: !md.cell, %q0: !real, %w: !real,
                   %lam: f64, %moduli: !grid) -> f64 {
  %q = md.map_particles gather(%q0, %w : !real, !real) {
  ^bb0(%a: f64, %b: f64):
    %s = arith.mulf %lam, %b : f64
    %c = arith.addf %a, %s : f64
    md.yield %c : f64
  } : !real
  %u, %f, %v = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  md.return %u : f64
}

md.function @requests(%x: !vec, %cell: !md.cell, %q0: !real, %w: !real,
                      %lam: f64, %moduli: !grid, %t: !vec) -> (f64, f64) {
  %p, %dp = md.evaluate @mesh(%x, %cell, %q0, %w, %lam, %moduli, %t)
      request [tangent, tangent_derivative(4)]
      : (!vec, !md.cell, !real, !real, f64, !grid, !vec) -> (f64, f64)
  md.return %p, %dp : f64, f64
}
