// RUN: mdir-opt %s --md-differentiate -split-input-file | FileCheck %s
// RUN: mdir-opt %s --md-differentiate="remarks=true" -split-input-file -verify-diagnostics -o /dev/null

// The derivative of the energy in a field of the particles
// (D230): `derivative(n)` of an argument that is a field of
// f64 is a field, g_i = dU/da_i. A sum over a relation gives a gather over
// it, a sum over tuples a gather over them, a sum over particles a map; a
// field that no sum of the energy takes has the derivative zero; a
// reciprocal sum of the charges yields it as a fourth result
// (D231). The errors are in
// differentiate-fields-invalid.mlir.

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// u = q_i q_j / r: the kernel of the gather is the derivative in the charge
// of the central particle, q_j / r, with no exchange contract, so that each
// particle of a pair takes its own.
//
// CHECK-LABEL: md.function @coulomb.derivative2(
// CHECK: %[[G:.*]] = md.gather_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%{{.*}} : !md.field<@atoms, f64>)
// CHECK-SAME: exchange(none, derived)
// CHECK: ^bb0(%[[R:[^:]*]]: f64, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: f64, %[[QJ:[^:]*]]: f64):
// CHECK: %[[K:.*]] = arith.divf %[[QJ]], %[[R]] : f64
// CHECK: md.yield %[[K]] : f64
// CHECK: -> !md.field<@atoms, f64>
// CHECK: md.return %[[G]] : !md.field<@atoms, f64>
md.potential @coulomb(%x: !vec, %cell: !md.cell, %q: !real) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell gather(%q : !real) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %q_i: f64, %q_j: f64):
    %qq = arith.mulf %q_i, %q_j : f64
    %k = arith.divf %qq, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

md.function @first(%x: !vec, %cell: !md.cell, %q: !real) -> !real {
  %g = md.evaluate @coulomb(%x, %cell, %q) request [derivative(2)]
      : (!vec, !md.cell, !real) -> !real
  md.return %g : !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!rel = !md.relation<@atoms, 2, unordered, @bonds>
md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// A sum over tuples that gathers the field: each member takes the
// derivative in its own value, times the weight of the sum in the energy,
// here 3.
//
// CHECK-LABEL: md.function @tuples.derivative2(
// CHECK: %[[THREE:.*]] = arith.constant 3.0{{.*}} : f64
// CHECK: %[[G:.*]] = md.gather_tuples
// CHECK: ^bb0(%{{[^:]*}}: f64, %[[QI:[^:]*]]: f64, %[[QJ:[^:]*]]: f64):
// CHECK-DAG: %[[A:.*]] = arith.mulf %[[THREE]], %[[QJ]] : f64
// CHECK-DAG: %[[B:.*]] = arith.mulf %[[THREE]], %[[QI]] : f64
// CHECK: md.yield %[[A]], %[[B]] : f64, f64
// CHECK: md.return %[[G]]
md.potential @tuples(%x: !vec, %cell: !md.cell, %q: !real, %r_bonds: !rel)
    -> f64 {
  %s = md.sum_tuples %r_bonds, %x, %cell coordinates(distance(0, 1))
      gather(%q : !real) {
  ^bb0(%r: f64, %q_i: f64, %q_j: f64):
    %k = arith.mulf %q_i, %q_j : f64
    md.yield %k : f64
  } : !rel, !vec -> f64
  %three = arith.constant 3.0 : f64
  %u = arith.mulf %three, %s : f64
  md.return %u : f64
}

md.function @second(%x: !vec, %cell: !md.cell, %q: !real, %r_bonds: !rel)
    -> !real {
  %g = md.evaluate @tuples(%x, %cell, %q, %r_bonds) request [derivative(2)]
      : (!vec, !md.cell, !real, !rel) -> !real
  md.return %g : !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
md.particle_set @atoms

// A sum over particles gives a map with the derivative of its kernel, and
// the two sums that take the field add particle by particle.
//
// CHECK-LABEL: md.function @particles.derivative2(
// CHECK: %[[A:.*]] = md.map_particles gather(%{{.*}}, %{{.*}} :
// CHECK: ^bb0(%[[Q:[^:]*]]: f64, %[[M:[^:]*]]: f64):
// CHECK: md.yield %[[M]] : f64
// CHECK: %[[B:.*]] = md.map_particles gather(%{{.*}} :
// CHECK: ^bb0(%[[Q2:[^:]*]]: f64):
// CHECK: %[[S:.*]] = arith.addf %[[Q2]], %[[Q2]] : f64
// CHECK: md.yield %[[S]] : f64
// CHECK: %[[T:.*]] = md.map_particles gather(%[[A]], %[[B]] :
// CHECK: md.return %[[T]]
md.potential @particles(%x: !vec, %cell: !md.cell, %q: !real, %m: !real)
    -> f64 {
  %s = md.sum_particles gather(%q, %m : !real, !real) {
  ^bb0(%q_i: f64, %m_i: f64):
    %k = arith.mulf %q_i, %m_i : f64
    md.yield %k : f64
  } : f64
  %t = md.sum_particles gather(%q : !real) {
  ^bb0(%q_i: f64):
    %k = arith.mulf %q_i, %q_i : f64
    md.yield %k : f64
  } : f64
  %u = arith.addf %s, %t : f64
  md.return %u : f64
}

md.function @third(%x: !vec, %cell: !md.cell, %q: !real, %m: !real) -> !real {
  %g = md.evaluate @particles(%x, %cell, %q, %m) request [derivative(2)]
      : (!vec, !md.cell, !real, !real) -> !real
  md.return %g : !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
md.particle_set @atoms

// The energy takes %m alone: the derivative in %q is zero by proof, a field
// of zeros, and no sum remains.
//
// CHECK-LABEL: md.function @untouched.derivative2(
// CHECK-NOT: md.sum_particles
// CHECK: %[[Z:.*]] = md.map_particles
// CHECK: %[[ZERO:.*]] = arith.constant 0.0{{.*}} : f64
// CHECK: md.yield %[[ZERO]] : f64
// CHECK: md.return %[[Z]]
// expected-remark@+1 {{independent of argument 2 of 'untouched': its derivative is zero}}
md.potential @untouched(%x: !vec, %cell: !md.cell, %q: !real, %m: !real)
    -> f64 {
  %s = md.sum_particles gather(%m : !real) {
  ^bb0(%m_i: f64):
    md.yield %m_i : f64
  } : f64
  md.return %s : f64
}

md.function @fourth(%x: !vec, %cell: !md.cell, %q: !real, %m: !real) -> !real {
  %g = md.evaluate @untouched(%x, %cell, %q, %m) request [derivative(2)]
      : (!vec, !md.cell, !real, !real) -> !real
  md.return %g : !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!grid = !md.table<2, f64>
md.particle_set @atoms

// The reciprocal sum of the charges: the derivative of its energy in the
// charge of a particle is the potential of the grid at the particle, the
// fourth result of the op, which takes the place of the op of the
// potential and yields its energy as before; here times the weight of the
// sum in the energy, 2, and with the energy requested as well.
//
// CHECK-LABEL: md.function @mesh.energy_derivative2(
// CHECK: %[[TWO:.*]] = arith.constant 2.0{{.*}} : f64
// CHECK: %[[U:.*]], %{{.*}}, %{{.*}}, %[[P:.*]] = md.reciprocal %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} grid([8, 8, 8]) order(4)
// CHECK-SAME: -> f64, !md.field<@atoms, 3 x f64>, vector<9xf64>, !md.field<@atoms, f64>
// CHECK-NOT: md.reciprocal
// CHECK: %[[E:.*]] = arith.mulf %[[TWO]], %[[U]] : f64
// CHECK: %[[G:.*]] = md.map_particles gather(%[[P]] :
// CHECK: ^bb0(%[[PI:[^:]*]]: f64):
// CHECK: %[[S:.*]] = arith.mulf %[[TWO]], %[[PI]] : f64
// CHECK: md.yield %[[S]] : f64
// CHECK: md.return %[[E]], %[[G]] : f64, !md.field<@atoms, f64>
md.potential @mesh(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> f64 {
  %two = arith.constant 2.0 : f64
  %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  %e = arith.mulf %two, %u : f64
  md.return %e : f64
}

md.function @fifth(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !real) {
  %e, %g = md.evaluate @mesh(%x, %cell, %q, %moduli)
      request [energy, derivative(2)]
      : (!vec, !md.cell, !real, !grid) -> (f64, !real)
  md.return %e, %g : f64, !real
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// The derivative in a number and the derivative in a field, of one energy
// (#256): two first derivatives, neither of which enters the other. The
// derivative in the number %a copies each sum that reads it with the fields
// that the sum gathers, among them %q, which the kernel of the first does
// not read; the derivative in %q, requested after it, follows the sums of
// the potential and not those copies. The term a / r gives nothing in %q,
// and a q_i q_j / r gives a q_j / r.
//
// CHECK-LABEL: md.function @mixed.energy_derivative3_derivative2(
// CHECK-SAME: %[[A:[^:]*]]: f64)
// CHECK: %[[DA1:.*]] = md.sum_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%{{.*}} : !md.field<@atoms, f64>) exchange(symmetric, derived)
// CHECK: %[[DA2:.*]] = md.sum_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%{{.*}} : !md.field<@atoms, f64>) exchange(symmetric, derived)
// CHECK: %[[DA:.*]] = arith.addf %[[DA1]], %[[DA2]] : f64
// CHECK: %[[G:.*]] = md.gather_relation %{{.*}}, %{{.*}}, %{{.*}} gather(%{{.*}} : !md.field<@atoms, f64>)
// CHECK-SAME: exchange(none, derived)
// CHECK: ^bb0(%[[R:[^:]*]]: f64, %{{[^:]*}}: vector<3xf64>, %{{[^:]*}}: f64, %[[QJ:[^:]*]]: f64):
// CHECK: %[[AQ:.*]] = arith.mulf %[[A]], %[[QJ]] : f64
// CHECK: %[[K:.*]] = arith.divf %[[AQ]], %[[R]] : f64
// CHECK: md.yield %[[K]] : f64
// CHECK-NOT: md.gather_relation
// CHECK: md.return %{{.*}}, %[[DA]], %[[G]] : f64, f64, !md.field<@atoms, f64>
//
// The same field with the requests in the other order.
// CHECK-LABEL: md.function @mixed.derivative2_derivative3(
// CHECK: %[[G2:.*]] = md.gather_relation
// CHECK-NOT: md.gather_relation
// CHECK: md.return %[[G2]], %{{.*}} : !md.field<@atoms, f64>, f64
//
// After the forces, whose gathers take the field as well.
// CHECK-LABEL: md.function @mixed.forces_derivative2(
// CHECK: %[[F:.*]] = md.gather_relation {{.*}} exchange(antisymmetric, derived)
// CHECK: %[[G3:.*]] = md.gather_relation {{.*}} exchange(none, derived)
// CHECK: md.return %{{.*}}, %[[G3]] : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>
md.potential @mixed(%x: !vec, %cell: !md.cell, %q: !real, %a: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0) : !vec -> !pairs
  %s = md.sum_relation %n, %x, %cell gather(%q : !real) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %q_i: f64, %q_j: f64):
    %k = arith.divf %a, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %c = md.sum_relation %n, %x, %cell gather(%q : !real) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %q_i: f64, %q_j: f64):
    %qq = arith.mulf %q_i, %q_j : f64
    %aqq = arith.mulf %a, %qq : f64
    %k = arith.divf %aqq, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %s, %c : f64
  md.return %u : f64
}

md.function @sixth(%x: !vec, %cell: !md.cell, %q: !real, %a: f64)
    -> (f64, f64, !real, !real, f64, !vec, !real) {
  %e, %da, %g = md.evaluate @mixed(%x, %cell, %q, %a)
      request [energy, derivative(3), derivative(2)]
      : (!vec, !md.cell, !real, f64) -> (f64, f64, !real)
  %g2, %da2 = md.evaluate @mixed(%x, %cell, %q, %a)
      request [derivative(2), derivative(3)]
      : (!vec, !md.cell, !real, f64) -> (!real, f64)
  %f, %g3 = md.evaluate @mixed(%x, %cell, %q, %a)
      request [forces, derivative(2)]
      : (!vec, !md.cell, !real, f64) -> (!vec, !real)
  md.return %e, %da, %g, %g2, %da2, %f, %g3
      : f64, f64, !real, !real, f64, !vec, !real
}
