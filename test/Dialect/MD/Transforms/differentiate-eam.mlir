// The derivative of EAM on domains (D[ml2-eam], docs/distributed-ml2.md,
// Section 15): the adjoint of the density, F'(ρ), is a map over the owned
// rows, copied to the ghosts between the two passes, and read at both
// particles of each pair by the gather of the forces, whose ghost
// contributions return to the owners by G^T. On the unique traversal the
// copy is the transpose of the density's halo_sum; on the directed one the
// derivative inserts it. md-dist-owner-computes removes G^T and keeps the
// copy.
//
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=unique potentials=eam" --md-differentiate | FileCheck %s --check-prefixes=CHECK,UNIQUE,FORM
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed potentials=eam" --md-differentiate | FileCheck %s --check-prefixes=CHECK,DIRECTED,FORM
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed potentials=eam" --md-differentiate --md-dist-owner-computes | FileCheck %s --check-prefixes=CHECK,DIRECTED,OWNER
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed potentials=many_body" --md-differentiate | FileCheck %s --check-prefix=FIELD

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @eam(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md.yield %f : f64
  } : !pairs, !vec -> !sca
  %e1 = md.sum_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    %m = arith.negf %s : f64
    md.yield %m : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %k = arith.mulf %t, %t : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @eam.energy_forces_virial
// CHECK: %[[H:.*]] = md_dist.halo
// CHECK: %[[XG:.*]] = md_dist.halo_copy %[[XO:.*]], %[[H]] image(%arg1)
// The density: complete at the owners (directed), or completed by G^T
// (unique).
// DIRECTED: %[[RHO:.*]] = md_dist.gather_pairs {{.*}} exchange(symmetric)
// DIRECTED: -> !md_dist.field<@eam.domains, owned, f64>
// UNIQUE: %[[RP:.*]]:2 = md_dist.gather_pairs {{.*}} exchange(symmetric)
// UNIQUE: %[[RHO:.*]] = md_dist.halo_sum %[[RP]]#0, %[[RP]]#1, %[[H]]
// CHECK: md_dist.sum_particles gather(%[[RHO]] :
// The forces of the pair term.
// FORM: %[[K:.*]]:2 = md_dist.gather_pairs {{.*}} exchange(antisymmetric)
// OWNER: %[[K:.*]] = md_dist.gather_pairs {{.*}} exchange(antisymmetric)
// The adjoint of the density, and its copy to the ghosts.
// CHECK: %[[RB:.*]] = md_dist.map_particles gather(%[[RHO]] :
// CHECK: %[[RBG:.*]] = md_dist.halo_copy %[[RB]], %[[H]] : !md_dist.field<@eam.domains, owned, f64>
// The forces of the density, from F'(ρ) at both particles.
// FORM: %[[G:.*]]:2 = md_dist.gather_pairs %{{.*}}, %[[XO]], %[[XG]] gather(%[[RB]], %[[RBG]] : {{.*}}) exchange(antisymmetric)
// FORM: -> !md_dist.field<@eam.domains, owned, partial, 3 x f64>, !md_dist.field<@eam.domains, ghost, partial, 3 x f64>
// FORM: md_dist.halo_sum %[[K]]#0, %[[K]]#1, %[[H]]
// FORM: md_dist.halo_sum %[[G]]#0, %[[G]]#1, %[[H]]
// OWNER: %[[G:.*]] = md_dist.gather_pairs %{{.*}}, %[[XO]], %[[XG]] gather(%[[RB]], %[[RBG]] : {{.*}}) exchange(antisymmetric)
// OWNER: -> !md_dist.field<@eam.domains, owned, 3 x f64>
// OWNER-NOT: md_dist.halo_sum
// CHECK: md_dist.collect
// The virial of the pair term and of the density.
// CHECK: md_dist.sum_pairs {{.*}} exchange(symmetric)
// CHECK: -> !md_dist.partial<vector<9xf64>>
// CHECK: md_dist.sum_pairs %{{.*}}, %[[XO]], %[[XG]] gather(%[[RB]], %[[RBG]] : {{.*}}) exchange(symmetric)
// CHECK: -> !md_dist.partial<vector<9xf64>>
// CHECK: md.return
func.func @f(%x: !vec, %cell: !md.cell) -> (f64, !vec, vector<9xf64>) {
  %u, %f, %w = md.evaluate @eam(%x, %cell) request [energy, forces, virial]
      : (!vec, !md.cell) -> (f64, !vec, vector<9xf64>)
  return %u, %f, %w : f64, !vec, vector<9xf64>
}

// A pair sum that reads ψ = √ρ at both particles: on the directed traversal
// the adjoint of ψ is the gather at the owners (Section 15.4), with nothing
// returned by G^T for ψ.
md.potential @many_body(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  %psi = md.map_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md.yield %s : f64
  } : !sca
  %u = md.sum_relation %n, %x, %cell gather(%psi : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %pi: f64, %pj: f64):
    %p = arith.mulf %pi, %pj : f64
    %k = arith.divf %p, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// FIELD-LABEL: md.function @many_body.forces
// FIELD: %[[H:.*]] = md_dist.halo
// FIELD: %[[RHO:.*]] = md_dist.gather_pairs
// FIELD: %[[PSI:.*]] = md_dist.map_particles gather(%[[RHO]] :
// FIELD: %[[PG:.*]] = md_dist.halo_copy %[[PSI]], %[[H]]
// FIELD: %[[PB:.*]] = md_dist.gather_pairs {{.*}} gather(%[[PSI]], %[[PG]] : {{.*}}) exchange(none)
// FIELD: -> !md_dist.field<@many_body.domains, owned, f64>
// FIELD: md_dist.gather_pairs {{.*}} gather(%[[PSI]], %[[PG]] : {{.*}}) exchange(antisymmetric)
// FIELD: %[[RB:.*]] = md_dist.map_particles gather(%[[RHO]], %[[PB]] :
// FIELD: %[[RBG:.*]] = md_dist.halo_copy %[[RB]], %[[H]]
// FIELD: md_dist.gather_pairs {{.*}} gather(%[[RB]], %[[RBG]] : {{.*}}) exchange(antisymmetric)
// FIELD: md.return
func.func @g(%x: !vec, %cell: !md.cell) -> !vec {
  %f = md.evaluate @many_body(%x, %cell) request [forces]
      : (!vec, !md.cell) -> !vec
  return %f : !vec
}
