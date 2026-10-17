// The derivative of a potential on domains carries the adjoints of the
// transfers back by their declared transposes: the forces of the ghost rows
// return to their owners by G^T (md_dist.halo_sum), those of the owned rows
// to the particle set by md_dist.collect (D276). On the
// directed traversal the derivative's form computes half of every pair at
// each of its particles; md-dist-owner-computes replaces it by the gather
// at the owners, with nothing returned.
//
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=unique" --md-differentiate | FileCheck %s --check-prefixes=CHECK,FORM
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed" --md-differentiate | FileCheck %s --check-prefixes=CHECK,FORM
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed" --md-differentiate --md-dist-owner-computes | FileCheck %s --check-prefixes=CHECK,OWNER
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=unique" --md-differentiate --md-dist-owner-computes | FileCheck %s --check-prefixes=CHECK,FORM

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// CHECK-LABEL: md.function @lj.energy_forces_virial
// CHECK: %[[L:.*]] = md_dist.partition
// CHECK: %[[XO:.*]] = md_dist.distribute %arg0, %[[L]] image(%arg1)
// CHECK: %[[H:.*]] = md_dist.halo
// CHECK: %[[XG:.*]] = md_dist.halo_copy %[[XO]], %[[H]] image(%arg1)
// CHECK: %[[N:.*]] = md_dist.neighborhood %[[XO]], %[[XG]], %[[H]]
// CHECK: %[[UP:.*]] = md_dist.sum_pairs %[[N]], %[[XO]], %[[XG]] exchange(symmetric)
// CHECK: %[[U:.*]] = md_dist.all_reduce %[[UP]]
// The forces of the pairs, at both of their particles, the ghosts' returned
// by the transpose of the copy of the positions.
// FORM: %[[G:.*]]:2 = md_dist.gather_pairs %[[N]], %[[XO]], %[[XG]] exchange(antisymmetric)
// FORM: } : {{.*}} -> !md_dist.field<@lj.domains, owned, partial, 3 x f64>, !md_dist.field<@lj.domains, ghost, partial, 3 x f64>
// FORM: %[[FO:.*]] = md_dist.halo_sum %[[G]]#0, %[[G]]#1, %[[H]]
// At the owners, nothing returned.
// OWNER: %[[FO:.*]] = md_dist.gather_pairs %[[N]], %[[XO]], %[[XG]] exchange(antisymmetric)
// OWNER: } : !md_dist.relation<@lj.domains, directed>, {{.*}} -> !md_dist.field<@lj.domains, owned, 3 x f64>
// OWNER-NOT: md_dist.halo_sum
// The transpose of the distribution.
// CHECK: %[[F:.*]] = md_dist.collect %[[FO]], %[[L]]
// The virial: a sum over the pairs of the traversal.
// CHECK: %[[WP:.*]] = md_dist.sum_pairs %[[N]], %[[XO]], %[[XG]] exchange(symmetric)
// CHECK: -> !md_dist.partial<vector<9xf64>>
// CHECK: %[[W:.*]] = md_dist.all_reduce %[[WP]] : !md_dist.partial<vector<9xf64>> -> vector<9xf64>
// CHECK: md.return %[[U]], %[[F]], %[[W]]
func.func @f(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64)
    -> (f64, !vec, vector<9xf64>) {
  %u, %f, %w = md.evaluate @lj(%x, %cell, %eps, %sigma)
      request [energy, forces, virial]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec, vector<9xf64>)
  return %u, %f, %w : f64, !vec, vector<9xf64>
}
