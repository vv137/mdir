// md-distribute rewrites a global potential into md_dist (D276,
// docs/distributed-ml2.md, Section 14).
// RUN: mdir-opt %s --md-distribute="grid=2,1,1" | FileCheck %s --check-prefixes=CHECK,DIRECTED
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=unique" | FileCheck %s --check-prefixes=CHECK,UNIQUE
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 reach=2.5 potentials=lj" | FileCheck %s --check-prefix=NAMED
// The automatic choice and an explicit override produce the same IR.
// RUN: mdir-opt %s --md-distribute="domains=6 box=6,3,1 potentials=lj" > %t.auto
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 potentials=lj" > %t.grid
// RUN: diff %t.auto %t.grid
// RUN: mdir-opt %s --md-distribute="domains=6 box=6,3,1 potentials=lj" | FileCheck %s --check-prefix=FACTOR
// RUN: mdir-opt %s --md-distribute="domains=6 box=1,3,6 potentials=lj" | FileCheck %s --check-prefix=PERMUTED
// RUN: mdir-opt %s --md-distribute="domains=2 box=1,1,1 potentials=lj" | FileCheck %s --check-prefix=TIE
// RUN: mdir-opt %s --md-distribute="domains=7 box=4,8,2 potentials=lj" | FileCheck %s --check-prefix=PRIME
// Every op of the result has a rule for the derivative or is semantic.
// RUN: mdir-opt %s --md-distribute="grid=2,1,1" --md-check-derivative-coverage -o /dev/null

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

// CHECK: md_dist.decomposition @lj.domains on(@atoms) grid([2, 1, 1])
// NAMED: md_dist.decomposition @lj.domains on(@atoms) grid([3, 2, 1])
// FACTOR: md_dist.decomposition @lj.domains on(@atoms) grid([3, 2, 1])
// PERMUTED: md_dist.decomposition @lj.domains on(@atoms) grid([1, 2, 3])
// TIE: md_dist.decomposition @lj.domains on(@atoms) grid([1, 1, 2])
// PRIME: md_dist.decomposition @lj.domains on(@atoms) grid([1, 7, 1])
// CHECK-LABEL: md.potential @lj
// The layout, the owned positions wrapped into the cell, one halo of the
// cutoff, and the ghost positions with their images.
// CHECK: %[[L:.*]] = md_dist.partition %arg0, %arg1 : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@lj.domains>
// CHECK: %[[XO:.*]] = md_dist.distribute %arg0, %[[L]] image(%arg1)
// CHECK: %[[H:.*]] = md_dist.halo %[[L]], %[[XO]], %arg1 : {{.*}} -> !md_dist.halo<@lj.domains, 2.0>
// NAMED: md_dist.halo {{.*}} -> !md_dist.halo<@lj.domains, 2.5>
// CHECK: %[[XG:.*]] = md_dist.halo_copy %[[XO]], %[[H]] image(%arg1)
// DIRECTED: %[[N:.*]] = md_dist.neighborhood %[[XO]], %[[XG]], %[[H]] cutoff(2.000000e+00) : {{.*}} -> !md_dist.relation<@lj.domains, directed>
// UNIQUE: %[[N:.*]] = md_dist.neighborhood %[[XO]], %[[XG]], %[[H]] cutoff(2.000000e+00) : {{.*}} -> !md_dist.relation<@lj.domains, unique>
// The truncation is in the kernel: u(r) − u(r_c).
// CHECK: %[[P:.*]] = md_dist.sum_pairs %[[N]], %[[XO]], %[[XG]] exchange(symmetric)
// CHECK: arith.subf
// CHECK: md_dist.yield
// CHECK: %[[U:.*]] = md_dist.all_reduce %[[P]] : !md_dist.partial<f64> -> f64
// CHECK: md.return %[[U]] : f64
md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) truncation(shift) {
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

// A many-body term: a density gathered over pairs, a map of it, a field
// argument, a sum over particles, and a pair sum that reads the map at
// both ends.
// CHECK-LABEL: md.potential @many_body
// NAMED-LABEL: md.potential @many_body
// NAMED-NOT: md_dist
// CHECK: %[[L:.*]] = md_dist.partition
// CHECK: %[[XO:.*]] = md_dist.distribute %arg0, %[[L]] image(%arg1)
// CHECK: %[[H:.*]] = md_dist.halo
// CHECK: %[[XG:.*]] = md_dist.halo_copy %[[XO]], %[[H]] image(%arg1)
// The field argument's owned rows, without images.
// CHECK: %[[Q:.*]] = md_dist.distribute %arg2, %[[L]] : !md.field<@atoms, f64> -> !md_dist.field<@many_body.domains, owned, f64>
// CHECK: %[[N:.*]] = md_dist.neighborhood
// Directed: the density is complete at the owners.
// DIRECTED: %[[RHO:.*]] = md_dist.gather_pairs %[[N]], %[[XO]], %[[XG]] exchange(symmetric)
// DIRECTED: } : {{.*}} -> !md_dist.field<@many_body.domains, owned, f64>
// Unique: each pair once, the contributions of the ghosts summed at their
// owners.
// UNIQUE: %[[G:.*]]:2 = md_dist.gather_pairs %[[N]], %[[XO]], %[[XG]] exchange(symmetric)
// UNIQUE: } : {{.*}} -> !md_dist.field<@many_body.domains, owned, partial, f64>, !md_dist.field<@many_body.domains, ghost, partial, f64>
// UNIQUE: %[[RHO:.*]] = md_dist.halo_sum %[[G]]#0, %[[G]]#1, %[[H]]
// CHECK: %[[PSI:.*]] = md_dist.map_particles gather(%[[RHO]] : !md_dist.field<@many_body.domains, owned, f64>)
// The map is read at neighbors: its copy to the ghosts, of this version.
// CHECK: %[[PG:.*]] = md_dist.halo_copy %[[PSI]], %[[H]] :
// CHECK: %[[S:.*]] = md_dist.sum_particles gather(%[[PSI]], %[[Q]] :
// CHECK: %[[E1:.*]] = md_dist.all_reduce %[[S]]
// CHECK: md_dist.sum_pairs %[[N]], %[[XO]], %[[XG]] gather(%[[PSI]], %[[PG]] :
// CHECK: md_dist.all_reduce
// CHECK: arith.addf %[[E1]],
md.potential @many_body(%x: !vec, %cell: !md.cell, %q: !sca) -> f64 {
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
  %e1 = md.sum_particles gather(%psi, %q : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell gather(%psi : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %pi: f64, %pj: f64):
    %p = arith.mulf %pi, %pj : f64
    %k = arith.divf %p, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
  md.return %u : f64
}
