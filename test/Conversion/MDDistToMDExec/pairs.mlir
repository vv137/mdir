// The executor of md_dist through md_exec (D290,
// docs/distributed-ml2.md, Section 7.2): for each of the two domains, the
// local rows (owned, then ghost) as a field of a particle set of its own,
// the neighbor structure with a center extent and explicit images, and the
// loops of md_exec over it; the transfers stay those of the reference
// executor.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-dist-to-md-exec | FileCheck %s

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// CHECK-DAG: md.particle_set @dd.d0.owned
// CHECK-DAG: md.particle_set @dd.d0.local
// CHECK-DAG: md.particle_set @dd.d1.owned
// CHECK-DAG: md.particle_set @dd.d1.local
// CHECK-NOT: md_dist.decomposition

// CHECK-LABEL: func.func @f
// The layout and the halo: calls of the transport.
// CHECK: call @mdrtDistPartition
// CHECK: call @mdrtDistHalo(
// CHECK: call @mdrtDistHaloCopy
// The directed relation: a structure for each domain.
// CHECK: %[[X0:.*]] = mdrt.from_buffer %{{.*}} : memref<?x3xf64> to !md.field<@dd.d0.local, 3 x f64>
// CHECK: %[[C0:.*]] = md_exec.build_cells %[[X0]]
// CHECK: %[[N0:.*]] = md_exec.build_neighbors %[[C0]], %[[X0]], %{{.*}} centers(%[[O0:.*]]) cutoff(2.000000e+00) skin(0.000000e+00) kind(matrix) width(64) images(explicit)
// CHECK: md_exec.build_neighbors {{.*}} -> !mdrt.neighbors<@dd.d1.local>
// The energy: a sum of weight 1/2 over each domain.
// CHECK: md_exec.pair_for %[[N0]], %[[X0]], %{{.*}} centers(%[[O0]]) reduce(%{{.*}} : f64) cutoff(2.000000e+00) weights [5.000000e-01] exchange [symmetric] images(explicit) policy(directed, owner_only)
// CHECK: md_exec.pair_for {{.*}}policy(directed, owner_only)
// CHECK: call @mdrtDistReleaseHalo
func.func @f(%x: !vec, %l: f64) -> f64 {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  %up = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %r : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  %u = md_dist.all_reduce %up : !md_dist.partial<f64> -> f64
  return %u : f64
}

// The unique relation: the identities of the local rows as a field of i64,
// the kernel's value where the center's is the lesser, and the two partial
// results from the prefix and the tail of one destination of the local rows,
// written at the neighbors, which G^T completes.
//
// CHECK-LABEL: func.func @g
// CHECK: %[[I0:.*]] = mdrt.from_buffer %{{.*}} : memref<?xi64> to !md.field<@dd.d0.local, i64>
// CHECK: %[[F0:.*]] = md_exec.pair_for {{.*}} ins(%[[I0]] : !md.field<@dd.d0.local, i64>) outs(%{{.*}} : !md.field<@dd.d0.local, 3 x f64>) cutoff(2.000000e+00) exchange [antisymmetric] images(explicit) policy(directed, atomic)
// CHECK: ^bb0(%{{.*}}: f64, %[[D:.*]]: vector<3xf64>, %[[IC:.*]]: i64, %[[IK:.*]]: i64):
// CHECK: %[[LESSER:.*]] = arith.cmpi slt, %[[IC]], %[[IK]] : i64
// CHECK: %[[V:.*]] = arith.select %[[LESSER]], %[[D]], %{{.*}} : vector<3xf64>
// CHECK: md_exec.yield %[[V]] : vector<3xf64>
// CHECK: mdrt.to_buffer %[[F0]]
// CHECK: mdrt.to_buffer %[[F0]]
// CHECK: call @mdrtDistHaloSum
func.func @g(%x: !vec, %l: f64) -> !vec {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho
      -> !md_dist.field<@dd, owned, partial, 3 x f64>,
         !md_dist.field<@dd, ghost, partial, 3 x f64>
  %f = md_dist.halo_sum %fo, %fg, %h
         : !md_dist.field<@dd, owned, partial, 3 x f64>,
           !md_dist.field<@dd, ghost, partial, 3 x f64>, !halo -> !own
  %fc = md_dist.collect %f, %lay : !own -> !vec
  return %fc : !vec
}

// The two-ended gather over a directed relation: the kernel halved, both
// ends written.
//
// CHECK-LABEL: func.func @h
// CHECK: md_exec.pair_for {{.*}} exchange [antisymmetric] images(explicit) policy(directed, atomic)
// CHECK: %[[HALF:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: %[[B:.*]] = vector.broadcast %[[HALF]] : f64 to vector<3xf64>
// CHECK: %[[S:.*]] = arith.mulf %{{.*}}, %[[B]] : vector<3xf64>
// CHECK: md_exec.yield %[[S]] : vector<3xf64>
func.func @h(%x: !vec, %l: f64) -> !vec {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho
      -> !md_dist.field<@dd, owned, partial, 3 x f64>,
         !md_dist.field<@dd, ghost, partial, 3 x f64>
  %f = md_dist.halo_sum %fo, %fg, %h
         : !md_dist.field<@dd, owned, partial, 3 x f64>,
           !md_dist.field<@dd, ghost, partial, 3 x f64>, !halo -> !own
  %fc = md_dist.collect %f, %lay : !own -> !vec
  return %fc : !vec
}
