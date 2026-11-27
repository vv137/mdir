// RUN: mdir-opt %s --convert-md-dist-to-loops="mpi=true" | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!layout = !md_dist.layout<@dd>
!group = !md.relation<@atoms, 3, ordered, @water>
module attributes {md_dist.driver} {
  md.particle_set @atoms
  md.tuple_set @water on(@atoms) arity(3) orientation(ordered) disjoint
  md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

  // CHECK-LABEL: func.func @solve
  // CHECK: call @mdrtDistUseIdentityGroups(%[[LAYOUT:[0-9]+]])
  // CHECK: call @mdrtDistGroupOwners(%[[LAYOUT]],
  // CHECK: scf.for
  func.func @solve(%x: !vec, %v: !vec, %f: !vec, %g: !group, %edge: f64, %n: index) -> !vec {
    %cell = md.orthorhombic_cell %edge, %edge, %edge
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %r:3 = scf.for %i = %zero to %n step %one
        iter_args(%p = %x, %u = %v, %a = %f) -> (!vec, !vec, !vec) {
      %l = md_dist.partition %p, %cell groups(%g : !group) {md_dist.constraint_layout} : !vec -> !layout
      %o = md_dist.distribute %p, %l image(%cell) : !vec -> !own
      %m = md_dist.migrate %l, %o, %cell : !layout, !own -> !layout
      %q = md_dist.redistribute %o, %l, %m image(%cell) : !own, !layout, !layout -> !own
      %global = md_dist.collect %q, %m : !own -> !vec
      scf.yield %global, %u, %a : !vec, !vec, !vec
    }
    return %r#0 : !vec
  }

  // CHECK-LABEL: func.func @spatial
  // CHECK-NOT: call @mdrtDistUseIdentityGroups
  // CHECK: call @mdrtDistGroupOwners
  // CHECK-NOT: call @mdrtDistUseIdentityGroups
  // CHECK: return
  func.func @spatial(%x: !vec, %g: !group, %edge: f64) {
    %cell = md.orthorhombic_cell %edge, %edge, %edge
    %l = md_dist.partition %x, %cell groups(%g : !group) : !vec -> !layout
    %o = md_dist.distribute %x, %l image(%cell) : !vec -> !own
    %next = md_dist.migrate %l, %o, %cell : !layout, !own -> !layout
    return
  }
}
