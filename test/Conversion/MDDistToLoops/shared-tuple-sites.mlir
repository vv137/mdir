// RUN: mdir-opt %s --convert-md-dist-to-loops="mpi=true" | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!owned = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 1.0>
!pairs = !md.relation<@atoms, 2, unordered, @pairs>

module attributes {md_dist.driver} {
  md.particle_set @atoms
  md.tuple_set @pairs on(@atoms) arity(2) orientation(unordered)
  md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

  // CHECK-LABEL: func.func @step
  // CHECK: %[[GROUP:.*]] = call @mdrtDistNewSiteRequests()
  // CHECK: call @mdrtDistRegisterTuplePlan(%[[GROUP]],
  // CHECK: call @mdrtDistRegisterTuplePlan(%[[GROUP]],
  // CHECK: scf.for
  // CHECK: call @mdrtDistTopologyShard
  // CHECK: call @mdrtDistScheduleTuples
  // CHECK: call @mdrtDistScheduleTuples
  // CHECK: call @mdrtDistFreeSiteRequests(%[[GROUP]])
  func.func @step(%x: !vec, %a: !pairs, %b: !pairs, %edge: f64, %n: index) {
    %cell = md.orthorhombic_cell %edge, %edge, %edge
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
      %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
      %o = md_dist.distribute %x, %l image(%cell) : !vec -> !owned
      %h = md_dist.halo %l, %o, %cell : !owned -> !halo
      %ra = md_dist.tuple_rows %a, %o, %h : !pairs, !owned, !halo -> memref<?x3xi64>
      %rb = md_dist.tuple_rows %b, %o, %h : !pairs, !owned, !halo -> memref<?x3xi64>
    }
    return
  }
}
