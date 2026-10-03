// RUN: not mdir-opt %s 2>&1 | FileCheck %s
// CHECK: bond dispatch requires completed topology
module {
  func.func private @pair(memref<?xi32>, memref<?x?xi32>, memref<?x3xf64>, memref<?x3xf64>, memref<10xf64>)
  func.func private @bond(memref<?x2xi32>, memref<?x2xf64>, memref<?x3xf64>, memref<?x6xf64>, memref<?x10xf64>)
  md_dist.reference_plan [2, 2, 2] {
  ^bb0(%layout: !mdrt.layout<@atoms>, %map: !mdrt.transfer_map<@atoms>, %topology: !mdrt.transfer_map<@atoms>):
    %te = md_dist.topology_start %layout via %topology : !mdrt.layout<@atoms>, !mdrt.transfer_map<@atoms> -> !mdrt.event
    %event = md_dist.halo_start %layout via %map : !mdrt.layout<@atoms>, !mdrt.transfer_map<@atoms> -> !mdrt.event
    md_dist.dispatch @pair "interior"
    md_dist.halo_wait %event : !mdrt.event
    md_dist.dispatch @pair "boundary"
    md_dist.bond_dispatch @bond
    md_dist.topology_wait %te : !mdrt.event
    %re = md_dist.reverse_start %layout via %topology : !mdrt.layout<@atoms>, !mdrt.transfer_map<@atoms> -> !mdrt.event
    md_dist.reverse_wait %re : !mdrt.event
  }
}
