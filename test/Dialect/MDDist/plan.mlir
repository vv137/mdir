// RUN: mdir-opt %s | mdir-opt | FileCheck %s
module {
  func.func private @pair(memref<?xi32>, memref<?x?xi32>, memref<?x3xf64>, memref<?x3xf64>, memref<10xf64>)
  // CHECK: md_dist.reference_plan [2, 2, 2]
  md_dist.reference_plan [2, 2, 2] {
    // CHECK: %[[E:.*]] = md_dist.halo_start %{{.*}} via %{{.*}} : !mdrt.layout<@atoms>, !mdrt.transfer_map<@atoms> -> !mdrt.event
  ^bb0(%layout: !mdrt.layout<@atoms>, %map: !mdrt.transfer_map<@atoms>):
    %event = md_dist.halo_start %layout via %map : !mdrt.layout<@atoms>, !mdrt.transfer_map<@atoms> -> !mdrt.event
    // CHECK: md_dist.dispatch @pair "interior"
    md_dist.dispatch @pair "interior"
    // CHECK: md_dist.halo_wait %[[E]] : !mdrt.event
    md_dist.halo_wait %event : !mdrt.event
    // CHECK: md_dist.dispatch @pair "boundary"
    md_dist.dispatch @pair "boundary"
  }
}
