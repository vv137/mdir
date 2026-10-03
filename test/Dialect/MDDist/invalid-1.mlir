// RUN: not mdir-opt %s 2>&1 | FileCheck %s
// CHECK: event requires exactly one wait
module {
  func.func private @pair(memref<?xi32>, memref<?x?xi32>, memref<?x3xf64>, memref<?x3xf64>, memref<10xf64>)
  md_dist.reference_plan [2, 2, 2] {
  ^bb0(%layout: !mdrt.layout<@atoms>, %map: !mdrt.transfer_map<@atoms>):
    %event = md_dist.halo_start %layout via %map : !mdrt.layout<@atoms>, !mdrt.transfer_map<@atoms> -> !mdrt.event
    md_dist.dispatch @pair "interior"
    md_dist.dispatch @pair "boundary"
  }
}
