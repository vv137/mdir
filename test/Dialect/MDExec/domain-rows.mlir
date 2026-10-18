// The rows of a domain in md_exec (D[ml2-exec], docs/distributed-ml2.md,
// Section 7.2): a center extent, explicit images, and the policy
// (directed, atomic), in the value and the storage form.
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s

md.particle_set @local

// CHECK-LABEL: md.function @value
// CHECK: md_exec.build_neighbors %{{.*}}, %{{.*}}, %{{.*}} centers(%{{.*}}) cutoff(2.000000e+00) skin(0.000000e+00) kind(matrix) width(64) images(explicit)
// CHECK: md_exec.pair_for %{{.*}}, %{{.*}}, %{{.*}} centers(%{{.*}}) outs(
// CHECK-SAME: exchange [antisymmetric] images(explicit) policy(directed, atomic)
// CHECK: md_exec.pair_for %{{.*}}, %{{.*}}, %{{.*}} centers(%{{.*}}) reduce(
// CHECK-SAME: images(explicit) policy(directed, owner_only)
md.function @value(%x: !md.field<@local, 3 x f64>, %cell: !md.cell,
                   %n: index) -> (!md.field<@local, 3 x f64>, f64) {
  %cells = md_exec.build_cells %x, %cell width(0.67)
             : !md.field<@local, 3 x f64> -> !mdrt.cells<@local>
  %nl = md_exec.build_neighbors %cells, %x, %cell centers(%n)
          cutoff(2.0) skin(0.0) kind(matrix) width(64) images(explicit)
          : !mdrt.cells<@local>, !md.field<@local, 3 x f64>
          -> !mdrt.neighbors<@local>
  %f0 = md_exec.zeros : !md.field<@local, 3 x f64>
  %f = md_exec.pair_for %nl, %x, %cell centers(%n)
         outs(%f0 : !md.field<@local, 3 x f64>) cutoff(2.0)
         exchange [antisymmetric] images(explicit)
         policy(directed, atomic) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@local>, !md.field<@local, 3 x f64>
      -> !md.field<@local, 3 x f64>
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell centers(%n) reduce(%u0 : f64)
         cutoff(2.0) weights [0.5] images(explicit)
         policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@local>, !md.field<@local, 3 x f64> -> f64
  md.return %f, %u : !md.field<@local, 3 x f64>, f64
}

// CHECK-LABEL: func.func @storage
// CHECK: md_exec.refresh_neighbors %{{.*}}, %{{.*}}, %{{.*}} centers(%{{.*}}) cutoff(2.000000e+00) skin(0.000000e+00) cell_width(6.700000e-01) images(explicit) policy(always)
func.func @storage(%x: memref<?x3xf64>, %cell: !md.cell, %n: index,
                   %size: index) {
  %nl = md_exec.empty_neighbors size(%size) positions(memref<?x3xf64>)
          kind(matrix) width(64) : !mdrt.neighbors<@local>
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell centers(%n)
           cutoff(2.0) skin(0.0) cell_width(0.67) images(explicit)
           policy(always) : !mdrt.neighbors<@local>, memref<?x3xf64>
  return
}
