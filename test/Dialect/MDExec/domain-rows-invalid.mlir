// What the verifier refuses of the rows of a domain (D290,
// docs/distributed-ml2.md, Section 7.2).
//
// RUN: mdir-opt %s -split-input-file -verify-diagnostics

md.particle_set @local

// A structure of groups takes every row as a center, in the minimum image.
md.function @f(%x: !md.field<@local, 3 x f64>, %cell: !md.cell, %n: index) {
  %cells = md_exec.build_cells %x, %cell width(0.67)
             : !md.field<@local, 3 x f64> -> !mdrt.cells<@local>
  // expected-error@+1 {{a structure with 'centers' or 'images(explicit)' is a matrix}}
  %nl = md_exec.build_neighbors %cells, %x, %cell centers(%n)
          cutoff(2.0) skin(0.0) kind(groups) width(64)
          : !mdrt.cells<@local>, !md.field<@local, 3 x f64>
          -> !mdrt.neighbors<@local>
  md.return
}

// -----

md.particle_set @local
md.particle_set @pairs

// Exclusions across domains are the topology of stage 2.
md.function @f(%x: !md.field<@local, 3 x f64>, %cell: !md.cell,
               %ex: !mdrt.incidence<@local, 2, @pairs>) {
  %cells = md_exec.build_cells %x, %cell width(0.67)
             : !md.field<@local, 3 x f64> -> !mdrt.cells<@local>
  // expected-error@+1 {{a structure with 'centers' or 'images(explicit)' leaves out no pairs}}
  %nl = md_exec.build_neighbors %cells, %x, %cell
          cutoff(2.0) skin(0.0) kind(matrix) width(64) images(explicit)
          exclude(%ex : !mdrt.incidence<@local, 2, @pairs>)
          : !mdrt.cells<@local>, !md.field<@local, 3 x f64>
          -> !mdrt.neighbors<@local>
  md.return
}

// -----

md.particle_set @local

// The rows of a domain are built at every refresh.
func.func @f(%x: memref<?x3xf64>, %cell: !md.cell, %n: index, %size: index) {
  %nl = md_exec.empty_neighbors size(%size) positions(memref<?x3xf64>)
          kind(matrix) width(64) : !mdrt.neighbors<@local>
  // expected-error@+1 {{a structure with 'centers' or 'images(explicit)' is refreshed with the policy 'always' and without 'prune_skin'}}
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell centers(%n)
           cutoff(2.0) skin(0.2) cell_width(0.67) policy(check)
           : !mdrt.neighbors<@local>, memref<?x3xf64>
  return
}

// -----

md.particle_set @local

// Each pair once over groups is not for the rows of a domain.
md.function @f(%x: !md.field<@local, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@local>, %n: index) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the policy (unique, atomic) takes a structure of groups, which has neither 'centers' nor 'images(explicit)'}}
  %u = md_exec.pair_for %nl, %x, %cell centers(%n) reduce(%u0 : f64)
         cutoff(2.0) exchange [symmetric] policy(unique, atomic) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@local>, !md.field<@local, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.particle_set @local

// The centers of a loop are those of its structure.
md.function @f(%x: !md.field<@local, 3 x f64>, %cell: !md.cell, %n: index,
               %m: index) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(0.67)
             : !md.field<@local, 3 x f64> -> !mdrt.cells<@local>
  %nl = md_exec.build_neighbors %cells, %x, %cell centers(%n)
          cutoff(2.0) skin(0.0) kind(matrix) width(64) images(explicit)
          : !mdrt.cells<@local>, !md.field<@local, 3 x f64>
          -> !mdrt.neighbors<@local>
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the centers of the loop are not those of the neighbor structure}}
  %u = md_exec.pair_for %nl, %x, %cell centers(%m) reduce(%u0 : f64)
         cutoff(2.0) images(explicit) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@local>, !md.field<@local, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.particle_set @local

// A loop takes the images that its structure holds.
md.function @f(%x: !md.field<@local, 3 x f64>, %cell: !md.cell, %n: index)
    -> f64 {
  %cells = md_exec.build_cells %x, %cell width(0.67)
             : !md.field<@local, 3 x f64> -> !mdrt.cells<@local>
  %nl = md_exec.build_neighbors %cells, %x, %cell centers(%n)
          cutoff(2.0) skin(0.0) kind(matrix) width(64) images(explicit)
          : !mdrt.cells<@local>, !md.field<@local, 3 x f64>
          -> !mdrt.neighbors<@local>
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the loop takes images(minimum), but the neighbor structure holds images(explicit)}}
  %u = md_exec.pair_for %nl, %x, %cell centers(%n) reduce(%u0 : f64)
         cutoff(2.0) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@local>, !md.field<@local, 3 x f64> -> f64
  md.return %u : f64
}
