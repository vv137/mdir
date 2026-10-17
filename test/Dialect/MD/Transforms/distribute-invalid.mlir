// md-distribute refuses what it cannot distribute, with the reason
// (D276).
// RUN: mdir-opt %s --md-distribute="grid=2,1,1" -split-input-file -verify-diagnostics
// RUN: not mdir-opt %s --md-distribute="grid=2,1,1 traversal=unique" -split-input-file 2>&1 | FileCheck %s --check-prefix=UNIQUE

// The options.
// RUN: not mdir-opt %s --md-distribute="grid=2,1" -split-input-file 2>&1 | FileCheck %s --check-prefix=GRID
// GRID: md-distribute needs a grid of three positive numbers of domains, 'grid=gx,gy,gz'
// RUN: not mdir-opt %s --md-distribute="grid=2,1,1 traversal=half" -split-input-file 2>&1 | FileCheck %s --check-prefix=TRAVERSAL
// TRAVERSAL: md-distribute: the traversal is 'directed' or 'unique', not 'half'
// RUN: not mdir-opt %s --md-distribute="grid=2,1,1 potentials=nothing" -split-input-file 2>&1 | FileCheck %s --check-prefix=NAME
// NAME: md-distribute: no potential is named 'nothing'
// RUN: not mdir-opt %s --md-distribute="grid=2,1,1 reach=1.5" -split-input-file 2>&1 | FileCheck %s --check-prefix=REACH
// REACH: error: 'md.potential' op the reach of the halo, 1.500000e+00, is less than the cutoff, 2.000000e+00

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
!bonds = !md.relation<@atoms, 2, unordered, @bonds>
md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// Exclusions are pairs of a tuple set: the topology across domains.
md.potential @excluded(%x: !vec, %cell: !md.cell, %bonds: !bonds) -> f64 {
  // expected-error @+1 {{cannot be distributed: exclusions are pairs of a tuple set}}
  %n = md.neighborhood %x, %cell cutoff(2.0) exclude(%bonds : !bonds)
         : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!bonds = !md.relation<@atoms, 2, unordered, @bonds>
md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// A sum over tuples.
md.potential @bonded(%x: !vec, %cell: !md.cell, %bonds: !bonds) -> f64 {
  // expected-error @+1 {{cannot be distributed: a sum over tuples needs the topology across domains}}
  %u = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !bonds, !vec -> f64
  md.return %u : f64
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A term of the absolute positions: the owned positions are wrapped into
// the cell.
md.potential @external(%x: !vec, %cell: !md.cell) -> f64 {
  // expected-error @+1 {{cannot be distributed: it reads the positions as a field}}
  %u = md.sum_particles gather(%x : !vec) {
  ^bb0(%p: vector<3xf64>):
    %s = vector.reduction <add>, %p : vector<3xf64> into f64
    md.yield %s : f64
  } : f64
  md.return %u : f64
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A contract that rests on an assertion: md_dist proves every contract.
md.potential @asserted(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  // expected-error @+1 {{cannot be distributed: md_dist proves every exchange contract (V8), and the kernel cannot be shown to be symmetric}}
  %u = md.sum_relation %n, %x, %cell exchange(symmetric, asserted) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %x0 = vector.extract %d[0] : f64 from vector<3xf64>
    md.yield %x0 : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A cell computed in the potential, not its argument.
md.potential @other_cell(%x: !vec, %cell: !md.cell, %l: f64) -> f64 {
  // expected-error @+1 {{cannot be distributed by md-distribute}}
  %c = md.orthorhombic_cell %l, %l, %l
  %n = md.neighborhood %x, %c cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %c exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// On the unique traversal a gather needs the value at the second particle
// of each pair from an exchange contract. The directed traversal takes
// any kernel.
md.potential @no_contract(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  // UNIQUE: error: 'md.gather_relation' op cannot be distributed on the unique traversal: each pair is computed once, and the value at its second particle needs an exchange contract
  %a = md.gather_relation %n, %x, %cell exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %x0 = vector.extract %d[0] : f64 from vector<3xf64>
    md.yield %x0 : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%a : !sca) {
  ^bb0(%v: f64):
    md.yield %v : f64
  } : f64
  md.return %u : f64
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// The name of the decomposition is taken.
// expected-error @+1 {{the name 'lj.domains' is needed for the decomposition of the potential, but is taken}}
md.function @lj.domains() {
  md.return
}
md.potential @lj(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}
