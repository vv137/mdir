// --md-check-transposes (D295): every op of the structure of a
// potential declares its transpose through AdjointOpInterface.
//
// The potentials of the derivative's tests pass: the sums and gathers over
// relations and tuples, the sums and maps over particles, and the stages.
// RUN: mdir-opt %s --split-input-file --md-check-transposes -verify-diagnostics -o /dev/null
// RUN: mdir-opt %S/differentiate.mlir --split-input-file --md-check-transposes -o /dev/null
// RUN: mdir-opt %S/differentiate-eam.mlir --split-input-file --md-check-transposes -o /dev/null
// RUN: mdir-opt %S/differentiate-tuples.mlir --split-input-file --md-check-transposes -o /dev/null
// RUN: mdir-opt %S/differentiate-fields.mlir --split-input-file --md-check-transposes -o /dev/null
// RUN: mdir-opt %S/differentiate-stages.mlir --split-input-file --md-check-transposes -o /dev/null
// RUN: mdir-opt %S/differentiate-parameters.mlir --split-input-file --md-check-transposes -o /dev/null

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!bonds = !md.relation<@atoms, 2, unordered, @bonds>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// Each op of md with a kernel that md-differentiate carries an adjoint
// through declares the transpose of its structure.
md.potential @declared(%x: !vec, %cell: !md.cell, %bonds: !bonds,
                       %q: !real) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !pairs, !vec -> !real
  %m = md.map_particles gather(%rho, %q : !real, !real) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : !real
  %e1 = md.sum_particles gather(%m : !real) {
  ^bb0(%a: f64):
    md.yield %a : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !pairs, !vec -> f64
  %e3 = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !bonds, !vec -> f64
  %s = arith.addf %e1, %e2 : f64
  %e = arith.addf %s, %e3 : f64
  md.return %e : f64
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!bonds = !md.relation<@atoms, 2, unordered, @bonds>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// A gather over tuples has no declared transpose yet: no derivative
// carries an adjoint through a field that one computes.
md.potential @undeclared(%x: !vec, %cell: !md.cell, %bonds: !bonds) -> f64 {
  // expected-error@+1 {{no declared transpose for 'md.gather_tuples' in potential 'undeclared'}}
  %f = md.gather_tuples %bonds, %x, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    %m = arith.negf %d : vector<3xf64>
    md.yield %d, %m : vector<3xf64>, vector<3xf64>
  } : !bonds, !vec -> !vec
  %e = md.sum_particles gather(%f : !vec) {
  ^bb0(%a: vector<3xf64>):
    %s = vector.reduction <add>, %a : vector<3xf64> into f64
    md.yield %s : f64
  } : f64
  md.return %e : f64
}
