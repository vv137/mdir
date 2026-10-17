// What md-differentiate does not differentiate through fields on domains
// computed from the positions, refused with the reason (D279,
// docs/distributed-ml2.md, Sections 15.2 and 15.4).
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=unique" --md-differentiate -split-input-file -verify-diagnostics

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A pair sum that reads a field computed from the positions, on the unique
// traversal: the derivative at the second particle has no exchange
// contract.
md.potential @many_body(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  // expected-error @+1 {{cannot differentiate on the unique traversal a sum over pairs that reads a field computed from the positions}}
  %u = md.sum_relation %n, %x, %cell gather(%rho : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

func.func @f(%x: !vec, %cell: !md.cell) -> !vec {
  %f = md.evaluate @many_body(%x, %cell) request [forces]
      : (!vec, !md.cell) -> !vec
  return %f : !vec
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A density whose kernel uses the displacement.
md.potential @anisotropic(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  // expected-error @+1 {{cannot differentiate a field gathered on domains from a kernel that uses the displacement}}
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %dz = vector.extract %d[2] : f64 from vector<3xf64>
    %e = arith.mulf %dz, %dz : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md.yield %s : f64
  } : f64
  md.return %u : f64
}

func.func @f(%x: !vec, %cell: !md.cell) -> !vec {
  %f = md.evaluate @anisotropic(%x, %cell) request [forces]
      : (!vec, !md.cell) -> !vec
  return %f : !vec
}

// -----

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A second density that reads the first at both particles.
md.potential @nested(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  // expected-error @+1 {{cannot differentiate a field gathered on domains from a kernel that reads a field computed from the positions}}
  %sigma = md.gather_relation %n, %x, %cell gather(%rho : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%sigma : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md.yield %s : f64
  } : f64
  md.return %u : f64
}

func.func @f(%x: !vec, %cell: !md.cell) -> !vec {
  %f = md.evaluate @nested(%x, %cell) request [forces]
      : (!vec, !md.cell) -> !vec
  return %f : !vec
}
