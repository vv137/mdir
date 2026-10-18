// RUN: mdir-opt %s --md-differentiate -split-input-file -verify-diagnostics

// What md-differentiate refuses around neural stages (D[ml4-neural-op]),
// each with its reason, so that a derivative it cannot take is never a
// zero.

!vec = !md.field<@atoms, 3 x f64>
!h = !md.field<@atoms, 2 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

md.stage @edge forward(@edge.f) vjp(@edge.b)
    : (f64, vector<2xf64>) -> vector<2xf64>
func.func @edge.f(%r: f64, %h: vector<2xf64>) -> vector<2xf64> {
  %b = vector.broadcast %r : f64 to vector<2xf64>
  %m = arith.mulf %b, %h : vector<2xf64>
  return %m : vector<2xf64>
}
func.func @edge.b(%r: f64, %h: vector<2xf64>, %y: vector<2xf64>)
    -> (f64, vector<2xf64>) {
  %p = arith.mulf %y, %h : vector<2xf64>
  %rb = vector.reduction <add>, %p : vector<2xf64> into f64
  %b = vector.broadcast %r : f64 to vector<2xf64>
  %hb = arith.mulf %b, %y : vector<2xf64>
  return %rb, %hb : f64, vector<2xf64>
}

// Arithmetic around the stage: a reverse rule of arith would be needed, and
// the envelope belongs to the body.
md.potential @scaled(%x: !vec, %cell: !md.cell, %h: !h) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  // expected-error@+1 {{cannot differentiate a kernel that applies a stage unless it yields the stage's value, with arguments of the kernel or constants as its operands: a cutoff function or an envelope belongs to the body of the stage}}
  %a = md.gather_relation %n, %x, %cell gather(%h : !h) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<2xf64>, %hj: vector<2xf64>):
    %m = md.apply_stage @edge(%r, %hj) : (f64, vector<2xf64>) -> vector<2xf64>
    %e = vector.extract %d[0] : f64 from vector<3xf64>
    %b = vector.broadcast %e : f64 to vector<2xf64>
    %s = arith.mulf %m, %b : vector<2xf64>
    md.yield %s : vector<2xf64>
  } : !pairs, !vec -> !h
  %u = md.sum_particles gather(%a : !h) {
  ^bb0(%ai: vector<2xf64>):
    %e = vector.reduction <add>, %ai : vector<2xf64> into f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

md.function @forces(%x: !vec, %cell: !md.cell, %h: !h) -> !vec {
  %f = md.evaluate @scaled(%x, %cell, %h) request [forces]
      : (!vec, !md.cell, !h) -> !vec
  md.return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!h = !md.field<@atoms, 2 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

md.stage @edge forward(@edge.f) vjp(@edge.b)
    : (f64, vector<2xf64>) -> vector<2xf64>
func.func @edge.f(%r: f64, %h: vector<2xf64>) -> vector<2xf64> {
  %b = vector.broadcast %r : f64 to vector<2xf64>
  %m = arith.mulf %b, %h : vector<2xf64>
  return %m : vector<2xf64>
}
func.func @edge.b(%r: f64, %h: vector<2xf64>, %y: vector<2xf64>)
    -> (f64, vector<2xf64>) {
  %p = arith.mulf %y, %h : vector<2xf64>
  %rb = vector.reduction <add>, %p : vector<2xf64> into f64
  %b = vector.broadcast %r : f64 to vector<2xf64>
  %hb = arith.mulf %b, %y : vector<2xf64>
  return %rb, %hb : f64, vector<2xf64>
}

// A field of features read by a kernel that does not yield a stage's value.
md.potential @summed(%x: !vec, %cell: !md.cell, %h: !h) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  %a = md.gather_relation %n, %x, %cell gather(%h : !h) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<2xf64>, %hj: vector<2xf64>):
    %m = md.apply_stage @edge(%r, %hj) : (f64, vector<2xf64>) -> vector<2xf64>
    md.yield %m : vector<2xf64>
  } : !pairs, !vec -> !h
  // expected-error@+1 {{cannot differentiate a field of 2 components that depends on the positions through a kernel that does not yield the value of a stage (md.apply_stage)}}
  %u = md.sum_particles gather(%a : !h) {
  ^bb0(%ai: vector<2xf64>):
    %e = vector.reduction <add>, %ai : vector<2xf64> into f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

md.function @forces(%x: !vec, %cell: !md.cell, %h: !h) -> !vec {
  %f = md.evaluate @summed(%x, %cell, %h) request [forces]
      : (!vec, !md.cell, !h) -> !vec
  md.return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!h = !md.field<@atoms, 2 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

// A stage that binds no product.
md.stage @edge forward(@edge.f) : (f64, vector<2xf64>) -> vector<2xf64>
md.stage @readout forward(@readout.f) : (vector<2xf64>) -> f64
func.func @edge.f(%r: f64, %h: vector<2xf64>) -> vector<2xf64> {
  %b = vector.broadcast %r : f64 to vector<2xf64>
  %m = arith.mulf %b, %h : vector<2xf64>
  return %m : vector<2xf64>
}
func.func @readout.f(%h: vector<2xf64>) -> f64 {
  %e = vector.reduction <add>, %h : vector<2xf64> into f64
  return %e : f64
}

md.potential @unbound(%x: !vec, %cell: !md.cell, %h: !h) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  %a = md.gather_relation %n, %x, %cell gather(%h : !h) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<2xf64>, %hj: vector<2xf64>):
    %m = md.apply_stage @edge(%r, %hj) : (f64, vector<2xf64>) -> vector<2xf64>
    md.yield %m : vector<2xf64>
  } : !pairs, !vec -> !h
  %u = md.sum_particles gather(%a : !h) {
  ^bb0(%ai: vector<2xf64>):
    // expected-error@+1 {{'md.apply_stage' op cannot be differentiated: the stage 'readout' binds no vector-Jacobian product}}
    %e = md.apply_stage @readout(%ai) : (vector<2xf64>) -> f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

md.function @forces(%x: !vec, %cell: !md.cell, %h: !h) -> !vec {
  %f = md.evaluate @unbound(%x, %cell, %h) request [forces]
      : (!vec, !md.cell, !h) -> !vec
  md.return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

md.stage @pair forward(@pair.f) vjp(@pair.b) : (f64, f64) -> f64
func.func @pair.f(%r: f64, %e: f64) -> f64 {
  %m = arith.mulf %r, %e : f64
  return %m : f64
}
func.func @pair.b(%r: f64, %e: f64, %y: f64) -> (f64, f64) {
  %rb = arith.mulf %y, %e : f64
  %eb = arith.mulf %y, %r : f64
  return %rb, %eb : f64, f64
}

// A derivative in a number through a stage: no forward rule is bound.
md.potential @parameter(%x: !vec, %cell: !md.cell, %e: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric, asserted) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    // expected-error@+1 {{has no forward derivative: a stage binds its vector-Jacobian product only}}
    %k = md.apply_stage @pair(%r, %e) : (f64, f64) -> f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

md.function @derivative(%x: !vec, %cell: !md.cell, %e: f64) -> f64 {
  %g = md.evaluate @parameter(%x, %cell, %e) request [derivative(2)]
      : (!vec, !md.cell, f64) -> f64
  md.return %g : f64
}
