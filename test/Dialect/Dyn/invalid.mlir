// RUN: mdir-opt %s -split-input-file -verify-diagnostics

dyn.program @p(%v: !md.field<@atoms, f64>, %m: !md.field<@atoms, f64>,
               %dt: f64) -> !md.field<@atoms, f64> {
  // expected-error@+1 {{expected a field with 3 components of f64, got '!md.field<@atoms, f64>'}}
  %v1 = dyn.kick %v, %v, %m, %dt : !md.field<@atoms, f64>
  dyn.return %v1 : !md.field<@atoms, f64>
}

// -----

// The masses must be on the particle set of the velocities.
// expected-note@+1 {{prior use here}}
dyn.program @p(%v: !md.field<@atoms, 3 x f64>, %m: !md.field<@ions, f64>,
               %dt: f64) -> !md.field<@atoms, 3 x f64> {
  // expected-error@+1 {{use of value '%m' expects different type than prior uses: '!md.field<@atoms, f64>' vs '!md.field<@ions, f64>'}}
  %v1 = dyn.kick %v, %v, %m, %dt : !md.field<@atoms, 3 x f64>
  dyn.return %v1 : !md.field<@atoms, 3 x f64>
}

// -----

// expected-error@+2 {{unknown name 'energy_conservation' in 'provides'}}
// expected-note@+1 {{known names: symplectic, time_reversible, thermostatting, barostatting}}
dyn.program @p(%dt: f64) -> f64
    attributes {provides = ["energy_conservation"]} {
  dyn.return %dt : f64
}

// -----

// expected-error@+2 {{unknown name 'volume' in 'requires'}}
// expected-note@+1 {{known names: temperature, pressure}}
dyn.program @p(%dt: f64) -> f64 attributes {requires = ["volume"]} {
  dyn.return %dt : f64
}

// -----

dyn.program @p(%dt: f64) -> (f64, f64) {
  // expected-error@+1 {{returns 1 values, but the enclosing program has 2 results}}
  dyn.return %dt : f64
}

// -----

func.func @f(%dt: f64) -> f64 {
  // expected-error@+1 {{'q' does not name a program}}
  %r = dyn.step @q(%dt) : (f64) -> f64
  return %r : f64
}

// -----

dyn.program @p(%dt: f64) -> f64 {
  dyn.return %dt : f64
}

func.func @f(%n: index) -> f64 {
  // expected-error@+1 {{expected operand 0 to have type 'f64', got 'index'}}
  %r = dyn.step @p(%n) : (index) -> f64
  return %r : f64
}
