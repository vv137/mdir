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
// expected-note@+1 {{known names: symplectic, time_reversible, thermostatting, barostatting, no_global_sums}}
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

// -----

func.func @f(%v: !md.field<@atoms, f64>, %m: !md.field<@atoms, f64>)
    -> vector<3xf64> {
  // expected-error@+1 {{expected a field with 3 components of f64, got '!md.field<@atoms, f64>'}}
  %p = dyn.momentum %v, %m : !md.field<@atoms, f64>
  return %p : vector<3xf64>
}

// -----

func.func @f(%k: f64, %step: i64) -> f64 {
  // expected-error@+1 {{expected 0 < 'decay' < 1, got 1}}
  %a = dyn.velocity_rescaling %k, %step
      {seed = 1 : i64, target_kinetic = 1.0, freedom = 3.0, decay = 1.0} : f64
  return %a : f64
}

// -----

func.func @f(%k: f64, %step: i64) -> f64 {
  // expected-error@+1 {{expected 'freedom' > 0, got 0}}
  %a = dyn.velocity_rescaling %k, %step
      {seed = 1 : i64, target_kinetic = 1.0, freedom = 0.0, decay = 0.5} : f64
  return %a : f64
}

// -----

func.func @f(%k: f64, %step: i64) -> f64 {
  // expected-error@+1 {{expected 'chain_length' >= 1, got 0}}
  %a = dyn.nose_hoover %k, %step
      {temperature = 300.0, time_constant = 1.0, chain_length = 0 : i64,
       period = 0.02, freedom = 3.0} : f64
  return %a : f64
}

// -----

func.func @f(%v: !md.field<@atoms, 3 x f64>, %m: !md.field<@atoms, f64>,
             %step: i64, %ids: !md.field<@atoms, i64>)
    -> !md.field<@atoms, 3 x f64> {
  // expected-error@+1 {{expected the numbers of the particles as a field of i32 on @atoms, got '!md.field<@atoms, i64>'}}
  %v1 = dyn.noise %v, %m, %step, %ids
      {seed = 1 : i64, decay = 0.5, spread = 0.1}
      : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i64>
  return %v1 : !md.field<@atoms, 3 x f64>
}

// -----

// A program with the noise claims what the friction and the noise make it.
// expected-error@+1 {{has a 'dyn.noise' and does not require 'temperature'}}
dyn.program @p(%v: !md.field<@atoms, 3 x f64>, %m: !md.field<@atoms, f64>,
               %step: i64, %ids: !md.field<@atoms, i32>)
    -> !md.field<@atoms, 3 x f64>
    attributes {provides = ["thermostatting"]} {
  %v1 = dyn.noise %v, %m, %step, %ids
      {seed = 1 : i64, decay = 0.5, spread = 0.1}
      : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i32>
  dyn.return %v1 : !md.field<@atoms, 3 x f64>
}

// -----

// expected-error@+1 {{has a 'dyn.noise' and does not provide 'thermostatting'}}
dyn.program @p(%v: !md.field<@atoms, 3 x f64>, %m: !md.field<@atoms, f64>,
               %step: i64, %ids: !md.field<@atoms, i32>)
    -> !md.field<@atoms, 3 x f64>
    attributes {requires = ["temperature"]} {
  %v1 = dyn.noise %v, %m, %step, %ids
      {seed = 1 : i64, decay = 0.5, spread = 0.1}
      : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i32>
  dyn.return %v1 : !md.field<@atoms, 3 x f64>
}

// -----

// expected-error@+1 {{provides 'symplectic', which the friction and the noise of 'dyn.noise' are not}}
dyn.program @p(%v: !md.field<@atoms, 3 x f64>, %m: !md.field<@atoms, f64>,
               %step: i64, %ids: !md.field<@atoms, i32>)
    -> !md.field<@atoms, 3 x f64>
    attributes {provides = ["symplectic", "thermostatting"],
                requires = ["temperature"]} {
  // expected-note@+1 {{the noise}}
  %v1 = dyn.noise %v, %m, %step, %ids
      {seed = 1 : i64, decay = 0.5, spread = 0.1}
      : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i32>
  dyn.return %v1 : !md.field<@atoms, 3 x f64>
}

// -----

// expected-error@+1 {{provides 'no_global_sums' and computes a global sum}}
dyn.program @p(%v: !md.field<@atoms, 3 x f64>, %m: !md.field<@atoms, f64>)
    -> f64 attributes {provides = ["no_global_sums"]} {
  // expected-note@+1 {{the sum}}
  %k = dyn.kinetic %v, %m : !md.field<@atoms, 3 x f64> -> f64
  dyn.return %k : f64
}

// -----

// expected-error@+1 {{provides 'no_global_sums' and computes a global sum}}
dyn.program @p(%v: !md.field<@atoms, 3 x f64>) -> vector<3xf64>
    attributes {provides = ["no_global_sums"]} {
  // expected-note@+1 {{the sum}}
  %s = md.sum_particles gather(%v : !md.field<@atoms, 3 x f64>) {
  ^bb0(%vi: vector<3xf64>):
    md.yield %vi : vector<3xf64>
  } : vector<3xf64>
  dyn.return %s : vector<3xf64>
}

// -----

func.func private @mdrtDeferStops()

// expected-error@+1 {{states its effects, and an op of its body has unknown effects}}
dyn.program @p(%dt: f64) -> f64 attributes {effects = {}} {
  // expected-note@+1 {{the op; a call has the effects that its function declares in 'dyn.effects'}}
  func.call @mdrtDeferStops() : () -> ()
  dyn.return %dt : f64
}

// -----

func.func private @mdrtDeferStops()
    attributes {dyn.effects = {write = ["dyn.stop"]}}

// expected-error@+1 {{does not state the effect of its body: write of 'dyn.stop'}}
dyn.program @p(%dt: f64) -> f64 attributes {effects = {read = ["dyn.stop"]}} {
  func.call @mdrtDeferStops() : () -> ()
  dyn.return %dt : f64
}

// -----

// expected-error@+1 {{in 'effects': unknown resource in 'write'; known: dyn.bath, dyn.chain, dyn.stop, dyn.record, dyn.statistics, <Default>}}
dyn.program @p(%dt: f64) -> f64 attributes {effects = {write = ["cell"]}} {
  dyn.return %dt : f64
}

// -----

// expected-error@+1 {{in 'effects': unknown kind of effect 'allocate'; expected 'read' or 'write'}}
dyn.program @p(%dt: f64) -> f64 attributes {effects = {allocate = []}} {
  dyn.return %dt : f64
}

// -----

func.func @f(%p: vector<3xf64>, %l: f64, %step: i64) -> vector<3xf64> {
  // expected-error@+1 {{expected 2 numbers in 'compressibility' for the coupling 'semi_isotropic', got 1}}
  %de = dyn.cell_rescaling %p, %l, %l, %l, %step
      {coupling = "semi_isotropic", seed = 1 : i64, target = 1.0,
       compressibility = [4.5e-05], rate = 0.01, kT = 2.5} : vector<3xf64>
  return %de : vector<3xf64>
}

// -----

func.func @f(%p: vector<3xf64>, %l: f64, %step: i64) -> vector<3xf64> {
  // expected-error@+1 {{'tension' needs the coupling 'semi_isotropic'}}
  %de = dyn.cell_rescaling %p, %l, %l, %l, %step
      {coupling = "isotropic", seed = 1 : i64, target = 1.0,
       compressibility = [4.5e-05], rate = 0.01, kT = 2.5, tension = 1.0}
      : vector<3xf64>
  return %de : vector<3xf64>
}

// -----

func.func @f(%p: vector<3xf64>, %l: f64, %step: i64) -> vector<3xf64> {
  // expected-error@+1 {{expected a 'compressibility' that is not 0}}
  %de = dyn.cell_rescaling %p, %l, %l, %l, %step
      {coupling = "anisotropic", seed = 1 : i64, target = 1.0,
       compressibility = [0.0, 0.0, 0.0], rate = 0.01, kT = 2.5}
      : vector<3xf64>
  return %de : vector<3xf64>
}

// -----

func.func @f(%box: memref<3xf64>) -> f64 {
  // expected-error@+1 {{expected the memory of 6 values of the cell, got 'memref<3xf64>'}}
  %t:6 = dyn.load_cell %box : memref<3xf64> -> f64, f64, f64, f64, f64, f64
  return %t#0 : f64
}

// -----

func.func @f(%l: f64, %box: memref<4xf64>) {
  // expected-error@+1 {{expected the 3 edges, or with the 3 tilts 6 values, of the cell, got 4}}
  dyn.store_cell %l, %l, %l, %l into %box : memref<4xf64>
  return
}

// -----

func.func @f(%k: vector<3xf64>, %mu: vector<3xf64>, %w: vector<3xf64>,
             %c: f64, %v: f64) -> f64 {
  // expected-error@+1 {{the count 'trotter' takes log_strain, virial_before, virial_after, and 2 volumes}}
  %work = dyn.scaling_work kinetic(%k) strain(%mu) virial_before(%w)
      constant(%c) volumes(%v) {count = "trotter"} : f64
  return %work : f64
}

// -----

func.func @f(%x: !md.field<@atoms, 3 x f64>, %mu: vector<3xf64>,
             %m: !md.field<@atoms, f64>, %cell: !md.cell,
             %r: !md.relation<@ions, 3, ordered, @settles>)
    -> !md.field<@atoms, 3 x f64> {
  // expected-error@+1 {{expected the groups as relations of two or more members of @atoms, got '!md.relation<@ions, 3, ordered, @settles>'}}
  %x1 = dyn.scale_positions %x, %mu, %m, %cell
      groups(%r : !md.relation<@ions, 3, ordered, @settles>)
      : !md.field<@atoms, 3 x f64>
  return %x1 : !md.field<@atoms, 3 x f64>
}

// -----

// A scaling within the step is neither symplectic nor time-reversible.
// expected-error@+1 {{provides 'symplectic', which a scaling within the step is not}}
dyn.program @p(%x: !md.field<@atoms, 3 x f64>, %mu: vector<3xf64>,
               %m: !md.field<@atoms, f64>, %cell: !md.cell)
    -> !md.field<@atoms, 3 x f64>
    attributes {provides = ["barostatting", "symplectic"],
                requires = ["pressure"]} {
  // expected-note@+1 {{the scaling}}
  %x1 = dyn.scale_positions %x, %mu, %m, %cell : !md.field<@atoms, 3 x f64>
  dyn.return %x1 : !md.field<@atoms, 3 x f64>
}

// -----

// expected-error@+1 {{has a 'dyn.scale_positions' and does not require 'pressure'}}
dyn.program @p(%x: !md.field<@atoms, 3 x f64>, %mu: vector<3xf64>,
               %m: !md.field<@atoms, f64>, %cell: !md.cell)
    -> !md.field<@atoms, 3 x f64> attributes {provides = ["barostatting"]} {
  %x1 = dyn.scale_positions %x, %mu, %m, %cell : !md.field<@atoms, 3 x f64>
  dyn.return %x1 : !md.field<@atoms, 3 x f64>
}

// -----

func.func @f(%state: memref<3xf64>) -> vector<3xf64> {
  // expected-error@+1 {{expected the memory of the 9 values of the state of the pressure, got 'memref<3xf64>'}}
  %w, %wg, %k = dyn.load_pressure_state %state : memref<3xf64>
  return %w : vector<3xf64>
}
