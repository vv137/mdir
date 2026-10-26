// RUN: mdir-opt %s | mdir-opt | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.5) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(switch, from = 2.0) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// CHECK-LABEL: dyn.program @velocity_verlet(
// CHECK-SAME:    -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>)
// CHECK-SAME:    attributes {provides = ["symplectic", "time_reversible"]}
dyn.program @velocity_verlet(%x: !vec, %v: !vec, %f: !vec, %m: !real,
                             %cell: !md.cell, %dt: f64,
                             %eps: f64, %sigma: f64) -> (!vec, !vec, !vec)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %c    = arith.constant 0.5 : f64
  %half = arith.mulf %c, %dt : f64
  // CHECK: %[[V1:[0-9]+]] = dyn.kick %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %[[HALF:[0-9]+]] : !md.field<@atoms, 3 x f64>
  %v1 = dyn.kick %v, %f, %m, %half : !vec
  // CHECK: %[[X1:[0-9]+]] = dyn.drift %{{[a-z0-9]+}}, %[[V1]], %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>
  %x1 = dyn.drift %x, %v1, %dt : !vec
  // CHECK: %[[F1:[0-9]+]] = md.evaluate @lj(%[[X1]],
  %f1 = md.evaluate @lj(%x1, %cell, %eps, %sigma) request [forces]
      : (!vec, !md.cell, f64, f64) -> !vec
  // CHECK: %[[V2:[0-9]+]] = dyn.kick %[[V1]], %[[F1]], %{{[a-z0-9]+}}, %[[HALF]]
  %v2 = dyn.kick %v1, %f1, %m, %half : !vec
  // CHECK: dyn.return %[[X1]], %[[V2]], %[[F1]]
  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec
}

// CHECK-LABEL: dyn.program @leapfrog(
// CHECK-SAME:    attributes {provides = ["symplectic", "time_reversible"], velocity_offset = -5.000000e-01 : f64}
dyn.program @leapfrog(%x: !vec, %v: !vec, %m: !real, %cell: !md.cell,
                      %dt: f64, %eps: f64, %sigma: f64) -> (!vec, !vec)
    attributes {velocity_offset = -0.5,
                provides = ["symplectic", "time_reversible"]} {
  %f = md.evaluate @lj(%x, %cell, %eps, %sigma) request [forces]
      : (!vec, !md.cell, f64, f64) -> !vec
  %v1 = dyn.kick %v, %f, %m, %dt : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  dyn.return %x1, %v1 : !vec, !vec
}

// The step loop carries the state as loop-carried values.
//
// CHECK-LABEL: func.func @run_segment(
func.func @run_segment(%x0: !vec, %v0: !vec, %f0: !vec, %m: !real,
                       %cell: !md.cell, %dt: f64, %eps: f64, %sigma: f64,
                       %n: index) -> (!vec, !vec, !vec) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.for
  // CHECK-SAME: iter_args(%[[XA:[a-z0-9]+]] = %{{[a-z0-9]+}}, %[[VA:[a-z0-9]+]] = %{{[a-z0-9]+}}, %[[FA:[a-z0-9]+]] = %{{[a-z0-9]+}})
  %x, %v, %f = scf.for %s = %c0 to %n step %c1
      iter_args(%xa = %x0, %va = %v0, %fa = %f0) -> (!vec, !vec, !vec) {
    // CHECK: %[[STATE:[0-9]+]]:3 = dyn.step @velocity_verlet(%[[XA]], %[[VA]], %[[FA]],
    %xb, %vb, %fb = dyn.step @velocity_verlet(
        %xa, %va, %fa, %m, %cell, %dt, %eps, %sigma)
        : (!vec, !vec, !vec, !real, !md.cell, f64, f64, f64)
        -> (!vec, !vec, !vec)
    // CHECK: scf.yield %[[STATE]]#0, %[[STATE]]#1, %[[STATE]]#2
    scf.yield %xb, %vb, %fb : !vec, !vec, !vec
  }
  return %x, %v, %f : !vec, !vec, !vec
}

// The ops of a coupling (docs/dynamics.md, Section 2.2.3): a period of
// stochastic velocity rescaling with the removal of the motion of the
// center of mass.
//
// CHECK-LABEL: func.func @couple(
func.func @couple(%v: !vec, %f: !vec, %m: !real, %dt: f64, %step: i64,
                  %total: vector<3xf64>) -> !vec {
  // CHECK: dyn.momentum %{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>
  %p = dyn.momentum %v, %m : !vec
  %vcm = arith.divf %p, %total : vector<3xf64>
  // CHECK: dyn.kinetic %{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64> -> f64
  %k = dyn.kinetic %v, %m : !vec -> f64
  // CHECK: dyn.kinetic %{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64> -> vector<3xf64>
  %ka = dyn.kinetic %v, %m : !vec -> vector<3xf64>
  // CHECK: dyn.half_step_excess %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>
  %e = dyn.half_step_excess %f, %m, %dt : !vec
  // CHECK: %[[ALPHA:[0-9]+]] = dyn.velocity_rescaling %{{[0-9]+}}, %{{[a-z0-9]+}} {decay = 0.98999999999999999 : f64, freedom = 8.900000e+03 : f64, seed = 314159 : i64, target_kinetic = 1.100000e+04 : f64} : f64
  %alpha = dyn.velocity_rescaling %k, %step
      {seed = 314159 : i64, target_kinetic = 1.1e+04, freedom = 8.9e+03,
       decay = 0.99} : f64
  // CHECK: dyn.nose_hoover %{{[0-9]+}}, %{{[a-z0-9]+}} {chain_length = 3 : i64, freedom = 8.900000e+03 : f64, period = 2.000000e-02 : f64, temperature = 3.000000e+02 : f64, time_constant = 1.000000e+00 : f64} : f64
  %beta = dyn.nose_hoover %k, %step
      {temperature = 300.0, time_constant = 1.0, chain_length = 3 : i64,
       period = 0.02, freedom = 8.9e+03} : f64
  // CHECK: %[[V1:[0-9]+]] = dyn.shift %{{[a-z0-9]+}}, %{{[0-9]+}} : !md.field<@atoms, 3 x f64>
  %v1 = dyn.shift %v, %vcm : !vec
  // CHECK: %[[V2:[0-9]+]] = dyn.scale %[[V1]], %[[ALPHA]] : !md.field<@atoms, 3 x f64>, f64
  %v2 = dyn.scale %v1, %alpha : !vec, f64
  // CHECK: dyn.scale %[[V2]], %{{[0-9]+}} : !md.field<@atoms, 3 x f64>, vector<3xf64>
  %v3 = dyn.scale %v2, %e : !vec, vector<3xf64>
  // CHECK: dyn.to_bath %{{[0-9]+}} : f64
  dyn.to_bath %beta : f64
  return %v3 : !vec
}

// The O step of Langevin dynamics, and the claims that a program with it
// makes.
//
// CHECK-LABEL: dyn.program @langevin(
// CHECK-SAME:    attributes {provides = ["thermostatting"], requires = ["temperature"]}
dyn.program @langevin(%v: !vec, %m: !real, %step: i64,
                      %ids: !md.field<@atoms, i32>) -> !vec
    attributes {provides = ["thermostatting"], requires = ["temperature"]} {
  // CHECK: dyn.noise %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} {decay = 0.98999999999999999 : f64, seed = 314159 : i64, spread = 2.200000e-01 : f64} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i32>
  %v1 = dyn.noise %v, %m, %step, %ids
      {seed = 314159 : i64, decay = 0.99, spread = 0.22}
      : !vec, !md.field<@atoms, i32>
  dyn.return %v1 : !vec
}

// A step between couplings computes no global sum (D67): the forces are
// not one.
//
// CHECK-LABEL: dyn.program @plain(
// CHECK-SAME:    attributes {provides = ["no_global_sums"]}
dyn.program @plain(%x: !vec, %v: !vec, %f: !vec, %m: !real,
                   %cell: !md.cell, %dt: f64, %eps: f64, %sigma: f64)
    -> (!vec, !vec, !vec) attributes {provides = ["no_global_sums"]} {
  %v1 = dyn.kick %v, %f, %m, %dt : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  %f1 = md.evaluate @lj(%x1, %cell, %eps, %sigma) request [forces]
      : (!vec, !md.cell, f64, f64) -> !vec
  dyn.return %x1, %v1, %f1 : !vec, !vec, !vec
}

// A period of stochastic cell rescaling (docs/dynamics.md, Section 2.2.5,
// item 2): the pressure of the groups, the strain, the cell, the positions,
// and the work. The memory of the cell may be in any memory space.
//
// CHECK-LABEL: func.func @barostat(
func.func @barostat(%x: !vec, %v: !vec, %f: !vec, %m: !real, %cell: !md.cell,
                    %dt: f64, %step: i64, %total: vector<3xf64>,
                    %w: vector<3xf64>, %c: f64, %box: memref<3xf64>,
                    %tilted: memref<6xf64, 1>,
                    %settles: !md.relation<@atoms, 3, ordered, @settles>)
    -> (!vec, f64) {
  // CHECK: %kinetic, %excess, %half = dyn.group_kinetic %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} groups(%{{[a-z0-9]+}} : !md.relation<@atoms, 3, ordered, @settles>) center(%{{[a-z0-9]+}}) : !md.field<@atoms, 3 x f64>
  %k, %e, %h = dyn.group_kinetic %x, %v, %f, %m, %cell, %dt
      groups(%settles : !md.relation<@atoms, 3, ordered, @settles>)
      center(%total) : !vec
  // CHECK: dyn.load_cell %{{[a-z0-9]+}} : memref<3xf64> -> f64, f64, f64
  %l:3 = dyn.load_cell %box : memref<3xf64> -> f64, f64, f64
  // CHECK: dyn.load_cell %{{[a-z0-9]+}} : memref<6xf64, 1> -> f64, f64, f64, f64, f64, f64
  %t:6 = dyn.load_cell %tilted : memref<6xf64, 1> -> f64, f64, f64, f64, f64, f64
  // CHECK: dyn.pressure
  %p = dyn.pressure %h, %w, %l#0, %l#1, %l#2, %c : vector<3xf64>
  // CHECK: dyn.cell_rescaling %{{[0-9]+}}, %{{[0-9]+}}#0, %{{[0-9]+}}#1, %{{[0-9]+}}#2, %{{[a-z0-9]+}} {compressibility = [4.500000e-05, 4.500000e-05], coupling = "semi_isotropic", kT = 2.494000e+00 : f64, rate = 1.000000e-02 : f64, seed = 314159 : i64, target = 1.013250e+00 : f64, tension = 0.000000e+00 : f64} : vector<3xf64>
  %de = dyn.cell_rescaling %p, %l#0, %l#1, %l#2, %step
      {coupling = "semi_isotropic", seed = 314159 : i64, target = 1.01325,
       compressibility = [4.5e-05, 4.5e-05], rate = 0.01, kT = 2.494,
       tension = 0.0} : vector<3xf64>
  %mu = math.exp %de : vector<3xf64>
  // CHECK: dyn.scale_cell %{{[0-9]+}}#0, %{{[0-9]+}}#1, %{{[0-9]+}}#2 by %{{[0-9]+}} : f64, f64, f64 -> f64, f64, f64
  %n:3 = dyn.scale_cell %l#0, %l#1, %l#2 by %mu : f64, f64, f64 -> f64, f64, f64
  // CHECK: dyn.store_cell %{{[0-9]+}}#0, %{{[0-9]+}}#1, %{{[0-9]+}}#2 into %{{[a-z0-9]+}} : memref<3xf64>
  dyn.store_cell %n#0, %n#1, %n#2 into %box : memref<3xf64>
  // CHECK: dyn.scale_positions %{{[a-z0-9]+}}, %{{[0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} groups(%{{[a-z0-9]+}} : !md.relation<@atoms, 3, ordered, @settles>) : !md.field<@atoms, 3 x f64>
  %x1 = dyn.scale_positions %x, %mu, %m, %cell
      groups(%settles : !md.relation<@atoms, 3, ordered, @settles>) : !vec
  %v0 = arith.mulf %l#0, %l#1 : f64
  %vol = arith.mulf %v0, %l#2 : f64
  // CHECK: dyn.scaling_work kinetic(%kinetic) strain(%{{[0-9]+}}) virial_before(%{{[a-z0-9]+}}) constant(%{{[a-z0-9]+}}) volumes(%{{[0-9]+}}) {count = "first_order"} : f64
  %work = dyn.scaling_work kinetic(%k) strain(%mu) virial_before(%w)
      constant(%c) volumes(%vol) {count = "first_order"} : f64
  return %x1, %work : !vec, f64
}
