// RUN: mdir-opt %s --dyn-expand-couplings | FileCheck %s

// Each op of a coupling expands to the text that the driver writes for it
// (lib/Driver/Builder.cpp; docs/dynamics.md, Section 2.2.3), in its order.
// expand-couplings-noise.mlir checks dyn.noise against the driver's text
// verbatim, and Lit/lift-couplings.test every op on whole programs.

!vec  = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>

md.particle_set @atoms

// The functions of the host are declared as the driver declares them.
// CHECK-DAG: func.func private @mdrtAddBath(f64) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @mdrtNoseHooverFactor(i64, f64) -> f64
// CHECK-DAG: func.func private @mdrtBussiFactor(i64, i64, f64, f64, f64, f64) -> f64

// The momentum, as the removal of the motion of the center of mass sums it
// (emitCoupling).
// CHECK-LABEL: func.func @momentum(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[M:[a-z0-9]+]]: !md.field<@atoms, f64>
// CHECK-NEXT:    %[[P:[0-9]+]] = md.sum_particles gather(%[[V]], %[[M]] : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>) {
// CHECK-NEXT:    ^bb0(%[[VI:[a-z0-9]+]]: vector<3xf64>, %[[MI:[a-z0-9]+]]: f64):
// CHECK-NEXT:      %[[MB:[0-9]+]] = vector.broadcast %[[MI]] : f64 to vector<3xf64>
// CHECK-NEXT:      %[[MV:[0-9]+]] = arith.mulf %[[MB]], %[[VI]] : vector<3xf64>
// CHECK-NEXT:      md.yield %[[MV]] : vector<3xf64>
// CHECK-NEXT:    } : vector<3xf64>
// CHECK-NEXT:    return %[[P]]
func.func @momentum(%v: !vec, %m: !real) -> vector<3xf64> {
  %p = dyn.momentum %v, %m : !vec
  return %p : vector<3xf64>
}

// The kinetic energy, of each axis (emitKineticVector) and in all
// (emitKineticEnergy), summed in different orders.
// CHECK-LABEL: func.func @kinetic(
// CHECK-NEXT:    md.sum_particles gather(
// CHECK-NEXT:    ^bb0(%[[VI:[a-z0-9]+]]: vector<3xf64>, %[[MI:[a-z0-9]+]]: f64):
// CHECK-NEXT:      %[[C:.+]] = arith.constant 5.000000e-01 : f64
// CHECK-NEXT:      %[[HM:[0-9]+]] = arith.mulf %[[C]], %[[MI]] : f64
// CHECK-NEXT:      %[[HMB:[0-9]+]] = vector.broadcast %[[HM]] : f64 to vector<3xf64>
// CHECK-NEXT:      %[[SQ:[0-9]+]] = arith.mulf %[[VI]], %[[VI]] : vector<3xf64>
// CHECK-NEXT:      %[[KE:[0-9]+]] = arith.mulf %[[HMB]], %[[SQ]] : vector<3xf64>
// CHECK-NEXT:      md.yield %[[KE]] : vector<3xf64>
// CHECK-NEXT:    } : vector<3xf64>
// CHECK-NEXT:    md.sum_particles gather(
// CHECK-NEXT:    ^bb0(%[[VJ:[a-z0-9]+]]: vector<3xf64>, %[[MJ:[a-z0-9]+]]: f64):
// CHECK-NEXT:      %[[H:.+]] = arith.constant 5.000000e-01 : f64
// CHECK-NEXT:      %[[SQ2:[0-9]+]] = arith.mulf %[[VJ]], %[[VJ]] : vector<3xf64>
// CHECK-NEXT:      %[[V2:[0-9]+]] = vector.reduction <add>, %[[SQ2]] : vector<3xf64> into f64
// CHECK-NEXT:      %[[MV2:[0-9]+]] = arith.mulf %[[MJ]], %[[V2]] : f64
// CHECK-NEXT:      %[[K:[0-9]+]] = arith.mulf %[[H]], %[[MV2]] : f64
// CHECK-NEXT:      md.yield %[[K]] : f64
// CHECK-NEXT:    } : f64
func.func @kinetic(%v: !vec, %m: !real) -> (vector<3xf64>, f64) {
  %ka = dyn.kinetic %v, %m : !vec -> vector<3xf64>
  %k = dyn.kinetic %v, %m : !vec -> f64
  return %ka, %k : vector<3xf64>, f64
}

// The excess of the kinetic energy of the half steps (emitKineticExcess).
// CHECK-LABEL: func.func @excess(
// CHECK-SAME:    %[[DT:[a-z0-9]+]]: f64
// CHECK-NEXT:    %[[S:[0-9]+]] = md.sum_particles gather(
// CHECK-NEXT:    ^bb0(%[[FI:[a-z0-9]+]]: vector<3xf64>, %[[MI:[a-z0-9]+]]: f64):
// CHECK-NEXT:      %[[SQ:[0-9]+]] = arith.mulf %[[FI]], %[[FI]] : vector<3xf64>
// CHECK-NEXT:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      %[[ONE:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:      %[[MASSLESS:[0-9]+]] = arith.cmpf oeq, %[[MI]], %[[ZERO]] : f64
// CHECK-NEXT:      %[[SAFE:[0-9]+]] = arith.select %[[MASSLESS]], %[[ONE]], %[[MI]] : f64
// CHECK-NEXT:      %[[INV:[0-9]+]] = arith.divf %[[ONE]], %[[SAFE]] : f64
// CHECK-NEXT:      %[[W:[0-9]+]] = arith.select %[[MASSLESS]], %[[ZERO]], %[[INV]] : f64
// CHECK-NEXT:      %[[WB:[0-9]+]] = vector.broadcast %[[W]] : f64 to vector<3xf64>
// CHECK-NEXT:      %[[G:[0-9]+]] = arith.mulf %[[SQ]], %[[WB]] : vector<3xf64>
// CHECK-NEXT:      md.yield %[[G]] : vector<3xf64>
// CHECK-NEXT:    } : vector<3xf64>
// CHECK-NEXT:    %[[D:[0-9]+]] = arith.mulf %[[DT]], %[[DT]] : f64
// CHECK-NEXT:    %[[EIGHTH:.+]] = arith.constant 1.250000e-01 : f64
// CHECK-NEXT:    %[[DS:[0-9]+]] = arith.mulf %[[D]], %[[EIGHTH]] : f64
// CHECK-NEXT:    %[[DB:[0-9]+]] = vector.broadcast %[[DS]] : f64 to vector<3xf64>
// CHECK-NEXT:    %[[E:[0-9]+]] = arith.mulf %[[S]], %[[DB]] : vector<3xf64>
// CHECK-NEXT:    return %[[E]]
func.func @excess(%f: !vec, %m: !real, %dt: f64) -> vector<3xf64> {
  %e = dyn.half_step_excess %f, %m, %dt : !vec
  return %e : vector<3xf64>
}

// The factors of the thermostats and the bath: calls of the host, with the
// constants of the driver as attributes.
// CHECK-LABEL: func.func @thermostats(
// CHECK-SAME:    %[[K:[a-z0-9]+]]: f64, %[[STEP:[a-z0-9]+]]: i64
// CHECK-NEXT:    %[[SEED:.+]] = arith.constant 314159 : i64
// CHECK-NEXT:    %[[TARGET:.+]] = arith.constant 1969.6421502334751 : f64
// CHECK-NEXT:    %[[FREEDOM:.+]] = arith.constant 2.589000e+03 : f64
// CHECK-NEXT:    %[[DECAY:.+]] = arith.constant 0.90483741803595952 : f64
// CHECK-NEXT:    %[[ALPHA:[0-9]+]] = call @mdrtBussiFactor(%[[SEED]], %[[STEP]], %[[K]], %[[TARGET]], %[[FREEDOM]], %[[DECAY]]) : (i64, i64, f64, f64, f64, f64) -> f64
// CHECK-NEXT:    %[[BETA:[0-9]+]] = call @mdrtNoseHooverFactor(%[[STEP]], %[[K]]) : (i64, f64) -> f64
// CHECK-NEXT:    call @mdrtAddBath(%[[K]]) : (f64) -> ()
// CHECK-NEXT:    return %[[ALPHA]], %[[BETA]]
func.func @thermostats(%k: f64, %step: i64) -> (f64, f64) {
  %alpha = dyn.velocity_rescaling %k, %step
      {seed = 314159 : i64, target_kinetic = 1969.6421502334751,
       freedom = 2.589000e+03, decay = 0.90483741803595952} : f64
  %beta = dyn.nose_hoover %k, %step
      {temperature = 300.0, time_constant = 1.0, chain_length = 3 : i64,
       period = 0.05, freedom = 2.589000e+03} : f64
  dyn.to_bath %k : f64
  return %alpha, %beta : f64, f64
}

// A chain of updates of the velocities is one loop that applies them in
// order: the center of mass subtracted, then α, then 1/μ (emitCoupling).
// CHECK-LABEL: func.func @chain(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[U:[a-z0-9]+]]: vector<3xf64>, %[[ALPHA:[a-z0-9]+]]: f64, %[[MUINV:[a-z0-9]+]]: vector<3xf64>
// CHECK-NEXT:    %[[R:[0-9]+]] = md.map_particles gather(%[[V]] : !md.field<@atoms, 3 x f64>) {
// CHECK-NEXT:    ^bb0(%[[VI:[a-z0-9]+]]: vector<3xf64>):
// CHECK-NEXT:      %[[D:[0-9]+]] = arith.subf %[[VI]], %[[U]] : vector<3xf64>
// CHECK-NEXT:      %[[AB:[0-9]+]] = vector.broadcast %[[ALPHA]] : f64 to vector<3xf64>
// CHECK-NEXT:      %[[S:[0-9]+]] = arith.mulf %[[AB]], %[[D]] : vector<3xf64>
// CHECK-NEXT:      %[[VS:[0-9]+]] = arith.mulf %[[MUINV]], %[[S]] : vector<3xf64>
// CHECK-NEXT:      md.yield %[[VS]] : vector<3xf64>
// CHECK-NEXT:    } : !md.field<@atoms, 3 x f64>
// CHECK-NEXT:    return %[[R]]
func.func @chain(%v: !vec, %u: vector<3xf64>, %alpha: f64,
                 %muinv: vector<3xf64>) -> !vec {
  %v1 = dyn.shift %v, %u : !vec
  %v2 = dyn.scale %v1, %alpha : !vec, f64
  %v3 = dyn.scale %v2, %muinv : !vec, vector<3xf64>
  return %v3 : !vec
}

// An update whose value has another use ends its loop.
// CHECK-LABEL: func.func @split(
// CHECK-NEXT:    %[[A:[0-9]+]] = md.map_particles
// CHECK:           arith.subf
// CHECK-NOT:       arith.mulf
// CHECK:         } : !md.field<@atoms, 3 x f64>
// CHECK-NEXT:    %[[B:[0-9]+]] = md.map_particles gather(%[[A]] :
// CHECK:           vector.broadcast
// CHECK-NEXT:      arith.mulf
// CHECK:         return %[[A]], %[[B]]
func.func @split(%v: !vec, %u: vector<3xf64>, %alpha: f64) -> (!vec, !vec) {
  %v1 = dyn.shift %v, %u : !vec
  %v2 = dyn.scale %v1, %alpha : !vec, f64
  return %v1, %v2 : !vec, !vec
}
