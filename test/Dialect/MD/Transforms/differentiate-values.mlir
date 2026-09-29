// Runs the kernels that differentiation generates and compares their values
// with closed-form expressions.
//
// RUN: mdir-opt %s --md-differentiate --test-md-outline-kernels=erase-md \
// RUN: | mlir-opt %lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// The reference values are for the Lennard-Jones potential
//
//   u(r) = 4 eps ((sigma/r)^12 - (sigma/r)^6)
//
// with eps = 1.7, sigma = 0.9, and a cutoff of 2.5. They come from the
// closed-form derivatives
//
//   du/dr     = -(24 eps / r)     (2 (sigma/r)^12 - (sigma/r)^6)
//   du/dsigma =  (24 eps / sigma) (2 (sigma/r)^12 - (sigma/r)^6)
//
// Each check prints the value and then 1 if it agrees with the reference to
// a relative tolerance of 1e-12, or 0 if it does not.

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @force_shift(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64)
    -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.5) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(force_shift) {
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

md.potential @switch(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64)
    -> f64 {
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

md.potential @shift(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64)
    -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.5) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(shift) {
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

md.function @requests(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64)
    -> (f64, !vec, vector<9xf64>, f64, f64, !vec, f64, !vec) {
  %u0, %f0, %w0, %g0 = md.evaluate @force_shift(%x, %cell, %eps, %sigma)
      request [energy, forces, virial, derivative(3)]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec, vector<9xf64>, f64)
  %u1, %f1 = md.evaluate @switch(%x, %cell, %eps, %sigma)
      request [energy, forces]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec)
  %u2, %f2 = md.evaluate @shift(%x, %cell, %eps, %sigma)
      request [energy, forces]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec)
  md.return %u0, %f0, %w0, %g0, %u1, %f1, %u2, %f2
      : f64, !vec, vector<9xf64>, f64, f64, !vec, f64, !vec
}

// The kernels of the generated functions, in the order of the requests. The
// arguments are the distance, the displacement, eps, and sigma.
func.func private @force_shift.energy_forces_virial_derivative3.kernel0(
    f64, vector<3xf64>, f64, f64) -> f64
func.func private @force_shift.energy_forces_virial_derivative3.kernel1(
    f64, vector<3xf64>, f64, f64) -> vector<3xf64>
func.func private @force_shift.energy_forces_virial_derivative3.kernel2(
    f64, vector<3xf64>, f64, f64) -> vector<9xf64>
func.func private @force_shift.energy_forces_virial_derivative3.kernel3(
    f64, vector<3xf64>, f64, f64) -> f64
func.func private @switch.energy_forces.kernel0(
    f64, vector<3xf64>, f64, f64) -> f64
func.func private @switch.energy_forces.kernel1(
    f64, vector<3xf64>, f64, f64) -> vector<3xf64>
func.func private @shift.energy_forces.kernel0(
    f64, vector<3xf64>, f64, f64) -> f64
func.func private @shift.energy_forces.kernel1(
    f64, vector<3xf64>, f64, f64) -> vector<3xf64>

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64) {
  %tolerance = arith.constant 1.0e-12 : f64
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %magnitude = math.absf %reference : f64
  %bound = arith.mulf %tolerance, %magnitude : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%value) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

// Energy and the x component of the force for a pair at distance `r` along
// the x axis.
func.func @check_switch(%r: f64, %energy: f64, %force: f64) {
  %d     = arith.constant dense<[1.0, 0.0, 0.0]> : vector<3xf64>
  %eps   = arith.constant 1.7 : f64
  %sigma = arith.constant 0.9 : f64
  %u = call @switch.energy_forces.kernel0(%r, %d, %eps, %sigma)
      : (f64, vector<3xf64>, f64, f64) -> f64
  %f = call @switch.energy_forces.kernel1(%r, %d, %eps, %sigma)
      : (f64, vector<3xf64>, f64, f64) -> vector<3xf64>
  %fx = vector.extract %f[0] : f64 from vector<3xf64>
  call @check(%u, %energy) : (f64, f64) -> ()
  call @check(%fx, %force) : (f64, f64) -> ()
  return
}

func.func @check_shift(%r: f64, %energy: f64, %force: f64) {
  %d     = arith.constant dense<[1.0, 0.0, 0.0]> : vector<3xf64>
  %eps   = arith.constant 1.7 : f64
  %sigma = arith.constant 0.9 : f64
  %u = call @shift.energy_forces.kernel0(%r, %d, %eps, %sigma)
      : (f64, vector<3xf64>, f64, f64) -> f64
  %f = call @shift.energy_forces.kernel1(%r, %d, %eps, %sigma)
      : (f64, vector<3xf64>, f64, f64) -> vector<3xf64>
  %fx = vector.extract %f[0] : f64 from vector<3xf64>
  call @check(%u, %energy) : (f64, f64) -> ()
  call @check(%fx, %force) : (f64, f64) -> ()
  return
}

func.func @main() {
  %eps   = arith.constant 1.7 : f64
  %sigma = arith.constant 0.9 : f64

  //===--------------------------------------------------------------------===//
  // Force shift, at r = 1.3 with d = (0.78, 1.04, 0).
  //===--------------------------------------------------------------------===//

  %r = arith.constant 1.3 : f64
  %d = arith.constant dense<[0.78, 1.04, 0.0]> : vector<3xf64>

  // Energy.
  // CHECK:      -0.609046
  // CHECK-NEXT: 1
  %u = call @force_shift.energy_forces_virial_derivative3.kernel0(
      %r, %d, %eps, %sigma) : (f64, vector<3xf64>, f64, f64) -> f64
  %u_ref = arith.constant -0.609046169505669 : f64
  call @check(%u, %u_ref) : (f64, f64) -> ()

  // Force.
  // CHECK-NEXT: -1.59553
  // CHECK-NEXT: 1
  // CHECK-NEXT: -2.12738
  // CHECK-NEXT: 1
  %f = call @force_shift.energy_forces_virial_derivative3.kernel1(
      %r, %d, %eps, %sigma) : (f64, vector<3xf64>, f64, f64) -> vector<3xf64>
  %fx = vector.extract %f[0] : f64 from vector<3xf64>
  %fy = vector.extract %f[1] : f64 from vector<3xf64>
  %fx_ref = arith.constant -1.595532257068931 : f64
  %fy_ref = arith.constant -2.127376342758575 : f64
  call @check(%fx, %fx_ref) : (f64, f64) -> ()
  call @check(%fy, %fy_ref) : (f64, f64) -> ()

  // Virial: the xx, xy, yx, and yy components.
  // CHECK-NEXT: -1.24452
  // CHECK-NEXT: 1
  // CHECK-NEXT: -1.65935
  // CHECK-NEXT: 1
  // CHECK-NEXT: -1.65935
  // CHECK-NEXT: 1
  // CHECK-NEXT: -2.21247
  // CHECK-NEXT: 1
  %w = call @force_shift.energy_forces_virial_derivative3.kernel2(
      %r, %d, %eps, %sigma) : (f64, vector<3xf64>, f64, f64) -> vector<9xf64>
  %wxx = vector.extract %w[0] : f64 from vector<9xf64>
  %wxy = vector.extract %w[1] : f64 from vector<9xf64>
  %wyx = vector.extract %w[3] : f64 from vector<9xf64>
  %wyy = vector.extract %w[4] : f64 from vector<9xf64>
  %wxx_ref = arith.constant -1.2445151605137663 : f64
  %wxy_ref = arith.constant -1.6593535473516883 : f64
  %wyy_ref = arith.constant -2.212471396468918 : f64
  call @check(%wxx, %wxx_ref) : (f64, f64) -> ()
  call @check(%wxy, %wxy_ref) : (f64, f64) -> ()
  call @check(%wyx, %wxy_ref) : (f64, f64) -> ()
  call @check(%wyy, %wyy_ref) : (f64, f64) -> ()

  // Derivative with respect to sigma.
  // CHECK-NEXT: -3.51221
  // CHECK-NEXT: 1
  %g = call @force_shift.energy_forces_virial_derivative3.kernel3(
      %r, %d, %eps, %sigma) : (f64, vector<3xf64>, f64, f64) -> f64
  %g_ref = arith.constant -3.512209475631653 : f64
  call @check(%g, %g_ref) : (f64, f64) -> ()

  //===--------------------------------------------------------------------===//
  // Switch from 2.0: below the switching distance, and inside the switch.
  //===--------------------------------------------------------------------===//

  // CHECK-NEXT: -0.666261
  // CHECK-NEXT: 1
  // CHECK-NEXT: -2.07276
  // CHECK-NEXT: 1
  %r1 = arith.constant 1.3 : f64
  %su1 = arith.constant -0.6662605807453331 : f64
  %sf1 = arith.constant -2.0727621964652734 : f64
  call @check_switch(%r1, %su1, %sf1) : (f64, f64, f64) -> ()

  // CHECK-NEXT: -0.0216535
  // CHECK-NEXT: 1
  // CHECK-NEXT: -0.0765522
  // CHECK-NEXT: 1
  %r2 = arith.constant 2.2 : f64
  %su2 = arith.constant -0.021653508840438488 : f64
  %sf2 = arith.constant -0.07655224262585227 : f64
  call @check_switch(%r2, %su2, %sf2) : (f64, f64, f64) -> ()

  //===--------------------------------------------------------------------===//
  // Shift: the energy moves, the force does not.
  //===--------------------------------------------------------------------===//

  // CHECK-NEXT: -0.651491
  // CHECK-NEXT: 1
  // CHECK-NEXT: -2.07276
  // CHECK-NEXT: 1
  %hu1 = arith.constant -0.6514906818536337 : f64
  call @check_shift(%r1, %hu1, %sf1) : (f64, f64, f64) -> ()

  // CHECK-NEXT: -0.0169541
  // CHECK-NEXT: 1
  // CHECK-NEXT: -0.039142
  // CHECK-NEXT: 1
  %hu2 = arith.constant -0.01695406506815541 : f64
  %hf2 = arith.constant -0.03914202303335085 : f64
  call @check_shift(%r2, %hu2, %hf2) : (f64, f64, f64) -> ()

  return
}
