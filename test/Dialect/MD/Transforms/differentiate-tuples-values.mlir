// Runs the kernels that differentiation generates for terms over tuples
// and compares their values with those of
// test/Integration/Inputs/tuples_reference.py, which checks its closed forms
// against finite differences.
//
// RUN: mdir-opt %s --md-differentiate --test-md-outline-kernels=erase-md \
// RUN: | mlir-opt %lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// The members of the tuples are at
//
//   (0.1, 0.2, -0.3), (1.0, 0.4, 0.2), (1.3, 1.5, -0.1), (2.4, 1.7, 0.9)
//
// A check prints 1 if every component of a value agrees with the reference
// to 1e-12 of the largest component of the reference, or 0 if one does not.

!vec       = !md.field<@atoms, 3 x f64>
!bonds     = !md.relation<@atoms, 2, unordered, @bonds>
!angles    = !md.relation<@atoms, 3, reversal, @angles>
!dihedrals = !md.relation<@atoms, 4, reversal, @dihedrals>
!of_bond     = !md.field<@bonds, f64>
!of_angle    = !md.field<@angles, f64>
!of_dihedral = !md.field<@dihedrals, f64>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
md.tuple_set @dihedrals on(@atoms) arity(4) orientation(reversal)

// u = scale k (r - r0)^2 / 2
md.potential @bond(%x: !vec, %cell: !md.cell, %bonds: !bonds,
                   %k: !of_bond, %r0: !of_bond, %scale: f64) -> f64 {
  %u = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1))
         tuple(%k, %r0 : !of_bond, !of_bond) {
  ^bb0(%r: f64, %k_t: f64, %r0_t: f64):
    %half = arith.constant 0.5 : f64
    %dr = arith.subf %r, %r0_t : f64
    %dr2 = arith.mulf %dr, %dr : f64
    %kh = arith.mulf %half, %k_t : f64
    %e0 = arith.mulf %kh, %dr2 : f64
    %e = arith.mulf %scale, %e0 : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64
  md.return %u : f64
}

// u = k (cos(theta) - c0)^2 / 2
md.potential @bend(%x: !vec, %cell: !md.cell, %angles: !angles,
                   %k: !of_angle, %c0: !of_angle) -> f64 {
  %u = md.sum_tuples %angles, %x, %cell coordinates(cosine(0, 1, 2))
         tuple(%k, %c0 : !of_angle, !of_angle) {
  ^bb0(%c: f64, %k_t: f64, %c0_t: f64):
    %half = arith.constant 0.5 : f64
    %dc = arith.subf %c, %c0_t : f64
    %dc2 = arith.mulf %dc, %dc : f64
    %kh = arith.mulf %half, %k_t : f64
    %e = arith.mulf %kh, %dc2 : f64
    md.yield %e : f64
  } : !angles, !vec -> f64
  md.return %u : f64
}

// u = k (theta - theta0)^2 / 2
md.potential @angle(%x: !vec, %cell: !md.cell, %angles: !angles,
                    %k: !of_angle, %t0: !of_angle) -> f64 {
  %u = md.sum_tuples %angles, %x, %cell coordinates(angle(0, 1, 2))
         tuple(%k, %t0 : !of_angle, !of_angle) {
  ^bb0(%t: f64, %k_t: f64, %t0_t: f64):
    %half = arith.constant 0.5 : f64
    %dt = arith.subf %t, %t0_t : f64
    %dt2 = arith.mulf %dt, %dt : f64
    %kh = arith.mulf %half, %k_t : f64
    %e = arith.mulf %kh, %dt2 : f64
    md.yield %e : f64
  } : !angles, !vec -> f64
  md.return %u : f64
}

// u = k (1 + cos(2 phi - phi0))
md.potential @twist(%x: !vec, %cell: !md.cell, %dihedrals: !dihedrals,
                    %k: !of_dihedral, %phi0: !of_dihedral) -> f64 {
  %u = md.sum_tuples %dihedrals, %x, %cell coordinates(dihedral(0, 1, 2, 3))
         tuple(%k, %phi0 : !of_dihedral, !of_dihedral) {
  ^bb0(%phi: f64, %k_t: f64, %phi0_t: f64):
    %one = arith.constant 1.0 : f64
    %two = arith.constant 2.0 : f64
    %p2 = arith.mulf %two, %phi : f64
    %a = arith.subf %p2, %phi0_t : f64
    %cos = math.cos %a : f64
    %s = arith.addf %one, %cos : f64
    %e = arith.mulf %k_t, %s : f64
    md.yield %e : f64
  } : !dihedrals, !vec -> f64
  md.return %u : f64
}

md.function @requests(%x: !vec, %cell: !md.cell, %bonds: !bonds,
                      %kb: !of_bond, %r0: !of_bond, %scale: f64,
                      %angles: !angles, %ka: !of_angle, %a0: !of_angle,
                      %dihedrals: !dihedrals, %kd: !of_dihedral,
                      %phi0: !of_dihedral)
    -> (f64, !vec, vector<9xf64>, f64, f64, !vec, vector<9xf64>,
        f64, !vec, vector<9xf64>, f64, !vec, vector<9xf64>) {
  %u0, %f0, %w0, %g0 = md.evaluate @bond(%x, %cell, %bonds, %kb, %r0, %scale)
      request [energy, forces, virial, derivative(5)]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, f64)
        -> (f64, !vec, vector<9xf64>, f64)
  %u1, %f1, %w1 = md.evaluate @bend(%x, %cell, %angles, %ka, %a0)
      request [energy, forces, virial]
      : (!vec, !md.cell, !angles, !of_angle, !of_angle)
        -> (f64, !vec, vector<9xf64>)
  %u2, %f2, %w2 = md.evaluate @angle(%x, %cell, %angles, %ka, %a0)
      request [energy, forces, virial]
      : (!vec, !md.cell, !angles, !of_angle, !of_angle)
        -> (f64, !vec, vector<9xf64>)
  %u3, %f3, %w3 = md.evaluate @twist(%x, %cell, %dihedrals, %kd, %phi0)
      request [energy, forces, virial]
      : (!vec, !md.cell, !dihedrals, !of_dihedral, !of_dihedral)
        -> (f64, !vec, vector<9xf64>)
  md.return %u0, %f0, %w0, %g0, %u1, %f1, %w1, %u2, %f2, %w2, %u3, %f3, %w3
      : f64, !vec, vector<9xf64>, f64, f64, !vec, vector<9xf64>,
        f64, !vec, vector<9xf64>, f64, !vec, vector<9xf64>
}

// The kernels of the generated functions, in the order of the requests. The
// arguments are the positions of the members, the two fields of the tuple,
// and, for the bond, the scale.
func.func private @bond.energy_forces_virial_derivative5.kernel0(
    vector<3xf64>, vector<3xf64>, f64, f64, f64) -> f64
func.func private @bond.energy_forces_virial_derivative5.kernel1(
    vector<3xf64>, vector<3xf64>, f64, f64, f64) -> (vector<3xf64>, vector<3xf64>)
func.func private @bond.energy_forces_virial_derivative5.kernel2(
    vector<3xf64>, vector<3xf64>, f64, f64, f64) -> vector<9xf64>
func.func private @bond.energy_forces_virial_derivative5.kernel3(
    vector<3xf64>, vector<3xf64>, f64, f64, f64) -> f64
func.func private @bend.energy_forces_virial.kernel0(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> f64
func.func private @bend.energy_forces_virial.kernel1(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> (vector<3xf64>, vector<3xf64>, vector<3xf64>)
func.func private @bend.energy_forces_virial.kernel2(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> vector<9xf64>
func.func private @angle.energy_forces_virial.kernel0(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> f64
func.func private @angle.energy_forces_virial.kernel1(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> (vector<3xf64>, vector<3xf64>, vector<3xf64>)
func.func private @angle.energy_forces_virial.kernel2(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> vector<9xf64>
func.func private @twist.energy_forces_virial.kernel0(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> f64
func.func private @twist.energy_forces_virial.kernel1(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> (vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>)
func.func private @twist.energy_forces_virial.kernel2(
    vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> vector<9xf64>

func.func private @printF64(f64)
func.func private @printNewline()

func.func @print(%agrees: i1) {
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @check(%value: f64, %reference: f64) {
  %tolerance = arith.constant 1.0e-12 : f64
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %magnitude = math.absf %reference : f64
  %bound = arith.mulf %tolerance, %magnitude : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  call @print(%agrees) : (i1) -> ()
  return
}

func.func @check3(%value: vector<3xf64>, %reference: vector<3xf64>) {
  %tolerance = arith.constant 1.0e-12 : f64
  %difference = arith.subf %value, %reference : vector<3xf64>
  %errors = math.absf %difference : vector<3xf64>
  %magnitudes = math.absf %reference : vector<3xf64>
  %error = vector.reduction <maxnumf>, %errors : vector<3xf64> into f64
  %magnitude = vector.reduction <maxnumf>, %magnitudes : vector<3xf64> into f64
  %bound = arith.mulf %tolerance, %magnitude : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  call @print(%agrees) : (i1) -> ()
  return
}

func.func @check9(%value: vector<9xf64>, %reference: vector<9xf64>) {
  %tolerance = arith.constant 1.0e-12 : f64
  %difference = arith.subf %value, %reference : vector<9xf64>
  %errors = math.absf %difference : vector<9xf64>
  %magnitudes = math.absf %reference : vector<9xf64>
  %error = vector.reduction <maxnumf>, %errors : vector<9xf64> into f64
  %magnitude = vector.reduction <maxnumf>, %magnitudes : vector<9xf64> into f64
  %bound = arith.mulf %tolerance, %magnitude : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  call @print(%agrees) : (i1) -> ()
  return
}

func.func @main() {
  %x0 = arith.constant dense<[0.1, 0.2, -0.3]> : vector<3xf64>
  %x1 = arith.constant dense<[1.0, 0.4, 0.2]> : vector<3xf64>
  %x2 = arith.constant dense<[1.3, 1.5, -0.1]> : vector<3xf64>
  %x3 = arith.constant dense<[2.4, 1.7, 0.9]> : vector<3xf64>
  %scale = arith.constant 1.0 : f64

  //===--------------------------------------------------------------------===//
  // A bond: k = 300, r0 = 0.8.
  //===--------------------------------------------------------------------===//

  %bond_k = arith.constant 300.0 : f64
  %bond_0 = arith.constant 0.8 : f64

  // Energy.
  // CHECK:      1
  %bond_u = call @bond.energy_forces_virial_derivative5.kernel0(%x0, %x1, %bond_k, %bond_0, %scale)
      : (vector<3xf64>, vector<3xf64>, f64, f64, f64) -> f64
  %bond_u_ref = arith.constant 9.285876439163632 : f64
  call @check(%bond_u, %bond_u_ref) : (f64, f64) -> ()

  // Forces on the members.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %bond_f0, %bond_f1 = call @bond.energy_forces_virial_derivative5.kernel1(%x0, %x1, %bond_k, %bond_0, %scale)
      : (vector<3xf64>, vector<3xf64>, f64, f64, f64) -> (vector<3xf64>, vector<3xf64>)
  %bond_f0_ref = arith.constant
      dense<[64.05208072295207, 14.233795716211569, 35.58448929052892]> : vector<3xf64>
  call @check3(%bond_f0, %bond_f0_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()
  %bond_f1_ref = arith.constant
      dense<[-64.05208072295207, -14.233795716211569, -35.58448929052892]> : vector<3xf64>
  call @check3(%bond_f1, %bond_f1_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()

  // Virial.
  // CHECK-NEXT: 1
  %bond_w = call @bond.energy_forces_virial_derivative5.kernel2(%x0, %x1, %bond_k, %bond_0, %scale)
      : (vector<3xf64>, vector<3xf64>, f64, f64, f64) -> vector<9xf64>
  %bond_w_ref = arith.constant dense<[
      -57.64687265065687, -12.810416144590413, -32.02604036147603,
      -12.810416144590414, -2.846759143242314, -7.1168978581057845,
      -32.026040361476035, -7.1168978581057845, -17.79224464526446]> : vector<9xf64>
  call @check9(%bond_w, %bond_w_ref)
      : (vector<9xf64>, vector<9xf64>) -> ()

  // Derivative with respect to the scale, which is the energy at a scale
  // of 1.
  // CHECK-NEXT: 1
  %bond_g = call @bond.energy_forces_virial_derivative5.kernel3(%x0, %x1, %bond_k, %bond_0, %scale)
      : (vector<3xf64>, vector<3xf64>, f64, f64, f64) -> f64
  call @check(%bond_g, %bond_u_ref) : (f64, f64) -> ()

  //===--------------------------------------------------------------------===//
  // An angle in the cosine: k = 40, c0 = -0.4.
  //===--------------------------------------------------------------------===//

  %bend_k = arith.constant 40.0 : f64
  %bend_0 = arith.constant -0.4 : f64

  // Energy.
  // CHECK-NEXT: 1
  %bend_u = call @bend.energy_forces_virial.kernel0(%x0, %x1, %x2, %bend_k, %bend_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> f64
  %bend_u_ref = arith.constant 0.31268212432209375 : f64
  call @check(%bend_u, %bend_u_ref) : (f64, f64) -> ()

  // Forces on the members.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %bend_f0, %bend_f1, %bend_f2 = call @bend.energy_forces_virial.kernel1(%x0, %x1, %x2, %bend_k, %bend_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> (vector<3xf64>, vector<3xf64>, vector<3xf64>)
  %bend_f0_ref = arith.constant
      dense<[-0.0882493132169786, -4.199196487241239, 1.8385273586870576]> : vector<3xf64>
  call @check3(%bend_f0, %bend_f0_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()
  %bend_f1_ref = arith.constant
      dense<[-3.2552251164874417, 4.4785468312374315, -4.157717193738775]> : vector<3xf64>
  call @check3(%bend_f1, %bend_f1_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()
  %bend_f2_ref = arith.constant
      dense<[3.3434744297044205, -0.2793503439961919, 2.319189835051717]> : vector<3xf64>
  call @check3(%bend_f2, %bend_f2_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()

  // Virial.
  // CHECK-NEXT: 1
  %bend_w = call @bend.energy_forces_virial.kernel2(%x0, %x1, %x2, %bend_k, %bend_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> vector<9xf64>
  %bend_w_ref = arith.constant dense<[
      1.082466710806607, 3.6954717353182582, -0.9589176723028379,
      3.6954717353182582, 0.5325539190524369, 2.183403346819477,
      -0.9589176723028369, 2.1834033468194773, -1.6150206298590442]> : vector<9xf64>
  call @check9(%bend_w, %bend_w_ref)
      : (vector<9xf64>, vector<9xf64>) -> ()

  //===--------------------------------------------------------------------===//
  // An angle: k = 50, theta0 = 1.9.
  //===--------------------------------------------------------------------===//

  %angle_k = arith.constant 50.0 : f64
  %angle_0 = arith.constant 1.9 : f64

  // Energy.
  // CHECK-NEXT: 1
  %angle_u = call @angle.energy_forces_virial.kernel0(%x0, %x1, %x2, %angle_k, %angle_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> f64
  %angle_u_ref = arith.constant 0.06414024938693 : f64
  call @check(%angle_u, %angle_u_ref) : (f64, f64) -> ()

  // Forces on the members.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %angle_f0, %angle_f1, %angle_f2 = call @angle.energy_forces_virial.kernel1(%x0, %x1, %x2, %angle_k, %angle_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> (vector<3xf64>, vector<3xf64>, vector<3xf64>)
  %angle_f0_ref = arith.constant
      dense<[-0.04647841775835347, -2.211598045001657, 0.9683003699656992]> : vector<3xf64>
  call @check3(%angle_f0, %angle_f0_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()
  %angle_f1_ref = arith.constant
      dense<[-1.7144350176368224, 2.3587239717187476, -2.189752074064877]> : vector<3xf64>
  call @check3(%angle_f1, %angle_f1_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()
  %angle_f2_ref = arith.constant
      dense<[1.760913435395176, -0.1471259267170905, 1.2214517040991775]> : vector<3xf64>
  call @check3(%angle_f2, %angle_f2_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()

  // Virial.
  // CHECK-NEXT: 1
  %angle_w = call @angle.energy_forces_virial.kernel2(%x0, %x1, %x2, %angle_k, %angle_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> vector<9xf64>
  %angle_w_ref = arith.constant dense<[
      0.5701046066010711, 1.9463004624863645, -0.5050348217393763,
      1.9463004624863645, 0.2804810896115319, 1.1499368005159554,
      -0.5050348217393761, 1.1499368005159556, -0.8505856962126029]> : vector<9xf64>
  call @check9(%angle_w, %angle_w_ref)
      : (vector<9xf64>, vector<9xf64>) -> ()

  //===--------------------------------------------------------------------===//
  // A dihedral: k = 2.5, phi0 = 0.7.
  //===--------------------------------------------------------------------===//

  %twist_k = arith.constant 2.5 : f64
  %twist_0 = arith.constant 0.7 : f64

  // Energy.
  // CHECK-NEXT: 1
  %twist_u = call @twist.energy_forces_virial.kernel0(%x0, %x1, %x2, %x3, %twist_k, %twist_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> f64
  %twist_u_ref = arith.constant 3.6538405599398587 : f64
  call @check(%twist_u, %twist_u_ref) : (f64, f64) -> ()

  // Forces on the members.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %twist_f0, %twist_f1, %twist_f2, %twist_f3 = call @twist.energy_forces_virial.kernel1(%x0, %x1, %x2, %x3, %twist_k, %twist_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> (vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>)
  %twist_f0_ref = arith.constant
      dense<[-2.2569669035489643, 1.553977212279615, 3.4409495414762894]> : vector<3xf64>
  call @check3(%twist_f0, %twist_f0_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()
  %twist_f1_ref = arith.constant
      dense<[2.453060944920859, -1.7407583173479975, -3.929719552021797]> : vector<3xf64>
  call @check3(%twist_f1, %twist_f1_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()
  %twist_f2_ref = arith.constant
      dense<[1.783098077426962, -0.8881249594516863, -1.4733601072292206]> : vector<3xf64>
  call @check3(%twist_f2, %twist_f2_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()
  %twist_f3_ref = arith.constant
      dense<[-1.979192118798857, 1.074906064520069, 1.9621301177747288]> : vector<3xf64>
  call @check3(%twist_f3, %twist_f3_ref)
      : (vector<3xf64>, vector<3xf64>) -> ()

  // Virial.
  // CHECK-NEXT: 1
  %twist_w = call @twist.energy_forces_virial.kernel2(%x0, %x1, %x2, %x3, %twist_k, %twist_0)
      : (vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, f64, f64) -> vector<9xf64>
  %twist_w_ref = arith.constant dense<[
      -0.20466932989624365, -0.16014848855906294, -0.791880454612806,
      -0.16014848855906294, 0.10964498602331174, 0.24188312685974678,
      -0.7918804546128064, 0.24188312685974678, 0.09502434387293146]> : vector<9xf64>
  call @check9(%twist_w, %twist_w_ref)
      : (vector<9xf64>, vector<9xf64>) -> ()

  // CHECK-NOT: 0
  return
}
