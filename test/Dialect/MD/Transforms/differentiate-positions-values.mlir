// Runs the kernels that differentiation generates for a sum over the
// particles of a function of their positions themselves, a term of the
// absolute positions (D148), and compares their values with the closed
// form: with a particle at x = (x, y, z),
//
//   k = a (x y + z^2) + sin x,
//
// the gradient is g = (a y + cos x, a x, 2 a z) and the force is -g. The
// virial is -dU/de when the positions scale by 1 + e about the origin
// (D154), x ⊗ (-g).
//
// RUN: mdir-opt %s --md-differentiate --test-md-outline-kernels=erase-md \
// RUN: | mlir-opt %lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// The particle is at (0.4, -0.7, 1.3) and a = 2.5: k = 3.9144183423086507,
// g = (-0.8289390059971149, 1.0, 6.5).

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>

md.particle_set @atoms

md.potential @wall(%x: !vec, %cell: !md.cell, %a: !real) -> f64 {
  %u = md.sum_particles gather(%x, %a : !vec, !real) {
  ^bb0(%p: vector<3xf64>, %ai: f64):
    %px = vector.extract %p[0] : f64 from vector<3xf64>
    %py = vector.extract %p[1] : f64 from vector<3xf64>
    %pz = vector.extract %p[2] : f64 from vector<3xf64>
    %xy = arith.mulf %px, %py : f64
    %zz = arith.mulf %pz, %pz : f64
    %s = arith.addf %xy, %zz : f64
    %as = arith.mulf %ai, %s : f64
    %sin = math.sin %px : f64
    %e = arith.addf %as, %sin : f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

md.function @requests(%x: !vec, %cell: !md.cell, %a: !real)
    -> (f64, !vec, vector<9xf64>) {
  %u, %f, %w = md.evaluate @wall(%x, %cell, %a)
      request [energy, forces, virial]
      : (!vec, !md.cell, !real) -> (f64, !vec, vector<9xf64>)
  md.return %u, %f, %w : f64, !vec, vector<9xf64>
}

func.func private @wall.energy_forces_virial.kernel0(vector<3xf64>, f64) -> f64
func.func private @wall.energy_forces_virial.kernel1(vector<3xf64>, f64)
    -> vector<3xf64>
func.func private @wall.energy_forces_virial.kernel2(vector<3xf64>, f64)
    -> vector<9xf64>

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
  %x = arith.constant dense<[0.4, -0.7, 1.3]> : vector<3xf64>
  %a = arith.constant 2.5 : f64

  // Energy.
  // CHECK: 1
  %u = call @wall.energy_forces_virial.kernel0(%x, %a)
      : (vector<3xf64>, f64) -> f64
  %u_ref = arith.constant 3.9144183423086507 : f64
  call @check(%u, %u_ref) : (f64, f64) -> ()

  // Force.
  // CHECK-NEXT: 1
  %f = call @wall.energy_forces_virial.kernel1(%x, %a)
      : (vector<3xf64>, f64) -> vector<3xf64>
  %f_ref = arith.constant dense<[0.8289390059971149, -1.0, -6.5]> : vector<3xf64>
  call @check3(%f, %f_ref) : (vector<3xf64>, vector<3xf64>) -> ()

  // Virial.
  // CHECK-NEXT: 1
  %w = call @wall.energy_forces_virial.kernel2(%x, %a)
      : (vector<3xf64>, f64) -> vector<9xf64>
  %w_ref = arith.constant dense<[
      0.331575602398846, -0.4, -2.6, -0.5802573041979804, 0.7, 4.55, 1.0776207077962494, -1.3, -8.450000000000001]> : vector<9xf64>
  call @check9(%w, %w_ref) : (vector<9xf64>, vector<9xf64>) -> ()
  return
}
