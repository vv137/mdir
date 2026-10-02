// Runs the kernels that differentiation generates for a term over tuples
// in the components of a displacement, and compares their values with the
// closed form: with d = x0 - x1 and
//
//   u = k (dx + 2 dy - dz) + k dx^2 / 2,
//
// the gradient is g = k (1 + dx, 2, -1), the forces are -g on x0 and g on
// x1, and the virial is d ⊗ (-g).
//
// RUN: mdir-opt %s --md-differentiate --test-md-outline-kernels=erase-md \
// RUN: | mlir-opt %lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// The members of the tuple are at (0.1, 0.2, -0.3) and (1.0, 0.4, 0.2):
// d = (-0.9, -0.2, -0.5), k = 3, u = -1.185, g = (0.3, 6, -3).

!vec   = !md.field<@atoms, 3 x f64>
!links = !md.relation<@atoms, 2, ordered, @links>
!of_link = !md.field<@links, f64>

md.particle_set @atoms
md.tuple_set @links on(@atoms) arity(2) orientation(ordered)

md.potential @pull(%x: !vec, %cell: !md.cell, %links: !links,
                   %k: !of_link) -> f64 {
  %u = md.sum_tuples %links, %x, %cell coordinates(displacement(0, 1))
         tuple(%k : !of_link) {
  ^bb0(%d: vector<3xf64>, %k_t: f64):
    %dx = vector.extract %d[0] : f64 from vector<3xf64>
    %dy = vector.extract %d[1] : f64 from vector<3xf64>
    %dz = vector.extract %d[2] : f64 from vector<3xf64>
    %two = arith.constant 2.0 : f64
    %half = arith.constant 0.5 : f64
    %dy2 = arith.mulf %two, %dy : f64
    %s0 = arith.addf %dx, %dy2 : f64
    %s = arith.subf %s0, %dz : f64
    %lin = arith.mulf %k_t, %s : f64
    %dxx = arith.mulf %dx, %dx : f64
    %kh = arith.mulf %half, %k_t : f64
    %quad = arith.mulf %kh, %dxx : f64
    %e = arith.addf %lin, %quad : f64
    md.yield %e : f64
  } : !links, !vec -> f64
  md.return %u : f64
}

md.function @requests(%x: !vec, %cell: !md.cell, %links: !links,
                      %k: !of_link) -> (f64, !vec, vector<9xf64>) {
  %u, %f, %w = md.evaluate @pull(%x, %cell, %links, %k)
      request [energy, forces, virial]
      : (!vec, !md.cell, !links, !of_link) -> (f64, !vec, vector<9xf64>)
  md.return %u, %f, %w : f64, !vec, vector<9xf64>
}

func.func private @pull.energy_forces_virial.kernel0(
    vector<3xf64>, vector<3xf64>, f64) -> f64
func.func private @pull.energy_forces_virial.kernel1(
    vector<3xf64>, vector<3xf64>, f64) -> (vector<3xf64>, vector<3xf64>)
func.func private @pull.energy_forces_virial.kernel2(
    vector<3xf64>, vector<3xf64>, f64) -> vector<9xf64>

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
  %k = arith.constant 3.0 : f64

  // Energy.
  // CHECK: 1
  %u = call @pull.energy_forces_virial.kernel0(%x0, %x1, %k)
      : (vector<3xf64>, vector<3xf64>, f64) -> f64
  %u_ref = arith.constant -1.185 : f64
  call @check(%u, %u_ref) : (f64, f64) -> ()

  // Forces on the members.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %f0, %f1 = call @pull.energy_forces_virial.kernel1(%x0, %x1, %k)
      : (vector<3xf64>, vector<3xf64>, f64) -> (vector<3xf64>, vector<3xf64>)
  %f0_ref = arith.constant dense<[-0.3, -6.0, 3.0]> : vector<3xf64>
  call @check3(%f0, %f0_ref) : (vector<3xf64>, vector<3xf64>) -> ()
  %f1_ref = arith.constant dense<[0.3, 6.0, -3.0]> : vector<3xf64>
  call @check3(%f1, %f1_ref) : (vector<3xf64>, vector<3xf64>) -> ()

  // Virial.
  // CHECK-NEXT: 1
  %w = call @pull.energy_forces_virial.kernel2(%x0, %x1, %k)
      : (vector<3xf64>, vector<3xf64>, f64) -> vector<9xf64>
  %w_ref = arith.constant dense<[
      0.27, 5.4, -2.7,
      0.06, 1.2, -0.6,
      0.15, 3.0, -1.5]> : vector<9xf64>
  call @check9(%w, %w_ref) : (vector<9xf64>, vector<9xf64>) -> ()
  return
}
