// The tangent of a potential in the positions and its mixed derivative
// (D[force-matching], #726), compiled and run on the CPU and checked by
// central differences along the tangent: `tangent` against differences of
// the energy, `tangent_derivative(n)` against differences of
// `derivative(n)` (the oracle of md-check-derivatives, which aborts with a
// message on a value outside the tolerance).
//
// RUN: mdir-opt %s --md-check-derivative-coverage \
// RUN:     --md-check-derivatives="step=1e-5 atol=1e-9 rtol=1e-6" %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// The same without the check, whose printed values are those that the
// checked run printed.
//
// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// Six particles in a periodic cube with the edge 3.0, a pair across the
// boundary among them; a Lennard-Jones and Coulomb pair term with a switch,
// charges as a field; a bond, an angle, and a dihedral on particles 0 to 3;
// and a harmonic term of the absolute positions.

!vec       = !md.field<@atoms, 3 x f64>
!real      = !md.field<@atoms, f64>
!pairs     = !md.relation<@atoms, 2, unordered>
!bonds     = !md.relation<@atoms, 2, unordered, @bonds>
!angles    = !md.relation<@atoms, 3, reversal, @angles>
!dihedrals = !md.relation<@atoms, 4, reversal, @dihedrals>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
md.tuple_set @dihedrals on(@atoms) arity(4) orientation(reversal)

md.potential @model(%x: !vec, %cell: !md.cell, %eps: f64, %q: !real,
                    %kd: f64, %c: f64, %bonds: !bonds, %angles: !angles,
                    %dihedrals: !dihedrals) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.4) : !vec -> !pairs
  %up = md.sum_relation %n, %x, %cell gather(%q : !real)
          exchange(symmetric) truncation(switch, from = 1.1) {
  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
    %c4 = arith.constant 4.0 : f64
    %sigma = arith.constant 0.35 : f64
    %i6 = arith.constant 6 : i32
    %sr = arith.divf %sigma, %r : f64
    %s6 = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t = arith.subf %s12, %s6 : f64
    %e4 = arith.mulf %c4, %eps : f64
    %lj = arith.mulf %e4, %t : f64
    %qq = arith.mulf %qi, %qj : f64
    %coulomb = arith.divf %qq, %r : f64
    %k = arith.addf %lj, %coulomb : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %ub = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    %half = arith.constant 50.0 : f64
    %r0 = arith.constant 0.6 : f64
    %dr = arith.subf %r, %r0 : f64
    %dr2 = arith.mulf %dr, %dr : f64
    %e = arith.mulf %half, %dr2 : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64
  %ua = md.sum_tuples %angles, %x, %cell coordinates(cosine(0, 1, 2))
          gather(%q : !real) {
  ^bb0(%cs: f64, %q0: f64, %q1: f64, %q2: f64):
    %k = arith.constant 20.0 : f64
    %c0 = arith.constant -0.3 : f64
    %dc = arith.subf %cs, %c0 : f64
    %dc2 = arith.mulf %dc, %dc : f64
    %e0 = arith.mulf %k, %dc2 : f64
    %qq = arith.mulf %q0, %q2 : f64
    %e = arith.mulf %e0, %qq : f64
    md.yield %e : f64
  } : !angles, !vec -> f64
  %ud = md.sum_tuples %dihedrals, %x, %cell coordinates(dihedral(0, 1, 2, 3)) {
  ^bb0(%phi: f64):
    %one = arith.constant 1.0 : f64
    %three = arith.constant 3.0 : f64
    %phase = arith.constant 0.4 : f64
    %p3 = arith.mulf %three, %phi : f64
    %a = arith.subf %p3, %phase : f64
    %cos = math.cos %a : f64
    %s = arith.addf %one, %cos : f64
    %e = arith.mulf %kd, %s : f64
    md.yield %e : f64
  } : !dihedrals, !vec -> f64
  %ux = md.sum_particles gather(%x, %q : !vec, !real) {
  ^bb0(%xi: vector<3xf64>, %qi: f64):
    %x2 = arith.mulf %xi, %xi : vector<3xf64>
    %s = vector.reduction <add>, %x2 : vector<3xf64> into f64
    %cq = arith.mulf %c, %qi : f64
    %e = arith.mulf %cq, %s : f64
    md.yield %e : f64
  } : f64
  // A term that does not read the positions has no tangent.
  %uq = md.sum_particles gather(%q : !real) {
  ^bb0(%qi: f64):
    %e = arith.mulf %qi, %qi : f64
    md.yield %e : f64
  } : f64
  %u1 = arith.addf %up, %ub : f64
  %u2 = arith.addf %u1, %ua : f64
  %u3 = arith.addf %u2, %ud : f64
  %u4 = arith.addf %u3, %ux : f64
  %u5 = arith.addf %u4, %uq : f64
  %two = arith.constant 2.0 : f64
  %u = arith.mulf %two, %u5 : f64
  md.return %u : f64
}

memref.global "private" constant @positions : memref<6x3xf64> = dense<[
  [0.30, 0.40, 0.50], [0.85, 0.55, 0.45], [1.10, 1.05, 0.70],
  [1.65, 1.20, 1.15], [2.90, 0.45, 0.60], [1.20, 0.20, 1.40]]>
memref.global "private" constant @tangents : memref<6x3xf64> = dense<[
  [0.7, -1.2, 0.4], [-0.3, 0.9, 1.1], [1.5, 0.2, -0.8],
  [-1.0, -0.6, 0.3], [0.5, 1.3, -0.2], [-0.4, 0.1, 0.9]]>
memref.global "private" constant @charges : memref<6xf64> = dense<
  [0.4, -0.7, 0.3, 0.5, -0.6, 0.2]>
memref.global "private" constant @bonds_members : memref<3x2xi32> = dense<
  [[0, 1], [1, 2], [2, 3]]>
memref.global "private" constant @angles_members : memref<2x3xi32> = dense<
  [[0, 1, 2], [1, 2, 3]]>
memref.global "private" constant @dihedrals_members : memref<1x4xi32> = dense<
  [[0, 1, 2, 3]]>

func.func private @printF64(f64)
func.func private @printNewline()

func.func @main() {
  %xs = memref.get_global @positions : memref<6x3xf64>
  %xd = memref.cast %xs : memref<6x3xf64> to memref<?x3xf64>
  %c0 = arith.constant 0 : index
  %n = memref.dim %xd, %c0 : memref<?x3xf64>
  %xb = memref.alloc(%n) : memref<?x3xf64>
  memref.copy %xd, %xb : memref<?x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
  %ts = memref.get_global @tangents : memref<6x3xf64>
  %td = memref.cast %ts : memref<6x3xf64> to memref<?x3xf64>
  %tb = memref.alloc(%n) : memref<?x3xf64>
  memref.copy %td, %tb : memref<?x3xf64> to memref<?x3xf64>
  %t = mdrt.from_buffer %tb : memref<?x3xf64> to !vec
  %qs = memref.get_global @charges : memref<6xf64>
  %qd = memref.cast %qs : memref<6xf64> to memref<?xf64>
  %qb = memref.alloc(%n) : memref<?xf64>
  memref.copy %qd, %qb : memref<?xf64> to memref<?xf64>
  %q = mdrt.from_buffer %qb : memref<?xf64> to !real
  %bs = memref.get_global @bonds_members : memref<3x2xi32>
  %bd = memref.cast %bs : memref<3x2xi32> to memref<?x2xi32>
  %bonds = mdrt.from_buffer %bd : memref<?x2xi32> to !bonds
  %as = memref.get_global @angles_members : memref<2x3xi32>
  %ad = memref.cast %as : memref<2x3xi32> to memref<?x3xi32>
  %angles = mdrt.from_buffer %ad : memref<?x3xi32> to !angles
  %ds = memref.get_global @dihedrals_members : memref<1x4xi32>
  %dd = memref.cast %ds : memref<1x4xi32> to memref<?x4xi32>
  %dihedrals = mdrt.from_buffer %dd : memref<?x4xi32> to !dihedrals
  %edge = arith.constant 3.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %eps = arith.constant 0.8 : f64
  %kd = arith.constant 2.5 : f64
  %c = arith.constant 0.3 : f64

  %u, %f, %p, %pe, %pk, %pc, %pq = md.evaluate @model(%x, %cell, %eps, %q,
        %kd, %c, %bonds, %angles, %dihedrals, %t)
      request [energy, forces, tangent, tangent_derivative(2),
               tangent_derivative(4), tangent_derivative(5),
               tangent_derivative(3)]
      : (!vec, !md.cell, f64, !real, f64, f64, !bonds, !angles, !dihedrals,
         !vec) -> (f64, !vec, f64, f64, f64, f64, !real)

  // Φ = Σ t_i · ∂U/∂x_i = −Σ t_i · F_i, from the forces of the same call.
  %forces = mdrt.to_buffer %f : !vec to memref<?x3xf64>
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %dot = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (f64) {
    %inner = scf.for %k = %c0 to %c3 step %c1 iter_args(%a = %acc) -> (f64) {
      %fk = memref.load %forces[%i, %k] : memref<?x3xf64>
      %tk = memref.load %tb[%i, %k] : memref<?x3xf64>
      %m = arith.mulf %fk, %tk : f64
      %s = arith.subf %a, %m : f64
      scf.yield %s : f64
    }
    scf.yield %inner : f64
  }
  // The two agree to the rounding: the flag is 1.
  // CHECK: 1
  %diff = arith.subf %p, %dot : f64
  %err = math.absf %diff : f64
  %scale = math.absf %dot : f64
  %tol = arith.constant 1.0e-12 : f64
  %bound = arith.mulf %tol, %scale : f64
  %ok = arith.cmpf ole, %err, %bound : f64
  %flag = arith.uitofp %ok : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()

  // The tangent of U is not zero, nor is the mixed derivative in each
  // argument: printed, to fail if a term is lost.
  // CHECK-NEXT: {{-?[1-9][0-9.e+-]*}}
  call @printF64(%p) : (f64) -> ()
  call @printNewline() : () -> ()
  // CHECK-NEXT: {{-?[1-9][0-9.e+-]*}}
  call @printF64(%pe) : (f64) -> ()
  call @printNewline() : () -> ()
  // CHECK-NEXT: {{-?[1-9][0-9.e+-]*}}
  call @printF64(%pk) : (f64) -> ()
  call @printNewline() : () -> ()
  // CHECK-NEXT: {{-?[1-9][0-9.e+-]*}}
  call @printF64(%pc) : (f64) -> ()
  call @printNewline() : () -> ()
  %pqs = mdrt.to_buffer %pq : !real to memref<?xf64>
  %pq0 = memref.load %pqs[%c0] : memref<?xf64>
  // CHECK-NEXT: {{-?[1-9][0-9.e+-]*}}
  call @printF64(%pq0) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
