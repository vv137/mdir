// The three-body term of Stillinger and Weber [StillingerWeber1985],
//
//   E = sum over (j, i, k) of lambda (cos theta_jik + 1/3)^2
//         exp(gamma sigma / (r_ij - a sigma)) exp(gamma sigma / (r_ik - a sigma)),
//
// over the triplets of md.triplets (D160) at the cutoff a sigma, on eight
// particles in a periodic cell, compiled and run on the CPU and compared
// with Inputs/generate_triplets_tests.py, which generates this file. The
// neighborhood reaches farther than the triplets, so that the legs between
// the two cutoffs (0-3, 2-3) are left out by md.triplets; the far leg 1-2
// is beyond the cutoff and its triplet is kept. 6 triplets.
//
// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \
// RUN: | FileCheck %s
// The energy to a relative tolerance of 1.0e-12, and each component of the
// forces and of the virial to 1.0e-9 of the largest: each check prints 1 if
// it agrees, or 0 if it does not.

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
!trip  = !md.relation<@atoms, 3, reversal>

md.particle_set @atoms

md.potential @three_body(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) : !vec -> !pairs
  %t = md.triplets %n cutoff(1.19625) : !pairs -> !trip
  %u = md.sum_tuples %t, %x, %cell
         coordinates(distance(0, 1), distance(2, 1), cosine(0, 1, 2)) {
  ^bb0(%r1: f64, %r2: f64, %c: f64):
    %lambda = arith.constant 3.0 : f64
    %gamma  = arith.constant 1.2 : f64
    %sigma  = arith.constant 0.6645833333333333 : f64
    %a      = arith.constant 1.8 : f64
    %third  = arith.constant 0.3333333333333333 : f64
    %as     = arith.mulf %a, %sigma : f64
    %gs     = arith.mulf %gamma, %sigma : f64
    %t1     = arith.addf %c, %third : f64
    %t2     = arith.mulf %t1, %t1 : f64
    %d1     = arith.subf %r1, %as : f64
    %q1     = arith.divf %gs, %d1 : f64
    %e1     = math.exp %q1 : f64
    %d2     = arith.subf %r2, %as : f64
    %q2     = arith.divf %gs, %d2 : f64
    %e2     = math.exp %q2 : f64
    %l2     = arith.mulf %lambda, %t2 : f64
    %l3     = arith.mulf %l2, %e1 : f64
    %e      = arith.mulf %l3, %e2 : f64
    md.yield %e : f64
  } : !trip, !vec -> f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64, %magnitude: f64,
                 %tolerance: f64) {
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %scale = math.absf %magnitude : f64
  %bound = arith.mulf %tolerance, %scale : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

memref.global "private" constant @positions : memref<8x3xf64> =
    dense<[[0.1, 2.0, 2.0], [3.3, 2.02, 1.97], [0.9, 2.3, 2.05], [0.12, 3.3, 2.01], [0.3, 2.1, 3.0], [2.0, 0.5, 0.5], [2.4, 0.52, 1.0], [3.5962499999999995, 0.52, 1.0]]>

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %xs = memref.get_global @positions : memref<8x3xf64>
  %xd = memref.cast %xs : memref<8x3xf64> to memref<?x3xf64>
  %n = memref.dim %xd, %c0 : memref<?x3xf64>
  %buffer = memref.alloc(%n) : memref<?x3xf64>
  memref.copy %xd, %buffer : memref<?x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %buffer : memref<?x3xf64> to !vec
  %edge = arith.constant 4.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @three_body(%x, %cell)
      request [energy, forces, virial]
      : (!vec, !md.cell) -> (f64, !vec, vector<9xf64>)
  %forces = mdrt.to_buffer %f : !vec to memref<?x3xf64>
  %tol_e = arith.constant 1.0e-12 : f64
  %tol_f = arith.constant 1.0e-9 : f64

  // Energy.
  // CHECK: 1
  %u_ref = arith.constant 0.01458330792500454 : f64
  call @check(%u, %u_ref, %u_ref, %tol_e) : (f64, f64, f64, f64) -> ()

  // Every component of the forces, on the scale of the largest.
  // CHECK-COUNT-24: 1
  %f_scale = arith.constant 0.08677196651974638 : f64
  %i0_0 = arith.constant 0 : index
  %f0_0 = memref.load %forces[%i0_0, %c0] : memref<?x3xf64>
  %r0_0 = arith.constant -0.02160667808360218 : f64
  call @check(%f0_0, %r0_0, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i0_1 = arith.constant 0 : index
  %f0_1 = memref.load %forces[%i0_1, %c1] : memref<?x3xf64>
  %r0_1 = arith.constant -0.07828607728882678 : f64
  call @check(%f0_1, %r0_1, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i0_2 = arith.constant 0 : index
  %f0_2 = memref.load %forces[%i0_2, %c2] : memref<?x3xf64>
  %r0_2 = arith.constant -0.02928819392520833 : f64
  call @check(%f0_2, %r0_2, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i1_0 = arith.constant 1 : index
  %f1_0 = memref.load %forces[%i1_0, %c0] : memref<?x3xf64>
  %r1_0 = arith.constant -0.0686064421796032 : f64
  call @check(%f1_0, %r1_0, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i1_1 = arith.constant 1 : index
  %f1_1 = memref.load %forces[%i1_1, %c1] : memref<?x3xf64>
  %r1_1 = arith.constant 0.022879323842321553 : f64
  call @check(%f1_1, %r1_1, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i1_2 = arith.constant 1 : index
  %f1_2 = memref.load %forces[%i1_2, %c2] : memref<?x3xf64>
  %r1_2 = arith.constant -0.002207429831591867 : f64
  call @check(%f1_2, %r1_2, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i2_0 = arith.constant 2 : index
  %f2_0 = memref.load %forces[%i2_0, %c0] : memref<?x3xf64>
  %r2_0 = arith.constant 0.08677196651974638 : f64
  call @check(%f2_0, %r2_0, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i2_1 = arith.constant 2 : index
  %f2_1 = memref.load %forces[%i2_1, %c1] : memref<?x3xf64>
  %r2_1 = arith.constant 0.05370947478538992 : f64
  call @check(%f2_1, %r2_1, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i2_2 = arith.constant 2 : index
  %f2_2 = memref.load %forces[%i2_2, %c2] : memref<?x3xf64>
  %r2_2 = arith.constant 0.0029547945303347845 : f64
  call @check(%f2_2, %r2_2, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i3_0 = arith.constant 3 : index
  %f3_0 = memref.load %forces[%i3_0, %c0] : memref<?x3xf64>
  %r3_0 = arith.constant -0.0 : f64
  call @check(%f3_0, %r3_0, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i3_1 = arith.constant 3 : index
  %f3_1 = memref.load %forces[%i3_1, %c1] : memref<?x3xf64>
  %r3_1 = arith.constant -0.0 : f64
  call @check(%f3_1, %r3_1, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i3_2 = arith.constant 3 : index
  %f3_2 = memref.load %forces[%i3_2, %c2] : memref<?x3xf64>
  %r3_2 = arith.constant -0.0 : f64
  call @check(%f3_2, %r3_2, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i4_0 = arith.constant 4 : index
  %f4_0 = memref.load %forces[%i4_0, %c0] : memref<?x3xf64>
  %r4_0 = arith.constant 0.003441153743459003 : f64
  call @check(%f4_0, %r4_0, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i4_1 = arith.constant 4 : index
  %f4_1 = memref.load %forces[%i4_1, %c1] : memref<?x3xf64>
  %r4_1 = arith.constant 0.001697278661115303 : f64
  call @check(%f4_1, %r4_1, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i4_2 = arith.constant 4 : index
  %f4_2 = memref.load %forces[%i4_2, %c2] : memref<?x3xf64>
  %r4_2 = arith.constant 0.028540829226465416 : f64
  call @check(%f4_2, %r4_2, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i5_0 = arith.constant 5 : index
  %f5_0 = memref.load %forces[%i5_0, %c0] : memref<?x3xf64>
  %r5_0 = arith.constant -0.0 : f64
  call @check(%f5_0, %r5_0, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i5_1 = arith.constant 5 : index
  %f5_1 = memref.load %forces[%i5_1, %c1] : memref<?x3xf64>
  %r5_1 = arith.constant -0.0 : f64
  call @check(%f5_1, %r5_1, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i5_2 = arith.constant 5 : index
  %f5_2 = memref.load %forces[%i5_2, %c2] : memref<?x3xf64>
  %r5_2 = arith.constant -0.0 : f64
  call @check(%f5_2, %r5_2, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i6_0 = arith.constant 6 : index
  %f6_0 = memref.load %forces[%i6_0, %c0] : memref<?x3xf64>
  %r6_0 = arith.constant -0.0 : f64
  call @check(%f6_0, %r6_0, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i6_1 = arith.constant 6 : index
  %f6_1 = memref.load %forces[%i6_1, %c1] : memref<?x3xf64>
  %r6_1 = arith.constant -0.0 : f64
  call @check(%f6_1, %r6_1, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i6_2 = arith.constant 6 : index
  %f6_2 = memref.load %forces[%i6_2, %c2] : memref<?x3xf64>
  %r6_2 = arith.constant -0.0 : f64
  call @check(%f6_2, %r6_2, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i7_0 = arith.constant 7 : index
  %f7_0 = memref.load %forces[%i7_0, %c0] : memref<?x3xf64>
  %r7_0 = arith.constant -0.0 : f64
  call @check(%f7_0, %r7_0, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i7_1 = arith.constant 7 : index
  %f7_1 = memref.load %forces[%i7_1, %c1] : memref<?x3xf64>
  %r7_1 = arith.constant -0.0 : f64
  call @check(%f7_1, %r7_1, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %i7_2 = arith.constant 7 : index
  %f7_2 = memref.load %forces[%i7_2, %c2] : memref<?x3xf64>
  %r7_2 = arith.constant -0.0 : f64
  call @check(%f7_2, %r7_2, %f_scale, %tol_f) : (f64, f64, f64, f64) -> ()

  // Every component of the virial, on the scale of the largest.
  // CHECK-COUNT-9: 1
  %w_scale = arith.constant 0.12499095770817148 : f64
  %w0 = vector.extract %w[0] : f64 from vector<9xf64>
  %w0_ref = arith.constant 0.12499095770817148 : f64
  call @check(%w0, %w0_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %w1 = vector.extract %w[1] : f64 from vector<9xf64>
  %w1_ref = arith.constant 0.025003576486677753 : f64
  call @check(%w1, %w1_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %w2 = vector.extract %w[2] : f64 from vector<9xf64>
  %w2_ref = arith.constant 0.009837945334834407 : f64
  call @check(%w2, %w2_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %w3 = vector.extract %w[3] : f64 from vector<9xf64>
  %w3_ref = arith.constant 0.025003576486677746 : f64
  call @check(%w3, %w3_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %w4 = vector.extract %w[4] : f64 from vector<9xf64>
  %w4_ref = arith.constant 0.016740156778574926 : f64
  call @check(%w4, %w4_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %w5 = vector.extract %w[5] : f64 from vector<9xf64>
  %w5_ref = arith.constant 0.0036963726851151416 : f64
  call @check(%w5, %w5_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %w6 = vector.extract %w[6] : f64 from vector<9xf64>
  %w6_ref = arith.constant 0.009837945334834404 : f64
  call @check(%w6, %w6_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %w7 = vector.extract %w[7] : f64 from vector<9xf64>
  %w7_ref = arith.constant 0.003696372685115142 : f64
  call @check(%w7, %w7_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  %w8 = vector.extract %w[8] : f64 from vector<9xf64>
  %w8_ref = arith.constant 0.02875479184792991 : f64
  call @check(%w8, %w8_ref, %w_scale, %tol_f) : (f64, f64, f64, f64) -> ()
  // CHECK-NOT: 0
  memref.dealloc %buffer : memref<?x3xf64>
  return
}
