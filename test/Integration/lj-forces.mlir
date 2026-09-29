// Energy and forces of a Lennard-Jones system, compiled and run, compared
// with a brute-force evaluation over all pairs.
//
// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" \
// RUN:     --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" \
// RUN:     --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \
// RUN: | FileCheck %s

// The system: 64 particles on a cubic lattice with the spacing 1.2, each
// displaced by up to 0.1 along every axis, in a periodic cube with the edge
// 4.8. The potential is Lennard-Jones with eps = 1 and sigma = 1, switched
// off between 1.6 and 2.0.
//
// The reference values come from Inputs/lj_reference.py, which evaluates the
// same system over all pairs. Each check prints the value and then 1 if it
// agrees with the reference to a relative tolerance of 1e-10, or 0 if not.

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(switch, from = 1.6) {
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

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64) {
  %tolerance = arith.constant 1.0e-10 : f64
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

// Places the particles. The displacements come from a linear congruential
// generator, which the reference script repeats.
func.func @place(%x: memref<?x3xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %spacing = arith.constant 1.2 : f64
  %amplitude = arith.constant 0.2 : f64
  %half = arith.constant 0.5 : f64
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 42 : i64
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      // The lattice site along axis k: digit k of i in base 4.
      %p0 = arith.cmpi eq, %k, %c0 : index
      %p1 = arith.cmpi eq, %k, %c1 : index
      %d1 = arith.divui %i, %c4 : index
      %d2 = arith.divui %d1, %c4 : index
      %q01 = arith.select %p1, %d1, %d2 : index
      %q = arith.select %p0, %i, %q01 : index
      %digit = arith.remui %q, %c4 : index
      %wide = arith.index_cast %digit : index to i64
      %site = arith.sitofp %wide : i64 to f64
      %base = arith.mulf %site, %spacing : f64

      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %centered = arith.subf %f, %half : f64
      %shift = arith.mulf %centered, %amplitude : f64
      %value = arith.addf %base, %shift : f64
      memref.store %value, %x[%i, %k] : memref<?x3xf64>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c17 = arith.constant 17 : index
  %count = arith.constant 64 : index
  %buffer = memref.alloc(%count) : memref<?x3xf64>
  call @place(%buffer) : (memref<?x3xf64>) -> ()

  %edge = arith.constant 4.8 : f64
  %eps = arith.constant 1.0 : f64
  %sigma = arith.constant 1.0 : f64
  %x = mdrt.from_buffer %buffer : memref<?x3xf64> to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  %u, %f = md.evaluate @lj(%x, %cell, %eps, %sigma) request [energy, forces]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec)
  %forces = mdrt.to_buffer %f : !vec to memref<?x3xf64>

  // Energy.
  // CHECK:      -215.197
  // CHECK-NEXT: 1
  %u_ref = arith.constant -215.19706025545707 : f64
  call @check(%u, %u_ref) : (f64, f64) -> ()

  // The force on particle 0.
  // CHECK-NEXT: 0.166889
  // CHECK-NEXT: 1
  // CHECK-NEXT: -1.13451
  // CHECK-NEXT: 1
  // CHECK-NEXT: -0.643662
  // CHECK-NEXT: 1
  %f0x = memref.load %forces[%c0, %c0] : memref<?x3xf64>
  %f0y = memref.load %forces[%c0, %c1] : memref<?x3xf64>
  %f0z = memref.load %forces[%c0, %c2] : memref<?x3xf64>
  %f0x_ref = arith.constant 0.1668894181446261 : f64
  %f0y_ref = arith.constant -1.1345108686059897 : f64
  %f0z_ref = arith.constant -0.6436619151093924 : f64
  call @check(%f0x, %f0x_ref) : (f64, f64) -> ()
  call @check(%f0y, %f0y_ref) : (f64, f64) -> ()
  call @check(%f0z, %f0z_ref) : (f64, f64) -> ()

  // The x component of the force on particle 17.
  // CHECK-NEXT: -1.81946
  // CHECK-NEXT: 1
  %f17x = memref.load %forces[%c17, %c0] : memref<?x3xf64>
  %f17x_ref = arith.constant -1.8194604580377653 : f64
  call @check(%f17x, %f17x_ref) : (f64, f64) -> ()

  // The sum over all particles of the squared force.
  // CHECK-NEXT: 3245.68
  // CHECK-NEXT: 1
  %zero = arith.constant 0.0 : f64
  %f2 = scf.for %i = %c0 to %count step %c1 iter_args(%acc = %zero) -> (f64) {
    %fx = memref.load %forces[%i, %c0] : memref<?x3xf64>
    %fy = memref.load %forces[%i, %c1] : memref<?x3xf64>
    %fz = memref.load %forces[%i, %c2] : memref<?x3xf64>
    %xx = arith.mulf %fx, %fx : f64
    %yy = arith.mulf %fy, %fy : f64
    %zz = arith.mulf %fz, %fz : f64
    %xy = arith.addf %xx, %yy : f64
    %sq = arith.addf %xy, %zz : f64
    %next = arith.addf %acc, %sq : f64
    scf.yield %next : f64
  }
  %f2_ref = arith.constant 3245.683825884532 : f64
  call @check(%f2, %f2_ref) : (f64, f64) -> ()
  return
}
