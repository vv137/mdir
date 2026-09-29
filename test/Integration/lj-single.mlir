// The system of lj-dynamics.mlir in single precision: the state is held by
// buffers of f32, and the kernels compute in f32. Global sums are
// accumulated in f64.
//
// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=single" \
// RUN:     --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// With the kernels rewritten in powers of the squared distance.
//
// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" \
// RUN:     %md_exec_fast_transforms \
// RUN:     --md-exec-assign-precision="mode=single" \
// RUN:     --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=single" \
// RUN:     --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \
// RUN: | FileCheck %s

// The reference values are those of lj-dynamics.mlir, which come from
// Inputs/lj_reference.py and are computed in double precision. Each check
// prints 1 if the value agrees with the reference to a relative tolerance
// of 1e-5, or 0 if not.

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
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

dyn.program @velocity_verlet(%x: !vec, %v: !vec, %f: !vec, %m: !real,
                             %cell: !md.cell, %dt: f64,
                             %eps: f64, %sigma: f64) -> (!vec, !vec, !vec)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %c    = arith.constant 0.5 : f64
  %half = arith.mulf %c, %dt : f64
  %v1 = dyn.kick %v, %f, %m, %half : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  %f1 = md.evaluate @lj(%x1, %cell, %eps, %sigma) request [forces]
      : (!vec, !md.cell, f64, f64) -> !vec
  %v2 = dyn.kick %v1, %f1, %m, %half : !vec
  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64) {
  %tolerance = arith.constant 1.0e-5 : f64
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %magnitude = math.absf %reference : f64
  %bound = arith.mulf %tolerance, %magnitude : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

// Places the particles. See lj-forces.mlir.
func.func @place(%x: memref<?x3xf32>) {
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
  %n = memref.dim %x, %c0 : memref<?x3xf32>
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
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
      %stored = arith.truncf %value : f64 to f32
      memref.store %stored, %x[%i, %k] : memref<?x3xf32>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

// Sets every component of every particle to `value`.
func.func @fill(%x: memref<?x3xf32>, %value: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = memref.dim %x, %c0 : memref<?x3xf32>
  scf.for %i = %c0 to %n step %c1 {
    scf.for %k = %c0 to %c3 step %c1 {
      memref.store %value, %x[%i, %k] : memref<?x3xf32>
    }
  }
  return
}

func.func @fill_scalar(%x: memref<?xf32>, %value: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %x, %c0 : memref<?xf32>
  scf.for %i = %c0 to %n step %c1 {
    memref.store %value, %x[%i] : memref<?xf32>
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %count = arith.constant 64 : index
  %steps = arith.constant 200 : index

  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %edge = arith.constant 4.8 : f64
  %eps = arith.constant 1.0 : f64
  %sigma = arith.constant 1.0 : f64
  %dt = arith.constant 0.004 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  // The buffers state the types that the fields are stored in. The fields
  // have the reference precision until the precision is assigned.
  %masses = memref.alloc(%count) : memref<?xf32>
  %positions = memref.alloc(%count) : memref<?x3xf32>
  %velocities = memref.alloc(%count) : memref<?x3xf32>
  call @fill_scalar(%masses, %one) : (memref<?xf32>, f32) -> ()
  call @place(%positions) : (memref<?x3xf32>) -> ()
  call @fill(%velocities, %zero) : (memref<?x3xf32>, f32) -> ()
  %m = mdrt.from_buffer %masses : memref<?xf32> to !real
  %x0 = mdrt.from_buffer %positions : memref<?x3xf32> to !vec
  %v0 = mdrt.from_buffer %velocities : memref<?x3xf32> to !vec

  %u0, %f0 = md.evaluate @lj(%x0, %cell, %eps, %sigma)
      request [energy, forces]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec)

  // The energy at the start.
  // CHECK: 1
  %u0_ref = arith.constant -215.19706025545707 : f64
  call @check(%u0, %u0_ref) : (f64, f64) -> ()

  %x, %v, %f = scf.for %step = %c0 to %steps step %c1
      iter_args(%xa = %x0, %va = %v0, %fa = %f0) -> (!vec, !vec, !vec) {
    %xb, %vb, %fb = dyn.step @velocity_verlet(
        %xa, %va, %fa, %m, %cell, %dt, %eps, %sigma)
        : (!vec, !vec, !vec, !real, !md.cell, f64, f64, f64)
        -> (!vec, !vec, !vec)
    scf.yield %xb, %vb, %fb : !vec, !vec, !vec
  }

  %u = md.evaluate @lj(%x, %cell, %eps, %sigma) request [energy]
      : (!vec, !md.cell, f64, f64) -> f64
  %k = md.sum_particles gather(%v, %m : !vec, !real) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %half = arith.constant 0.5 : f64
    %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
    %mv2  = arith.mulf %m_i, %v2 : f64
    %ke   = arith.mulf %half, %mv2 : f64
    md.yield %ke : f64
  } : f64
  %total = arith.addf %u, %k : f64

  // The potential, kinetic, and total energy after 200 steps.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %u_ref = arith.constant -221.98158745232965 : f64
  %k_ref = arith.constant 6.781805602672273 : f64
  %total_ref = arith.constant -215.19978184965737 : f64
  call @check(%u, %u_ref) : (f64, f64) -> ()
  call @check(%k, %k_ref) : (f64, f64) -> ()
  call @check(%total, %total_ref) : (f64, f64) -> ()

  // The x coordinate of particle 5.
  // CHECK-NEXT: 1
  %final = mdrt.to_buffer %x : !vec to memref<?x3xf32>
  %stored = memref.load %final[%c5, %c0] : memref<?x3xf32>
  %x5 = arith.extf %stored : f32 to f64
  %x5_ref = arith.constant 1.2295663551964633 : f64
  call @check(%x5, %x5_ref) : (f64, f64) -> ()
  return
}
