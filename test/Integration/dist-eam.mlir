// EAM on several domains through the compiler (D279,
// docs/distributed-ml2.md, Section 15): an embedded-atom potential,
// distributed by md-distribute and differentiated by md-differentiate
// through its density and embedding, against the same potential on one
// domain and against central differences of the distributed energy.
//
// The derivative copies the density's adjoint, F'(ρ), to the ghosts
// between the two passes, and the forces of the density's gather return to
// the owners by G^T with those of the pair term. Each grid of dist-lj.mlir
// (one domain whose ghosts are its own images; two; 3x2, uneven; 16, half
// of them empty) runs with the unique traversal, with the directed
// traversal in the derivative's form, and with the directed traversal
// after md-dist-owner-computes.
//
// RUN: mdir-opt %s --md-distribute="grid=1,1,1 traversal=unique potentials=eam,linear" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=1,1,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=1,1,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=unique potentials=eam,linear" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 traversal=unique potentials=eam,linear" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=16,1,1 traversal=unique potentials=eam,linear" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=16,1,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=16,1,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// The system of dist-lj.mlir: 64 particles near a cubic lattice of spacing
// 1.2 in a periodic cube of edge 4.8. The potential has the analytic form
// of Finnis and Sinclair [Finnis1984], with parameters chosen for this
// system in reduced units:
//
//   U = Σ_i F(ρ_i) + Σ_{i<j} φ(r_ij),  ρ_i = Σ_{j≠i} f(r_ij),
//   f(r) = (r − 2)²,  F(ρ) = −√ρ,  φ(r) = (r − 2)² (3 − 2r + 0.4r²),
//
// for r < 2, each zero with its slope at the cutoff.
//
// Each check prints 1 if it holds: energies to a relative 1e-10, every
// force component to 1e-10 of the largest component, every virial
// component to 1e-10 of the trace; central differences, with a step of
// 1e-5, of the distributed energy in six position components and in the
// three stretches of the cell against the distributed forces and the
// diagonal of the virial, to 1e-7 of the largest force component and of
// the trace; and the adjoint identity of the density, to a relative 1e-12.

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!buf   = memref<?x3xf64>

md.particle_set @atoms

md.potential @eam(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md.yield %f : f64
  } : !pairs, !vec -> !sca
  %e1 = md.sum_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    %m = arith.negf %s : f64
    md.yield %m : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %c0 = arith.constant 3.0 : f64
    %c1 = arith.constant -2.0 : f64
    %cq = arith.constant 0.4 : f64
    %t = arith.subf %r, %c2 : f64
    %t2 = arith.mulf %t, %t : f64
    %a = arith.mulf %c1, %r : f64
    %r2 = arith.mulf %r, %r : f64
    %b = arith.mulf %cq, %r2 : f64
    %p0 = arith.addf %c0, %a : f64
    %p = arith.addf %p0, %b : f64
    %k = arith.mulf %t2, %p : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
  md.return %u : f64
}

// The same, on one domain: the reference.
md.potential @eam_ref(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md.yield %f : f64
  } : !pairs, !vec -> !sca
  %e1 = md.sum_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    %m = arith.negf %s : f64
    md.yield %m : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %c0 = arith.constant 3.0 : f64
    %c1 = arith.constant -2.0 : f64
    %cq = arith.constant 0.4 : f64
    %t = arith.subf %r, %c2 : f64
    %t2 = arith.mulf %t, %t : f64
    %a = arith.mulf %c1, %r : f64
    %r2 = arith.mulf %r, %r : f64
    %b = arith.mulf %cq, %r2 : f64
    %p0 = arith.addf %c0, %a : f64
    %p = arith.addf %p0, %b : f64
    %k = arith.mulf %t2, %p : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
  md.return %u : f64
}

// U = <c, ρ(x)>: its forces are −J^T c, with J = ∂ρ/∂x, carried back
// through the copy of the density's adjoint, c, to the ghosts.
md.potential @linear(%x: !vec, %cell: !md.cell, %c: !sca) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md.yield %f : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%rho, %c : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

// Prints 1 if |value - reference| <= tolerance * |scale|, else 0.
func.func @check(%value: f64, %reference: f64, %scale: f64, %tolerance: f64) {
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %magnitude = math.absf %scale : f64
  %bound = arith.mulf %tolerance, %magnitude : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

// The largest |component| of `b`.
func.func @largest(%b: !buf) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %b, %c0 : !buf
  %m = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> (f64) {
    %r = scf.for %k = %c0 to %c3 step %c1 iter_args(%s1 = %s) -> (f64) {
      %vb = memref.load %b[%i, %k] : !buf
      %ab = math.absf %vb : f64
      %s2 = arith.maximumf %s1, %ab : f64
      scf.yield %s2 : f64
    }
    scf.yield %r : f64
  }
  return %m : f64
}

// Prints 1 if every component of `a` is within 1e-10 * max |b| of `b`.
func.func @check_forces(%a: !buf, %b: !buf) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %b, %c0 : !buf
  %e = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> (f64) {
    %r = scf.for %k = %c0 to %c3 step %c1 iter_args(%s1 = %s) -> (f64) {
      %va = memref.load %a[%i, %k] : !buf
      %vb = memref.load %b[%i, %k] : !buf
      %d = arith.subf %va, %vb : f64
      %ad = math.absf %d : f64
      %s2 = arith.maximumf %s1, %ad : f64
      scf.yield %s2 : f64
    }
    scf.yield %r : f64
  }
  %scale = call @largest(%b) : (!buf) -> f64
  %tolerance = arith.constant 1.0e-10 : f64
  call @check(%e, %zero, %scale, %tolerance) : (f64, f64, f64, f64) -> ()
  return
}

// Prints 1 if every component of `w` agrees with `ref` to 1e-10 of the
// trace of `ref`.
func.func @check_virial(%w: vector<9xf64>, %ref: vector<9xf64>) {
  %d = arith.subf %w, %ref : vector<9xf64>
  %ad = math.absf %d : vector<9xf64>
  %e = vector.reduction <maximumf>, %ad : vector<9xf64> into f64
  %xx = vector.extract %ref[0] : f64 from vector<9xf64>
  %yy = vector.extract %ref[4] : f64 from vector<9xf64>
  %zz = vector.extract %ref[8] : f64 from vector<9xf64>
  %t0 = arith.addf %xx, %yy : f64
  %trace = arith.addf %t0, %zz : f64
  %zero = arith.constant 0.0 : f64
  %tolerance = arith.constant 1.0e-10 : f64
  call @check(%e, %zero, %trace, %tolerance) : (f64, f64, f64, f64) -> ()
  return
}

// Places the particles, as lj-forces.mlir does.
func.func @place(%x: !buf) {
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
  %n = memref.dim %x, %c0 : !buf
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
      memref.store %value, %x[%i, %k] : !buf
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

// A copy of `x` that the program may take, with each component a scaled by
// scale[a].
func.func @copy(%x: !buf, %sx: f64, %sy: f64, %sz: f64) -> !buf {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %n = memref.dim %x, %c0 : !buf
  %y = memref.alloc(%n) : !buf
  scf.for %i = %c0 to %n step %c1 {
    %a = memref.load %x[%i, %c0] : !buf
    %b = memref.load %x[%i, %c1] : !buf
    %c = memref.load %x[%i, %c2] : !buf
    %a2 = arith.mulf %a, %sx : f64
    %b2 = arith.mulf %b, %sy : f64
    %c2v = arith.mulf %c, %sz : f64
    memref.store %a2, %y[%i, %c0] : !buf
    memref.store %b2, %y[%i, %c1] : !buf
    memref.store %c2v, %y[%i, %c2] : !buf
  }
  return %y : !buf
}

// c_i = 1 + (i mod 3) / 2 and v_i = ((i mod 5) − 2, 1 − (i mod 3), (i mod 7) / 3):
// fields that differ between particles.
func.func @coefficients(%n: index) -> (memref<?xf64>, !buf) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %three = arith.constant 3.0 : f64
  %half = arith.constant 0.5 : f64
  %cb = memref.alloc(%n) : memref<?xf64>
  %vb = memref.alloc(%n) : !buf
  scf.for %i = %c0 to %n step %c1 {
    %r3 = arith.remui %i, %c3 : index
    %r5 = arith.remui %i, %c5 : index
    %r7 = arith.remui %i, %c7 : index
    %i3 = arith.index_cast %r3 : index to i64
    %i5 = arith.index_cast %r5 : index to i64
    %i7 = arith.index_cast %r7 : index to i64
    %f3 = arith.sitofp %i3 : i64 to f64
    %f5 = arith.sitofp %i5 : i64 to f64
    %f7 = arith.sitofp %i7 : i64 to f64
    %h = arith.mulf %f3, %half : f64
    %ci = arith.addf %one, %h : f64
    memref.store %ci, %cb[%i] : memref<?xf64>
    %vx = arith.subf %f5, %two : f64
    %vy = arith.subf %one, %f3 : f64
    %vz = arith.divf %f7, %three : f64
    memref.store %vx, %vb[%i, %c0] : !buf
    memref.store %vy, %vb[%i, %c1] : !buf
    memref.store %vz, %vb[%i, %c2] : !buf
  }
  return %cb, %vb : memref<?xf64>, !buf
}

// <c, J v> = Σ_i c_i Σ_{j≠i} f'(r_ij) d_ij · (v_i − v_j) / r_ij, the
// derivative of <c, ρ> along v, summed directly over all ordered pairs in
// the minimum image of the cube: no relation, no domains, no derivative.
func.func @jvp(%xb: !buf, %edge: f64, %cb: memref<?xf64>, %vb: !buf) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %two = arith.constant 2.0 : f64
  %cut2 = arith.constant 4.0 : f64
  %n = memref.dim %xb, %c0 : !buf
  %total = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> (f64) {
    %ci = memref.load %cb[%i] : memref<?xf64>
    %si = scf.for %j = %c0 to %n step %c1 iter_args(%sj = %s) -> (f64) {
      %same = arith.cmpi eq, %i, %j : index
      %next = scf.if %same -> (f64) {
        scf.yield %sj : f64
      } else {
        %acc:2 = scf.for %k = %c0 to %c3 step %c1
            iter_args(%r2 = %zero, %dv = %zero) -> (f64, f64) {
          %xi = memref.load %xb[%i, %k] : !buf
          %xj = memref.load %xb[%j, %k] : !buf
          %dk = arith.subf %xi, %xj : f64
          %q = arith.divf %dk, %edge : f64
          %nq = math.roundeven %q : f64
          %sh = arith.mulf %nq, %edge : f64
          %d = arith.subf %dk, %sh : f64
          %vi = memref.load %vb[%i, %k] : !buf
          %vj = memref.load %vb[%j, %k] : !buf
          %w = arith.subf %vi, %vj : f64
          %dd = arith.mulf %d, %d : f64
          %r2n = arith.addf %r2, %dd : f64
          %dw = arith.mulf %d, %w : f64
          %dvn = arith.addf %dv, %dw : f64
          scf.yield %r2n, %dvn : f64, f64
        }
        %inside = arith.cmpf olt, %acc#0, %cut2 : f64
        %term = scf.if %inside -> (f64) {
          %r = math.sqrt %acc#0 : f64
          %t = arith.subf %r, %two : f64
          %fp = arith.mulf %two, %t : f64
          %a = arith.mulf %fp, %acc#1 : f64
          %b = arith.divf %a, %r : f64
          %e = arith.mulf %ci, %b : f64
          scf.yield %e : f64
        } else {
          scf.yield %zero : f64
        }
        %sn = arith.addf %sj, %term : f64
        scf.yield %sn : f64
      }
      scf.yield %next : f64
    }
    scf.yield %si : f64
  }
  return %total : f64
}

// <F, v> of the forces of U = <c, ρ> on domains: −<J^T c, v>.
func.func @vjp(%xb: !buf, %edge: f64, %cb: memref<?xf64>, %vb: !buf) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %y = call @copy(%xb, %one, %one, %one) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %cs = mdrt.from_buffer %cb : memref<?xf64> to !sca
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %f = md.evaluate @linear(%x, %cell, %cs) request [forces]
      : (!vec, !md.cell, !sca) -> !vec
  %fb = mdrt.to_buffer %f : !vec to !buf
  %n = memref.dim %xb, %c0 : !buf
  %total = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> (f64) {
    %si = scf.for %k = %c0 to %c3 step %c1 iter_args(%sk = %s) -> (f64) {
      %fv = memref.load %fb[%i, %k] : !buf
      %vv = memref.load %vb[%i, %k] : !buf
      %p = arith.mulf %fv, %vv : f64
      %sn = arith.addf %sk, %p : f64
      scf.yield %sn : f64
    }
    scf.yield %si : f64
  }
  return %total : f64
}

// The reference on one domain.
func.func @reference(%xb: !buf, %edge: f64) -> (f64, !buf, vector<9xf64>) {
  %one = arith.constant 1.0 : f64
  %y = call @copy(%xb, %one, %one, %one) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @eam_ref(%x, %cell)
      request [energy, forces, virial]
      : (!vec, !md.cell) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to !buf
  return %u, %fb, %w : f64, !buf, vector<9xf64>
}

// The potential on domains.
func.func @derived(%xb: !buf, %edge: f64) -> (f64, !buf, vector<9xf64>) {
  %one = arith.constant 1.0 : f64
  %y = call @copy(%xb, %one, %one, %one) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @eam(%x, %cell)
      request [energy, forces, virial]
      : (!vec, !md.cell) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to !buf
  return %u, %fb, %w : f64, !buf, vector<9xf64>
}

// The energy on domains with positions and cell stretched by `s`.
func.func @energy(%xb: !buf, %edge: f64, %sx: f64, %sy: f64, %sz: f64) -> f64 {
  %y = call @copy(%xb, %sx, %sy, %sz) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %lx = arith.mulf %edge, %sx : f64
  %ly = arith.mulf %edge, %sy : f64
  %lz = arith.mulf %edge, %sz : f64
  %cell = md.orthorhombic_cell %lx, %ly, %lz
  %u = md.evaluate @eam(%x, %cell) request [energy]
      : (!vec, !md.cell) -> f64
  return %u : f64
}

// Prints 1 if −dU/dx_ic by central differences is within 1e-7 * scale of
// f_ic.
func.func @check_difference(%xb: !buf, %edge: f64, %i: index, %c: index,
                            %f: !buf, %scale: f64) {
  %one = arith.constant 1.0 : f64
  %h = arith.constant 1.0e-5 : f64
  %two_h = arith.constant 2.0e-5 : f64
  %tolerance = arith.constant 1.0e-7 : f64
  %v = memref.load %xb[%i, %c] : !buf
  %plus = arith.addf %v, %h : f64
  %minus = arith.subf %v, %h : f64
  memref.store %plus, %xb[%i, %c] : !buf
  %up = call @energy(%xb, %edge, %one, %one, %one)
      : (!buf, f64, f64, f64, f64) -> f64
  memref.store %minus, %xb[%i, %c] : !buf
  %um = call @energy(%xb, %edge, %one, %one, %one)
      : (!buf, f64, f64, f64, f64) -> f64
  memref.store %v, %xb[%i, %c] : !buf
  %du = arith.subf %um, %up : f64
  %fd = arith.divf %du, %two_h : f64
  %analytic = memref.load %f[%i, %c] : !buf
  call @check(%fd, %analytic, %scale, %tolerance) : (f64, f64, f64, f64) -> ()
  return
}

// Prints 1 if −dU/dε of the stretch of axis `a` (positions and edge scaled
// by 1 + ε), by central differences, is within 1e-7 * scale of W_aa.
func.func @check_stretch(%xb: !buf, %edge: f64, %sp: vector<3xf64>,
                         %sm: vector<3xf64>, %w: f64, %scale: f64) {
  %two_h = arith.constant 2.0e-5 : f64
  %tolerance = arith.constant 1.0e-7 : f64
  %px = vector.extract %sp[0] : f64 from vector<3xf64>
  %py = vector.extract %sp[1] : f64 from vector<3xf64>
  %pz = vector.extract %sp[2] : f64 from vector<3xf64>
  %mx = vector.extract %sm[0] : f64 from vector<3xf64>
  %my = vector.extract %sm[1] : f64 from vector<3xf64>
  %mz = vector.extract %sm[2] : f64 from vector<3xf64>
  %up = call @energy(%xb, %edge, %px, %py, %pz)
      : (!buf, f64, f64, f64, f64) -> f64
  %um = call @energy(%xb, %edge, %mx, %my, %mz)
      : (!buf, f64, f64, f64, f64) -> f64
  %du = arith.subf %um, %up : f64
  %fd = arith.divf %du, %two_h : f64
  call @check(%fd, %w, %scale, %tolerance) : (f64, f64, f64, f64) -> ()
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c17 = arith.constant 17 : index
  %c30 = arith.constant 30 : index
  %c45 = arith.constant 45 : index
  %c63 = arith.constant 63 : index
  %count = arith.constant 64 : index
  %x = memref.alloc(%count) : !buf
  call @place(%x) : (!buf) -> ()
  %edge = arith.constant 4.8 : f64
  %tol = arith.constant 1.0e-10 : f64
  %u, %f, %w = call @reference(%x, %edge)
      : (!buf, f64) -> (f64, !buf, vector<9xf64>)
  %ud, %fd, %wd = call @derived(%x, %edge)
      : (!buf, f64) -> (f64, !buf, vector<9xf64>)
  // One domain against Inputs/eam_reference.py: the energy
  // 31.860504482882163, the force component 0 of particle 0
  // 0.44715410193808536 (of the largest, 2.240583255195037), and the trace
  // of the virial 584.8240522027986.
  // CHECK: 31.8605
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  call @printF64(%u) : (f64) -> ()
  call @printNewline() : () -> ()
  %u_ref = arith.constant 31.860504482882163 : f64
  call @check(%u, %u_ref, %u_ref, %tol) : (f64, f64, f64, f64) -> ()
  %f00 = memref.load %f[%c0, %c0] : !buf
  %f00_ref = arith.constant 0.44715410193808536 : f64
  %f_scale = arith.constant 2.240583255195037 : f64
  call @check(%f00, %f00_ref, %f_scale, %tol) : (f64, f64, f64, f64) -> ()
  %rxx = vector.extract %w[0] : f64 from vector<9xf64>
  %ryy = vector.extract %w[4] : f64 from vector<9xf64>
  %rzz = vector.extract %w[8] : f64 from vector<9xf64>
  %r0 = arith.addf %rxx, %ryy : f64
  %rtrace = arith.addf %r0, %rzz : f64
  %trace_ref = arith.constant 584.8240522027986 : f64
  call @check(%rtrace, %trace_ref, %trace_ref, %tol) : (f64, f64, f64, f64) -> ()
  // On domains against one domain: energy, forces, virial.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  call @check(%ud, %u, %u, %tol) : (f64, f64, f64, f64) -> ()
  call @check_forces(%fd, %f) : (!buf, !buf) -> ()
  call @check_virial(%wd, %w) : (vector<9xf64>, vector<9xf64>) -> ()
  // Central differences of the distributed energy in positions: particle 0
  // along x, 17 along y, 30 along z, 45 along x, 63 along y and z.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %scale = call @largest(%fd) : (!buf) -> f64
  call @check_difference(%x, %edge, %c0, %c0, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
  call @check_difference(%x, %edge, %c17, %c1, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
  call @check_difference(%x, %edge, %c30, %c2, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
  call @check_difference(%x, %edge, %c45, %c0, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
  call @check_difference(%x, %edge, %c63, %c1, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
  call @check_difference(%x, %edge, %c63, %c2, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
  // Central differences in the three stretches of the cell against the
  // diagonal of the virial.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %one = arith.constant 1.0 : f64
  %p = arith.constant 1.00001 : f64
  %m = arith.constant 0.99999 : f64
  %wxx = vector.extract %wd[0] : f64 from vector<9xf64>
  %wyy = vector.extract %wd[4] : f64 from vector<9xf64>
  %wzz = vector.extract %wd[8] : f64 from vector<9xf64>
  %t0 = arith.addf %wxx, %wyy : f64
  %trace = arith.addf %t0, %wzz : f64
  %spx = vector.from_elements %p, %one, %one : vector<3xf64>
  %smx = vector.from_elements %m, %one, %one : vector<3xf64>
  %spy = vector.from_elements %one, %p, %one : vector<3xf64>
  %smy = vector.from_elements %one, %m, %one : vector<3xf64>
  %spz = vector.from_elements %one, %one, %p : vector<3xf64>
  %smz = vector.from_elements %one, %one, %m : vector<3xf64>
  call @check_stretch(%x, %edge, %spx, %smx, %wxx, %trace)
      : (!buf, f64, vector<3xf64>, vector<3xf64>, f64, f64) -> ()
  call @check_stretch(%x, %edge, %spy, %smy, %wyy, %trace)
      : (!buf, f64, vector<3xf64>, vector<3xf64>, f64, f64) -> ()
  call @check_stretch(%x, %edge, %spz, %smz, %wzz, %trace)
      : (!buf, f64, vector<3xf64>, vector<3xf64>, f64, f64) -> ()
  // The adjoint identity of the density: <F, v> = −<J^T c, v> on domains
  // against −<c, J v> summed directly.
  // CHECK-NEXT: 1
  %cb, %vb = call @coefficients(%count) : (index) -> (memref<?xf64>, !buf)
  %left = call @vjp(%x, %edge, %cb, %vb)
      : (!buf, f64, memref<?xf64>, !buf) -> f64
  %right0 = call @jvp(%x, %edge, %cb, %vb)
      : (!buf, f64, memref<?xf64>, !buf) -> f64
  %right = arith.negf %right0 : f64
  %tight = arith.constant 1.0e-12 : f64
  call @check(%left, %right, %right, %tight) : (f64, f64, f64, f64) -> ()
  return
}
