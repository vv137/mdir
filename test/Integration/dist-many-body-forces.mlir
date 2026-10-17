// The forces and the virial of a many-body energy on several domains
// through the compiler (D[ml2-eam], docs/distributed-ml2.md, Section 15.4):
// the energy of dist-many-body.mlir, whose pair sum reads ψ = √ρ, a field
// computed from the positions, at both particles. On the directed
// traversal the adjoint of ψ is the gather at the owners; the adjoint of
// ρ is copied to the ghosts, and the forces of the density's gather return
// by G^T, or are computed at the owners after md-dist-owner-computes.
// Against the same potential on one domain, and central differences of the
// distributed energy.
//
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=16,1,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: mdir-opt %s --md-distribute="grid=16,1,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// The 64 particles of dist-lj.mlir and the energy of dist-many-body.mlir:
//
//   U = Σ_i q_i ψ_i + Σ_{i<j} ψ_i ψ_j / r_ij,  ψ = √ρ,  ρ_i = Σ_j exp(−r_ij),
//
// with q_i = 1 + (i mod 3) / 2. Each check prints 1 if it holds: the energy
// to a relative 1e-10, every force component to 1e-10 of the largest
// component, every virial component to 1e-10 of the trace, and central
// differences with a step of 1e-5 to 1e-7 of the largest force component
// and of the trace.

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!buf   = memref<?x3xf64>

md.particle_set @atoms

md.potential @many_body(%x: !vec, %cell: !md.cell, %q: !sca) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  %psi = md.map_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md.yield %s : f64
  } : !sca
  %e1 = md.sum_particles gather(%psi, %q : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell gather(%psi : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %pi: f64, %pj: f64):
    %p = arith.mulf %pi, %pj : f64
    %k = arith.divf %p, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
  md.return %u : f64
}

// The same, on one domain: the reference.
md.potential @reference_body(%x: !vec, %cell: !md.cell, %q: !sca) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  %psi = md.map_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md.yield %s : f64
  } : !sca
  %e1 = md.sum_particles gather(%psi, %q : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell gather(%psi : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %pi: f64, %pj: f64):
    %p = arith.mulf %pi, %pj : f64
    %k = arith.divf %p, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
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

// q_i = 1 + (i mod 3) / 2.
func.func @charges(%n: index) -> memref<?xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %one = arith.constant 1.0 : f64
  %half = arith.constant 0.5 : f64
  %q = memref.alloc(%n) : memref<?xf64>
  scf.for %i = %c0 to %n step %c1 {
    %r = arith.remui %i, %c3 : index
    %ri = arith.index_cast %r : index to i64
    %rf = arith.sitofp %ri : i64 to f64
    %h = arith.mulf %rf, %half : f64
    %v = arith.addf %one, %h : f64
    memref.store %v, %q[%i] : memref<?xf64>
  }
  return %q : memref<?xf64>
}

// The reference on one domain.
func.func @reference(%xb: !buf, %edge: f64) -> (f64, !buf, vector<9xf64>) {
  %one = arith.constant 1.0 : f64
  %c0 = arith.constant 0 : index
  %n = memref.dim %xb, %c0 : !buf
  %qb = call @charges(%n) : (index) -> memref<?xf64>
  %q = mdrt.from_buffer %qb : memref<?xf64> to !sca
  %y = call @copy(%xb, %one, %one, %one) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @reference_body(%x, %cell, %q)
      request [energy, forces, virial]
      : (!vec, !md.cell, !sca) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to !buf
  return %u, %fb, %w : f64, !buf, vector<9xf64>
}

// The potential on domains.
func.func @derived(%xb: !buf, %edge: f64) -> (f64, !buf, vector<9xf64>) {
  %one = arith.constant 1.0 : f64
  %c0 = arith.constant 0 : index
  %n = memref.dim %xb, %c0 : !buf
  %qb = call @charges(%n) : (index) -> memref<?xf64>
  %q = mdrt.from_buffer %qb : memref<?xf64> to !sca
  %y = call @copy(%xb, %one, %one, %one) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @many_body(%x, %cell, %q)
      request [energy, forces, virial]
      : (!vec, !md.cell, !sca) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to !buf
  return %u, %fb, %w : f64, !buf, vector<9xf64>
}

// The energy on domains with positions and cell stretched by `s`.
func.func @energy(%xb: !buf, %edge: f64, %sx: f64, %sy: f64, %sz: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %n = memref.dim %xb, %c0 : !buf
  %qb = call @charges(%n) : (index) -> memref<?xf64>
  %q = mdrt.from_buffer %qb : memref<?xf64> to !sca
  %y = call @copy(%xb, %sx, %sy, %sz) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %lx = arith.mulf %edge, %sx : f64
  %ly = arith.mulf %edge, %sy : f64
  %lz = arith.mulf %edge, %sz : f64
  %cell = md.orthorhombic_cell %lx, %ly, %lz
  %u = md.evaluate @many_body(%x, %cell, %q) request [energy]
      : (!vec, !md.cell, !sca) -> f64
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
  %c45 = arith.constant 45 : index
  %count = arith.constant 64 : index
  %x = memref.alloc(%count) : !buf
  call @place(%x) : (!buf) -> ()
  %edge = arith.constant 4.8 : f64
  %tol = arith.constant 1.0e-10 : f64
  %u, %f, %w = call @reference(%x, %edge)
      : (!buf, f64) -> (f64, !buf, vector<9xf64>)
  %ud, %fd, %wd = call @derived(%x, %edge)
      : (!buf, f64) -> (f64, !buf, vector<9xf64>)
  // The energy of dist-many-body.mlir on one domain.
  // CHECK: 1896.65
  call @printF64(%u) : (f64) -> ()
  call @printNewline() : () -> ()
  // On domains against one domain: energy, forces, virial.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  call @check(%ud, %u, %u, %tol) : (f64, f64, f64, f64) -> ()
  call @check_forces(%fd, %f) : (!buf, !buf) -> ()
  call @check_virial(%wd, %w) : (vector<9xf64>, vector<9xf64>) -> ()
  // Central differences of the distributed energy: particle 0 along x, 17
  // along y, 45 along z; the stretch of the cell along x.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %scale = call @largest(%fd) : (!buf) -> f64
  call @check_difference(%x, %edge, %c0, %c0, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
  call @check_difference(%x, %edge, %c17, %c1, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
  call @check_difference(%x, %edge, %c45, %c2, %fd, %scale)
      : (!buf, f64, index, index, !buf, f64) -> ()
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
  call @check_stretch(%x, %edge, %spx, %smx, %wxx, %trace)
      : (!buf, f64, vector<3xf64>, vector<3xf64>, f64, f64) -> ()
  return
}
