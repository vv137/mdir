// Lennard-Jones on several domains through the compiler (D276,
// docs/distributed-ml2.md, Section 14): the global potential of
// dist-lj.mlir, distributed by md-distribute and differentiated by
// md-differentiate, against the same potential on one domain and against
// the program of dist-lj.mlir written by hand in md_dist.
//
// Each grid of dist-lj.mlir runs with the unique traversal, with the
// directed traversal in the derivative's form (the forces of the ghosts
// returned by G^T, each pair half at each of its owners), and with the
// directed traversal after md-dist-owner-computes (every pair at both of
// its owners, nothing returned). The grid of the hand-written program comes
// from sed and that of md-distribute from its option.
//
// RUN: sed 's/GRID/1, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=1,1,1 traversal=unique potentials=lj" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/1, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=1,1,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/1, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=1,1,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/2, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=2,1,1 traversal=unique potentials=lj" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/2, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=2,1,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/2, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=2,1,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/3, 2, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=unique potentials=lj" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/3, 2, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/3, 2, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/16, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=unique potentials=lj" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/16, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/16, 1, 1/' %s \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --md-dist-owner-computes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// The system of dist-lj.mlir and lj-forces.mlir: 64 particles near a cubic
// lattice of spacing 1.2 in a periodic cube of edge 4.8; Lennard-Jones with
// eps = 1, sigma = 1, truncated at 2.0 without a switch.
//
// Each check prints 1 if it holds: energies to a relative 1e-10, every force
// component to 1e-10 of the largest component, every virial component to
// 1e-10 of the trace (as dist-lj.mlir); and central differences, with a
// step of 1e-6, of the distributed energy in six position components and in
// the three stretches of the cell against the distributed forces and the
// diagonal of the virial, to 1e-7 of the largest force component and of
// the trace.

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
!own   = !md_dist.field<@dd, owned, 3 x f64>
!gho   = !md_dist.field<@dd, ghost, 3 x f64>
!halo  = !md_dist.halo<@dd, 2.0>
!buf   = memref<?x3xf64>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([GRID])

// Distributed by md-distribute.
md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
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

// The same, on one domain: the reference.
md.potential @lj_ref(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
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

// Prints 1 if every component of `a` is within tolerance * max |b| of `b`.
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

// The reference on one domain.
func.func @reference(%xb: !buf, %edge: f64) -> (f64, !buf, vector<9xf64>) {
  %one = arith.constant 1.0 : f64
  %eps = arith.constant 1.0 : f64
  %sigma = arith.constant 1.0 : f64
  %y = call @copy(%xb, %one, %one, %one) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @lj_ref(%x, %cell, %eps, %sigma)
      request [energy, forces, virial]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to !buf
  return %u, %fb, %w : f64, !buf, vector<9xf64>
}

// Distributed by md-distribute and differentiated by md-differentiate.
func.func @derived(%xb: !buf, %edge: f64) -> (f64, !buf, vector<9xf64>) {
  %one = arith.constant 1.0 : f64
  %eps = arith.constant 1.0 : f64
  %sigma = arith.constant 1.0 : f64
  %y = call @copy(%xb, %one, %one, %one) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @lj(%x, %cell, %eps, %sigma)
      request [energy, forces, virial]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to !buf
  return %u, %fb, %w : f64, !buf, vector<9xf64>
}

// The distributed energy at `xb` with each axis scaled by s, in a cell of
// edges `edge` times s.
func.func @energy(%xb: !buf, %edge: f64, %sx: f64, %sy: f64, %sz: f64) -> f64 {
  %eps = arith.constant 1.0 : f64
  %sigma = arith.constant 1.0 : f64
  %y = call @copy(%xb, %sx, %sy, %sz) : (!buf, f64, f64, f64) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %lx = arith.mulf %edge, %sx : f64
  %ly = arith.mulf %edge, %sy : f64
  %lz = arith.mulf %edge, %sz : f64
  %cell = md.orthorhombic_cell %lx, %ly, %lz
  %u = md.evaluate @lj(%x, %cell, %eps, %sigma) request [energy]
      : (!vec, !md.cell, f64, f64) -> f64
  return %u : f64
}

// Prints 1 if −(U(x + h e) − U(x − h e)) / 2h, with e the unit vector of
// component c of particle i, is within 1e-7 * scale of the force `f`.
func.func @check_difference(%xb: !buf, %edge: f64, %i: index, %c: index,
                            %f: !buf, %scale: f64) {
  %one = arith.constant 1.0 : f64
  %h = arith.constant 1.0e-6 : f64
  %two_h = arith.constant 2.0e-6 : f64
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
  %two_h = arith.constant 2.0e-6 : f64
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

// The program of dist-lj.mlir, written by hand in md_dist: energy, forces,
// and virial on the directed traversal, energy and forces on the unique.
func.func @handwritten(%buffer: !buf, %edge: f64)
    -> (f64, !buf, vector<9xf64>, f64, !buf) {
  %one = arith.constant 1.0 : f64
  %eps = arith.constant 1.0 : f64
  %sigma = arith.constant 1.0 : f64
  %copy = call @copy(%buffer, %one, %one, %one) : (!buf, f64, f64, f64) -> !buf
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %xs = mdrt.from_buffer %copy : !buf to !vec
  %lay = md_dist.partition %xs, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %xs, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho

  %nd = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
          : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  %udp = md_dist.sum_pairs %nd, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md_dist.yield %k : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  %ud = md_dist.all_reduce %udp : !md_dist.partial<f64> -> f64
  %fdo = md_dist.gather_pairs %nd, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %sv  = vector.broadcast %s : f64 to vector<3xf64>
    %f   = arith.mulf %sv, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !own
  %fd = md_dist.collect %fdo, %lay : !own -> !vec
  %wdp = md_dist.sum_pairs %nd, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %left = vector.shuffle %d, %d [0, 0, 0, 1, 1, 1, 2, 2, 2] : vector<3xf64>, vector<3xf64>
    %right = vector.shuffle %d, %d [0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xf64>, vector<3xf64>
    %dd  = arith.mulf %left, %right : vector<9xf64>
    %sv  = vector.broadcast %s : f64 to vector<9xf64>
    %w   = arith.mulf %sv, %dd : vector<9xf64>
    md_dist.yield %w : vector<9xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<vector<9xf64>>
  %wd = md_dist.all_reduce %wdp : !md_dist.partial<vector<9xf64>> -> vector<9xf64>

  %nu = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
          : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  %uup = md_dist.sum_pairs %nu, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md_dist.yield %k : f64
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !md_dist.partial<f64>
  %uu = md_dist.all_reduce %uup : !md_dist.partial<f64> -> f64
  %fuo, %fug = md_dist.gather_pairs %nu, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %sv  = vector.broadcast %s : f64 to vector<3xf64>
    %f   = arith.mulf %sv, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho
      -> !md_dist.field<@dd, owned, partial, 3 x f64>,
         !md_dist.field<@dd, ghost, partial, 3 x f64>
  %fuc = md_dist.halo_sum %fuo, %fug, %h
           : !md_dist.field<@dd, owned, partial, 3 x f64>,
             !md_dist.field<@dd, ghost, partial, 3 x f64>, !halo -> !own
  %fu = md_dist.collect %fuc, %lay : !own -> !vec
  %fdb = mdrt.to_buffer %fd : !vec to !buf
  %fub = mdrt.to_buffer %fu : !vec to !buf
  return %ud, %fdb, %wd, %uu, %fub : f64, !buf, vector<9xf64>, f64, !buf
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
  %uh, %fh, %wh, %uhu, %fhu = call @handwritten(%x, %edge)
      : (!buf, f64) -> (f64, !buf, vector<9xf64>, f64, !buf)

  // The energy on one domain, and against Inputs/lj_reference.py with the
  // switch taken out (SWITCH_FROM = CUTOFF): -226.88590520072805.
  // CHECK: -226.886
  // CHECK-NEXT: 1
  call @printF64(%u) : (f64) -> ()
  call @printNewline() : () -> ()
  %u_ref = arith.constant -226.88590520072805 : f64
  call @check(%u, %u_ref, %u_ref, %tol) : (f64, f64, f64, f64) -> ()

  // Derived on domains against one domain: energy, forces, virial.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  call @check(%ud, %u, %u, %tol) : (f64, f64, f64, f64) -> ()
  call @check_forces(%fd, %f) : (!buf, !buf) -> ()
  call @check_virial(%wd, %w) : (vector<9xf64>, vector<9xf64>) -> ()

  // Derived against the hand-written program, directed and unique.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  call @check(%ud, %uh, %uh, %tol) : (f64, f64, f64, f64) -> ()
  call @check_forces(%fd, %fh) : (!buf, !buf) -> ()
  call @check_virial(%wd, %wh) : (vector<9xf64>, vector<9xf64>) -> ()
  call @check(%ud, %uhu, %uhu, %tol) : (f64, f64, f64, f64) -> ()
  call @check_forces(%fd, %fhu) : (!buf, !buf) -> ()

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
  %p = arith.constant 1.000001 : f64
  %m = arith.constant 0.999999 : f64
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
  return
}
