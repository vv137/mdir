// A Lennard-Jones trajectory on domains against the same trajectory on one
// domain (D[distributed-migration]). The RUN lines are in
// dist-trajectory-lj.test; GRID, STEPS, and TOLERANCES are put in by sed.
//
// The system of lj-dynamics.mlir: 64 particles on a cubic lattice of
// spacing 1.2, each displaced by up to 0.1 along every axis, in
// a periodic cube of edge 4.8, here with velocities drawn uniformly in
// [-1, 1) per component. The lattice puts particles at 0 and 2.4 along each
// axis, the faces of the domains of a 2 x 2 x 2 grid and of the periodic
// boundary, so that particles cross faces, edges, and corners. The masses
// are 1, 1.25, 1.5, and 1.75 in turn (exact in binary), so that a mass
// carried to another particle by a migration changes the trajectory.
//
// Lennard-Jones with eps = sigma = 1, shifted so that the energy and the
// force vanish at the cutoff 2.0, so that the energy is conserved by the
// integrator to second order in the step:
//
//   u(r) = 4 (r^-12 - r^-6) - u_c - (r - r_c) u'_c,
//   u_c = -0.0615234375, u'_c = 0.181640625 (exact in binary).
//
// One domain: md.potential through md-differentiate and the neighbor
// structures of md_exec, stepped by dyn.step (velocity Verlet). Domains: a
// md_dist.state of the positions, velocities, forces, and masses, a halo of
// reach 2.4 (a skin of 0.4), and at every step the collective skin test
// md_dist.covered: the halo and the layout are kept while it holds, and the
// particles migrate and the halo is rebuilt when it does not. At every step
// the fields unpacked after the branch are compared, in the order of the
// set, with those before it: the forces carried through a rebuild are not
// read by the step (they are evaluated again), so this is where a force
// field given to the wrong particles would show.

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!own   = !md_dist.field<@dd, owned, 3 x f64>
!mass  = !md_dist.field<@dd, owned, f64>
!gho   = !md_dist.field<@dd, ghost, 3 x f64>
!halo  = !md_dist.halo<@dd, 2.4>
!lay   = !md_dist.layout<@dd>
!rel   = !md_dist.relation<@dd, directed>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0,
                        fields = [3 x f64, 3 x f64, 3 x f64, f64]>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([GRID])

md.potential @lj(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %rc  = arith.constant 2.0 : f64
    %uc  = arith.constant -0.0615234375 : f64
    %duc = arith.constant 0.181640625 : f64
    %one = arith.constant 1.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %one, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %u0  = arith.mulf %c4, %t : f64
    %dr  = arith.subf %r, %rc : f64
    %lin = arith.mulf %dr, %duc : f64
    %u1  = arith.subf %u0, %uc : f64
    %k   = arith.subf %u1, %lin : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

dyn.program @velocity_verlet(%x: !vec, %v: !vec, %f: !vec, %m: !real,
                             %cell: !md.cell, %dt: f64) -> (!vec, !vec, !vec)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %c    = arith.constant 0.5 : f64
  %half = arith.mulf %c, %dt : f64
  %v1 = dyn.kick %v, %f, %m, %half : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  %f1 = md.evaluate @lj(%x1, %cell) request [forces]
      : (!vec, !md.cell) -> !vec
  %v2 = dyn.kick %v1, %f1, %m, %half : !vec
  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec
}

func.func private @printF64(f64)
func.func private @printI64(i64)
func.func private @printNewline()

// Prints `value`, then 1 if |value - reference| <= tolerance * |scale|, else
// 0.
func.func @check(%value: f64, %reference: f64, %scale: f64, %tolerance: f64) {
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %magnitude = math.absf %scale : f64
  %bound = arith.mulf %tolerance, %magnitude : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%error) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

// The largest difference of two fields per component, with `period` the
// difference is taken to the nearest image (positions wrapped on domains).
func.func @difference(%a: memref<?x3xf64>, %b: memref<?x3xf64>, %period: f64)
    -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %half = arith.constant 0.5 : f64
  %n = memref.dim %a, %c0 : memref<?x3xf64>
  %periodic = arith.cmpf ogt, %period, %zero : f64
  %error = scf.for %i = %c0 to %n step %c1 iter_args(%e = %zero) -> f64 {
    %row = scf.for %k = %c0 to %c3 step %c1 iter_args(%r = %e) -> f64 {
      %va = memref.load %a[%i, %k] : memref<?x3xf64>
      %vb = memref.load %b[%i, %k] : memref<?x3xf64>
      %raw = arith.subf %va, %vb : f64
      %q = arith.divf %raw, %period : f64
      %qh = arith.addf %q, %half : f64
      %whole = math.floor %qh : f64
      %shift = arith.mulf %whole, %period : f64
      %wrapped = arith.subf %raw, %shift : f64
      %delta = arith.select %periodic, %wrapped, %raw : f64
      %ad = math.absf %delta : f64
      %next = arith.maximumf %r, %ad : f64
      scf.yield %next : f64
    }
    scf.yield %row : f64
  }
  return %error : f64
}

// The largest absolute component of a field.
func.func @largest(%a: memref<?x3xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %a, %c0 : memref<?x3xf64>
  %m = scf.for %i = %c0 to %n step %c1 iter_args(%e = %zero) -> f64 {
    %row = scf.for %k = %c0 to %c3 step %c1 iter_args(%r = %e) -> f64 {
      %va = memref.load %a[%i, %k] : memref<?x3xf64>
      %ad = math.absf %va : f64
      %next = arith.maximumf %r, %ad : f64
      scf.yield %next : f64
    }
    scf.yield %row : f64
  }
  return %m : f64
}

// Places the particles, as lj-dynamics.mlir does.
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

// Velocities uniform in [-1, 1) per component, from another seed.
func.func @shake(%v: memref<?x3xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %two = arith.constant 2.0 : f64
  %one = arith.constant 1.0 : f64
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 7 : i64
  %n = memref.dim %v, %c0 : memref<?x3xf64>
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %g = arith.mulf %f, %two : f64
      %value = arith.subf %g, %one : f64
      memref.store %value, %v[%i, %k] : memref<?x3xf64>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

// The 8 particles at the lattice sites whose three coordinates are all
// on faces of the 2 x 2 x 2 grid (0 or 2.4) are aimed at the corner they
// are next to: each component 1.5 towards its face, so that each crosses
// three faces within a few steps, a corner of the domains.
func.func @aim(%x: memref<?x3xf64>, %v: memref<?x3xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %spacing = arith.constant 1.2 : f64
  %fast = arith.constant 1.5 : f64
  %slow = arith.constant -1.5 : f64
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  scf.for %i = %c0 to %n step %c1 {
    %d0 = arith.remui %i, %c4 : index
    %q1 = arith.divui %i, %c4 : index
    %d1 = arith.remui %q1, %c4 : index
    %d2 = arith.divui %i, %c16 : index
    %e0 = arith.remui %d0, %c2 : index
    %e1 = arith.remui %d1, %c2 : index
    %e2 = arith.remui %d2, %c2 : index
    %o01 = arith.ori %e0, %e1 : index
    %odd = arith.ori %o01, %e2 : index
    %corner = arith.cmpi eq, %odd, %c0 : index
    scf.if %corner {
      scf.for %k = %c0 to %c3 step %c1 {
        %p0 = arith.cmpi eq, %k, %c0 : index
        %p1 = arith.cmpi eq, %k, %c1 : index
        %d12 = arith.select %p1, %d1, %d2 : index
        %d = arith.select %p0, %d0, %d12 : index
        %wide = arith.index_cast %d : index to i64
        %site = arith.sitofp %wide : i64 to f64
        %face = arith.mulf %site, %spacing : f64
        %xk = memref.load %x[%i, %k] : memref<?x3xf64>
        %below = arith.cmpf olt, %xk, %face : f64
        %vk = arith.select %below, %fast, %slow : f64
        memref.store %vk, %v[%i, %k] : memref<?x3xf64>
      }
    }
  }
  return
}

// The masses 1 + (i mod 4) / 4.
func.func @weigh(%x: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %one = arith.constant 1.0 : f64
  %quarter = arith.constant 0.25 : f64
  %n = memref.dim %x, %c0 : memref<?xf64>
  scf.for %i = %c0 to %n step %c1 {
    %r = arith.remui %i, %c4 : index
    %ri = arith.index_cast %r : index to i64
    %rf = arith.sitofp %ri : i64 to f64
    %q = arith.mulf %rf, %quarter : f64
    %value = arith.addf %one, %q : f64
    memref.store %value, %x[%i] : memref<?xf64>
  }
  return
}

// The largest difference of two scalar fields.
func.func @difference_scalar(%a: memref<?xf64>, %b: memref<?xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %a, %c0 : memref<?xf64>
  %error = scf.for %i = %c0 to %n step %c1 iter_args(%e = %zero) -> f64 {
    %va = memref.load %a[%i] : memref<?xf64>
    %vb = memref.load %b[%i] : memref<?xf64>
    %raw = arith.subf %va, %vb : f64
    %ad = math.absf %raw : f64
    %next = arith.maximumf %e, %ad : f64
    scf.yield %next : f64
  }
  return %error : f64
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = arith.constant 64 : index
  %steps = arith.constant STEPS : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %edge = arith.constant 4.8 : f64
  %dt = arith.constant 0.005 : f64
  %halfdt = arith.constant 0.0025 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  %masses = memref.alloc(%count) : memref<?xf64>
  call @weigh(%masses) : (memref<?xf64>) -> ()
  %m = mdrt.from_buffer %masses : memref<?xf64> to !real

  %positions = memref.alloc(%count) : memref<?x3xf64>
  %velocities = memref.alloc(%count) : memref<?x3xf64>
  call @place(%positions) : (memref<?x3xf64>) -> ()
  call @shake(%velocities) : (memref<?x3xf64>) -> ()
  call @aim(%positions, %velocities) : (memref<?x3xf64>, memref<?x3xf64>) -> ()
  %x0 = mdrt.from_buffer %positions : memref<?x3xf64> to !vec
  %v0 = mdrt.from_buffer %velocities : memref<?x3xf64> to !vec

  //===--------------------------------------------------------------------===//
  // One domain
  //===--------------------------------------------------------------------===//

  %e0, %f0 = md.evaluate @lj(%x0, %cell) request [energy, forces]
      : (!vec, !md.cell) -> (f64, !vec)
  %k0 = md.sum_particles gather(%v0, %m : !vec, !real) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %half = arith.constant 0.5 : f64
    %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
    %mv2  = arith.mulf %m_i, %v2 : f64
    %ke   = arith.mulf %half, %mv2 : f64
    md.yield %ke : f64
  } : f64
  %t0 = arith.addf %e0, %k0 : f64
  %xr, %vr, %fr = scf.for %s = %c0 to %steps step %c1
      iter_args(%xa = %x0, %va = %v0, %fa = %f0) -> (!vec, !vec, !vec) {
    %xb, %vb, %fb = dyn.step @velocity_verlet(%xa, %va, %fa, %m, %cell, %dt)
        : (!vec, !vec, !vec, !real, !md.cell, f64) -> (!vec, !vec, !vec)
    scf.yield %xb, %vb, %fb : !vec, !vec, !vec
  }
  %ur = md.evaluate @lj(%xr, %cell) request [energy]
      : (!vec, !md.cell) -> f64
  %kr = md.sum_particles gather(%vr, %m : !vec, !real) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %half = arith.constant 0.5 : f64
    %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
    %mv2  = arith.mulf %m_i, %v2 : f64
    %ke   = arith.mulf %half, %mv2 : f64
    md.yield %ke : f64
  } : f64

  //===--------------------------------------------------------------------===//
  // Domains
  //===--------------------------------------------------------------------===//

  // The same initial state, in buffers of its own.
  %positions2 = memref.alloc(%count) : memref<?x3xf64>
  %velocities2 = memref.alloc(%count) : memref<?x3xf64>
  call @place(%positions2) : (memref<?x3xf64>) -> ()
  call @shake(%velocities2) : (memref<?x3xf64>) -> ()
  call @aim(%positions2, %velocities2) : (memref<?x3xf64>, memref<?x3xf64>) -> ()
  %y0 = mdrt.from_buffer %positions2 : memref<?x3xf64> to !vec
  %w0 = mdrt.from_buffer %velocities2 : memref<?x3xf64> to !vec
  %l0 = md_dist.partition %y0, %cell : !vec -> !lay
  %xo = md_dist.distribute %y0, %l0 image(%cell) : !vec -> !own
  %vo = md_dist.distribute %w0, %l0 : !vec -> !own
  %mo = md_dist.distribute %m, %l0 : !real -> !mass
  %h0 = md_dist.halo %l0, %xo, %cell : !own -> !halo
  %xg0 = md_dist.halo_copy %xo, %h0 image(%cell) : !own, !halo -> !gho
  %n0 = md_dist.neighborhood %xo, %xg0, %h0 cutoff(2.0)
      : !own, !gho, !halo -> !rel
  %fo = md_dist.gather_pairs %n0, %xo, %xg0 exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %duc = arith.constant 0.181640625 : f64
    %unit = arith.constant 1.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %unit, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %num = arith.mulf %c4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s0  = arith.divf %num, %r2 : f64
    %sc  = arith.mulf %duc, %sr : f64
    %s   = arith.addf %s0, %sc : f64
    %sv  = vector.broadcast %s : f64 to vector<3xf64>
    %f   = arith.mulf %sv, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !rel, !own, !gho -> !own
  %start = md_dist.pack %l0, %h0, %cell
      fields(%xo, %vo, %fo, %mo : !own, !own, !own, !mass) : !state
  %c0i = arith.constant 0 : i64
  %c1i = arith.constant 1 : i64

  %final, %rebuilds, %carried, %carried_x = scf.for %s = %c0 to %steps step %c1
      iter_args(%sa = %start, %nb = %c0i, %ca = %zero, %cx = %zero)
      -> (!state, i64, f64, f64) {
    %l, %h, %xref, %x, %v, %f, %mm = md_dist.unpack %sa, %cell : !state
    // The first half kick and the drift, on the owned rows.
    %v1 = md_dist.map_particles gather(%v, %f, %mm : !own, !own, !mass) {
    ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
      %w = arith.divf %halfdt, %m_i : f64
      %wv = vector.broadcast %w : f64 to vector<3xf64>
      %dv = arith.mulf %wv, %f_i : vector<3xf64>
      %next = arith.addf %v_i, %dv : vector<3xf64>
      md_dist.yield %next : vector<3xf64>
    } : !own
    %x1 = md_dist.map_particles gather(%x, %v1 : !own, !own) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
      %dtv = vector.broadcast %dt : f64 to vector<3xf64>
      %dx = arith.mulf %dtv, %v_i : vector<3xf64>
      %next = arith.addf %x_i, %dx : vector<3xf64>
      md_dist.yield %next : vector<3xf64>
    } : !own
    // The collective skin test: one bit, the same on every domain.
    %ok = md_dist.covered %h, %xref, %x1, %cell cutoff(2.0) : !halo, !own -> i1
    %moved, %nb1 = scf.if %ok -> (!state, i64) {
      // The halo still covers the positions: keep the layout and the halo.
      %kept = md_dist.pack %l, %h, %cell
          fields(%x1, %v1, %f, %mm : !own, !own, !own, !mass) : !state
      scf.yield %kept, %nb : !state, i64
    } else {
      // A new ownership epoch: every carried field migrates, the positions
      // wrapped into the cell, and the halo is rebuilt at them.
      %l2 = md_dist.migrate %l, %x1, %cell : !lay, !own -> !lay
      %x2 = md_dist.redistribute %x1, %l, %l2 image(%cell)
          : !own, !lay, !lay -> !own
      %v2 = md_dist.redistribute %v1, %l, %l2 : !own, !lay, !lay -> !own
      %f2 = md_dist.redistribute %f, %l, %l2 : !own, !lay, !lay -> !own
      %m2 = md_dist.redistribute %mm, %l, %l2 : !mass, !lay, !lay -> !mass
      %h2 = md_dist.halo %l2, %x2, %cell : !own -> !halo
      %rebuilt = md_dist.pack %l2, %h2, %cell
          fields(%x2, %v2, %f2, %m2 : !own, !own, !own, !mass) : !state
      %nb2 = arith.addi %nb, %c1i : i64
      scf.yield %rebuilt, %nb2 : !state, i64
    }
    // The forces at the new positions and the second half kick.
    %lb, %hb, %xrb, %xb, %vb, %fb_old, %mb = md_dist.unpack %moved, %cell
        : !state
    // The fields after the branch against those before it, in the order of
    // the set: a migration copies them, so the velocities, forces, and
    // masses agree exactly, and the positions to the rounding of their wrap.
    %cx0 = md_dist.collect %x1, %l : !own -> !vec
    %cx1 = md_dist.collect %xb, %lb : !own -> !vec
    %cv0 = md_dist.collect %v1, %l : !own -> !vec
    %cv1 = md_dist.collect %vb, %lb : !own -> !vec
    %cf0 = md_dist.collect %f, %l : !own -> !vec
    %cf1 = md_dist.collect %fb_old, %lb : !own -> !vec
    %cm0 = md_dist.collect %mm, %l : !mass -> !real
    %cm1 = md_dist.collect %mb, %lb : !mass -> !real
    %bx0 = mdrt.to_buffer %cx0 : !vec to memref<?x3xf64>
    %bx1 = mdrt.to_buffer %cx1 : !vec to memref<?x3xf64>
    %bv0 = mdrt.to_buffer %cv0 : !vec to memref<?x3xf64>
    %bv1 = mdrt.to_buffer %cv1 : !vec to memref<?x3xf64>
    %bf0 = mdrt.to_buffer %cf0 : !vec to memref<?x3xf64>
    %bf1 = mdrt.to_buffer %cf1 : !vec to memref<?x3xf64>
    %bm0 = mdrt.to_buffer %cm0 : !real to memref<?xf64>
    %bm1 = mdrt.to_buffer %cm1 : !real to memref<?xf64>
    %ex = func.call @difference(%bx0, %bx1, %edge)
        : (memref<?x3xf64>, memref<?x3xf64>, f64) -> f64
    %ev = func.call @difference(%bv0, %bv1, %zero)
        : (memref<?x3xf64>, memref<?x3xf64>, f64) -> f64
    %ef = func.call @difference(%bf0, %bf1, %zero)
        : (memref<?x3xf64>, memref<?x3xf64>, f64) -> f64
    %em = func.call @difference_scalar(%bm0, %bm1)
        : (memref<?xf64>, memref<?xf64>) -> f64
    %evf = arith.maximumf %ev, %ef : f64
    %evfm = arith.maximumf %evf, %em : f64
    %ca1 = arith.maximumf %ca, %evfm : f64
    %cx1m = arith.maximumf %cx, %ex : f64
    %xgb = md_dist.halo_copy %xb, %hb image(%cell) : !own, !halo -> !gho
    %nbr = md_dist.neighborhood %xb, %xgb, %hb cutoff(2.0)
        : !own, !gho, !halo -> !rel
    %fb = md_dist.gather_pairs %nbr, %xb, %xgb exchange(antisymmetric) {
    ^bb0(%r: f64, %d: vector<3xf64>):
      %c4  = arith.constant 4.0 : f64
      %c6  = arith.constant 6.0 : f64
      %c12 = arith.constant 12.0 : f64
      %duc = arith.constant 0.181640625 : f64
      %unit = arith.constant 1.0 : f64
      %i6  = arith.constant 6 : i32
      %sr  = arith.divf %unit, %r : f64
      %s6  = math.fpowi %sr, %i6 : f64, i32
      %s12 = arith.mulf %s6, %s6 : f64
      %a12 = arith.mulf %c12, %s12 : f64
      %a6  = arith.mulf %c6, %s6 : f64
      %a   = arith.subf %a12, %a6 : f64
      %num = arith.mulf %c4, %a : f64
      %r2  = arith.mulf %r, %r : f64
      %s0  = arith.divf %num, %r2 : f64
      %sc  = arith.mulf %duc, %sr : f64
      %sf  = arith.addf %s0, %sc : f64
      %sv  = vector.broadcast %sf : f64 to vector<3xf64>
      %fv  = arith.mulf %sv, %d : vector<3xf64>
      md_dist.yield %fv : vector<3xf64>
    } : !rel, !own, !gho -> !own
    %v3 = md_dist.map_particles gather(%vb, %fb, %mb : !own, !own, !mass) {
    ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
      %w = arith.divf %halfdt, %m_i : f64
      %wv = vector.broadcast %w : f64 to vector<3xf64>
      %dv = arith.mulf %wv, %f_i : vector<3xf64>
      %next = arith.addf %v_i, %dv : vector<3xf64>
      md_dist.yield %next : vector<3xf64>
    } : !own
    %next = md_dist.pack %lb, %hb, %cell
        fields(%xb, %v3, %fb, %mb : !own, !own, !own, !mass) : !state
    scf.yield %next, %nb1, %ca1, %cx1m : !state, i64, f64, f64
  }

  // The energies and the fields after the steps, in the order of the set.
  %lf, %hf, %xrf, %xf, %vf, %ff, %mf = md_dist.unpack %final, %cell : !state
  %xgf = md_dist.halo_copy %xf, %hf image(%cell) : !own, !halo -> !gho
  %nf = md_dist.neighborhood %xf, %xgf, %hf cutoff(2.0)
      : !own, !gho, !halo -> !rel
  %up = md_dist.sum_pairs %nf, %xf, %xgf exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %rc  = arith.constant 2.0 : f64
    %uc  = arith.constant -0.0615234375 : f64
    %duc = arith.constant 0.181640625 : f64
    %unit = arith.constant 1.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %unit, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %u0  = arith.mulf %c4, %t : f64
    %dr  = arith.subf %r, %rc : f64
    %lin = arith.mulf %dr, %duc : f64
    %u1  = arith.subf %u0, %uc : f64
    %k   = arith.subf %u1, %lin : f64
    md_dist.yield %k : f64
  } : !rel, !own, !gho -> !md_dist.partial<f64>
  %ud = md_dist.all_reduce %up : !md_dist.partial<f64> -> f64
  %kp = md_dist.sum_particles gather(%vf, %mf : !own, !mass) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %half = arith.constant 0.5 : f64
    %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
    %mv2  = arith.mulf %m_i, %v2 : f64
    %ke   = arith.mulf %half, %mv2 : f64
    md_dist.yield %ke : f64
  } : !md_dist.partial<f64>
  %kd = md_dist.all_reduce %kp : !md_dist.partial<f64> -> f64
  %xglobal = md_dist.collect %xf, %lf : !own -> !vec
  %vglobal = md_dist.collect %vf, %lf : !own -> !vec
  %fglobal = md_dist.collect %ff, %lf : !own -> !vec

  //===--------------------------------------------------------------------===//
  // The comparison
  //===--------------------------------------------------------------------===//

  // The rebuilds of the halo: more than none and fewer than the steps, so
  // that both branches ran.
  call @printI64(%rebuilds) : (i64) -> ()
  call @printNewline() : () -> ()
  %steps64 = arith.index_cast %steps : index to i64
  %some = arith.cmpi sgt, %rebuilds, %c0i : i64
  %fewer = arith.cmpi slt, %rebuilds, %steps64 : i64
  %both = arith.andi %some, %fewer : i1
  %bothf = arith.uitofp %both : i1 to f64
  call @printF64(%bothf) : (f64) -> ()
  call @printNewline() : () -> ()

  %xa = mdrt.to_buffer %xglobal : !vec to memref<?x3xf64>
  %xb = mdrt.to_buffer %xr : !vec to memref<?x3xf64>
  %va = mdrt.to_buffer %vglobal : !vec to memref<?x3xf64>
  %vb = mdrt.to_buffer %vr : !vec to memref<?x3xf64>
  %fa = mdrt.to_buffer %fglobal : !vec to memref<?x3xf64>
  %fb = mdrt.to_buffer %fr : !vec to memref<?x3xf64>
  %dx = call @difference(%xa, %xb, %edge)
      : (memref<?x3xf64>, memref<?x3xf64>, f64) -> f64
  %dv = call @difference(%va, %vb, %zero)
      : (memref<?x3xf64>, memref<?x3xf64>, f64) -> f64
  %df = call @difference(%fa, %fb, %zero)
      : (memref<?x3xf64>, memref<?x3xf64>, f64) -> f64
  %vmax = call @largest(%vb) : (memref<?x3xf64>) -> f64
  %fmax = call @largest(%fb) : (memref<?x3xf64>) -> f64
  %tx = arith.constant XTOL : f64
  %tv = arith.constant VTOL : f64
  %te = arith.constant ETOL : f64
  %tf = arith.constant FTOL : f64
  // Positions: the largest difference, to the nearest image, against the
  // edge of the cell.
  call @check(%dx, %zero, %edge, %tx) : (f64, f64, f64, f64) -> ()
  // Velocities: against the largest component.
  call @check(%dv, %zero, %vmax, %tv) : (f64, f64, f64, f64) -> ()
  // Forces of the last step: against the largest component.
  call @check(%df, %zero, %fmax, %tf) : (f64, f64, f64, f64) -> ()
  // The potential and the kinetic energy, and the total, against the
  // kinetic energy.
  call @check(%ud, %ur, %kr, %te) : (f64, f64, f64, f64) -> ()
  call @check(%kd, %kr, %kr, %te) : (f64, f64, f64, f64) -> ()
  %td = arith.addf %ud, %kd : f64
  %tr = arith.addf %ur, %kr : f64
  call @check(%td, %tr, %kr, %te) : (f64, f64, f64, f64) -> ()
  // The drift of the total energy on domains since the start, against the
  // kinetic energy, and the same on one domain.
  %td_drift = arith.constant DTOL : f64
  call @check(%td, %t0, %kr, %td_drift) : (f64, f64, f64, f64) -> ()
  call @check(%tr, %t0, %kr, %td_drift) : (f64, f64, f64, f64) -> ()
  // The fields carried through the branches: the velocities, forces, and
  // masses exactly; the positions within 1e-15 of the edge, a few roundings
  // of the wrap x - n L (half an ulp of 4.8 is 9.3e-17 of it).
  %exact = arith.constant 0.0 : f64
  %wrap = arith.constant 1.0e-15 : f64
  call @check(%carried, %zero, %one, %exact) : (f64, f64, f64, f64) -> ()
  call @check(%carried_x, %zero, %edge, %wrap) : (f64, f64, f64, f64) -> ()
  call @printF64(%tr) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%kr) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%vmax) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%fmax) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
