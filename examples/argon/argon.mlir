// Liquid argon at constant energy.
//
// 864 atoms in a periodic cube, after A. Rahman, Phys. Rev. 136, A405
// (1964): the density is 1.374 g/cm³, and the atoms interact through the
// Lennard-Jones potential with sigma = 0.34 nm and epsilon / k_B = 120 K.
//
// Units: nm, ps, amu, and kJ/mol. They are consistent: 1 amu nm²/ps² is
// 1 kJ/mol.
//
// The atoms start on a face-centered cubic lattice with random velocities
// that correspond to 183 K. About half of the kinetic energy goes into the
// potential energy within the first picosecond, which leaves the liquid
// near 98 K. The original study was at 94 K.
//
// The program integrates 2000 steps of 5 fs with velocity Verlet and prints
// a row every 100 steps:
//
//   time (ps), potential energy, kinetic energy, total energy (kJ/mol),
//   temperature (K)
//
// The last row is the relative change of the total energy between the start
// and the end of the run.
//
// Run it with examples/argon/run.sh.

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.85) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(switch, from = 0.75) {
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

// One step. The program returns the potential energy of the new positions
// together with the state.
dyn.program @velocity_verlet(%x: !vec, %v: !vec, %f: !vec, %m: !real,
                             %cell: !md.cell, %dt: f64,
                             %eps: f64, %sigma: f64)
    -> (!vec, !vec, !vec, f64)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %c    = arith.constant 0.5 : f64
  %half = arith.mulf %c, %dt : f64
  %v1 = dyn.kick %v, %f, %m, %half : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  %u1, %f1 = md.evaluate @lj(%x1, %cell, %eps, %sigma)
      request [energy, forces]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec)
  %v2 = dyn.kick %v1, %f1, %m, %half : !vec
  dyn.return %x1, %v2, %f1, %u1 : !vec, !vec, !vec, f64
}

func.func private @printF64(f64)
func.func private @printComma()
func.func private @printNewline()

func.func @print_row(%time: f64, %potential: f64, %kinetic: f64,
                     %degrees: f64) {
  // k_B in kJ/(mol K).
  %boltzmann = arith.constant 0.0083144626 : f64
  %two = arith.constant 2.0 : f64
  %total = arith.addf %potential, %kinetic : f64
  %twice = arith.mulf %two, %kinetic : f64
  %scale = arith.mulf %degrees, %boltzmann : f64
  %temperature = arith.divf %twice, %scale : f64
  call @printF64(%time) : (f64) -> ()
  call @printComma() : () -> ()
  call @printF64(%potential) : (f64) -> ()
  call @printComma() : () -> ()
  call @printF64(%kinetic) : (f64) -> ()
  call @printComma() : () -> ()
  call @printF64(%total) : (f64) -> ()
  call @printComma() : () -> ()
  call @printF64(%temperature) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

// Places the atoms on a face-centered cubic lattice of `cells` unit cells
// along each direction, with the lattice constant `a`.
func.func @place(%x: memref<?x3xf64>, %cells: index, %a: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f64
  %half = arith.constant 0.5 : f64
  %quarter = arith.constant 0.25 : f64
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  scf.for %i = %c0 to %n step %c1 {
    %cell = arith.divui %i, %c4 : index
    %basis = arith.remui %i, %c4 : index
    %rest = arith.divui %cell, %cells : index
    %cx = arith.remui %cell, %cells : index
    %cy = arith.remui %rest, %cells : index
    %cz = arith.divui %rest, %cells : index
    scf.for %k = %c0 to %c3 step %c1 {
      %p0 = arith.cmpi eq, %k, %c0 : index
      %p1 = arith.cmpi eq, %k, %c1 : index
      %cyz = arith.select %p1, %cy, %cz : index
      %c = arith.select %p0, %cx, %cyz : index
      %wide = arith.index_cast %c : index to i64
      %site = arith.sitofp %wide : i64 to f64

      // Atom 0 of a unit cell is at its corner. Atom b, for b from 1 to 3,
      // is at the center of the face that is normal to direction 3 - b.
      %normal = arith.subi %c3, %basis : index
      %corner = arith.cmpi eq, %basis, %c0 : index
      %along = arith.cmpi eq, %k, %normal : index
      %none = arith.ori %corner, %along : i1
      %offset = arith.select %none, %zero, %half : f64

      %s0 = arith.addf %site, %offset : f64
      %s1 = arith.addf %s0, %quarter : f64
      %value = arith.mulf %s1, %a : f64
      memref.store %value, %x[%i, %k] : memref<?x3xf64>
    }
  }
  return
}

// Gives every component of every velocity a value between -amplitude and
// amplitude, and subtracts the mean, so that the center of mass is at rest.
func.func @thermalize(%v: memref<?x3xf64>, %amplitude: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 42 : i64
  %n = memref.dim %v, %c0 : memref<?x3xf64>
  %wide = arith.index_cast %n : index to i64
  %count = arith.sitofp %wide : i64 to f64

  scf.for %k = %c0 to %c3 step %c1 {
    %shift = arith.index_cast %k : index to i64
    %start = arith.addi %seed, %shift : i64
    %state, %sum = scf.for %i = %c0 to %n step %c1
        iter_args(%s = %start, %partial = %zero) -> (i64, f64) {
      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %fraction = arith.divf %u, %mf : f64
      %scaled = arith.mulf %fraction, %two : f64
      %centered = arith.subf %scaled, %one : f64
      %value = arith.mulf %centered, %amplitude : f64
      memref.store %value, %v[%i, %k] : memref<?x3xf64>
      %next = arith.addf %partial, %value : f64
      scf.yield %t2, %next : i64, f64
    }
    %mean = arith.divf %sum, %count : f64
    scf.for %i = %c0 to %n step %c1 {
      %value = memref.load %v[%i, %k] : memref<?x3xf64>
      %centered = arith.subf %value, %mean : f64
      memref.store %centered, %v[%i, %k] : memref<?x3xf64>
    }
  }
  return
}

func.func @fill(%x: memref<?xf64>, %value: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %x, %c0 : memref<?xf64>
  scf.for %i = %c0 to %n step %c1 {
    memref.store %value, %x[%i] : memref<?xf64>
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cells = arith.constant 6 : index
  %count = arith.constant 864 : index
  %rows = arith.constant 20 : index
  %steps = arith.constant 100 : index

  %mass = arith.constant 39.95 : f64
  %sigma = arith.constant 0.34 : f64
  // 120 K times k_B.
  %eps = arith.constant 0.99773551 : f64
  %edge = arith.constant 3.47786 : f64
  %lattice = arith.constant 0.57964333 : f64
  %dt = arith.constant 0.005 : f64
  // The time between two rows.
  %interval = arith.constant 0.5 : f64
  // Three per atom, less three for the center of mass.
  %degrees = arith.constant 2589.0 : f64
  // sqrt(3 k_B T / m) at 183 K.
  %amplitude = arith.constant 0.33802 : f64
  %zero = arith.constant 0.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  %masses = memref.alloc(%count) : memref<?xf64>
  %positions = memref.alloc(%count) : memref<?x3xf64>
  %velocities = memref.alloc(%count) : memref<?x3xf64>
  call @fill(%masses, %mass) : (memref<?xf64>, f64) -> ()
  call @place(%positions, %cells, %lattice)
      : (memref<?x3xf64>, index, f64) -> ()
  call @thermalize(%velocities, %amplitude) : (memref<?x3xf64>, f64) -> ()
  %m = mdrt.from_buffer %masses : memref<?xf64> to !real
  %x0 = mdrt.from_buffer %positions : memref<?x3xf64> to !vec
  %v0 = mdrt.from_buffer %velocities : memref<?x3xf64> to !vec

  %u0, %f0 = md.evaluate @lj(%x0, %cell, %eps, %sigma)
      request [energy, forces]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec)
  %k0 = md.sum_particles gather(%v0, %m : !vec, !real) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %half = arith.constant 0.5 : f64
    %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
    %mv2  = arith.mulf %m_i, %v2 : f64
    %ke   = arith.mulf %half, %mv2 : f64
    md.yield %ke : f64
  } : f64
  %e0 = arith.addf %u0, %k0 : f64
  call @print_row(%zero, %u0, %k0, %degrees) : (f64, f64, f64, f64) -> ()

  %x, %v, %f, %e = scf.for %row = %c0 to %rows step %c1
      iter_args(%xr = %x0, %vr = %v0, %fr = %f0, %er = %e0)
      -> (!vec, !vec, !vec, f64) {
    %xs, %vs, %fs, %us = scf.for %step = %c0 to %steps step %c1
        iter_args(%xa = %xr, %va = %vr, %fa = %fr, %ua = %zero)
        -> (!vec, !vec, !vec, f64) {
      %xb, %vb, %fb, %ub = dyn.step @velocity_verlet(
          %xa, %va, %fa, %m, %cell, %dt, %eps, %sigma)
          : (!vec, !vec, !vec, !real, !md.cell, f64, f64, f64)
          -> (!vec, !vec, !vec, f64)
      scf.yield %xb, %vb, %fb, %ub : !vec, !vec, !vec, f64
    }

    %k = md.sum_particles gather(%vs, %m : !vec, !real) {
    ^bb0(%v_i: vector<3xf64>, %m_i: f64):
      %half = arith.constant 0.5 : f64
      %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
      %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
      %mv2  = arith.mulf %m_i, %v2 : f64
      %ke   = arith.mulf %half, %mv2 : f64
      md.yield %ke : f64
    } : f64
    %total = arith.addf %us, %k : f64

    %next = arith.addi %row, %c1 : index
    %wide = arith.index_cast %next : index to i64
    %number = arith.sitofp %wide : i64 to f64
    %time = arith.mulf %number, %interval : f64
    func.call @print_row(%time, %us, %k, %degrees) : (f64, f64, f64, f64) -> ()
    scf.yield %xs, %vs, %fs, %total : !vec, !vec, !vec, f64
  }

  %change = arith.subf %e, %e0 : f64
  %relative = arith.divf %change, %e0 : f64
  %drift = math.absf %relative : f64
  call @printF64(%drift) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
