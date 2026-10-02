// Generalized Born of Onufriev, Bashford, and Case (OBC II) at the md
// level: the integral of the descreening of each particle by its
// neighbors [HawkinsCramerTruhlar], a gather over pairs whose kernel is not
// symmetric, the Born radii from it, a map over particles, and the energy,
// a sum over pairs that reads the radii and a sum over particles. The
// forces and the virial come through the radii, intermediate fields whose
// adjoints differentiation carries back to the gather (D143). The values are
// those of Inputs/obc_reference.py: the energy from the formulas, the
// forces from central differences, extrapolated, and the virial Σ x ⊗ F,
// at positions where no pair is within 0.03 nm of a branch of the integral,
// at which the differences would straddle a kink.
//
// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// The energy, the forces, and the virial agree, each printing 1.
// CHECK:      1
// CHECK-NEXT: 1
// CHECK-NEXT: 1

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.potential @obc(%x: !vec, %cell: !md.cell, %q: !real, %rt: !real,
                  %sr: !real, %rho: !real) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(4.0) : !vec -> !pairs
  // The integral of the descreening of i by j, 0 beyond its reach, with a
  // term of its own where i lies within the sphere of j.
  %integral = md.gather_relation %n, %x, %cell gather(%rt, %sr : !real, !real)
      exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %ri: f64, %rj: f64, %si: f64, %sj: f64):
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %two = arith.constant 2.0 : f64
    %quarter = arith.constant 0.25 : f64
    %half = arith.constant 0.5 : f64
    %upper = arith.addf %r, %sj : f64
    %diff = arith.subf %r, %sj : f64
    %adiff = math.absf %diff : f64
    %lower = arith.maximumf %ri, %adiff : f64
    %l = arith.divf %one, %lower : f64
    %u = arith.divf %one, %upper : f64
    %l2 = arith.mulf %l, %l : f64
    %u2 = arith.mulf %u, %u : f64
    %t0 = arith.subf %l, %u : f64
    %du = arith.subf %u2, %l2 : f64
    %qr = arith.mulf %quarter, %r : f64
    %t1 = arith.mulf %qr, %du : f64
    %ratio = arith.divf %u, %l : f64
    %log = math.log %ratio : f64
    %hr = arith.divf %half, %r : f64
    %t2 = arith.mulf %hr, %log : f64
    %sj2 = arith.mulf %sj, %sj : f64
    %qs = arith.mulf %quarter, %sj2 : f64
    %qsr = arith.divf %qs, %r : f64
    %dl = arith.subf %l2, %u2 : f64
    %t3 = arith.mulf %qsr, %dl : f64
    %a0 = arith.addf %t0, %t1 : f64
    %a1 = arith.addf %a0, %t2 : f64
    %a2 = arith.addf %a1, %t3 : f64
    // Inside the sphere of j.
    %inner = arith.subf %sj, %r : f64
    %isin = arith.cmpf olt, %ri, %inner : f64
    %iri = arith.divf %one, %ri : f64
    %e0 = arith.subf %iri, %l : f64
    %e1 = arith.mulf %two, %e0 : f64
    %a3 = arith.addf %a2, %e1 : f64
    %a4 = arith.select %isin, %a3, %a2 : f64
    %reach = arith.cmpf olt, %ri, %upper : f64
    %k = arith.select %reach, %a4, %zero : f64
    md.yield %k : f64
  } : !pairs, !vec -> !real
  // The Born radii (OBC II).
  %born = md.map_particles gather(%integral, %rt, %rho : !real, !real, !real) {
  ^bb0(%i: f64, %ri: f64, %r0: f64):
    %half = arith.constant 0.5 : f64
    %one = arith.constant 1.0 : f64
    %alpha = arith.constant 1.0 : f64
    %beta = arith.constant 0.8 : f64
    %gamma = arith.constant 4.85 : f64
    %hi = arith.mulf %half, %i : f64
    %psi = arith.mulf %hi, %ri : f64
    %psi2 = arith.mulf %psi, %psi : f64
    %psi3 = arith.mulf %psi2, %psi : f64
    %a = arith.mulf %alpha, %psi : f64
    %b = arith.mulf %beta, %psi2 : f64
    %c = arith.mulf %gamma, %psi3 : f64
    %s0 = arith.subf %a, %b : f64
    %s1 = arith.addf %s0, %c : f64
    %t = math.tanh %s1 : f64
    %tr = arith.divf %t, %r0 : f64
    %iri = arith.divf %one, %ri : f64
    %den = arith.subf %iri, %tr : f64
    %bi = arith.divf %one, %den : f64
    md.yield %bi : f64
  } : !real
  %pair = md.sum_relation %n, %x, %cell gather(%q, %born : !real, !real)
      exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qj: f64, %bi: f64, %bj: f64):
    %c = arith.constant -137.16557920267516 : f64
    %quarter = arith.constant 0.25 : f64
    %bb = arith.mulf %bi, %bj : f64
    %r2 = arith.mulf %r, %r : f64
    %bb4 = arith.divf %quarter, %bb : f64
    %x0 = arith.mulf %r2, %bb4 : f64
    %nx = arith.negf %x0 : f64
    %ex = math.exp %nx : f64
    %be = arith.mulf %bb, %ex : f64
    %s = arith.addf %r2, %be : f64
    %fgb = math.sqrt %s : f64
    %qq = arith.mulf %qi, %qj : f64
    %cq = arith.mulf %c, %qq : f64
    %e = arith.divf %cq, %fgb : f64
    md.yield %e : f64
  } : !pairs, !vec -> f64
  %self = md.sum_particles gather(%q, %born : !real, !real) {
  ^bb0(%qi: f64, %bi: f64):
    %c = arith.constant -68.58278960133758 : f64
    %qq = arith.mulf %qi, %qi : f64
    %cq = arith.mulf %c, %qq : f64
    %e = arith.divf %cq, %bi : f64
    md.yield %e : f64
  } : f64
  %u = arith.addf %pair, %self : f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

// 1 if every element of `values` is within `tolerance` of `reference`,
// relative to the largest element of the reference.
func.func @agrees(%values: memref<?xf64>, %reference: memref<?xf64>,
                  %tolerance: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %values, %c0 : memref<?xf64>
  %largest = scf.for %i = %c0 to %n step %c1 iter_args(%m = %zero) -> (f64) {
    %v = memref.load %reference[%i] : memref<?xf64>
    %a = math.absf %v : f64
    %b = arith.maximumf %m, %a : f64
    scf.yield %b : f64
  }
  %worst = scf.for %i = %c0 to %n step %c1 iter_args(%m = %zero) -> (f64) {
    %v = memref.load %values[%i] : memref<?xf64>
    %r = memref.load %reference[%i] : memref<?xf64>
    %d = arith.subf %v, %r : f64
    %a = math.absf %d : f64
    %b = arith.maximumf %m, %a : f64
    scf.yield %b : f64
  }
  %bound = arith.mulf %tolerance, %largest : f64
  %ok = arith.cmpf ole, %worst, %bound : f64
  %flag = arith.uitofp %ok : i1 to f64
  return %flag : f64
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %count = arith.constant 5 : index
  %xb = memref.alloc(%count) : memref<?x3xf64>
  %qb = memref.alloc(%count) : memref<?xf64>
  %rtb = memref.alloc(%count) : memref<?xf64>
  %srb = memref.alloc(%count) : memref<?xf64>
  %rhob = memref.alloc(%count) : memref<?xf64>
  %xs = arith.constant dense<[0.09,0.18,0.29, 0.27,0.20,0.31, 0.33,0.36,0.29, 0.20,0.42,0.45, 0.41,0.17,0.51]> : vector<15xf64>
  %qs = arith.constant dense<[-0.5, 0.3, 0.25, -0.35, 0.3]> : vector<5xf64>
  %rhos = arith.constant dense<[0.17, 0.12, 0.155, 0.15, 0.13]> : vector<5xf64>
  %ss = arith.constant dense<[0.72, 0.85, 0.72, 0.79, 0.85]> : vector<5xf64>
  %off = arith.constant 0.009 : f64
  scf.for %i = %c0 to %count step %c1 {
    %ii = arith.index_cast %i : index to i64
    %i32 = arith.index_cast %i : index to i32
    %q = vector.extract %qs[%i] : f64 from vector<5xf64>
    memref.store %q, %qb[%i] : memref<?xf64>
    %rho = vector.extract %rhos[%i] : f64 from vector<5xf64>
    memref.store %rho, %rhob[%i] : memref<?xf64>
    %rt = arith.subf %rho, %off : f64
    memref.store %rt, %rtb[%i] : memref<?xf64>
    %s = vector.extract %ss[%i] : f64 from vector<5xf64>
    %sr = arith.mulf %s, %rt : f64
    memref.store %sr, %srb[%i] : memref<?xf64>
    %c3 = arith.constant 3 : index
    %base = arith.muli %i, %c3 : index
    scf.for %k = %c0 to %c3 step %c1 {
      %p = arith.addi %base, %k : index
      %v = vector.extract %xs[%p] : f64 from vector<15xf64>
      memref.store %v, %xb[%i, %k] : memref<?x3xf64>
    }
  }
  %x = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
  %q = mdrt.from_buffer %qb : memref<?xf64> to !real
  %rt = mdrt.from_buffer %rtb : memref<?xf64> to !real
  %sr = mdrt.from_buffer %srb : memref<?xf64> to !real
  %rho = mdrt.from_buffer %rhob : memref<?xf64> to !real
  %edge = arith.constant 10.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @obc(%x, %cell, %q, %rt, %sr, %rho)
      request [energy, forces, virial]
      : (!vec, !md.cell, !real, !real, !real, !real) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to memref<?x3xf64>

  %c15 = arith.constant 15 : index
  %c9 = arith.constant 9 : index
  %one = arith.constant 1 : index
  %ub = memref.alloc(%one) : memref<?xf64>
  %ur = memref.alloc(%one) : memref<?xf64>
  memref.store %u, %ub[%c0] : memref<?xf64>
  %u_ref = arith.constant -146.48786605464932 : f64
  memref.store %u_ref, %ur[%c0] : memref<?xf64>
  %fv = memref.alloc(%c15) : memref<?xf64>
  %fr = memref.alloc(%c15) : memref<?xf64>
  %f_ref = arith.constant dense<[-488.52533770348333, 14.114545687059868, -15.733642832695219, 493.44860059666945, 12.384413341758696, 44.004213892634425, 199.16514606142263, -140.07370128046168, -42.860913201820949, -286.63740538069266, 137.40567889395075, 95.882297394969854, 82.548996426557622, -23.830936641928702, -81.291955252188089]> : vector<15xf64>
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %count step %c1 {
    scf.for %k = %c0 to %c3 step %c1 {
      %p0 = arith.muli %i, %c3 : index
      %p = arith.addi %p0, %k : index
      %v = memref.load %fb[%i, %k] : memref<?x3xf64>
      memref.store %v, %fv[%p] : memref<?xf64>
      %r = vector.extract %f_ref[%p] : f64 from vector<15xf64>
      memref.store %r, %fr[%p] : memref<?xf64>
    }
  }
  %wv = memref.alloc(%c9) : memref<?xf64>
  %wr = memref.alloc(%c9) : memref<?xf64>
  %w_ref = arith.constant dense<[131.50594742680681, -23.899768952842738, -17.832233634935331, -23.899768952557075, 8.2500943373877504, 16.989790829001571, -17.832233634997408, 16.989790828752881, -1.662978294172504]> : vector<9xf64>
  scf.for %k = %c0 to %c9 step %c1 {
    %v = vector.extract %w[%k] : f64 from vector<9xf64>
    memref.store %v, %wv[%k] : memref<?xf64>
    %r = vector.extract %w_ref[%k] : f64 from vector<9xf64>
    memref.store %r, %wr[%k] : memref<?xf64>
  }
  %tight = arith.constant 1.0e-12 : f64
  %loose = arith.constant 1.0e-9 : f64
  %differences = arith.constant 1.0e-9 : f64
  %e_ok = func.call @agrees(%ub, %ur, %tight) : (memref<?xf64>, memref<?xf64>, f64) -> f64
  %f_ok = func.call @agrees(%fv, %fr, %loose) : (memref<?xf64>, memref<?xf64>, f64) -> f64
  %w_ok = func.call @agrees(%wv, %wr, %differences) : (memref<?xf64>, memref<?xf64>, f64) -> f64
  func.call @printF64(%e_ok) : (f64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printF64(%f_ok) : (f64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printF64(%w_ok) : (f64) -> ()
  func.call @printNewline() : () -> ()
  return
}
