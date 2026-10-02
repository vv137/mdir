// The reciprocal sum of smooth particle mesh Ewald on the host
// [Essmann1995]: spreading the charges to a grid with B-splines, the product
// of the transform of the grid with the influence function, and the forces
// from the potential on the grid (docs/pme-m1.md). The keys are those of
// docs/references.md. The FFT between the steps is a call of the runtime.
//
// This is a template. The compiler adds it to a module that has a
// reciprocal sum, where it is specialized like any other code. `!pme_pos`,
// `!pme_chg`, and `!pme_frc` are the types that the positions, the charges,
// and the forces are stored in; the instance replaces them, and the ops
// PME_EXTEND_POS, PME_EXTEND_CHG, and PME_NARROW_FRC with the conversions
// between them and f64. The arithmetic is in f64.
//
// The grid has k1 x k2 x k3 points, the third index the fastest. A particle
// at the fractional coordinate u (from 0 to k along each edge) adds to the
// points floor(u) − n + 1 + j, j = 0 .. n − 1, taken modulo k, with the
// weights M_n(u − floor(u) + n − 1 − j) of the cardinal B-spline of order n.
// The charges are added in fixed point, at the scale 2^40, so that the sum
// does not depend on the order in which they are added (D70).

!pme_pos = f64
!pme_chg = f64
!pme_frc = f64

// Raises the weights `a` of the B-spline at the fraction `w` from order
// k − 1 to order k, in place (the recursion of Essmann et al., Eq. 4.1).
func.func private @mdrt.pme_raise(%w: f64, %k: index, %a: memref<8xf64>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %one = arith.constant 1.0 : f64
  %km1 = arith.subi %k, %c1 : index
  %km2 = arith.subi %k, %c2 : index
  %km1i = arith.index_cast %km1 : index to i64
  %km1f = arith.sitofp %km1i : i64 to f64
  %div = arith.divf %one, %km1f : f64
  %ki = arith.index_cast %k : index to i64
  %kf = arith.sitofp %ki : i64 to f64
  // a[k − 1] = div · w · a[k − 2]
  %last = memref.load %a[%km2] : memref<8xf64>
  %dw = arith.mulf %div, %w : f64
  %top = arith.mulf %dw, %last : f64
  memref.store %top, %a[%km1] : memref<8xf64>
  // a[k − j − 1] = div ((w + j) a[k − j − 2] + (k − j − w) a[k − j − 1])
  scf.for %j = %c1 to %km1 step %c1 {
    %ji = arith.index_cast %j : index to i64
    %jf = arith.sitofp %ji : i64 to f64
    %at = arith.subi %km1, %j : index
    %below = arith.subi %at, %c1 : index
    %lo = memref.load %a[%below] : memref<8xf64>
    %hi = memref.load %a[%at] : memref<8xf64>
    %wj = arith.addf %w, %jf : f64
    %kj = arith.subf %kf, %jf : f64
    %kjw = arith.subf %kj, %w : f64
    %t1 = arith.mulf %wj, %lo : f64
    %t2 = arith.mulf %kjw, %hi : f64
    %sum = arith.addf %t1, %t2 : f64
    %value = arith.mulf %div, %sum : f64
    memref.store %value, %a[%at] : memref<8xf64>
  }
  // a[0] = div · (1 − w) · a[0]
  %c0 = arith.constant 0 : index
  %first = memref.load %a[%c0] : memref<8xf64>
  %omw = arith.subf %one, %w : f64
  %d0 = arith.mulf %div, %omw : f64
  %bottom = arith.mulf %d0, %first : f64
  memref.store %bottom, %a[%c0] : memref<8xf64>
  return
}

// The weights `values` of the B-spline of order `n` at the fraction `w`,
// and their derivatives `slopes` with respect to the fractional coordinate.
func.func private @mdrt.pme_bspline(%w: f64, %n: index, %values: memref<8xf64>,
                                    %slopes: memref<8xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  scf.for %j = %c0 to %c8 step %c1 {
    memref.store %zero, %values[%j] : memref<8xf64>
  }
  %omw = arith.subf %one, %w : f64
  memref.store %omw, %values[%c0] : memref<8xf64>
  memref.store %w, %values[%c1] : memref<8xf64>
  scf.for %k = %c3 to %n step %c1 {
    func.call @mdrt.pme_raise(%w, %k, %values) : (f64, index, memref<8xf64>) -> ()
  }
  // The derivative of M_n is M_{n−1}(x) − M_{n−1}(x − 1).
  %v0 = memref.load %values[%c0] : memref<8xf64>
  %s0 = arith.negf %v0 : f64
  memref.store %s0, %slopes[%c0] : memref<8xf64>
  scf.for %j = %c1 to %n step %c1 {
    %jm1 = arith.subi %j, %c1 : index
    %before = memref.load %values[%jm1] : memref<8xf64>
    %here = memref.load %values[%j] : memref<8xf64>
    %slope = arith.subf %before, %here : f64
    memref.store %slope, %slopes[%j] : memref<8xf64>
  }
  func.call @mdrt.pme_raise(%w, %n, %values) : (f64, index, memref<8xf64>) -> ()
  return
}

// The first point of the grid that a particle at `x` along an edge of the
// length `length` with `k` points adds to, plus k so that it is not
// negative, and the fraction of its coordinate.
func.func private @mdrt.pme_place(%x: f64, %length: f64, %k: index, %n: index)
    -> (index, f64) {
  %c1 = arith.constant 1 : index
  %s = arith.divf %x, %length : f64
  %fs = math.floor %s : f64
  %frac = arith.subf %s, %fs : f64
  %ki = arith.index_cast %k : index to i64
  %kf = arith.sitofp %ki : i64 to f64
  %u = arith.mulf %frac, %kf : f64
  %fu = math.floor %u : f64
  %w = arith.subf %u, %fu : f64
  %bi = arith.fptosi %fu : f64 to i64
  %b = arith.index_cast %bi : i64 to index
  // floor(u) − n + 1 + k
  %bk = arith.addi %b, %k : index
  %bkn = arith.subi %bk, %n : index
  %start = arith.addi %bkn, %c1 : index
  return %start, %w : index, f64
}

// Spreads the charges to `grid`, in fixed point.
func.func private @mdrt.pme_spread(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %box: vector<3xf64>, %grid: memref<?xi64>,
                                   %k1: index, %k2: index, %k3: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %none = arith.constant 0 : i64
  %scale = arith.constant 1099511627776.0 : f64
  %points = memref.dim %grid, %c0 : memref<?xi64>
  scf.for %p = %c0 to %points step %c1 {
    memref.store %none, %grid[%p] : memref<?xi64>
  }
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %wx = memref.alloca() : memref<8xf64>
  %wy = memref.alloca() : memref<8xf64>
  %wz = memref.alloca() : memref<8xf64>
  %dx = memref.alloca() : memref<8xf64>
  %dy = memref.alloca() : memref<8xf64>
  %dz = memref.alloca() : memref<8xf64>
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.for %i = %c0 to %count step %c1 {
    %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
    %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
    %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
    %qs = memref.load %q[%i] : memref<?x!pme_chg>
    %xi = PME_EXTEND_POS %xs : !pme_pos to f64
    %yi = PME_EXTEND_POS %ys : !pme_pos to f64
    %zi = PME_EXTEND_POS %zs : !pme_pos to f64
    %qi = PME_EXTEND_CHG %qs : !pme_chg to f64
    %sx, %fx = func.call @mdrt.pme_place(%xi, %lx, %k1, %n) : (f64, f64, index, index) -> (index, f64)
    %sy, %fy = func.call @mdrt.pme_place(%yi, %ly, %k2, %n) : (f64, f64, index, index) -> (index, f64)
    %sz, %fz = func.call @mdrt.pme_place(%zi, %lz, %k3, %n) : (f64, f64, index, index) -> (index, f64)
    func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
    func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
    func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
    %qscaled = arith.mulf %qi, %scale : f64
    scf.for %j1 = %c0 to %n step %c1 {
      %g1s = arith.addi %sx, %j1 : index
      %g1 = arith.remui %g1s, %k1 : index
      %w1 = memref.load %wx[%j1] : memref<8xf64>
      %q1 = arith.mulf %qscaled, %w1 : f64
      scf.for %j2 = %c0 to %n step %c1 {
        %g2s = arith.addi %sy, %j2 : index
        %g2 = arith.remui %g2s, %k2 : index
        %w2 = memref.load %wy[%j2] : memref<8xf64>
        %q12 = arith.mulf %q1, %w2 : f64
        %row1 = arith.muli %g1, %k2 : index
        %row = arith.addi %row1, %g2 : index
        %base = arith.muli %row, %k3 : index
        scf.for %j3 = %c0 to %n step %c1 {
          %g3s = arith.addi %sz, %j3 : index
          %g3 = arith.remui %g3s, %k3 : index
          %w3 = memref.load %wz[%j3] : memref<8xf64>
          %value = arith.mulf %q12, %w3 : f64
          %rounded = math.roundeven %value : f64
          %fixed = arith.fptosi %rounded : f64 to i64
          %at = arith.addi %base, %g3 : index
          %old = memref.load %grid[%at] : memref<?xi64>
          %new = arith.addi %old, %fixed : i64
          memref.store %new, %grid[%at] : memref<?xi64>
        }
      }
    }
  }
  return
}

// The grid in f64, from fixed point.
func.func private @mdrt.pme_real(%grid: memref<?xi64>, %real: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %inverse = arith.constant 9.094947017729282e-13 : f64
  %points = memref.dim %grid, %c0 : memref<?xi64>
  scf.parallel (%p) = (%c0) to (%points) step (%c1) {
    %fixed = memref.load %grid[%p] : memref<?xi64>
    %value = arith.sitofp %fixed : i64 to f64
    %scaled = arith.mulf %value, %inverse : f64
    memref.store %scaled, %real[%p] : memref<?xf64>
    scf.reduce
  }
  return
}

// The component of the wave vector of index `k` of `count` along an edge of
// the length `length`: k / L, or (k − count) / L beyond count / 2.
func.func private @mdrt.pme_wave(%k: index, %count: index, %length: f64) -> f64 {
  %c2 = arith.constant 2 : index
  %half = arith.divui %count, %c2 : index
  %ki = arith.index_cast %k : index to i64
  %ni = arith.index_cast %count : index to i64
  %beyond = arith.cmpi ugt, %k, %half : index
  %shifted = arith.subi %ki, %ni : i64
  %signed = arith.select %beyond, %shifted, %ki : i64
  %sf = arith.sitofp %signed : i64 to f64
  %m = arith.divf %sf, %length : f64
  return %m : f64
}

// Multiplies the half-complex transform `c` by the influence function
//
//   B C(m) = f / (π V) · exp(−π² m² / β²) / m² · B_1(k_1) B_2(k_2) B_3(k_3)
//
// with the factors B_a of each edge in the rows of `moduli`, and returns the
// energy and the virial
//
//   E = ½ Σ_m w(m) B C(m) |G(m)|²
//   W_ab = Σ_m E_m (δ_ab − 2 (1 / m² + π² / β²) m_a m_b)
//
// with w(m) = 2 for the points whose conjugate is not stored.
func.func private @mdrt.pme_convolve(%c: memref<?xf64>, %moduli: memref<?x?xf64>,
                                     %box: vector<3xf64>, %beta: f64, %coulomb: f64,
                                     %k1: index, %k2: index, %k3: index)
    -> (f64, vector<9xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %half = arith.constant 0.5 : f64
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %pi2 = arith.constant 9.869604401089358 : f64
  %none = arith.constant dense<0.0> : vector<9xf64>
  %identity = arith.constant dense<[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]> : vector<9xf64>
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %beta2 = arith.mulf %beta, %beta : f64
  %gauss = arith.divf %pi2, %beta2 : f64
  %pi = arith.constant 3.141592653589793 : f64
  %lxy = arith.mulf %lx, %ly : f64
  %volume = arith.mulf %lxy, %lz : f64
  %piv = arith.mulf %pi, %volume : f64
  %prefactor = arith.divf %coulomb, %piv : f64
  %h = arith.divui %k3, %c2 : index
  %h1 = arith.addi %h, %c1 : index
  %odd = arith.remui %k3, %c2 : index
  %even = arith.cmpi eq, %odd, %c0 : index
  %energy, %virial = scf.for %a = %c0 to %k1 step %c1
      iter_args(%ea = %zero, %wa = %none) -> (f64, vector<9xf64>) {
    %m1 = func.call @mdrt.pme_wave(%a, %k1, %lx) : (index, index, f64) -> f64
    %mod1 = memref.load %moduli[%c0, %a] : memref<?x?xf64>
    %eb, %wb = scf.for %b = %c0 to %k2 step %c1
        iter_args(%e2 = %ea, %w2 = %wa) -> (f64, vector<9xf64>) {
      %m2 = func.call @mdrt.pme_wave(%b, %k2, %ly) : (index, index, f64) -> f64
      %mod2 = memref.load %moduli[%c1, %b] : memref<?x?xf64>
      %mod12 = arith.mulf %mod1, %mod2 : f64
      %row1 = arith.muli %a, %k2 : index
      %row = arith.addi %row1, %b : index
      %ec, %wc = scf.for %z = %c0 to %h1 step %c1
          iter_args(%e3 = %e2, %w3 = %w2) -> (f64, vector<9xf64>) {
        %zi = arith.index_cast %z : index to i64
        %zf = arith.sitofp %zi : i64 to f64
        %m3 = arith.divf %zf, %lz : f64
        %mod3 = memref.load %moduli[%c2, %z] : memref<?x?xf64>
        %m1s = arith.mulf %m1, %m1 : f64
        %m2s = arith.mulf %m2, %m2 : f64
        %m3s = arith.mulf %m3, %m3 : f64
        %m12 = arith.addf %m1s, %m2s : f64
        %msq = arith.addf %m12, %m3s : f64
        %origin = arith.cmpf oeq, %msq, %zero : f64
        %safe = arith.select %origin, %one, %msq : f64
        %inverse = arith.divf %one, %safe : f64
        %gm = arith.mulf %gauss, %msq : f64
        %ngm = arith.negf %gm : f64
        %ex = math.exp %ngm : f64
        %exm = arith.mulf %ex, %inverse : f64
        %pexm = arith.mulf %prefactor, %exm : f64
        %mods = arith.mulf %mod12, %mod3 : f64
        %bc0 = arith.mulf %pexm, %mods : f64
        %bc = arith.select %origin, %zero, %bc0 : f64
        %base1 = arith.muli %row, %h1 : index
        %base2 = arith.addi %base1, %z : index
        %re_at = arith.muli %base2, %c2 : index
        %im_at = arith.addi %re_at, %c1 : index
        %re = memref.load %c[%re_at] : memref<?xf64>
        %im = memref.load %c[%im_at] : memref<?xf64>
        %re2 = arith.mulf %re, %re : f64
        %im2 = arith.mulf %im, %im : f64
        %g2 = arith.addf %re2, %im2 : f64
        // w = 1 for the plane z = 0 and, with k3 even, z = k3 / 2.
        %first = arith.cmpi eq, %z, %c0 : index
        %middle = arith.cmpi eq, %z, %h : index
        %middle_even = arith.andi %middle, %even : i1
        %single = arith.ori %first, %middle_even : i1
        %weight = arith.select %single, %one, %two : f64
        %hw = arith.mulf %half, %weight : f64
        %hwb = arith.mulf %hw, %bc : f64
        %em = arith.mulf %hwb, %g2 : f64
        // The virial of the point; nothing at m = 0, whose influence is 0.
        %sum = arith.addf %inverse, %gauss : f64
        %factor0 = arith.mulf %two, %sum : f64
        %factor = arith.select %origin, %zero, %factor0 : f64
        %mv = vector.from_elements %m1, %m2, %m3 : vector<3xf64>
        %mm = vector.outerproduct %mv, %mv : vector<3xf64>, vector<3xf64>
        %mmf = vector.shape_cast %mm : vector<3x3xf64> to vector<9xf64>
        %fb = vector.broadcast %factor : f64 to vector<9xf64>
        %scaled = arith.mulf %fb, %mmf : vector<9xf64>
        %diff = arith.subf %identity, %scaled : vector<9xf64>
        %eb9 = vector.broadcast %em : f64 to vector<9xf64>
        %wm = arith.mulf %eb9, %diff : vector<9xf64>
        %w3n = arith.addf %w3, %wm : vector<9xf64>
        %e3n = arith.addf %e3, %em : f64
        // The product with the influence function, for the forces.
        %reb = arith.mulf %re, %bc : f64
        %imb = arith.mulf %im, %bc : f64
        memref.store %reb, %c[%re_at] : memref<?xf64>
        memref.store %imb, %c[%im_at] : memref<?xf64>
        scf.yield %e3n, %w3n : f64, vector<9xf64>
      }
      scf.yield %ec, %wc : f64, vector<9xf64>
    }
    scf.yield %eb, %wb : f64, vector<9xf64>
  }
  return %energy, %virial : f64, vector<9xf64>
}

// The force on particle `i`, F_i = −q_i Σ_k φ(k) ∇_i θ_i(k), from the
// potential `phi` on the grid, which is the backward transform of the
// product of the transform of the charges with the influence function. A
// function of its own, so that its arrays live on the stack of a call.
func.func private @mdrt.pme_gather_one(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                       %phi: memref<?xf64>, %box: vector<3xf64>,
                                       %k1: index, %k2: index, %k3: index, %n: index,
                                       %f: memref<?x3x!pme_frc>, %i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %k1i = arith.index_cast %k1 : index to i64
  %k2i = arith.index_cast %k2 : index to i64
  %k3i = arith.index_cast %k3 : index to i64
  %k1f = arith.sitofp %k1i : i64 to f64
  %k2f = arith.sitofp %k2i : i64 to f64
  %k3f = arith.sitofp %k3i : i64 to f64
  %rx = arith.divf %k1f, %lx : f64
  %ry = arith.divf %k2f, %ly : f64
  %rz = arith.divf %k3f, %lz : f64
  %wx = memref.alloca() : memref<8xf64>
  %wy = memref.alloca() : memref<8xf64>
  %wz = memref.alloca() : memref<8xf64>
  %dx = memref.alloca() : memref<8xf64>
  %dy = memref.alloca() : memref<8xf64>
  %dz = memref.alloca() : memref<8xf64>
  %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
  %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
  %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
  %qs = memref.load %q[%i] : memref<?x!pme_chg>
  %xi = PME_EXTEND_POS %xs : !pme_pos to f64
  %yi = PME_EXTEND_POS %ys : !pme_pos to f64
  %zi = PME_EXTEND_POS %zs : !pme_pos to f64
  %qi = PME_EXTEND_CHG %qs : !pme_chg to f64
  %sx, %fx = func.call @mdrt.pme_place(%xi, %lx, %k1, %n) : (f64, f64, index, index) -> (index, f64)
  %sy, %fy = func.call @mdrt.pme_place(%yi, %ly, %k2, %n) : (f64, f64, index, index) -> (index, f64)
  %sz, %fz = func.call @mdrt.pme_place(%zi, %lz, %k3, %n) : (f64, f64, index, index) -> (index, f64)
  func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
  func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
  func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
  %gx, %gy, %gz = scf.for %j1 = %c0 to %n step %c1
      iter_args(%ax = %zero, %ay = %zero, %az = %zero) -> (f64, f64, f64) {
    %g1s = arith.addi %sx, %j1 : index
    %g1 = arith.remui %g1s, %k1 : index
    %w1 = memref.load %wx[%j1] : memref<8xf64>
    %d1 = memref.load %dx[%j1] : memref<8xf64>
    %bx, %by, %bz = scf.for %j2 = %c0 to %n step %c1
        iter_args(%cx = %ax, %cy = %ay, %cz = %az) -> (f64, f64, f64) {
      %g2s = arith.addi %sy, %j2 : index
      %g2 = arith.remui %g2s, %k2 : index
      %w2 = memref.load %wy[%j2] : memref<8xf64>
      %d2 = memref.load %dy[%j2] : memref<8xf64>
      %row1 = arith.muli %g1, %k2 : index
      %row = arith.addi %row1, %g2 : index
      %base = arith.muli %row, %k3 : index
      %d1w2 = arith.mulf %d1, %w2 : f64
      %w1d2 = arith.mulf %w1, %d2 : f64
      %w1w2 = arith.mulf %w1, %w2 : f64
      %ex, %ey, %ez = scf.for %j3 = %c0 to %n step %c1
          iter_args(%tx = %cx, %ty = %cy, %tz = %cz) -> (f64, f64, f64) {
        %g3s = arith.addi %sz, %j3 : index
        %g3 = arith.remui %g3s, %k3 : index
        %w3 = memref.load %wz[%j3] : memref<8xf64>
        %d3 = memref.load %dz[%j3] : memref<8xf64>
        %at = arith.addi %base, %g3 : index
        %p = memref.load %phi[%at] : memref<?xf64>
        %px = arith.mulf %d1w2, %w3 : f64
        %py = arith.mulf %w1d2, %w3 : f64
        %pz = arith.mulf %w1w2, %d3 : f64
        %vx = arith.mulf %p, %px : f64
        %vy = arith.mulf %p, %py : f64
        %vz = arith.mulf %p, %pz : f64
        %nx = arith.addf %tx, %vx : f64
        %ny = arith.addf %ty, %vy : f64
        %nz = arith.addf %tz, %vz : f64
        scf.yield %nx, %ny, %nz : f64, f64, f64
      }
      scf.yield %ex, %ey, %ez : f64, f64, f64
    }
    scf.yield %bx, %by, %bz : f64, f64, f64
  }
  %mq = arith.negf %qi : f64
  %sxq = arith.mulf %mq, %rx : f64
  %syq = arith.mulf %mq, %ry : f64
  %szq = arith.mulf %mq, %rz : f64
  %fx64 = arith.mulf %sxq, %gx : f64
  %fy64 = arith.mulf %syq, %gy : f64
  %fz64 = arith.mulf %szq, %gz : f64
  %fxs = PME_NARROW_FRC %fx64 : f64 to !pme_frc
  %fys = PME_NARROW_FRC %fy64 : f64 to !pme_frc
  %fzs = PME_NARROW_FRC %fz64 : f64 to !pme_frc
  memref.store %fxs, %f[%i, %c0] : memref<?x3x!pme_frc>
  memref.store %fys, %f[%i, %c1] : memref<?x3x!pme_frc>
  memref.store %fzs, %f[%i, %c2] : memref<?x3x!pme_frc>
  return
}

// The forces on all particles.
func.func private @mdrt.pme_gather(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %phi: memref<?xf64>, %box: vector<3xf64>,
                                   %k1: index, %k2: index, %k3: index, %n: index,
                                   %f: memref<?x3x!pme_frc>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    func.call @mdrt.pme_gather_one(%x, %q, %phi, %box, %k1, %k2, %k3, %n, %f, %i)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, memref<?xf64>, vector<3xf64>,
           index, index, index, index, memref<?x3x!pme_frc>, index) -> ()
    scf.reduce
  }
  return
}

// Spreads the charges to `grid` in a triclinic cell, `box` = (a_x, b_y,
// c_z, b_x, c_x, c_y), from the fractional coordinates s = x H⁻¹.
func.func private @mdrt.pme_spread_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %box: vector<6xf64>, %grid: memref<?xi64>,
                                   %k1: index, %k2: index, %k3: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %none = arith.constant 0 : i64
  %scale = arith.constant 1099511627776.0 : f64
  %points = memref.dim %grid, %c0 : memref<?xi64>
  scf.for %p = %c0 to %points step %c1 {
    memref.store %none, %grid[%p] : memref<?xi64>
  }
  %lx = vector.extract %box[0] : f64 from vector<6xf64>
  %ly = vector.extract %box[1] : f64 from vector<6xf64>
  %lz = vector.extract %box[2] : f64 from vector<6xf64>
  %tbx = vector.extract %box[3] : f64 from vector<6xf64>
  %tcx = vector.extract %box[4] : f64 from vector<6xf64>
  %tcy = vector.extract %box[5] : f64 from vector<6xf64>
  %unit = arith.constant 1.0 : f64
  // The rows of H⁻¹: h00 = 1/a_x; h10 = −b_x/(a_x b_y), h11 = 1/b_y;
  // h20 = (b_x c_y − b_y c_x)/(a_x b_y c_z), h21 = −c_y/(b_y c_z),
  // h22 = 1/c_z.
  %h00 = arith.divf %unit, %lx : f64
  %h11 = arith.divf %unit, %ly : f64
  %h22 = arith.divf %unit, %lz : f64
  %bx_h00 = arith.mulf %tbx, %h00 : f64
  %bx_h00_h11 = arith.mulf %bx_h00, %h11 : f64
  %h10 = arith.negf %bx_h00_h11 : f64
  %cy_h11 = arith.mulf %tcy, %h11 : f64
  %cy_h11_h22 = arith.mulf %cy_h11, %h22 : f64
  %h21 = arith.negf %cy_h11_h22 : f64
  %bxcy = arith.mulf %tbx, %tcy : f64
  %bycx = arith.mulf %ly, %tcx : f64
  %h20n = arith.subf %bxcy, %bycx : f64
  %h20a = arith.mulf %h20n, %h00 : f64
  %h20b = arith.mulf %h20a, %h11 : f64
  %h20 = arith.mulf %h20b, %h22 : f64
  %wx = memref.alloca() : memref<8xf64>
  %wy = memref.alloca() : memref<8xf64>
  %wz = memref.alloca() : memref<8xf64>
  %dx = memref.alloca() : memref<8xf64>
  %dy = memref.alloca() : memref<8xf64>
  %dz = memref.alloca() : memref<8xf64>
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.for %i = %c0 to %count step %c1 {
    %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
    %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
    %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
    %qs = memref.load %q[%i] : memref<?x!pme_chg>
    %xi = PME_EXTEND_POS %xs : !pme_pos to f64
    %yi = PME_EXTEND_POS %ys : !pme_pos to f64
    %zi = PME_EXTEND_POS %zs : !pme_pos to f64
    %qi = PME_EXTEND_CHG %qs : !pme_chg to f64
    // s = x H⁻¹.
    %s3 = arith.mulf %zi, %h22 : f64
    %s2y = arith.mulf %yi, %h11 : f64
    %s2z = arith.mulf %zi, %h21 : f64
    %s2 = arith.addf %s2y, %s2z : f64
    %s1x = arith.mulf %xi, %h00 : f64
    %s1y = arith.mulf %yi, %h10 : f64
    %s1z = arith.mulf %zi, %h20 : f64
    %s1xy = arith.addf %s1x, %s1y : f64
    %s1 = arith.addf %s1xy, %s1z : f64
    %sx, %fx = func.call @mdrt.pme_place(%s1, %unit, %k1, %n) : (f64, f64, index, index) -> (index, f64)
    %sy, %fy = func.call @mdrt.pme_place(%s2, %unit, %k2, %n) : (f64, f64, index, index) -> (index, f64)
    %sz, %fz = func.call @mdrt.pme_place(%s3, %unit, %k3, %n) : (f64, f64, index, index) -> (index, f64)
    func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
    func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
    func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
    %qscaled = arith.mulf %qi, %scale : f64
    scf.for %j1 = %c0 to %n step %c1 {
      %g1s = arith.addi %sx, %j1 : index
      %g1 = arith.remui %g1s, %k1 : index
      %w1 = memref.load %wx[%j1] : memref<8xf64>
      %q1 = arith.mulf %qscaled, %w1 : f64
      scf.for %j2 = %c0 to %n step %c1 {
        %g2s = arith.addi %sy, %j2 : index
        %g2 = arith.remui %g2s, %k2 : index
        %w2 = memref.load %wy[%j2] : memref<8xf64>
        %q12 = arith.mulf %q1, %w2 : f64
        %row1 = arith.muli %g1, %k2 : index
        %row = arith.addi %row1, %g2 : index
        %base = arith.muli %row, %k3 : index
        scf.for %j3 = %c0 to %n step %c1 {
          %g3s = arith.addi %sz, %j3 : index
          %g3 = arith.remui %g3s, %k3 : index
          %w3 = memref.load %wz[%j3] : memref<8xf64>
          %value = arith.mulf %q12, %w3 : f64
          %rounded = math.roundeven %value : f64
          %fixed = arith.fptosi %rounded : f64 to i64
          %at = arith.addi %base, %g3 : index
          %old = memref.load %grid[%at] : memref<?xi64>
          %new = arith.addi %old, %fixed : i64
          memref.store %new, %grid[%at] : memref<?xi64>
        }
      }
    }
  }
  return
}

// As pme_convolve, in a triclinic cell, with the wave vectors k = H⁻¹ m.
// Multiplies the half-complex transform `c` by the influence function
//
//   B C(m) = f / (π V) · exp(−π² m² / β²) / m² · B_1(k_1) B_2(k_2) B_3(k_3)
//
// with the factors B_a of each edge in the rows of `moduli`, and returns the
// energy and the virial
//
//   E = ½ Σ_m w(m) B C(m) |G(m)|²
//   W_ab = Σ_m E_m (δ_ab − 2 (1 / m² + π² / β²) m_a m_b)
//
// with w(m) = 2 for the points whose conjugate is not stored.
func.func private @mdrt.pme_convolve_triclinic(%c: memref<?xf64>, %moduli: memref<?x?xf64>,
                                     %box: vector<6xf64>, %beta: f64, %coulomb: f64,
                                     %k1: index, %k2: index, %k3: index)
    -> (f64, vector<9xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %half = arith.constant 0.5 : f64
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %pi2 = arith.constant 9.869604401089358 : f64
  %none = arith.constant dense<0.0> : vector<9xf64>
  %identity = arith.constant dense<[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]> : vector<9xf64>
  %lx = vector.extract %box[0] : f64 from vector<6xf64>
  %ly = vector.extract %box[1] : f64 from vector<6xf64>
  %lz = vector.extract %box[2] : f64 from vector<6xf64>
  %tbx = vector.extract %box[3] : f64 from vector<6xf64>
  %tcx = vector.extract %box[4] : f64 from vector<6xf64>
  %tcy = vector.extract %box[5] : f64 from vector<6xf64>
  %unit = arith.constant 1.0 : f64
  // The rows of H⁻¹: h00 = 1/a_x; h10 = −b_x/(a_x b_y), h11 = 1/b_y;
  // h20 = (b_x c_y − b_y c_x)/(a_x b_y c_z), h21 = −c_y/(b_y c_z),
  // h22 = 1/c_z.
  %h00 = arith.divf %unit, %lx : f64
  %h11 = arith.divf %unit, %ly : f64
  %h22 = arith.divf %unit, %lz : f64
  %bx_h00 = arith.mulf %tbx, %h00 : f64
  %bx_h00_h11 = arith.mulf %bx_h00, %h11 : f64
  %h10 = arith.negf %bx_h00_h11 : f64
  %cy_h11 = arith.mulf %tcy, %h11 : f64
  %cy_h11_h22 = arith.mulf %cy_h11, %h22 : f64
  %h21 = arith.negf %cy_h11_h22 : f64
  %bxcy = arith.mulf %tbx, %tcy : f64
  %bycx = arith.mulf %ly, %tcx : f64
  %h20n = arith.subf %bxcy, %bycx : f64
  %h20a = arith.mulf %h20n, %h00 : f64
  %h20b = arith.mulf %h20a, %h11 : f64
  %h20 = arith.mulf %h20b, %h22 : f64
  %beta2 = arith.mulf %beta, %beta : f64
  %gauss = arith.divf %pi2, %beta2 : f64
  %pi = arith.constant 3.141592653589793 : f64
  %lxy = arith.mulf %lx, %ly : f64
  %volume = arith.mulf %lxy, %lz : f64
  %piv = arith.mulf %pi, %volume : f64
  %prefactor = arith.divf %coulomb, %piv : f64
  %h = arith.divui %k3, %c2 : index
  %h1 = arith.addi %h, %c1 : index
  %odd = arith.remui %k3, %c2 : index
  %even = arith.cmpi eq, %odd, %c0 : index
  %energy, %virial = scf.for %a = %c0 to %k1 step %c1
      iter_args(%ea = %zero, %wa = %none) -> (f64, vector<9xf64>) {
    // k = H⁻¹ m: its x from m1, its y from m1 and m2, its z from all.
    %i1 = func.call @mdrt.pme_wave(%a, %k1, %unit) : (index, index, f64) -> f64
    %m1 = arith.mulf %i1, %h00 : f64
    %i1_h10 = arith.mulf %i1, %h10 : f64
    %i1_h20 = arith.mulf %i1, %h20 : f64
    %mod1 = memref.load %moduli[%c0, %a] : memref<?x?xf64>
    %eb, %wb = scf.for %b = %c0 to %k2 step %c1
        iter_args(%e2 = %ea, %w2 = %wa) -> (f64, vector<9xf64>) {
      %i2 = func.call @mdrt.pme_wave(%b, %k2, %unit) : (index, index, f64) -> f64
      %i2_h11 = arith.mulf %i2, %h11 : f64
      %m2 = arith.addf %i1_h10, %i2_h11 : f64
      %i2_h21 = arith.mulf %i2, %h21 : f64
      %i12_z = arith.addf %i1_h20, %i2_h21 : f64
      %mod2 = memref.load %moduli[%c1, %b] : memref<?x?xf64>
      %mod12 = arith.mulf %mod1, %mod2 : f64
      %row1 = arith.muli %a, %k2 : index
      %row = arith.addi %row1, %b : index
      %ec, %wc = scf.for %z = %c0 to %h1 step %c1
          iter_args(%e3 = %e2, %w3 = %w2) -> (f64, vector<9xf64>) {
        %zi = arith.index_cast %z : index to i64
        %zf = arith.sitofp %zi : i64 to f64
        %i3_h22 = arith.mulf %zf, %h22 : f64
        %m3 = arith.addf %i12_z, %i3_h22 : f64
        %mod3 = memref.load %moduli[%c2, %z] : memref<?x?xf64>
        %m1s = arith.mulf %m1, %m1 : f64
        %m2s = arith.mulf %m2, %m2 : f64
        %m3s = arith.mulf %m3, %m3 : f64
        %m12 = arith.addf %m1s, %m2s : f64
        %msq = arith.addf %m12, %m3s : f64
        %origin = arith.cmpf oeq, %msq, %zero : f64
        %safe = arith.select %origin, %one, %msq : f64
        %inverse = arith.divf %one, %safe : f64
        %gm = arith.mulf %gauss, %msq : f64
        %ngm = arith.negf %gm : f64
        %ex = math.exp %ngm : f64
        %exm = arith.mulf %ex, %inverse : f64
        %pexm = arith.mulf %prefactor, %exm : f64
        %mods = arith.mulf %mod12, %mod3 : f64
        %bc0 = arith.mulf %pexm, %mods : f64
        %bc = arith.select %origin, %zero, %bc0 : f64
        %base1 = arith.muli %row, %h1 : index
        %base2 = arith.addi %base1, %z : index
        %re_at = arith.muli %base2, %c2 : index
        %im_at = arith.addi %re_at, %c1 : index
        %re = memref.load %c[%re_at] : memref<?xf64>
        %im = memref.load %c[%im_at] : memref<?xf64>
        %re2 = arith.mulf %re, %re : f64
        %im2 = arith.mulf %im, %im : f64
        %g2 = arith.addf %re2, %im2 : f64
        // w = 1 for the plane z = 0 and, with k3 even, z = k3 / 2.
        %first = arith.cmpi eq, %z, %c0 : index
        %middle = arith.cmpi eq, %z, %h : index
        %middle_even = arith.andi %middle, %even : i1
        %single = arith.ori %first, %middle_even : i1
        %weight = arith.select %single, %one, %two : f64
        %hw = arith.mulf %half, %weight : f64
        %hwb = arith.mulf %hw, %bc : f64
        %em = arith.mulf %hwb, %g2 : f64
        // The virial of the point; nothing at m = 0, whose influence is 0.
        %sum = arith.addf %inverse, %gauss : f64
        %factor0 = arith.mulf %two, %sum : f64
        %factor = arith.select %origin, %zero, %factor0 : f64
        %mv = vector.from_elements %m1, %m2, %m3 : vector<3xf64>
        %mm = vector.outerproduct %mv, %mv : vector<3xf64>, vector<3xf64>
        %mmf = vector.shape_cast %mm : vector<3x3xf64> to vector<9xf64>
        %fb = vector.broadcast %factor : f64 to vector<9xf64>
        %scaled = arith.mulf %fb, %mmf : vector<9xf64>
        %diff = arith.subf %identity, %scaled : vector<9xf64>
        %eb9 = vector.broadcast %em : f64 to vector<9xf64>
        %wm = arith.mulf %eb9, %diff : vector<9xf64>
        %w3n = arith.addf %w3, %wm : vector<9xf64>
        %e3n = arith.addf %e3, %em : f64
        // The product with the influence function, for the forces.
        %reb = arith.mulf %re, %bc : f64
        %imb = arith.mulf %im, %bc : f64
        memref.store %reb, %c[%re_at] : memref<?xf64>
        memref.store %imb, %c[%im_at] : memref<?xf64>
        scf.yield %e3n, %w3n : f64, vector<9xf64>
      }
      scf.yield %ec, %wc : f64, vector<9xf64>
    }
    scf.yield %eb, %wb : f64, vector<9xf64>
  }
  return %energy, %virial : f64, vector<9xf64>
}

// As pme_gather_one, in a triclinic cell.
// The force on particle `i`, F_i = −q_i Σ_k φ(k) ∇_i θ_i(k), from the
// potential `phi` on the grid, which is the backward transform of the
// product of the transform of the charges with the influence function. A
// function of its own, so that its arrays live on the stack of a call.
func.func private @mdrt.pme_gather_one_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                       %phi: memref<?xf64>, %box: vector<6xf64>,
                                       %k1: index, %k2: index, %k3: index, %n: index,
                                       %f: memref<?x3x!pme_frc>, %i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %lx = vector.extract %box[0] : f64 from vector<6xf64>
  %ly = vector.extract %box[1] : f64 from vector<6xf64>
  %lz = vector.extract %box[2] : f64 from vector<6xf64>
  %tbx = vector.extract %box[3] : f64 from vector<6xf64>
  %tcx = vector.extract %box[4] : f64 from vector<6xf64>
  %tcy = vector.extract %box[5] : f64 from vector<6xf64>
  %unit = arith.constant 1.0 : f64
  // The rows of H⁻¹: h00 = 1/a_x; h10 = −b_x/(a_x b_y), h11 = 1/b_y;
  // h20 = (b_x c_y − b_y c_x)/(a_x b_y c_z), h21 = −c_y/(b_y c_z),
  // h22 = 1/c_z.
  %h00 = arith.divf %unit, %lx : f64
  %h11 = arith.divf %unit, %ly : f64
  %h22 = arith.divf %unit, %lz : f64
  %bx_h00 = arith.mulf %tbx, %h00 : f64
  %bx_h00_h11 = arith.mulf %bx_h00, %h11 : f64
  %h10 = arith.negf %bx_h00_h11 : f64
  %cy_h11 = arith.mulf %tcy, %h11 : f64
  %cy_h11_h22 = arith.mulf %cy_h11, %h22 : f64
  %h21 = arith.negf %cy_h11_h22 : f64
  %bxcy = arith.mulf %tbx, %tcy : f64
  %bycx = arith.mulf %ly, %tcx : f64
  %h20n = arith.subf %bxcy, %bycx : f64
  %h20a = arith.mulf %h20n, %h00 : f64
  %h20b = arith.mulf %h20a, %h11 : f64
  %h20 = arith.mulf %h20b, %h22 : f64
  %k1i = arith.index_cast %k1 : index to i64
  %k2i = arith.index_cast %k2 : index to i64
  %k3i = arith.index_cast %k3 : index to i64
  %k1f = arith.sitofp %k1i : i64 to f64
  %k2f = arith.sitofp %k2i : i64 to f64
  %k3f = arith.sitofp %k3i : i64 to f64
  %wx = memref.alloca() : memref<8xf64>
  %wy = memref.alloca() : memref<8xf64>
  %wz = memref.alloca() : memref<8xf64>
  %dx = memref.alloca() : memref<8xf64>
  %dy = memref.alloca() : memref<8xf64>
  %dz = memref.alloca() : memref<8xf64>
  %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
  %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
  %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
  %qs = memref.load %q[%i] : memref<?x!pme_chg>
  %xi = PME_EXTEND_POS %xs : !pme_pos to f64
  %yi = PME_EXTEND_POS %ys : !pme_pos to f64
  %zi = PME_EXTEND_POS %zs : !pme_pos to f64
  %qi = PME_EXTEND_CHG %qs : !pme_chg to f64
  // s = x H⁻¹.
  %s3 = arith.mulf %zi, %h22 : f64
  %s2y = arith.mulf %yi, %h11 : f64
  %s2z = arith.mulf %zi, %h21 : f64
  %s2 = arith.addf %s2y, %s2z : f64
  %s1x = arith.mulf %xi, %h00 : f64
  %s1y = arith.mulf %yi, %h10 : f64
  %s1z = arith.mulf %zi, %h20 : f64
  %s1xy = arith.addf %s1x, %s1y : f64
  %s1 = arith.addf %s1xy, %s1z : f64
  %sx, %fx = func.call @mdrt.pme_place(%s1, %unit, %k1, %n) : (f64, f64, index, index) -> (index, f64)
  %sy, %fy = func.call @mdrt.pme_place(%s2, %unit, %k2, %n) : (f64, f64, index, index) -> (index, f64)
  %sz, %fz = func.call @mdrt.pme_place(%s3, %unit, %k3, %n) : (f64, f64, index, index) -> (index, f64)
  func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
  func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
  func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz) : (f64, index, memref<8xf64>, memref<8xf64>) -> ()
  %gx, %gy, %gz = scf.for %j1 = %c0 to %n step %c1
      iter_args(%ax = %zero, %ay = %zero, %az = %zero) -> (f64, f64, f64) {
    %g1s = arith.addi %sx, %j1 : index
    %g1 = arith.remui %g1s, %k1 : index
    %w1 = memref.load %wx[%j1] : memref<8xf64>
    %d1 = memref.load %dx[%j1] : memref<8xf64>
    %bx, %by, %bz = scf.for %j2 = %c0 to %n step %c1
        iter_args(%cx = %ax, %cy = %ay, %cz = %az) -> (f64, f64, f64) {
      %g2s = arith.addi %sy, %j2 : index
      %g2 = arith.remui %g2s, %k2 : index
      %w2 = memref.load %wy[%j2] : memref<8xf64>
      %d2 = memref.load %dy[%j2] : memref<8xf64>
      %row1 = arith.muli %g1, %k2 : index
      %row = arith.addi %row1, %g2 : index
      %base = arith.muli %row, %k3 : index
      %d1w2 = arith.mulf %d1, %w2 : f64
      %w1d2 = arith.mulf %w1, %d2 : f64
      %w1w2 = arith.mulf %w1, %w2 : f64
      %ex, %ey, %ez = scf.for %j3 = %c0 to %n step %c1
          iter_args(%tx = %cx, %ty = %cy, %tz = %cz) -> (f64, f64, f64) {
        %g3s = arith.addi %sz, %j3 : index
        %g3 = arith.remui %g3s, %k3 : index
        %w3 = memref.load %wz[%j3] : memref<8xf64>
        %d3 = memref.load %dz[%j3] : memref<8xf64>
        %at = arith.addi %base, %g3 : index
        %p = memref.load %phi[%at] : memref<?xf64>
        %px = arith.mulf %d1w2, %w3 : f64
        %py = arith.mulf %w1d2, %w3 : f64
        %pz = arith.mulf %w1w2, %d3 : f64
        %vx = arith.mulf %p, %px : f64
        %vy = arith.mulf %p, %py : f64
        %vz = arith.mulf %p, %pz : f64
        %nx = arith.addf %tx, %vx : f64
        %ny = arith.addf %ty, %vy : f64
        %nz = arith.addf %tz, %vz : f64
        scf.yield %nx, %ny, %nz : f64, f64, f64
      }
      scf.yield %ex, %ey, %ez : f64, f64, f64
    }
    scf.yield %bx, %by, %bz : f64, f64, f64
  }
  // F = −q Σ_a K_a g_a ∂s_a/∂x, with ∂s_a/∂x_i = (H⁻¹)_ia.
  %mq = arith.negf %qi : f64
  %u1 = arith.mulf %gx, %k1f : f64
  %u2 = arith.mulf %gy, %k2f : f64
  %u3 = arith.mulf %gz, %k3f : f64
  %gx_lab = arith.mulf %u1, %h00 : f64
  %gy_1 = arith.mulf %u1, %h10 : f64
  %gy_2 = arith.mulf %u2, %h11 : f64
  %gy_lab = arith.addf %gy_1, %gy_2 : f64
  %gz_1 = arith.mulf %u1, %h20 : f64
  %gz_2 = arith.mulf %u2, %h21 : f64
  %gz_3 = arith.mulf %u3, %h22 : f64
  %gz_12 = arith.addf %gz_1, %gz_2 : f64
  %gz_lab = arith.addf %gz_12, %gz_3 : f64
  %fx64 = arith.mulf %mq, %gx_lab : f64
  %fy64 = arith.mulf %mq, %gy_lab : f64
  %fz64 = arith.mulf %mq, %gz_lab : f64
  %fxs = PME_NARROW_FRC %fx64 : f64 to !pme_frc
  %fys = PME_NARROW_FRC %fy64 : f64 to !pme_frc
  %fzs = PME_NARROW_FRC %fz64 : f64 to !pme_frc
  memref.store %fxs, %f[%i, %c0] : memref<?x3x!pme_frc>
  memref.store %fys, %f[%i, %c1] : memref<?x3x!pme_frc>
  memref.store %fzs, %f[%i, %c2] : memref<?x3x!pme_frc>
  return
}

// The forces on all particles in a triclinic cell.
func.func private @mdrt.pme_gather_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %phi: memref<?xf64>, %box: vector<6xf64>,
                                   %k1: index, %k2: index, %k3: index, %n: index,
                                   %f: memref<?x3x!pme_frc>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    func.call @mdrt.pme_gather_one_triclinic(%x, %q, %phi, %box, %k1, %k2, %k3, %n, %f, %i)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, memref<?xf64>, vector<6xf64>,
           index, index, index, index, memref<?x3x!pme_frc>, index) -> ()
    scf.reduce
  }
  return
}
