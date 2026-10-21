// The reciprocal sum of smooth particle mesh Ewald on the host
// [Essmann1995]: spreading the charges to a grid with B-splines, the product
// of the transform of the grid with the influence function, and the forces
// from the potential on the grid (docs/pme.md). The keys are those of
// docs/references.md. The FFT between the steps is pocketfft's, in the
// runtime [Reinecke2019].
//
// This is a template. The compiler adds it to a module that has a
// reciprocal sum, where it is specialized like any other code. `!pme_pos`,
// `!pme_chg`, and `!pme_frc` are the types that the positions, the charges,
// and the forces are stored in; the instance replaces them, and the ops
// PME_EXTEND_POS, PME_EXTEND_CHG, and PME_NARROW_FRC with the conversions
// between them and f64. `!pme_real` is the type of the forces, as in the
// template for devices (PMEGPU.mlir): the weights of the B-splines, the
// grid and its transform, the factors of the influence function, the
// product, and the sums of the gathering are in it, so in f32 in the mixed
// mode, where the mixed mode of a device takes f32 (D271).
// The places of the particles, the cell, and the sums over the rows of the
// grid are in f64.
//
// The grid has k1 x k2 x k3 points, the third index the fastest. A particle
// at the fractional coordinate u (from 0 to k along each edge) adds to the
// points floor(u) − n + 1 + j, j = 0 .. n − 1, taken modulo k, with the
// weights M_n(u − floor(u) + n − 1 − j) of the cardinal B-spline of order n.
// The charges are added in fixed point, at the scale 2^40, so that the sum
// does not depend on the order in which they are added (D70), as the
// fixed point of Le Grand et al. accumulates forces [LeGrand2013].
//
// Every loop over the particles, the points, the rows of the grid, and the
// lines of the FFT is parallel (`scf.parallel`, D171), and its result does
// not depend on the number of threads: the spreading adds integers with
// atomics, the FFT transforms each line by the same code whatever thread
// takes it, and the energy and the virial are reduced in a fixed order
// (D271). The arrays of a particle live on the stack of a
// function of its own, which each iteration calls.

!pme_pos = f64
!pme_chg = f64
!pme_frc = f64
!pme_real = f64

// The transforms of the runtime (runtime/mdrt_fft.c), for a grid in
// !pme_real: PME_FFT names those of its type.
func.func private @mdrtFFTPrepare(i64, i64, i64)
func.func private @mdrtFFTColumnBlocks(i64, i64) -> i64
func.func private @PME_FFTRowsForward(memref<?x!pme_real>, memref<?x!pme_real>, i64, i64, i64)
    attributes {llvm.emit_c_interface}
func.func private @PME_FFTRowsBackward(memref<?x!pme_real>, memref<?x!pme_real>, i64, i64, i64)
    attributes {llvm.emit_c_interface}
func.func private @PME_FFTColumns(memref<?x!pme_real>, i64, i64, i64, i64, i64, i64, i64)
    attributes {llvm.emit_c_interface}

// Raises the weights `a` of the B-spline at the fraction `w` from order
// k − 1 to order k, in place (the recursion of Essmann et al., Eq. 2.28).
func.func private @mdrt.pme_raise(%w: !pme_real, %k: index, %a: memref<8x!pme_real>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %one = arith.constant 1.0 : !pme_real
  %km1 = arith.subi %k, %c1 : index
  %km2 = arith.subi %k, %c2 : index
  %km1i = arith.index_cast %km1 : index to i64
  %km1f = arith.sitofp %km1i : i64 to !pme_real
  %div = arith.divf %one, %km1f : !pme_real
  %ki = arith.index_cast %k : index to i64
  %kf = arith.sitofp %ki : i64 to !pme_real
  // a[k − 1] = div · w · a[k − 2]
  %last = memref.load %a[%km2] : memref<8x!pme_real>
  %dw = arith.mulf %div, %w : !pme_real
  %top = arith.mulf %dw, %last : !pme_real
  memref.store %top, %a[%km1] : memref<8x!pme_real>
  // a[k − j − 1] = div ((w + j) a[k − j − 2] + (k − j − w) a[k − j − 1])
  scf.for %j = %c1 to %km1 step %c1 {
    %ji = arith.index_cast %j : index to i64
    %jf = arith.sitofp %ji : i64 to !pme_real
    %at = arith.subi %km1, %j : index
    %below = arith.subi %at, %c1 : index
    %lo = memref.load %a[%below] : memref<8x!pme_real>
    %hi = memref.load %a[%at] : memref<8x!pme_real>
    %wj = arith.addf %w, %jf : !pme_real
    %kj = arith.subf %kf, %jf : !pme_real
    %kjw = arith.subf %kj, %w : !pme_real
    %t1 = arith.mulf %wj, %lo : !pme_real
    %t2 = arith.mulf %kjw, %hi : !pme_real
    %sum = arith.addf %t1, %t2 : !pme_real
    %value = arith.mulf %div, %sum : !pme_real
    memref.store %value, %a[%at] : memref<8x!pme_real>
  }
  // a[0] = div · (1 − w) · a[0]
  %c0 = arith.constant 0 : index
  %first = memref.load %a[%c0] : memref<8x!pme_real>
  %omw = arith.subf %one, %w : !pme_real
  %d0 = arith.mulf %div, %omw : !pme_real
  %bottom = arith.mulf %d0, %first : !pme_real
  memref.store %bottom, %a[%c0] : memref<8x!pme_real>
  return
}

// The weights `values` of the B-spline of order `n` at the fraction `w`,
// and their derivatives `slopes` with respect to the fractional coordinate.
// The fraction is in f64, and is rounded once to !pme_real.
func.func private @mdrt.pme_bspline(%w: f64, %n: index, %values: memref<8x!pme_real>,
                                    %slopes: memref<8x!pme_real>, %moduli: memref<?x?xf64>) {
  // The table of the prolate kernel has five rows (D266).
  %r0 = arith.constant 0 : index
  %r5 = arith.constant 5 : index
  %rows = memref.dim %moduli, %r0 : memref<?x?xf64>
  %prolate = arith.cmpi eq, %rows, %r5 : index
  scf.if %prolate {
    func.call @mdrt.pme_window(%w, %n, %values, %slopes, %moduli)
        : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  } else {
    %wr = PME_F64_TO_REAL %w : f64 to !pme_real
    func.call @mdrt.pme_spline(%wr, %n, %values, %slopes)
        : (!pme_real, index, memref<8x!pme_real>, memref<8x!pme_real>) -> ()
  }
  return
}

// The weights of the window of the prolate kernel at the fraction `w`
// of the place (Builder::collectProlate, D266): weight j is the
// Chebyshev series of row 3 of `moduli`, at offset j D, in t = 2w − 1,
// and its slope in the scaled coordinate the series at offset (n + j) D,
// by Clenshaw's recurrence [Liang2026, eq. 14]. Computed in f64 and
// rounded once to !pme_real.
func.func private @mdrt.pme_window(%w: f64, %n: index, %values: memref<8x!pme_real>,
                                   %slopes: memref<8x!pme_real>, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %df = memref.load %moduli[%c4, %c0] : memref<?x?xf64>
  %di = arith.fptosi %df : f64 to i64
  %d = arith.index_cast %di : i64 to index
  %tw = arith.mulf %two, %w : f64
  %t = arith.subf %tw, %one : f64
  %t2 = arith.mulf %two, %t : f64
  %nd = arith.muli %n, %d : index
  scf.for %j = %c0 to %n step %c1 {
    %jd = arith.muli %j, %d : index
    %sd = arith.addi %nd, %jd : index
    %v = func.call @mdrt.pme_clenshaw(%moduli, %c3, %jd, %d, %t2, %t)
        : (memref<?x?xf64>, index, index, index, f64, f64) -> f64
    %sl = func.call @mdrt.pme_clenshaw(%moduli, %c3, %sd, %d, %t2, %t)
        : (memref<?x?xf64>, index, index, index, f64, f64) -> f64
    %vr = PME_F64_TO_REAL %v : f64 to !pme_real
    %slr = PME_F64_TO_REAL %sl : f64 to !pme_real
    memref.store %vr, %values[%j] : memref<8x!pme_real>
    memref.store %slr, %slopes[%j] : memref<8x!pme_real>
  }
  return
}

// Σ_k a_k T_k(t) of the `count` coefficients from `moduli`[row, at], by
// Clenshaw's recurrence; `t2` = 2t.
func.func private @mdrt.pme_clenshaw(%moduli: memref<?x?xf64>, %row: index, %at: index,
                                     %count: index, %t2: f64, %t: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %b1, %b2 = scf.for %i = %c1 to %count step %c1
      iter_args(%p1 = %zero, %p2 = %zero) -> (f64, f64) {
    // k = count − i, from count − 1 down to 1.
    %k = arith.subi %count, %i : index
    %ak = arith.addi %at, %k : index
    %a = memref.load %moduli[%row, %ak] : memref<?x?xf64>
    %m = arith.mulf %t2, %p1 : f64
    %s = arith.subf %m, %p2 : f64
    %b = arith.addf %s, %a : f64
    scf.yield %b, %p1 : f64, f64
  }
  %a0 = memref.load %moduli[%row, %at] : memref<?x?xf64>
  %m0 = arith.mulf %t, %b1 : f64
  %s0 = arith.subf %m0, %b2 : f64
  %v = arith.addf %s0, %a0 : f64
  return %v : f64
}

// The weights of the B-spline, as pme_bspline gives them for a table of
// the Gaussian kernel.
func.func private @mdrt.pme_spline(%w: !pme_real, %n: index, %values: memref<8x!pme_real>,
                                   %slopes: memref<8x!pme_real>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : !pme_real
  %one = arith.constant 1.0 : !pme_real
  scf.for %j = %c0 to %c8 step %c1 {
    memref.store %zero, %values[%j] : memref<8x!pme_real>
  }
  %omw = arith.subf %one, %w : !pme_real
  memref.store %omw, %values[%c0] : memref<8x!pme_real>
  memref.store %w, %values[%c1] : memref<8x!pme_real>
  scf.for %k = %c3 to %n step %c1 {
    func.call @mdrt.pme_raise(%w, %k, %values) : (!pme_real, index, memref<8x!pme_real>) -> ()
  }
  // The derivative of M_n is M_{n−1}(x) − M_{n−1}(x − 1).
  %v0 = memref.load %values[%c0] : memref<8x!pme_real>
  %s0 = arith.negf %v0 : !pme_real
  memref.store %s0, %slopes[%c0] : memref<8x!pme_real>
  scf.for %j = %c1 to %n step %c1 {
    %jm1 = arith.subi %j, %c1 : index
    %before = memref.load %values[%jm1] : memref<8x!pme_real>
    %here = memref.load %values[%j] : memref<8x!pme_real>
    %slope = arith.subf %before, %here : !pme_real
    memref.store %slope, %slopes[%j] : memref<8x!pme_real>
  }
  func.call @mdrt.pme_raise(%w, %n, %values) : (!pme_real, index, memref<8x!pme_real>) -> ()
  return
}

// The first point of the grid that a particle at `x` along an edge of the
// length `length` with `k` points adds to, plus k so that it is not
// negative, and the fraction of its coordinate.
func.func private @mdrt.pme_place(%x: f64, %length: f64, %k: index, %n: index,
                                  %moduli: memref<?x?xf64>) -> (index, f64) {
  %r0 = arith.constant 0 : index
  %r2 = arith.constant 2 : index
  %r5 = arith.constant 5 : index
  %rows = memref.dim %moduli, %r0 : memref<?x?xf64>
  %prolate = arith.cmpi eq, %rows, %r5 : index
  %c1 = arith.constant 1 : index
  %s = arith.divf %x, %length : f64
  %fs = math.floor %s : f64
  %frac = arith.subf %s, %fs : f64
  %ki = arith.index_cast %k : index to i64
  %kf = arith.sitofp %ki : i64 to f64
  %u = arith.mulf %frac, %kf : f64
  %fu0 = math.floor %u : f64
  // A coordinate that is not a number takes the point 0, so that the
  // conversion to an integer is defined and every point is in the grid
  // (#168); the fraction is in [0, 1] otherwise, and so is this.
  %nil = arith.constant 0.0 : f64
  %fu = arith.maxnumf %fu0, %nil : f64
  %w = arith.subf %u, %fu : f64
  %bi = arith.fptosi %fu : f64 to i64
  %b = arith.index_cast %bi : i64 to index
  // floor(u) − n + 1 + k
  %bk = arith.addi %b, %k : index
  %bkn = arith.subi %bk, %n : index
  %start = arith.addi %bkn, %c1 : index
  // The window of the prolate kernel (D266): the P points from
  // floor(u − h) − floor(P/2) + 1, with h = 1/2 for odd P and 0 otherwise,
  // and the fraction of u − h; u + k − h is positive.
  %nodd = arith.remui %n, %r2 : index
  %odd = arith.cmpi ne, %nodd, %r0 : index
  %halfp = arith.constant 0.5 : f64
  %h = arith.select %odd, %halfp, %nil : f64
  %uk = arith.addf %u, %kf : f64
  %uh = arith.subf %uk, %h : f64
  %fp0 = math.floor %uh : f64
  %fp = arith.maxnumf %fp0, %nil : f64
  %wp = arith.subf %uh, %fp : f64
  %pi = arith.fptosi %fp : f64 to i64
  %pb = arith.index_cast %pi : i64 to index
  %below = arith.divui %n, %r2 : index
  %pbn = arith.subi %pb, %below : index
  %pstart = arith.addi %pbn, %c1 : index
  %first = arith.select %prolate, %pstart, %start : index
  %fraction = arith.select %prolate, %wp, %w : f64
  return %first, %fraction : index, f64
}

// The rows of H⁻¹ of a triclinic cell `box` = (a_x, b_y, c_z, b_x, c_x,
// c_y): h00 = 1/a_x; h10 = −b_x/(a_x b_y), h11 = 1/b_y;
// h20 = (b_x c_y − b_y c_x)/(a_x b_y c_z), h21 = −c_y/(b_y c_z),
// h22 = 1/c_z.
func.func private @mdrt.pme_inverse_cell(%box: vector<6xf64>)
    -> (f64, f64, f64, f64, f64, f64) {
  %lx = vector.extract %box[0] : f64 from vector<6xf64>
  %ly = vector.extract %box[1] : f64 from vector<6xf64>
  %lz = vector.extract %box[2] : f64 from vector<6xf64>
  %tbx = vector.extract %box[3] : f64 from vector<6xf64>
  %tcx = vector.extract %box[4] : f64 from vector<6xf64>
  %tcy = vector.extract %box[5] : f64 from vector<6xf64>
  %unit = arith.constant 1.0 : f64
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
  return %h00, %h10, %h11, %h20, %h21, %h22 : f64, f64, f64, f64, f64, f64
}

// Zeroes the fixed-point grid.
func.func private @mdrt.pme_zero(%grid: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %none = arith.constant 0 : i64
  %points = memref.dim %grid, %c0 : memref<?xi64>
  scf.parallel (%p) = (%c0) to (%points) step (%c1) {
    memref.store %none, %grid[%p] : memref<?xi64>
    scf.reduce
  }
  return
}

// Adds the charge `qs` of a particle with the weights `wx`, `wy`, `wz`
// from the points `sx`, `sy`, `sz` on to the fixed-point grid, with
// integer atomics: the threads of the spreading add to the same points, and
// an integer sum does not depend on their order (D70).
func.func private @mdrt.pme_spread_point(%grid: memref<?xi64>, %qs: !pme_real,
                                         %placex: index, %placey: index, %placez: index,
                                         %wx: memref<8x!pme_real>, %wy: memref<8x!pme_real>,
                                         %wz: memref<8x!pme_real>,
                                         %k1: index, %k2: index, %k3: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // The first points within the grid, so that a point needs one
  // subtraction at most to wrap around, not a division.
  %sx = arith.remui %placex, %k1 : index
  %sy = arith.remui %placey, %k2 : index
  %sz = arith.remui %placez, %k3 : index
  %scale = arith.constant 1099511627776.0 : !pme_real
  %qscaled = arith.mulf %qs, %scale : !pme_real
  scf.for %j1 = %c0 to %n step %c1 {
    %g1s = arith.addi %sx, %j1 : index
    %g1w = arith.subi %g1s, %k1 : index
    %g1o = arith.cmpi uge, %g1s, %k1 : index
    %g1 = arith.select %g1o, %g1w, %g1s : index
    %w1 = memref.load %wx[%j1] : memref<8x!pme_real>
    %q1 = arith.mulf %qscaled, %w1 : !pme_real
    scf.for %j2 = %c0 to %n step %c1 {
      %g2s = arith.addi %sy, %j2 : index
      %g2w = arith.subi %g2s, %k2 : index
      %g2o = arith.cmpi uge, %g2s, %k2 : index
      %g2 = arith.select %g2o, %g2w, %g2s : index
      %w2 = memref.load %wy[%j2] : memref<8x!pme_real>
      %q12 = arith.mulf %q1, %w2 : !pme_real
      %row1 = arith.muli %g1, %k2 : index
      %row = arith.addi %row1, %g2 : index
      %base = arith.muli %row, %k3 : index
      scf.for %j3 = %c0 to %n step %c1 {
        %g3s = arith.addi %sz, %j3 : index
        %g3w = arith.subi %g3s, %k3 : index
        %g3o = arith.cmpi uge, %g3s, %k3 : index
        %g3 = arith.select %g3o, %g3w, %g3s : index
        %w3 = memref.load %wz[%j3] : memref<8x!pme_real>
        %value = arith.mulf %q12, %w3 : !pme_real
        %rounded = math.roundeven %value : !pme_real
        %fixed = arith.fptosi %rounded : !pme_real to i64
        %at = arith.addi %base, %g3 : index
        %old = memref.atomic_rmw addi %fixed, %grid[%at] : (i64, memref<?xi64>) -> i64
      }
    }
  }
  return
}

// Spreads the charge of particle `i` to `grid`, in fixed point.
func.func private @mdrt.pme_spread_one(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                       %box: vector<3xf64>, %grid: memref<?xi64>,
                                       %k1: index, %k2: index, %k3: index, %n: index,
                                       %moduli: memref<?x?xf64>, %i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %wx = memref.alloca() : memref<8x!pme_real>
  %wy = memref.alloca() : memref<8x!pme_real>
  %wz = memref.alloca() : memref<8x!pme_real>
  %dx = memref.alloca() : memref<8x!pme_real>
  %dy = memref.alloca() : memref<8x!pme_real>
  %dz = memref.alloca() : memref<8x!pme_real>
  %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
  %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
  %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
  %qs = memref.load %q[%i] : memref<?x!pme_chg>
  %xi = PME_EXTEND_POS %xs : !pme_pos to f64
  %yi = PME_EXTEND_POS %ys : !pme_pos to f64
  %zi = PME_EXTEND_POS %zs : !pme_pos to f64
  %qi = PME_CHG_TO_REAL %qs : !pme_chg to !pme_real
  %sx, %fx = func.call @mdrt.pme_place(%xi, %lx, %k1, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sy, %fy = func.call @mdrt.pme_place(%yi, %ly, %k2, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sz, %fz = func.call @mdrt.pme_place(%zi, %lz, %k3, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_spread_point(%grid, %qi, %sx, %sy, %sz, %wx, %wy, %wz, %k1, %k2, %k3, %n)
      : (memref<?xi64>, !pme_real, index, index, index, memref<8x!pme_real>, memref<8x!pme_real>,
         memref<8x!pme_real>, index, index, index, index) -> ()
  return
}

// Spreads the charge of particle `i` to `grid` in a triclinic cell, from
// its fractional coordinates s = x H⁻¹.
func.func private @mdrt.pme_spread_one_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                                 %box: vector<6xf64>, %grid: memref<?xi64>,
                                                 %k1: index, %k2: index, %k3: index, %n: index,
                                                 %moduli: memref<?x?xf64>, %i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %unit = arith.constant 1.0 : f64
  %h00, %h10, %h11, %h20, %h21, %h22 = func.call @mdrt.pme_inverse_cell(%box)
      : (vector<6xf64>) -> (f64, f64, f64, f64, f64, f64)
  %wx = memref.alloca() : memref<8x!pme_real>
  %wy = memref.alloca() : memref<8x!pme_real>
  %wz = memref.alloca() : memref<8x!pme_real>
  %dx = memref.alloca() : memref<8x!pme_real>
  %dy = memref.alloca() : memref<8x!pme_real>
  %dz = memref.alloca() : memref<8x!pme_real>
  %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
  %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
  %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
  %qs = memref.load %q[%i] : memref<?x!pme_chg>
  %xi = PME_EXTEND_POS %xs : !pme_pos to f64
  %yi = PME_EXTEND_POS %ys : !pme_pos to f64
  %zi = PME_EXTEND_POS %zs : !pme_pos to f64
  %qi = PME_CHG_TO_REAL %qs : !pme_chg to !pme_real
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
  %sx, %fx = func.call @mdrt.pme_place(%s1, %unit, %k1, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sy, %fy = func.call @mdrt.pme_place(%s2, %unit, %k2, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sz, %fz = func.call @mdrt.pme_place(%s3, %unit, %k3, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_spread_point(%grid, %qi, %sx, %sy, %sz, %wx, %wy, %wz, %k1, %k2, %k3, %n)
      : (memref<?xi64>, !pme_real, index, index, index, memref<8x!pme_real>, memref<8x!pme_real>,
         memref<8x!pme_real>, index, index, index, index) -> ()
  return
}

// Spreads the charges to `grid`, in fixed point: a particle an iteration
// of a parallel loop.
func.func private @mdrt.pme_spread(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %box: vector<3xf64>, %grid: memref<?xi64>,
                                   %k1: index, %k2: index, %k3: index, %n: index, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @mdrt.pme_zero(%grid) : (memref<?xi64>) -> ()
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    func.call @mdrt.pme_spread_one(%x, %q, %box, %grid, %k1, %k2, %k3, %n, %moduli, %i)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, vector<3xf64>, memref<?xi64>,
           index, index, index, index, memref<?x?xf64>, index) -> ()
    scf.reduce
  }
  return
}

// Spreads the charges to `grid` in a triclinic cell, `box` = (a_x, b_y,
// c_z, b_x, c_x, c_y), from the fractional coordinates s = x H⁻¹.
func.func private @mdrt.pme_spread_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %box: vector<6xf64>, %grid: memref<?xi64>,
                                   %k1: index, %k2: index, %k3: index, %n: index, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @mdrt.pme_zero(%grid) : (memref<?xi64>) -> ()
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    func.call @mdrt.pme_spread_one_triclinic(%x, %q, %box, %grid, %k1, %k2, %k3, %n, %moduli, %i)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, vector<6xf64>, memref<?xi64>,
           index, index, index, index, memref<?x?xf64>, index) -> ()
    scf.reduce
  }
  return
}

// The grid in !pme_real, from fixed point.
func.func private @mdrt.pme_real(%grid: memref<?xi64>, %real: memref<?x!pme_real>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %inverse = arith.constant 9.094947017729282e-13 : !pme_real
  %points = memref.dim %grid, %c0 : memref<?xi64>
  scf.parallel (%p) = (%c0) to (%points) step (%c1) {
    %fixed = memref.load %grid[%p] : memref<?xi64>
    %value = arith.sitofp %fixed : i64 to !pme_real
    %scaled = arith.mulf %value, %inverse : !pme_real
    memref.store %scaled, %real[%p] : memref<?x!pme_real>
    scf.reduce
  }
  return
}

// The forward transform, exp(−2π i k·m / K), of the real grid `real` of
// k1 x k2 x k3 points into the half-complex grid `complex` of
// k1 x k2 x (k3 / 2 + 1) numbers, interleaved: the lines along z, then
// along y, then along x, each pass a parallel loop over blocks of lines
// (runtime/mdrt_fft.c).
func.func private @mdrt.pme_fft_forward(%real: memref<?x!pme_real>, %complex: memref<?x!pme_real>,
                                        %k1: index, %k2: index, %k3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i0 = arith.constant 0 : i64
  %i1 = arith.constant 1 : i64
  %k1i = arith.index_cast %k1 : index to i64
  %k2i = arith.index_cast %k2 : index to i64
  %k3i = arith.index_cast %k3 : index to i64
  func.call @mdrtFFTPrepare(%k1i, %k2i, %k3i) : (i64, i64, i64) -> ()
  // The rows along z, a plane of x a block.
  scf.parallel (%a) = (%c0) to (%k1) step (%c1) {
    %ai = arith.index_cast %a : index to i64
    %first = arith.muli %ai, %k2i : i64
    %last = arith.addi %first, %k2i : i64
    func.call @PME_FFTRowsForward(%real, %complex, %k3i, %first, %last)
        : (memref<?x!pme_real>, memref<?x!pme_real>, i64, i64, i64) -> ()
    scf.reduce
  }
  %ny = func.call @mdrtFFTColumnBlocks(%k1i, %k3i) : (i64, i64) -> i64
  %nyx = arith.index_cast %ny : i64 to index
  scf.parallel (%u) = (%c0) to (%nyx) step (%c1) {
    %ui = arith.index_cast %u : index to i64
    %next = arith.addi %ui, %i1 : i64
    func.call @PME_FFTColumns(%complex, %k1i, %k2i, %k3i, %i1, %i1, %ui, %next)
        : (memref<?x!pme_real>, i64, i64, i64, i64, i64, i64, i64) -> ()
    scf.reduce
  }
  %nx = func.call @mdrtFFTColumnBlocks(%k2i, %k3i) : (i64, i64) -> i64
  %nxx = arith.index_cast %nx : i64 to index
  scf.parallel (%u) = (%c0) to (%nxx) step (%c1) {
    %ui = arith.index_cast %u : index to i64
    %next = arith.addi %ui, %i1 : i64
    func.call @PME_FFTColumns(%complex, %k1i, %k2i, %k3i, %i0, %i1, %ui, %next)
        : (memref<?x!pme_real>, i64, i64, i64, i64, i64, i64, i64) -> ()
    scf.reduce
  }
  return
}

// The backward transform, exp(+2π i k·m / K), not normalized, of the
// half-complex grid `complex`, which it overwrites, into `real`: the lines
// along y, then along x, then the rows along z.
func.func private @mdrt.pme_fft_backward(%complex: memref<?x!pme_real>, %real: memref<?x!pme_real>,
                                         %k1: index, %k2: index, %k3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i0 = arith.constant 0 : i64
  %i1 = arith.constant 1 : i64
  %k1i = arith.index_cast %k1 : index to i64
  %k2i = arith.index_cast %k2 : index to i64
  %k3i = arith.index_cast %k3 : index to i64
  func.call @mdrtFFTPrepare(%k1i, %k2i, %k3i) : (i64, i64, i64) -> ()
  %ny = func.call @mdrtFFTColumnBlocks(%k1i, %k3i) : (i64, i64) -> i64
  %nyx = arith.index_cast %ny : i64 to index
  scf.parallel (%u) = (%c0) to (%nyx) step (%c1) {
    %ui = arith.index_cast %u : index to i64
    %next = arith.addi %ui, %i1 : i64
    func.call @PME_FFTColumns(%complex, %k1i, %k2i, %k3i, %i1, %i0, %ui, %next)
        : (memref<?x!pme_real>, i64, i64, i64, i64, i64, i64, i64) -> ()
    scf.reduce
  }
  %nx = func.call @mdrtFFTColumnBlocks(%k2i, %k3i) : (i64, i64) -> i64
  %nxx = arith.index_cast %nx : i64 to index
  scf.parallel (%u) = (%c0) to (%nxx) step (%c1) {
    %ui = arith.index_cast %u : index to i64
    %next = arith.addi %ui, %i1 : i64
    func.call @PME_FFTColumns(%complex, %k1i, %k2i, %k3i, %i0, %i0, %ui, %next)
        : (memref<?x!pme_real>, i64, i64, i64, i64, i64, i64, i64) -> ()
    scf.reduce
  }
  scf.parallel (%a) = (%c0) to (%k1) step (%c1) {
    %ai = arith.index_cast %a : index to i64
    %first = arith.muli %ai, %k2i : i64
    %last = arith.addi %first, %k2i : i64
    func.call @PME_FFTRowsBackward(%complex, %real, %k3i, %first, %last)
        : (memref<?x!pme_real>, memref<?x!pme_real>, i64, i64, i64) -> ()
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

// The factors of the influence function along the edges of a rectangular
// cell, as a device computes them (D104): for each point k of each edge a,
// at 3 (o_a + k) with o = (0, k1, k1 + k2), the wave number m_a, m_a², and
// exp(−π² m_a² / β²) times the factor B_a of the B-splines (row a of
// `moduli`), along x also times f / (π V); each computed in f64 and
// rounded once to !pme_real. The Gaussian influence function of a point is
// then the product of three of them over m².
//
// With the rows of influence "P3M" in `moduli` (more than five,
// D304), eq. 2.28 of [Ballenegger2012] in its form for the
// edges (docs/pme.md, Section 1.1): with the weights u_j of the aliases
// ν_j = n + j K of the point, μ_j = ν_j / L, g_j = exp(−π² μ_j² / β²),
// the three values are Σ u μ, Σ u μ², and Σ u g / S times the prefactor
// along x, so that the influence is again the product of the third over
// the sum of the second; after the 3 (k1 + k2 + k3) values, at
// 3 (k1 + k2 + k3) + 2 (o_a + k), Σ u g μ / Σ u g and Σ u g μ² / Σ u g,
// which the virial takes.
func.func private @mdrt.pme_tables(%moduli: memref<?x?xf64>, %tables: memref<?x!pme_real>,
                                   %box: vector<3xf64>, %beta: f64, %coulomb: f64,
                                   %k1: index, %k2: index, %k3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %pi2 = arith.constant 9.869604401089358 : f64
  %pi = arith.constant 3.141592653589793 : f64
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %beta2 = arith.mulf %beta, %beta : f64
  %gauss = arith.divf %pi2, %beta2 : f64
  %lxy = arith.mulf %lx, %ly : f64
  %volume = arith.mulf %lxy, %lz : f64
  %piv = arith.mulf %pi, %volume : f64
  %prefactor = arith.divf %coulomb, %piv : f64
  %k12 = arith.addi %k1, %k2 : index
  // Influence "P3M": 2J + 1 rows of weights for each edge after the three
  // rows of 1 / S (D304).
  %c5 = arith.constant 5 : index
  %rows = memref.dim %moduli, %c0 : memref<?x?xf64>
  %p3m = arith.cmpi ugt, %rows, %c5 : index
  %rows3 = arith.subi %rows, %c3 : index
  %span = arith.divui %rows3, %c3 : index
  %span1 = arith.subi %span, %c1 : index
  %aliases = arith.divui %span1, %c2 : index
  %entries0 = arith.addi %k12, %k3 : index
  %extra = arith.muli %entries0, %c3 : index
  scf.for %axis = %c0 to %c3 step %c1 {
    %on_x = arith.cmpi eq, %axis, %c0 : index
    %on_y = arith.cmpi eq, %axis, %c1 : index
    %count0 = arith.select %on_y, %k2, %k3 : index
    %count = arith.select %on_x, %k1, %count0 : index
    %length0 = arith.select %on_y, %ly, %lz : f64
    %length = arith.select %on_x, %lx, %length0 : f64
    %offset0 = arith.select %on_y, %k1, %k12 : index
    %offset = arith.select %on_x, %c0, %offset0 : index
    scf.for %k = %c0 to %count step %c1 {
      %point = arith.addi %offset, %k : index
      %m = func.call @mdrt.pme_wave(%k, %count, %length) : (index, index, f64) -> f64
      %v0, %v1, %factor0 = scf.if %p3m -> (f64, f64, f64) {
        %zero = arith.constant 0.0 : f64
        %unit = arith.constant 1.0 : f64
        %n = func.call @mdrt.pme_wave(%k, %count, %unit) : (index, index, f64) -> f64
        %ci = arith.index_cast %count : index to i64
        %cf = arith.sitofp %ci : i64 to f64
        %ai = arith.index_cast %aliases : index to i64
        %first = arith.muli %axis, %span : index
        %row0 = arith.addi %first, %c3 : index
        %sm, %sq, %sg, %sgm, %sgm2 = scf.for %j = %c0 to %span step %c1
            iter_args(%a0 = %zero, %a1 = %zero, %a2 = %zero, %a3 = %zero, %a4 = %zero)
            -> (f64, f64, f64, f64, f64) {
          %ji = arith.index_cast %j : index to i64
          %js = arith.subi %ji, %ai : i64
          %jf = arith.sitofp %js : i64 to f64
          %jk = arith.mulf %jf, %cf : f64
          %nu = arith.addf %n, %jk : f64
          %mu = arith.divf %nu, %length : f64
          %mu2 = arith.mulf %mu, %mu : f64
          %gmu = arith.mulf %gauss, %mu2 : f64
          %ngmu = arith.negf %gmu : f64
          %g = math.exp %ngmu : f64
          %row = arith.addi %row0, %j : index
          %u = memref.load %moduli[%row, %k] : memref<?x?xf64>
          %umu = arith.mulf %u, %mu : f64
          %umu2 = arith.mulf %u, %mu2 : f64
          %ug = arith.mulf %u, %g : f64
          %ugmu = arith.mulf %ug, %mu : f64
          %ugmu2 = arith.mulf %ug, %mu2 : f64
          %b0 = arith.addf %a0, %umu : f64
          %b1 = arith.addf %a1, %umu2 : f64
          %b2 = arith.addf %a2, %ug : f64
          %b3 = arith.addf %a3, %ugmu : f64
          %b4 = arith.addf %a4, %ugmu2 : f64
          scf.yield %b0, %b1, %b2, %b3, %b4 : f64, f64, f64, f64, f64
        }
        %inverse = memref.load %moduli[%axis, %k] : memref<?x?xf64>
        %f0 = arith.mulf %sg, %inverse : f64
        // The means under the Gaussian, 0 where it underflows.
        %some = arith.cmpf ogt, %sg, %zero : f64
        %sgs = arith.select %some, %sg, %unit : f64
        %gm0 = arith.divf %sgm, %sgs : f64
        %gq0 = arith.divf %sgm2, %sgs : f64
        %gm = arith.select %some, %gm0, %zero : f64
        %gq = arith.select %some, %gq0, %zero : f64
        %two = arith.constant 2 : index
        %at3a = arith.muli %point, %two : index
        %at3 = arith.addi %extra, %at3a : index
        %at4 = arith.addi %at3, %c1 : index
        %gmr = PME_F64_TO_REAL %gm : f64 to !pme_real
        %gqr = PME_F64_TO_REAL %gq : f64 to !pme_real
        memref.store %gmr, %tables[%at3] : memref<?x!pme_real>
        memref.store %gqr, %tables[%at4] : memref<?x!pme_real>
        scf.yield %sm, %sq, %f0 : f64, f64, f64
      } else {
        %m2 = arith.mulf %m, %m : f64
        %gm = arith.mulf %gauss, %m2 : f64
        %ngm = arith.negf %gm : f64
        %ex = math.exp %ngm : f64
        %mod = memref.load %moduli[%axis, %k] : memref<?x?xf64>
        %f0 = arith.mulf %ex, %mod : f64
        scf.yield %m, %m2, %f0 : f64, f64, f64
      }
      %scaled = arith.mulf %factor0, %prefactor : f64
      %factor = arith.select %on_x, %scaled, %factor0 : f64
      %at0 = arith.muli %point, %c3 : index
      %at1 = arith.addi %at0, %c1 : index
      %at2 = arith.addi %at0, %c2 : index
      %mr = PME_F64_TO_REAL %v0 : f64 to !pme_real
      %m2r = PME_F64_TO_REAL %v1 : f64 to !pme_real
      %fr = PME_F64_TO_REAL %factor : f64 to !pme_real
      memref.store %mr, %tables[%at0] : memref<?x!pme_real>
      memref.store %m2r, %tables[%at1] : memref<?x!pme_real>
      memref.store %fr, %tables[%at2] : memref<?x!pme_real>
    }
  }
  return
}

// The influence function of the point of the wave vector m, in f64, for
// the sums that are not separable along the edges: the Gaussian sum in a
// triclinic cell, the sum of the dispersion, and the prolate kernel.
// `mods` is the product of the factors of the B-splines of the three
// edges, `volume` that of the cell. Returns B C(m) and the parts of the
// virial of the point, E_m (δ − factor m m) − ½ w vd |G|² m m:
//
//   B C(m) = f / (π V) · exp(−π² m² / β²) / m² · B_1(k_1) B_2(k_2) B_3(k_3)
//   W_ab = Σ_m E_m (δ_ab − 2 (1 / m² + π² / β²) m_a m_b)
func.func private @mdrt.pme_influence(%m1: f64, %m2: f64, %m3: f64, %mods: f64,
                                      %moduli: memref<?x?xf64>, %volume: f64,
                                      %beta: f64, %coulomb: f64, %dispersion: i1)
    -> (f64, f64, f64) {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %pi2 = arith.constant 9.869604401089358 : f64
  %pi = arith.constant 3.141592653589793 : f64
  %beta2 = arith.mulf %beta, %beta : f64
  %gauss = arith.divf %pi2, %beta2 : f64
  %piv = arith.mulf %pi, %volume : f64
  %prefactor = arith.divf %coulomb, %piv : f64
  %m1s = arith.mulf %m1, %m1 : f64
  %m2s = arith.mulf %m2, %m2 : f64
  %m3s = arith.mulf %m3, %m3 : f64
  %m12 = arith.addf %m1s, %m2s : f64
  %msq = arith.addf %m12, %m3s : f64
  %origin = arith.cmpf oeq, %msq, %zero : f64
  %safe = arith.select %origin, %one, %msq : f64
  %inverse = arith.divf %one, %safe : f64
  // The influence function: of the Coulomb sum, nothing at m = 0;
  // of the dispersion, B · P_d · F(b) with b = π |m| / β and
  //   F(b) = (1 − 2b²) exp(−b²) + 2 √π b³ erfc(b)
  // [Essmann1995], m = 0 included. `factor` and `vd` are the parts
  // of the virial: E_m (δ − factor m m) − ½ w vd |G|² m m.
  %bcg, %factorg, %vdg = scf.if %dispersion -> (f64, f64, f64) {
    // The sum of the dispersion (D162): its prefactor −π^{3/2} β³ / (3V)
    // times the factor, π / β, which turns |m| into b, and 6 π² / β², which
    // the virial takes.
    %pi32 = arith.constant 5.568327996831708 : f64
    %beta3 = arith.mulf %beta2, %beta : f64
    %pb3 = arith.mulf %pi32, %beta3 : f64
    %three = arith.constant 3.0 : f64
    %v3 = arith.mulf %three, %volume : f64
    %pd0 = arith.divf %pb3, %v3 : f64
    %pdc = arith.mulf %pd0, %coulomb : f64
    %dprefactor = arith.negf %pdc : f64
    %pib = arith.divf %pi, %beta : f64
    %sqrtpi = arith.constant 1.7724538509055159 : f64
    %six = arith.constant 6.0 : f64
    %sixgauss = arith.mulf %six, %gauss : f64
    %mabs = math.sqrt %msq : f64
    %bd = arith.mulf %pib, %mabs : f64
    %bd2 = arith.mulf %bd, %bd : f64
    %nbd2 = arith.negf %bd2 : f64
    %ebd = math.exp %nbd2 : f64
    %erfcb = math.erfc %bd : f64
    %spb = arith.mulf %sqrtpi, %bd : f64
    %sbd = arith.mulf %spb, %erfcb : f64
    %twob2 = arith.mulf %two, %bd2 : f64
    %omt = arith.subf %one, %twob2 : f64
    %fa = arith.mulf %omt, %ebd : f64
    %fb2 = arith.mulf %twob2, %sbd : f64
    %fd = arith.addf %fa, %fb2 : f64
    %pmods = arith.mulf %dprefactor, %mods : f64
    %dbc = arith.mulf %pmods, %fd : f64
    // −∂F/∂ε_a: 6 π² / β² (exp(−b²) − √π b erfc(b)) m_a m_a.
    %vd0 = arith.subf %ebd, %sbd : f64
    %vd1 = arith.mulf %sixgauss, %vd0 : f64
    %dvd = arith.mulf %pmods, %vd1 : f64
    scf.yield %dbc, %zero, %dvd : f64, f64, f64
  } else {
    %gm = arith.mulf %gauss, %msq : f64
    %ngm = arith.negf %gm : f64
    %ex = math.exp %ngm : f64
    %exm = arith.mulf %ex, %inverse : f64
    %pexm = arith.mulf %prefactor, %exm : f64
    %bc0 = arith.mulf %pexm, %mods : f64
    %bcc = arith.select %origin, %zero, %bc0 : f64
    // Nothing at m = 0, whose influence is 0.
    %sum = arith.addf %inverse, %gauss : f64
    %factor0 = arith.mulf %two, %sum : f64
    %factorc = arith.select %origin, %zero, %factor0 : f64
    scf.yield %bcc, %factorc, %zero : f64, f64, f64
  }
  // The prolate kernel (D266): B · f/(πV) · χ̂(2π r_c |m|)
  // / m², with χ̂ and k χ̂'(k) the Chebyshev series of row 4 of the
  // table in y = 2x² − 1, x = 2π r_c |m| / c, and nothing at m = 0 or
  // beyond x = 1 [Liang2026, eq. 5]. The virial of a point is
  // E_m (δ − (2 − g) m m / m²) with g = d ln χ̂ / d ln |m|: factor
  // 2/m², and vd = −B f/(πV) k χ̂'(k) / m⁴.
  %p_r0 = arith.constant 0 : index
  %p_r4 = arith.constant 4 : index
  %p_r5 = arith.constant 5 : index
  %p_rows = memref.dim %moduli, %p_r0 : memref<?x?xf64>
  %prolate = arith.cmpi eq, %p_rows, %p_r5 : index
  %bc, %factor, %vd = scf.if %prolate -> (f64, f64, f64) {
    %p_c2 = arith.constant 2 : index
    %p_c3 = arith.constant 3 : index
    %p_tf = memref.load %moduli[%p_r4, %p_c2] : memref<?x?xf64>
    %p_ti = arith.fptosi %p_tf : f64 to i64
    %p_terms = arith.index_cast %p_ti : i64 to index
    %p_scale = memref.load %moduli[%p_r4, %p_c3] : memref<?x?xf64>
    %p_mabs = math.sqrt %msq : f64
    %p_x = arith.mulf %p_scale, %p_mabs : f64
    %p_x2 = arith.mulf %p_x, %p_x : f64
    %p_2x2 = arith.mulf %two, %p_x2 : f64
    %p_y = arith.subf %p_2x2, %one : f64
    %p_y2 = arith.mulf %two, %p_y : f64
    %p_at = arith.addi %p_r4, %p_terms : index
    %p_ch = func.call @mdrt.pme_clenshaw(%moduli, %p_r4, %p_r4, %p_terms, %p_y2, %p_y)
        : (memref<?x?xf64>, index, index, index, f64, f64) -> f64
    %p_gch = func.call @mdrt.pme_clenshaw(%moduli, %p_r4, %p_at, %p_terms, %p_y2, %p_y)
        : (memref<?x?xf64>, index, index, index, f64, f64) -> f64
    %p_pm = arith.mulf %prefactor, %mods : f64
    %p_pmi = arith.mulf %p_pm, %inverse : f64
    %p_bc0 = arith.mulf %p_pmi, %p_ch : f64
    %p_f0 = arith.mulf %two, %inverse : f64
    %p_vd1 = arith.mulf %p_pmi, %inverse : f64
    %p_vd2 = arith.mulf %p_vd1, %p_gch : f64
    %p_vd0 = arith.negf %p_vd2 : f64
    %p_band = arith.cmpf olt, %p_x, %one : f64
    %p_true = arith.constant true
    %p_nonzero = arith.xori %origin, %p_true : i1
    %p_in = arith.andi %p_band, %p_nonzero : i1
    %p_bc = arith.select %p_in, %p_bc0, %zero : f64
    %p_f = arith.select %p_in, %p_f0, %zero : f64
    %p_vd = arith.select %p_in, %p_vd0, %zero : f64
    scf.yield %p_bc, %p_f, %p_vd : f64, f64, f64
  } else {
    scf.yield %bcg, %factorg, %vdg : f64, f64, f64
  }
  return %bc, %factor, %vd : f64, f64, f64
}

// The energy and the virial of a point of the half-complex transform,
// in !pme_real, added to `e` and `w`, from its value `re` + i `im`, its
// wave vector, the influence `bc`, and the parts of the virial: with
// w(m) = 2 for the points whose conjugate is not stored,
//
//   E_m = ½ w(m) B C(m) |G(m)|²,  W_m = E_m (δ − factor m m) − ½ w vd |G|² m m.
func.func private @mdrt.pme_point_sums(%re: !pme_real, %im: !pme_real, %m1: !pme_real,
                                       %m2: !pme_real, %m3: !pme_real, %bc: !pme_real,
                                       %factor: !pme_real, %vd: !pme_real, %single: i1,
                                       %e: !pme_real, %w: vector<9x!pme_real>)
    -> (!pme_real, vector<9x!pme_real>) {
  %half = arith.constant 0.5 : !pme_real
  %one = arith.constant 1.0 : !pme_real
  %identity = arith.constant dense<[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]> : vector<9x!pme_real>
  %re2 = arith.mulf %re, %re : !pme_real
  %im2 = arith.mulf %im, %im : !pme_real
  %g2 = arith.addf %re2, %im2 : !pme_real
  %hw = arith.select %single, %half, %one : !pme_real
  %hwb = arith.mulf %hw, %bc : !pme_real
  %em = arith.mulf %hwb, %g2 : !pme_real
  %mv = vector.from_elements %m1, %m2, %m3 : vector<3x!pme_real>
  %mm = vector.outerproduct %mv, %mv : vector<3x!pme_real>, vector<3x!pme_real>
  %mmf = vector.shape_cast %mm : vector<3x3x!pme_real> to vector<9x!pme_real>
  %fb = vector.broadcast %factor : !pme_real to vector<9x!pme_real>
  %scaled = arith.mulf %fb, %mmf : vector<9x!pme_real>
  %diff = arith.subf %identity, %scaled : vector<9x!pme_real>
  %eb9 = vector.broadcast %em : !pme_real to vector<9x!pme_real>
  %wmc = arith.mulf %eb9, %diff : vector<9x!pme_real>
  %hvd = arith.mulf %hw, %vd : !pme_real
  %evd = arith.mulf %hvd, %g2 : !pme_real
  %evb = vector.broadcast %evd : !pme_real to vector<9x!pme_real>
  %evm = arith.mulf %evb, %mmf : vector<9x!pme_real>
  %wm = arith.subf %wmc, %evm : vector<9x!pme_real>
  %wn = arith.addf %w, %wm : vector<9x!pme_real>
  %en = arith.addf %e, %em : !pme_real
  return %en, %wn : !pme_real, vector<9x!pme_real>
}

// As pme_point_sums, for influence "P3M" (D304): the virial of
// the point is the derivative of its energy under a strain of the cell,
// through every alias in the sums of eq. 2.28 (docs/pme.md, Section 1.1),
//
//   W_ab = E_m (δ_ab − 2 (π²/β²) A_ab − 2 B_ab / D),
//
// with A_ab = ⟨μ_a⟩_g ⟨μ_b⟩_g off the diagonal and ⟨μ_a²⟩_g on it (`ga`,
// `gs`), B likewise of the plain means (`pa`, `ps`), D = Σ_a ⟨μ_a²⟩, and
// `inverse` 1 / D.
func.func private @mdrt.pme_point_sums_p3m(%re: !pme_real, %im: !pme_real,
                                           %pa: vector<3x!pme_real>, %ps: vector<3x!pme_real>,
                                           %ga: vector<3x!pme_real>, %gs: vector<3x!pme_real>,
                                           %bc: !pme_real, %inverse: !pme_real, %gauss: !pme_real,
                                           %single: i1, %e: !pme_real, %w: vector<9x!pme_real>)
    -> (!pme_real, vector<9x!pme_real>) {
  %half = arith.constant 0.5 : !pme_real
  %one = arith.constant 1.0 : !pme_real
  %two = arith.constant 2.0 : !pme_real
  %identity = arith.constant dense<[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]> : vector<9x!pme_real>
  %re2 = arith.mulf %re, %re : !pme_real
  %im2 = arith.mulf %im, %im : !pme_real
  %g2 = arith.addf %re2, %im2 : !pme_real
  %hw = arith.select %single, %half, %one : !pme_real
  %hwb = arith.mulf %hw, %bc : !pme_real
  %em = arith.mulf %hwb, %g2 : !pme_real
  // The outer products, with their diagonals replaced by the means of
  // the squares.
  %aa = vector.outerproduct %ga, %ga : vector<3x!pme_real>, vector<3x!pme_real>
  %bb = vector.outerproduct %pa, %pa : vector<3x!pme_real>, vector<3x!pme_real>
  %gs0 = vector.extract %gs[0] : !pme_real from vector<3x!pme_real>
  %gs1 = vector.extract %gs[1] : !pme_real from vector<3x!pme_real>
  %gs2 = vector.extract %gs[2] : !pme_real from vector<3x!pme_real>
  %ps0 = vector.extract %ps[0] : !pme_real from vector<3x!pme_real>
  %ps1 = vector.extract %ps[1] : !pme_real from vector<3x!pme_real>
  %ps2 = vector.extract %ps[2] : !pme_real from vector<3x!pme_real>
  %aa0 = vector.insert %gs0, %aa[0, 0] : !pme_real into vector<3x3x!pme_real>
  %aa1 = vector.insert %gs1, %aa0[1, 1] : !pme_real into vector<3x3x!pme_real>
  %aa2 = vector.insert %gs2, %aa1[2, 2] : !pme_real into vector<3x3x!pme_real>
  %bb0 = vector.insert %ps0, %bb[0, 0] : !pme_real into vector<3x3x!pme_real>
  %bb1 = vector.insert %ps1, %bb0[1, 1] : !pme_real into vector<3x3x!pme_real>
  %bb2 = vector.insert %ps2, %bb1[2, 2] : !pme_real into vector<3x3x!pme_real>
  %af = vector.shape_cast %aa2 : vector<3x3x!pme_real> to vector<9x!pme_real>
  %bf = vector.shape_cast %bb2 : vector<3x3x!pme_real> to vector<9x!pme_real>
  %ca = arith.mulf %two, %gauss : !pme_real
  %cb = arith.mulf %two, %inverse : !pme_real
  %cav = vector.broadcast %ca : !pme_real to vector<9x!pme_real>
  %cbv = vector.broadcast %cb : !pme_real to vector<9x!pme_real>
  %ta = arith.mulf %cav, %af : vector<9x!pme_real>
  %tb = arith.mulf %cbv, %bf : vector<9x!pme_real>
  %t = arith.addf %ta, %tb : vector<9x!pme_real>
  %diff = arith.subf %identity, %t : vector<9x!pme_real>
  %eb9 = vector.broadcast %em : !pme_real to vector<9x!pme_real>
  %wm = arith.mulf %eb9, %diff : vector<9x!pme_real>
  %wn = arith.addf %w, %wm : vector<9x!pme_real>
  %en = arith.addf %e, %em : !pme_real
  return %en, %wn : !pme_real, vector<9x!pme_real>
}

// Multiplies the row `row` = a k2 + b of the half-complex transform `c` by
// the influence function, and returns its energy and virial (with `sums`),
// summed along the row in !pme_real and returned in f64. The Gaussian sum
// takes the factors of the edges from `tables` (pme_tables); the others
// compute the influence of each point in f64 (pme_influence), rounded once.
func.func private @mdrt.pme_convolve_row(%c: memref<?x!pme_real>, %tables: memref<?x!pme_real>,
                                         %moduli: memref<?x?xf64>, %box: vector<3xf64>,
                                         %beta: f64, %coulomb: f64, %dispersion: i1,
                                         %k1: index, %k2: index, %k3: index, %row: index,
                                         %sums: i1)
    -> (f64, vector<9xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %rzero = arith.constant 0.0 : !pme_real
  %rone = arith.constant 1.0 : !pme_real
  %rtwo = arith.constant 2.0 : !pme_real
  %rnone = arith.constant dense<0.0> : vector<9x!pme_real>
  %pi2 = arith.constant 9.869604401089358 : f64
  %true = arith.constant true
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %lxy = arith.mulf %lx, %ly : f64
  %volume = arith.mulf %lxy, %lz : f64
  %beta2 = arith.mulf %beta, %beta : f64
  %gauss = arith.divf %pi2, %beta2 : f64
  %gaussr = PME_F64_TO_REAL %gauss : f64 to !pme_real
  %rows = memref.dim %moduli, %c0 : memref<?x?xf64>
  %prolate = arith.cmpi eq, %rows, %c5 : index
  %other = arith.ori %dispersion, %prolate : i1
  %separable = arith.xori %other, %true : i1
  // Influence "P3M" (D304): the separable product, and the
  // virial from the means of the edges after the 3 (k1 + k2 + k3) values.
  %p3m = arith.cmpi ugt, %rows, %c5 : index
  %k12e = arith.addi %k1, %k2 : index
  %entries = arith.addi %k12e, %k3 : index
  %extra = arith.muli %entries, %c3 : index
  %a = arith.divui %row, %k2 : index
  %b = arith.remui %row, %k2 : index
  %h = arith.divui %k3, %c2 : index
  %h1 = arith.addi %h, %c1 : index
  %odd = arith.remui %k3, %c2 : index
  %even = arith.cmpi eq, %odd, %c0 : index
  %ta = arith.muli %a, %c3 : index
  %ta1 = arith.addi %ta, %c1 : index
  %ta2 = arith.addi %ta, %c2 : index
  %m1 = memref.load %tables[%ta] : memref<?x!pme_real>
  %m1s = memref.load %tables[%ta1] : memref<?x!pme_real>
  %t1 = memref.load %tables[%ta2] : memref<?x!pme_real>
  %pb = arith.addi %k1, %b : index
  %tb = arith.muli %pb, %c3 : index
  %tb1 = arith.addi %tb, %c1 : index
  %tb2 = arith.addi %tb, %c2 : index
  %m2 = memref.load %tables[%tb] : memref<?x!pme_real>
  %m2s = memref.load %tables[%tb1] : memref<?x!pme_real>
  %t2 = memref.load %tables[%tb2] : memref<?x!pme_real>
  %t12 = arith.mulf %t1, %t2 : !pme_real
  %m12s = arith.addf %m1s, %m2s : !pme_real
  %k12 = arith.addi %k1, %k2 : index
  // In f64, for the sums that are not separable.
  %m1d = func.call @mdrt.pme_wave(%a, %k1, %lx) : (index, index, f64) -> f64
  %m2d = func.call @mdrt.pme_wave(%b, %k2, %ly) : (index, index, f64) -> f64
  %mod1 = memref.load %moduli[%c0, %a] : memref<?x?xf64>
  %mod2 = memref.load %moduli[%c1, %b] : memref<?x?xf64>
  %mod12 = arith.mulf %mod1, %mod2 : f64
  %base = arith.muli %row, %h1 : index
  %e, %w = scf.for %z = %c0 to %h1 step %c1
      iter_args(%e3 = %rzero, %w3 = %rnone) -> (!pme_real, vector<9x!pme_real>) {
    %pz = arith.addi %k12, %z : index
    %tz = arith.muli %pz, %c3 : index
    %tz1 = arith.addi %tz, %c1 : index
    %tz2 = arith.addi %tz, %c2 : index
    %m3 = memref.load %tables[%tz] : memref<?x!pme_real>
    %m3s = memref.load %tables[%tz1] : memref<?x!pme_real>
    %t3 = memref.load %tables[%tz2] : memref<?x!pme_real>
    %bc, %factor, %vd = scf.if %separable -> (!pme_real, !pme_real, !pme_real) {
      %msq = arith.addf %m12s, %m3s : !pme_real
      %origin = arith.cmpf oeq, %msq, %rzero : !pme_real
      %safe = arith.select %origin, %rone, %msq : !pme_real
      %inverse = arith.divf %rone, %safe : !pme_real
      %t123 = arith.mulf %t12, %t3 : !pme_real
      %bc0 = arith.mulf %t123, %inverse : !pme_real
      %bcs = arith.select %origin, %rzero, %bc0 : !pme_real
      // Nothing at m = 0, whose influence is 0.
      %sum = arith.addf %inverse, %gaussr : !pme_real
      %factor0 = arith.mulf %rtwo, %sum : !pme_real
      %factors = arith.select %origin, %rzero, %factor0 : !pme_real
      scf.yield %bcs, %factors, %rzero : !pme_real, !pme_real, !pme_real
    } else {
      %zi = arith.index_cast %z : index to i64
      %zf = arith.sitofp %zi : i64 to f64
      %m3d = arith.divf %zf, %lz : f64
      %mod3 = memref.load %moduli[%c2, %z] : memref<?x?xf64>
      %mods = arith.mulf %mod12, %mod3 : f64
      %bcd, %fd, %vdd = func.call @mdrt.pme_influence(%m1d, %m2d, %m3d, %mods, %moduli, %volume,
                                                       %beta, %coulomb, %dispersion)
          : (f64, f64, f64, f64, memref<?x?xf64>, f64, f64, f64, i1) -> (f64, f64, f64)
      %bcr = PME_F64_TO_REAL %bcd : f64 to !pme_real
      %fr = PME_F64_TO_REAL %fd : f64 to !pme_real
      %vdr = PME_F64_TO_REAL %vdd : f64 to !pme_real
      scf.yield %bcr, %fr, %vdr : !pme_real, !pme_real, !pme_real
    }
    %base2 = arith.addi %base, %z : index
    %re_at = arith.muli %base2, %c2 : index
    %im_at = arith.addi %re_at, %c1 : index
    %re = memref.load %c[%re_at] : memref<?x!pme_real>
    %im = memref.load %c[%im_at] : memref<?x!pme_real>
    // w = 1 for the plane z = 0 and, with k3 even, z = k3 / 2.
    %first = arith.cmpi eq, %z, %c0 : index
    %middle = arith.cmpi eq, %z, %h : index
    %middle_even = arith.andi %middle, %even : i1
    %single = arith.ori %first, %middle_even : i1
    %sums_p3m = arith.andi %sums, %p3m : i1
    %sums_plain = arith.xori %p3m, %true : i1
    %sums_other = arith.andi %sums, %sums_plain : i1
    %en, %wn = scf.if %sums_other -> (!pme_real, vector<9x!pme_real>) {
      %es, %ws = func.call @mdrt.pme_point_sums(%re, %im, %m1, %m2, %m3, %bc, %factor, %vd,
                                                %single, %e3, %w3)
          : (!pme_real, !pme_real, !pme_real, !pme_real, !pme_real, !pme_real, !pme_real,
             !pme_real, i1, !pme_real, vector<9x!pme_real>) -> (!pme_real, vector<9x!pme_real>)
      scf.yield %es, %ws : !pme_real, vector<9x!pme_real>
    } else {
      %ep, %wp = scf.if %sums_p3m -> (!pme_real, vector<9x!pme_real>) {
        %c2p = arith.constant 2 : index
        %xa = arith.muli %a, %c2p : index
        %xa0 = arith.addi %extra, %xa : index
        %xa1 = arith.addi %xa0, %c1 : index
        %pbx = arith.addi %k1, %b : index
        %xb = arith.muli %pbx, %c2p : index
        %xb0 = arith.addi %extra, %xb : index
        %xb1 = arith.addi %xb0, %c1 : index
        %xz = arith.muli %pz, %c2p : index
        %xz0 = arith.addi %extra, %xz : index
        %xz1 = arith.addi %xz0, %c1 : index
        %ga1 = memref.load %tables[%xa0] : memref<?x!pme_real>
        %gs1 = memref.load %tables[%xa1] : memref<?x!pme_real>
        %ga2 = memref.load %tables[%xb0] : memref<?x!pme_real>
        %gs2 = memref.load %tables[%xb1] : memref<?x!pme_real>
        %ga3 = memref.load %tables[%xz0] : memref<?x!pme_real>
        %gs3 = memref.load %tables[%xz1] : memref<?x!pme_real>
        %pav = vector.from_elements %m1, %m2, %m3 : vector<3x!pme_real>
        %psv = vector.from_elements %m1s, %m2s, %m3s : vector<3x!pme_real>
        %gav = vector.from_elements %ga1, %ga2, %ga3 : vector<3x!pme_real>
        %gsv = vector.from_elements %gs1, %gs2, %gs3 : vector<3x!pme_real>
        %msqp = arith.addf %m12s, %m3s : !pme_real
        %originp = arith.cmpf oeq, %msqp, %rzero : !pme_real
        %safep = arith.select %originp, %rone, %msqp : !pme_real
        %inversep = arith.divf %rone, %safep : !pme_real
        %es, %ws = func.call @mdrt.pme_point_sums_p3m(%re, %im, %pav, %psv, %gav, %gsv, %bc,
                                                      %inversep, %gaussr, %single, %e3, %w3)
            : (!pme_real, !pme_real, vector<3x!pme_real>, vector<3x!pme_real>,
               vector<3x!pme_real>, vector<3x!pme_real>, !pme_real, !pme_real, !pme_real, i1,
               !pme_real, vector<9x!pme_real>) -> (!pme_real, vector<9x!pme_real>)
        scf.yield %es, %ws : !pme_real, vector<9x!pme_real>
      } else {
        scf.yield %e3, %w3 : !pme_real, vector<9x!pme_real>
      }
      scf.yield %ep, %wp : !pme_real, vector<9x!pme_real>
    }
    // The product with the influence function, for the forces.
    %reb = arith.mulf %re, %bc : !pme_real
    %imb = arith.mulf %im, %bc : !pme_real
    memref.store %reb, %c[%re_at] : memref<?x!pme_real>
    memref.store %imb, %c[%im_at] : memref<?x!pme_real>
    scf.yield %en, %wn : !pme_real, vector<9x!pme_real>
  }
  %e64 = PME_REAL_TO_F64 %e : !pme_real to f64
  %w64 = PME_REAL_TO_F64 %w : vector<9x!pme_real> to vector<9xf64>
  return %e64, %w64 : f64, vector<9xf64>
}

// As pme_convolve_row, in a triclinic cell, with the wave vectors
// k = H⁻¹ m, whose influence is not separable along the edges.
func.func private @mdrt.pme_convolve_row_triclinic(%c: memref<?x!pme_real>, %tables: memref<?x!pme_real>,
                                                   %moduli: memref<?x?xf64>, %box: vector<6xf64>,
                                                   %beta: f64, %coulomb: f64, %dispersion: i1,
                                                   %k1: index, %k2: index, %k3: index, %row: index,
                                                   %sums: i1)
    -> (f64, vector<9xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %rzero = arith.constant 0.0 : !pme_real
  %rnone = arith.constant dense<0.0> : vector<9x!pme_real>
  %unit = arith.constant 1.0 : f64
  %lx = vector.extract %box[0] : f64 from vector<6xf64>
  %ly = vector.extract %box[1] : f64 from vector<6xf64>
  %lz = vector.extract %box[2] : f64 from vector<6xf64>
  %lxy = arith.mulf %lx, %ly : f64
  %volume = arith.mulf %lxy, %lz : f64
  %h00, %h10, %h11, %h20, %h21, %h22 = func.call @mdrt.pme_inverse_cell(%box)
      : (vector<6xf64>) -> (f64, f64, f64, f64, f64, f64)
  %a = arith.divui %row, %k2 : index
  %b = arith.remui %row, %k2 : index
  %h = arith.divui %k3, %c2 : index
  %h1 = arith.addi %h, %c1 : index
  %odd = arith.remui %k3, %c2 : index
  %even = arith.cmpi eq, %odd, %c0 : index
  // k = H⁻¹ m: its x from m1, its y from m1 and m2, its z from all.
  %i1 = func.call @mdrt.pme_wave(%a, %k1, %unit) : (index, index, f64) -> f64
  %m1 = arith.mulf %i1, %h00 : f64
  %i1_h10 = arith.mulf %i1, %h10 : f64
  %i1_h20 = arith.mulf %i1, %h20 : f64
  %i2 = func.call @mdrt.pme_wave(%b, %k2, %unit) : (index, index, f64) -> f64
  %i2_h11 = arith.mulf %i2, %h11 : f64
  %m2 = arith.addf %i1_h10, %i2_h11 : f64
  %i2_h21 = arith.mulf %i2, %h21 : f64
  %i12_z = arith.addf %i1_h20, %i2_h21 : f64
  %m1r = PME_F64_TO_REAL %m1 : f64 to !pme_real
  %m2r = PME_F64_TO_REAL %m2 : f64 to !pme_real
  %mod1 = memref.load %moduli[%c0, %a] : memref<?x?xf64>
  %mod2 = memref.load %moduli[%c1, %b] : memref<?x?xf64>
  %mod12 = arith.mulf %mod1, %mod2 : f64
  %base = arith.muli %row, %h1 : index
  %e, %w = scf.for %z = %c0 to %h1 step %c1
      iter_args(%e3 = %rzero, %w3 = %rnone) -> (!pme_real, vector<9x!pme_real>) {
    %zi = arith.index_cast %z : index to i64
    %zf = arith.sitofp %zi : i64 to f64
    %i3_h22 = arith.mulf %zf, %h22 : f64
    %m3 = arith.addf %i12_z, %i3_h22 : f64
    %m3r = PME_F64_TO_REAL %m3 : f64 to !pme_real
    %mod3 = memref.load %moduli[%c2, %z] : memref<?x?xf64>
    %mods = arith.mulf %mod12, %mod3 : f64
    %bcd, %fd, %vdd = func.call @mdrt.pme_influence(%m1, %m2, %m3, %mods, %moduli, %volume,
                                                     %beta, %coulomb, %dispersion)
        : (f64, f64, f64, f64, memref<?x?xf64>, f64, f64, f64, i1) -> (f64, f64, f64)
    %bc = PME_F64_TO_REAL %bcd : f64 to !pme_real
    %factor = PME_F64_TO_REAL %fd : f64 to !pme_real
    %vd = PME_F64_TO_REAL %vdd : f64 to !pme_real
    %base2 = arith.addi %base, %z : index
    %re_at = arith.muli %base2, %c2 : index
    %im_at = arith.addi %re_at, %c1 : index
    %re = memref.load %c[%re_at] : memref<?x!pme_real>
    %im = memref.load %c[%im_at] : memref<?x!pme_real>
    // w = 1 for the plane z = 0 and, with k3 even, z = k3 / 2.
    %first = arith.cmpi eq, %z, %c0 : index
    %middle = arith.cmpi eq, %z, %h : index
    %middle_even = arith.andi %middle, %even : i1
    %single = arith.ori %first, %middle_even : i1
    %en, %wn = scf.if %sums -> (!pme_real, vector<9x!pme_real>) {
      %es, %ws = func.call @mdrt.pme_point_sums(%re, %im, %m1r, %m2r, %m3r, %bc, %factor, %vd,
                                                %single, %e3, %w3)
          : (!pme_real, !pme_real, !pme_real, !pme_real, !pme_real, !pme_real, !pme_real,
             !pme_real, i1, !pme_real, vector<9x!pme_real>) -> (!pme_real, vector<9x!pme_real>)
      scf.yield %es, %ws : !pme_real, vector<9x!pme_real>
    } else {
      scf.yield %e3, %w3 : !pme_real, vector<9x!pme_real>
    }
    // The product with the influence function, for the forces.
    %reb = arith.mulf %re, %bc : !pme_real
    %imb = arith.mulf %im, %bc : !pme_real
    memref.store %reb, %c[%re_at] : memref<?x!pme_real>
    memref.store %imb, %c[%im_at] : memref<?x!pme_real>
    scf.yield %en, %wn : !pme_real, vector<9x!pme_real>
  }
  %e64 = PME_REAL_TO_F64 %e : !pme_real to f64
  %w64 = PME_REAL_TO_F64 %w : vector<9x!pme_real> to vector<9xf64>
  return %e64, %w64 : f64, vector<9xf64>
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
// with w(m) = 2 for the points whose conjugate is not stored: a row of the
// grid an iteration of a parallel loop, the rows summed in f64 in the fixed
// order of D171.
func.func private @mdrt.pme_convolve(%c: memref<?x!pme_real>, %moduli: memref<?x?xf64>,
                                     %tables: memref<?x!pme_real>,
                                     %box: vector<3xf64>, %beta: f64, %coulomb: f64, %dispersion: i1,
                                     %k1: index, %k2: index, %k3: index)
    -> (f64, vector<9xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %none = arith.constant dense<0.0> : vector<9xf64>
  %true = arith.constant true
  func.call @mdrt.pme_tables(%moduli, %tables, %box, %beta, %coulomb, %k1, %k2, %k3)
      : (memref<?x?xf64>, memref<?x!pme_real>, vector<3xf64>, f64, f64, index, index, index) -> ()
  %rows = arith.muli %k1, %k2 : index
  %energy, %virial = scf.parallel (%r) = (%c0) to (%rows) step (%c1)
      init (%zero, %none) -> (f64, vector<9xf64>) {
    %er, %wr = func.call @mdrt.pme_convolve_row(%c, %tables, %moduli, %box, %beta, %coulomb,
                                                %dispersion, %k1, %k2, %k3, %r, %true)
        : (memref<?x!pme_real>, memref<?x!pme_real>, memref<?x?xf64>, vector<3xf64>, f64, f64,
           i1, index, index, index, index, i1) -> (f64, vector<9xf64>)
    scf.reduce(%er, %wr : f64, vector<9xf64>) {
    ^bb0(%l: f64, %rr: f64):
      %s = arith.addf %l, %rr : f64
      scf.reduce.return %s : f64
    }, {
    ^bb0(%l: vector<9xf64>, %rr: vector<9xf64>):
      %s = arith.addf %l, %rr : vector<9xf64>
      scf.reduce.return %s : vector<9xf64>
    }
  }
  return %energy, %virial : f64, vector<9xf64>
}

// As pme_convolve, for a step that writes neither the energy nor the
// virial: the product alone, as on a device.
func.func private @mdrt.pme_scale(%c: memref<?x!pme_real>, %moduli: memref<?x?xf64>,
                                  %tables: memref<?x!pme_real>,
                                  %box: vector<3xf64>, %beta: f64, %coulomb: f64, %dispersion: i1,
                                  %k1: index, %k2: index, %k3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  func.call @mdrt.pme_tables(%moduli, %tables, %box, %beta, %coulomb, %k1, %k2, %k3)
      : (memref<?x?xf64>, memref<?x!pme_real>, vector<3xf64>, f64, f64, index, index, index) -> ()
  %rows = arith.muli %k1, %k2 : index
  scf.parallel (%r) = (%c0) to (%rows) step (%c1) {
    %er, %wr = func.call @mdrt.pme_convolve_row(%c, %tables, %moduli, %box, %beta, %coulomb,
                                                %dispersion, %k1, %k2, %k3, %r, %false)
        : (memref<?x!pme_real>, memref<?x!pme_real>, memref<?x?xf64>, vector<3xf64>, f64, f64,
           i1, index, index, index, index, i1) -> (f64, vector<9xf64>)
    scf.reduce
  }
  return
}

// As pme_convolve, in a triclinic cell, with the wave vectors k = H⁻¹ m.
func.func private @mdrt.pme_convolve_triclinic(%c: memref<?x!pme_real>, %moduli: memref<?x?xf64>,
                                     %tables: memref<?x!pme_real>,
                                     %box: vector<6xf64>, %beta: f64, %coulomb: f64, %dispersion: i1,
                                     %k1: index, %k2: index, %k3: index)
    -> (f64, vector<9xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %none = arith.constant dense<0.0> : vector<9xf64>
  %true = arith.constant true
  %rows = arith.muli %k1, %k2 : index
  %energy, %virial = scf.parallel (%r) = (%c0) to (%rows) step (%c1)
      init (%zero, %none) -> (f64, vector<9xf64>) {
    %er, %wr = func.call @mdrt.pme_convolve_row_triclinic(%c, %tables, %moduli, %box, %beta,
                                                          %coulomb, %dispersion, %k1, %k2, %k3,
                                                          %r, %true)
        : (memref<?x!pme_real>, memref<?x!pme_real>, memref<?x?xf64>, vector<6xf64>, f64, f64,
           i1, index, index, index, index, i1) -> (f64, vector<9xf64>)
    scf.reduce(%er, %wr : f64, vector<9xf64>) {
    ^bb0(%l: f64, %rr: f64):
      %s = arith.addf %l, %rr : f64
      scf.reduce.return %s : f64
    }, {
    ^bb0(%l: vector<9xf64>, %rr: vector<9xf64>):
      %s = arith.addf %l, %rr : vector<9xf64>
      scf.reduce.return %s : vector<9xf64>
    }
  }
  return %energy, %virial : f64, vector<9xf64>
}

// As pme_scale, in a triclinic cell.
func.func private @mdrt.pme_scale_triclinic(%c: memref<?x!pme_real>, %moduli: memref<?x?xf64>,
                                  %tables: memref<?x!pme_real>,
                                  %box: vector<6xf64>, %beta: f64, %coulomb: f64, %dispersion: i1,
                                  %k1: index, %k2: index, %k3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %rows = arith.muli %k1, %k2 : index
  scf.parallel (%r) = (%c0) to (%rows) step (%c1) {
    %er, %wr = func.call @mdrt.pme_convolve_row_triclinic(%c, %tables, %moduli, %box, %beta,
                                                          %coulomb, %dispersion, %k1, %k2, %k3,
                                                          %r, %false)
        : (memref<?x!pme_real>, memref<?x!pme_real>, memref<?x?xf64>, vector<6xf64>, f64, f64,
           i1, index, index, index, index, i1) -> (f64, vector<9xf64>)
    scf.reduce
  }
  return
}

// The sums Σ_k φ(k) ∂θ_i(k)/∂u_a of particle `i` along the three edges,
// from its first points and its weights and slopes, in !pme_real.
func.func private @mdrt.pme_gather_sums(%phi: memref<?x!pme_real>,
                                        %placex: index, %placey: index, %placez: index,
                                        %wx: memref<8x!pme_real>, %wy: memref<8x!pme_real>,
                                        %wz: memref<8x!pme_real>, %dx: memref<8x!pme_real>,
                                        %dy: memref<8x!pme_real>, %dz: memref<8x!pme_real>,
                                        %k1: index, %k2: index, %k3: index, %n: index)
    -> (f64, f64, f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // The first points within the grid, so that a point needs one
  // subtraction at most to wrap around, not a division.
  %sx = arith.remui %placex, %k1 : index
  %sy = arith.remui %placey, %k2 : index
  %sz = arith.remui %placez, %k3 : index
  %zero = arith.constant 0.0 : !pme_real
  %gx, %gy, %gz = scf.for %j1 = %c0 to %n step %c1
      iter_args(%ax = %zero, %ay = %zero, %az = %zero) -> (!pme_real, !pme_real, !pme_real) {
    %g1s = arith.addi %sx, %j1 : index
    %g1w = arith.subi %g1s, %k1 : index
    %g1o = arith.cmpi uge, %g1s, %k1 : index
    %g1 = arith.select %g1o, %g1w, %g1s : index
    %w1 = memref.load %wx[%j1] : memref<8x!pme_real>
    %d1 = memref.load %dx[%j1] : memref<8x!pme_real>
    %bx, %by, %bz = scf.for %j2 = %c0 to %n step %c1
        iter_args(%cx = %ax, %cy = %ay, %cz = %az) -> (!pme_real, !pme_real, !pme_real) {
      %g2s = arith.addi %sy, %j2 : index
      %g2w = arith.subi %g2s, %k2 : index
      %g2o = arith.cmpi uge, %g2s, %k2 : index
      %g2 = arith.select %g2o, %g2w, %g2s : index
      %w2 = memref.load %wy[%j2] : memref<8x!pme_real>
      %d2 = memref.load %dy[%j2] : memref<8x!pme_real>
      %row1 = arith.muli %g1, %k2 : index
      %row = arith.addi %row1, %g2 : index
      %base = arith.muli %row, %k3 : index
      %d1w2 = arith.mulf %d1, %w2 : !pme_real
      %w1d2 = arith.mulf %w1, %d2 : !pme_real
      %w1w2 = arith.mulf %w1, %w2 : !pme_real
      %ex, %ey, %ez = scf.for %j3 = %c0 to %n step %c1
          iter_args(%tx = %cx, %ty = %cy, %tz = %cz) -> (!pme_real, !pme_real, !pme_real) {
        %g3s = arith.addi %sz, %j3 : index
        %g3w = arith.subi %g3s, %k3 : index
        %g3o = arith.cmpi uge, %g3s, %k3 : index
        %g3 = arith.select %g3o, %g3w, %g3s : index
        %w3 = memref.load %wz[%j3] : memref<8x!pme_real>
        %d3 = memref.load %dz[%j3] : memref<8x!pme_real>
        %at = arith.addi %base, %g3 : index
        %p = memref.load %phi[%at] : memref<?x!pme_real>
        %px = arith.mulf %d1w2, %w3 : !pme_real
        %py = arith.mulf %w1d2, %w3 : !pme_real
        %pz = arith.mulf %w1w2, %d3 : !pme_real
        %vx = arith.mulf %p, %px : !pme_real
        %vy = arith.mulf %p, %py : !pme_real
        %vz = arith.mulf %p, %pz : !pme_real
        %nx = arith.addf %tx, %vx : !pme_real
        %ny = arith.addf %ty, %vy : !pme_real
        %nz = arith.addf %tz, %vz : !pme_real
        scf.yield %nx, %ny, %nz : !pme_real, !pme_real, !pme_real
      }
      scf.yield %ex, %ey, %ez : !pme_real, !pme_real, !pme_real
    }
    scf.yield %bx, %by, %bz : !pme_real, !pme_real, !pme_real
  }
  %gx64 = PME_REAL_TO_F64 %gx : !pme_real to f64
  %gy64 = PME_REAL_TO_F64 %gy : !pme_real to f64
  %gz64 = PME_REAL_TO_F64 %gz : !pme_real to f64
  return %gx64, %gy64, %gz64 : f64, f64, f64
}

// The sum Σ_k φ(k) θ_i(k) of particle `i`, in !pme_real.
func.func private @mdrt.pme_potential_sum(%phi: memref<?x!pme_real>,
                                          %placex: index, %placey: index, %placez: index,
                                          %wx: memref<8x!pme_real>, %wy: memref<8x!pme_real>,
                                          %wz: memref<8x!pme_real>,
                                          %k1: index, %k2: index, %k3: index, %n: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // The first points within the grid, so that a point needs one
  // subtraction at most to wrap around, not a division.
  %sx = arith.remui %placex, %k1 : index
  %sy = arith.remui %placey, %k2 : index
  %sz = arith.remui %placez, %k3 : index
  %zero = arith.constant 0.0 : !pme_real
  %g = scf.for %j1 = %c0 to %n step %c1 iter_args(%a = %zero) -> (!pme_real) {
    %g1s = arith.addi %sx, %j1 : index
    %g1w = arith.subi %g1s, %k1 : index
    %g1o = arith.cmpi uge, %g1s, %k1 : index
    %g1 = arith.select %g1o, %g1w, %g1s : index
    %w1 = memref.load %wx[%j1] : memref<8x!pme_real>
    %b = scf.for %j2 = %c0 to %n step %c1 iter_args(%c = %a) -> (!pme_real) {
      %g2s = arith.addi %sy, %j2 : index
      %g2w = arith.subi %g2s, %k2 : index
      %g2o = arith.cmpi uge, %g2s, %k2 : index
      %g2 = arith.select %g2o, %g2w, %g2s : index
      %w2 = memref.load %wy[%j2] : memref<8x!pme_real>
      %row1 = arith.muli %g1, %k2 : index
      %row = arith.addi %row1, %g2 : index
      %base = arith.muli %row, %k3 : index
      %w1w2 = arith.mulf %w1, %w2 : !pme_real
      %e = scf.for %j3 = %c0 to %n step %c1 iter_args(%t = %c) -> (!pme_real) {
        %g3s = arith.addi %sz, %j3 : index
        %g3w = arith.subi %g3s, %k3 : index
        %g3o = arith.cmpi uge, %g3s, %k3 : index
        %g3 = arith.select %g3o, %g3w, %g3s : index
        %w3 = memref.load %wz[%j3] : memref<8x!pme_real>
        %at = arith.addi %base, %g3 : index
        %p = memref.load %phi[%at] : memref<?x!pme_real>
        %w = arith.mulf %w1w2, %w3 : !pme_real
        %v = arith.mulf %p, %w : !pme_real
        %next = arith.addf %t, %v : !pme_real
        scf.yield %next : !pme_real
      }
      scf.yield %e : !pme_real
    }
    scf.yield %b : !pme_real
  }
  %g64 = PME_REAL_TO_F64 %g : !pme_real to f64
  return %g64 : f64
}

// The force on particle `i`, F_i = −q_i Σ_k φ(k) ∇_i θ_i(k), from the
// potential `phi` on the grid, which is the backward transform of the
// product of the transform of the charges with the influence function. A
// function of its own, so that its arrays live on the stack of a call.
func.func private @mdrt.pme_gather_one(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                       %phi: memref<?x!pme_real>, %box: vector<3xf64>,
                                       %k1: index, %k2: index, %k3: index, %n: index,
                                       %f: memref<?x3x!pme_frc>, %i: index, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
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
  %wx = memref.alloca() : memref<8x!pme_real>
  %wy = memref.alloca() : memref<8x!pme_real>
  %wz = memref.alloca() : memref<8x!pme_real>
  %dx = memref.alloca() : memref<8x!pme_real>
  %dy = memref.alloca() : memref<8x!pme_real>
  %dz = memref.alloca() : memref<8x!pme_real>
  %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
  %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
  %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
  %qs = memref.load %q[%i] : memref<?x!pme_chg>
  %xi = PME_EXTEND_POS %xs : !pme_pos to f64
  %yi = PME_EXTEND_POS %ys : !pme_pos to f64
  %zi = PME_EXTEND_POS %zs : !pme_pos to f64
  %qi = PME_EXTEND_CHG %qs : !pme_chg to f64
  %sx, %fx = func.call @mdrt.pme_place(%xi, %lx, %k1, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sy, %fy = func.call @mdrt.pme_place(%yi, %ly, %k2, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sz, %fz = func.call @mdrt.pme_place(%zi, %lz, %k3, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  %gx, %gy, %gz = func.call @mdrt.pme_gather_sums(%phi, %sx, %sy, %sz, %wx, %wy, %wz, %dx, %dy, %dz,
                                                  %k1, %k2, %k3, %n)
      : (memref<?x!pme_real>, index, index, index, memref<8x!pme_real>, memref<8x!pme_real>,
         memref<8x!pme_real>, memref<8x!pme_real>, memref<8x!pme_real>, memref<8x!pme_real>,
         index, index, index, index) -> (f64, f64, f64)
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
                                   %phi: memref<?x!pme_real>, %box: vector<3xf64>,
                                   %k1: index, %k2: index, %k3: index, %n: index,
                                   %f: memref<?x3x!pme_frc>, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    func.call @mdrt.pme_gather_one(%x, %q, %phi, %box, %k1, %k2, %k3, %n, %f, %i, %moduli)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, memref<?x!pme_real>, vector<3xf64>,
           index, index, index, index, memref<?x3x!pme_frc>, index, memref<?x?xf64>) -> ()
    scf.reduce
  }
  return
}

// The potential of the grid at particle `i`, Σ_k φ(k) θ_i(k): the
// derivative of the energy of the sum in the charge of the particle, the
// fourth result of `md.reciprocal` (D231). The places
// and the splines are those of @mdrt.pme_gather_one; the buffer holds
// one number for each particle, in the type that the op stores it in.
func.func private @mdrt.pme_potential_one(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                       %phi: memref<?x!pme_real>, %box: vector<3xf64>,
                                       %k1: index, %k2: index, %k3: index, %n: index,
                                       %f: memref<?x!pme_frc>, %i: index, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %wx = memref.alloca() : memref<8x!pme_real>
  %wy = memref.alloca() : memref<8x!pme_real>
  %wz = memref.alloca() : memref<8x!pme_real>
  %dx = memref.alloca() : memref<8x!pme_real>
  %dy = memref.alloca() : memref<8x!pme_real>
  %dz = memref.alloca() : memref<8x!pme_real>
  %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
  %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
  %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
  %xi = PME_EXTEND_POS %xs : !pme_pos to f64
  %yi = PME_EXTEND_POS %ys : !pme_pos to f64
  %zi = PME_EXTEND_POS %zs : !pme_pos to f64
  %sx, %fx = func.call @mdrt.pme_place(%xi, %lx, %k1, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sy, %fy = func.call @mdrt.pme_place(%yi, %ly, %k2, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sz, %fz = func.call @mdrt.pme_place(%zi, %lz, %k3, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  %g = func.call @mdrt.pme_potential_sum(%phi, %sx, %sy, %sz, %wx, %wy, %wz, %k1, %k2, %k3, %n)
      : (memref<?x!pme_real>, index, index, index, memref<8x!pme_real>, memref<8x!pme_real>,
         memref<8x!pme_real>, index, index, index, index) -> f64
  %stored = PME_NARROW_FRC %g : f64 to !pme_frc
  memref.store %stored, %f[%i] : memref<?x!pme_frc>
  return
}

// The potential at all particles.
func.func private @mdrt.pme_potential(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %phi: memref<?x!pme_real>, %box: vector<3xf64>,
                                   %k1: index, %k2: index, %k3: index, %n: index,
                                   %f: memref<?x!pme_frc>, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    func.call @mdrt.pme_potential_one(%x, %q, %phi, %box, %k1, %k2, %k3, %n, %f, %i, %moduli)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, memref<?x!pme_real>, vector<3xf64>,
           index, index, index, index, memref<?x!pme_frc>, index, memref<?x?xf64>) -> ()
    scf.reduce
  }
  return
}

// The places and the weights of particle `i` in a triclinic cell, from its
// fractional coordinates s = x H⁻¹: its first points, written to `starts`.
func.func private @mdrt.pme_splines_triclinic(%x: memref<?x3x!pme_pos>, %box: vector<6xf64>,
                                              %k1: index, %k2: index, %k3: index, %n: index,
                                              %i: index, %moduli: memref<?x?xf64>,
                                              %wx: memref<8x!pme_real>, %wy: memref<8x!pme_real>,
                                              %wz: memref<8x!pme_real>, %dx: memref<8x!pme_real>,
                                              %dy: memref<8x!pme_real>, %dz: memref<8x!pme_real>)
    -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %unit = arith.constant 1.0 : f64
  %h00, %h10, %h11, %h20, %h21, %h22 = func.call @mdrt.pme_inverse_cell(%box)
      : (vector<6xf64>) -> (f64, f64, f64, f64, f64, f64)
  %xs = memref.load %x[%i, %c0] : memref<?x3x!pme_pos>
  %ys = memref.load %x[%i, %c1] : memref<?x3x!pme_pos>
  %zs = memref.load %x[%i, %c2] : memref<?x3x!pme_pos>
  %xi = PME_EXTEND_POS %xs : !pme_pos to f64
  %yi = PME_EXTEND_POS %ys : !pme_pos to f64
  %zi = PME_EXTEND_POS %zs : !pme_pos to f64
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
  %sx, %fx = func.call @mdrt.pme_place(%s1, %unit, %k1, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sy, %fy = func.call @mdrt.pme_place(%s2, %unit, %k2, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  %sz, %fz = func.call @mdrt.pme_place(%s3, %unit, %k3, %n, %moduli) : (f64, f64, index, index, memref<?x?xf64>) -> (index, f64)
  func.call @mdrt.pme_bspline(%fx, %n, %wx, %dx, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fy, %n, %wy, %dy, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  func.call @mdrt.pme_bspline(%fz, %n, %wz, %dz, %moduli) : (f64, index, memref<8x!pme_real>, memref<8x!pme_real>, memref<?x?xf64>) -> ()
  return %sx, %sy, %sz : index, index, index
}

// As pme_gather_one, in a triclinic cell.
func.func private @mdrt.pme_gather_one_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                       %phi: memref<?x!pme_real>, %box: vector<6xf64>,
                                       %k1: index, %k2: index, %k3: index, %n: index,
                                       %f: memref<?x3x!pme_frc>, %i: index, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %h00, %h10, %h11, %h20, %h21, %h22 = func.call @mdrt.pme_inverse_cell(%box)
      : (vector<6xf64>) -> (f64, f64, f64, f64, f64, f64)
  %k1i = arith.index_cast %k1 : index to i64
  %k2i = arith.index_cast %k2 : index to i64
  %k3i = arith.index_cast %k3 : index to i64
  %k1f = arith.sitofp %k1i : i64 to f64
  %k2f = arith.sitofp %k2i : i64 to f64
  %k3f = arith.sitofp %k3i : i64 to f64
  %wx = memref.alloca() : memref<8x!pme_real>
  %wy = memref.alloca() : memref<8x!pme_real>
  %wz = memref.alloca() : memref<8x!pme_real>
  %dx = memref.alloca() : memref<8x!pme_real>
  %dy = memref.alloca() : memref<8x!pme_real>
  %dz = memref.alloca() : memref<8x!pme_real>
  %qs = memref.load %q[%i] : memref<?x!pme_chg>
  %qi = PME_EXTEND_CHG %qs : !pme_chg to f64
  %sx, %sy, %sz = func.call @mdrt.pme_splines_triclinic(%x, %box, %k1, %k2, %k3, %n, %i, %moduli,
                                                        %wx, %wy, %wz, %dx, %dy, %dz)
      : (memref<?x3x!pme_pos>, vector<6xf64>, index, index, index, index, index, memref<?x?xf64>,
         memref<8x!pme_real>, memref<8x!pme_real>, memref<8x!pme_real>, memref<8x!pme_real>,
         memref<8x!pme_real>, memref<8x!pme_real>) -> (index, index, index)
  %gx, %gy, %gz = func.call @mdrt.pme_gather_sums(%phi, %sx, %sy, %sz, %wx, %wy, %wz, %dx, %dy, %dz,
                                                  %k1, %k2, %k3, %n)
      : (memref<?x!pme_real>, index, index, index, memref<8x!pme_real>, memref<8x!pme_real>,
         memref<8x!pme_real>, memref<8x!pme_real>, memref<8x!pme_real>, memref<8x!pme_real>,
         index, index, index, index) -> (f64, f64, f64)
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
                                   %phi: memref<?x!pme_real>, %box: vector<6xf64>,
                                   %k1: index, %k2: index, %k3: index, %n: index,
                                   %f: memref<?x3x!pme_frc>, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    func.call @mdrt.pme_gather_one_triclinic(%x, %q, %phi, %box, %k1, %k2, %k3, %n, %f, %i, %moduli)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, memref<?x!pme_real>, vector<6xf64>,
           index, index, index, index, memref<?x3x!pme_frc>, index, memref<?x?xf64>) -> ()
    scf.reduce
  }
  return
}

// The potential of the grid at particle `i` in a triclinic cell, as
// pme_potential_one gives it.
func.func private @mdrt.pme_potential_one_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                       %phi: memref<?x!pme_real>, %box: vector<6xf64>,
                                       %k1: index, %k2: index, %k3: index, %n: index,
                                       %f: memref<?x!pme_frc>, %i: index, %moduli: memref<?x?xf64>) {
  %wx = memref.alloca() : memref<8x!pme_real>
  %wy = memref.alloca() : memref<8x!pme_real>
  %wz = memref.alloca() : memref<8x!pme_real>
  %dx = memref.alloca() : memref<8x!pme_real>
  %dy = memref.alloca() : memref<8x!pme_real>
  %dz = memref.alloca() : memref<8x!pme_real>
  %sx, %sy, %sz = func.call @mdrt.pme_splines_triclinic(%x, %box, %k1, %k2, %k3, %n, %i, %moduli,
                                                        %wx, %wy, %wz, %dx, %dy, %dz)
      : (memref<?x3x!pme_pos>, vector<6xf64>, index, index, index, index, index, memref<?x?xf64>,
         memref<8x!pme_real>, memref<8x!pme_real>, memref<8x!pme_real>, memref<8x!pme_real>,
         memref<8x!pme_real>, memref<8x!pme_real>) -> (index, index, index)
  %g = func.call @mdrt.pme_potential_sum(%phi, %sx, %sy, %sz, %wx, %wy, %wz, %k1, %k2, %k3, %n)
      : (memref<?x!pme_real>, index, index, index, memref<8x!pme_real>, memref<8x!pme_real>,
         memref<8x!pme_real>, index, index, index, index) -> f64
  %stored = PME_NARROW_FRC %g : f64 to !pme_frc
  memref.store %stored, %f[%i] : memref<?x!pme_frc>
  return
}

// The potential at all particles in a triclinic cell.
func.func private @mdrt.pme_potential_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %phi: memref<?x!pme_real>, %box: vector<6xf64>,
                                   %k1: index, %k2: index, %k3: index, %n: index,
                                   %f: memref<?x!pme_frc>, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    func.call @mdrt.pme_potential_one_triclinic(%x, %q, %phi, %box, %k1, %k2, %k3, %n, %f, %i, %moduli)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, memref<?x!pme_real>, vector<6xf64>,
           index, index, index, index, memref<?x!pme_frc>, index, memref<?x?xf64>) -> ()
    scf.reduce
  }
  return
}

// The number of the listed particles of `list` (D262):
// its entries before the first negative one.
func.func private @mdrt.pme_count_listed(%list: memref<?xi32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %none = arith.constant 0 : i32
  %size = memref.dim %list, %c0 : memref<?xi32>
  %count = scf.for %k = %c0 to %size step %c1 iter_args(%n = %c0) -> (index) {
    %entry = memref.load %list[%k] : memref<?xi32>
    %listed = arith.cmpi sge, %entry, %none : i32
    %more = arith.addi %n, %c1 : index
    %next = arith.select %listed, %more, %n : index
    scf.yield %next : index
  }
  return %count : index
}

// Spreads the charges of the listed particles to `grid`, as
// @mdrt.pme_spread spreads those of all (D262): `list` holds
// their places, then −1 − the place of each of the others.
func.func private @mdrt.pme_spread_listed(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %list: memref<?xi32>,
                                   %box: vector<3xf64>, %grid: memref<?xi64>,
                                   %k1: index, %k2: index, %k3: index, %n: index, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @mdrt.pme_zero(%grid) : (memref<?xi64>) -> ()
  %listed_count = func.call @mdrt.pme_count_listed(%list) : (memref<?xi32>) -> index
  scf.parallel (%k) = (%c0) to (%listed_count) step (%c1) {
    %listed = memref.load %list[%k] : memref<?xi32>
    %i = arith.index_cast %listed : i32 to index
    func.call @mdrt.pme_spread_one(%x, %q, %box, %grid, %k1, %k2, %k3, %n, %moduli, %i)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, vector<3xf64>, memref<?xi64>,
           index, index, index, index, memref<?x?xf64>, index) -> ()
    scf.reduce
  }
  return
}

// The forces on the listed particles, as @mdrt.pme_gather gives them, and
// a force of zero on the others (D262).
func.func private @mdrt.pme_gather_listed(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %list: memref<?xi32>,
                                   %phi: memref<?x!pme_real>, %box: vector<3xf64>,
                                   %k1: index, %k2: index, %k3: index, %n: index,
                                   %f: memref<?x3x!pme_frc>, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  %zero = arith.constant 0.0 : f64
  %none = PME_NARROW_FRC %zero : f64 to !pme_frc
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    scf.for %c = %c0 to %c3 step %c1 {
      memref.store %none, %f[%i, %c] : memref<?x3x!pme_frc>
    }
    scf.reduce
  }
  %listed_count = func.call @mdrt.pme_count_listed(%list) : (memref<?xi32>) -> index
  scf.parallel (%k) = (%c0) to (%listed_count) step (%c1) {
    %listed = memref.load %list[%k] : memref<?xi32>
    %i = arith.index_cast %listed : i32 to index
    func.call @mdrt.pme_gather_one(%x, %q, %phi, %box, %k1, %k2, %k3, %n, %f, %i, %moduli)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, memref<?x!pme_real>, vector<3xf64>,
           index, index, index, index, memref<?x3x!pme_frc>, index, memref<?x?xf64>) -> ()
    scf.reduce
  }
  return
}

// Spreads the charges of the listed particles to `grid`, as
// @mdrt.pme_spread_triclinic spreads those of all (D262): `list` holds
// their places, then −1 − the place of each of the others.
func.func private @mdrt.pme_spread_listed_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %list: memref<?xi32>,
                                   %box: vector<6xf64>, %grid: memref<?xi64>,
                                   %k1: index, %k2: index, %k3: index, %n: index, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  func.call @mdrt.pme_zero(%grid) : (memref<?xi64>) -> ()
  %listed_count = func.call @mdrt.pme_count_listed(%list) : (memref<?xi32>) -> index
  scf.parallel (%k) = (%c0) to (%listed_count) step (%c1) {
    %listed = memref.load %list[%k] : memref<?xi32>
    %i = arith.index_cast %listed : i32 to index
    func.call @mdrt.pme_spread_one_triclinic(%x, %q, %box, %grid, %k1, %k2, %k3, %n, %moduli, %i)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, vector<6xf64>, memref<?xi64>,
           index, index, index, index, memref<?x?xf64>, index) -> ()
    scf.reduce
  }
  return
}

// The forces on the listed particles, as @mdrt.pme_gather_triclinic gives them, and
// a force of zero on the others (D262).
func.func private @mdrt.pme_gather_listed_triclinic(%x: memref<?x3x!pme_pos>, %q: memref<?x!pme_chg>,
                                   %list: memref<?xi32>,
                                   %phi: memref<?x!pme_real>, %box: vector<6xf64>,
                                   %k1: index, %k2: index, %k3: index, %n: index,
                                   %f: memref<?x3x!pme_frc>, %moduli: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %count = memref.dim %x, %c0 : memref<?x3x!pme_pos>
  %zero = arith.constant 0.0 : f64
  %none = PME_NARROW_FRC %zero : f64 to !pme_frc
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    scf.for %c = %c0 to %c3 step %c1 {
      memref.store %none, %f[%i, %c] : memref<?x3x!pme_frc>
    }
    scf.reduce
  }
  %listed_count = func.call @mdrt.pme_count_listed(%list) : (memref<?xi32>) -> index
  scf.parallel (%k) = (%c0) to (%listed_count) step (%c1) {
    %listed = memref.load %list[%k] : memref<?xi32>
    %i = arith.index_cast %listed : i32 to index
    func.call @mdrt.pme_gather_one_triclinic(%x, %q, %phi, %box, %k1, %k2, %k3, %n, %f, %i, %moduli)
        : (memref<?x3x!pme_pos>, memref<?x!pme_chg>, memref<?x!pme_real>, vector<6xf64>,
           index, index, index, index, memref<?x3x!pme_frc>, index, memref<?x?xf64>) -> ()
    scf.reduce
  }
  return
}
