"""Writes test/Integration/triplets.mlir and triplets-mixed.mlir: the
three-body term of Stillinger and Weber [StillingerWeber1985] over the
triplets of md.triplets, on eight particles, against the energy, the forces,
and the virial that this script computes.

The forces are the derivatives of the energy by the complex step, exact to
rounding; central differences check them. Run it from the root of the
repository after a change:

  python3 test/Integration/Inputs/generate_triplets_tests.py
"""

import cmath
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))

# The cell, the cutoffs, and the parameters of the term. The cutoff of the
# triplets is a sigma, where the term has its pole; that of the
# neighborhood is wider, so that the test of the triplets is the one that
# leaves the far legs out.
EDGE = 4.0
NEIGHBORHOOD = 1.5
LAMBDA = 3.0
GAMMA = 1.2
# sigma is that of mW over 3.6, so that a sigma in f32 rounds below the
# cutoff as it does for mW (D159).
SIGMA = 2.3925 / 3.6
A = 1.8
CUTOFF = A * SIGMA

# Particle 0 is a center with three legs within the cutoff, one of them
# through a face of the cell (to 1); 2 and 4 have two each, the leg 2-4
# within 0.02 of the cutoff; the far leg 1-2 is beyond the cutoff and is
# not tested; 3 is within the neighborhood of 0 and 2 but beyond the
# cutoff; 5 has one leg; 6 has two, of which the leg to 7 is below the
# cutoff by a few units in the last place of f64 (find_edge_end), so that
# a kernel in f32 meets the pole of the term there unless the cutoff is
# pulled in (D159).
POSITIONS = [
    [0.10, 2.00, 2.00],
    [3.30, 2.02, 1.97],
    [0.90, 2.30, 2.05],
    [0.12, 3.30, 2.01],
    [0.30, 2.10, 3.00],
    [2.00, 0.50, 0.50],
    [2.40, 0.52, 1.00],
    None,
]


def f32(value):
    """`value` rounded to f32."""
    return struct.unpack("f", struct.pack("f", value))[0]


def find_edge_end(start):
    """The coordinate of an end along x from `start` whose leg is just below
    CUTOFF in f64 but that a kernel in f32 computes at or beyond a sigma
    as f32 computes it, while its square passes a test of the cutoff in f32
    that is not pulled in: the pole of the term."""
    a_sigma = f32(f32(A) * f32(SIGMA))
    square = f32(CUTOFF * CUTOFF)
    end = start + CUTOFF
    for _ in range(1000):
        end = math.nextafter(end, 0.0)
        d = end - start
        r32 = f32(d)
        if d < CUTOFF and r32 >= a_sigma and f32(r32 * r32) < square:
            return end
    raise RuntimeError("no end at the pole")


POSITIONS[7] = [find_edge_end(POSITIONS[6][0]), POSITIONS[6][1],
                POSITIONS[6][2]]
EDGE_LEG = POSITIONS[7][0] - POSITIONS[6][0]


def image(d):
    return [c - EDGE * round(c.real / EDGE) for c in d]


def leg(x, end, center):
    d = image([x[end][a] - x[center][a] for a in range(3)])
    return d, cmath.sqrt(sum(c * c for c in d))


def triplets(x):
    """For each center, each pair of its legs within the cutoff once."""
    found = []
    n = len(x)
    for i in range(n):
        near = [j for j in range(n)
                if j != i and leg(x, j, i)[1].real < CUTOFF
                and leg(x, j, i)[1].real < NEIGHBORHOOD]
        for a in range(len(near)):
            for b in range(a + 1, len(near)):
                found.append((near[a], i, near[b]))
    return found


def term(r1, r2, c):
    a_sigma = A * SIGMA
    g_sigma = GAMMA * SIGMA
    t = c + 1.0 / 3.0
    return (LAMBDA * t * t * cmath.exp(g_sigma / (r1 - a_sigma))
            * cmath.exp(g_sigma / (r2 - a_sigma)))


def energy(x, found):
    total = 0.0
    for j, i, k in found:
        u, r1 = leg(x, j, i)
        v, r2 = leg(x, k, i)
        c = sum(p * q for p, q in zip(u, v)) / (r1 * r2)
        total += term(r1, r2, c)
    return total


def evaluate(x):
    found = triplets(x)
    e = energy(x, found).real
    h = 1e-30
    forces = []
    for m in range(len(x)):
        force = []
        for a in range(3):
            y = [list(map(complex, p)) for p in x]
            y[m][a] += 1j * h
            force.append(-energy(y, found).imag / h)
        forces.append(force)
    # The virial: over the triplets and their members, the displacement of
    # the member from the center times its force, as one triplet alone
    # gives it.
    virial = [[0.0] * 3 for _ in range(3)]
    for t in found:
        for m in t:
            force = []
            for a in range(3):
                y = [list(map(complex, p)) for p in x]
                y[m][a] += 1j * h
                force.append(-energy(y, [t]).imag / h)
            d = [c.real for c in image([x[m][a] - x[t[1]][a]
                                        for a in range(3)])]
            for a in range(3):
                for b in range(3):
                    virial[a][b] += d[a] * force[b]
    return found, e, forces, virial


def check(x, forces):
    """The largest difference of the forces from central differences of the
    energy, on the scale of the largest force."""
    h = 1e-6
    worst = 0.0
    largest = max(abs(c) for f in forces for c in f)
    for m in range(len(x)):
        for a in range(3):
            plus = [list(p) for p in x]
            minus = [list(p) for p in x]
            plus[m][a] += h
            minus[m][a] -= h
            # The triplets are found again: the leg 6-7 leaves the
            # cutoff, where the term vanishes.
            slope = (energy(plus, triplets(plus)).real
                     - energy(minus, triplets(minus)).real) / (2 * h)
            worst = max(worst, abs(-slope - forces[m][a]) / largest)
    return worst


def f(v):
    return repr(float(v))


def write(name, mixed, found, e, forces, virial):
    tolerance = "1.0e-5" if mixed else "1.0e-12"
    force_tolerance = "1.0e-5" if mixed else "1.0e-9"
    if mixed:
        run = '''// RUN: mdir-opt %s %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \\
// RUN:     --md-exec-assign-precision="mode=mixed" \\
// RUN:     --md-exec-assign-storage --convert-md-exec-to-loops \\
// RUN: | mlir-opt %lower_loops_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \\
// RUN: | FileCheck %s
'''
        what = ('''// The test of triplets.mlir in mixed precision: the positions are held in
// f64 and the kernels compute in f32. The leg 6-7 is at the cutoff, where
// the term has its pole, but for a few units in the last place of f64; in
// f32 it rounds onto the pole, and the triplets leave it out only because
// the cutoff is pulled in below it (D159, D160). Each check prints 1 if
// the value agrees with the reference to a relative tolerance of %s, or
// 0 if it does not.''' % tolerance)
    else:
        run = '''// RUN: mdir-opt %s %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \\
// RUN: | mlir-opt %lower_loops_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \\
// RUN: | FileCheck %s

// RUN: mdir-opt %s %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \\
// RUN: | mlir-opt %lower_loops_to_openmp \\
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \\
// RUN: | FileCheck %s
'''
        what = ('''// The energy to a relative tolerance of %s, and each component of the
// forces and of the virial to %s of the largest: each check prints 1 if
// it agrees, or 0 if it does not.''' % (tolerance, force_tolerance))

    lines = []
    lines.append('''// The three-body term of Stillinger and Weber [StillingerWeber1985],
//
//   E = sum over (j, i, k) of lambda (cos theta_jik + 1/3)^2
//         exp(gamma sigma / (r_ij - a sigma)) exp(gamma sigma / (r_ik - a sigma)),
//
// over the triplets of md.triplets (D160) at the cutoff a sigma, on eight
// particles in a periodic cell, compiled and run on the CPU and compared
// with Inputs/generate_triplets_tests.py, which generates this file. The
// neighborhood reaches farther than the triplets, so that the legs between
// the two cutoffs (0-3, 2-3) are left out by md.triplets; the far leg 1-2
// is beyond the cutoff and its triplet is kept. %d triplets.
//
''' % len(found))
    lines.append(run)
    lines.append(what + "\n")
    module = '''
!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
!trip  = !md.relation<@atoms, 3, reversal>

md.particle_set @atoms

md.potential @three_body(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(@NEIGHBORHOOD@) : !vec -> !pairs
  %t = md.triplets %n cutoff(@CUTOFF@) : !pairs -> !trip
  %u = md.sum_tuples %t, %x, %cell
         coordinates(distance(0, 1), distance(2, 1), cosine(0, 1, 2)) {
  ^bb0(%r1: f64, %r2: f64, %c: f64):
    %lambda = arith.constant @LAMBDA@ : f64
    %gamma  = arith.constant @GAMMA@ : f64
    %sigma  = arith.constant @SIGMA@ : f64
    %a      = arith.constant @A@ : f64
    %third  = arith.constant 0.3333333333333333 : f64
    %as     = arith.mulf %a, %sigma : f64
    %gs     = arith.mulf %gamma, %sigma : f64
    %t1     = arith.addf %c, %third : f64
    %t2     = arith.mulf %t1, %t1 : f64
    %d1     = arith.subf %r1, %as : f64
    %q1     = arith.divf %gs, %d1 : f64
    %e1     = math.exp %q1 : f64
    %d2     = arith.subf %r2, %as : f64
    %q2     = arith.divf %gs, %d2 : f64
    %e2     = math.exp %q2 : f64
    %l2     = arith.mulf %lambda, %t2 : f64
    %l3     = arith.mulf %l2, %e1 : f64
    %e      = arith.mulf %l3, %e2 : f64
    md.yield %e : f64
  } : !trip, !vec -> f64
  md.return %u : f64
}
'''
    for key, value in [("NEIGHBORHOOD", NEIGHBORHOOD), ("CUTOFF", CUTOFF),
                       ("LAMBDA", LAMBDA), ("GAMMA", GAMMA),
                       ("SIGMA", SIGMA), ("A", A)]:
        module = module.replace("@" + key + "@", f(value))
    lines.append(module)
    lines.append('''
func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64, %magnitude: f64,
                 %tolerance: f64) {
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %scale = math.absf %magnitude : f64
  %bound = arith.mulf %tolerance, %scale : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
''')
    rows = ", ".join("[" + ", ".join(f(c) for c in p) + "]" for p in POSITIONS)
    lines.append('''
memref.global "private" constant @positions : memref<%dx3xf64> =
    dense<[%s]>

func.func @main() {
  %%c0 = arith.constant 0 : index
  %%c1 = arith.constant 1 : index
  %%c2 = arith.constant 2 : index
  %%xs = memref.get_global @positions : memref<%dx3xf64>
  %%xd = memref.cast %%xs : memref<%dx3xf64> to memref<?x3xf64>
  %%n = memref.dim %%xd, %%c0 : memref<?x3xf64>
  %%buffer = memref.alloc(%%n) : memref<?x3xf64>
  memref.copy %%xd, %%buffer : memref<?x3xf64> to memref<?x3xf64>
  %%x = mdrt.from_buffer %%buffer : memref<?x3xf64> to !vec
  %%edge = arith.constant %s : f64
  %%cell = md.orthorhombic_cell %%edge, %%edge, %%edge
  %%u, %%f, %%w = md.evaluate @three_body(%%x, %%cell)
      request [energy, forces, virial]
      : (!vec, !md.cell) -> (f64, !vec, vector<9xf64>)
  %%forces = mdrt.to_buffer %%f : !vec to memref<?x3xf64>
  %%tol_e = arith.constant %s : f64
  %%tol_f = arith.constant %s : f64

  // Energy.
  // CHECK: 1
  %%u_ref = arith.constant %s : f64
  call @check(%%u, %%u_ref, %%u_ref, %%tol_e) : (f64, f64, f64, f64) -> ()
''' % (len(POSITIONS), rows, len(POSITIONS), len(POSITIONS), f(EDGE),
       tolerance, force_tolerance, f(e)))
    largest = max(abs(c) for fr in forces for c in fr)
    lines.append('''
  // Every component of the forces, on the scale of the largest.
  // CHECK-COUNT-%d: 1
  %%f_scale = arith.constant %s : f64
''' % (3 * len(forces), f(largest)))
    for m, fr in enumerate(forces):
        for a in range(3):
            idx = "%c" + str(m) if m < 3 else None
            lines.append('''  %%i%d_%d = arith.constant %d : index
  %%f%d_%d = memref.load %%forces[%%i%d_%d, %%c%d] : memref<?x3xf64>
  %%r%d_%d = arith.constant %s : f64
  call @check(%%f%d_%d, %%r%d_%d, %%f_scale, %%tol_f) : (f64, f64, f64, f64) -> ()
''' % (m, a, m, m, a, m, a, a, m, a, f(fr[a]), m, a, m, a))
    flat = [c for row in virial for c in row]
    lines.append('''
  // Every component of the virial, on the scale of the largest.
  // CHECK-COUNT-9: 1
  %%w_scale = arith.constant %s : f64
''' % f(max(abs(c) for c in flat)))
    for k, c in enumerate(flat):
        lines.append('''  %%w%d = vector.extract %%w[%d] : f64 from vector<9xf64>
  %%w%d_ref = arith.constant %s : f64
  call @check(%%w%d, %%w%d_ref, %%w_scale, %%tol_f) : (f64, f64, f64, f64) -> ()
''' % (k, k, k, f(c), k, k))
    lines.append('''  // CHECK-NOT: 0
  memref.dealloc %buffer : memref<?x3xf64>
  return
}
''')
    path = os.path.join(HERE, "..", name)
    with open(path, "w") as out:
        out.write("".join(lines))


def main():
    found, e, forces, virial = evaluate(POSITIONS)
    worst = check(POSITIONS, forces)
    assert worst < 1e-6, worst
    centers = [t[1] for t in found]
    # The cases that the arrangement is for.
    assert centers.count(0) == 3 and centers.count(2) == 1
    assert centers.count(4) == 1 and centers.count(6) == 1
    assert (1, 0, 2) in found or (2, 0, 1) in found
    assert leg(POSITIONS, 1, 2)[1].real > CUTOFF
    assert CUTOFF < leg(POSITIONS, 3, 0)[1].real < NEIGHBORHOOD
    assert CUTOFF < leg(POSITIONS, 3, 2)[1].real < NEIGHBORHOOD
    print("triplets", found)
    print("energy", e, "central differences", worst)
    print("edge leg", repr(EDGE_LEG), "cutoff", repr(CUTOFF))
    write("triplets.mlir", False, found, e, forces, virial)
    write("triplets-mixed.mlir", True, found, e, forces, virial)


if __name__ == "__main__":
    main()
