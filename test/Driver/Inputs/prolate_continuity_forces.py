"""Binary64 mesh and point-charge Ewald force references for D440."""
from itertools import product
import math
from pathlib import Path
import sys

import mdir
import numpy as np
from numpy.polynomial import legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from mdir_physical_constants import COULOMB_NATIVE

target, work = sys.argv[1], Path(sys.argv[2])
edge, cutoff, bandwidth, grid = 2.4, 0.8, 9.5392, 48
q = np.array([1., -.7, .3, -.8, .6, -.4, .9, -.9])
# Keep the first frame free of direct pairs, with off-grid sites and a
# nonzero force on each ion. In the second frame several pairs interact.
x = np.array(list(product([.42, 1.62], repeat=3))) + np.random.default_rng(938).uniform(
    -.09, .09, size=(8, 3))
close = x.copy()
close[1] = x[0] + [.23, .17, -.11]
close[6] = x[7] + [-.28, .09, .13]


def density(continuity, terms):
    # Integral kernel sinc(c(x-y)/pi) on [-1,1], independent of the
    # frequency quadrature and Jacobi diagonalization used in the builder.
    u, w = legendre.leggauss(128)
    n = np.arange(0, 2*terms, 2)
    basis = legendre.legvander(u, 2*terms-2)[:, ::2] * np.sqrt(2*n + 1)
    constraints = np.array([np.ones(terms), n*(n+1)/2, (n-1)*n*(n+1)*(n+2)/8])
    z = np.linalg.svd(constraints[:continuity] * np.sqrt(2*n + 1), full_matrices=True)[2][continuity:].T
    projected = (w[:, None] * basis) @ z
    kernel = np.sinc(bandwidth * (u[:, None] - u) / np.pi)
    _, vectors = np.linalg.eigh(projected.T @ kernel @ projected)
    samples = w * (basis @ z @ vectors[:, -1])
    return u, samples / samples.sum()


def mesh_force(continuity, terms, window):
    axis = np.rint(np.fft.fftfreq(grid) * grid).astype(int)
    m2 = sum(a*a for a in np.meshgrid(axis, axis, axis, indexing="ij"))
    shells, indices = np.unique(m2, return_inverse=True)
    k = 2*np.pi / edge * np.sqrt(shells)
    u, weights = density(continuity, terms or 40)
    transform = np.cos(cutoff * k[:, None] * u) @ weights
    if not terms and continuity == 0:
        transform[k * cutoff >= bandwidth] = 0
    influence = np.zeros_like(k)
    influence[1:] = 4*np.pi*COULOMB_NATIVE/edge**3 * transform[1:] / k[1:]**2
    influence = influence[indices].reshape(m2.shape)
    rho = np.zeros_like(m2, dtype=float)
    t = x / edge * grid
    base = np.floor(t).astype(int)
    f = t - base
    if window == "BSPLINE":
        order = 4
        squared = np.sinc(axis / grid)**8
        weights = np.array([(1-f)**3, 4-6*f*f+3*f**3, 1+3*f+3*f*f-3*f**3, f**3])/6
        slopes = np.array([-(1-f)**2, -4*f+3*f*f, 1+2*f-3*f*f, f*f])/2
    else:
        # Independent differential-operator eigenproblem for the assignment.
        # Set tolerance from this window's endpoint so the default c1 is known.
        order = 6
        u, w = legendre.leggauss(128)
        values = window_polynomial(u) - window_polynomial(1.)
        squared = (3 * np.cos(6*np.pi*axis[:, None]*u/grid) @ (w*values))**2
        displacements = np.array([(f + 2 - j) / 3 for j in range(order)])
        weights = window_polynomial(displacements) - window_polynomial(1.)
        slopes = window_polynomial.deriv()(displacements) / 3
    for axis_factor in np.ix_(*([squared]*3)):
        influence /= axis_factor
    stencil = []
    for offsets in product(range(order), repeat=3):
        idx = tuple(((base + offsets - (order//2-1)) % grid).T)
        w = np.array([weights[j, :, a] for a, j in enumerate(offsets)])
        np.add.at(rho, idx, q * np.prod(w, axis=0))
        grad = [slopes[j, :, a] * w[(a+1)%3] * w[(a+2)%3] for a, j in enumerate(offsets)]
        stencil.append((idx, np.array(grad).T))
    potential = np.fft.ifftn(influence * np.fft.fftn(rho)).real * grid**3
    return -q[:, None] * grid/edge * sum(potential[idx][:, None]*g for idx, g in stencil)


def ewald_force():
    # beta*edge/2 > 6 and the omitted Gaussian modes are below 1e-22.
    beta = 5.6
    axis = np.arange(-32, 33)
    modes = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3)
    k = modes[np.any(modes, axis=1)] * (2*np.pi/edge)
    k2 = np.sum(k*k, axis=1)
    factor = 4*np.pi/edge**3 * np.exp(-k2/(4*beta**2)) / k2
    phase = np.exp(1j * (close @ k.T))
    force = q[:, None] * ((phase * (q @ phase.conj())).imag * factor) @ k
    for i, xi in enumerate(close):
        for j in range(i):
            d = xi - close[j]
            d -= edge * np.rint(d/edge)
            r = np.linalg.norm(d)
            pair = q[i]*q[j]*d/r**3 * (math.erfc(beta*r) +
                   2/math.sqrt(np.pi)*beta*r*np.exp(-(beta*r)**2))
            force[i] += pair
            force[j] -= pair
    return COULOMB_NATIVE * force


work.mkdir(parents=True, exist_ok=True)
top = work / "ions.top"
top.write_text("""[ defaults ]
1 2 no 1 1
[ atomtypes ]
X 1 1.0 0 A 0.3 0
[ moleculetype ]
ions 0
[ atoms ]
""" + "".join(f"{i} X 1 ION X {i} {charge} 1.0\n" for i, charge in enumerate(q, 1)) +
               "[ system ]\nEight ions\n[ molecules ]\nions 1\n")
loaded = mdir.load_gromacs(str(top), str(Path(__file__).parent / "zero-epsilon/argon.gro"))
system, state = loaded.make_system(), loaded.make_state()
system.electrostatics = mdir.Electrostatics.PME
system.cutoff, system.pairlist_distance = cutoff, .95
system.pme_kernel, system.pme_bandwidth = mdir.PMEKernel.Prolate, bandwidth
system.pme_grid = [grid]*3
# Diagonalize the differential operator in normalized even Legendre functions.
n = 2*np.arange(48)
diag = n*(n+1) + 15.0**2*(2*n*(n+1)-1)/((2*n+3)*(2*n-1))
off = 15.0**2*(n+2)*(n+1)/((2*n+3)*np.sqrt((2*n+1)*(2*n+5)))
_, vectors = np.linalg.eigh(np.diag(diag)+np.diag(off[:-1],1)+np.diag(off[:-1],-1))
coeff = np.zeros(2*len(n))
coeff[::2] = vectors[:,0] * np.sqrt(n+.5)
window_polynomial = legendre.Legendre(coeff)
if window_polynomial(0.) < 0:
    window_polynomial = -window_polynomial
system.pme_tolerance = 2 * window_polynomial(1.)
state.velocities = np.zeros_like(x)
full_reference = ewald_force()
for continuity, terms, window in [(0, 0, "PROLATE"), (2, 8, "BSPLINE"),
                                  (2, 40, "BSPLINE"), (3, 8, "BSPLINE"),
                                  (1, 8, "BSPLINE"), (0, 8, "BSPLINE"),
                                  (2, 8, "PROLATE")]:
    bandwidth = 14.471 if not terms else 9.5392
    system.pme_bandwidth = bandwidth
    system.pme_continuity = continuity
    if terms:
        system.pme_basis_terms = terms
    if (window == "PROLATE" and continuity) or (terms and not continuity):
        system.pme_window = window
    refs = [mesh_force(continuity, terms, window), full_reference]
    print(f"k={continuity} N={terms or 'default'} window={window}", flush=True)
    for mode in ("Mixed", "Double"):
        execution = mdir.Execution()
        execution.target, execution.precision = getattr(mdir.Target, target), getattr(mdir.Precision, mode)
        integrator = mdir.Integrator()
        integrator.method = mdir.IntegratorMethod.VelocityVerlet
        integrator.timestep = 1e-12
        for name, frame, ref in zip(("reciprocal", "full"), (x, close), refs):
            state.positions = frame
            program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution, mdir.Schedule())
            sim = mdir.Simulation(program)
            sim.run(1)
            sampled = sim.state()
            assert np.max(np.abs(sampled.positions - frame)) < 1e-7
            actual = np.asarray(sampled.forces, dtype=float)
            rms = np.sqrt(np.mean(ref**2))
            diff = np.sqrt(np.mean((actual-ref)**2))
            tol = (2e-4 if mode == "Mixed" else 2e-6) if name == "reciprocal" else 1e-3
            print(f"{target} {mode} {name}: reference RMS {rms:.9g}, difference {diff:.8g} "
                  f"kJ/mol/nm, relative {diff/rms:.7g}, tolerance {tol}", flush=True)
            assert diff < tol*rms
