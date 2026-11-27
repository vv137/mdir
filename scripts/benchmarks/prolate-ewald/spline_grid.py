"""The automatic grid of the prolate kernel with the B-spline window
(D444, issue #985): the mesh error estimate against the
measured force error.

spline_grid.py functional C K0 K1 K2 L0 L1 L2 RC [CONTINUITY TERMS]
    The estimate sqrt(Q/R) of Builder::collectProlate by NumPy, summed over
    the whole grid and its images |j_a| <= 1 without the symmetries of the
    builder, with chi of the constrained density built here (the null space
    of the endpoint constraints and the eigenvector of the band-limited
    sinc kernel, as test/Driver/Inputs/prolate_continuity_forces.py builds
    it) and its transform by Gauss-Legendre quadrature: the reference of
    test/Driver/prolate-function.test.
spline_grid.py uncorrelated COUNT EDGE CUTOFF C...
    COUNT charges +1 and -1 at uncorrelated places in a cube of EDGE nm,
    cutoff CUTOFF nm, on the CPU in double precision through the Python
    API: for each bandwidth C, at 1, 1.5, 2, and 3 times the grid of eq. 15,
    the error of the reciprocal force (against six times that grid) over
    the reciprocal force (that grid less the direct sum by NumPy), and its
    ratio to the estimate (`functional`).
spline_grid.py make DIR NAME PRMTOP INPCRD CUTOFF [TARGET:PRECISION...]
    DIR/NAME/: the topology with every term but the charges set to zero
    (ParmEd), so that the forces are the Coulomb forces of the topology with
    its exclusions and 1-4 scaling; the reference, the Gaussian kernel at
    erfc 1e-10 of order 8 on 0.5 A in double precision; the prolate kernel
    (continuity 2, 8 terms, the B-spline window of order 4) at tolerances
    1e-3, 1e-4, 1e-5 on the grid of eq. 15 times 1 to 3 and at the automatic
    grid, for each TARGET:PRECISION (default GPU:MIXED GPU:DOUBLE).
spline_grid.py run MDIR DIR...
    Runs every case without f.h5 (one step of 1e-12 ps).
spline_grid.py report MDIR DIR...
    Delta = |F - F_ref| / |F_ref| over all particles, the grid, and the
    estimate of each case from its log, as JSON lines.
"""
import json, math, os, re, subprocess, sys
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

BANDWIDTH = {1e-3: 9.5392, 1e-4: 12.024, 1e-5: 14.471}


def density(c, continuity, terms):
    """chi on [0, 1] as a Legendre series, int_0^1 chi = 1."""
    u, w = legendre.leggauss(128)
    n = np.arange(0, 2 * terms, 2)
    basis = legendre.legvander(u, 2 * terms - 2)[:, ::2] * np.sqrt(2 * n + 1)
    constraints = np.array([np.ones(terms), n * (n + 1) / 2,
                            (n - 1) * n * (n + 1) * (n + 2) / 8])
    z = np.linalg.svd(constraints[:continuity] * np.sqrt(2 * n + 1),
                      full_matrices=True)[2][continuity:].T
    projected = (w[:, None] * basis) @ z
    kernel = np.sinc(c * (u[:, None] - u) / np.pi)
    _, vectors = np.linalg.eigh(projected.T @ kernel @ projected)
    coefficients = np.zeros(2 * terms - 1)
    coefficients[::2] = (z @ vectors[:, -1]) * np.sqrt(2 * n + 1)
    chi = legendre.Legendre(coefficients)
    return chi / (chi.integ()(1.0) - chi.integ()(0.0))


def functional(c, grid, edges, cutoff, continuity=2, terms=8, order=4):
    chi = density(c, continuity, terms)
    t, w = legendre.leggauss(600)
    t, w = 0.5 * (t + 1), 0.5 * w          # on [0, 1]
    weights = w * chi(t)

    def spectrum(k):                        # chi_hat(k) = int_0^1 chi cos(k t)
        flat = k.ravel()
        out = np.empty_like(flat)
        for s in range(0, flat.size, 4096):
            out[s:s + 4096] = np.cos(np.outer(flat[s:s + 4096], t)) @ weights
        return out.reshape(k.shape)

    n = np.meshgrid(*[np.fft.fftfreq(K) * K for K in grid], indexing='ij')
    def wave2(j):
        return sum(((n[a] + j[a] * grid[a]) / edges[a]) ** 2 for a in range(3))
    base2 = wave2((0, 0, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(base2 > 0, spectrum(2 * np.pi * cutoff * np.sqrt(base2)) / base2, 0.0)
    images = [j for j in np.ndindex(3, 3, 3)]
    images = [tuple(x - 1 for x in j) for j in images]
    rho2, k2, exact_g = {}, {}, {}
    for j in images:
        r = np.ones_like(base2)
        for a in range(3):
            if j[a]:
                image = n[a] + j[a] * grid[a]
                r = r * (n[a] / image) ** (2 * order)
        rho2[j], k2[j] = r, wave2(j)
        if j != (0, 0, 0):
            exact_g[j] = spectrum(2 * np.pi * cutoff * np.sqrt(k2[j])) / k2[j]
    error = np.zeros_like(base2)
    exact = g * g * base2
    for j in images:
        if j != (0, 0, 0):
            error += (rho2[j] * g - exact_g[j]) ** 2 * k2[j]
            exact += exact_g[j] ** 2 * k2[j]
        for l in images:
            if l != j:
                error += g * g * rho2[j] * rho2[l] * k2[l]
    return math.sqrt(error.sum() / exact.sum())


def uncorrelated(count, edge, cutoff, bandwidths, work):
    import mdir
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mdir_physical_constants import COULOMB_NATIVE
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(985)
    q = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    gro = work / 'ions.gro'
    gro.write_text(f'ions\n{count}\n' + ''.join(
        f'{(i + 1) % 100000:5d}ION     X{(i + 1) % 100000:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n'
        for i, p in enumerate(rng.uniform(0, edge, size=(count, 3)))) + f'{edge:10.5f}' * 3 + '\n')
    x = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])]
                  for l in gro.read_text().splitlines()[2:2 + count]])
    top = work / 'ions.top'
    top.write_text('[ defaults ]\n1 2 no 1 1\n[ atomtypes ]\nX 1 1.0 0 A 0.3 0\n'
                   '[ moleculetype ]\nions 0\n[ atoms ]\n' +
                   ''.join(f'{i} X 1 ION X {i} {c} 1.0\n' for i, c in enumerate(q, 1)) +
                   '[ system ]\nions\n[ molecules ]\nions 1\n')
    loaded = mdir.load_gromacs(str(top), str(gro))
    system, state = loaded.make_system(), loaded.make_state()
    system.electrostatics = mdir.Electrostatics.PME
    system.cutoff, system.pairlist_distance = cutoff, cutoff + 0.1
    system.pme_kernel = mdir.PMEKernel.Prolate
    system.pme_continuity, system.pme_basis_terms = 2, 8
    state.velocities = np.zeros_like(x)

    def forces(c, k):
        system.pme_bandwidth, system.pme_grid = c, [k] * 3
        execution = mdir.Execution()
        execution.target, execution.precision = mdir.Target.CPU, mdir.Precision.Double
        integrator = mdir.Integrator()
        integrator.method = mdir.IntegratorMethod.VelocityVerlet
        integrator.timestep = 1e-12
        simulation = mdir.Simulation(mdir.compile(system, state, integrator, mdir.Ensemble(),
                                                  execution, mdir.Schedule()))
        simulation.run(1)
        return np.asarray(simulation.state().forces, dtype=float)

    for c in bandwidths:
        chi = density(c, 2, 8)
        phi = chi.integ(lbnd=0)
        direct = np.zeros((count, 3))
        for i in range(count - 1):
            d = x[i] - x[i + 1:]
            d -= edge * np.rint(d / edge)
            r = np.linalg.norm(d, axis=1)
            inside = r < cutoff
            d, r, t = d[inside], r[inside], r[inside] / cutoff
            pair = (q[i] * q[i + 1:][inside] *
                    ((1 - phi(t)) / r ** 3 + chi(t) / (cutoff * r ** 2)))[:, None] * d
            direct[i] += pair.sum(0)
            np.subtract.at(direct, np.arange(i + 1, count)[inside], pair)
        start = math.ceil(edge / (math.pi * cutoff / c) - 1e-9)
        fine = forces(c, 6 * start + 6 * start % 2)
        reciprocal = fine - COULOMB_NATIVE * direct
        for scale in (1.0, 1.5, 2.0, 3.0):
            k = math.ceil(start * scale)
            k += k % 2
            error = np.sqrt(np.mean((forces(c, k) - fine) ** 2) / np.mean(reciprocal ** 2))
            estimate = functional(c, [k] * 3, [10 * edge] * 3, 10 * cutoff)
            print(f'c {c} grid {k}^3: error {error:.4e}, estimate {estimate:.4e}, '
                  f'ratio {error / estimate:.3f}', flush=True)


def smooth(n):
    while True:
        if n % 2 == 0:
            r = n
            for f in (2, 3, 5, 7):
                while r % f == 0:
                    r //= f
            if r == 1:
                return n
        n += 1


def control(topology, coordinates, cutoff, pme, target, precision):
    return f'''[input]
format      = "AMBER"
topology    = "{topology}"
coordinates = "{coordinates}"

[output]
energy_interval     = 1
checkpoint          = "f.h5"
checkpoint_interval = 1

[energy]
cutoff            = {cutoff}
pairlist_distance = {cutoff + 2}
electrostatics    = "PME"

[pme]
''' + '\n'.join(pme) + f'''

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 1e-12
steps      = 1

[ensemble]
ensemble = "NVE"

[constraints]
hydrogen_bonds = false
rigid_water    = false

[boundary]
type = "PERIODIC"

[execution]
target    = "{target}"
precision = "{precision}"
'''


def make(d, name, prmtop, inpcrd, cutoff, modes):
    import parmed
    root = Path(d) / name
    root.mkdir(parents=True, exist_ok=True)
    # The force constants and the Lennard-Jones coefficients set to zero in
    # the tables of the file, which are written as they are, so that the
    # charges, the exclusions, and the 1-4 scaling stay.
    parm = parmed.amber.AmberFormat(prmtop)
    for flag in ('BOND_FORCE_CONSTANT', 'ANGLE_FORCE_CONSTANT',
                 'DIHEDRAL_FORCE_CONSTANT', 'LENNARD_JONES_ACOEF',
                 'LENNARD_JONES_BCOEF', 'LENNARD_JONES_14_ACOEF',
                 'LENNARD_JONES_14_BCOEF', 'UREY_BRADLEY_FORCE_CONSTANT',
                 'CHARMM_IMPROPER_FORCE_CONSTANT', 'HBOND_ACOEF', 'HBOND_BCOEF'):
        if flag in parm.parm_data:
            parm.parm_data[flag] = [0.0] * len(parm.parm_data[flag])
    assert not any(f.startswith('CMAP') for f in parm.parm_data), 'CMAP left'
    parm.write_parm(str(root / 'charges.prmtop'))
    top = (root / 'charges.prmtop').resolve()
    crd = Path(inpcrd).resolve()
    edges = [float(v) for v in open(inpcrd).read().splitlines()[-1].split()[:3]]
    cases = {'reference': (['tolerance = 1e-10', 'order = 8', 'max_spacing = 0.5'], 'GPU', 'DOUBLE')}
    prolate = ['kernel = "PROLATE"', 'continuity = 2', 'basis_terms = 8', 'window = "BSPLINE"', 'order = 4']
    for eps, c in BANDWIDTH.items():
        base = [math.ceil(L / (math.pi * cutoff / c) - 1e-9) for L in edges]
        grids = []
        for s_ in (1, 1.25, 1.5, 1.75, 2, 2.5, 3):
            g = tuple(smooth(math.ceil(b * s_ - 1e-9)) for b in base)
            if g not in grids:
                grids.append(g)
        for mode in modes:
            target, precision = mode.split(':')
            cases[f'{eps:g}-{mode}-auto'.replace(':', '-')] = (prolate + [f'tolerance = {eps:g}'], target, precision)
            for g in grids:
                cases[f'{eps:g}-{mode}-{g[0]}x{g[1]}x{g[2]}'.replace(':', '-')] = (
                    prolate + [f'tolerance = {eps:g}', f'grid = [{g[0]}, {g[1]}, {g[2]}]'], target, precision)
    for case, (pme, target, precision) in cases.items():
        (root / case).mkdir(exist_ok=True)
        (root / case / 'control.toml').write_text(control(top, crd, cutoff, pme, target, precision))


def forces(mdir, path):
    out = subprocess.run([mdir, 'checkpoint', str(path), '--print=forces'],
                         capture_output=True, text=True, check=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip()]
    return np.array([[float(v) for v in r[2:5]] for r in rows])


def main():
    mode = sys.argv[1]
    if mode == 'functional':
        a = sys.argv[2:]
        c, grid, edges, rc = float(a[0]), [int(v) for v in a[1:4]], [float(v) for v in a[4:7]], float(a[7])
        extra = [int(v) for v in a[8:10]]
        print(f'{functional(c, grid, edges, rc, *extra):.12e}')
    elif mode == 'uncorrelated':
        count, edge, cutoff = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        uncorrelated(count, edge, cutoff, [float(v) for v in sys.argv[5:]],
                     Path(os.environ.get('TMPDIR', '/tmp')) / f'spline-grid-{os.getpid()}')
    elif mode == 'make':
        make(*sys.argv[2:6], float(sys.argv[6]), sys.argv[7:] or ['GPU:MIXED', 'GPU:DOUBLE'])
    elif mode == 'run':
        mdir = sys.argv[2]
        for d in sys.argv[3:]:
            for case in sorted(Path(d).iterdir()):
                if not (case / 'control.toml').exists() or (case / 'f.h5').exists():
                    continue
                with open(case / 'run.log', 'w') as log:
                    p = subprocess.run([mdir, 'run', 'control.toml'], cwd=case, stdout=log,
                                       stderr=subprocess.STDOUT)
                print(case.name, 'exit', p.returncode, flush=True)
    elif mode == 'report':
        mdir = sys.argv[2]
        for d in sys.argv[3:]:
            ref = forces(mdir, Path(d) / 'reference' / 'f.h5')
            for case in sorted(Path(d).iterdir()):
                if case.name == 'reference' or not (case / 'f.h5').exists():
                    continue
                f = forces(mdir, case / 'f.h5')
                row = {'case': f'{Path(d).name}/{case.name}',
                       'delta': float(np.linalg.norm(f - ref) / np.linalg.norm(ref))}
                m = re.search(r'a grid of (\d+) x (\d+) x (\d+) \((\w[\w ]*)\), the mesh error '
                              r'estimate ([0-9.e+-]+), the bound of the automatic grid ([0-9.e+-]+)',
                              (case / 'run.log').read_text())
                if m:
                    row.update(grid=[int(m[1]), int(m[2]), int(m[3])], source=m[4],
                               estimate=float(m[5]), bound=float(m[6]))
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
