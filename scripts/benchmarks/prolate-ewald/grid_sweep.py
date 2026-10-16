"""The grid of the prolate kernel against the measured force error
(D266, issue #334): mdir run on the CPU in double precision, one
step of a tiny time step from the coordinates, at explicit grids, against a
reference of the Gaussian kernel at a tight tolerance on a fine grid.

Delta is eq. 16 of Liang et al. (2026), the RMS of the difference of the
forces over the RMS of the Coulomb forces of the reference; the Coulomb forces
are those of the reference less those of the same system with every charge
zero (`make` writes that topology with ParmEd), so that the denominator is
that of the paper and the numerator the error of the Coulomb sum.

grid_sweep.py make DIR TOPOLOGY COORDINATES CUTOFF [SUITE]  the references and the sweep;
                                                     SUITE (jac, cel, stmv) adds the suite's [pme]
grid_sweep.py run MDIR DIR [JOBS]                    runs what has no run.h5 yet
grid_sweep.py report MDIR DIR [OUT.jsonl]            Delta, the eq. 10 estimate
"""
import json, math, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# The reference: the Gaussian at erfc 1e-13 on 0.33 A of order 8, checked by
# the prolate kernel at 1e-9 with 8 points, an independent splitting; and the
# reference of validate.py, erfc 1e-11 on 0.5 A, whose own error this shows.
REFERENCES = {
    'reference': ['tolerance = 1e-13', 'max_spacing = 0.33', 'order = 8'],
    'reference-loose': ['tolerance = 1e-11', 'max_spacing = 0.5', 'order = 8'],
    # its automatic grid is refused at 1e-9 with 8 points (the aliasing
    # estimate stays above half the tolerance): 0.15 A gives 420³ on JAC
    'reference-prolate': ['kernel = "PROLATE"', 'tolerance = 1e-9', 'order = 8',
                          'max_spacing = 0.15'],
}
# (tolerance, points of the window); each at the spacings below.
PROLATE = [(1e-3, 5), (1e-3, 6), (1e-4, 5), (1e-4, 6), (1e-4, 7),
           (2e-5, 6), (1e-5, 6), (1e-5, 7), (1e-5, 8), (1e-6, 7), (1e-6, 8)]
# The Gaussian, for matching by the measured Delta: (tolerance, order) at the
# spacings below, MDIR's default, and the suite's settings (pmemd's beta and
# grid, by the name of the system).
GAUSSIAN = [(1e-5, 4), (3e-6, 6), (1e-6, 6), (1e-6, 8)]
GAUSSIAN_SPACINGS = [1.2, 1.1, 1.0, 0.95, 0.9, 0.864, 0.8, 0.75, 0.7, 0.65, 0.6]
SUITE_GRID = {'jac': [64, 64, 64], 'cel': [256, 128, 128], 'stmv': [224, 224, 240]}
SPACINGS = [3.0, 2.8, 2.6, 2.4, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.45,
            1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9]


def smooth(least):
    """getSmoothSize of lib/Driver/Builder.cpp: even and 7-smooth."""
    size = least
    while True:
        if size % 2 == 0:
            rest = size
            for f in (2, 3, 5, 7):
                while rest % f == 0:
                    rest //= f
            if rest == 1:
                return size
        size += 1


def control(topology, coordinates, cutoff, pme):
    return f'''[input]
format      = "AMBER"
topology    = "{topology}"
coordinates = "{coordinates}"

[output]
energy_interval     = 1
checkpoint          = "run.h5"
checkpoint_interval = 1

[energy]
cutoff            = {cutoff}
pairlist_distance = {cutoff + 2}
electrostatics    = "PME"

[pme]
''' + '\n'.join(pme) + '''

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 1e-06
steps      = 1

[ensemble]
ensemble    = "NVE"
temperature = 300.0

[constraints]
hydrogen_bonds = false
rigid_water    = false

[boundary]
type = "PERIODIC"

[execution]
target    = "CPU"
precision = "DOUBLE"
'''


def box(coordinates):
    last = open(coordinates).read().splitlines()[-1].split()
    return [float(v) for v in last[:3]]


def forces(mdir, path):
    out = subprocess.run([mdir, 'checkpoint', path, '--print=forces'],
                         capture_output=True, text=True, check=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip()]
    return np.array([[float(v) for v in r[2:5]] for r in rows])


def main():
    mode = sys.argv[1]
    if mode == 'make':
        d, top, crd, cutoff = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
        os.makedirs(d, exist_ok=True)
        import parmed
        neutral = parmed.load_file(top)
        for atom in neutral.atoms:
            atom.charge = 0.0
        neutral.save(f'{d}/uncharged.prmtop', overwrite=True)
        cases = dict(REFERENCES)
        cases['uncharged'] = ['tolerance = 1e-5']
        cases['gaussian-default'] = []
        if len(sys.argv) > 6:
            cases['gaussian-suite'] = ['beta = 0.39467022824439113', 'order = 4', 'influence = "OPTIMAL"',
                                       'grid = [%d, %d, %d]' % tuple(SUITE_GRID[sys.argv[6]])]
        edges = box(crd)
        for tol, order in GAUSSIAN:
            seen = set()
            for h in GAUSSIAN_SPACINGS:
                grid = tuple(smooth(math.ceil(e / h - 1e-9)) for e in edges)
                if grid in seen:
                    continue
                seen.add(grid)
                cases[f'g{order}-{tol:g}-{grid[0]}x{grid[1]}x{grid[2]}'] = [
                    f'tolerance = {tol:g}', f'order = {order}', f'grid = [{grid[0]}, {grid[1]}, {grid[2]}]']
        for eps, points in PROLATE:
            seen = set()
            for h in SPACINGS:
                grid = tuple(smooth(math.ceil(e / h - 1e-9)) for e in edges)
                if grid in seen or min(grid) < 2 * points:
                    continue
                seen.add(grid)
                cases[f'p{points}-{eps:g}-{grid[0]}x{grid[1]}x{grid[2]}'] = [
                    'kernel = "PROLATE"', f'tolerance = {eps:g}', f'order = {points}',
                    f'grid = [{grid[0]}, {grid[1]}, {grid[2]}]']
            # and the grid of the default rule
            cases[f'p{points}-{eps:g}-auto'] = ['kernel = "PROLATE"', f'tolerance = {eps:g}',
                                                f'order = {points}']
        for case, pme in cases.items():
            os.makedirs(f'{d}/{case}', exist_ok=True)
            t = f'{d}/uncharged.prmtop' if case == 'uncharged' else top
            open(f'{d}/{case}/run.toml', 'w').write(control(t, crd, cutoff, pme))
    elif mode == 'run':
        mdir, d = sys.argv[2], sys.argv[3]
        jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        todo = sorted(c for c in os.listdir(d)
                      if os.path.isdir(f'{d}/{c}') and not os.path.exists(f'{d}/{c}/run.h5'))

        def one(case):
            p = subprocess.run([mdir, 'run', 'run.toml'], cwd=f'{d}/{case}',
                               capture_output=True, text=True,
                               env=dict(os.environ, OMP_NUM_THREADS='4'))
            open(f'{d}/{case}/run.log', 'w').write(p.stdout + p.stderr)
            print(case, 'exit', p.returncode, flush=True)
        with ThreadPoolExecutor(jobs) as pool:
            list(pool.map(one, todo))
    elif mode == 'report':
        mdir, d = sys.argv[2], sys.argv[3]
        out = open(sys.argv[4], 'w') if len(sys.argv) > 4 else None
        ref = forces(mdir, f'{d}/reference/run.h5')
        coul = ref - forces(mdir, f'{d}/uncharged/run.h5')
        norm = np.sqrt(np.mean(np.sum(coul ** 2, 1)))
        print(f'rms Coulomb force of the reference {norm:.3f}')
        for case in sorted(os.listdir(d)):
            if case in ('reference', 'uncharged') or not os.path.exists(f'{d}/{case}/run.h5'):
                continue
            f = forces(mdir, f'{d}/{case}/run.h5')
            delta = float(np.sqrt(np.mean(np.sum((f - ref) ** 2, 1))) / norm)
            log = open(f'{d}/{case}/run.log').read()
            m = re.search(r'c = ([0-9.]+), c1 = ([0-9.]+), (\d+) points of the window, '
                          r'a grid of (\d+) x (\d+) x (\d+), the aliasing estimate ([0-9.e+-]+)', log)
            row = {'case': case, 'delta': delta}
            g = re.search(r'grid = \[(\d+), (\d+), (\d+)\]', open(f'{d}/{case}/run.toml').read())
            if g:
                row['grid'] = [int(g[1]), int(g[2]), int(g[3])]
            if m:
                row.update(c=float(m[1]), c1=float(m[2]), P=int(m[3]),
                           grid=[int(m[4]), int(m[5]), int(m[6])], estimate=float(m[7]))
                tol = re.search(r'tolerance = ([0-9.e+-]+)', open(f'{d}/{case}/run.toml').read())
                row['eps'] = float(tol[1])
            print(json.dumps(row), flush=True)
            if out:
                out.write(json.dumps(row) + '\n')


main()
