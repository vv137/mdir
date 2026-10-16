"""Dynamics of JAC with the prolate kernel against the Gaussian kernel
(D[prolate-ewald], issue #334): the drift of the total energy in NVE and the
mean volume in NPT, mixed precision on a GPU.

dynamics.py make DIR TOPOLOGY COORDINATES NVE_STEPS NPT_STEPS   control files
dynamics.py run MDIR DIR                                      on the GPU, in turn
dynamics.py report DIR                                        the table

The drift is the slope of a straight line fitted to the total energy of the
rows of the energy file, in kcal/mol/ns and per particle, and the slopes of
its two halves, which show how much a slope varies by itself; the
volume is the mean of the second half of the NPT run, with the standard
error of the mean of 10 blocks.
"""
import os, re, subprocess, sys

import numpy as np

CASES = {
    'gaussian-default': [],
    'gaussian-suite': ['beta = 0.39467022824439113', 'grid = [64, 64, 64]', 'order = 4',
                       'influence = "OPTIMAL"'],
    'prolate-1e-5': ['kernel = "PROLATE"'],
    'prolate-1e-3': ['kernel = "PROLATE"', 'tolerance = 1e-3', 'order = 5'],
    # The force of the direct sum jumps at the cutoff by about the tolerance
    # with either kernel: the Gaussian at the same tolerances, for the drift.
    'gaussian-1e-3': ['tolerance = 1e-3'],
    'gaussian-1e-4': ['tolerance = 1e-4'],
    'prolate-1e-4': ['kernel = "PROLATE"', 'tolerance = 1e-4'],
    'prolate-1e-5-double': ['kernel = "PROLATE"'],
    'gaussian-default-double': [],
}


NVE_ONLY = ('gaussian-1e-3', 'gaussian-1e-4', 'prolate-1e-4', 'prolate-1e-5-double',
            'gaussian-default-double')


def control(top, crd, steps, npt, pme):
    ensemble = ('[ensemble]\nensemble    = "NPT"\ntemperature = 300.0\npressure    = 0.986923\n\n'
                '[thermostat]\nmethod        = "V-RESCALE"\ntime_constant = 1.0\n\n'
                '[barostat]\nmethod        = "C-RESCALE"\ntime_constant = 2.0\n') if npt else \
               '[ensemble]\nensemble    = "NVE"\ntemperature = 300.0\n'
    return f'''[input]
format      = "AMBER"
topology    = "{top}"
coordinates = "{crd}"

[output]
energy          = "energy.txt"
energy_interval = 500

[energy]
cutoff            = 8.0
pairlist_distance = 11.0
electrostatics    = "PME"

[pme]
''' + '\n'.join(pme) + f'''

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {steps}

{ensemble}
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
'''


def table(path):
    lines = [l for l in open(path) if l.strip()]
    header = lines[0].lstrip('#').split()
    rows = np.array([[float(v) for v in l.split()] for l in lines[1:] if not l.startswith('#')])
    return header, rows


mode = sys.argv[1]
if mode == 'make':
    d, top, crd, nve, npt = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
    for case, pme in CASES.items():
        for kind, steps in (('nve', nve), ('npt', npt)):
            if kind == 'npt' and case in NVE_ONLY:
                continue
            os.makedirs(f'{d}/{kind}-{case}', exist_ok=True)
            text = control(top, crd, steps, kind == 'npt', pme)
            if case.endswith('-double'):
                text = text.replace('"MIXED"', '"DOUBLE"')
            open(f'{d}/{kind}-{case}/run.toml', 'w').write(text)
elif mode == 'run':
    mdir, d = sys.argv[2], sys.argv[3]
    fail = 0
    for case in sorted(os.listdir(d)):
        if os.path.exists(f'{d}/{case}/run.log'):
            continue
        p = subprocess.run([mdir, 'run', 'run.toml'], cwd=f'{d}/{case}', capture_output=True, text=True)
        open(f'{d}/{case}/run.log', 'w').write(p.stdout + p.stderr)
        print(case, 'exit', p.returncode, flush=True)
        fail += p.returncode != 0
    sys.exit(1 if fail else 0)
elif mode == 'report':
    d = sys.argv[2]
    for case in sorted(os.listdir(d)):
        header, rows = table(f'{d}/{case}/energy.txt')
        log = open(f'{d}/{case}/run.log').read()
        grid = re.search(r'grid of (\d+ x \d+ x \d+)', log)
        col = {name: k for k, name in enumerate(header)}
        if case.startswith('nve'):
            time = rows[:, col['time']] / 1000.0  # ns
            total = rows[:, col['total']]
            count = int(re.search(r'MDIR: (\d+) particles', log)[1])
            slope = np.polyfit(time, total, 1)[0]
            half = len(time) // 2
            halves = [np.polyfit(time[a:b], total[a:b], 1)[0] for a, b in ((0, half), (half, None))]
            print(f'{case}: {len(rows)} rows, {time[-1]:.2f} ns, drift {slope:.3f} kcal/mol/ns '
                  f'(halves {halves[0]:.2f}, {halves[1]:.2f}), '
                  f'{slope / count:.2e} kcal/mol/ns per particle, fluctuation {np.std(total):.2f} '
                  f'kcal/mol' + (f', grid {grid[1]}' if grid else ''))
        else:
            v = rows[len(rows) // 2:, col['volume']]
            blocks = np.array_split(v, 10)
            sem = np.std([b.mean() for b in blocks], ddof=1) / np.sqrt(10)
            print(f'{case}: {len(rows)} rows, volume {v.mean():.1f} ± {sem:.1f} Å³ (second half, '
                  f'10 blocks), sd {v.std():.1f}' + (f', grid {grid[1]}' if grid else ''))
