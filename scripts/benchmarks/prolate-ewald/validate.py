"""Validation of the prolate kernel in mdir run (D266): the
forces of one step of a tiny time step, from a checkpoint, against a
reference of the Gaussian kernel in double precision on the CPU at a tight
tolerance (erfc 1e-13, 0.33 A, order 8: erfc 1e-11 at 0.5 A was itself off
by 1.9e-6; grid_sweep.py measures Delta over the Coulomb forces alone), as the relative force error Δ of eq. 16 of Liang et al. (2026)
over all forces (the terms other than the Coulomb sum are those of the
reference, so the difference is that of the Coulomb sum, and the
denominator the norm of all forces, as stated); the Coulomb energy
against the reference's; and the bits of two runs in the deterministic
mode.

validate.py make NAME DIR TOPOLOGY COORDINATES CUTOFF   writes the control files
validate.py run MDIR DIR TARGET                         runs those of TARGET (cpu, gpu)
validate.py report MDIR DIR                             prints the table
"""
import os, re, subprocess, sys

import numpy as np

CASES = {
    # name: (target, precision, deterministic, [pme] lines)
    'reference': ('CPU', 'DOUBLE', False, ['tolerance = 1e-13', 'max_spacing = 0.33', 'order = 8']),
    'prolate-cpu': ('CPU', 'DOUBLE', False, ['kernel = "PROLATE"']),
    'prolate-mixed': ('GPU', 'MIXED', False, ['kernel = "PROLATE"']),
    'prolate-double': ('GPU', 'DOUBLE', False, ['kernel = "PROLATE"']),
    'prolate-det-a': ('GPU', 'MIXED', True, ['kernel = "PROLATE"']),
    'prolate-det-b': ('GPU', 'MIXED', True, ['kernel = "PROLATE"']),
    'prolate-1e-3-mixed': ('GPU', 'MIXED', False, ['kernel = "PROLATE"', 'tolerance = 1e-3', 'order = 5']),
    'prolate-1e-4-mixed': ('GPU', 'MIXED', False, ['kernel = "PROLATE"', 'tolerance = 1e-4']),
    'gaussian-mixed': ('GPU', 'MIXED', False, []),
    'gaussian-double': ('CPU', 'DOUBLE', False, []),
}


def control(topology, coordinates, cutoff, target, precision, deterministic, pme):
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
''' + '\n'.join(pme) + f'''

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
target    = "{target}"
precision = "{precision}"
deterministic = {'true' if deterministic else 'false'}
'''


def forces(mdir, path):
    out = subprocess.run([mdir, 'checkpoint', path, '--print=forces'], capture_output=True, text=True, check=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip()]
    return np.array([[float(v) for v in r[2:5]] for r in rows])


def coulomb(log):
    total = 0.0
    for line in open(log):
        m = re.match(r'MDIR:   (Coulomb(?: [a-z]+)?)\s+(-?[0-9.]+)\s*$', line)
        if m and m.group(1) in ('Coulomb', 'Coulomb excluded', 'Coulomb reciprocal', 'Coulomb self', 'Coulomb background'):
            total += float(m.group(2))
    return total


mode = sys.argv[1]
if mode == 'make':
    name, d, top, crd, cutoff = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6])
    for case, (target, precision, det, pme) in CASES.items():
        os.makedirs(f'{d}/{case}', exist_ok=True)
        open(f'{d}/{case}/run.toml', 'w').write(control(top, crd, cutoff, target, precision, det, pme))
elif mode == 'run':
    mdir, d, target = sys.argv[2], sys.argv[3], sys.argv[4].upper()
    fail = 0
    for case, (t, *_rest) in CASES.items():
        if t != target:
            continue
        p = subprocess.run([mdir, 'run', 'run.toml'], cwd=f'{d}/{case}', capture_output=True, text=True)
        open(f'{d}/{case}/run.log', 'w').write(p.stdout + p.stderr)
        print(case, 'exit', p.returncode, flush=True)
        fail += p.returncode != 0
    sys.exit(1 if fail else 0)
elif mode == 'report':
    mdir, d = sys.argv[2], sys.argv[3]
    ref = forces(mdir, f'{d}/reference/run.h5')
    eref = coulomb(f'{d}/reference/run.log')
    norm = np.sqrt(np.mean(np.sum(ref ** 2, 1)))
    print(f'reference: Coulomb {eref:.6f} kcal/mol, rms force {norm:.3f} kJ/mol/nm')
    print('| Case | Δ (all forces) | Coulomb − reference (kcal/mol) | relative |')
    print('|---|---|---|---|')
    for case in CASES:
        if case == 'reference' or not os.path.exists(f'{d}/{case}/run.h5'):
            continue
        f = forces(mdir, f'{d}/{case}/run.h5')
        delta = np.sqrt(np.mean(np.sum((f - ref) ** 2, 1))) / norm
        e = coulomb(f'{d}/{case}/run.log')
        print(f'| {case} | {delta:.2e} | {e - eref:+.6f} | {abs(e - eref) / abs(eref):.1e} |')
    a, b = f'{d}/prolate-det-a/run.h5', f'{d}/prolate-det-b/run.h5'
    if os.path.exists(a) and os.path.exists(b):
        same = subprocess.run([mdir, 'checkpoint', a, b], capture_output=True, text=True)
        print('deterministic, two runs:', same.stdout.strip().splitlines()[-1] if same.stdout.strip() else same.stderr.strip())
