"""The force error of the prolate kernel with its direct sum as a table of
r² against the cells of D266 (D280, issue #380).

Delta is that of grid_sweep.py and docs/pme-m1.md Section 7: the RMS of the
difference of the forces from those of the Gaussian at erfc 1e-13, 0.33 A,
order 8 (CPU, double), over the RMS of the Coulomb forces of that reference
(the reference less the same system with every charge zero), one step of a
tiny time step from the coordinates, at a cutoff of 9 A. Each tier
(epsilon = 1e-3, 1e-4, 1e-5, P = 6, the automatic grid) runs on the CPU
and a GPU, in mixed and double precision, with the series of r² (the
default) and with the cells (MDIR_PROLATE_CELLS=1).

    pair_delta.py make DIR TOPOLOGY COORDINATES
    pair_delta.py run MDIR DIR [JOBS]       CPU cases JOBS at once, GPU ones in turn
    pair_delta.py report MDIR DIR [OUT.jsonl]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_sweep import REFERENCES, control, forces  # noqa: E402

CUTOFF = 9.0
EPSILONS = [1e-3, 1e-4, 1e-5]


def write(d, case, topology, coordinates, pme, target='CPU', precision='DOUBLE', cells=False):
    os.makedirs(f'{d}/{case}', exist_ok=True)
    text = control(topology, coordinates, CUTOFF, pme)
    text = text.replace('target    = "CPU"', f'target    = "{target}"')
    text = text.replace('precision = "DOUBLE"', f'precision = "{precision}"')
    open(f'{d}/{case}/run.toml', 'w').write(text)
    open(f'{d}/{case}/env', 'w').write('MDIR_PROLATE_CELLS=1\n' if cells else '')


def main():
    mode = sys.argv[1]
    if mode == 'make':
        d, top, crd = sys.argv[2], os.path.abspath(sys.argv[3]), os.path.abspath(sys.argv[4])
        os.makedirs(d, exist_ok=True)
        import parmed
        neutral = parmed.load_file(top)
        for atom in neutral.atoms:
            atom.charge = 0.0
        neutral.save(f'{d}/uncharged.prmtop', overwrite=True)
        write(d, 'reference', top, crd, REFERENCES['reference'])
        write(d, 'uncharged', os.path.abspath(f'{d}/uncharged.prmtop'), crd, ['tolerance = 1e-5'])
        for eps in EPSILONS:
            pme = ['kernel = "PROLATE"', f'tolerance = {eps:g}', 'order = 6']
            for target in ('CPU', 'GPU'):
                for precision in ('MIXED', 'DOUBLE'):
                    for cells in (False, True):
                        case = f'{eps:g}-{target}-{precision}-{"cells" if cells else "series"}'
                        write(d, case, top, crd, pme, target, precision, cells)
    elif mode == 'run':
        mdir, d = sys.argv[2], sys.argv[3]
        jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        todo = sorted(c for c in os.listdir(d)
                      if os.path.isdir(f'{d}/{c}') and not os.path.exists(f'{d}/{c}/run.h5'))

        def one(case):
            env = dict(os.environ, OMP_NUM_THREADS='4')
            for line in open(f'{d}/{case}/env'):
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    env[k] = v
            p = subprocess.run([mdir, 'run', 'run.toml'], cwd=f'{d}/{case}',
                               capture_output=True, text=True, env=env)
            open(f'{d}/{case}/run.log', 'w').write(p.stdout + p.stderr)
            print(case, 'exit', p.returncode, flush=True)
        with ThreadPoolExecutor(jobs) as pool:
            list(pool.map(one, [c for c in todo if '-GPU-' not in c]))
        for case in todo:
            if '-GPU-' in case:
                one(case)
    elif mode == 'report':
        mdir, d = sys.argv[2], sys.argv[3]
        out = open(sys.argv[4], 'w') if len(sys.argv) > 4 else None
        ref = forces(mdir, f'{d}/reference/run.h5')
        coul = ref - forces(mdir, f'{d}/uncharged/run.h5')
        norm = np.sqrt(np.mean(np.sum(coul ** 2, 1)))
        print(f'rms Coulomb force of the reference {norm:.3f}')
        rows = {}
        for case in sorted(os.listdir(d)):
            if case in ('reference', 'uncharged') or not os.path.exists(f'{d}/{case}/run.h5'):
                continue
            rows[case] = forces(mdir, f'{d}/{case}/run.h5')
        for case, f in rows.items():
            delta = float(np.sqrt(np.mean(np.sum((f - ref) ** 2, 1))) / norm)
            row = {'case': case, 'delta': delta}
            if case.endswith('-series'):
                # the series against the cells of the same case
                other = rows.get(case[:-len('series')] + 'cells')
                if other is not None:
                    row['series_vs_cells'] = float(
                        np.sqrt(np.mean(np.sum((f - other) ** 2, 1))) / norm)
            print(json.dumps(row), flush=True)
            if out:
                out.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
