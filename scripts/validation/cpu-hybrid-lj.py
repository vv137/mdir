#!/usr/bin/env python3
"""Independent unique-pair LJ oracle for the fixed-layout CPU prototype.

No NumPy or production MDIR force code is used. The oracle searches periodic
images explicitly instead of using the generated kernel's rounding operation.
"""
import argparse
import itertools
import math
import os
from pathlib import Path
import subprocess
import tempfile


def oracle(points, box, cutoff, sigma=1.0, epsilon=1.0):
    energy = 0.0
    forces = [[0.0] * 3 for _ in points]
    virial = [0.0] * 9
    for i, j in itertools.combinations(range(len(points)), 2):
        images = ([points[i][k] - points[j][k] + s * box[k]
                   for s in (-1, 0, 1)] for k in range(3))
        d = min(itertools.product(*images), key=lambda v: sum(x*x for x in v))
        r2 = sum(x*x for x in d)
        if r2 >= cutoff**2:
            continue
        r = math.sqrt(r2)
        energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
        force = [24 * epsilon * (2*sigma**12*r**-14 - sigma**6*r**-8) * v for v in d]
        for a in range(3):
            forces[i][a] += force[a]
            forces[j][a] -= force[a]
            for b in range(3):
                virial[3*a+b] += d[a]*force[b]
    return energy, forces, virial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('executable')
    ap.add_argument('--mpiexec', default='mpiexec')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--case', help='Run one named fixture')
    args = ap.parse_args()
    coords = [[0.3,1,1], [1.45,1.2,1], [2.7,1,1.3], [4.1,1.1,1],
              [6.6,2,1], [7.85,2.2,1.2], [9.3,1.8,1.1], [11.15,1.1,1.1]]
    ids = [101,9,55,402,18,230,4,77]
    cases = [('periodic', coords, ids, 2.95),
             ('empty-ranks', [coords[0],coords[-1]], [101,77], 2.95),
             ('empty-system', [], [], 2.95),
             ('reordered', coords[::-1], ids[::-1], 2.95),
             ('nonunit-parameters', coords, ids, 2.95),
             ('multi-slab', [[2.3,1,1], [4.85,1,1]], [901,902], 2.95)]
    if args.smoke:
        cases = [cases[0], cases[1], cases[-1]]
    if args.case:
        cases = [case for case in cases if case[0] == args.case]
        if not cases:
            ap.error('unknown fixture')
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    env.pop('DISPLAY', None)
    tested = 0
    reproducible = {}
    with tempfile.TemporaryDirectory(prefix='mdir-cpu-lj-') as directory:
        path = Path(directory)/'snapshot.txt'
        for name, points, gids, rc in cases:
            sigma, epsilon = ((0.8, 1.3) if name=='nonunit-parameters' else (1.0, 1.0))
            energy, force, virial = oracle(points, [12]*3, rc, sigma, epsilon)
            path.write_text(f'{len(points)} 12 12 12 {rc} {sigma} {epsilon}\n' + ''.join(
                f'{gid} {p[0]} {p[1]} {p[2]}\n' for gid,p in zip(gids,points)))
            for precision, ranks, threads, width in itertools.product(
                    ('double','mixed'), (((5,) if name=='multi-slab' else (2,)) if args.smoke else (1,2,5)),
                    ((2,) if args.smoke else (1,2)), ((4,) if args.smoke else (1,4,8))):
                command = [args.mpiexec, '--oversubscribe', '-n', str(ranks),
                           args.executable, str(path), f'--precision={precision}',
                           f'--threads={threads}', f'--simd-width={width}']
                run = subprocess.run(command, text=True, capture_output=True,
                                     env=env, timeout=90)
                if run.returncode:
                    raise RuntimeError(f'{command}\n{run.stdout}\n{run.stderr}')
                lines = [line.split() for line in run.stdout.splitlines()]
                got_e = float(next(row[1] for row in lines if row[0]=='energy'))
                got_v = list(map(float,next(row[1:] for row in lines if row[0]=='virial')))
                got_f = {int(row[1]):list(map(float,row[2:])) for row in lines if row[0]=='force'}
                assert set(got_f)==set(gids)
                got = [got_e]+got_v+sum((got_f[gid] for gid in gids), [])
                ref = [energy]+virial+sum(force, [])
                atol, rtol = ((2e-11,2e-11) if precision=='double' else (3e-5,3e-5))
                for a,b in zip(got,ref):
                    assert math.isfinite(a) and abs(a-b)<=atol+rtol*abs(b), (command,a,b)
                key = (name, precision, ranks, width)
                if key in reproducible:
                    assert got == reproducible[key], ('thread-count mismatch', command)
                reproducible[key] = got
                force_errors = [a-b for a,b in zip(got[10:],ref[10:])]
                force_max = max(map(abs,force_errors), default=0)
                force_rms = math.sqrt(sum(v*v for v in force_errors)/max(len(force_errors),1))
                virial_max = max(abs(a-b) for a,b in zip(got_v,virial))
                maximum = max((abs(a-b) for a,b in zip(got,ref)),default=0)
                print(f'{name} {precision} ranks={ranks} threads={threads} width={width} '
                      f'energy_ref={energy:.17g} energy_error={got_e-energy:.3g} max_abs_error={maximum:.3g} '
                      f'force_max={force_max:.3g} force_rms={force_rms:.3g} virial_max={virial_max:.3g} '
                      f'atol={atol} rtol={rtol}', flush=True)
                tested += 1
        rejected = [
            ('2 12 12 12 2.5 1 1\n7 1 1 1\n7 2 1 1\n', 'duplicate atom ID'),
            ('2 12 12 12 2.5 1 1\n7 1 1 1\n8 1 1 1\n', 'coincident particles'),
            ('0 12 12 12 6 1 1\n', 'cutoff < half'),
            ('1 12 12 12 2.5 1 1\n7 -1 1 1\n', 'canonical and finite'),
        ]
        for text, diagnostic in rejected:
            path.write_text(text)
            run = subprocess.run([args.mpiexec, '--oversubscribe', '-n', '2',
                                  args.executable, str(path)], text=True,
                                 capture_output=True, env=env, timeout=30)
            assert run.returncode != 0 and diagnostic in run.stderr, run.stderr
        # Energy finite differences validate the independent force convention.
        _, force, _ = oracle(coords,[12]*3,2.95)
        step = 1e-6
        for i in range(len(coords)):
            for axis in range(3):
                plus = [p[:] for p in coords]; minus = [p[:] for p in coords]
                plus[i][axis] += step; minus[i][axis] -= step
                derivative = -(oracle(plus,[12]*3,2.95)[0]-oracle(minus,[12]*3,2.95)[0])/(2*step)
                assert abs(derivative-force[i][axis]) < 2e-7
        print(f'PASS: {tested} MPI configurations and 24 finite differences; 4 invalid snapshots rejected')


if __name__ == '__main__':
    main()
