#!/usr/bin/env python3
"""Independent Cartesian/async LJ checks; imports only the Python oracle."""
import argparse
import importlib.util
import itertools
import math
import os
from pathlib import Path
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location('oracle', Path(__file__).with_name('cpu-hybrid-lj.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    # Every octant, across internal faces/edges/corners and all periodic seams;
    # add a genuine interior interacting pair so async does useful local work.
    points = list(itertools.product((0.6, 5.4, 6.6, 11.4), repeat=3))
    points += [(3., 3., 3.), (4.2, 3., 3.)]
    cases = [('cube', (12.,)*3, points),
             ('empty-ranks', (12.,)*3, [(5.4,5.4,5.4),(6.6,6.6,6.6)]),
             ('empty', (12.,)*3, []),
             ('elongated', (48.,12.,12.), [(4*x,y,z) for x,y,z in points])]
    grids = [(1,'1,1,1'), (2,'1,2,1'), (4,'1,2,2'), (8,'2,2,2'),
             (8,'auto'), (5,'5,1,1')]
    if args.smoke:
        cases = cases[:2]
        grids = [(8,'2,2,2')]
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    env.pop('DISPLAY', None)
    tested = 0
    maxima = {'double': 0., 'mixed': 0.}
    with tempfile.TemporaryDirectory(prefix='mdir-cartesian-') as temporary:
        path = Path(temporary)/'input.txt'
        for name, box, points in cases:
            ids = [101+7*i for i in range(len(points))]
            path.write_text(f'{len(points)} {box[0]} {box[1]} {box[2]} 2.5 1 1\n'+''.join(
                f'{gid} {x} {y} {z}\n' for gid,(x,y,z) in zip(ids,points)))
            energy, force, virial = module.oracle(points, box, 2.5)
            reference = [energy]+virial+sum(force, [])
            for (ranks,grid), precision in itertools.product(grids, ('double','mixed')):
                outputs = []
                for halo in ('sync','async'):
                    command = ['mpiexec','--oversubscribe','-n',str(ranks),args.executable,
                               str(path),f'--grid={grid}',f'--halo={halo}',f'--precision={precision}',
                               '--threads=2','--simd-width=4',
                               '--repeat=3' if halo=='async' else '--repeat=1']
                    run = subprocess.run(command, text=True, capture_output=True, env=env, timeout=120)
                    if run.returncode:
                        raise RuntimeError(f'{command}\n{run.stdout}\n{run.stderr}')
                    rows = [line.split() for line in run.stdout.splitlines()]
                    forces = {int(r[1]): list(map(float,r[2:])) for r in rows if r[0]=='force'}
                    assert set(forces)==set(ids)
                    got = [float(next(r[1] for r in rows if r[0]=='energy'))]
                    got += list(map(float,next(r[1:] for r in rows if r[0]=='virial')))
                    got += sum((forces[gid] for gid in ids), [])
                    tol = 2e-11 if precision=='double' else 3e-5
                    for a,b in zip(got,reference):
                        assert math.isfinite(a) and abs(a-b)<=tol*(1+abs(b)), (command,a,b)
                    error = max(map(lambda ab: abs(ab[0]-ab[1]), zip(got,reference)), default=0.)
                    maxima[precision] = max(maxima[precision],error)
                    outputs.append(got)
                    tested += 1
                    print(name, grid, precision, halo, f'energy_ref={energy:.17g}', f'max_abs_error={error:.3g}', flush=True)
                assert outputs[0] == outputs[1], ('sync/async mismatch',name,grid,precision)
        for option, diagnostic in [('--grid=2,2,2','grid product'),
                                   ('--grid=0,1,2','invalid grid'),
                                   ('--grid=1,2','three dimensions'),
                                   ('--repeat=0','usage:'),
                                   ('--halo=unknown','usage:')]:
            run = subprocess.run(['mpiexec','--oversubscribe','-n','2',args.executable,
                                  str(path),option], text=True, capture_output=True,
                                 env=env, timeout=30)
            assert run.returncode != 0 and diagnostic in run.stderr, run.stderr
        print(f'PASS: {tested} Cartesian configurations; max errors {maxima}')


if __name__ == '__main__':
    main()
