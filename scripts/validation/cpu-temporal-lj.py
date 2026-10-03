#!/usr/bin/env python3
"""Independent velocity-Verlet oracle for the experimental Cartesian driver.

This exercises reduced-unit LJ only, not the production simulation workflow.
"""
import argparse
import importlib.util
import itertools
import math
import os
from pathlib import Path
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location('oracle', Path(__file__).with_name('cpu-hybrid-lj.py'))
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def evolve(points, velocities, masses, dt, steps):
    x = [list(p) for p in points]
    v = [list(p) for p in velocities]
    energy, f, virial = oracle.oracle(x, (12.,)*3, 2.5)
    initial = energy + sum(m*sum(a*a for a in row)/2 for m,row in zip(masses,v))
    for _ in range(steps):
        for i,k in itertools.product(range(len(x)), range(3)):
            v[i][k] += dt*f[i][k]/(2*masses[i])
            x[i][k] = (x[i][k]+dt*v[i][k]) % 12
        energy, f, virial = oracle.oracle(x, (12.,)*3, 2.5)
        for i,k in itertools.product(range(len(x)), range(3)):
            v[i][k] += dt*f[i][k]/(2*masses[i])
    final = energy + sum(m*sum(a*a for a in row)/2 for m,row in zip(masses,v))
    return x,v,energy,f,virial,initial,final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    # Translation crosses an internal corner, then a periodic corner. Unequal
    # masses expose lost/reordered state fields on an empty-to-populated rank.
    cases = [
        ('corner', [(5.99,5.99,5.99),(7.19,5.99,5.99)], [(2.,2.,2.)]*2, .001, 16, .4),
        ('face', [(5.99,3.,3.),(7.19,3.,3.)], [(2.,0.,0.)]*2, .001,16,.4),
        ('edge', [(5.99,5.99,3.),(7.19,5.99,3.)], [(2.,2.,0.)]*2, .001,16,.4),
        ('enter-cutoff', [(4.5,3.,3.),(7.5,3.,3.)], [(1.,0.,0.),(-1.,0.,0.)], .01,32,.2),
        ('periodic', [(11.99,11.99,11.99),(1.19,11.99,11.99)], [(2.,2.,2.)]*2, .001,16,.4),
        ('reuse', [(5.4,3.,3.),(6.6,3.,3.)], [(0.,.03,0.),(0.,-.03,0.)], .001,32,.4),
        ('rebuild', [(5.4,3.,3.),(6.6,3.,3.)], [(0.,.03,0.),(0.,-.03,0.)], .001,32,.0001),
        ('multiwrap', [(1.,2.,3.)], [(2500.,-2500.,2500.)], .01,2,.4),
        ('empty', [], [], .001,2,.4),
    ]
    grids = [(1,'1,1,1'),(2,'2,1,1'),(8,'2,2,2')]
    if args.smoke:
        grids = [(8,'2,2,2')]
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    env.pop('DISPLAY', None)
    maxima = {'double':0., 'mixed':0.}
    tested = 0
    with tempfile.TemporaryDirectory(prefix='mdir-temporal-') as temp:
        snapshot, state = Path(temp)/'snapshot', Path(temp)/'state'
        for name,points,velocities,dt,steps,skin in cases:
            ids = [101+7*i for i in range(len(points))]
            masses = [1.+i*.7 for i in range(len(points))]
            snapshot.write_text(f'{len(points)} 12 12 12 2.5 1 1\n'+''.join(
                f'{gid} {x} {y} {z}\n' for gid,(x,y,z) in zip(ids,points)))
            state.write_text(''.join(f'{gid} {m} {vx} {vy} {vz}\n' for gid,m,(vx,vy,vz) in
                                     reversed(list(zip(ids,masses,velocities)))))
            x,v,e,f,w,e0,e1 = evolve(points,velocities,masses,dt,steps)
            for (ranks,grid),precision,halo in itertools.product(grids,('double','mixed'),('sync','async')):
                command = ['mpiexec','--oversubscribe','-n',str(ranks),args.executable,str(snapshot),
                           f'--state={state}',f'--dt={dt}',f'--steps={steps}',f'--skin={skin}',
                           f'--grid={grid}',f'--precision={precision}',f'--halo={halo}',
                           '--threads=2','--simd-width=4']
                run = subprocess.run(command,text=True,capture_output=True,env=env,timeout=120)
                assert run.returncode==0, (command,run.stdout,run.stderr)
                rows = [r.split() for r in run.stdout.splitlines()]
                gotstates = {int(r[1]):list(map(float,r[2:])) for r in rows if r[0]=='state'}
                gotforces = {int(r[1]):list(map(float,r[2:])) for r in rows if r[0]=='force'}
                assert set(gotstates)==set(ids)==set(gotforces)
                got = [float(next(r[1] for r in rows if r[0]=='energy'))]
                ref = [e]
                got += list(map(float,next(r[1:] for r in rows if r[0]=='virial')))
                ref += w
                for i,gid in enumerate(ids):
                    got += gotstates[gid]+gotforces[gid]
                    ref += [masses[i]]+x[i]+v[i]+f[i]
                tol = 3e-10 if precision=='double' else 5e-5
                for a,b in zip(got,ref):
                    assert math.isfinite(a) and abs(a-b)<=tol*(1+abs(b)), (name,command,a,b)
                diagnostics = [r.split() for r in run.stderr.splitlines() if r.startswith('step ')]
                assert len(diagnostics)==steps+1
                last = dict(zip(diagnostics[-1][::2], diagnostics[-1][1::2]))
                if name=='reuse': assert int(last['rebuilds'])==1, last
                if name in ('rebuild','multiwrap'): assert int(last['rebuilds'])>1, last
                if name in ('corner','face','edge','periodic') and ranks>1: assert int(last['migrations'])>0, last
                if name in ('reuse','rebuild'):
                    assert abs(float(last['total'])-e0)<2e-7, (name,e0,last)
                error = max((abs(a-b) for a,b in zip(got,ref)),default=0.)
                maxima[precision] = max(maxima[precision],error)
                tested += 1
                print(name,grid,precision,halo,f'energy_ref={e:.17g}',f'max_error={error:.4g}',
                      f'rebuilds={last["rebuilds"]}',f'migrations={last["migrations"]}',flush=True)
        print('PASS',tested,'trajectory configurations',maxima)


if __name__ == '__main__':
    main()
