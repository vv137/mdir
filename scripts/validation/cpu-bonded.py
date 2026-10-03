#!/usr/bin/env python3
"""Independent harmonic/LJ, finite-difference, and moving-state MPI checks."""
import argparse
import importlib.util
import itertools
import math
import os
from pathlib import Path
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location('oracle', Path(__file__).with_name('cpu-hybrid-lj.py'))
lj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lj)


def evaluate(x, bonds):
    energy,force,virial = lj.oracle(x, (12.,)*3, 2.5)
    for i,j,k,r0 in bonds:
        # Explicit image enumeration, independent of kernel rounding.
        shifts = ([x[i][a]-x[j][a]+12*s for s in (-1,0,1)] for a in range(3))
        d = min(itertools.product(*shifts), key=lambda q:sum(v*v for v in q))
        r = math.sqrt(sum(v*v for v in d))
        energy += .5*k*(r-r0)**2
        f = [-k*(r-r0)*v/r for v in d]
        for a in range(3):
            force[i][a] += f[a]
            force[j][a] -= f[a]
            for b in range(3): virial[3*a+b] += d[a]*f[b]
    return energy,force,virial


def evolve(points, velocity, masses, bonds, steps):
    x = [list(p) for p in points]
    v = [list(p) for p in velocity]
    e,f,w = evaluate(x,bonds)
    for _ in range(steps):
        for i,a in itertools.product(range(len(x)),range(3)):
            v[i][a] += .0005*f[i][a]/masses[i]
            x[i][a] = (x[i][a]+.001*v[i][a]) % 12
        e,f,w = evaluate(x,bonds)
        for i,a in itertools.product(range(len(x)),range(3)):
            v[i][a] += .0005*f[i][a]/masses[i]
    return x,v,e,f,w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('executable')
    ap.add_argument('--smoke',action='store_true')
    args = ap.parse_args()
    cases = [
        ('outside-halo',[(1.,3.,3.),(8.,3.,3.)],[(0,1,.7,4.)],[(0.,0.,0.)]*2),
        ('shared-endpoint',[(5.99,5.99,5.99),(9.49,5.99,5.99),(5.99,9.49,5.99),
                             (5.99,5.99,9.49)],[(0,1,.7,3.),(0,2,.8,3.),(0,3,.9,3.)],[(2.,2.,2.)]*4),
        ('periodic',[(11.99,11.99,11.99),(3.49,11.99,11.99)],[(0,1,.7,3.)],[(2.,2.,2.)]*2),
        ('lj-and-bond',[(5.99,3.,3.),(7.19,3.,3.)],[(0,1,.7,1.)],[(2.,0.,0.)]*2),
        ('empty-topology',[(5.4,3.,3.),(6.6,3.,3.)],[],[(0.,0.,0.)]*2),
        ('empty-system',[],[],[]),
    ]
    grids = [(1,'1,1,1'),(2,'2,1,1'),(8,'2,2,2')]
    if args.smoke:
        cases = [cases[0],cases[1],cases[4]]
        grids = [(8,'2,2,2')]
    maxima = dict(double=0.,mixed=0.)
    tested = fdtests = 0
    env = dict(os.environ,OMP_DYNAMIC='FALSE')
    env.pop('DISPLAY',None)
    with tempfile.TemporaryDirectory(prefix='mdir-bonded-') as temp:
        snapshot,state,bondfile = [Path(temp)/n for n in ('snapshot','state','bonds')]
        for name,points,bonds,velocities in cases:
            ids = [100+7*i for i in range(len(points))]
            masses = [1.+.7*i for i in range(len(points))]
            snapshot.write_text(f'{len(points)} 12 12 12 2.5 1 1\n'+''.join(
                f'{gid} {x} {y} {z}\n' for gid,(x,y,z) in reversed(list(zip(ids,points)))))
            state.write_text(''.join(f'{gid} {m} {vx} {vy} {vz}\n' for gid,m,(vx,vy,vz)
                                    in zip(ids,masses,velocities)))
            # Reverse row order/endpoints to test canonical work assignment.
            bondfile.write_text(''.join(f'{ids[j]} {ids[i]} {k} {r0}\n' for i,j,k,r0 in reversed(bonds)))
            _,ref_force,_ = evaluate(points,bonds)
            for i,a in itertools.product(range(len(points)),range(3)):
                energies=[]
                for sign in (-1,1):
                    x=[list(p) for p in points];x[i][a]+=sign*1e-6
                    energies.append(evaluate(x,bonds)[0])
                finite_difference=-(energies[1]-energies[0])/2e-6
                assert abs(finite_difference-ref_force[i][a])<2e-7
                fdtests+=1
            for steps in (0,16):
                x,v,e,f,w = evolve(points,velocities,masses,bonds,steps)
                for (ranks,grid),precision,halo in itertools.product(grids,('double','mixed'),('sync','async')):
                    cmd=['mpiexec','--oversubscribe','-n',str(ranks),args.executable,str(snapshot),
                         f'--bonds={bondfile}',f'--grid={grid}',f'--precision={precision}',
                         f'--halo={halo}','--skin=.4','--threads=2','--simd-width=4']
                    if steps: cmd += [f'--steps={steps}','--dt=.001',f'--state={state}']
                    run=subprocess.run(cmd,text=True,capture_output=True,env=env,timeout=120)
                    assert run.returncode==0,(cmd,run.stdout,run.stderr)
                    rows=[r.split() for r in run.stdout.splitlines()]
                    got=[float(next(r[1] for r in rows if r[0]=='energy'))]
                    got+=list(map(float,next(r[1:] for r in rows if r[0]=='virial')))
                    ref=[e]+w
                    forces={int(r[1]):list(map(float,r[2:])) for r in rows if r[0]=='force'}
                    assert set(forces)==set(ids)
                    for i,gid in enumerate(ids): got+=forces[gid];ref+=f[i]
                    if steps:
                        states={int(r[1]):list(map(float,r[2:])) for r in rows if r[0]=='state'}
                        assert set(states)==set(ids)
                        for i,gid in enumerate(ids): got+=states[gid];ref += [masses[i]]+x[i]+v[i]
                        if name in ('shared-endpoint','periodic','lj-and-bond') and ranks>1:
                            last=[r.split() for r in run.stderr.splitlines() if r.startswith('step ')][-1]
                            assert int(last[last.index('migrations')+1])>0,last
                    tol=3e-10 if precision=='double' else 5e-5
                    for a,b in zip(got,ref):
                        assert math.isfinite(a) and abs(a-b)<=tol*(1+abs(b)),(cmd,a,b)
                    error=max((abs(a-b) for a,b in zip(got,ref)),default=0.)
                    maxima[precision]=max(maxima[precision],error)
                    tested+=1
                    print(name,steps,grid,precision,halo,f'energy_ref={e:.17g}',f'max_error={error:.4g}',flush=True)
        print('PASS',tested,'bonded configurations',fdtests,'finite differences',maxima)


if __name__=='__main__': main()
