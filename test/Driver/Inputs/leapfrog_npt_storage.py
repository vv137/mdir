"""D[leapfrog-npt-stored-velocity]: independent RATTLE/SHAKE and main comparison.

Bounds declared in the decision before the first run. Forces are inputs to
the independent step oracle, as in leapfrog_langevin_reference.py; force
evaluation itself is not the subject of this storage-contract check.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('mode',choices=['collect','check','compare'])
p.add_argument('inputs',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--target',choices=['CPU','GPU'],default='CPU')
p.add_argument('--precision',choices=['Double','Mixed'],default='Double')
p.add_argument('--work',choices=['Trotter','TrotterFirstOrder','Exact','FirstOrder'],default='Trotter')
p.add_argument('--base',type=Path)
p.add_argument('--integrator',choices=['Leapfrog','VelocityVerlet'],default='Leapfrog')
p.add_argument('--reference-integrator',choices=['Leapfrog','VelocityVerlet'],default='Leapfrog')
p.add_argument('--base-module',type=Path)
p.add_argument('--branch-module',type=Path)
p.add_argument('--baseline-double',type=Path)
p.add_argument('--baseline-ulp',type=Path)
p.add_argument('--perturb-f32-ulp',action='store_true')
a=p.parse_args()
DT=.002
# Same-frame storage, bond tangency and independent-step limits never
# inherit the mixed trajectory baseline multiplier.
ORACLE_BOUND={'Double':(1e-11,1e-9,1e-10),'Mixed':(1e-7,1e-4,1e-5)}[a.precision]
TRAJECTORY_FLOOR={'Double':(1e-11,1e-11,1e-9),'Mixed':(1e-7,1e-7,1e-4)}[a.precision]

def section(lines,flag):
    start=next(i for i,s in enumerate(lines) if s.strip()=='%FLAG '+flag)
    import re
    match=re.fullmatch(r'%FORMAT\((\d+)([a-zA-Z])(\d+)(?:\.\d+)?\)',lines[start+1].strip())
    width=int(match.group(3));values=[]
    for row in lines[start+2:]:
        if row.startswith('%'):break
        values.extend(row[i:i+width].strip() for i in range(0,len(row.rstrip()),width) if row[i:i+width].strip())
    return values

lines=(a.inputs/'dipeptide/dipeptide.prmtop').read_text().splitlines(keepends=True)
masses=np.array(section(lines,'MASS'),float)
rows=np.array(section(lines,'BONDS_INC_HYDROGEN'),int).reshape(-1,3)
pairs=rows[:,:2]//3
lengths=.1*np.array(section(lines,'BOND_EQUIL_VALUE'),float)[rows[:,2]-1]
inverse=1/masses

def image(d,cell):
    return d-np.rint(np.linalg.solve(cell.T,d))@cell

def project(x,v,cell):
    v=v.copy()
    for _ in range(1000):
        worst=0.
        for i,j in pairs:
            d=image(x[j]-x[i],cell);rate=d@(v[j]-v[i])
            worst=max(worst,abs(rate)/np.linalg.norm(d))
            impulse=rate/((inverse[i]+inverse[j])*(d@d))
            v[i]+=inverse[i]*impulse*d;v[j]-=inverse[j]*impulse*d
        if worst<1e-13:return v
    raise AssertionError('independent mass-weighted projection did not converge')

def shake(old,trial,cell):
    x=trial.copy()
    for _ in range(1000):
        for (i,j),length in zip(pairs,lengths):
            old_d=image(old[j]-old[i],cell);d=image(x[j]-x[i],cell)
            error=d@d-length*length
            impulse=error/(2*(inverse[i]+inverse[j])*(d@old_d))
            x[i]+=inverse[i]*impulse*old_d;x[j]-=inverse[j]*impulse*old_d
        # Judge the completed array, not stale errors from intermediate
        # bond corrections. Its propagated f64 coordinate-rounding level
        # is 1.42e-14 for bond 24--22 in the diagnosed state. The approved
        # 1e-13 oracle criterion is about seven times that level; the old
        # 1e-14 could reject correct solutions farther from the origin.
        # This changes no position/velocity comparison bound.
        worst=0.
        for (i,j),length in zip(pairs,lengths):
            d=image(x[j]-x[i],cell)
            worst=max(worst,abs(d@d-length*length)/(length*length))
        if worst<1e-13:return x
    raise AssertionError('independent SHAKE did not converge')

def normalized_rate(x,v,cell):
    return max(abs(image(x[j]-x[i],cell)@(v[j]-v[i]))/np.linalg.norm(image(x[j]-x[i],cell)) for i,j in pairs)

def full(x,v,f,cell):
    return project(x,v+.5*DT*f*inverse[:,None],cell)

if a.mode=='compare':
    assert a.base_module is not None and a.branch_module is not None
    a.output.mkdir(parents=True,exist_ok=True)
    common=['--target',a.target,'--precision',a.precision,'--work',a.work]
    for name,module in [('main',a.base_module),('branch',a.branch_module)]:
        env=dict(os.environ);env['PYTHONPATH']=str(module)
        subprocess.run([sys.executable,__file__,'collect',str(a.inputs),str(a.output/(name+'.npz')),*common],env=env,check=True)
    baseline_args=[]
    if a.precision=='Mixed':
        env=dict(os.environ);env['PYTHONPATH']=str(a.base_module)
        for name,options in [('main-double',['--precision','Double']),
                             ('main-ulp',['--precision','Mixed','--perturb-f32-ulp'])]:
            subprocess.run([sys.executable,__file__,'collect',str(a.inputs),str(a.output/(name+'.npz')),
                            '--target',a.target,'--work',a.work,*options],env=env,check=True)
        baseline_args=['--baseline-double',str(a.output/'main-double.npz'),'--baseline-ulp',str(a.output/'main-ulp.npz')]
    subprocess.run([sys.executable,__file__,'check',str(a.inputs),str(a.output/'branch.npz'),
                    '--base',str(a.output/'main.npz'),*common,*baseline_args],check=True)
elif a.mode=='collect':
    import mdir
    loaded=mdir.load_amber(str(a.inputs/'dipeptide/dipeptide.prmtop'),str(a.inputs/'dipeptide/dipeptide.inpcrd'))
    system,state=loaded.make_system(),loaded.make_state()
    system.cutoff,system.pairlist_distance=.8,.9
    system.electrostatics=mdir.Electrostatics.PME;system.pme_grid=[32]*3
    system.truncation=mdir.Truncation.None_
    system.rigid_water=system.rigid_hydrogen_bonds=True
    state=state.draw_velocities(system,300.,2024)
    perturb_increment=0.
    if a.perturb_f32_ulp:
        # Fixed before acquisition, with no scan/retries. Add one f32
        # spacing to the original f64 coordinate without prerounding it.
        coordinates=np.array(state.positions,copy=True)
        original=coordinates.copy();value=coordinates[606,0]
        perturb_increment=float(np.nextafter(np.float32(value),np.float32(np.inf))-np.float32(value))
        coordinates[606,0]=value+perturb_increment
        assert np.count_nonzero(coordinates!=original)==1
        state.positions=coordinates
    integrator,bath,execution,schedule=mdir.Integrator(),mdir.Ensemble(),mdir.Execution(),mdir.Schedule()
    integrator.method=getattr(mdir.IntegratorMethod,a.integrator);integrator.timestep=DT
    bath.kind=mdir.EnsembleKind.NPT;bath.temperature=300.;bath.seed=2024
    bath.pressure=2000*1.01325;bath.com_period=bath.coupling_period=10
    bath.barostat_work=getattr(mdir.BarostatWork,a.work)
    execution.target=getattr(mdir.Target,a.target);execution.precision=getattr(mdir.Precision,a.precision)
    execution.deterministic=True;schedule.steps=30;schedule.energy_period=10
    sim=mdir.Simulation(mdir.compile(system,state,integrator,bath,execution,schedule))
    data={'_target':np.array(a.target),'_precision':np.array(a.precision),
          '_integrator':np.array(a.integrator),'_work':np.array(a.work),
          '_perturb_particle':np.array(606),'_perturb_axis':np.array(0),
          '_perturb_increment':np.array(perturb_increment)};previous=0
    for step in [10,11,20,21]:
        assert sim.run(step-previous,energy=True)==step-previous
        s=sim.state()
        for field in ['positions','velocities','forces']:data[f'{step}_{field}']=np.array(getattr(s,field))
        data[f'{step}_cell']=np.array(s.cell.vectors)
        data[f'{step}_offset']=np.array(s.velocity_offset)
        previous=step
    np.savez(a.output,**data)
    print(f'{a.target} {a.precision} {a.work}: collected 4 states',flush=True)
else:
    assert a.base is not None
    fixed,base=np.load(a.output),np.load(a.base)
    if a.precision=='Mixed':
        assert a.baseline_double is not None and a.baseline_ulp is not None
        baseline_double,baseline_ulp=np.load(a.baseline_double),np.load(a.baseline_ulp)
        for reference in [base,baseline_double,baseline_ulp]:
            # Preserved pre-metadata acquisitions exist only on CPU.
            assert '_target' in reference or a.target=='CPU'
            if '_target' in reference:assert str(reference['_target'])==a.target
        if '_perturb_increment' in baseline_ulp:assert float(baseline_ulp['_perturb_increment'])>0
    largest_x=largest_v=largest_projection=largest_rate=largest_step=0.
    for step in [10,11,20,21]:
        x,v,f,cell=[fixed[f'{step}_{k}'] for k in ['positions','velocities','forces','cell']]
        bx,bv,bf,bc=[base[f'{step}_{k}'] for k in ['positions','velocities','forces','cell']]
        dx=float(np.max(np.abs(x-bx)));dc=float(np.max(np.abs(cell-bc)))
        base_full=full(bx,bv,bf,bc) if a.reference_integrator=='Leapfrog' else project(bx,bv,bc)
        dv=float(np.max(np.abs(full(x,v,f,cell)-base_full)))
        limits=TRAJECTORY_FLOOR
        if a.precision=='Mixed':
            # Finite-trajectory rounding is calibrated against the same
            # reference route/target: its mixed-double and one-f32-ulp
            # traces. A fixed 1e-7 nm can be smaller than one extra f32
            # constraint correction accumulated over these steps.
            baselines=[]
            for reference in [baseline_double,baseline_ulp]:
                rx,rv,rf,rc=[reference[f'{step}_{k}'] for k in ['positions','velocities','forces','cell']]
                reference_full=full(rx,rv,rf,rc) if a.reference_integrator=='Leapfrog' else project(rx,rv,rc)
                baselines.append((float(np.max(np.abs(bx-rx))),float(np.max(np.abs(bc-rc))),
                                  float(np.max(np.abs(base_full-reference_full)))))
            rounding=tuple(max(left,right) for left,right in zip(*baselines))
            limits=tuple(max(3*b,floor) for b,floor in zip(rounding,TRAJECTORY_FLOOR))
            print(f'{step}: mixed trajectory baseline {rounding}; limits {limits}',flush=True)
        largest_x=max(largest_x,dx,dc);largest_v=max(largest_v,dv)
        print(f'{step}: reference position {dx:.17g} cell {dc:.17g}; projected full velocity {dv:.17g}',flush=True)
        assert dx<=limits[0] and dc<=limits[1] and dv<=limits[2],('trajectory changed beyond rounding',step,dx,dc,dv,limits)
        if step in [10,20]:
            expected=project(bx,bv,bc) if a.reference_integrator=='Leapfrog' else project(bx,bv-.5*DT*bf*inverse[:,None],bc)
            difference=float(np.max(np.abs(v-expected)));rate=normalized_rate(x,v,cell)
            largest_projection=max(largest_projection,difference);largest_rate=max(largest_rate,rate)
            print(f'{step}: independent storage projection {difference:.17g}; normalized bond velocity {rate:.17g}',flush=True)
            assert difference<=ORACLE_BOUND[1] and rate<=ORACLE_BOUND[2],('stored velocity oracle/tangency failed',step,difference,rate,ORACLE_BOUND)
        else:
            before=step-1
            x0,v0,f0,c0=[fixed[f'{before}_{k}'] for k in ['positions','velocities','forces','cell']]
            u=v0+DT*f0*inverse[:,None];trial=x0+DT*u
            predicted=shake(x0,trial,c0)
            corrected=u+(predicted-trial)/DT
            dx_ref=float(np.max(np.abs(x-predicted)))
            dv_ref=float(np.max(np.abs(full(x,v,f,cell)-full(predicted,corrected,f,c0))))
            largest_step=max(largest_step,dx_ref)
            print(f'{step}: independent SHAKE step position {dx_ref:.17g}; full velocity {dv_ref:.17g}',flush=True)
            assert dx_ref<=ORACLE_BOUND[0] and dv_ref<=ORACLE_BOUND[1],('independent step failed',step,dx_ref,dv_ref,ORACLE_BOUND)
    print(f'{a.precision} {a.work}: reference/oracle passed; maximum x {largest_x:.3e}, full v {largest_v:.3e}, storage {largest_projection:.3e}, bond rate {largest_rate:.3e}, independent step x {largest_step:.3e}',flush=True)
