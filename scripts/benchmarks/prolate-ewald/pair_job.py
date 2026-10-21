"""The step and the loop over pairs of the prolate kernel against the
Gaussian, alternated over programs in one job (issue #380,
D280).

    pair_job.py OUT SYSTEM TIERS ROUNDS THREADS BUILDS [--gpu] [--keys K,..]
                [--sample] [--profile] [--steps N] [--subset K,..]

SYSTEM: a directory under $PAIR_DIR/sys (sys.prmtop, sys.inpcrd, npt.h5 if
NPT); the cases of each tier are the prolate default and the fastest
Gaussian of docs/pme.md, Section 7 (#371). TIERS: 1e-3,1e-4,1e-5. THREADS: 16,4,1 (CPU; the first for every
case, the others for --subset). BUILDS: name=PATH[:VAR=VAL;VAR=VAL],...
Everything but [pme] is the control of cpu_job.py of #371 (9 A cutoff).
Round 0 sets the steps (about 20 s; --steps fixes them) and is dropped by
the table. --sample: one run of each under $SAMPLER after the rounds;
--profile (GPU): one run of each with MDRT_PROFILE and MDRT_WAIT.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

S = os.environ['PAIR_DIR']
SAMPLER = os.environ.get('SAMPLER', '')
CUTOFF = 9.0
TARGET_S = 20.0

GRIDS = {  # the fastest Gaussian of docs/pme.md section 7 (#371)
    'jac': {'1e-03': (4, 1e-4, (50, 50, 50)), '1e-04': (4, 1e-5, (96, 96, 96)),
            '1e-05': (6, 1e-6, (90, 90, 90))},
    'tip3p-34k': {'1e-03': (4, 1e-4, (84, 84, 84)), '1e-04': (6, 1e-5, (90, 90, 90)),
                  '1e-05': (6, 1e-6, (140, 140, 140))},
    'tip3p-103k': {'1e-03': (4, 1e-4, (120, 120, 120)), '1e-04': (6, 1e-5, (126, 126, 126)),
                   '1e-05': (6, 1e-6, (192, 192, 192))},
    'cellulose': {'1e-03': (4, 1e-4, (210, 100, 100)), '1e-04': (6, 1e-5, (224, 108, 108)),
                  '1e-05': (6, 1e-6, (336, 160, 160))},
}


def cases(name, tier):
    o, tol, g = GRIDS[name][tier]
    return {
        'prolate-p6': ['kernel = "PROLATE"', f'tolerance = {float(tier):g}', 'order = 6'],
        'gaussian': [f'tolerance = {tol:g}', f'order = {o}', 'grid = [%d, %d, %d]' % g],
    }


out, name, tiers, rounds, threads, builds = sys.argv[1:7]
tiers = [f'{float(t):.0e}' for t in tiers.split(',')]
rounds = int(rounds)
threads = [int(t) for t in threads.split(',')]
B = {}
for spec in builds.split(','):
    n, rest = spec.split('=', 1)
    path, _, envs = rest.partition(':')
    B[n] = (path, dict(e.split('=', 1) for e in envs.split(';') if e))
gpu = sample = profile = False
keys = None
subset = None
fixed = None
a = sys.argv[7:]
while a:
    if a[0] == '--gpu':
        gpu = True; a = a[1:]
    elif a[0] == '--sample':
        sample = True; a = a[1:]
    elif a[0] == '--profile':
        profile = True; a = a[1:]
    elif a[0] == '--keys':
        keys = a[1].split(','); a = a[2:]
    elif a[0] == '--subset':
        subset = a[1].split(','); a = a[2:]
    elif a[0] == '--steps':
        fixed = int(a[1]); a = a[2:]
    else:
        sys.exit('unknown ' + a[0])
D = f'{S}/sys/{name}'
NPT = os.path.exists(f'{D}/npt.h5')
OUT = os.path.abspath(out)
os.makedirs(OUT, exist_ok=True)
MODEL = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?')
if gpu:
    MODEL = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True).stdout.strip() + ' / ' + MODEL


def control(tier, pme, nthreads, steps):
    head = f'[input]\nformat      = "AMBER"\ntopology    = "{D}/sys.prmtop"\ncoordinates = "{D}/sys.inpcrd"\n'
    if NPT:
        head += f'checkpoint  = "{D}/npt.h5"\n'
        ensemble = ('[ensemble]\nensemble    = "NPT"\ntemperature = 300.0\npressure    = 1.0\n'
                    '[barostat]\nmethod        = "C-RESCALE"\ntime_constant = 2.0\n')
    else:
        ensemble = '[ensemble]\nensemble    = "NVT"\ntemperature = 300.0\n'
    target = 'GPU' if gpu else 'CPU'
    execution = f'[execution]\ntarget    = "{target}"\nprecision = "MIXED"\n'
    if not gpu:
        execution += f'threads   = {nthreads}\n'
    return (f'# {name} at {tier}: issue #380\n' + head +
            f'[output]\nenergy_interval = {steps // 2}\n'
            f'[dynamics]\nintegrator = "VELOCITY_VERLET"\ntime_step  = 0.002\nsteps      = {steps}\n'
            + ensemble +
            '[thermostat]\nmethod        = "V-RESCALE"\ntime_constant = 1.0\ninterval      = 25\n'
            f'[energy]\ncutoff            = {CUTOFF}\nelectrostatics    = "PME"\n'
            'coulomb_modifier  = "POTENTIAL_SHIFT"\n'
            '[pme]\n' + '\n'.join(pme) + '\n'
            '[constraints]\nhydrogen_bonds = true\nrigid_water    = true\n'
            '[boundary]\ntype = "PERIODIC"\n' + execution)


variants = []
for tier in tiers:
    cs = cases(name, tier)
    for k in cs:
        if keys and k not in keys:
            continue
        for b in B:
            variants.append((tier, k, b, threads[0]))
    for t in threads[1:]:
        for k in cs:
            if (subset or keys) and k not in (subset or keys):
                continue
            for b in B:
                variants.append((tier, k, b, t))
steps = {}


def run(v, r, mode=''):
    tier, key, b, nthreads = v
    tagv = f'{tier}-{key}-{b}@{nthreads}'
    d = f'{OUT}/{tagv}' + (f'-{mode}' if mode else f'-r{r}')
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    n = fixed or steps.get(tagv, 50)
    if mode:
        n = steps.get(tagv, 50) if not gpu else min(steps.get(tagv, 50), 2000)
    open(d + '/mdir.toml', 'w').write(control(tier, cases(name, tier)[key], nthreads, n))
    path, envs = B[b]
    env = dict(os.environ, OMP_PROC_BIND='close', OMP_PLACES='cores', **envs)
    if mode == 'sampled':
        os.makedirs(d + '/prof', exist_ok=True)
        env.update(LD_PRELOAD=SAMPLER, SAMPLER_OUT=d + '/prof')
    if mode == 'profile':
        env.update(MDRT_PROFILE='1', MDRT_WAIT='1')
    t0 = time.time()
    p = subprocess.run([path, 'run', 'mdir.toml'], cwd=d, capture_output=True, text=True, env=env)
    log = p.stdout + p.stderr
    open(d + '/run.log', 'w').write(log)
    rate = re.search(r'from step [0-9]+ to step [0-9]+, ([0-9.]+) ms per step', log)
    rate = float(rate.group(1)) if rate else None
    if r == 0 and not mode and rate and not fixed:
        steps[tagv] = max(50, min(20000, int(TARGET_S / (rate / 1000.0) / 50) * 50))
    pairs = None
    if mode == 'profile':
        tot = 0.0
        calls = 0
        for line in log.splitlines():
            m = re.match(r'mdrt: (\S*pair_for\S*)\s+(\d+)\s+([0-9.]+)', line)
            if m:
                tot += float(m.group(3)); calls = max(calls, int(m.group(2)))
        pairs = dict(seconds=tot, calls=calls, lines=[l for l in log.splitlines() if 'pair_for' in l])
    g = re.search(r'grid[^0-9\n]*(\d+) x (\d+) x (\d+)', log)
    res = dict(system=name, tier=tier, key=key, build=b, threads=nthreads, round=r, mode=mode,
               rate_ms=rate, steps=n, ok=p.returncode == 0, wall_s=time.time() - t0, dir=d,
               host=os.uname().nodename, cpu=MODEL, pairs=pairs,
               grid=[int(x) for x in g.groups()] if g else None)
    with open(f'{OUT}/results.jsonl', 'a') as f:
        f.write(json.dumps(res) + '\n')
    print(json.dumps({k: res[k] for k in ('tier', 'key', 'build', 'threads', 'round', 'mode', 'rate_ms', 'ok')}),
          flush=True)


print(f'{name}: {len(variants)} variants; host {os.uname().nodename}; {MODEL}; '
      f'affinity {len(os.sched_getaffinity(0))} cores', flush=True)
for r in range(rounds):
    for v in (variants if r % 2 == 0 else variants[::-1]):
        run(v, r)
if sample and SAMPLER and not gpu:
    for v in variants:
        run(v, 0, 'sampled')
if profile and gpu:
    for v in variants:
        run(v, 0, 'profile')
