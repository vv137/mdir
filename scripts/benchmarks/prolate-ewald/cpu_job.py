"""The step of the prolate kernel against the Gaussian on the CPU target, one
job a comparison (issue #371).

    cpu_job.py NAME CASES.json TIER ROUNDS THREADS [--tag T] [--subset KEY,...]
               [--resample KEY@THREADS,...]

Runs the cases of one tier (TIER: 1e-3, 1e-4, 1e-5; CASES.json from
cpu_cases.py pick) of the system NAME (a directory under $ESP_CPU_DIR/sys)
with `mdir run`, target CPU, mixed precision, a cutoff of 9 Å, at each
number of threads of THREADS (e.g. 16,4,1): every case at the first number,
and the cases of --subset (by default prolate-p6 and the Gaussian of each
order) at the others. The runs alternate in rounds, the order reversed from
round to round; round 0 (50 steps) is dropped by cpu_table.py and sets the
number of steps of the later rounds, about 20 s of steps a run (at least
50, a multiple of 50, so that half of them is a multiple of the interval of
coupling). The rate is that of the second half of a run ("from
step"). Then, with $SAMPLER (the library of
scripts/benchmarks/cpu-breakdown/sampler.c), one run of each under the
sampler, not timed, for the parts of the step (cpu-breakdown/prof.py).

Everything but [pme] is the same in every run: the control of
scripts/benchmarks/cpu-breakdown/bench.py (D267) at 9 Å, its default list
reach (cutoff + 1.5 Å), `coulomb_modifier = "POTENTIAL_SHIFT"` (which
changes nothing with the prolate kernel, whose direct term is 0 at the
cutoff), rigid water and bonds to hydrogen constrained, 2 fs, V-rescale and
C-rescale every 25 steps; NPT from npt.h5 where the system has one, NVT from
the coordinates otherwise.

The environment: ESP_CPU_DIR (required), MDIR (a frozen copy of the
program), SAMPLER. Writes $ESP_CPU_DIR/runs/NAME/TAG/<variant>-r<round>/ and
a JSON line a run to results.jsonl there. --resample repeats only the
sampled runs named, with the steps of their timed runs in the tag (for a
sampled run that did not finish).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

S = os.environ['ESP_CPU_DIR']
MDIR = os.environ.get('MDIR', 'mdir')
SAMPLER = os.environ.get('SAMPLER', '')
CUTOFF = 9.0
TARGET_S = 20.0

name, cases_file, tier, rounds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
threads = [int(t) for t in sys.argv[5].split(',')]
tag = 't' + time.strftime('%m%d-%H%M%S')
subset = None
resample = None
a = sys.argv[6:]
while a:
    if a[0] == '--tag':
        tag = a[1]; a = a[2:]
    elif a[0] == '--subset':
        subset = a[1].split(','); a = a[2:]
    elif a[0] == '--resample':
        resample = a[1].split(','); a = a[2:]
    else:
        sys.exit('unknown ' + a[0])
tier = f'{float(tier):g}'
cases = {c['key']: c for c in json.load(open(cases_file))[tier]}
if subset is None:
    subset = [k for k in cases if k == 'prolate-p6' or re.match(r'gaussian-o\d$', k)]
D = f'{S}/sys/{name}'
NPT = os.path.exists(f'{D}/npt.h5')
OUT = f'{S}/runs/{name}/{tag}'
os.makedirs(OUT, exist_ok=True)
MODEL = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?')
variants = [(k, threads[0]) for k in cases] + [(k, t) for t in threads[1:] for k in subset if k in cases]


def control(case, nthreads, steps):
    head = f'[input]\nformat      = "AMBER"\ntopology    = "{D}/sys.prmtop"\ncoordinates = "{D}/sys.inpcrd"\n'
    if NPT:
        head += f'checkpoint  = "{D}/npt.h5"\n'
        ensemble = ('[ensemble]\nensemble    = "NPT"\ntemperature = 300.0\npressure    = 1.0\n'
                    '[barostat]\nmethod        = "C-RESCALE"\ntime_constant = 2.0\n')
    else:
        ensemble = '[ensemble]\nensemble    = "NVT"\ntemperature = 300.0\n'
    return (f'# {name} {case["key"]} at {tier}: issue #371\n' + head +
            f'[output]\nenergy_interval = {steps // 2}\n'
            f'[dynamics]\nintegrator = "VELOCITY_VERLET"\ntime_step  = 0.002\nsteps      = {steps}\n'
            + ensemble +
            '[thermostat]\nmethod        = "V-RESCALE"\ntime_constant = 1.0\ninterval      = 25\n'
            f'[energy]\ncutoff            = {CUTOFF}\nelectrostatics    = "PME"\n'
            'coulomb_modifier  = "POTENTIAL_SHIFT"\n'
            '[pme]\n' + '\n'.join(case['pme']) + '\n'
            '[constraints]\nhydrogen_bonds = true\nrigid_water    = true\n'
            '[boundary]\ntype = "PERIODIC"\n'
            f'[execution]\ntarget    = "CPU"\nthreads   = {nthreads}\nprecision = "MIXED"\n')


steps = {}


def run(key, nthreads, r, sampled=False):
    variant = f'{key}@{nthreads}'
    d = f'{OUT}/{variant}' + ('-sampled' if sampled else f'-r{r}')
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    n = steps.get(variant, 50)
    open(d + '/mdir.toml', 'w').write(control(cases[key], nthreads, n))
    env = dict(os.environ, OMP_PROC_BIND='close', OMP_PLACES='cores')
    if sampled:
        os.makedirs(d + '/prof', exist_ok=True)
        env.update(LD_PRELOAD=SAMPLER, SAMPLER_OUT=d + '/prof')
    t0 = time.time()
    out = subprocess.run([MDIR, 'run', 'mdir.toml'], cwd=d, capture_output=True, text=True, env=env)
    log = out.stdout + out.stderr
    open(d + '/run.log', 'w').write(log)
    rate = re.search(r'from step [0-9]+ to step [0-9]+, ([0-9.]+) ms per step', log)
    rate = float(rate.group(1)) if rate else None
    b = re.search(r'built (\d+) times, every ([0-9.]+) steps', log)
    if r == 0 and not sampled and rate:
        steps[variant] = max(50, min(20000, int(TARGET_S / (rate / 1000.0) / 50) * 50))
    res = dict(system=name, tier=tier, key=key, threads=nthreads, round=r, sampled=sampled,
               rate_ms=rate, steps=n, grid=cases[key]['grid'], points=cases[key]['points'],
               delta=cases[key]['delta'], delta_mixed=cases[key].get('delta_mixed'),
               builds=int(b.group(1)) if b else None, wall_s=time.time() - t0, ok=out.returncode == 0,
               dir=d, tag=tag, host=os.uname().nodename, cpu=MODEL, program=MDIR)
    with open(f'{OUT}/results.jsonl', 'a') as f:
        f.write(json.dumps(res) + '\n')
    print(json.dumps(res), flush=True)


print(f'{name} at {tier}: {len(variants)} variants; host {os.uname().nodename}; cpu {MODEL}; '
      f'affinity {sorted(os.sched_getaffinity(0))}; program {MDIR}', flush=True)
if resample:
    for line in open(f'{OUT}/results.jsonl'):
        r = json.loads(line)
        if r['round'] > 0 and not r['sampled']:
            steps[f"{r['key']}@{r['threads']}"] = r['steps']
    variants = [(v.split('@')[0], int(v.split('@')[1])) for v in resample]
else:
    for r in range(rounds):
        for key, t in (variants if r % 2 == 0 else variants[::-1]):
            run(key, t, r)
if SAMPLER:
    for key, t in variants:
        run(key, t, 0, sampled=True)
