"""Control files of mdir run for the step of the prolate kernel against the
Gaussian kernel on the systems of the Amber suite, with the suite's
settings but [pme] (D266).  perf_cases.py NAME STEPS OUTDIR [MAIN]

The cases are matched by the force error Δ measured on JAC by
grid_sweep.py (eq. 16 over the Coulomb forces of the topology), each grid
given by its spacing so that it scales to the other systems:
- Δ ≈ 1e-3: MDIR's default (1.4e-3) and the suite's [pme] (pmemd's β and
  grid, OPTIMAL; 7.3e-4) against the prolate kernel at ε = 1e-3, P = 5 and 6
  (6e-4);
- Δ ≈ 1e-4: the Gaussian of orders 4, 6, 8 (1.2e-4, 6.7e-5, 4.9e-5) against
  the prolate kernel at ε = 1e-4, P = 6 (30³, 32³) and P = 7 (30³);
- Δ ≈ 1e-5: the Gaussian of orders 6 and 8 (1.1e-5) against the prolate
  kernel at ε = 1e-5, P = 6 (48³, 50³), 7 (40³), and 8 (36³).
With MAIN, the path of a build of main, its runs of the Gaussian default and
of the suite's settings are added, to show the order-4 path unchanged.
Writes runs-NAME.json for mdir_job.py."""
import json, os, sys

name, steps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
main = sys.argv[4] if len(sys.argv) > 4 else None
SUITE = os.path.join(os.environ.get('MDIR_BENCH_DIR', '.'), 'Amber24_Benchmark_Suite', 'PME', '')
src = {'jac': 'JAC_production_NVE', 'cel': 'Cellulose_production_NVE', 'stmv': 'STMV_production_NPT_4fs'}[name]
suite_grid = {'jac': [64, 64, 64], 'cel': [256, 128, 128], 'stmv': [224, 224, 240]}[name]
npt = name == 'stmv'
JAC = 62.23  # the edge of JAC, Å: a grid of K³ there is a spacing of JAC / K


def spacing(k):
    return 'max_spacing = %.4f' % (JAC / k)


def prolate(eps, points, k):
    return ['kernel = "PROLATE"', f'tolerance = {eps:g}', f'order = {points}', spacing(k)]


cases = {
    'gaussian-suite': ['beta = 0.39467022824439113', 'grid = [%d, %d, %d]' % tuple(suite_grid), 'order = 4', 'influence = "OPTIMAL"'],
    'gaussian-default': [],
    'prolate-1e-3-p5': prolate(1e-3, 5, 24),
    'prolate-1e-3-p6': prolate(1e-3, 6, 24),
    'gaussian-4-1e-4': ['tolerance = 1e-5', 'order = 4', spacing(96)],
    'gaussian-6-1e-4': ['tolerance = 1e-6', 'order = 6', spacing(70)],
    'gaussian-8-1e-4': ['tolerance = 1e-6', 'order = 8', spacing(60)],
    'prolate-1e-4-p6-30': prolate(1e-4, 6, 30),
    'prolate-1e-4-p6-32': prolate(1e-4, 6, 32),
    'prolate-1e-4-p7-30': prolate(1e-4, 7, 30),
    'gaussian-6-1e-5': ['tolerance = 1e-6', 'order = 6', spacing(90)],
    'gaussian-8-1e-5': ['tolerance = 1e-6', 'order = 8', spacing(70)],
    'prolate-1e-5-p6-48': prolate(1e-5, 6, 48),
    'prolate-1e-5-p6-50': prolate(1e-5, 6, 50),
    'prolate-1e-5-p7-40': prolate(1e-5, 7, 40),
    'prolate-1e-5-p8-36': prolate(1e-5, 8, 36),
}
binaries = {key: None for key in cases}
if main:
    for key in ('gaussian-suite', 'gaussian-default'):
        cases['main-' + key] = cases[key]
        binaries['main-' + key] = main
runs = []
for case, pme in cases.items():
    d = os.path.join(out, name, case)
    os.makedirs(d, exist_ok=True)
    ens = ('[ensemble]\nensemble    = "NPT"\ntemperature = 300.0\npressure    = 0.986923\n\n'
           '[thermostat]\nmethod        = "V-RESCALE"\ntime_constant = 1.0\ninterval      = 25\n\n'
           '[barostat]\nmethod        = "C-RESCALE"\ntime_constant = 2.0\n') if npt else \
          '[ensemble]\nensemble    = "NVE"\ntemperature = 300.0\n'
    text = f'''# {name} {case}: the step of the prolate kernel (issue #334)
[input]
format      = "AMBER"
topology    = "{SUITE}{src}/prmtop"
coordinates = "{SUITE}{src}/inpcrd"

[output]
energy_interval = {steps // 2}

[energy]
cutoff            = 8.0
pairlist_distance = 11.0
pruned_distance   = {9.2 if npt else 8.6}
electrostatics    = "PME"

[pme]
''' + '\n'.join(pme) + f'''

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = {0.004 if npt else 0.002}
steps      = {steps}

{ens}
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
neighbor_structure = "GROUPS"
'''
    open(os.path.join(d, 'mdir.toml'), 'w').write(text)
    runs.append(dict(tag=f'{name}-{case}', dir=d, case=dict(name=name, key=case), order=0, mdir=binaries[case]))
json.dump(runs, open(os.path.join(out, f'runs-{name}.json'), 'w'), indent=1)
print(len(runs), 'control files')
