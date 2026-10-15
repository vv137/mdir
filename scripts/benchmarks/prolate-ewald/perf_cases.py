"""Control files of mdir run for the step of the prolate kernel against the
Gaussian kernel on the systems of the Amber suite, with the suite's
settings but [pme] (D[prolate-ewald]).  perf_cases.py NAME STEPS OUTDIR

Cases: the suite's own [pme] (pmemd's β and grid, OPTIMAL); MDIR's
default; the Gaussian of order 6 at the grid that reached Δ ≤ 1e-5 on JAC
(scaled by the spacing); the prolate kernel at its default (ε = 1e-5,
P = 6), at 1e-4 (P = 6), and at 1e-3 (P = 5). Writes runs.json for
mdir_job.py."""
import json, os, sys

name, steps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SUITE = os.path.join(os.environ.get('MDIR_BENCH_DIR', '.'), 'Amber24_Benchmark_Suite', 'PME', '')
src = {'jac': 'JAC_production_NVE', 'cel': 'Cellulose_production_NVE', 'stmv': 'STMV_production_NPT_4fs'}[name]
suite_grid = {'jac': [64, 64, 64], 'cel': [256, 128, 128], 'stmv': [224, 224, 240]}[name]
# Gaussian of order 6 at Δ ≤ 1e-5: 72³ on JAC (62.23 Å), i.e. a spacing of 0.864 Å.
edges = {'jac': [62.23, 62.23, 62.23], 'cel': [259.23, 124.56, 123.50], 'stmv': [221.17, 223.20, 224.49]}[name]
npt = name == 'stmv'
cases = {
    'gaussian-suite': ['beta = 0.39467022824439113', 'grid = [%d, %d, %d]' % tuple(suite_grid), 'order = 4', 'influence = "OPTIMAL"'],
    'gaussian-default': [],
    'gaussian-6-1e-5': ['tolerance = 3e-6', 'max_spacing = 0.864', 'order = 6'],
    'prolate-1e-5': ['kernel = "PROLATE"'],
    'prolate-1e-4': ['kernel = "PROLATE"', 'tolerance = 1e-4'],
    'prolate-1e-3': ['kernel = "PROLATE"', 'tolerance = 1e-3', 'order = 5'],
}
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
    runs.append(dict(tag=f'{name}-{case}', dir=d, case=dict(name=name, key=case), order=0))
json.dump(runs, open(os.path.join(out, f'runs-{name}.json'), 'w'), indent=1)
print(len(runs), 'control files')
