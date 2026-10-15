"""Control files of MDIR for the cases of a system: each Gaussian case as
it is (order 4, 6, 8 at its grid and beta), and, for each ESP case, MDIR's
order 4 at the ESP grid (the emulation of a coarser grid with the
spreading of order 4).  mk_mdir.py NAME STEPS"""
import json, os, sys

D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
SUITE = os.path.join(os.environ.get('MDIR_BENCH_DIR', '.'), 'Amber24_Benchmark_Suite', 'PME', '')
name, steps = sys.argv[1], int(sys.argv[2])
cases = json.load(open(f'{D}cases/cases-{name}.json'))
src = {'jac': 'JAC_production_NVE', 'cel': 'Cellulose_production_NVE', 'stmv': 'STMV_production_NPT_4fs'}[name]
npt = name == 'stmv'
plist = {'jac': (11.0, 8.6), 'cel': (11.0, 8.6), 'stmv': (11.0, 9.2)}[name]
runs = []
for c in cases:
    if c['key'].startswith('gauss'):
        beta_A = c['beta'] / 10.0
        order = c['order']
        tag = f"{name}-{c['target']}-{c['key']}"
    else:
        # beta irrelevant for the cost; that of the target's Gaussian
        g = [x for x in cases if x['target'] == c['target'] and x['key'] == 'gauss4'][0]
        beta_A = g['beta'] / 10.0
        order = 4
        tag = f"{name}-{c['target']}-{c['key']}-emul4"
    d = f'{D}runs/{tag}'
    os.makedirs(d, exist_ok=True)
    ens = ('[ensemble]\nensemble    = "NPT"\ntemperature = 300.0\npressure    = 0.986923\n\n'
           '[thermostat]\nmethod        = "V-RESCALE"\ntime_constant = 1.0\ninterval      = 25\n\n'
           '[barostat]\nmethod        = "C-RESCALE"\ntime_constant = 2.0\n') if npt else \
          '[ensemble]\nensemble    = "NVE"\ntemperature = 300.0\n'
    txt = f'''# {tag}: measurement of issue #334
[input]
format      = "AMBER"
topology    = "{SUITE}{src}/prmtop"
coordinates = "{SUITE}{src}/inpcrd"

[output]
energy_interval = {steps // 2}

[energy]
cutoff            = 8.0
pairlist_distance = {plist[0]}
pruned_distance   = {plist[1]:.2f}
electrostatics    = "PME"

[pme]
beta      = {beta_A!r}
grid      = [{c['K'][0]}, {c['K'][1]}, {c['K'][2]}]
order     = {order}

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
    open(f'{d}/mdir.toml', 'w').write(txt)
    runs.append(dict(tag=tag, dir=d, case=c, order=order))
json.dump(runs, open(f'{D}runs/runs-{name}.json', 'w'), indent=1)
print(len(runs), 'control files')
