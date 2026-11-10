"""Native barostat crossing, checkpoint cold refresh and exact entry replay.

D80 keeps the old list valid across a geometric regime boundary. Force a
new allocation by continuing the exact compressed checkpoint, not by
changing validity criteria or tuning compression after observing a result.
"""
import argparse
import json
import pathlib
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument('fixtures', type=pathlib.Path)
parser.add_argument('work', type=pathlib.Path)
parser.add_argument('helper')
parser.add_argument('runtime_cuda')
parser.add_argument('runtime_base')
parser.add_argument('runner_utils')
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)
declared = dict(steps=100, seed=11, pressure_atm=20000.0,
                barostat_tau_ps=0.5, barostat_interval=5,
                timestep_ps=0.001, pairlist_angstrom=18.0,
                pruned_angstrom=17.9, cutoff_angstrom=9.0,
                deterministic=False, continuation_steps=[1, 4],
                physical_frames=[5, 100, 101, 105],
                cold_refresh='new native allocation from exact step-100 checkpoint',
                entry_reference='forced general iterator on each actual refresh snapshot')
# One predeclared protocol, including checkpoint restart. This does not
# claim an uninterrupted validity-triggered transition. Failure stops;
# pressure, duration, seed and physical bounds are never searched.
(args.work / 'predeclared.json').write_text(json.dumps(declared, indent=2))
control = args.work / 'control.toml'
control.write_text(f'''[input]
topology = "{args.fixtures / 'three.prmtop'}"
coordinates = "{args.fixtures / 'three.inpcrd'}"
[output]
energy_interval = 5
trajectory = "frames.h5md"
trajectory_interval = 5
[energy]
cutoff = 9.0
pairlist_distance = 18.0
pruned_distance = 17.9
electrostatics = "CUTOFF"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 100
seed = 11
[ensemble]
ensemble = "NPT"
temperature = 300.0
pressure = 20000.0
[thermostat]
method = "V-RESCALE"
time_constant = 0.5
interval = 5
[barostat]
method = "C-RESCALE"
time_constant = 0.5
interval = 5
[constraints]
rigid_water = true
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
neighbor_structure = "GROUPS"
precision = "MIXED"
deterministic = false
[performance]
"water.template" = "ON"
''')
def command(argv, stdout, stdin=None):
    with stdout.open('w') as out, stdout.with_suffix(stdout.suffix + '.stderr').open('w') as err:
        if stdin is None:
            result = subprocess.run(argv, cwd=args.work, stdout=out, stderr=err)
        else:
            with stdin.open() as source:
                result = subprocess.run(argv, cwd=args.work, stdin=source, stdout=out, stderr=err)
    assert result.returncode == 0, ('native crossing/replay failed; stop', argv, result.returncode,
                                    stdout.read_text(), stdout.with_suffix(stdout.suffix + '.stderr').read_text())


command([args.helper, str(control), str(args.work), args.runtime_cuda], args.work / 'run.log')
native = json.loads((args.work / 'native.json').read_text())
assert native['physical_frames'] == 4
assert native['nearest_refreshes'] > 0 and native['conditional_refreshes'] > 0
assert native['general_refreshes'] == 0
observations = native['observations']
assert observations and observations[0]['phase'] == 0 and observations[0]['path'] == 0
assert any(o['phase'] == 1 and o['path'] == 1 for o in observations)
replay = pathlib.Path(__file__).with_name('template_search_native_replay.py')
for observation in observations:
    snapshot = args.work / observation['file']
    captured = json.loads(snapshot.read_text())
    program = snapshot.with_suffix('.mlir')
    converted = snapshot.with_suffix('.gpu.mlir')
    lowered = snapshot.with_suffix('.llvm.mlir')
    records = snapshot.with_suffix('.records')
    command([sys.executable, str(replay), 'generate', str(snapshot), str(program)],
            snapshot.with_suffix('.generate.log'))
    option = ('--test-md-template-search=reuse-published-groups force-general '
              f'columns-x={captured["columns_x"]} columns-y={captured["columns_y"]}')
    command(['mdir-opt', str(program), option, '--convert-md-exec-to-gpu'], converted)
    command(['mlir-opt', '--gpu-lower-to-nvvm-pipeline=cubin-format=isa',
             '--reconcile-unrealized-casts'], lowered, converted)
    command(['mlir-runner', '-e', 'main', '--entry-point-result=void',
             '--shared-libs=' + ','.join([args.runner_utils, args.runtime_base, args.runtime_cuda])],
            records, lowered)
    command([sys.executable, str(replay), 'check', str(snapshot), str(records)],
            snapshot.with_suffix('.check.log'))
native['ordered_refresh_replays'] = len(observations)
(args.work / 'crossing.json').write_text(json.dumps(native, indent=2))
print('native checkpoint refresh: nearest then conditional; general fallback 0')
print(f'native refresh/general iterator: {len(observations)} exact ordered snapshots passed')
print('barostat physical checks: four frames passed unchanged bounds')
