"""Predeclared native C-RESCALE crossing gate; never retune or retry here.

The actual owner logs establish dispatch. The nominal anchor reach below
only checks the observed cell contraction; it is not the f64 margin policy.
Ordered entry equality is tested separately on identical seeded frames.
"""
import argparse
import json
import pathlib
import re
import subprocess

import mdir


parser = argparse.ArgumentParser()
parser.add_argument('fixtures', type=pathlib.Path)
parser.add_argument('work', type=pathlib.Path)
parser.add_argument('mdir')
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)
declared = dict(steps=100, seed=11, pressure_atm=20000.0,
                barostat_tau_ps=0.5, barostat_interval=5,
                timestep_ps=0.001, pairlist_angstrom=18.0,
                pruned_angstrom=17.9, cutoff_angstrom=9.0,
                deterministic=False)
# The tight outer/inner gap requests cold refreshes as the cell deforms;
# a cell crossing without any refresh would not exercise the selector.
# Chosen before the first device run, without changing a validity bound.
# Written before the one subprocess: a failed crossing is an unmet gate,
# not a reason to search seeds, change pressure, or run the test again.
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
with (args.work / 'run.log').open('w') as log:
    status = subprocess.run([args.mdir, 'run', str(control)], cwd=args.work,
                            stdout=log, stderr=subprocess.STDOUT).returncode
text = (args.work / 'run.log').read_text()
assert status == 0 and 'MDIR: ran in ' in text, ('native barostat failed; stop', status)
assert not re.search(r'\b(?:nan|inf)\b', text, re.I), 'nonfinite native result; stop'
paths = re.findall(r'MDIR: template search image path (nearest|conditional)', text)
assert paths and paths[0] == 'nearest', ('initial actual path', paths)
assert 'conditional' in paths, ('barostat did not cross in the declared run; stop', paths)
assert paths.index('conditional') > paths.index('nearest')
assert 'template search general fallback' not in text, 'unexpected general fallback; stop'


def amber_section(name):
    raw = (args.fixtures / 'three.prmtop').read_text()
    section = raw.split('%FLAG ' + name + '\n', 1)[1].split('%FLAG ', 1)[0]
    return [word for line in section.splitlines() if not line.startswith('%')
            for word in line.split()]


numbers = list(map(int, amber_section('ATOMIC_NUMBER')))
bonds = list(map(int, amber_section('BONDS_INC_HYDROGEN')))
equilibria = [float(x.replace('D', 'E')) for x in amber_section('BOND_EQUIL_VALUE')]
oh = []
for i in range(0, len(bonds), 3):
    a, b, kind = bonds[i:i + 3]
    if {numbers[a // 3], numbers[b // 3]} == {1, 8}:
        oh.append(equilibria[kind - 1])
assert oh
# OPC's O-H distance controls this nominal molecular extent. Include the
# model's fixed 0.001-A extent allowance, not the image-policy margin.
nominal_reach_nm = (18.0 + 2 * (max(oh) + 0.001)) * 0.1
initial_box = list(map(float, (args.fixtures / 'three.inpcrd').read_text().splitlines()[-1].split()))
assert len(initial_box) == 6 and initial_box[3:] == [90.0] * 3
initial_half_nm = 0.05 * min(initial_box[:3])
assert initial_half_nm > nominal_reach_nm
frames = list(mdir.read_h5md(str(args.work / 'frames.h5md')))
assert frames, 'missing native barostat frames'
cells = [dict(step=int(frame.step), diagonal_nm=list(map(float, frame.cell.diagonal)))
         for frame in frames]
halves = [0.5 * min(frame['diagonal_nm']) for frame in cells]
assert min(halves) < nominal_reach_nm, ('actual cells did not cross nominal reach; stop', halves)
assert min(halves) < initial_half_nm
(args.work / 'crossing.json').write_text(json.dumps(dict(
    actual_paths=paths, cells=cells, nominal_reach_nm=nominal_reach_nm,
    initial_half_nm=initial_half_nm, general_fallbacks=0,
    note='Actual path logs establish selection; nominal reach is not the production margin policy.'),
    indent=2))
print('native barostat: nearest then conditional; general fallback 0')
print('barostat cells cross the nominal anchor half-diagonal threshold')
