"""Controls for alanine dipeptide in OPC or TIP3P with the template ON.

The protein's X-H bonds are constrained (hydrogen_bonds), so its heavy
atoms anchor the solute units of D[rigid-water-template]; --loose leaves
them free, which the template refuses. PME, 2 fs, and an NPT option with
C-RESCALE exercise cell changes between builds. --split writes the
coordinates translated so that the first X-H bond (HH31-CH3 of ACE) lies
across a face of the cell, and wraps every atom into the cell on its own, as
a GROMACS .gro file wrapped per atom is: that bond and the waters on the
faces are split across the cell.

    water_template_solute.py INPUTS WORK MODEL ENSEMBLE STEPS [--loose | --split]
"""
from pathlib import Path
import math
import sys

inputs, work = (Path(a).resolve() for a in sys.argv[1:3])
model, ensemble, steps = sys.argv[3], sys.argv[4], int(sys.argv[5])
loose = sys.argv[6:] == ['--loose']
split = sys.argv[6:] == ['--split']
assert model in ('OPC', 'TIP3P') and ensemble in ('NVE', 'NVT', 'NPT')
assert len(sys.argv) in (6, 7) and (len(sys.argv) == 6 or loose or split)
files = (inputs / 'opc' / 'opc', ) if model == 'OPC' else (inputs / 'dipeptide' / 'dipeptide', )
coordinates = f'{files[0]}.inpcrd'
if split:
    lines = Path(coordinates).read_text().splitlines()
    count = int(lines[1].split()[0])
    values = [float(line[k:k + 12]) for line in lines[2:-1] for k in range(0, len(line), 12)]
    assert len(values) == 3 * count, (len(values), count)
    box = [float(v) for v in lines[-1].split()[:3]]
    xyz = [values[3 * i:3 * i + 3] for i in range(count)]
    # The midpoint of atoms 0 and 1 (HH31 and CH3 of ACE) goes to the
    # face 0 of the axis along which the bond is longest.
    axis = max(range(3), key=lambda a: abs(xyz[1][a] - xyz[0][a]))
    shift = [-(xyz[0][a] + xyz[1][a]) / 2 if a == axis else 0.0 for a in range(3)]
    xyz = [[(x + s) - box[a] * math.floor((x + s) / box[a])
            for a, (x, s) in enumerate(zip(p, shift))] for p in xyz]
    def crosses(i, j):
        return any(abs(xyz[i][a] - xyz[j][a]) > box[a] / 2 for a in range(3))
    assert crosses(0, 1), 'the HH31-CH3 bond is not split'
    sites = 4 if model == 'OPC' else 3
    solute = 22
    waters = sum(crosses(o, o + k) for o in range(solute, count, sites) for k in (1, 2))
    assert waters > 0, 'no water is split'
    print(f'split input: the HH31-CH3 bond and {waters} O-H bonds of water cross a face')
    flat = [v for p in xyz for v in p]
    body = [''.join(f'{v:12.7f}' for v in flat[k:k + 6]) for k in range(0, len(flat), 6)]
    coordinates = str(work / 'split.inpcrd')
    Path(coordinates).write_text('\n'.join([lines[0], lines[1], *body, lines[-1]]) + '\n')
coupling = ''
if ensemble != 'NVE':
    coupling += '''
[thermostat]
method = "V-RESCALE"
time_constant = 1.0
interval = 10
'''
if ensemble == 'NPT':
    coupling += '''
[barostat]
method = "C-RESCALE"
time_constant = 2.0
interval = 10
'''
(work / 'control.toml').write_text(f'''
[input]
topology = "{files[0]}.prmtop"
coordinates = "{coordinates}"
[output]
energy_interval = {max(1, steps // 10)}
[energy]
cutoff = 8.0
pairlist_distance = 10.0
pruned_distance = 8.6
electrostatics = "PME"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = {0.001 if loose else 0.002}
steps = {steps}
[ensemble]
ensemble = "{ensemble}"
temperature = 300.0
{'pressure = 1.0' if ensemble == 'NPT' else ''}
[constraints]
rigid_water = true
hydrogen_bonds = {'false' if loose else 'true'}
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
[performance]
"water.template" = "ON"
{coupling}''')
