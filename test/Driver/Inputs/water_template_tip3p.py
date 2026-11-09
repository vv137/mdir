"""Three nearby waters from the tracked TIP3P fixture, centered in 4 nm.

The parameters and coordinates come from Inputs/triclinic. Retain whole
molecules, no velocities, and plain-cutoff interactions for the pair oracle.
"""
from pathlib import Path
import sys

root, work = map(Path, sys.argv[1:])
source = (root / 'dodecahedron.gro').read_text().splitlines()
count = int(source[1])
assert count % 3 == 0
atoms = source[2:2 + count]
coordinates = [[float(row[20 + 8*k:28 + 8*k]) for k in range(3)] for row in atoms]
oxygens = coordinates[::3]
closest = sorted(range(len(oxygens)),
                 key=lambda i: sum((a-b)**2 for a, b in zip(oxygens[i], oxygens[0])))[:3]
chosen = [3*i+j for i in closest for j in range(3)]
center = [sum(coordinates[i][k] for i in chosen)/9 for k in range(3)]
out = ['three nearby TIP3P waters', '    9']
for index, original in enumerate(chosen):
    xyz = [coordinates[original][k]-center[k]+2.0 for k in range(3)]
    out.append(f'{index//3+1:5d}{"SOL":<5}{atoms[original][10:15]:>5}{index+1:5d}'
               + ''.join(f'{v:8.3f}' for v in xyz))
out.append('   4.00000   4.00000   4.00000')
(work / 'three.gro').write_text('\n'.join(out)+'\n')
top = (root / 'water.top').read_text()
assert top.count('SOL 403') == 1
(work / 'three.top').write_text(top.replace('SOL 403', 'SOL 3'))
(work / 'control.toml').write_text('''
[input]
topology = "three.top"
coordinates = "three.gro"
[output]
energy_interval = 1
[energy]
cutoff = 9.0
pairlist_distance = 10.0
electrostatics = "CUTOFF"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 2
[ensemble]
ensemble = "NVE"
temperature = 300.0
[constraints]
rigid_water = true
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
[performance]
"water.template" = "ON"
''')
