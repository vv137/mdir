"""A rhombic dodecahedron of TIP3P water with the template ON.

The cell of Inputs/triclinic (403 waters, a reduced triclinic cell with
a = (d, 0, 0), b = (0, d, 0), c = (d/2, d/2, d/sqrt 2), d = 2.6 nm) is
replicated twice along each of its lattice vectors: the 3,224 waters of a
dodecahedron with d = 5.2 nm, the same periodic system, whose least face
width (c_z, 3.68 nm) holds the covering reach of the template twice over.
PME, 2 fs, C-RESCALE at constant pressure, which moves the tilts.

--cell keeps the cell of Inputs/triclinic itself (403 waters, d = 2.6 nm,
least face width c_z = 1.8385 nm, every edge 2.6 nm) under NVT, with a
cutoff of 6 Å, a pruned distance of 6.6 Å and 'pairlist_distance' D
(--pairlist=D, Å): the run-time check of the covering reach against half
the least face width. --stretch then lengthens the first water's O-HW1
bond by 30 %, which the first, unconstrained build takes as the water's
extent.

    water_template_dodecahedron.py ROOT WORK STEPS [--cell [--pairlist=D] [--stretch]]
        (STEPS a multiple of 10)
"""
from pathlib import Path
import sys

root, work = map(Path, sys.argv[1:3])
steps = int(sys.argv[3])
assert steps % 10 == 0
options = dict(a[2:].split('=', 1) for a in sys.argv[4:] if '=' in a)
flags = set(a for a in sys.argv[4:] if '=' not in a)
assert set(options) <= {'pairlist'} and flags <= {'--cell', '--stretch'}
cell = '--cell' in flags
assert cell or not (options or flags)
pairlist = float(options.get('pairlist', 10.0))
source = (root / 'dodecahedron.gro').read_text().splitlines()
count = int(source[1])
box = [float(v) for v in source[2 + count].split()]
# v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
a = (box[0], box[3], box[4])
b = (box[5], box[1], box[6])
c = (box[7], box[8], box[2])
atoms = source[2:2 + count]
if '--stretch' in flags:
    row = lambda r: [float(r[20 + 8 * n:28 + 8 * n]) for n in range(3)]
    o, h = row(atoms[0]), row(atoms[1])
    assert atoms[0][10:15].strip() == 'OW' and atoms[1][10:15].strip() == 'HW1'
    h = [x + 1.3 * (y - x) for x, y in zip(o, h)]
    atoms[1] = atoms[1][:20] + ''.join(f'{v:8.3f}' for v in h) + atoms[1][44:]
copies = 1 if cell else 2
out = ['dodecahedron of TIP3P water, ' + ('the cell' if cell else 'twice the cell')
       + ' of Inputs/triclinic', f'{copies ** 3 * count:5d}']
serial = 0
for i in range(copies):
    for j in range(copies):
        for k in range(copies):
            shift = [i * a[n] + j * b[n] + k * c[n] for n in range(3)]
            for row in atoms:
                xyz = [float(row[20 + 8 * n:28 + 8 * n]) + shift[n] for n in range(3)]
                residue = (serial // 3) + 1
                serial += 1
                out.append(f'{residue % 100000:5d}{"SOL":<5}{row[10:15]:>5}{serial % 100000:5d}'
                           + ''.join(f'{v:8.3f}' for v in xyz))
out.append(''.join(f'{copies * v:10.5f}' for v in box))
(work / 'dodecahedron.gro').write_text('\n'.join(out) + '\n')
top = (root / 'water.top').read_text()
assert top.count('SOL 403') == 1
waters = copies ** 3 * count // 3
(work / 'water.top').write_text(top.replace('SOL 403', f'SOL {waters}'))
ensemble = 'NVT' if cell else 'NPT'
barostat = '' if cell else '''[barostat]
method = "C-RESCALE"
time_constant = 2.0
interval = 10
'''
(work / 'control.toml').write_text(f'''
[input]
topology = "{work / 'water.top'}"
coordinates = "{work / 'dodecahedron.gro'}"
[output]
energy_interval = 10
[energy]
cutoff = {6.0 if cell else 8.0}
pairlist_distance = {pairlist}
pruned_distance = {6.6 if cell else 8.6}
electrostatics = "PME"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.002
steps = {steps}
[ensemble]
ensemble = "{ensemble}"
temperature = 300.0
{'' if cell else 'pressure = 1.0'}
[constraints]
rigid_water = true
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
[performance]
"water.template" = "ON"
[thermostat]
method = "V-RESCALE"
time_constant = 1.0
interval = 10
{barostat}''')
print(f'dodecahedron: {waters} waters, a = {copies * box[0]:.2f} nm, c_z = {copies * box[2]:.4f} nm')
