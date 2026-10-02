"""Give the vacuum peptide an octahedral cell and shift one XH hydrogen
by its tilted b lattice vector, forcing a constraint across that face.
"""
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
sections, flag = {}, None
for line in (root / 'vacuum.prmtop').read_text().splitlines():
    if line.startswith('%FLAG '):
        flag = line.split()[1]
        sections[flag] = []
    elif not line.startswith('%') and flag:
        sections[flag].extend(line.split())
masses = list(map(float, sections['MASS']))
bonds = list(map(int, sections['BONDS_INC_HYDROGEN']))
groups = {}
for a, b in zip(bonds[::3], bonds[1::3]):
    a, b = a // 3, b // 3
    if masses[a] < masses[b]:
        a, b = b, a
    groups.setdefault(a, []).append(b)
hydrogen = next(h[0] for h in groups.values() if len(h) == 1)
path = root / 'vacuum.inpcrd'
lines = path.read_text().splitlines()
n = int(lines[1].split()[0])
values = [float(x) for line in lines[2:] for x in line.split()]
assert len(values) == 3*n+6
angle = 109.4712190
values[3*hydrogen] += 60 * math.cos(math.radians(angle))
values[3*hydrogen+1] += 60 * math.sin(math.radians(angle))
values[-6:] = [60., 60., 60., angle, angle, angle]
path.write_text('\n'.join(lines[:2])+'\n'+''.join(
    ''.join(f'{x:12.7f}' for x in values[i:i+6])+'\n'
    for i in range(0, len(values), 6)))
