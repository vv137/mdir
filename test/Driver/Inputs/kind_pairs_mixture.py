"""Writes a system of four kinds of rigid groups for D405:

    python kind_pairs_mixture.py DIRECTORY [incomplete | outside]

three waters of three sites (TIP3P), two of four sites whose site is a
linear combination of the frame ([ virtual_sites3 ] of type 1, the
parameters of OPC), two rigid molecules of three atoms that all have
Lennard-Jones, and one ion, in a cell of 4 nm, with plain cutoffs of
0.9 nm, as system.top, system.gro and run.toml. The molecules are placed at
random orientations (a fixed seed) within 1.3 nm of each other, no two
sites of different molecules nearer than 0.25 nm, so that most site pairs
are inside the cutoff and some outside. Two variants break a fact of
D403 for the molecules of three Lennard-Jones atoms: `incomplete` drops
their exclusions, `outside` gives each an ion of its own, not bonded to it,
that it excludes.
"""
import math
import random
import sys
from pathlib import Path

work = Path(sys.argv[1])
variant = sys.argv[2] if len(sys.argv) > 2 else None
assert variant in (None, 'incomplete', 'outside')
random.seed(859)

# name: (atoms as (name, type, charge, mass), the frame distances
# d(1,2) = d(1,3) and d(2,3) in nm, the site weights a, b over atoms 2 and 3)
OPC_A = 0.14772060947071208
MOLECULES = {
    'SOL': ([('OW', 'OW', -0.834, 16.0), ('HW1', 'HW', 0.417, 1.008),
             ('HW2', 'HW', 0.417, 1.008)], (0.09572, 0.15139), None),
    'OPC': ([('O', 'OP', 0.0, 16.0), ('H1', 'HP', 0.679142, 1.008),
             ('H2', 'HP', 0.679142, 1.008), ('EP', 'EP', -1.358284, 0.0)],
            (0.0872433, 0.1371205), (OPC_A, OPC_A)),
    'LJ3': ([('S', 'S', 0.2, 32.06), ('C1', 'C', -0.1, 12.011),
             ('C2', 'C', -0.1, 12.011)], (0.12, 0.2), None),
    'NA': ([('NA', 'NA', 1.0, 22.99)], None, None),
}
if variant == 'outside':
    atoms, frame, site = MOLECULES['LJ3']
    MOLECULES['LJ3'] = (atoms + [('NA', 'NA', 1.0, 22.99)], frame, site)
COUNTS = [('SOL', 3), ('OPC', 2), ('LJ3', 2), ('NA', 1)]
# type: (sigma nm, epsilon kJ/mol)
TYPES = {'OW': (0.315061, 0.636386), 'HW': (0.0, 0.0), 'OP': (0.316655, 0.890359),
         'HP': (0.0, 0.0), 'EP': (0.0, 0.0), 'S': (0.36, 1.0), 'C': (0.34, 0.4),
         'NA': (0.24, 0.37)}


def rotation():
    q = [random.gauss(0, 1) for _ in range(4)]
    n = math.sqrt(sum(v * v for v in q))
    w, x, y, z = (v / n for v in q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def shape(name):
    """Positions of a molecule about its first atom, before rotation."""
    atoms, frame, site = MOLECULES[name]
    if frame is None:
        return [(0.0, 0.0, 0.0)]
    d, e = frame
    half = math.asin(e / (2 * d))
    points = [(0.0, 0.0, 0.0), (d * math.cos(half), d * math.sin(half), 0.0),
              (d * math.cos(half), -d * math.sin(half), 0.0)]
    if site:
        a, b = site
        points.append(tuple(a * points[1][k] + b * points[2][k] for k in range(3)))
    if len(atoms) > len(points):
        points.append((0.0, 0.0, 0.4))
    return points


placed = []
center = (2.0, 2.0, 2.0)
for name, count in COUNTS:
    for _ in range(count):
        for attempt in range(10000):
            origin = [center[k] + random.uniform(-0.65, 0.65) for k in range(3)]
            r = rotation()
            points = [tuple(origin[i] + sum(r[i][k] * p[k] for k in range(3))
                            for i in range(3)) for p in shape(name)]
            if all(math.dist(p, q) > 0.25 for _, other in placed for q in other
                   for p in points):
                break
        else:
            raise SystemExit('cannot place the molecules')
        placed.append((name, points))

top = ['[ defaults ]', '  1  2  yes  0.5  0.8333', '', '[ atomtypes ]']
for type, (sigma, epsilon) in TYPES.items():
    ptype = 'D' if type == 'EP' else 'A'
    top.append(f'  {type}  0  1.0  0.0  {ptype}  {sigma}  {epsilon}')
for name, (atoms, frame, site) in MOLECULES.items():
    top += ['', '[ moleculetype ]', f'  {name}  2', '[ atoms ]']
    for index, (atom, type, charge, mass) in enumerate(atoms, 1):
        top.append(f'  {index}  {type}  1  {name}  {atom}  1  {charge}  {mass}')
    if frame:
        top += ['[ settles ]', f'  1  1  {frame[0]}  {frame[1]}']
    if site:
        top += ['[ virtual_sites3 ]', f'  4  1  2  3  1  {site[0]!r}  {site[1]!r}']
    if len(atoms) > 1 and not (variant == 'incomplete' and name == 'LJ3'):
        top.append('[ exclusions ]')
        for i in range(1, len(atoms) + 1):
            top.append('  ' + '  '.join(str(j) for j in [i] + [k for k in range(1, len(atoms) + 1) if k != i]))
top += ['', '[ system ]', 'four kinds', '', '[ molecules ]']
top += [f'  {name}  {count}' for name, count in COUNTS]
(work / 'system.top').write_text('\n'.join(top) + '\n')

gro = ['four kinds of rigid groups', f'{sum(len(p) for _, p in placed):5d}']
number = 0
for residue, (name, points) in enumerate(placed, 1):
    for (atom, *_), point in zip(MOLECULES[name][0], points):
        number += 1
        gro.append(f'{residue:5d}{name:<5}{atom:>5}{number:5d}'
                   + ''.join(f'{v:8.3f}' for v in point))
gro.append('   4.00000   4.00000   4.00000')
(work / 'system.gro').write_text('\n'.join(gro) + '\n')

(work / 'run.toml').write_text('''[input]
topology = "system.top"
coordinates = "system.gro"
[energy]
cutoff = 9.0
pairlist_distance = 10.0
electrostatics = "CUTOFF"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 10
[ensemble]
ensemble = "NVE"
temperature = 300.0
[constraints]
rigid_water = true
[boundary]
type = "PERIODIC"
''')
