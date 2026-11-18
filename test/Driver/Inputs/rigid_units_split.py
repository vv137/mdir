"""rigid_units_split.py IN.gro OUT.gro ATOM AXIS: moves atom ATOM (from 1) of
a GROMACS coordinate file by one lattice vector AXIS (a, b, or c) of its
cell, as a file that wraps each atom into the cell on its own does to a
molecule split by a face (rigid-units-route.test)."""
import sys

src, out, atom, axis = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = open(src).read().splitlines()
box = [float(v) for v in rows[-1].split()] + [0.0] * 6
# a = (v1x, 0, 0), b = (v2x, v2y, 0), c = (v3x, v3y, v3z); the file gives
# v1x v2y v3z v1y v1z v2x v2z v3x v3y.
vectors = {'a': (box[0], 0.0, 0.0), 'b': (box[5], box[1], 0.0), 'c': (box[7], box[8], box[2])}
row = rows[1 + atom]
x = [float(row[20 + 8 * k:28 + 8 * k]) for k in range(3)]
x = [v + d for v, d in zip(x, vectors[axis])]
rows[1 + atom] = row[:20] + ''.join(f'{v:8.3f}' for v in x) + row[44:]
open(out, 'w').write('\n'.join(rows) + '\n')
