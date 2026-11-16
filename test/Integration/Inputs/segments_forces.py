"""The serial definition of D417 over the lists of units that the device
builds (D[md-alg-segments-build]), against the forces of another build at the
same positions: `main`'s general groups in double precision, or #795's
template in mixed precision.

  segments_forces.py emit SYSTEM.top SYSTEM.gro PROGRAM.mlir KINDS OUT.mlir DUMP POSITIONS [--drop-record]
      a program that runs the op of PROGRAM.mlir (convert-md-exec-to-md-alg,
      double) in its buffer form over the records of DUMP (dump-segments)
      at POSITIONS (`mdir checkpoint --print=positions`), in the cell of
      SYSTEM.gro, and prints the energy, the forces and the virial
      (segments_serial.py's probe)
  segments_forces.py compare SYSTEM.top SYSTEM.gro KINDS POSITIONS VALUES FORCES PRECISION CUTOFF [--taken-only]
      compares the probe's forces with FORCES (`--print=forces`) within the
      error model of two evaluations of the same sum in different orders:
      2 gamma_n M + 32 u M + dx (|u''| + |u'|/r) + 4 u max M for each
      component, M the sum of the magnitudes of its terms, n = 1024 (D415),
      u the unit roundoff of the precision of FORCES (f64, or f32 for
      mixed), dx = 8 u L the rounding of a displacement, L the largest
      coordinate (D405's oracle). The magnitudes are this script's, over
      every site pair within the cutoff in the minimum image.
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'Driver' / 'Inputs'))
import segments_serial  # noqa: E402
from kind_pairs_oracle import COULOMB, System  # noqa: E402

TERMS = 1024


def read_rows(path):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    return np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])


def gro_cell(gro):
    box = [float(v) for v in Path(gro).read_text().splitlines()[-1].split()]
    box += [0.0] * (9 - len(box))
    if not any(box[3:]):
        return box[:3]
    return [box[0], box[1], box[2], box[5], box[7], box[8]]


def read_dump(path):
    raw = Path(path).read_bytes()
    offset, parts = 48, {}
    for name in ['order', 'shifts', 'entries', 'masks', 'groups', 'counts', 'offsets',
                 'inner_entries', 'inner_masks', 'inner_counts', 'active', 'active_count']:
        length, = struct.unpack_from('<q', raw, offset)
        offset += 8
        parts[name] = np.frombuffer(raw[offset:offset + 4 * length], np.int32)
        offset += 4 * length
    return parts


if sys.argv[1] == 'emit':
    top, gro, program, kinds, out, dump, positions = sys.argv[2:9]
    # --drop-record, for the test of this test: the first record's entries
    # left out, which the comparison must find.
    drop = '--drop-record' in sys.argv
    system = System(top, gro, kinds.split(','))
    parts = read_dump(dump)
    # The members of the units of the taken kinds, kind by kind, frame atoms
    # in frame order: the table that the lowering forms (empty_segmented_pairs).
    members = []
    for name in system.taken:
        frame = system.frame(name)
        for unit in system.molecules[name]:
            members += [unit[i] for i in frame]
    counts = parts['counts'].tolist()
    if drop:
        counts[0] = 0
    lists = {'entries': parts['entries'].tolist(), 'masks': parts['masks'].tolist(),
             'counts': counts, 'units': parts['groups'].tolist(),
             'ordinals': [0] * len(parts['counts']), 'places': parts['order'].tolist(),
             'shifts': parts['shifts'].tolist(), 'segments': parts['offsets'].tolist(),
             'members': members or [0]}
    stored = read_rows(positions).tolist()
    segments_serial.emit(system, program, out, None, False, given=(lists, stored, gro_cell(gro)))
    sys.exit(0)

top, gro, kinds, positions, values, forces, precision, cutoff = sys.argv[2:10]
# --taken-only: the components of the particles of the taken kinds alone,
# whose forces in a run are those of the pair term (the rest has bonded terms).
taken_only = '--taken-only' in sys.argv
cutoff = float(cutoff)
system = System(top, gro, kinds.split(','))
x = read_rows(positions)
reference = read_rows(forces)
bits = [int(v) for v in Path(values).read_text().split()]
got = np.array([struct.unpack('<d', struct.pack('<q', b))[0] for b in bits])
n = len(x)
probe = got[1:1 + 3 * n].reshape(n, 3)
cell = gro_cell(gro)
if len(cell) == 3:
    H = np.diag(cell)
else:
    ax, by, cz, bx, cx, cy = cell
    H = np.array([[ax, 0, 0], [bx, by, 0], [cx, cy, cz]])
Hinv = np.linalg.inv(H)
types = np.array(system.type_of)
charges = np.array(system.charge_of)
sigma = np.array([system.sigma[k] for k in range(len(system.sigma))])
epsilon = np.array([system.epsilon[k] for k in range(len(system.epsilon))])
excluded_molecule = {}
for name in system.taken:
    for k, unit in enumerate(system.molecules[name]):
        for p in unit:
            excluded_molecule[p] = (name, k)
for p, m in system.rest:
    excluded_molecule[p] = ('rest', m)
# The magnitudes of the terms of each component, over the site pairs within
# the cutoff in the minimum image (every pair of particles of two molecules).
from scipy.spatial import cKDTree  # noqa: E402
frac = x @ Hinv
wrapped = (frac - np.floor(frac)) @ H
M = np.zeros((n, 3))
curvature = np.zeros((n, 3))
images = [np.array([a, b, c]) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
tree = cKDTree(wrapped)
for img in images:
    shift = img @ H
    other = cKDTree(wrapped + shift)
    for i, js in enumerate(tree.query_ball_tree(other, cutoff)):
        for j in js:
            if excluded_molecule[i] == excluded_molecule[j]:
                continue
            d = wrapped[i] - (wrapped[j] + shift)
            r = math.sqrt(d @ d)
            if r == 0.0 or r >= cutoff:
                continue
            s = 0.5 * (sigma[types[i]] + sigma[types[j]])
            e = math.sqrt(epsilon[types[i]] * epsilon[types[j]])
            s6 = (s / r) ** 6
            lj12, lj6, coulomb = 4 * e * s6 * s6, 4 * e * s6, COULOMB * charges[i] * charges[j] / r
            dmag = (12 * abs(lj12) + 6 * abs(lj6) + abs(coulomb)) / r
            ddmag = (156 * abs(lj12) + 42 * abs(lj6) + 2 * abs(coulomb)) / (r * r)
            M[i] += dmag * np.abs(d) / r
            curvature[i] += ddmag + dmag / r
u = 2.0 ** -53 if precision == 'double' else 2.0 ** -24
gamma = (TERMS - 1) * u / (1 - (TERMS - 1) * u)
dx = 8 * u * float(np.abs(x).max())
bound = 2 * gamma * M + 32 * u * M + dx * curvature + 4 * u * float(M.max())
difference = np.abs(probe - reference)
ratio = difference / bound
if taken_only:
    rest = [p for p, _ in system.rest]
    ratio[rest] = 0.0
    difference[rest] = 0.0
    reference = reference.copy()
    reference[rest] = 0.0
worst = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
outside = int((ratio > 1).sum())
counted = 3 * (n - (len(system.rest) if taken_only else 0))
print(f'{precision}: {counted // 3} particles, largest difference {float(ratio.max()):.3g} of its bound '
      f'(force {worst[0]}.{worst[1]}: {float(difference[worst]):.3g} against {float(bound[worst]):.3g}), '
      f'{outside} of {counted} components outside, relative RMS {float(np.sqrt((difference ** 2).sum() / (reference ** 2).sum())):.3g}')
sys.exit(0 if outside == 0 else 1)
