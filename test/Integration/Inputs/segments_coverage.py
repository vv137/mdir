"""The coverage of the lists of units that the device builds
(D419; D417, Section 8, the gate of S2), against every
image of every pair of reference points in f64.

  segments_coverage.py system SYSTEM ROOT WORK STEPS [--cell] [--npt] [--forces] [--cutoff=Å] [--pairlist=Å] [--pruned=Å]
      WORK/control.toml, a general run of groups whose trajectory (DCD, every
      step) the harness replays. SYSTEM:
        dodecahedron     TIP3P in a rhombic dodecahedron, NPT (--cell: the
                         small one of Inputs/triclinic, NVT, or NPT with
                         --npt; with --cutoff=6 --pairlist=7 --pruned=6.6
                         its groups take the images kernel of D115)
        mixture          the same dodecahedron, a third of its waters OPC and
                         four of them sodium ions: two taken kinds of
                         different extents, one with a site, and the rest
        dipeptide-opc, dipeptide-tip3p
                         alanine dipeptide in water, NVT (--npt: C-rescale),
                         with the anchors and the excluded pairs of the rest
        dipeptide-gro    the dipeptide in TIP3P from its GROMACS files
        growth           no run: two frames written here, two waters and a
                         molecule of two atoms whose second is anchored,
                         the cell grown by half at the second and the waters
                         moved towards each other by more than the least
                         half-margin of the segments but less than that of
                         the largest radii (D80 under a barostat)
      --forces: one step in double precision with Coulomb cut off, NVE, and a
      checkpoint (segments_forces.py), the bonded terms of the dipeptide set
      to 0 (without_bonded).
  segments_coverage.py emit SYSTEM ROOT WORK [--frames=N | --positions=FILE] [--drop-exclusion]
      WORK/harness.mlir: an empty list of units carried through a loop over
      the frames, a refresh at each frame's positions and cell
      (md-exec-reuse-neighbors's form), for convert-md-exec-to-gpu with
      dump-segments. --drop-exclusion leaves the first excluded pair of the
      rest out of the list's exclusions; --join-exclusion adds an excluded
      pair of the rest and a water.
  segments_coverage.py check SYSTEM ROOT WORK DUMPS [--scale=S] [--limit=L] [--expect-missing] [--any-points]
      Each dump is the list after the refresh of a frame. Over its records:
        - each pair of units in each image at most once (`unique`);
        - each entry in the segment of the kinds of its two units;
        - no excluded pair and no pair within one unit;
        - its inner records (the outer ones without pruning) hold every pair
          of reference points within r_c + e_f + e_g of its segment at that
          frame, which every site pair within r_c needs; right after a build
          its outer records hold every pair within the covering reach R_s.
      Prints the counts and the frames just before a pruning, just before a
      build, and right after a build, which must all occur (not with
      --any-points). With --scale=S the cutoff is S r_c, and
      --expect-missing requires pairs without an entry; --limit=L checks the
      first L dumps.

Lengths in nm. The covering radii: the water's (TIP3P 0.09582 nm, OPC
0.0873433 nm, those of #795) and the rest's anchors (0.10901 nm, D417's rho).
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np

mode = sys.argv[1]
name = sys.argv[2]
root, work = Path(sys.argv[3]), Path(sys.argv[4])
options = dict(a[2:].split('=', 1) for a in sys.argv[5:] if a.startswith('--') and '=' in a)
flags = {a for a in sys.argv[5:] if a.startswith('--') and '=' not in a}
CUTOFF = float(options.get('cutoff', 8.0)) / 10.0
PRUNED = float(options.get('pruned', 8.6)) / 10.0
RADIUS = 0.10901
OPC_A = 0.14772060947071208
MODELS = {
    'tip3p': dict(frame=[0, 1, 2], sites=[], weights=[], extent=0.09582,
                  oh=0.09572, hh=0.15136),
    'opc': dict(frame=[0, 1, 2], sites=[3], weights=[1 - 2 * OPC_A, OPC_A, OPC_A],
                extent=0.0873433, oh=0.0872433, hh=0.1371205),
}
TRICLINIC = name in ('dodecahedron', 'mixture')


def without_bonded(top):
    """The topology with the force constants of its bonds, angles, and
    dihedrals 0 and no pairs of 1-4: its forces are those of the pair term,
    with the same exclusions (the bonds still make them)."""
    out, section = [], None
    column = {'bonds': 4, 'angles': 5, 'dihedrals': 6}
    for line in top.splitlines():
        stripped = line.split(';')[0].strip()
        if stripped.startswith('['):
            section = stripped.strip('[] ')
            out.append(line)
            continue
        if not stripped or stripped.startswith('#'):
            out.append(line)
            continue
        if section == 'pairs':
            continue
        fields = stripped.split()
        if section in column and len(fields) > column[section]:
            fields[column[section]] = '0.0'
            line = ' '.join(fields)
        out.append(line)
    return '\n'.join(out) + '\n'


def common(steps):
    pairlist = float(options.get('pairlist', 9.0))
    if '--forces' in flags:
        return f'''
[output]
energy_interval = 1
checkpoint = "out.h5"
checkpoint_interval = 1
[energy]
cutoff = {10 * CUTOFF}
pairlist_distance = {pairlist}
pruned_distance = {10 * PRUNED}
electrostatics = "CUTOFF"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.002
steps = 1
[constraints]
rigid_water = true
hydrogen_bonds = true
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "DOUBLE"
neighbor_structure = "GROUPS"
'''
    return f'''
[output]
energy_interval = {steps}
trajectory = "traj.dcd"
trajectory_interval = 1
[energy]
cutoff = {10 * CUTOFF}
pairlist_distance = {pairlist}
pruned_distance = {10 * PRUNED}
electrostatics = "PME"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.002
steps = {steps}
[constraints]
rigid_water = true
hydrogen_bonds = true
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
neighbor_structure = "GROUPS"
[thermostat]
method = "V-RESCALE"
time_constant = 1.0
interval = 1
'''


def ensemble(npt):
    if '--forces' in flags:
        return '[ensemble]\nensemble = "NVE"\ntemperature = 300.0\n'
    if npt:
        return ('[ensemble]\nensemble = "NPT"\ntemperature = 300.0\npressure = 1.0\n'
                '[barostat]\nmethod = "C-RESCALE"\ntime_constant = 0.2\ninterval = 1\n')
    return '[ensemble]\nensemble = "NVT"\ntemperature = 300.0\n'


def dodecahedron_rows(copies):
    """The waters of Inputs/triclinic replicated `copies` times along each
    lattice vector: rows (O, H1, H2) of positions, and the cell line."""
    source = (root / 'triclinic' / 'dodecahedron.gro').read_text().splitlines()
    count = int(source[1])
    box = [float(v) for v in source[2 + count].split()]
    a, b, c = (box[0], box[3], box[4]), (box[5], box[1], box[6]), (box[7], box[8], box[2])
    rows = [[float(r[20 + 8 * n:28 + 8 * n]) for n in range(3)] for r in source[2:2 + count]]
    waters = []
    for i in range(copies):
        for j in range(copies):
            for k in range(copies):
                shift = np.array([i * a[n] + j * b[n] + k * c[n] for n in range(3)])
                for w in range(0, count, 3):
                    waters.append(np.array(rows[w:w + 3]) + shift)
    return waters, ''.join(f'{copies * v:10.5f}' for v in box)


def opc_geometry(water):
    """An OPC water at the place of a TIP3P one: its O, H1 and H2 in the
    plane and on the bisector of the TIP3P's, at OPC's lengths, and its site
    at the weights of the frame."""
    o, h1, h2 = water
    bisector = (h1 + h2) / 2 - o
    bisector /= np.linalg.norm(bisector)
    across = h1 - h2
    across -= (across @ bisector) * bisector
    across /= np.linalg.norm(across)
    m = MODELS['opc']
    half = math.asin(m['hh'] / (2 * m['oh']))
    p1 = o + m['oh'] * (math.cos(half) * bisector + math.sin(half) * across)
    p2 = o + m['oh'] * (math.cos(half) * bisector - math.sin(half) * across)
    site = (1 - 2 * OPC_A) * o + OPC_A * p1 + OPC_A * p2
    return [o, p1, p2, site]


def write_gro(path, title, molecules, box):
    rows = [title, f'{sum(len(atoms) for _, atoms in molecules):5d}']
    serial = 0
    for residue, (resname, atoms) in enumerate(molecules, 1):
        for atom, position in atoms:
            serial += 1
            rows.append(f'{residue % 100000:5d}{resname:<5}{atom:>5}{serial % 100000:5d}'
                        + ''.join(f'{v:8.3f}' for v in position))
    rows.append(box)
    Path(path).write_text('\n'.join(rows) + '\n')


MIXTURE_TOP = '''[ defaults ]
  1  2  yes  0.5  0.8333

[ atomtypes ]
  OW  8   15.9994  0.0  A  0.315061  0.636386
  HW  1   1.008    0.0  A  0.0       0.0
  OP  8   15.9994  0.0  A  0.316655  0.890359
  HP  1   1.008    0.0  A  0.0       0.0
  EP  0   0.0      0.0  D  0.0       0.0
  NA  11  22.99    0.0  A  0.24      0.37

[ moleculetype ]
  SOL  2
[ atoms ]
  1  OW  1  SOL  OW   1  -0.834  15.9994
  2  HW  1  SOL  HW1  1  0.417  1.008
  3  HW  1  SOL  HW2  1  0.417  1.008
[ settles ]
  1  1  0.09572  0.15136
[ exclusions ]
  1  2  3
  2  1  3
  3  1  2

[ moleculetype ]
  OPC  2
[ atoms ]
  1  OP  1  OPC  O   1  0.0  15.9994
  2  HP  1  OPC  H1  1  0.679142  1.008
  3  HP  1  OPC  H2  1  0.679142  1.008
  4  EP  1  OPC  EP  1  -1.358284  0.0
[ settles ]
  1  1  0.0872433  0.1371205
[ virtual_sites3 ]
  4  1  2  3  1  {a!r}  {a!r}
[ exclusions ]
  1  2  3  4
  2  1  3  4
  3  1  2  4
  4  1  2  3

[ moleculetype ]
  NA  1
[ atoms ]
  1  NA  1  NA  NA  1  1.0  22.99

[ system ]
waters of two kinds and ions

[ molecules ]
  SOL  {sol}
  OPC  {opc}
  NA   {na}
'''


def system():
    steps = int(sys.argv[5])
    if name == 'dodecahedron':
        copies = 1 if '--cell' in flags else 2
        waters, box = dodecahedron_rows(copies)
        names = ('OW', 'HW1', 'HW2')
        write_gro(work / 'system.gro', 'dodecahedron of TIP3P water',
                  [('SOL', list(zip(names, w))) for w in waters], box)
        top = (root / 'triclinic' / 'water.top').read_text()
        (work / 'system.top').write_text(top.replace('SOL 403', f'SOL {len(waters)}'))
        npt = '--cell' not in flags or '--npt' in flags
        (work / 'control.toml').write_text(
            f'[input]\ntopology = "{work / "system.top"}"\ncoordinates = "{work / "system.gro"}"\n'
            + ensemble(npt) + common(steps))
    elif name == 'mixture':
        waters, box = dodecahedron_rows(2)
        ions = {len(waters) * k // 4 + 7 for k in range(4)}
        sol, opc, na = [], [], []
        for k, w in enumerate(waters):
            if k in ions:
                na.append(('NA', [('NA', w[0])]))
            elif k % 3 == 1:
                opc.append(('OPC', list(zip(('O', 'H1', 'H2', 'EP'), opc_geometry(w)))))
            else:
                sol.append(('SOL', list(zip(('OW', 'HW1', 'HW2'), w))))
        write_gro(work / 'system.gro', 'waters of two kinds and ions', sol + opc + na, box)
        (work / 'system.top').write_text(MIXTURE_TOP.format(a=OPC_A, sol=len(sol), opc=len(opc), na=len(na)))
        (work / 'control.toml').write_text(
            f'[input]\ntopology = "{work / "system.top"}"\ncoordinates = "{work / "system.gro"}"\n'
            + ensemble('--forces' not in flags) + common(steps))
    elif name == 'dipeptide-gro':
        top = (root / 'dipeptide' / 'dipeptide.top').read_text()
        if '--forces' in flags:
            top = without_bonded(top)
        (work / 'system.top').write_text(top)
        # The coordinates in the columns of 8.3 that the oracle's reader takes.
        rows = (root / 'dipeptide' / 'dipeptide.gro').read_text().splitlines()
        count = int(rows[1])
        out = rows[:2]
        for row in rows[2:2 + count]:
            fields = row[20:].split()
            out.append(row[:20] + ''.join(f'{float(v):8.3f}' for v in fields[:3]))
        out.append(rows[2 + count])
        (work / 'system.gro').write_text('\n'.join(out) + '\n')
        (work / 'control.toml').write_text(
            f'[input]\ntopology = "{work / "system.top"}"\ncoordinates = "{work / "system.gro"}"\n'
            + ensemble(False) + common(steps))
    elif name == 'growth':
        growth()
    else:
        files = root / ('opc/opc' if name == 'dipeptide-opc' else 'dipeptide/dipeptide')
        (work / 'control.toml').write_text(
            f'[input]\ntopology = "{files}.prmtop"\ncoordinates = "{files}.inpcrd"\n'
            + ensemble('--npt' in flags) + common(steps))


def growth():
    """Two frames of a cubic cell of 4 nm: waters A and B with their oxygens
    1.092 nm apart, beyond their covering reach 0.9 + 2 (0.09582), and a
    molecule of two atoms, its second anchored to its first, 1.5 nm away.
    At the second frame the cell and every position are scaled by 1.5, and
    A and B are moved towards each other by d = 0.326 nm each: more than the
    least half-margin of the segments, (1.5 (0.9 + E) - (0.8 + E)) / 2 at E =
    0.19164 (the waters'), 0.3229, so the list must be built again, which
    brings A and B within 0.8 + E of each other; less than that of the
    largest radii, at E = 0.21802 (the rest's), 0.3295."""
    m = MODELS['tip3p']
    half = math.asin(m['hh'] / (2 * m['oh']))

    def water(o, up):
        o = np.array(o)
        h1 = o + m['oh'] * np.array([0.0, math.cos(half), up * math.sin(half)])
        h2 = o + m['oh'] * np.array([0.0, math.cos(half), -up * math.sin(half)])
        return [o, h1, h2]
    a = water([1.454, 1.0, 1.0], 1.0)
    b = water([1.454 + 1.092, 1.0, 1.0], -1.0)
    rest = [np.array([2.0, 2.5, 2.6]), np.array([2.0, 2.5, 2.6 + 0.109])]
    first = np.array(a + b + rest)
    second = first * 1.5
    d = 0.326
    second[0:3] += [d, 0.0, 0.0]
    second[3:6] -= [d, 0.0, 0.0]
    np.save(work / 'frames.npy', np.array([first, second], np.float32))
    np.save(work / 'cells.npy', np.array([[4.0, 4.0, 4.0, 0, 0, 0], [6.0, 6.0, 6.0, 0, 0, 0]]))


def topology():
    """The taken kinds, as (model, tuples of their places), the anchors of
    the rest, its excluded pairs, and the number of particles."""
    if name == 'dodecahedron':
        n = int((work / 'system.gro').read_text().splitlines()[1])
        return [('tip3p', np.arange(n, dtype=np.int64).reshape(-1, 3))], \
            np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64), n
    if name == 'mixture':
        rows = (work / 'system.gro').read_text().splitlines()
        n = int(rows[1])
        resnames = [r[5:10].strip() for r in rows[2:2 + n]]
        sol = [k for k, r in enumerate(resnames) if r == 'SOL']
        opc = [k for k, r in enumerate(resnames) if r == 'OPC']
        return [('tip3p', np.array(sol, np.int64).reshape(-1, 3)),
                ('opc', np.array(opc, np.int64).reshape(-1, 4))], \
            np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64), n
    if name == 'growth':
        return [('tip3p', np.arange(6, dtype=np.int64).reshape(-1, 3))], \
            np.array([[7, 6]], np.int64), np.array([[6, 7]], np.int64), 8
    import parmed
    # dipeptide-gro: the GROMACS files of the same system, in the same order.
    files = root / ('opc/opc' if name == 'dipeptide-opc' else 'dipeptide/dipeptide')
    parm = parmed.load_file(f'{files}.prmtop')
    n = len(parm.atoms)
    waters = [[a.idx for a in r.atoms] for r in parm.residues
              if r.name in ('WAT', 'HOH', 'OPC', 'TP3', 'SOL') and r.atoms[0].name.startswith('O')]
    inwater = np.zeros(n, bool)
    inwater[[i for w in waters for i in w]] = True
    anchors = []
    for atom in parm.atoms:
        if not inwater[atom.idx] and atom.element == 1:
            heavy = [b for b in atom.bond_partners if b.element != 1]
            if heavy:
                anchors.append((atom.idx, heavy[0].idx))
    pairs = set()
    k = 0
    excluded = parm.parm_data['EXCLUDED_ATOMS_LIST']
    for i, count in enumerate(parm.parm_data['NUMBER_EXCLUDED_ATOMS']):
        for j in excluded[k:k + count]:
            j -= 1
            if j >= 0 and not inwater[i] and not inwater[j]:
                pairs.add((min(i, j), max(i, j)))
        k += count
    model = 'opc' if len(waters[0]) == 4 else 'tip3p'
    return [(model, np.array(waters, np.int64))], np.array(anchors, np.int64).reshape(-1, 2), \
        np.array(sorted(pairs), np.int64).reshape(-1, 2), n


def frames(count=None):
    """The positions (nm) and the cells (a_x, b_y, c_z, b_x, c_x, c_y) of the
    trajectory: DCD, or the frames that `system` wrote."""
    if (work / 'cells.npy').exists() and not (work / 'traj.dcd').exists():
        return np.load(work / 'frames.npy'), np.load(work / 'cells.npy')
    data = (work / 'traj.dcd').read_bytes()
    offset = 0

    def record():
        nonlocal offset
        size, = struct.unpack_from('<i', data, offset)
        body = data[offset + 4:offset + 4 + size]
        offset += size + 8
        return body
    header = record()
    has_cell = struct.unpack_from('<20i', header, 4)[10] == 1
    assert has_cell
    record()
    record()
    positions, cells = [], []
    while offset < len(data) and (count is None or len(positions) < count):
        a, cg, b, cb, ca, c = struct.unpack('<6d', record())
        xyz = [np.frombuffer(record(), np.float32) for _ in range(3)]
        positions.append(np.stack(xyz, 1) / np.float32(10.0))
        sg = math.sqrt(1 - cg * cg)
        cx = c * cb
        cy = c * (ca - cb * cg) / sg
        cz = math.sqrt(c * c - cx * cx - cy * cy)
        cells.append(np.array([a, b * sg, cz, b * cg, cx, cy]) / 10.0)
    return np.array(positions, np.float32), np.array(cells)


def gro_cell():
    box = [float(v) for v in (work / 'system.gro').read_text().splitlines()[-1].split()]
    box += [0.0] * (9 - len(box))
    # v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
    return np.array([box[0], box[1], box[2], box[5], box[7], box[8]])


if mode == 'system':
    system()
    sys.exit(0)

kinds, anchors, excluded, N = topology()
K = len(kinds)
S = (K + 1) * (K + 2) // 2
# Kind f of the type is kinds[f]; the rest is kind K. Rank 0 the rest, rank
# K - f kind f.
kind_of = np.full(N, K, np.int64)
for f, (model, tuples) in enumerate(kinds):
    kind_of[tuples.ravel()] = f
rest = np.nonzero(kind_of == K)[0]
anchor_of = {int(m): int(a) for m, a in anchors}
# The units in the order of the lowering: the rest, then the kinds from K - 1
# to 0; each unit's reference particle and kind.
unit_ref, unit_kind, unit_first = [], [], {}
for p in rest:
    unit_ref.append(anchor_of.get(int(p), int(p)))
    unit_kind.append(K)
for f in reversed(range(K)):
    for tup in kinds[f][1]:
        unit_first[int(tup[0])] = len(unit_ref)
        unit_ref.append(int(tup[0]))
        unit_kind.append(f)
unit_ref, unit_kind = np.array(unit_ref), np.array(unit_kind)
U, R0 = len(unit_ref), len(rest)
unit_of_rest = {int(p): k for k, p in enumerate(rest)}
radius = RADIUS if len(anchors) else 0.0
extent = [MODELS[model]['extent'] for model, _ in kinds] + [radius]
pairlist = float(options.get('pairlist', 9.0)) / 10.0
skin, prune_skin = pairlist - CUTOFF, PRUNED - CUTOFF

if mode == 'emit':
    if 'positions' in options:
        # One frame: the positions that `mdir checkpoint --print=positions`
        # prints (particle, mass, x, y, z in nm), in the cell of the input.
        rows = [line.split() for line in Path(options['positions']).read_text().splitlines() if line.strip()]
        pos = np.array([[[float(r[2]), float(r[3]), float(r[4])] for r in rows]], np.float32)
        cells = np.array([gro_cell()])
    else:
        pos, cells = frames(int(options['frames']) if 'frames' in options else None)
    T = len(pos)
    np.save(work / 'frames.npy', pos)
    np.save(work / 'cells.npy', cells)
    np.save(work / 'excluded.npy', excluded)
    if '--drop-exclusion' in flags:
        excluded = excluded[1:]
    if '--join-exclusion' in flags:
        excluded = np.concatenate([excluded, [[int(rest[0]), int(kinds[0][1][0][0])]]])
    refs = np.unique(unit_ref)
    M = len(refs)

    def hexed(array):
        # The bytes of the array, little endian, as MLIR reads a dense
        # literal in hexadecimal.
        return '"0x' + np.ascontiguousarray(array).tobytes().hex().upper() + '"'
    descriptors = []
    for model, tuples in kinds:
        m = MODELS[model]
        descriptors.append(f'#md.rigid_kind<frame = {m["frame"]}, sites = {m["sites"]}, '
                           f'weights = [{", ".join(f"{w:.17g}" for w in (m["weights"] if m["sites"] else []))}], '
                           f'extent = {m["extent"]:.17e}>')
    seg = 'unique, minimum, removed, grouped_by(group)'
    segments = ', '.join(f'({f}, {g}) {seg}' for f in range(K + 1) for g in range(f + 1))
    sp = f'!mdrt.segmented_pairs<@atoms, kinds [{", ".join(descriptors)}], segments [{segments}]>'
    nc = 6 if TRICLINIC else 3
    anchors_ = anchors if len(anchors) else np.zeros((1, 2), np.int64)
    excluded_ = excluded if len(excluded) else np.array([kinds[0][1][0][:2]])
    cellmake = ('md.triclinic_cell %c0v, %c1v, %c2v, %c3v, %c4v, %c5v' if TRICLINIC else
                'md.orthorhombic_cell %c0v, %c1v, %c2v')
    anchor = f'anchor(%da : memref<?x2xi32, 1>) radius({RADIUS})' if len(anchors) else ''
    lines = ['module {',
             f'  memref.global "private" constant @frames : memref<{T}x{M}x3xf32> = dense<{hexed(pos[:, refs, :].astype("<f4"))}>',
             f'  memref.global "private" constant @cells : memref<{T}x{nc}xf64> = dense<{hexed(cells[:, :nc].astype("<f8"))}>',
             f'  memref.global "private" constant @refs : memref<{M}xi32> = dense<{hexed(refs.astype("<i4"))}>',
             f'  memref.global "private" constant @anchors : memref<{len(anchors_)}x2xi32> = dense<{hexed(anchors_.astype("<i4"))}>',
             f'  memref.global "private" constant @excluded : memref<{len(excluded_)}x2xi32> = dense<{hexed(excluded_.astype("<i4"))}>']
    for f, (model, tuples) in enumerate(kinds):
        lines.append(f'  memref.global "private" constant @tuples{f} : memref<{len(tuples)}x{tuples.shape[1]}xi32> = '
                     f'dense<{hexed(tuples.astype("<i4"))}>')
    body = [f'''  func.func @main() {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %N = arith.constant {N} : index
    %T = arith.constant {T} : index
    %M = arith.constant {M} : index
    %zero = arith.constant 0.0 : f64
    %fr = memref.get_global @frames : memref<{T}x{M}x3xf32>
    %ce = memref.get_global @cells : memref<{T}x{nc}xf64>
    %rf = memref.get_global @refs : memref<{M}xi32>''']

    def device(name, global_, shape):
        rank_two = 'x' in shape
        dynamic = f'?x{shape.split("x", 1)[1]}' if rank_two else '?xi32'
        static = shape
        return f'''    %g_{name} = memref.get_global @{global_} : memref<{static}>
    %h_{name} = memref.cast %g_{name} : memref<{static}> to memref<{dynamic}>
    %n_{name} = memref.dim %h_{name}, %c0 : memref<{dynamic}>
    %{name} = gpu.alloc (%n_{name}) : memref<{dynamic}, 1>
    %w0_{name} = gpu.wait async
    %w1_{name} = gpu.memcpy async [%w0_{name}] %{name}, %h_{name} : memref<{dynamic}, 1>, memref<{dynamic}>
    gpu.wait [%w1_{name}]'''
    body.append(device('da', 'anchors', f'{len(anchors_)}x2xi32'))
    body.append(device('dx', 'excluded', f'{len(excluded_)}x2xi32'))
    for f, (model, tuples) in enumerate(kinds):
        body.append(device(f'dk{f}', f'tuples{f}', f'{len(tuples)}x{tuples.shape[1]}xi32'))
    kinds_operands = ', '.join(f'%dk{f}' for f in range(K))
    kinds_types = ', '.join(f'memref<?x{t.shape[1]}xi32, 1>' for _, t in kinds)
    body.append(f'''    %hu = memref.alloc(%N) : memref<?xi32>
    %du = gpu.alloc (%N) : memref<?xi32, 1>
    %hp = memref.alloc(%N) : memref<?x3xf64>
    scf.for %i = %c0 to %N step %c1 {{
      memref.store %zero, %hp[%i, %c0] : memref<?x3xf64>
      memref.store %zero, %hp[%i, %c1] : memref<?x3xf64>
      memref.store %zero, %hp[%i, %c2] : memref<?x3xf64>
    }}
    %dp = gpu.alloc (%N) : memref<?x3xf64, 1>
    %l0 = md_exec.empty_segmented_pairs %du kinds({kinds_operands} : {kinds_types}) exclude(%dx : memref<?x2xi32, 1>) {anchor} cutoff({CUTOFF}) skin({skin:.9f}) : memref<?xi32, 1> -> {sp}
    %lend = scf.for %t = %c0 to %T step %c1 iter_args(%l = %l0) -> ({sp}) {{
      scf.for %m = %c0 to %M step %c1 {{
        %r32 = memref.load %rf[%m] : memref<{M}xi32>
        %r = arith.index_cast %r32 : i32 to index
        %x0 = memref.load %fr[%t, %m, %c0] : memref<{T}x{M}x3xf32>
        %x1 = memref.load %fr[%t, %m, %c1] : memref<{T}x{M}x3xf32>
        %x2 = memref.load %fr[%t, %m, %c2] : memref<{T}x{M}x3xf32>
        %y0 = arith.extf %x0 : f32 to f64
        %y1 = arith.extf %x1 : f32 to f64
        %y2 = arith.extf %x2 : f32 to f64
        memref.store %y0, %hp[%r, %c0] : memref<?x3xf64>
        memref.store %y1, %hp[%r, %c1] : memref<?x3xf64>
        memref.store %y2, %hp[%r, %c2] : memref<?x3xf64>
      }}
      %w8 = gpu.wait async
      %w9 = gpu.memcpy async [%w8] %dp, %hp : memref<?x3xf64, 1>, memref<?x3xf64>
      gpu.wait [%w9]''')
    for k in range(nc):
        body.append(f'''      %k{k} = arith.constant {k} : index
      %c{k}v = memref.load %ce[%t, %k{k}] : memref<{T}x{nc}xf64>''')
    body.append(f'''      %cell = {cellmake}
      %l1 = md_exec.refresh_segmented_pairs %l at(%dp, %cell) cutoff({CUTOFF}) skin({skin:.9f}) prune_skin({prune_skin:.9f}) : {sp}, memref<?x3xf64, 1>
      scf.yield %l1 : {sp}
    }}
    return
  }}
}}''')
    (work / 'harness.mlir').write_text('\n'.join(lines + body) + '\n')
    print(f'{name}: {N} particles, {K} taken kinds of {", ".join(str(len(t)) for _, t in kinds)} copies, '
          f'{R0} units of the rest, {T} frames')
    sys.exit(0)

# check
from scipy.spatial import cKDTree  # noqa: E402
dumps = Path(sys.argv[5])
scale = float(options.get('scale', 1.0))
pos = np.load(work / 'frames.npy').astype(np.float64)
cells = np.load(work / 'cells.npy')
names = ['order', 'shifts', 'entries', 'masks', 'groups', 'counts', 'offsets',
         'inner_entries', 'inner_masks', 'inner_counts', 'active', 'active_count', 'ref_order']


def read_dump(path):
    raw = path.read_bytes()
    head = np.frombuffer(raw[:48], np.int64)
    offset, parts = 48, {}
    for part in names:
        length, = struct.unpack_from('<q', raw, offset)
        offset += 8
        parts[part] = np.frombuffer(raw[offset:offset + 4 * length], np.int32)
        offset += 4 * length
    return head, parts


def matrix(cell):
    ax, by, cz, bx, cx, cy = cell if TRICLINIC else (cell[0], cell[1], cell[2], 0, 0, 0)
    return np.array([[ax, 0, 0], [bx, by, 0], [cx, cy, cz]])


segkind = [(f, g) for f in range(K + 1) for g in range(f + 1)]
wbits, woff = (5, 16) if TRICLINIC else (4, 4)
excluded_units = {(min(unit_of_rest[int(a)], unit_of_rest[int(b)]), max(unit_of_rest[int(a)], unit_of_rest[int(b)]))
                  for a, b in excluded if int(a) in unit_of_rest and int(b) in unit_of_rest}
slot_of = {}
base = 0
for f, (model, tuples) in enumerate(kinds):
    for c, tup in enumerate(tuples):
        slot_of[unit_first[int(tup[0])]] = base + 3 * c
    base += 3 * len(tuples)


def listed(parts, inner, faults):
    """The pairs of the records, as (unit, unit, image) with their counts;
    `faults` counts the entries in a wrong segment, of a place whose order
    disagrees with its unit, of an excluded pair, and of a pair within one
    unit."""
    entries = parts['inner_entries' if inner else 'entries']
    masks = parts['inner_masks' if inner else 'masks']
    counts = parts['inner_counts' if inner else 'counts']
    offsets = parts['offsets']
    records = parts['active'][:parts['active_count'][0]] if inner else range(len(parts['groups']))
    out = {}

    def unit(place):
        ref, value = int(parts['ref_order'][place]), int(parts['order'][place])
        f = int(kind_of[ref])
        u = unit_of_rest.get(value, -1) if f == K else unit_first.get(ref, -1)
        if u < 0 or (f < K and slot_of[u] != value) or (f == K and unit_ref[u] != ref):
            faults['order'] += 1
        return u, f

    def cells_of(packed):
        return np.array([(packed >> (10 * a)) & 1023 for a in range(3)]) - 512
    for w in records:
        w = int(w)
        s = max(t for t in range(S) if w >= offsets[t])
        f_seg, g_seg = segkind[s]
        group = int(parts['groups'][w])
        for e in range(int(counts[w])):
            q, m = int(entries[64 * w + e]), int(masks[64 * w + e])
            eshift = np.array([(m >> (16 + wbits * a)) & ((1 << wbits) - 1) for a in range(3)]) - woff
            uq, gq = unit(q)
            kq = cells_of(int(parts['shifts'][q]))
            for u in range(16):
                if not (m >> u) & 1:
                    continue
                p = 16 * group + u
                up, fp = unit(p)
                if (fp, gq) != (f_seg, g_seg):
                    faults['segment'] += 1
                n = tuple(int(v) for v in kq + eshift - cells_of(int(parts['shifts'][p])))
                key = (up, uq, n) if up < uq else (uq, up, tuple(-v for v in n))
                if not any(n) and (up == uq or (key[0], key[1]) in excluded_units):
                    faults['excluded'] += 1
                out[key] = out.get(key, 0) + 1
    return out


def expected(x, H, reach_of):
    X = x[unit_ref]
    rmax = max(reach_of.values())
    tree = cKDTree(X)
    out = set()
    for img in [(a, b, c) for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)]:
        shift = np.array(img) @ H
        other = cKDTree(X + shift)
        for i, js in enumerate(tree.query_ball_tree(other, rmax)):
            for j in js:
                if j <= i:
                    continue
                d = X[j] + shift - X[i]
                segment = (max(unit_kind[i], unit_kind[j]), min(unit_kind[i], unit_kind[j]))
                if math.sqrt(d @ d) >= reach_of[segment]:
                    continue
                if (i, j) in excluded_units and not any(img):
                    continue
                out.add((i, j, img))
    return out


def reaches(r):
    return {(f, g): r + extent[f] + extent[g] for f, g in segkind}


paths = sorted(dumps.glob('dump_*.bin'))
heads = [read_dump(p)[0] for p in paths]
limit = int(options.get('limit', len(paths)))
faults = {'order': 0, 'segment': 0, 'excluded': 0, 'twice': 0}
missing_all, built_after, before_build, before_prune, checked = 0, [], [], [], 0
for k, path in enumerate(paths[:limit]):
    head, parts = read_dump(path)
    x, H = pos[k], matrix(cells[k])
    built = k == 0 or head[4] > heads[k - 1][4]
    pruned = k == 0 or head[5] > heads[k - 1][5]
    if k + 1 < len(paths):
        if heads[k + 1][4] > head[4] and not built:
            before_build.append(k)
        elif heads[k + 1][5] > head[5] and not pruned:
            before_prune.append(k)
    if built:
        built_after.append(k)
    dual = len(parts['inner_counts']) > 0
    outer = listed(parts, False, faults)
    faults['twice'] += sum(1 for v in outer.values() if v > 1)
    have = outer
    if dual:
        have = listed(parts, True, faults)
        faults['twice'] += sum(1 for v in have.values() if v > 1)
    missing = len(expected(x, H, reaches(scale * CUTOFF)) - set(have))
    if built and scale == 1.0:
        # Right after a build: the outer records hold the covering reach.
        missing += len(expected(x, H, reaches(pairlist)) - set(outer))
    missing_all += missing
    checked += 1
print(f'{name}: {checked} refreshes checked, {len(built_after)} right after a build, '
      f'{len(before_build)} just before a build, {len(before_prune)} just before a pruning; '
      f'cutoff {scale * CUTOFF:.3g} nm: {missing_all} pairs without an entry; '
      f'{faults["twice"]} pairs listed twice, {faults["segment"]} in a wrong segment, '
      f'{faults["excluded"]} excluded, {faults["order"]} places of a wrong order')
bad = faults['twice'] + faults['segment'] + faults['excluded'] + faults['order']
if '--expect-missing' in flags:
    sys.exit(0 if missing_all > 0 and bad == 0 else 1)
if '--any-points' not in flags:
    assert before_build and before_prune and len(built_after) > 1, 'a point of the gate was not reached'
sys.exit(0 if missing_all == 0 and bad == 0 else 1)
