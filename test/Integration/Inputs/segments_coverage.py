"""The coverage of the lists of units that the device builds
(D[md-alg-segments-build]; D417, Section 8, the gate of S2), against every
image of every pair of reference points in f64.

  segments_coverage.py system KIND ROOT WORK STEPS [--cell] [--pairlist=Å]
      KIND dodecahedron (TIP3P in a rhombic dodecahedron, NPT), dipeptide-opc
      or dipeptide-tip3p (alanine dipeptide in water, NVT): WORK/control.toml,
      a general run of groups whose trajectory (DCD, every step) the harness
      replays. --cell keeps the small dodecahedron of Inputs/triclinic.
  segments_coverage.py emit KIND ROOT WORK [--frames N]
      WORK/harness.mlir: an empty list of units and, for each frame of
      WORK/traj.dcd, a refresh at its positions and cell
      (md-exec-reuse-neighbors's form), for convert-md-exec-to-gpu with
      dump-segments.
  segments_coverage.py check KIND ROOT WORK DUMPS [--scale=S] [--limit=L] [--expect-missing]
      Each dump is the list after the refresh of a frame. Its inner records
      (the outer ones without pruning) must hold every pair of reference
      points within r_c + e_f + e_g of its segment at that frame, which every
      site pair within r_c needs; right after a build, its outer records
      every pair within the covering reach R_s. Prints the counts, the frames
      just before a pruning and just before a build, and those right after a
      build. With --scale=S the cutoff is S r_c, and --expect-missing
      requires that pairs be reported without an entry; --limit=L checks the
      first L dumps.

Lengths in nm. The covering radii: the water's (the TIP3P O-H length, 0.09572
nm, OPC's 0.0873433 nm) and the rest's anchors (0.10901 nm, D417's rho).
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np

mode = sys.argv[1]
kind = sys.argv[2]
root, work = Path(sys.argv[3]), Path(sys.argv[4])
options = dict(a[2:].split('=', 1) for a in sys.argv[5:] if a.startswith('--') and '=' in a)
flags = {a for a in sys.argv[5:] if a.startswith('--') and '=' not in a}
CUTOFF, PRUNED = 0.8, 0.86
RADIUS = 0.10901


def system():
    steps = int(sys.argv[5])
    pairlist = float(options.get('pairlist', 9.0))
    common = f'''
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
    if kind == 'dodecahedron':
        source = (root / 'triclinic' / 'dodecahedron.gro').read_text().splitlines()
        count = int(source[1])
        box = [float(v) for v in source[2 + count].split()]
        a, b, c = (box[0], box[3], box[4]), (box[5], box[1], box[6]), (box[7], box[8], box[2])
        copies = 1 if '--cell' in flags else 2
        out = [f'dodecahedron of TIP3P water, {copies} cells along each vector', f'{copies ** 3 * count:5d}']
        serial = 0
        for i in range(copies):
            for j in range(copies):
                for k in range(copies):
                    shift = [i * a[n] + j * b[n] + k * c[n] for n in range(3)]
                    for row in source[2:2 + count]:
                        xyz = [float(row[20 + 8 * n:28 + 8 * n]) + shift[n] for n in range(3)]
                        serial += 1
                        out.append(f'{((serial - 1) // 3 + 1) % 100000:5d}{"SOL":<5}{row[10:15]:>5}{serial % 100000:5d}'
                                   + ''.join(f'{v:8.3f}' for v in xyz))
        out.append(''.join(f'{copies * v:10.5f}' for v in box))
        (work / 'system.gro').write_text('\n'.join(out) + '\n')
        top = (root / 'triclinic' / 'water.top').read_text()
        (work / 'system.top').write_text(top.replace('SOL 403', f'SOL {copies ** 3 * count // 3}'))
        npt = '--cell' not in flags
        (work / 'control.toml').write_text(f'''
[input]
topology = "{work / 'system.top'}"
coordinates = "{work / 'system.gro'}"
[ensemble]
ensemble = "{'NPT' if npt else 'NVT'}"
temperature = 300.0
{'pressure = 1.0' if npt else ''}
{'[barostat]' if npt else ''}
{'method = "C-RESCALE"' if npt else ''}
{'time_constant = 0.2' if npt else ''}
{'interval = 1' if npt else ''}
''' + common)
    else:
        files = root / ('opc/opc' if kind == 'dipeptide-opc' else 'dipeptide/dipeptide')
        (work / 'control.toml').write_text(f'''
[input]
topology = "{files}.prmtop"
coordinates = "{files}.inpcrd"
[ensemble]
ensemble = "NVT"
temperature = 300.0
''' + common)


def topology():
    """The waters (tuples of the kind's places), the anchors of the rest,
    the excluded pairs of the rest, the particles, and the kind."""
    if kind == 'dodecahedron':
        gro = (work / 'system.gro').read_text().splitlines()
        n = int(gro[1])
        waters = np.arange(n, dtype=np.int64).reshape(-1, 3)
        return waters, np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64), n, 'tip3p'
    import parmed
    files = root / ('opc/opc' if kind == 'dipeptide-opc' else 'dipeptide/dipeptide')
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
    return (np.array(waters, np.int64), np.array(anchors, np.int64).reshape(-1, 2),
            np.array(sorted(pairs), np.int64).reshape(-1, 2), n, 'opc' if len(waters[0]) == 4 else 'tip3p')


def frames(count=None):
    """The positions (nm) and the cells (a_x, b_y, c_z, b_x, c_x, c_y) of the
    trajectory, as f32 and f64."""
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
    record()
    n, = struct.unpack_from('<i', record(), 0)
    positions, cells = [], []
    while offset < len(data) and (count is None or len(positions) < count):
        cell = struct.unpack('<6d', record()) if has_cell else None
        xyz = [np.frombuffer(record(), np.float32) for _ in range(3)]
        positions.append(np.stack(xyz, 1) / np.float32(10.0))
        a, cg, b, cb, ca, c = cell
        sg = math.sqrt(1 - cg * cg)
        bx, by = b * cg, b * sg
        cx = c * cb
        cy = c * (ca - cb * cg) / sg
        cz = math.sqrt(c * c - cx * cx - cy * cy)
        cells.append(np.array([a, by, cz, bx, cx, cy]) / 10.0)
    return np.array(positions, np.float32), np.array(cells)


def units():
    waters, anchors, excluded, n, model = topology()
    inwater = np.zeros(n, bool)
    inwater[waters.ravel()] = True
    rest = np.nonzero(~inwater)[0]
    anchor_of = {int(m): int(a) for m, a in anchors}
    ref = np.array([anchor_of.get(int(p), int(p)) for p in rest] + list(waters[:, 0]), np.int64)
    unit_of = np.full(n, -1, np.int64)
    unit_of[rest] = np.arange(len(rest))
    extent = {'tip3p': 0.09582, 'opc': 0.0873433}[model]
    return waters, anchors, excluded, n, model, rest, ref, unit_of, extent


if mode == 'system':
    system()
    sys.exit(0)

waters, anchors, excluded, N, model, rest, ref, unit_of, extent = units()
tri = kind == 'dodecahedron'
pairlist = float(options.get('pairlist', 9.0)) / 10.0
skin, prune_skin = pairlist - CUTOFF, PRUNED - CUTOFF
R0, U = len(rest), len(rest) + len(waters)
refs = np.unique(ref)

if mode == 'emit':
    pos, cells = frames(int(options['frames']) if 'frames' in options else None)
    T, M, n = len(pos), len(refs), waters.shape[1]
    np.save(work / 'frames.npy', pos)
    np.save(work / 'cells.npy', cells)
    weights = (', weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208]'
               if n == 4 else ', weights = []')
    sites = '[3]' if n == 4 else '[]'
    seg = 'unique, minimum, removed, grouped_by(group)'
    sp = (f'!mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = {sites}{weights}, '
          f'extent = {extent:.17e}>], segments [(0, 0) {seg}, (1, 0) {seg}, (1, 1) {seg}]>')
    def dense(values, fmt):
        return ', '.join(fmt % v for v in values)
    def hexed(array):
        # The bytes of the array, little endian, as MLIR reads a dense
        # literal in hexadecimal.
        return '"0x' + np.ascontiguousarray(array).tobytes().hex().upper() + '"'
    framevals = hexed(pos[:, refs, :].astype('<f4'))
    cellvals = hexed((cells if tri else cells[:, :3]).astype('<f8'))
    nc = 6 if tri else 3
    tuples = dense(waters.ravel(), '%d')
    anchors_ = anchors if len(anchors) else np.zeros((1, 2), np.int64)
    excluded_ = excluded if len(excluded) else np.array([[waters[0, 0], waters[0, 1]]])
    cellmake = (f'md.triclinic_cell %c0v, %c1v, %c2v, %c3v, %c4v, %c5v' if tri else
                f'md.orthorhombic_cell %c0v, %c1v, %c2v')
    radius = f'anchor(%da : memref<?x2xi32, 1>) radius({RADIUS})' if len(anchors) else ''
    text = f'''module {{
  memref.global "private" constant @frames : memref<{T}x{M}x3xf32> = dense<{framevals}>
  memref.global "private" constant @cells : memref<{T}x{nc}xf64> = dense<{cellvals}>
  memref.global "private" constant @refs : memref<{M}xi32> = dense<{hexed(refs.astype('<i4'))}>
  memref.global "private" constant @tuples : memref<{len(waters)}x{n}xi32> = dense<{hexed(waters.astype('<i4'))}>
  memref.global "private" constant @anchors : memref<{len(anchors_)}x2xi32> = dense<{hexed(anchors_.astype('<i4'))}>
  memref.global "private" constant @excluded : memref<{len(excluded_)}x2xi32> = dense<{hexed(excluded_.astype('<i4'))}>
  func.func @main() {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %N = arith.constant {N} : index
    %T = arith.constant {T} : index
    %M = arith.constant {M} : index
    %zero = arith.constant 0.0 : f64
    %fr = memref.get_global @frames : memref<{T}x{M}x3xf32>
    %ce = memref.get_global @cells : memref<{T}x{nc}xf64>
    %rf = memref.get_global @refs : memref<{M}xi32>
    %tu = memref.get_global @tuples : memref<{len(waters)}x{n}xi32>
    %an = memref.get_global @anchors : memref<{len(anchors_)}x2xi32>
    %ex = memref.get_global @excluded : memref<{len(excluded_)}x2xi32>
    %hu = memref.alloc(%N) : memref<?xi32>
    %du = gpu.alloc (%N) : memref<?xi32, 1>
    %t0 = gpu.wait async
    %t1 = gpu.memcpy async [%t0] %du, %hu : memref<?xi32, 1>, memref<?xi32>
    gpu.wait [%t1]
    %tu1 = memref.cast %tu : memref<{len(waters)}x{n}xi32> to memref<?x{n}xi32>
    %c_tu = memref.dim %tu1, %c0 : memref<?x{n}xi32>
    %dk = gpu.alloc (%c_tu) : memref<?x{n}xi32, 1>
    %t2 = gpu.wait async
    %t3 = gpu.memcpy async [%t2] %dk, %tu1 : memref<?x{n}xi32, 1>, memref<?x{n}xi32>
    gpu.wait [%t3]
    %an1 = memref.cast %an : memref<{len(anchors_)}x2xi32> to memref<?x2xi32>
    %c_an = memref.dim %an1, %c0 : memref<?x2xi32>
    %da = gpu.alloc (%c_an) : memref<?x2xi32, 1>
    %t4 = gpu.wait async
    %t5 = gpu.memcpy async [%t4] %da, %an1 : memref<?x2xi32, 1>, memref<?x2xi32>
    gpu.wait [%t5]
    %ex1 = memref.cast %ex : memref<{len(excluded_)}x2xi32> to memref<?x2xi32>
    %c_ex = memref.dim %ex1, %c0 : memref<?x2xi32>
    %dx = gpu.alloc (%c_ex) : memref<?x2xi32, 1>
    %t6 = gpu.wait async
    %t7 = gpu.memcpy async [%t6] %dx, %ex1 : memref<?x2xi32, 1>, memref<?x2xi32>
    gpu.wait [%t7]
    %hp = memref.alloc(%N) : memref<?x3xf64>
    scf.for %i = %c0 to %N step %c1 {{
      memref.store %zero, %hp[%i, %c0] : memref<?x3xf64>
      memref.store %zero, %hp[%i, %c1] : memref<?x3xf64>
      memref.store %zero, %hp[%i, %c2] : memref<?x3xf64>
    }}
    %dp = gpu.alloc (%N) : memref<?x3xf64, 1>
    %l0 = md_exec.empty_segmented_pairs %du kinds(%dk : memref<?x{n}xi32, 1>) exclude(%dx : memref<?x2xi32, 1>) {radius} cutoff({CUTOFF}) skin({skin:.9f}) : memref<?xi32, 1> -> {sp}
    scf.for %t = %c0 to %T step %c1 {{
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
      %t8 = gpu.wait async
      %t9 = gpu.memcpy async [%t8] %dp, %hp : memref<?x3xf64, 1>, memref<?x3xf64>
      gpu.wait [%t9]
''' + ''.join(f'''      %k{k} = arith.constant {k} : index
      %c{k}v = memref.load %ce[%t, %k{k}] : memref<{T}x{nc}xf64>
''' for k in range(nc)) + f'''      %cell = {cellmake}
      %l1 = md_exec.refresh_segmented_pairs %l0 at(%dp, %cell) cutoff({CUTOFF}) skin({skin:.9f}) prune_skin({prune_skin:.9f}) : {sp}, memref<?x3xf64, 1>
    }}
    return
  }}
}}
'''
    (work / 'harness.mlir').write_text(text)
    print(f'{kind}: {N} particles, {len(waters)} waters, {R0} units of the rest, {T} frames')
    sys.exit(0)

# check
from scipy.spatial import cKDTree
dumps = Path(sys.argv[5])
scale = float(options.get('scale', 1.0))
pos = np.load(work / 'frames.npy').astype(np.float64)
cells = np.load(work / 'cells.npy')
names = ['order', 'shifts', 'entries', 'masks', 'groups', 'counts', 'offsets',
         'inner_entries', 'inner_masks', 'inner_counts', 'active', 'active_count']


def read_dump(path):
    raw = path.read_bytes()
    head = np.frombuffer(raw[:48], np.int64)
    offset, parts = 48, {}
    for name in names:
        length, = struct.unpack_from('<q', raw, offset)
        offset += 8
        parts[name] = np.frombuffer(raw[offset:offset + 4 * length], np.int32)
        offset += 4 * length
    return head, parts


def matrix(cell):
    ax, by, cz, bx, cx, cy = cell if tri else (cell[0], cell[1], cell[2], 0, 0, 0)
    return np.array([[ax, 0, 0], [bx, by, 0], [cx, cy, cz]])


e_rank = {1: RADIUS if len(anchors) else 0.0, 0: extent}   # kind 1 the rest, 0 the water
segkind = {0: (0, 0), 1: (1, 0), 2: (1, 1)}
wbits, woff = (5, 16) if tri else (4, 4)
excluded_units = {(min(unit_of[a], unit_of[b]), max(unit_of[a], unit_of[b])) for a, b in excluded}


def listed_pairs(parts, inner):
    entries = parts['inner_entries' if inner else 'entries']
    masks = parts['inner_masks' if inner else 'masks']
    counts = parts['inner_counts' if inner else 'counts']
    offsets = parts['offsets']
    records = range(len(parts['groups'])) if not inner else parts['active'][:parts['active_count'][0]]
    out = set()
    for w in records:
        w = int(w)
        s = max(t for t in range(3) if w >= offsets[t])
        f, g = segkind[s]
        group = int(parts['groups'][w])
        for e in range(int(counts[w])):
            q, m = int(entries[64 * w + e]), int(masks[64 * w + e])
            def unit(place, k):
                v = int(parts['order'][place])
                return int(unit_of[v]) if k == 1 else R0 + v // 3
            def cells_of(packed):
                return np.array([(packed >> (10 * a)) & 1023 for a in range(3)]) - 512
            eshift = np.array([(m >> (16 + wbits * a)) & ((1 << wbits) - 1) for a in range(3)]) - woff
            uq = unit(q, g)
            kq = cells_of(int(parts['shifts'][q]))
            for u in range(16):
                if not (m >> u) & 1:
                    continue
                p = 16 * group + u
                up = unit(p, f)
                n = tuple(kq + eshift - cells_of(int(parts['shifts'][p])))
                key = (up, uq, n) if up < uq else (uq, up, tuple(-np.array(n)))
                out.add(key)
    return out


def expected_pairs(x, H, reach_of):
    X = x[ref]
    kinds = np.where(np.arange(U) < R0, 1, 0)
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
                segment = (max(kinds[i], kinds[j]), min(kinds[i], kinds[j]))
                if math.sqrt(d @ d) >= reach_of[segment]:
                    continue
                if (i, j) in excluded_units and not any(img):
                    continue
                out.add((i, j, img))
    return out


def reaches(r):
    return {(f, g): r + e_rank[f] + e_rank[g] for f, g in segkind.values()}


paths = sorted(dumps.glob('dump_*.bin'))
heads = [read_dump(p)[0] for p in paths]
limit = int(options.get('limit', len(paths)))
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
    need = expected_pairs(x, H, reaches(scale * CUTOFF))
    have = listed_pairs(parts, inner=dual)
    missing = len(need - have)
    if built and scale == 1.0:
        # Right after a build: the outer records hold the covering reach.
        full = expected_pairs(x, H, reaches(pairlist))
        missing += len(full - listed_pairs(parts, inner=False))
    missing_all += missing
    checked += 1
print(f'{kind}: {checked} refreshes checked, {len(built_after)} right after a build, '
      f'{len(before_build)} just before a build, {len(before_prune)} just before a pruning; '
      f'cutoff {scale * CUTOFF:.3g} nm: {missing_all} pairs without an entry')
if '--expect-missing' in flags:
    sys.exit(0 if missing_all > 0 else 1)
assert before_build and before_prune and len(built_after) > 1, 'a point of the gate was not reached'
sys.exit(0 if missing_all == 0 else 1)
