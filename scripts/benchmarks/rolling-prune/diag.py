"""Offline diagnostics of the group lists on one frame (#357).

    diag.py FRAMES.h5md | INPCRD [GROUPS]

For a sample of groups (made as in trigger.py): the outer entries (each
particle at a place not before the group with a member within R) and the
three cases of each against the bounding box of the group (beyond R_in;
wholly within R_in; between), also by runs of 32 entries in the order of
the candidate cells; the blocks of 64 of the outer and inner lists; the
candidates of the build's cube of cells against rows of cells trimmed to
the x-interval [b- - sqrt(t), b+ + sqrt(t)], t = R^2 - dy^2 - dz^2; the
populations of the bins of the two sorts of the build.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

from groups import make_groups

sys.argv += [] if len(sys.argv) > 2 else ["600"]
path, nsample = sys.argv[1], int(sys.argv[2])
RC, RIN, R = (8.0, 8.6, 11.0) if "jac" in path else (9.0, 9.6, 12.0)
if path.endswith(".h5md"):
    import h5py
    f = h5py.File(path, "r")
    part = f["particles/all"]
    x = np.asarray(part["position/value"][0], dtype=np.float64) * 10
    box = np.asarray(part["box/edges/value"][0], dtype=np.float64) * 10
else:
    lines = open(path).read().splitlines()
    n = int(lines[1].split()[0])
    vals = "".join(l.ljust(72)[:72] for l in lines[2:2 + (n * 3 + 5) // 6])
    x = np.array([float(vals[12 * k:12 * k + 12]) for k in range(3 * n)]).reshape(n, 3)
    box = np.array([float(v) for v in lines[-1].split()[:3]])
n = len(x)
x = np.mod(x, box)
groups, gid = make_groups(x, box)
ng = len(groups)
print(f"{path}: {n} particles, {ng} groups, box {box}, R {R}, R_in {RIN}")
tree = cKDTree(x, boxsize=box)
cell = R / 2
gn = np.maximum(3, np.floor(box / cell).astype(int))
cw = box / gn
cidx = np.minimum((x / cw).astype(int), gn - 1)
cflat = (cidx[:, 2] * gn[1] + cidx[:, 1]) * gn[0] + cidx[:, 0]
cell_pop = np.bincount(cflat, minlength=gn.prod())
rng = np.random.default_rng(2)
tot = dict(out=0, inn=0, bnd=0, runs=0, runs_dec=0, bo=0, bi=0, cube=0, trim=0, entries=0)
for g in rng.choice(ng, size=min(nsample, ng), replace=False):
    q = groups[g]
    rel = x[q] - x[q[0]]
    rel -= box * np.round(rel / box)
    lo, hi = rel.min(0), rel.max(0)
    js = np.unique(np.concatenate(tree.query_ball_point(x[q], R)))
    js = js[gid[js] >= g]
    d = x[js] - x[q[0]]
    d -= box * np.round(d / box)
    near = np.maximum(np.maximum(lo - d, d - hi), 0)
    far = np.maximum(np.abs(d - lo), np.abs(d - hi))
    out = (near ** 2).sum(1) > RIN ** 2
    inn = (far ** 2).sum(1) <= RIN ** 2
    tot["out"] += out.sum(); tot["inn"] += inn.sum(); tot["bnd"] += (~out & ~inn).sum()
    tot["entries"] += len(js)
    # Entry order: by candidate cell (z, y, x), then place.
    order = np.lexsort((gid[js], cflat[js]))
    dec = (out | inn)[order]
    for k in range(0, len(dec), 32):
        tot["runs"] += 1; tot["runs_dec"] += bool(dec[k:k + 32].all())
    # Inner entries: a member within R_in.
    dm = d[:, None, :] - rel[None, :, :]
    r2 = (dm ** 2).sum(2).min(1)
    nin = int((r2 <= RIN ** 2).sum())
    tot["bo"] += -(-len(js) // 64); tot["bi"] += -(-nin // 64)
    # Build candidates: the cube of cells of bbox +- R, against trimmed rows.
    blo, bhi = x[q[0]] + lo - R, x[q[0]] + hi + R
    c0, c1 = np.floor(blo / cw).astype(int), np.floor(bhi / cw).astype(int)
    for cz in range(c0[2], c1[2] + 1):
        for cy in range(c0[1], c1[1] + 1):
            row = [(cz % gn[2]) * gn[1] + (cy % gn[1])]
            cube = sum(cell_pop[row[0] * gn[0] + (cx % gn[0])] for cx in range(c0[0], c1[0] + 1))
            tot["cube"] += cube
            dy = max(cy * cw[1] - (x[q[0], 1] + hi[1]), (x[q[0], 1] + lo[1]) - (cy + 1) * cw[1], 0)
            dz = max(cz * cw[2] - (x[q[0], 2] + hi[2]), (x[q[0], 2] + lo[2]) - (cz + 1) * cw[2], 0)
            t = R * R - dy * dy - dz * dz
            if t < 0:
                continue
            a, b = x[q[0], 0] + lo[0] - np.sqrt(t), x[q[0], 0] + hi[0] + np.sqrt(t)
            tot["trim"] += sum(cell_pop[row[0] * gn[0] + (cx % gn[0])]
                               for cx in range(int(np.floor(a / cw[0])), int(np.floor(b / cw[0])) + 1))
e = tot["entries"]
print(f"entries per group {e / nsample:.0f}: beyond R_in of the box {tot['out'] / e:.3f}, "
      f"wholly within {tot['inn'] / e:.3f}, between {tot['bnd'] / e:.3f}")
print(f"runs of 32 entries all decided by the box: {tot['runs_dec'] / tot['runs']:.3f}")
print(f"blocks of 64 per group: outer {tot['bo'] / nsample:.2f}, inner {tot['bi'] / nsample:.2f}, "
      f"empty after pruning {1 - tot['bi'] / tot['bo']:.3f}")
print(f"build candidates per group: cube {tot['cube'] / nsample:.0f}, trimmed rows {tot['trim'] / nsample:.0f} "
      f"({1 - tot['trim'] / tot['cube']:.3f} fewer)")
print(f"candidate cells {gn.prod()} of {cw.round(2)} A: population mean {cell_pop.mean():.1f}, "
      f"percentiles 50/90/99/max {np.percentile(cell_pop, [50, 90, 99]).round(1)} {cell_pop.max()}")
cols = gn[:2]
w = (box.prod() / n * 64) ** (1 / 3)
nx, ny = max(1, int(box[0] // w)), max(1, int(box[1] // w))
nzb = max(1, 2 * n // (nx * ny))
col = np.minimum((x[:, 0] / box[0] * nx).astype(int), nx - 1) * ny + np.minimum((x[:, 1] / box[1] * ny).astype(int), ny - 1)
zb = np.minimum((x[:, 2] / box[2] * nzb).astype(int), nzb - 1)
kpop = np.bincount(col * nzb + zb, minlength=nx * ny * nzb)
print(f"key bins {len(kpop)}: population mean {kpop.mean():.2f}, histogram 0..8+ "
      f"{np.bincount(np.minimum(kpop, 8), minlength=9).tolist()}")
