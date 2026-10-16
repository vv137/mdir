"""Offline study of the trigger of the pruning (#357) on frames at every step.

    trigger.py FRAMES.h5md [--check K]

Groups of 16 are made as the build makes them (columns of the x-y grid of
about 64 particles, sorted by z, chunks of 64 halved by x and by y), anew at
each simulated build. The outer list (reach R) is rebuilt when the test of
D80 fails; the inner list (R_in) is pruned by each rule:

  G    global (main): every group, when 2 max_k d_k > s
  B    per group: max_{i in g} d_i + sum over steps since t_g of the global
       largest one-step displacement > s
  Bx   per group: own + the exact global max displacement since t_g
  C    per group: own + sum over steps of the largest one-step displacement
       among the groups within reach of g
  A    per group: own + the exact max displacement since t_g among the
       groups within reach of g (an ideal bound)

d_k is the displacement since the governing pruning (of the group, or the
global one), s = R_in - r_c. All rules force a pruning of every group at a
build. Prints the group-prunings per step of each rule, relative to G.
--check K: for rule B, K random groups at every step: every pair of a
particle of g and any particle within r_c now was within R_in at g's last
pruning and within R at the last build.
"""
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

from groups import make_groups

RC, RIN, R = 9.0, 9.6, 12.0
S_IN, S_OUT = RIN - RC, R - RC

path = sys.argv[1]
check = int(sys.argv[sys.argv.index("--check") + 1]) if "--check" in sys.argv else 0
f = h5py.File(path, "r")
part = f["particles"][list(f["particles"].keys())[0]]
X = part["position/value"]
nframes, n, _ = X.shape
box_ds = part["box/edges/value"] if "value" in part["box/edges"] else None
box0 = np.asarray(part["box/edges"] if box_ds is None else box_ds[0], dtype=np.float64)
if box0.ndim == 2:
    box0 = np.diag(box0)
scale = 10.0 if box0.max() < 100 else 1.0   # nm -> Å
print(f"{path}: {nframes} frames, {n} particles, box {box0 * scale} Å", flush=True)


def frame(t):
    b = np.asarray(box0 if box_ds is None else box_ds[t], dtype=np.float64)
    if b.ndim == 2:
        b = np.diag(b)
    return np.asarray(X[t], dtype=np.float64) * scale, b * scale


def group_neighbors(x, box, groups, gid):
    """Groups h within reach of g: any pair within R (via the spheres)."""
    xw = np.mod(x, box)
    centers = np.array([xw[q[0]] + ((xw[q] - xw[q[0]] + box / 2) % box - box / 2).mean(0) for q in groups])
    centers = np.mod(centers, box)
    radius = np.array([np.sqrt((((xw[q] - c + box / 2) % box - box / 2) ** 2).sum(1)).max()
                       for q, c in zip(groups, centers)])
    tree = cKDTree(centers, boxsize=box)
    reach = R + 2 * radius.max()
    nb = tree.query_ball_point(centers, reach)
    out = []
    for g, hs in enumerate(nb):
        hs = np.array(hs)
        d = np.sqrt((((centers[hs] - centers[g] + box / 2) % box - box / 2) ** 2).sum(1))
        out.append(hs[d <= R + radius[g] + radius[hs]])
    return out


def disp(x, xref, box, boxref):
    m = box / boxref
    d = x - m * xref
    d -= box * np.round(d / box)
    return np.sqrt((d * d).sum(1)), m.min()


def groupmax(v, gid, ng):
    out = np.zeros(ng)
    np.maximum.at(out, gid, v)
    return out


def nbmax(vg, nbs):
    return np.array([vg[h].max() for h in nbs])


rules = ["G", "B", "Bx", "C", "A"]
counts = {r: 0 for r in rules}
steps_with = {r: 0 for r in rules}
builds = 0
x, box = frame(0)
xb, boxb = x.copy(), box.copy()
groups, gid = make_groups(x, box)
ng = len(groups)
nbs = group_neighbors(x, box, groups, gid)
hist = {0: (x.copy(), box.copy())}          # frames kept for the exact rules
tg = {r: np.zeros(ng, dtype=np.int64) for r in rules}
xg = {r: np.zeros((ng, 0)) for r in rules}
sum_global = [0.0]                        # cumulative sum of the largest one-step displacement
cum_nb = [np.zeros(ng)]                   # cumulative neighborhood sums (by group, current groups)
xprev, boxprev = x, box
tp_global = 0
groupsizes = []
viol = {r: 0 for r in rules}
checked = {r: 0 for r in rules}
rng = np.random.default_rng(1)
for t in range(1, nframes):
    x, box = frame(t)
    hist[t] = (x, box)
    step, _ = disp(x, xprev, box, boxprev)
    sum_global.append(sum_global[-1] + step.max())
    cum_nb.append(cum_nb[-1] + nbmax(groupmax(step, gid, ng), nbs))
    xprev, boxprev = x, box
    db, mb = disp(x, xb, box, boxb)
    if 2 * db.max() > mb * R - RC:
        # A build: new groups, every rule prunes every group.
        builds += 1
        xb, boxb = x.copy(), box.copy()
        groups, gid = make_groups(x, box)
        ng = len(groups)
        nbs = group_neighbors(x, box, groups, gid)
        cum_nb = [np.zeros(ng) for _ in range(t)] + [np.zeros(ng)]
        for r in rules:
            counts[r] += ng
            steps_with[r] += 1
            tg[r] = np.full(ng, t, dtype=np.int64)
        tp_global = t
        groupsizes.append(ng)
        continue
    # G: the global test against the global pruning.
    d, m = disp(x, hist[tp_global][0], box, hist[tp_global][1])
    if 2 * d.max() > m * RIN - RC:
        counts["G"] += ng
        steps_with["G"] += 1
        tg["G"][:] = t
        tp_global = t
    for r in ("B", "Bx", "C", "A"):
        due = np.zeros(ng, dtype=bool)
        for t0 in np.unique(tg[r]):
            sel = tg[r] == t0
            dk, m0 = disp(x, hist[t0][0], box, hist[t0][1])
            own = groupmax(dk, gid, ng)
            limit = m0 * RIN - RC
            if r == "B":
                partner = sum_global[t] - sum_global[t0]
            elif r == "Bx":
                partner = dk.max()
            elif r == "C":
                partner = cum_nb[t] - cum_nb[t0]
            else:
                partner = nbmax(own, nbs)
            bad = own + partner > limit
            due |= sel & bad
        if due.any():
            counts[r] += int(due.sum())
            steps_with[r] += 1
            tg[r][due] = t
    if check and t % 5 == 0:
        # Exactness of the per-group rules on K random groups: every pair
        # of the group's list within r_c now was within R_in at the
        # group's last pruning.
        tree = cKDTree(np.mod(x, box), boxsize=box)
        for g in rng.choice(ng, size=min(check, ng), replace=False):
            for i in groups[g]:
                js = np.array(tree.query_ball_point(np.mod(x[i], box), RC))
                js = js[(js != i) & (gid[js] >= g)]
                if not len(js):
                    continue
                for r in ("B", "C", "A"):
                    x0, b0 = hist[tg[r][g]]
                    dd = x0[js] - x0[i]
                    dd -= b0 * np.round(dd / b0)
                    r0 = np.sqrt((dd * dd).sum(1))
                    checked[r] += len(js)
                    viol[r] += int((r0 > RIN).sum())
    # Forget frames no rule refers to.
    live = {tp_global} | {int(v) for r in rules for v in np.unique(tg[r])}
    for k in [k for k in hist if k not in live and k != t]:
        del hist[k]
steps = nframes - 1
print(f"{steps} steps, {builds} builds, {ng} groups", flush=True)
for r in rules:
    rate = counts[r] / steps / ng
    print(f"rule {r}: group-prunings {counts[r]}, {rate:.3f} of the groups a step "
          f"(every {1 / rate:.2f} steps a group), {counts['G'] / counts[r]:.2f}x fewer than G; "
          f"steps with a pruning {steps_with[r]}", flush=True)
if check:
    for r in ("B", "C", "A"):
        print(f"check of rule {r}: {checked[r]} pairs within r_c, {viol[r]} beyond R_in at the group's pruning",
              flush=True)
