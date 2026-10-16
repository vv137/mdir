"""Pairs within distances in the equilibrated boxes (eq.gro, periodic).

    pairs.py SYSTEM... > $CPU_BENCH_DIR/pairs.json

For each system: sites, and the number of ordered pairs (i != j) within the
cutoff, cutoff + 1.0 A, cutoff + 1.5 A (MDIR's default reach on the CPU),
and cutoff + 3 A, per step of a full matrix; half of them are the unique pairs.
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

import os

S = os.environ["CPU_BENCH_DIR"]
out = {}
for s in sys.argv[1:]:
    lines = open(f"{S}/{s}/eq.gro").read().splitlines()
    n = int(lines[1])
    x = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in lines[2:2 + n]]) * 10
    box = np.array([float(v) for v in lines[2 + n].split()[:3]]) * 10
    x = np.mod(x, box)
    tree = cKDTree(x, boxsize=box)
    cut = 8.0 if s == "jac" else 9.0
    res = dict(sites=n, box=box.tolist(), cutoff=cut)
    for extra in (0.0, 1.0, 1.5, 3.0):
        r = cut + extra
        res[f"ordered_pairs_{r:g}"] = int(tree.count_neighbors(tree, r)) - n
    out[s] = res
    print(s, res, file=sys.stderr, flush=True)
json.dump(out, sys.stdout, indent=1)
