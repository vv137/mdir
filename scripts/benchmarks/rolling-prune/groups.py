"""Groups of 16 places as the build of the lists of groups makes them
(docs/groups-m1.md), for the offline studies of #357."""
import numpy as np


def make_groups(x, box):
    """Places of 16 as the build orders them (the shape, not the exact order)."""
    n = len(x)
    w = (box[0] * box[1] * box[2] / n * 64) ** (1 / 3)   # columns of about 64
    nx, ny = max(1, int(box[0] // w)), max(1, int(box[1] // w))
    xw = np.mod(x, box)
    col = (np.minimum((xw[:, 0] / box[0] * nx).astype(int), nx - 1) * ny
           + np.minimum((xw[:, 1] / box[1] * ny).astype(int), ny - 1))
    order = np.lexsort((xw[:, 2], col))
    groups = []
    starts = np.r_[0, np.flatnonzero(np.diff(col[order])) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        c = order[a:b]
        for k in range(0, len(c), 64):
            chunk = c[k:k + 64]
            chunk = chunk[np.argsort(xw[chunk, 0], kind="stable")]
            for half in np.array_split(chunk, 2):
                half = half[np.argsort(xw[half, 1], kind="stable")]
                for q in np.array_split(half, 2):
                    if len(q):
                        groups.append(q)
    gid = np.empty(n, dtype=np.int64)
    for g, q in enumerate(groups):
        gid[q] = g
    return groups, gid


