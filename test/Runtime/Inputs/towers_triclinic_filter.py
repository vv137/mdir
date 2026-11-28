"""The lateral filter of TOWERS on sheared cells (D[towers-triclinic-filter]).

Two sheared lattices whose lateral face normals are not orthogonal. Each
holds random dyadic particles and one pair inside the cutoff whose lateral
target lies off a corner of the owner's column: the distance of the pair is
below the cutoff, while the sum of the squared face gaps of the target is
above the padded filter radius. A filter by that sum drops the pair; the
four-axis bound keeps it. The forces of a spring within the cutoff are
compared word for word with an exhaustive loop over lattice images, which
counts every unordered pair within the cutoff once: a missing pair or a
duplicate changes the words.
"""

import ctypes as C
import itertools
import math
from pathlib import Path
import random
import sys

import towers_rebuild as rebuild

scale = 2**18
# (lattice rows, fractional target, fractional source) per case. The target
# lies in the lateral field (-1, -1) of the source's column.
cases = [
    (
        ((8, 0, 0), (0, 8, 0), (4, 4, 6)),
        (1 / 4 + 33 / 256, 1 / 4 + 34 / 256, 96 / 256),
        (2 / 4, 2 / 4, 67 / 256),
    ),
    (
        ((8, 0, 0), (-3, 8, 0), (-2, -3, 8)),
        (1 / 4 + 34 / 256, 1 / 4 + 35 / 256, 96 / 256),
        (2 / 4, 2 / 4, 109 / 256),
    ),
]
dims_xy = (4, 4)


def cartesian(lattice, u):
    return [sum(u[k] * lattice[k][a] for k in range(3)) for a in range(3)]


def dot(u, v):
    return sum(x * y for x, y in zip(u, v))


def premise(lattice, target, source, reach):
    """The corner pair: within the cutoff, dropped by the sum of the squared
    face gaps, kept by the four-axis bound, which stays below the distance."""
    rows = [list(map(float, r)) for r in lattice]

    def cross(u, v):
        return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]]

    normals, slabs = [], []
    for a in range(2):
        n = cross(rows[(a + 1) % 3], rows[(a + 2) % 3])
        n = [x / math.sqrt(dot(n, n)) for x in n]
        normals.append(n)
        slabs.append(dot(rows[a], n) / dims_xy[a])
    nz = 4 * reach
    owner = [math.floor(source[0] * 4), math.floor(source[1] * 4),
             math.floor(target[2] * nz)]
    assert [math.floor(target[0] * 4), math.floor(target[1] * 4)] == [
        owner[0] - 1, owner[1] - 1]
    assert abs(math.floor(source[2] * nz) - owner[2]) <= reach
    center = [(owner[0] + 0.5) / 4, (owner[1] + 0.5) / 4, (owner[2] + 0.5) / nz]
    p = cartesian(rows, [t - c for t, c in zip(target, center)])
    gaps = [max(0.0, abs(dot(n, p)) - h / 2) for n, h in zip(normals, slabs)]
    pair = cartesian(rows, [t - s for t, s in zip(target, source)])
    distance = math.sqrt(dot(pair, pair))
    bound = max(gaps)
    for sign in (1, -1):
        u = [x + sign * y for x, y in zip(*normals)]
        norm = math.sqrt(dot(u, u))
        support = max(
            abs(dot(u, [(e0 * x + e1 * y) / 2 for x, y in zip(
                [r / 4 for r in rows[0]], [r / 4 for r in rows[1]])]))
            for e0 in (1, -1) for e1 in (1, -1)) / norm
        bound = max(bound, abs(dot(u, p)) / norm - support)
    padded = 1 + 2**-10
    assert distance < 1 and math.hypot(*gaps) > padded and bound < distance, (
        distance, gaps, bound)
    return distance, math.hypot(*gaps), bound


def reference(lattice, fractional):
    result = [0] * (3 * len(fractional))
    pairs = 0
    for i, j in itertools.combinations(range(len(fractional)), 2):
        u = [x % 1 - y % 1 for x, y in zip(fractional[i], fractional[j])]
        images = [
            cartesian(lattice, [u[k] - m[k] for k in range(3)])
            for m in itertools.product(range(-2, 3), repeat=3)
        ]
        inside = [d for d in images if dot(d, d) < 1]
        assert len(inside) <= 1
        if not inside:
            continue
        pairs += 1
        for a, x in enumerate(inside[0]):
            word = round(scale * 8 * x)
            result[3 * i + a] += word
            result[3 * j + a] -= word
    return result, pairs


def check(directory, lattice, fractional, reach, workers, packed):
    source = Path(__file__).with_name("towers-force.mlir").read_bytes()
    dims = (*dims_xy, 4 * reach)
    n, split = len(fractional), workers // 32
    opts = {
        "triclinic": 1,
        "certify-force": 1,
        "packed-charge-type": packed,
        "force-workers": workers,
        "force-panel": 32,
        "force-stripe": 32,
        "force-records": 4,
        "force-split": split,
        "exclusions": 0,
    }
    capacity = 16
    image = rebuild.compile_entries(directory, dims, reach, n, capacity, True,
                                    True, False, ("spring", source), opts)
    e = rebuild.Entries(image, True)
    r = rebuild.Rebuild(e, dims, reach, n, capacity, True)
    r.box = e.array(C.c_double, 9)
    r.geometry = e.array(C.c_double, 54)
    r.box[:] = [x for row in lattice for x in row]
    r.setup()
    r.raw[:] = [p[a] for a in range(3) for p in fractional]
    r.charge[:] = [1] * n
    r.types[:] = [0] * n
    # The packed entry takes the binary32 geometry of towers_prepare_force;
    # the other reads the binary64 geometry of the rebuild, as a run does.
    geom = r.geometry
    if packed:
        geom = e.array(rebuild.F, 22 + 3 * r.fields + 10)
        e.run("prepare_force", 1, r.geometry, geom)
    table = e.array(rebuild.F, 4)
    table[:] = [8] * 4
    state, failed = e.array(rebuild.I, 2), e.array(C.c_int64, 1)
    words, low, high, output = [e.array(C.c_int64, 3 * n) for _ in range(4)]
    items = r.cells * workers * split
    prefix = r.spans, r.packed, r.ids, geom
    # The packed entry reads the types by slot; the other reads the charges
    # and the types by particle.
    atoms = (r.packed_types,) if packed else (r.charge, r.types)
    fields = (state, rebuild.F(scale), failed, *atoms, table)
    expected, pairs = reference(lattice, fractional)
    for reverse in (False, True):
        r.step(reverse)
        e.sync()
        assert max(r.count[: r.cells]) <= capacity
        state[:] = [0, 0]
        failed[0] = 0
        words[:] = [0] * (3 * n)
        e.run("force", items, *prefix, words, *fields, block=workers, shared=32)
        e.run("certify", n, words, failed, state)
        e.sync()
        assert state[:] == [0, 0] and failed[0] == 0
        if reverse:
            # The replay of the certificate visits the same tiles.
            state[1] = 1
            e.run("recover_once", workers, *prefix, low, *fields, high,
                  block=workers, shared=32)
        e.run("publish", n, state, words, low, output)
        e.sync()
        assert state[0] == 0 and output[:] == expected, (
            reach, workers, reverse,
            [(i // 3, a, b) for i, (a, b) in enumerate(zip(output, expected))
             if a != b][:6])
    e.close()
    return pairs


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rebuild.MDIR_OPT = sys.argv[2]
    rng = random.Random(1035)
    for index, (lattice, target, source) in enumerate(cases):
        # Random dyadic particles keep every displacement exact in binary32.
        fractional = [target, source] + [
            tuple(rng.randrange(256) / 256 for _ in range(3)) for _ in range(150)
        ]
        for reach, workers, packed in itertools.product((1, 2), (32, 64), (1, 0)):
            distance, summed, bound = premise(lattice, target, source, reach)
            pairs = check(directory, lattice, fractional, reach, workers, packed)
            print(f"lattice {index} reach {reach} workers {workers} packed {packed}: "
                  f"{pairs} pairs; corner pair at {distance:.4f}, "
                  f"summed gap {summed:.4f}, four-axis bound {bound:.4f}")
