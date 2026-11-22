"""Compare fractional rebuild and exclusion recovery to exhaustive lattice images."""

import ctypes as C
import itertools
from pathlib import Path
import sys

import towers_rebuild as rebuild


lattice = ((8, 0, 0), (-3, 8, 0), (-2, -3, 8))
# Dyadic coordinates give exactly representable cell displacements, including
# translated images and a distant bond whose closest image needs a search.
fractional = [
    (63 / 64, 63 / 64, 63 / 64),
    (1 / 64, 1 / 64, 1 / 64),
    (17 / 32, 15 / 32, 1 / 32),
    (1 / 32, 1 / 32, 1 / 32),
    (5 / 4, -1 / 2, 13 / 8),
    (1 / 4, 1 / 2, 37 / 64),
    # The last pair is two float ULPs inside the physical cutoff.
    (1 / 8, 1 / 4, 3 / 4),
    (1 / 8 + (1 - 2**-23) / 8, 1 / 4, 3 / 4),
]
bonds = [(0, 1), (0, 2), (1, 2), (2, 5)]
scale = 2**18


def displacement(u, image):
    return tuple(
        sum((u[k] - image[k]) * lattice[k][a] for k in range(3)) for a in range(3)
    )


def reference(finite_at_cutoff):
    result = [0] * (3 * len(fractional))
    missed = 0
    for i, j in itertools.combinations(range(len(fractional)), 2):
        u = [x % 1 - y % 1 for x, y in zip(fractional[i], fractional[j])]
        rounded = displacement(u, list(map(round, u)))
        choices = [rounded] + [
            displacement(u, image)
            for image in itertools.product(range(-2, 3), repeat=3)
        ]
        norm2 = lambda v: sum(x * x for x in v)
        d = min(choices, key=norm2)
        if (i, j) in bonds:
            missed += norm2(d) < norm2(rounded)
            k = 2
        elif (i, j) == (6, 7) and not finite_at_cutoff:
            continue
        elif norm2(d) < 1:
            k = 8
        else:
            continue
        for a, x in enumerate(d):
            word = round(scale * k * x)
            result[3 * i + a] += word
            result[3 * j + a] -= word
    assert missed
    return result


def check(directory, gpu, reach, workers, finite_at_cutoff=True):
    source = Path(__file__).with_name("towers-force.mlir").read_bytes()
    body = source[source.index(b"  md_exec.pair_for") : source.index(b"  return")]
    physical = body.replace(
        b"%kq = arith.mulf %spring, %charge : f32",
        b"%two = arith.constant 2.0 : f32\n    %kq = arith.mulf %two, %charge : f32",
    )
    dims = (4, 4, 4 * reach)
    n, split = len(fractional), workers // 32
    opts = {
        "triclinic": 1,
        "finite-at-cutoff": int(finite_at_cutoff),
        "certify-force": 1,
        "packed-charge-type": 1,
        "force-workers": workers,
        "force-panel": 32,
        "force-stripe": 32,
        "force-records": 1,
        "force-split": split,
        "exclusions": len(bonds),
    }
    image = rebuild.compile_entries(
        directory,
        dims,
        reach,
        n,
        8,
        True,
        gpu,
        False,
        ("spring", source.replace(b"  return", physical + b"  return")),
        opts,
    )
    e = rebuild.Entries(image, gpu)
    r = rebuild.Rebuild(e, dims, reach, n, 8, True)
    r.box = e.array(C.c_double, 9)
    r.geometry = e.array(C.c_double, 39)
    r.box[:] = [x for row in lattice for x in row]
    r.setup()
    r.raw[:] = [p[a] for a in range(3) for p in fractional]
    r.charge[:] = [1] * n
    r.types[:] = [0] * n
    geom = r.geometry
    if gpu:
        geom = e.array(rebuild.F, 22 + 3 * r.fields)
        e.run("prepare_force", 1, r.geometry, geom)
    table = e.array(rebuild.F, 4)
    table[:] = [8] * 4
    edges = e.array(rebuild.I, 2 * len(bonds))
    edges[:] = [atom for edge in bonds for atom in edge]
    state, failed = e.array(rebuild.I, 2), e.array(C.c_int64, 1)
    words, low, high, output = [e.array(C.c_int64, 3 * n) for _ in range(4)]
    items = r.cells * workers * split + len(bonds)
    prefix = r.spans, r.packed, r.ids, geom
    fields = (
        state,
        rebuild.F(scale),
        failed,
        edges,
        r.inverse,
        r.raw,
        r.box,
        r.packed_types,
        table,
    )
    expected = reference(finite_at_cutoff)
    for reverse in (False, True):
        r.step(reverse)
        state[:] = [0, 0]
        failed[0] = 0
        words[:] = [0] * (3 * n)
        e.run("force", items, *prefix, words, *fields, block=workers, shared=8)
        e.run("certify", n, words, failed, state)
        e.sync()
        assert state[:] == [0, 0] and failed[0] == 0
        if reverse:
            state[1] = 1
            if gpu:
                e.run(
                    "recover_once",
                    workers,
                    *prefix,
                    low,
                    *fields,
                    high,
                    block=workers,
                    shared=8,
                )
            else:
                e.run("clear_wide", n, state, low, high)
                e.run("recover", items, *prefix, low, *fields, high)
                e.run("check_wide", n, state, low, high)
        e.run("publish", n, state, words, low, output)
        e.sync()
        assert state[0] == 0 and output[:] == expected, (
            reach,
            workers,
            reverse,
            output[:],
            expected,
        )
    e.close()


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rebuild.MDIR_OPT = sys.argv[3]
    for reach, workers in itertools.product((1, 2), (32, 64)):
        check(directory, sys.argv[2] == "gpu", reach, workers)
    check(directory, sys.argv[2] == "gpu", 2, 64, finite_at_cutoff=False)
