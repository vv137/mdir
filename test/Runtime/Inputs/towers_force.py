"""Check the emitted force region and masked rank traversal by unordered atom ID."""

import ctypes as C
import itertools
import random
import sys
from pathlib import Path

import towers_rebuild as rebuild


def check(directory, gpu, reach):
    dims = (3, 3, 2 * reach + 1)
    lengths = tuple(2 * k if a < 2 else 2 * k / reach for a, k in enumerate(dims))
    rng = random.Random(41)
    # Dyadic coordinates and coefficients make the Q18 force oracle exact.
    # The crowded home cell streams in both dimensions and has several masks
    # per target. Other cells exercise wraps, empty lanes and clipped tiles.
    xyz = [tuple(rng.randrange(128) / 128 for _ in dims) for _ in range(157)]
    xyz += [
        tuple(rng.randrange(int(128 * box)) / 128 for box in lengths)
        for _ in range(121)
    ]
    xyz += [(0, 0, 0), (1, 0, 0), (127 / 128, 0, 0), (129 / 128, 0, 0)]
    n = len(xyz)
    capacity = 29 if reach == 2 else 512
    source = Path(__file__).with_name("towers-force.mlir").read_bytes()
    image = rebuild.compile_entries(
        directory,
        dims,
        reach,
        n,
        192,
        True,
        gpu,
        False,
        ("spring", source),
        {"force-records": capacity},
    )
    if gpu:
        assert b"mma.sync.aligned.m16n8k8" in image and b"cvt.rm.f16.f32" in image
    entries = rebuild.Entries(image, gpu)
    r = rebuild.Rebuild(entries, dims, reach, n, 192, True)
    r.box[:] = lengths
    r.raw[:] = [p[a] for a in range(3) for p in xyz]
    r.types[:] = [i % 2 for i in range(n)]
    r.setup()
    words = entries.array(C.c_int64, 3 * n)
    status = entries.array(rebuild.I, 1)
    table = entries.array(rebuild.F, 4)
    for reverse in (False, True):
        table[:] = [2**16, 3, 3, -(2**17)] if reverse else [2, -0.5, -0.5, 4]
        expected = [0] * (3 * n)
        pairs = 0
        for i, j in itertools.combinations(range(n), 2):
            d = [
                u - v - round((u - v) / box) * box
                for u, v, box in zip(xyz[i], xyz[j], lengths)
            ]
            if sum(v * v for v in d) < 1:
                pairs += 1
                k = table[2 * (i % 2) + (j % 2)]
                for a, v in enumerate(d):
                    w = round(k * v * 2**18)
                    expected[3 * i + a] += w
                    expected[3 * j + a] -= w
        r.step(reverse)
        entries.sync()
        assert r.status[0] == 0
        words[:] = [0] * (3 * n)
        status[0] = 0
        entries.run(
            "force",
            r.cells * 128,
            r.spans,
            r.packed,
            r.ids,
            r.geometry,
            words,
            status,
            rebuild.F(2**18),
            r.types,
            table,
            shared=8 * capacity,
        )
        entries.sync()
        assert status[0] == 0
        assert words[:] == expected, max(abs(a - b) for a, b in zip(words, expected))
    table[0] = float("inf")
    entries.run(
        "force",
        r.cells * 128,
        r.spans,
        r.packed,
        r.ids,
        r.geometry,
        words,
        status,
        rebuild.F(2**18),
        r.types,
        table,
        shared=8 * capacity,
    )
    entries.sync()
    assert status[0] == 1
    r.box[0] = 0.75 * dims[0]
    r.setup()
    status[0] = 0
    entries.run(
        "force",
        r.cells * 128,
        r.spans,
        r.packed,
        r.ids,
        r.geometry,
        words,
        status,
        rebuild.F(2**18),
        r.types,
        table,
        shared=8 * capacity,
    )
    entries.sync()
    assert status[0] == 2
    entries.close()
    print(f"reach {reach}: {pairs} pairs, {3*n} force words match in both orders")


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rebuild.MDIR_OPT = sys.argv[3]
    for reach in (1, 2):
        check(directory, sys.argv[2] == "gpu", reach)
