"""Exercise narrow certification and replacement by an independent wide replay."""

import ctypes as C
import itertools
import sys
from pathlib import Path

import towers_rebuild as rebuild


def check(directory, gpu, workers):
    # Dyadic coordinates keep every product and rounding exact. Repeated
    # positions let several distinct interactions overflow a single endpoint.
    line = [(x, 0, 0) for x in (0, 0, 0, 0.5, 0.5, 0.5, 0.5)]
    corners = list(itertools.product((1.75, 2.25), repeat=3))[:7]
    n = len(line)
    dims = (3, 3, 3)
    packed = workers == 64
    source = Path(__file__).with_name("towers-force.mlir").read_bytes()
    image = rebuild.compile_entries(
        directory,
        dims,
        1,
        n,
        32,
        True,
        gpu,
        False,
        ("spring", source),
        {
            "certify-force": 1,
            "packed-charge-type": int(packed),
            "force-workers": workers,
            "force-panel": 32,
            "force-stripe": 32,
            "force-records": 1,
        },
    )
    e = rebuild.Entries(image, gpu)
    r = rebuild.Rebuild(e, dims, 1, n, 32, True)
    r.box[:] = (6, 6, 6)
    r.types[:] = [0] * n
    r.setup()
    geom = r.geometry
    if packed and gpu:
        geom = e.array(rebuild.F, 7)
        e.run("prepare_force", 1, r.geometry, geom)
    fields = (r.packed_types,) if packed else (r.charge, r.types)
    table = e.array(rebuild.F, 4)
    state = e.array(rebuild.I, 2)
    failed = e.array(C.c_int64, 1)
    provisional, lo, hi, output = [e.array(C.c_int64, 3 * n) for _ in range(4)]
    sentinel = 747
    common = (state, rebuild.F(1), failed, *fields, table)
    fixtures = [
        (line, 4, [1] * n, False, 0),  # small exact force
        (line, 2**31, [1] * n, True, 0),  # only the endpoint sum exceeds i32
        (line, 2**32, [1] * n, True, 0),  # -2^31 fits the narrow conversion
        (line, -(2**32), [1] * n, True, 0),  # +2^31 requires wide conversion
        (line, 2**33, [1] * n, True, 0),  # failed narrow conversions
        (line, 2**63, [1, -1, 1, 1, 1, -1, -1], True, 0),  # cancel wide partials
        (line, 2**63, [1] * n, True, 4),  # exact sum is outside signed i64
        (line, 2**64, [1] * n, True, 4),  # -2^63 fits before the endpoint sum
        (line, -(2**64), [1] * n, True, 1),  # +2^63 fails physical conversion
        (line, float("inf"), [1] * n, True, 1),
        (corners, 2**63, [-1, -1, 1, 1, 1, 1, -1], True, 0),  # seven owning boxes
        (line, 4, [1] * n, False, 0),  # reuse after failure
    ]
    for reverse, (points, coefficient, charges, retry, error) in enumerate(fixtures):
        r.raw[:] = [p[axis] for axis in range(3) for p in points]
        r.charge[:] = charges
        table[:] = [coefficient] * 4
        r.step(bool(reverse % 2))
        e.sync()
        assert r.status[0] == 0
        state[:] = [0, 0]
        failed[0] = 0
        provisional[:] = [0] * (3 * n)
        for buf in (lo, hi, output):
            buf[:] = [sentinel] * (3 * n)
        prefix = (r.spans, r.packed, r.ids, geom)
        e.run(
            "force",
            r.cells * workers,
            *prefix,
            provisional,
            *common,
            block=workers,
            shared=8,
        )
        e.run("certify", n, provisional, failed, state)
        single = gpu and workers == 64
        if not single:
            e.run("clear_wide", n, state, lo, hi)
        e.run(
            "recover_once" if single else "recover",
            workers if single else r.cells * (workers if gpu else 1),
            *prefix,
            lo,
            *common,
            hi,
            block=workers,
            shared=8,
        )
        if not single:
            e.run("check_wide", n, state, lo, hi)
        e.run("publish", n, state, provisional, lo, output)
        e.sync()
        assert list(state) == [error, int(retry)], list(state)
        if error != 1:
            expected = [0] * (3 * n)
            bad = 0
            for i, j in itertools.combinations(range(n), 2):
                delta = [a - b for a, b in zip(points[i], points[j])]
                if sum(v * v for v in delta) >= 1:
                    continue
                w = [round(v * coefficient * charges[i] * charges[j]) for v in delta]
                bad += any(not -(2**31) <= v < 2**31 for v in w)
                for axis, word in enumerate(w):
                    expected[3 * i + axis] += word
                    expected[3 * j + axis] -= word
            assert failed[0] == bad, (failed[0], bad)
            if retry:
                for k, word in enumerate(expected):
                    assert lo[k] == C.c_int64(word).value
                    assert hi[k] == word >> 64
        else:
            assert failed[0] > 0
        if error:
            assert list(output) == [sentinel] * (3 * n)
        else:
            assert list(output) == expected
        if not retry:
            assert list(lo) == list(hi) == [sentinel] * (3 * n)

    # Both signed endpoints pass final validation. The adjacent exact values
    # have the same low limbs as the opposite endpoints and must be rejected.
    for value in (-(2**63), 2**63 - 1, -(2**63) - 1, 2**63):
        state[:] = [0, 1]
        lo[:] = [0] * (3 * n)
        hi[:] = [0] * (3 * n)
        lo[0], hi[0] = C.c_int64(value).value, value >> 64
        e.run("check_wide", n, state, lo, hi)
        e.sync()
        assert state[0] == 4 * int(not -(2**63) <= value < 2**63)
    for value in (-(2**31), 2**31 - 1, -(2**31) - 1, 2**31):
        state[:] = [0, 0]
        failed[0] = 0
        provisional[:] = [0] * (3 * n)
        provisional[0] = value
        e.run("certify", n, provisional, failed, state)
        e.sync()
        assert state[1] == int(not -(2**31) <= value < 2**31)
    # A signed debit canceled by a later correction can reach zero. An
    # outstanding count in either direction must select recovery.
    provisional[:] = [0] * (3 * n)
    for balance in (1, -1, 0):
        state[:] = [0, 0]
        failed[0] = balance
        e.run("certify", n, provisional, failed, state)
        e.sync()
        assert state[1] == int(balance != 0)
    e.close()
    print(
        f"{workers} workers: twelve frame evaluations and eight range endpoints passed"
    )


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rebuild.MDIR_OPT = sys.argv[3]
    for workers in (32, 64):
        check(directory, sys.argv[2] == "gpu", workers)
