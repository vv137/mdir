"""Check exclusion debits and physical recovery against exact signed sums."""

import ctypes as C
import itertools
from pathlib import Path
import sys

import towers_rebuild as rebuild


def evaluate(directory, gpu, workers, shape=None):
    source = Path(__file__).with_name("towers-force.mlir").read_bytes()
    body = source[source.index(b"  md_exec.pair_for"):source.index(b"  return")]
    # The unconditional physical spring is independent of the direct table.
    physical = body.replace(b"%kq = arith.mulf %spring, %charge : f32",
                            b"%two = arith.constant 2.0 : f32\n    %kq = arith.mulf %two, %charge : f32")
    source = source.replace(b"  return", physical + b"  return")
    x = (0.125, 5.875, 2.125, 0.375)
    bonds = [(0, 1), (0, 2)]
    n = len(x)
    dims = (6, 6, 6)
    split = 2 if workers == 64 else 1
    panel = stripe = 32
    records = residency = 1
    if shape:
        panel, stripe = shape["panel"], shape["stripe"]
        records, residency, split = shape["records"], shape["blocks_per_sm"], shape["split"]
    image = rebuild.compile_entries(
        directory, dims, 1, n, 8, True, gpu, False, ("spring", source),
        {"certify-force": 1, "exclusions": len(bonds), "packed-charge-type": 1,
         "force-workers": workers, "force-panel": panel, "force-stripe": stripe,
         "force-records": records, "force-residency": residency, "force-split": split},
    )
    e = rebuild.Entries(image, gpu)
    r = rebuild.Rebuild(e, dims, 1, n, 8, True)
    r.box[:] = [6] * 3
    r.setup()
    r.raw[:] = list(x) + [0] * (2 * n)
    r.types[:] = [1, 1, 0, 0]
    r.charge[:] = [1] * n
    geom = r.geometry
    if gpu:
        geom = e.array(rebuild.F, 7)
        e.run("prepare_force", 1, r.geometry, geom)
    table = e.array(rebuild.F, 4)
    topology = e.array(rebuild.I, 2 * len(bonds))
    topology[:] = [v for edge in bonds for v in edge]
    state = e.array(rebuild.I, 2)
    failed = e.array(C.c_int64, 1)
    words, lo, hi, output = [e.array(C.c_int64, 3 * n) for _ in range(4)]
    results = []
    for step, (ordinary, isolated_charge) in enumerate(
        [(8, 1), (2**35, 1), (8, 2**30), (8, 2**70), (8, 1)]
    ):
        r.charge[2] = isolated_charge
        r.step(bool(step % 2))
        # The excluded direct term cannot even be converted to signed i64.
        table[:] = [ordinary, ordinary, ordinary, 2**70]
        state[:] = [0, 0]
        failed[0] = 0
        words[:] = [0] * (3 * n)
        for dest in (lo, hi, output):
            dest[:] = [987] * (3 * n)
        prefix = (r.spans, r.packed, r.ids, geom)
        fields = (state, rebuild.F(4), failed, topology, r.inverse, r.raw,
                  r.box, r.packed_types, table)
        items = r.cells * split * workers + len(bonds)
        e.run("force", items, *prefix, words, *fields, block=workers, shared=8 * records)
        e.run("certify", n, words, failed, state)
        if gpu and workers == 64:
            e.run("recover_once", workers, *prefix, lo, *fields, hi,
                  block=workers, shared=8 * records)
        else:
            e.run("clear_wide", n, state, lo, hi)
            e.run("recover", items, *prefix, lo, *fields, hi,
                  block=workers, shared=8 * records)
            e.run("check_wide", n, state, lo, hi)
        e.run("publish", n, state, words, lo, output)
        e.sync()
        expected = [0] * (3 * n)
        bad = 0
        for i, j in itertools.combinations(range(n), 2):
            delta = x[i] - x[j]
            delta -= round(delta / 6) * 6
            if (i, j) in bonds:
                word = round(2 * 4 * delta * (isolated_charge if j == 2 else 1))
            elif abs(delta) < 1:
                word = round(ordinary * 4 * delta)
            else:
                continue
            bad += int(abs(word) > 2**31)
            expected[3 * i] += word
            expected[3 * j] -= word
        fatal = int(isolated_charge == 2**70)
        assert list(state) == [fatal, int(bool(bad))], (list(state), bad)
        assert failed[0] == bad, failed[0]
        if fatal:
            assert output[:] == [987] * (3 * n)
        else:
            assert output[:] == expected, (output[:], expected)
        if not bad:
            assert lo[:] == hi[:] == [987] * (3 * n)
        results.append(output[:])
    e.close()
    return results


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rebuild.MDIR_OPT = sys.argv[3]
    gpu = sys.argv[2] == "gpu"
    assert evaluate(directory, gpu, 32) == evaluate(directory, gpu, 64)
