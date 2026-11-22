"""Check the separate direct energy and virial entries on one packed frame."""

import ctypes as C
import itertools
import math
from pathlib import Path
import sys

import towers_rebuild as rebuild


def source():
    force = Path(__file__).with_name("towers-force.mlir").read_text()
    body = force[force.index("  md_exec.pair_for") : force.index("  return")]
    correction = body.replace("%spring, %charge", "%two, %charge").replace(
        "    %kq =", "    %two = arith.constant 2.0 : f32\n    %kq ="
    )
    force = force.replace("  return", correction + "  return")
    energy = force[force.index("func.func") :].replace("@spring(", "@potential(")
    energy = energy.replace(
        "  %built =", "  %zero = arith.constant 0.0 : f64\n  %built ="
    )
    energy = energy.replace("outs(%f : memref<?x3xf32>)", "reduce(%zero : f64)")
    energy = energy.replace(
        "exchange [antisymmetric]", "weights [0.5] exchange [symmetric]"
    )
    energy = energy.replace("  md_exec.pair_for", "  %u = md_exec.pair_for", 1)
    energy = energy.replace("  md_exec.pair_for", "  %c = md_exec.pair_for", 1)
    energy = energy.replace(
        "    md_exec.yield %df : vector<3xf32>",
        """    %minusHalf = arith.constant -0.5 : f32
    %kr2 = arith.mulf %kq, %r2 : f32
    %e = arith.mulf %kr2, %minusHalf : f32
    %sum = arith.extf %e : f32 to f64
    md_exec.yield %sum : f64""",
    )
    energy = energy.replace(
        "} : !nl, memref<?x3xf64>", "} : !nl, memref<?x3xf64> -> f64"
    )
    return (force + energy).encode()


def evaluate(directory, gpu, reach, workers):
    dims = (6, 6, 6 * reach)
    xyz = [
        (0.0123, 0.13, 0.27891),
        (5.895, 0.21, 0.189),
        (2.231, 0.38, 0.19),
        (0.4027, 0.14, 0.481),
    ]
    n = len(xyz)
    bonds = [(0, 1), (0, 2)]
    image = rebuild.compile_entries(
        directory,
        dims,
        reach,
        n,
        8,
        True,
        gpu,
        False,
        ("spring", source()),
        {
            "certify-force": 1,
            "exclusions": len(bonds),
            "packed-charge-type": 1,
            "energy-body": "potential",
            "direct-virial": 1,
            "force-workers": workers,
            "force-panel": workers,
            "force-stripe": workers,
            "force-records": workers // 32,
            "force-residency": workers // 32,
            "force-split": workers // 32,
        },
    )
    e = rebuild.Entries(image, gpu)
    r = rebuild.Rebuild(e, dims, reach, n, 8, True)
    r.box[:] = [6] * 3
    r.setup()
    r.raw[:] = [v[a] for a in range(3) for v in xyz]
    r.charge[:] = [1] * n
    r.types[:] = [1, 1, 0, 0]
    geom = r.geometry
    if gpu:
        geom = e.array(rebuild.F, 7)
        e.run("prepare_force", 1, r.geometry, geom)
    table = e.array(rebuild.F, 4)
    table[:] = [8, 8, 8, 2**70]
    topology = e.array(rebuild.I, 4)
    topology[:] = [i for edge in bonds for i in edge]
    state = e.array(rebuild.I, 2)
    failed = e.array(C.c_int64, 1)
    sums, wide = [e.array(C.c_int64, 2) for _ in range(2)]
    word = e.array(C.c_int64, 1)
    tensor, virial = [e.array(C.c_double, 6) for _ in range(2)]
    items = r.cells * (workers // 32) * workers + len(bonds)

    energy = 0.0
    reference = [0.0] * 6
    for i, j in itertools.combinations(range(n), 2):
        d = [a - b - round((a - b) / 6) * 6 for a, b in zip(xyz[i], xyz[j])]
        r2 = sum(v * v for v in d)
        k = 2 if (i, j) in bonds else 8 if r2 < 1 else 0
        energy -= k * r2 / 2
        for h, (a, b) in enumerate(
            itertools.combinations_with_replacement(range(3), 2)
        ):
            reference[h] += k * d[a] * d[b]

    force, recovered_force, published_force = [
        e.array(C.c_int64, 3 * n) for _ in range(3)
    ]
    outputs = []
    for replay, fatal, overflow in [
        (0, False, False),
        (1, False, False),
        (0, True, False),
        (0, False, True),
    ]:
        if overflow:
            # Three ordinary springs each fit in Q32. Their sum exceeds i64.
            table[:] = [2**32] * 4
            r.raw[:] = [0, 0.9, 0, 0.9] + [0] * (2 * n)
        published_force[:] = [987] * (3 * n)
        force[:] = recovered_force[:] = list(range(3 * n))
        r.charge[2] = 2**70 if fatal else 1
        r.step()
        state[:] = [0, replay]
        failed[0] = 0
        sums[:] = wide[:] = [0, 0]
        tensor[:] = [0] * 6
        word[0] = 987
        virial[:] = [987] * 6
        prefix = (r.spans, r.packed, r.ids, geom)
        fields = (
            state,
            rebuild.F(2**32),
            failed,
            topology,
            r.inverse,
            r.raw,
            r.box,
            r.packed_types,
            table,
        )
        kw = dict(block=workers, shared=8 * (workers // 32))
        e.run("energy", items, *prefix, sums, *fields, **kw)
        e.run("virial", items, *prefix, tensor, *fields, **kw)
        e.run("certify_outputs", n, force, sums, failed, state)
        if gpu and workers == 64:
            e.run("energy_recover_once", workers, *prefix, wide, *fields, **kw)
        else:
            e.run("energy_recover", items, *prefix, wide, *fields, **kw)
            e.run("check_recovered_energy", 1, wide, state)
        e.run("virial_check", 6, tensor, state)
        e.run(
            "publish_outputs",
            n,
            state,
            force,
            recovered_force,
            published_force,
            sums,
            wide,
            word,
        )
        e.run("publish_virial", 6, state, tensor, virial)
        e.sync()
        if fatal or overflow:
            if overflow:
                assert failed[0] == 0 and state[:] == [4, 1], state[:]
            assert published_force[:] == [987] * (3 * n)
            assert state[0] and word[0] == 987 and virial[:] == [987] * 6
            continue
        assert state[:] == [0, replay] and failed[0] == 0
        assert published_force[:] == list(range(3 * n))
        actual = word[0] / 2**32
        er = abs(actual - energy) / abs(energy)
        vr = math.dist(virial[:], reference) / math.hypot(*reference)
        assert er < 1.4e-5 and vr < 2e-4, (actual, energy, virial[:], reference)
        outputs.append(word[0])
        print(
            f"height={reach} workers={workers} replay={replay} energy={actual:.10g} "
            f"reference={energy:.10g} relative={er:.6g} virial_relative={vr:.6g}"
        )
    assert outputs[0] == outputs[1]
    e.close()
    return outputs[0]


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rebuild.MDIR_OPT = sys.argv[3]
    for reach in (1, 2):
        assert evaluate(directory, sys.argv[2] == "gpu", reach, 32) == evaluate(
            directory, sys.argv[2] == "gpu", reach, 64
        )
