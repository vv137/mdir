"""Compare integer force words on moving, periodically rebuilt pressure frames."""

import ctypes
import itertools
import math
import sys
from pathlib import Path

import towers_rebuild as rebuild


def frames(directory, fused):
    dims = (3, 3, 3)
    n = 97
    source = Path(__file__).with_name("towers-force.mlir").read_text()
    source = source.replace("  md_exec.pair_for", """  %zero = arith.constant dense<0.0> : vector<3xf64>
  %vir = md_exec.pair_for""")
    source = source.replace("cutoff(1.0) exchange [antisymmetric]", """reduce(%zero : vector<3xf64>)
      cutoff(1.0) weights [0.5] exchange [antisymmetric, symmetric]""")
    source = source.replace("md_exec.yield %df : vector<3xf32>", """%fd = arith.extf %df : vector<3xf32> to vector<3xf64>
    %dd = arith.extf %d : vector<3xf32> to vector<3xf64>
    %vdiag = arith.mulf %dd, %fd : vector<3xf64>
    md_exec.yield %df, %vdiag : vector<3xf32>, vector<3xf64>""")
    source = source.replace("} : !nl, memref<?x3xf64>", "} : !nl, memref<?x3xf64> -> vector<3xf64>")
    image = rebuild.compile_entries(directory, dims, 1, n, 128, True, True, False,
                                    ("spring", source.encode()), {"direct-virial": int(fused),
                                    "certify-force": 1, "force-panel": 32, "force-stripe": 32, "force-records": 8})
    entries = rebuild.Entries(image, True)
    frame = rebuild.Rebuild(entries, dims, 1, n, 128, True)
    frame.charge[:] = [1 if i % 3 else -1 for i in range(n)]
    frame.types[:] = [i % 2 for i in range(n)]
    table = entries.array(rebuild.F, 4)
    table[:] = [1.531, -0.753, -0.753, 2.039]
    force = entries.array(ctypes.c_int64, n * 3)
    flag = entries.array(rebuild.I, 2)
    failures = entries.array(ctypes.c_int64, 1)
    virial = entries.array(rebuild.D, 6)
    result = []
    for step in range(4):
        frame.box[:] = [6 + 3 * step / 16] * 3
        xyz = [
            tuple(((i * (a + 3) + step * (i % 7 - 3)) % 80) / 64
                  for a in range(3))
            for i in range(n)
        ]
        frame.raw[:] = [p[a] for a in range(3) for p in xyz]
        frame.setup()
        frame.step(step % 2)
        force[:] = [0] * (n * 3)
        failures[0] = 0
        flag[:] = [0, 0]
        virial[:] = [0] * 6
        entries.run("force_virial" if fused else "force", frame.cells * 128,
                    frame.spans, frame.packed, frame.ids, frame.geometry, force,
                    flag, rebuild.F(2**18), failures, frame.charge, frame.types, table,
                    *([virial] if fused else []), shared=8 * 8)
        entries.sync()
        assert frame.status[0] == flag[0] == failures[0] == 0
        assert any(force[:])
        if fused:
            reference = [0.0] * 3
            for i, j in itertools.combinations(range(n), 2):
                delta = [a - b for a, b in zip(xyz[i], xyz[j])]
                if sum(d * d for d in delta) < 1:
                    k = table[2 * frame.types[i] + frame.types[j]]
                    k *= frame.charge[i] * frame.charge[j]
                    for a, d in enumerate(delta):
                        reference[a] += d * rebuild.F(k * d).value
            result_diag = [virial[j] for j in (0, 3, 5)]
            assert all(math.isclose(x, y, abs_tol=1e-9)
                       for x, y in zip(reference, result_diag)), (reference, result_diag)
        result.append(force[:])
    entries.close()
    return result


rebuild.MDIR_OPT = sys.argv[2]
directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
assert frames(directory, False) == frames(directory, True)
