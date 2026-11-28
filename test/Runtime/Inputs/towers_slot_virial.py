"""Check exact subtraction of narrow and wide excluded pairs in staged images."""

import ctypes as C
from pathlib import Path
import sys

import towers_rebuild as rebuild


def source():
    force = Path(__file__).with_name("towers-force.mlir").read_text()
    start = force.index("  md_exec.pair_for")
    end = force.index("  return", start)
    body = force[start:end]
    direct = body.replace("  md_exec.pair_for", "  %virial = md_exec.pair_for")
    direct = direct.replace("cutoff(1.0) exchange [antisymmetric]",
                            "reduce(%zero : vector<3xf64>)\n"
                            "      cutoff(1.0) weights [0.5] "
                            "exchange [antisymmetric, symmetric]")
    direct = direct.replace(
        "    md_exec.yield %df : vector<3xf32>",
        """    %d64 = arith.extf %d : vector<3xf32> to vector<3xf64>
    %f64 = arith.extf %df : vector<3xf32> to vector<3xf64>
    %w = arith.mulf %d64, %f64 : vector<3xf64>
    md_exec.yield %df, %w : vector<3xf32>, vector<3xf64>""",
    )
    direct = direct.replace("} : !nl, memref<?x3xf64>",
                            "} : !nl, memref<?x3xf64> -> vector<3xf64>")
    correction = body.replace("    %spring = memref.load %k[%a, %b] : memref<2x2xf32>",
                              "    %spring = arith.constant 0.0 : f32")
    before = force[:start] + "  %zero = arith.constant dense<0.0> : vector<3xf64>\n"
    return (before + direct + correction + force[end:]).encode()


def check(work, workers, split):
    dims = (6, 6, 6)
    # The first excluded pair crosses a periodic face at 0.05 angstrom.
    # The second shares a staged slot with a retained interaction.
    xyz = [(0.002, 0.5, 0.5), (5.997, 0.5, 0.5),
           (3.125, 3.5, 3.5), (3.25, 3.5, 3.5), (3.5, 3.5, 3.5)]
    exclusions = (0, 1, 2, 3)
    n = len(xyz)
    image = rebuild.compile_entries(
        work, dims, 1, n, 8, True, True, False, ("spring", source()),
        {"certify-force": 1, "exclusions": len(exclusions) // 2,
         "packed-charge-type": 1, "direct-virial": 1,
         "force-workers": workers, "force-panel": workers, "force-stripe": workers,
         "force-records": workers // 32, "force-split": split},
    )
    e = rebuild.Entries(image, True)
    r = rebuild.Rebuild(e, dims, 1, n, 8, True)
    r.box[:] = [6] * 3
    r.setup()
    r.raw[:] = [v[a] for a in range(3) for v in xyz]
    r.charge[:] = [1] * n
    r.types[:] = [1, 1, 1, 1, 0]
    geom = e.array(rebuild.F, 7)
    e.run("prepare_force", 1, r.geometry, geom)
    r.step()
    table = e.array(rebuild.F, 4)
    bonds = e.array(rebuild.I, len(exclusions))
    bonds[:] = exclusions
    state = e.array(rebuild.I, 2)
    failed = e.array(C.c_int64, 1)
    force = e.array(C.c_int64, 3 * n)
    virial = e.array(C.c_uint64, 12)
    items = r.cells * split * workers + len(exclusions) // 2
    reference = None
    for excluded_spring in (0, 2**15, 2**30, 2**15):
        table[:] = [8, 8, 8, excluded_spring]
        force[:] = [0] * len(force)
        virial[:] = [0] * len(virial)
        state[:] = [0, 0]
        failed[0] = 0
        args = (r.spans, r.packed, r.ids, geom, force, state, rebuild.F(2**18),
                failed, bonds, r.inverse, r.raw, r.box, r.packed_types, table)
        kw = dict(block=workers, shared=8 * (workers // 32))
        e.run("force_virial", items, *args, virial, **kw)
        e.sync()
        assert state[:] == [0, 0] and failed[0] == 0, (state[:], failed[0])
        # Two active springs: 8 * (0.375^2 + 0.25^2) = 1.625.
        expected = [0] * 12
        expected[0] = 13 * 2**47
        assert virial[:] == expected, (excluded_spring, virial[:], expected)
        observed = (tuple(force), tuple(virial))
        if reference is None:
            reference = observed
        assert observed == reference, (excluded_spring, observed, reference)
        force[:] = [0] * len(force)
        e.run("force", items, *args, **kw)
        e.sync()
        assert tuple(force) == reference[0], (excluded_spring, force[:], reference[0])
    e.close()
    return reference


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rebuild.MDIR_OPT = sys.argv[2]
    assert check(directory, 32, 1) == check(directory, 64, 2)
