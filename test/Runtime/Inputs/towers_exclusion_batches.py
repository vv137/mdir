"""Check compact exclusion blocks and their last partial slice against integer sums."""

import ctypes as C
from itertools import combinations
from pathlib import Path
import sys

import towers_rebuild as rebuild
from towers_outputs import source


def check(directory):
    workers = 32
    n = 193
    dims = (6, 6, 6)
    # All pairs are inside the cutoff. Dyadic coordinates give exact Q10
    # force and Q32 energy words for both the direct and corrected springs.
    x = [(7 * i) % 24 for i in range(n)]
    bonds = [(i, j) for i, j in combinations(range(n), 2) if (i + j) % 19]
    excluded = set(bonds)
    expected = [0] * (3 * n)
    energy = 0
    for i, j in combinations(range(n), 2):
        k = 2 if (i, j) in excluded else 8
        dx = x[i] - x[j]
        expected[3 * i] += k * dx * 32
        expected[3 * j] -= k * dx * 32
        energy -= k * dx**2 * (1 << 21)

    image = rebuild.compile_entries(
        directory, dims, 1, n, n, True, True, False, ("spring", source()),
        {"certify-force": 1, "exclusions": len(bonds), "packed-charge-type": 1,
         "energy-body": "potential", "direct-virial": 1, "force-workers": workers,
         "force-panel": workers, "force-stripe": workers, "force-records": 1,
         "force-residency": 1, "force-split": 1},
    )
    e = rebuild.Entries(image, True)
    r = rebuild.Rebuild(e, dims, 1, n, n, True)
    r.box[:] = [6] * 3
    r.setup()
    r.raw[:] = [v / 32 + 0.125 for v in x] + [0.125] * (2 * n)
    r.charge[:] = [1] * n
    r.types[:] = [0] * n
    r.step()
    geometry = e.array(rebuild.F, 7)
    e.run("prepare_force", 1, r.geometry, geometry)
    table = e.array(rebuild.F, 4)
    table[:] = [8] * 4
    topology = e.array(rebuild.I, 2 * len(bonds))
    topology[:] = [v for edge in bonds for v in edge]
    state = e.array(rebuild.I, 2)
    failed = e.array(C.c_int64, 1)
    force = e.array(C.c_int64, 3 * n)
    sums = e.array(C.c_int64, 2)
    force[:] = [0] * (3 * n)
    sums[:] = state[:] = [0, 0]
    failed[0] = 0

    # Launch the exact compact grid. Missing or duplicate exclusion rows
    # affect the force distribution and the signed two-limb energy.
    width = 4 * workers
    blocks = r.cells + (len(bonds) + width - 1) // width
    prefix = (r.spans, r.packed, r.ids, geometry)
    fields = (failed, topology, r.inverse, r.raw, r.box, r.packed_types, table)
    for name, dest, scale in (("force", force, 1024), ("energy", sums, 2**32)):
        e.run(name, blocks * workers, *prefix, dest, state, rebuild.F(scale),
              *fields, block=workers, shared=8)
    e.sync()
    assert force[:] == expected
    assert sums[:] == [energy, -1]
    assert state[:] == [0, 0] and failed[0] == 0
    e.close()


if __name__ == "__main__":
    rebuild.MDIR_OPT = sys.argv[2]
    check(Path(sys.argv[1]))
