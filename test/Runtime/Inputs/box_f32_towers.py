"""Periodic winding through the host TOWERS rebuild, pack and energy entries."""
import ctypes as C
from pathlib import Path
import sys

import towers_outputs
import towers_rebuild as rebuild

directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
rebuild.MDIR_OPT = sys.argv[2]
for tilted in (False, True):
    dims = (4, 4, 4)
    image = rebuild.compile_entries(
        directory, dims, 1, 3, 8, True, False, False, ("spring", towers_outputs.source()),
        {"triclinic": int(tilted), "certify-force": 1, "exclusions": 1,
         "packed-charge-type": 1, "energy-body": "potential",
         "force-workers": 32, "force-panel": 32, "force-stripe": 32, "force-records": 1})
    e = rebuild.Entries(image, False)
    r = rebuild.Rebuild(e, dims, 1, 3, 8, True)
    edge = C.c_float(8.00000042).value
    if tilted:
        r.box = e.array(C.c_double, 9)
        r.geometry = e.array(C.c_double, 39)
        r.box[:] = [edge, 0, 0, 1, edge, 0, 0.5, -0.25, edge]
    else:
        r.box[:] = [edge] * 3
    r.setup()
    r.charge[:] = [1, 1, 0]
    r.types[:] = [0, 0, 0]
    table = e.array(rebuild.F, 4)
    table[:] = [8] * 4
    exclusion = e.array(rebuild.I, 2)
    exclusion[:] = [0, 2]
    sums = e.array(C.c_int64, 2)
    state = e.array(rebuild.I, 2)
    failed = e.array(C.c_int64, 1)
    energies = []
    for x in (7.875, 8.5, 11.875, 15.875):
        # The triclinic pack entry receives fractional coordinates.
        values = [0.125, x, 2, 0, 0, 3, 0, 0, 0]
        r.raw[:] = [v / edge if tilted else v for v in values]
        r.step()
        sums[:] = [0, 0]
        e.run("energy", r.cells * 32 + 1, r.spans, r.packed, r.ids, r.geometry,
              sums, state, rebuild.F(2**32), failed, exclusion, r.inverse,
              r.raw, r.box, r.packed_types, table)
        assert state[:] == [0, 0] and failed[0] == 0
        energies.append(sums[0] / 2**32)
    assert energies == [-0.25, -0.5625, 0, -0.25], (tilted, energies)
    e.close()
