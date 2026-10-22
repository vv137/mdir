"""Check the water-box writer at the five-column GRO residue boundary."""

import importlib.util
import pathlib
import sys

import mdir
import numpy as np


work = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("water_box", sys.argv[2])
water_box = importlib.util.module_from_spec(spec)
spec.loader.exec_module(water_box)

waters = 100001
positions = np.zeros((3 * waters, 3), dtype=np.float64)
cell = np.diag([500.0] * 3)
water_box.write(str(work), positions, cell)

gro = work / "system.gro"
with gro.open() as file:
    assert file.readline().strip() == "water for a learned potential"
    assert int(file.readline()) == 3 * waters
    for water in range(waters):
        for atom in range(3):
            line = file.readline().rstrip("\n")
            assert len(line) == 56, (water, atom, len(line))
            if water in (99998, 99999, 100000):
                assert line[:5] == f"{(water + 1) % 100000:5d}", (water, line)
                assert line[5:10] == "MOL  ", (water, line)
                assert line[10:15] == f"{('O', 'H1', 'H2')[atom]:>5s}"
                assert line[15:20] == f"{(3 * water + atom + 1) % 100000:5d}"
                assert [float(line[i:i + 12]) for i in (20, 32, 44)] == [0.0] * 3
    assert len(file.readline().split()) == 9
    assert not file.readline()
print("GRO identifiers wrap from residue 99999 to 0 to 1 in fixed-width fields")

state = mdir.load_gromacs(str(work / "system.top"), str(gro)).make_state()
assert np.asarray(state.positions).shape == (3 * waters, 3)
assert np.count_nonzero(state.positions) == 0
print("MDIR reads all 300003 generated coordinates")
