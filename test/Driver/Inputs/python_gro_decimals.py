"""Independent binary64 oracle for decimal fields of a GRO file."""

import pathlib
import sys

import mdir
import numpy as np


work = pathlib.Path(sys.argv[1])
top = work / "argon.top"
gro = work / "argon.gro"
top.write_text("""[ defaults ]
1 2 no 1 1
[ atomtypes ]
AR 18 39.948 0 A 0.34 0
[ moleculetype ]
AR 1
[ atoms ]
1 AR 1 AR AR 1 0 39.948
[ system ]
Argon
[ molecules ]
AR 256
""")

rng = np.random.default_rng(525)
values = [["0.00000000000000"] * 3 + ["0.000000000000000"] * 3]
for _ in range(252):
    values.append([format(x, ".17g") for x in rng.uniform(-1000, 1000, 6)])
values += [
    ["-0.0", "+0.125", "1e-300", "-0.0", "+1.25", "-1e-300"],
    [".5", "1.", "-1e-200", "-.5", "1.", "2e-200"],
    ["1.0000000000000002", "-999.9999999999999", "1e-12",
     "-1.0000000000000002", "999.9999999999999", "-1e-12"],
]
assert len(values) == 256

with gro.open("w") as file:
    file.write("Argon decimal oracle\n256\n")
    for i, row in enumerate(values, 1):
        file.write(f"{i:5d}{'AR':<5s}{'AR':>5s}{i:5d}")
        file.write("".join(f"{token:>32s}" for token in row))
        file.write("\n")
    file.write("2000 2000 2000\n")

state = mdir.load_gromacs(str(top), str(gro)).make_state()
expected = np.asarray([[float(x) for x in row] for row in values],
                      dtype=np.float64)
for observed, oracle in ((state.positions, expected[:, :3]),
                         (state.velocities, expected[:, 3:])):
    assert np.array_equal(np.asarray(observed).view(np.uint64),
                          oracle.view(np.uint64))
print("1536 coordinate and velocity fields equal the decimal oracle bitwise")

# A failed direct conversion must take the general reader's diagnostic path.
bad = values[1].copy()
bad[0] = "not-a-number"
with gro.open("w") as file:
    file.write("Argon decimal error\n256\n")
    for i, row in enumerate(values, 1):
        if i == 2:
            row = bad
        file.write(f"{i:5d}{'AR':<5s}{'AR':>5s}{i:5d}")
        file.write("".join(f"{token:>32s}" for token in row))
        file.write("\n")
    file.write("2000 2000 2000\n")
try:
    mdir.load_gromacs(str(top), str(gro))
except mdir.InputError as error:
    assert ":4: cannot read a coordinate in '" in str(error), error
else:
    raise AssertionError("expected a malformed coordinate to be rejected")
print("malformed coordinate retains its line-level diagnostic")
