"""Accumulate sub-ulp isotropic strain in a force-free, nearly cold gas."""
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

fixture, root = map(Path, sys.argv[1:3])
binary, target = sys.argv[3:]
template = "[input]\n" + fixture.read_text().split("\n[input]\n", 1)[1]
template = template.replace("Inputs/", str(fixture.parent / "Inputs") + "/")
template = template.replace("0.238465", "0.0").replace("0.3253", "0.0")
template = template.replace("0.004", "0.002").replace("23.2", "25.0")
template = template.replace('"NVE"', '"NPT"\npressure = 0.01')
template = template.replace("120.0", "1.0e-30")
template = template.replace("= 50", "= 1")
template += """
checkpoint = "state.h5"
checkpoint_interval = 1
[thermostat]
method = "V-RESCALE"
interval = 1
[barostat]
method = "C-RESCALE"
work = "EXACT"
compressibility = 0.001
time_constant = 1.0
[execution]
deterministic = true
threads = 1
"""


def run(mode, name, steps, checkpoint=None):
    path = root / mode / name
    path.mkdir(parents=True, exist_ok=True)
    control = template.replace("= 200", f"= {steps}")
    control += f'precision = "{mode}"\ntarget = "{target}"\n'
    if checkpoint:
        control = control.replace("[input]", f'[input]\ncheckpoint = "{checkpoint}"')
    (path / "run.toml").write_text(control)
    result = subprocess.run([binary, "run", "run.toml"], cwd=path,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    with h5py.File(path / "state.h5") as f:
        step = f["particles/all/box/edges"][:]
        master = f["parameters/mdir/master_cell"][:3] if mode != "double" else step
        positions = f["particles/all/position/value"][0]
    return master, step, positions


# With negligible kinetic pressure and noise, sqrt(V) changes by the
# factor below. The nm edge changes by less than half a binary32 spacing.
factor = (1 - 0.5 * 0.001 * 0.002 * 0.01) ** (2 / 3)
for mode in ("double", "mixed", "single"):
    early, step1, x1 = run(mode, "first", 1)
    final, step16, x = run(mode, "whole", 16)
    assert np.all(final < early) and np.all(early < 2.5)
    np.testing.assert_allclose(early, 2.5 * factor, rtol=0, atol=2e-14)
    np.testing.assert_allclose(final, 2.5 * factor**16, rtol=0, atol=4e-14)
    if mode != "double":
        np.testing.assert_array_equal(step1, np.full(3, 2.5))
        np.testing.assert_array_equal(step16, final.astype("f"))
        assert np.all(step16 < step1)
        tolerance = 6e-8 if mode == "single" else 1e-14
        np.testing.assert_allclose(x / step16, x1 / step1, atol=tolerance, rtol=0)
        resumed = run(mode, "resumed", 15, root / mode / "first" / "state.h5")
        for actual, expected in zip(resumed, (final, step16, x)):
            np.testing.assert_array_equal(actual, expected)
    print(mode, "first master", early[0], "final master", final[0],
          "final step box", step16[0])

# Independent rounding deforms a tilted step cell slightly even under
# isotropic pressure. Test the full map with force-free, unconstrained atoms.
points = np.array([[0.24, 0.67, 0.83], [1.31, 0.84, 0.45], [0.55, 1.23, 1.37]])
cell = np.array([[2.5, 0, 0], [0.36812344, 2.5, 0], [0.11626, -0.243137, 2.5]])
(root / "gas.top").write_text("""[ defaults ]
1 2 yes 1.0 1.0
[ atomtypes ]
A 18 40 0 A 0.34 0
[ moleculetype ]
A 1
[ atoms ]
1 A 1 A A 1 0 40
[ system ]
tilted gas
[ molecules ]
A 3
""")
rows = [f"{i:5d}{'A':<5}{'A':>5}{i:5d}" + "".join(f"{v:12.8f}" for v in point)
        for i, point in enumerate(points, 1)]
(root / "gas.gro").write_text("tilted gas\n3\n" + "\n".join(rows) +
                              "\n2.5 2.5 2.5 0 0 0.36812344 0 0.11626 -0.243137\n")
template = template[template.index("[dynamics]"):]
template = template[:template.index("[boundary]")] + template[template.index("[output]"):]
template = (f'[input]\nformat = "GROMACS"\ntopology = "{root}/gas.top"\n'
            f'coordinates = "{root}/gas.gro"\n[boundary]\ntype = "PERIODIC"\n'
            '[energy]\ncutoff = 8.0\npairlist_distance = 10.0\n' + template)
template = template.replace("interval = 1", "interval = 2")
initial = cell.astype("f").astype("d")
for work in ("EXACT", "TROTTER"):
    template = template.replace('work = "EXACT"', f'work = "{work}"')
    _, final, placed = run("mixed", f"tilted-{work}", 16)
    fractional = points @ np.linalg.inv(initial)
    expected = fractional @ final
    diagonal_only = points * (final.diagonal() / initial.diagonal())
    assert np.max(np.abs(expected - diagonal_only)) > 1e-9
    np.testing.assert_allclose(placed, expected, rtol=0, atol=1e-13)
