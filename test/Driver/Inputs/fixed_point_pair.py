"""Check integer momentum, residual placement, and a force beyond i32 range."""
import os
import sys
from pathlib import Path

import mdir
from mdir._physical_constants import COULOMB_NATIVE
import numpy as np

mode, directory = sys.argv[1:]
work = Path(directory)
work.mkdir(parents=True, exist_ok=True)
site = mode == "residual"
sum_overflow = mode == "sum-overflow"
expect_overflow = mode == "overflow" or sum_overflow
if site:
    os.environ["MDIR_RIGID_UNITS_TEST"] = "1"
charge = -0.25 / (4096 * COULOMB_NATIVE) if mode != "overflow" else -1000.0
if sum_overflow:
    # Each x contribution is below 2^30 quanta. Their sum exceeds 2^31.
    charge = -0.0004
top = """[ defaults ]
1 2 yes 0.5 0.833333
[ atomtypes ]
O 8 16.0 0 A 0.3 0
H 1 1.0 0 A 0.1 0
E 0 0.0 0 D 0.0 0
I 11 23.0 0 A 0.2 0
"""
if site:
    top += """[ moleculetype ]
WAT 2
[ atoms ]
1 O 1 WAT O 1 0 16
2 H 1 WAT H1 1 0 1
3 H 1 WAT H2 1 0 1
4 E 1 WAT EP 1 1 0
[ settles ]
1 1 0.1 0.16
[ virtual_sites3 ]
4 1 2 3 1 0.5 0.5
[ exclusions ]
1 2 3 4
2 1 3 4
3 1 2 4
4 1 2 3
"""
else:
    top += f"""[ moleculetype ]
A 1
[ atoms ]
1 I 1 A A 1 {1e6 if sum_overflow else 1} 23
"""
top += f"""[ moleculetype ]
B 1
[ atoms ]
1 I 1 B B 1 {charge:.17g} 23
[ system ]
pair
[ molecules ]
{'WAT' if site else 'A'} 1
B {4 if sum_overflow else 1}
"""
(work / "pair.top").write_text(top)
first = ([("O", .44, .5), ("H1", .50, .58), ("H2", .50, .42), ("EP", .50, .5)]
         if site else [("A", .50, .5)])
rows = []
ys = [.47, .49, .51, .53] if sum_overflow else [.5]
for n, (name, x, y) in enumerate(first + [("B", 1., y) for y in ys], 1):
    residue = 1 if n <= len(first) else n - len(first) + 1
    label = ("WAT" if site else "A") if residue == 1 else "B"
    rows.append(f"{residue:5d}{label:<5}{name:>5}{n:5d}{x:8.3f}{y:8.3f}{.5:8.3f}")
(work / "pair.gro").write_text(f"pair\n{len(rows)}\n" + "\n".join(rows) + "\n3 3 3\n")
loaded = mdir.load_gromacs(str(work / "pair.top"), str(work / "pair.gro"))
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = .8, .9
system.periodic, system.rigid_water = True, site
system.electrostatics = mdir.Electrostatics.Cutoff
execution = mdir.Execution()
execution.target, execution.precision = mdir.Target.GPU, mdir.Precision.Mixed
execution.neighbor_structure = mdir.NeighborStructure.Groups
ensemble = mdir.Ensemble()
ensemble.kind = mdir.EnsembleKind.NVE
sim = mdir.Simulation(mdir.compile(system, state, mdir.Integrator(), ensemble,
                                   execution, mdir.Schedule()))
try:
    sim.run(0, energy=True)
except RuntimeError as error:
    if not expect_overflow or "fixed-point force exceeds its range at particle" not in str(error):
        raise
    assert "component x" in str(error), str(error)
    print("overflow identified the particle and x component")
else:
    assert not expect_overflow, "the out-of-range force was accepted"
    forces = np.asarray(sim.state().forces) * 4096
    expected = np.zeros_like(forces)
    expected[1 if site else 0, 0], expected[-1, 0] = 1, -1
    np.testing.assert_array_equal(forces, expected)
    print("exact member quanta and zero net force")
