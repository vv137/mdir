"""Check the fused pressure virial against an isolated Lennard-Jones pair."""

import math
from pathlib import Path
import re
import subprocess
import sys

mdir = sys.argv[1]
work = Path(sys.argv[2]).resolve()

# The excluded Lennard-Jones sites straddle the periodic boundary. Their
# IDs span more than one exclusion window, with a short edge in that window.
# The remaining Lennard-Jones pair occupies the other half of the box.
xyz = [(0.05, 0.5, 0.5), (0.25, 0.5, 0.5), (2.0, 2.0, 2.0),
       (2.4, 2.0, 2.0)]
xyz += [(0.3 + 0.1 * i, 3.1, 3.0) for i in range(36)]
xyz += [(3.85, 0.5, 0.5)]

atoms = []
positions = []
for i, (x, y, z) in enumerate(xyz, 1):
    kind = "A" if i <= 4 or i == len(xyz) else "Z"
    atoms.append(f"{i} {kind} 1 BOX {kind} {i} 0.0 40.0")
    positions.append(f"{1:5d}{'BOX':<5}{kind:>5}{i:5d}"
                     f"{x:8.3f}{y:8.3f}{z:8.3f}")
(work / "system.gro").write_text(
    f"excluded-pair virial\n{len(xyz)}\n" + "\n".join(positions) + "\n4.0 4.0 4.0\n")
(work / "system.top").write_text(
    """[ defaults ]
1 2 yes 1.0 1.0
[ atomtypes ]
A 18 40.0 0.0 A 0.34 1.0
Z 18 40.0 0.0 A 0.00 0.0
[ moleculetype ]
BOX 0
[ atoms ]
""" + "\n".join(atoms) + """
[ exclusions ]
1 2 41
2 41
[ system ]
excluded-pair virial
[ molecules ]
BOX 1
""")
(work / "run.toml").write_text("""[input]
format = "GROMACS"
topology = "system.top"
coordinates = "system.gro"
[output]
energy_interval = 50
[energy]
cutoff = 8.0
electrostatics = "CUTOFF"
dispersion_correction = "NONE"
[constraints]
hydrogen_bonds = false
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.00001
steps = 50
seed = 12345
[ensemble]
ensemble = "NPT"
temperature = 300.0
pressure = 1.0
[thermostat]
method = "V-RESCALE"
time_constant = 1.0
[barostat]
method = "C-RESCALE"
time_constant = 1.0
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
neighbor_structure = "TOWERS"
towers_shape = {threads=128, blocks_per_sm=1, panel=64, stripe=64, records=16, split=1, cell_height="FULL"}
step_graph = "LAUNCH"
spatial_order = false
""")

output = subprocess.check_output([mdir, "run", "run.toml"], cwd=work, text=True)
(work / "run.log").write_text(output)
assert '"force_kernel":"towers_force_virial"' in output, output
line = re.search(r"the diagonal of the virial.*\nMDIR: +([^\n]+)", output)
observed = [float(part) for part in line[1].split()]
ratio6 = (0.34 / 0.4)**6
expected = (4 * (12 * ratio6**2 - 6 * ratio6) / 4.184, 0.0, 0.0)
assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=2e-5)
           for a, b in zip(observed, expected, strict=True)), (observed, expected)
