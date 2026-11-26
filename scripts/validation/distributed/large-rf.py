"""Print a 2 fs reaction-field benchmark deck for a prepared Amber system.

Pass the cellulose_nve or stmv_npt_4fs directory produced by the Amber
benchmark preparation script. Omit --domains for ordinary MATRIX.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", type=Path)
    parser.add_argument("--precision", choices=("double", "mixed"), required=True)
    parser.add_argument("--domains", type=int, choices=(1, 2, 4, 8))
    args = parser.parse_args()
    topology = (args.system / "system.parm7").resolve(strict=True)
    restart = (args.system / "system.rst7").resolve(strict=True)
    print(f"""[input]
format = "AMBER"
topology = {json.dumps(str(topology), ensure_ascii=False)}
coordinates = {json.dumps(str(restart), ensure_ascii=False)}

[output]
energy = "energy.dat"
energy_interval = 1

[energy]
cutoff = 8.0
pairlist_distance = 10.0
electrostatics = "REACTION_FIELD"
reaction_field_dielectric = 0

[dynamics]
seed = 314159
integrator = "VELOCITY_VERLET"
time_step = 0.002
steps = 204

[ensemble]
ensemble = "NVE"
temperature = 300.0

[constraints]
hydrogen_bonds = true
rigid_water = true
analytic_bonds = true

[boundary]
type = "PERIODIC"

[execution]
target = "GPU"
precision = "{args.precision}"
threads = 1
deterministic = false""")
    if args.domains is None:
        print('neighbor_structure = "MATRIX"')
    else:
        print(f"""
[distribute]
domains = {args.domains}
transport = "nccl"
traversal = "unique"
""", end="")


if __name__ == "__main__":
    main()
