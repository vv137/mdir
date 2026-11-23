"""Compare moving mixed Amber reductions with the matrix traversal."""
from pathlib import Path
import re
import subprocess
import sys

inputs, work = map(Path, sys.argv[2:])


def run(structure):
    control = work / (structure + ".toml")
    control.write_text(f'''[input]
format = "AMBER"
topology = "{inputs}/dipeptide.prmtop"
coordinates = "{inputs}/dipeptide.inpcrd"
[output]
energy_interval = 2
[energy]
cutoff = 8.0
electrostatics = "PME"
[pme]
grid = [32,32,32]
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 6
[ensemble]
ensemble = "NVE"
temperature = 300.0
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
neighbor_structure = "{structure}"
''' + ('towers_shape = {threads=128, blocks_per_sm=1, panel=64, stripe=64, records=16, split=1, cell_height="FULL"}\n'
       if structure == "TOWERS" else ''))
    output = subprocess.check_output([sys.argv[1], "run", str(control)], text=True)
    print(output, flush=True)
    terms = {}
    for term in ("Lennard-Jones", "Coulomb", "Lennard-Jones 1-4", "Coulomb 1-4",
                 "Coulomb excluded", "total"):
        terms[term] = float(re.search(r"^MDIR: +"+term+r" +([-\d.]+)$", output, re.M)[1])
    virial = re.search(r"the diagonal of the virial.*\nMDIR: +([^\n]+)", output)
    terms.update(zip(("xx", "yy", "zz"), map(float, virial[1].split()), strict=True))
    for step in (0, 2, 4, 6):
        row = re.search(rf"^INFO: +{step} +(.+)$", output, re.M)[1].split()
        terms[f"energy at step {step}"] = float(row[1])
    return terms


matrix, towers = map(run, ("MATRIX", "TOWERS"))
for term in matrix:
    assert abs(matrix[term] - towers[term]) <= 0.05, (term, matrix[term], towers[term])
