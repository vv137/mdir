"""Dyadic spring forces at a half word and at the signed i64 limits."""
import os
from pathlib import Path
import subprocess
import sys

binary, directory = sys.argv[1:]
work = Path(directory)
work.mkdir(parents=True, exist_ok=True)
(work / "atoms.gro").write_text(
    "Three collinear atoms\n3\n" + "".join(
        f"{1:5}AR   {'AR':>5}{i:5}{x:8.3f}{1.:8.3f}{1.:8.3f}\n"
        for i, x in enumerate((1.0, 1.25, 1.5), 1)
    ) + "3.0 3.0 3.0\n")
(work / "run.toml").write_text('''[input]
format = "GROMACS"
topology = "atoms.top"
coordinates = "atoms.gro"
[output]
energy_interval = 1
[energy]
cutoff = 8.0
pairlist_distance = 10.0
[dynamics]
integrator = "VELOCITY_VERLET"
steps = 1
time_step = 0.000001
seed = 314159
[ensemble]
ensemble = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
[constraints]
hydrogen_bonds = false
rigid_water = false
[execution]
precision = "MIXED"
target = "GPU"
''')
topology = '''[ defaults ]
1 2 no 1.0 1.0
[ atomtypes ]
AR 18 39.948 0.0 A 0.34 0.0
[ moleculetype ]
chain 1
[ atoms ]
1 AR 1 AR AR 1 0.0 39.948
2 AR 1 AR AR 2 0.0 39.948
3 AR 1 AR AR 3 0.0 39.948
[ bonds ]
1 2 1 0.125 {k}
1 3 1 0.375 {k}
[ system ]
Springs
[ molecules ]
chain 1
'''
for name, k, mode in (("ties", 12 * 2**-18, "workunits-check"),
                      ("component overflow", 2**48, "workunits"),
                      ("shared sum overflow", 2**47, "workunits"),
                      ("global sum overflow", 2**47, "integer-direct")):
    (work / "atoms.top").write_text(topology.format(k=k))
    env = dict(os.environ, MDIR_TUPLE_SUMS=mode, MDRT_WAIT="1")
    run = subprocess.run([binary, "run", "run.toml"], cwd=work, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if mode != "workunits-check":
        assert run.returncode and any(s in run.stdout for s in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION", "CUDA_ERROR_LAUNCH_FAILED")), run.stdout
    else:
        assert run.returncode == 0 and "0 word mismatches" in run.stdout, run.stdout
    print(name, "passed")
