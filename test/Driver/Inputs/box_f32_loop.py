"""Wind a freely moving massive particle once around the periodic x axis.

The transverse separation keeps the LJ force finite. The large mass makes
deflection negligible over one winding. Halfway around, the pair is outside
the cutoff. The endpoints have the same separation and pair energy.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

mdir, target, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
for structure in (("MATRIX",) if target == "CPU" else ("MATRIX", "GROUPS", "TOWERS")):
    for tilted in (False, True):
        d = root / f"{structure}-{int(tilted)}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "system.top").write_text("""[ defaults ]
1 2 yes 1.0 1.0
[ atomtypes ]
A 18 1e12 0 A 0.34 4.184
[ moleculetype ]
A 1
[ atoms ]
1 A 1 A A 1 0 1e12
[ system ]
periodic winding
[ molecules ]
A 2
""")
        rows = []
        for i, (x, y, vx) in enumerate(((0.125, 1.0, 0.0), (7.875, 1.25, 7.8125)), 1):
            rows.append(f"{i:5d}{'A':<5}{'A':>5}{i:5d}" +
                        "".join(f"{v:12.8f}" for v in (x, y, 1., vx, 0., 0.)))
        box = "8.00000042 8 8" + (" 0 0 1.00000004 0 0.5 -0.25" if tilted else "")
        (d / "system.gro").write_text("winding\n2\n" + "\n".join(rows) + "\n" + box + "\n")
        (d / "run.toml").write_text(f"""[input]
topology = "system.top"
coordinates = "system.gro"
[output]
energy_interval = 256
manifest = "run.jsonl"
energy = "energy.dat"
[energy]
cutoff = 8.0
pairlist_distance = 10.0
electrostatics = "CUTOFF"
dispersion_correction = "NONE"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.002
steps = 512
[ensemble]
ensemble = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
[execution]
threads = 1
target = "{target}"
precision = "MIXED"
neighbor_structure = "{structure}"
""")
        result = subprocess.run([mdir, "run", "run.toml"], cwd=d,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode == 0, result.stdout
        energies = [float(row.split()[3]) for row in (d / "energy.dat").read_text().splitlines()
                    if row.strip() and not row.startswith("#")]
        assert len(energies) == 3, energies
        assert abs(energies[0]) > 0.1 and abs(energies[1]) < 1e-6, energies
        assert abs(energies[2] - energies[0]) <= 2e-6, energies
        event = json.loads((d / "run.jsonl").read_text().splitlines()[0])
        changes = event["effective"]["box_rounding_delta_nm"]
        assert changes == [8.0 - 8.00000042, 0, 0,
                           1.0 - 1.00000004 if tilted else 0, 0, 0], changes
        logged = re.search(r"binary32 box delta .*?: (.*)", result.stdout)
        assert list(map(float, logged[1].split())) == changes
        print(structure, tilted, energies)
