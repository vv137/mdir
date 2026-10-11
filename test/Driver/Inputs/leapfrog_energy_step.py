"""Under leapfrog a step that writes a row of energies leaves the state of
the plain step (#285).

    leapfrog_energy_step.py MDIR INPUTS WORK TARGET PRECISION

The dipeptide in water (1,168 particles, PME), 20 steps from velocities
drawn with a seed, in the deterministic mode. For each case two runs that
differ only in `energy_interval` (10 and 20, so that step 10 writes a row
in one and is a plain step in the other; with a barostat it is then the
step that scales the cell with or without the energies) end in the same
state to the bit, which `mdir checkpoint` compares.

With constraints, where the velocities change within the drift (the
friction and the noise of Langevin dynamics, the scaling of the barostat
of Trotter type), the steps are those of velocity Verlet from the
projected velocities of the time of the positions: the positions of
leapfrog at step 20 are then those of velocity Verlet within `CLOSE`, in
double precision. A kick of a whole step without the projection left them
2e-5 nm apart under Langevin dynamics and 8e-7 nm under the barostat.
"""
import pathlib
import subprocess
import sys

mdir, inputs, work, target, precision = sys.argv[1:6]
work = pathlib.Path(work)
CLOSE = 1e-11  # nm

LANGEVIN = '[thermostat]\nmethod = "LANGEVIN"\nfriction = 1.0\n'
RESCALE = '[thermostat]\nmethod = "V-RESCALE"\ntime_constant = 0.1\n'
FIRST = '[barostat]\nmethod = "C-RESCALE"\nwork = "TROTTER_FIRST_ORDER"\n'
TROTTER = '[barostat]\nmethod = "C-RESCALE"\n'
# The name, the ensemble, the tables of the couplings, whether the bonds
# are constrained, and whether velocity Verlet takes the same steps.
CASES = [
    ("nve", "NVE", "", True, True),
    ("nve-free", "NVE", "", False, True),
    # Velocity rescaling scales the stored velocities, those of the half
    # step under leapfrog: not the steps of velocity Verlet.
    ("v-rescale", "NVT", RESCALE + "interval = 10\n", True, False),
    ("langevin", "NVT", LANGEVIN + "interval = 0\n", True, True),
    ("langevin-free", "NVT", LANGEVIN + "interval = 0\n", False, True),
    ("langevin-npt", "NPT", LANGEVIN + TROTTER, True, True),
    ("langevin-npt-first-order", "NPT", LANGEVIN + FIRST, True, True),
    ("v-rescale-npt-first-order", "NPT", RESCALE + FIRST, True, True),
]


def run(name, integrator, kind, couplings, constrained, interval):
    directory = work / f"{name}-{integrator.lower()}-{interval}"
    directory.mkdir(parents=True)
    constraints = ("[constraints]\nhydrogen_bonds = true\nrigid_water = true\n"
                   if constrained else "")
    pressure = "pressure = 2000.0\n" if kind == "NPT" else ""
    (directory / "run.toml").write_text(f"""[input]
topology = "{inputs}/dipeptide/dipeptide.prmtop"
coordinates = "{inputs}/dipeptide/dipeptide.inpcrd"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "PME"
[pme]
grid = [32, 32, 32]
{constraints}[dynamics]
integrator = "{integrator}"
time_step = {0.002 if constrained else 0.0005}
steps = 20
seed = 2024
[output]
energy_interval = {interval}
checkpoint = "run.h5"
checkpoint_interval = 20
[ensemble]
ensemble = "{kind}"
temperature = 300.0
{pressure}{couplings}[boundary]
type = "PERIODIC"
[execution]
target = "{target}"
precision = "{precision}"
deterministic = true
""")
    result = subprocess.run([mdir, "run", "run.toml"], cwd=directory,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    return str(directory / "run.h5")


def positions(checkpoint):
    text = subprocess.run([mdir, "checkpoint", "--print=positions", checkpoint],
                          check=True, stdout=subprocess.PIPE, text=True).stdout
    return [[float(v) for v in line.split()[2:5]] for line in text.splitlines()]


failed = 0
for name, kind, couplings, constrained, verlet in CASES:
    rows = run(name, "LEAPFROG", kind, couplings, constrained, 10)
    plain = run(name, "LEAPFROG", kind, couplings, constrained, 20)
    same = subprocess.run([mdir, "checkpoint", rows, plain], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True).stdout.strip()
    ok = same == "the states are identical"
    line = f"{target} {precision} {name}: {same}"
    if verlet and precision == "DOUBLE":
        other = positions(run(name, "VELOCITY_VERLET", kind, couplings, constrained, 20))
        apart = max(abs(a - b) for p, q in zip(positions(plain), other) for a, b in zip(p, q))
        ok = ok and apart < CLOSE
        line += f"; within {apart:.1e} nm of velocity Verlet"
    failed += not ok
    print(line + ("" if ok else ": FAILED"), flush=True)
print(f"leapfrog {target} {precision}: {len(CASES) - failed} of {len(CASES)} cases passed")
sys.exit(1 if failed else 0)
