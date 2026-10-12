"""Under leapfrog a step that writes a row of energies leaves the state of
the plain step (#285).

    leapfrog_energy_step.py MDIR INPUTS WORK TARGET PRECISION [CASE...]

Without the name of a case all the cases run; with names, those cases in
the order of the table, and a name that the table does not have, or a
name given twice, is an error. The last line counts the cases that ran, so
that a test checks how many it believes it runs.

The dipeptide in water (1,168 particles, PME), 20 steps from velocities
drawn with a seed, in the deterministic mode. For each case two runs that
differ only in `energy_interval` (10 and 20, so that step 10 writes a row
in one and is a plain step in the other; with a barostat it is then the
step that closes a period of coupling with or without the energies) end in
the same state to the bit, which `mdir checkpoint` compares: at constant
energy, with velocity rescaling, a Nose-Hoover chain, and Langevin
dynamics, and with stochastic cell rescaling by each count of its work and
each coupling of the cell.

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
CHAIN = '[thermostat]\nmethod = "NOSE-HOOVER"\ntime_constant = 0.5\n'


def barostat(work, coupling="ISOTROPIC"):
    return f'[barostat]\nmethod = "C-RESCALE"\nwork = "{work}"\ncoupling = "{coupling}"\n'


# The name, the ensemble, the tables of the couplings, whether the bonds
# are constrained, and whether velocity Verlet takes the same steps: it
# does not where a thermostat alone scales the stored velocities, those of
# the half step under leapfrog (design-m1.md, Section 23).
#
# What the two runs of a case take at step 10: without a barostat a step
# of energy against a plain step; with one, whose period is 10 steps, the
# step that closes the period with against without the energies. With
# "TROTTER_FIRST_ORDER" that is the scaling step with against without the
# virial. With "TROTTER", "EXACT", and "FIRST_ORDER" both take the virial
# for the barostat, so the two runs take the same programs but for the
# energies and what the log measures, and the plain steps of the period
# are the same in both: what tells there that the plain step, the step
# with the virial, and the scaling step agree is velocity Verlet.
#
# With constraints the step has three forms (#285), and the tests that do
# not run all the cases run one case of each (leapfrog-energy-step.test):
# under Langevin dynamics the velocities change within the drift and
# every program takes the projected form ("langevin"); where they do not
# change, the step with the virial keeps the kick of the plain step and
# takes the drift of velocity Verlet on the side ("nve"); and the scaling
# step of the barostat, which "TROTTER_FIRST_ORDER" takes without the
# virial, has the projected form with and without it
# ("v-rescale-anisotropic").
CASES = [
    ("nve", "NVE", "", True, True),
    ("nve-free", "NVE", "", False, True),
    ("v-rescale", "NVT", RESCALE + "interval = 10\n", True, False),
    ("nose-hoover", "NVT", CHAIN + "interval = 10\n", True, False),
    ("langevin", "NVT", LANGEVIN + "interval = 0\n", True, True),
    ("langevin-free", "NVT", LANGEVIN + "interval = 0\n", False, True),
    ("langevin-trotter", "NPT", LANGEVIN + barostat("TROTTER"), True, True),
    ("langevin-trotter-first-order", "NPT", LANGEVIN + barostat("TROTTER_FIRST_ORDER"),
     True, True),
    ("v-rescale-trotter", "NPT", RESCALE + barostat("TROTTER"), True, True),
    ("v-rescale-trotter-first-order", "NPT", RESCALE + barostat("TROTTER_FIRST_ORDER"),
     True, True),
    ("v-rescale-exact", "NPT", RESCALE + barostat("EXACT"), True, True),
    ("v-rescale-first-order", "NPT", RESCALE + barostat("FIRST_ORDER"), True, True),
    ("nose-hoover-trotter", "NPT", CHAIN + barostat("TROTTER"), True, True),
    ("v-rescale-semi-isotropic", "NPT",
     RESCALE + barostat("TROTTER_FIRST_ORDER", "SEMI_ISOTROPIC"), True, True),
    ("v-rescale-anisotropic", "NPT",
     RESCALE + barostat("TROTTER_FIRST_ORDER", "ANISOTROPIC"), True, True),
    ("v-rescale-anisotropic-exact", "NPT", RESCALE + barostat("EXACT", "ANISOTROPIC"),
     True, True),
    ("langevin-anisotropic", "NPT", LANGEVIN + barostat("TROTTER", "ANISOTROPIC"),
     True, True),
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


names = sys.argv[6:]
unknown = sorted(set(names) - {case[0] for case in CASES})
if unknown:
    sys.exit(f"leapfrog_energy_step.py: no such case: {' '.join(unknown)}")
if len(set(names)) != len(names):
    sys.exit(f"leapfrog_energy_step.py: a case is given twice: {' '.join(names)}")
selected = [case for case in CASES if not names or case[0] in names]

failed = 0
for name, kind, couplings, constrained, verlet in selected:
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
print(f"leapfrog {target} {precision}: {len(selected) - failed} of {len(selected)} cases passed")
sys.exit(1 if failed else 0)
