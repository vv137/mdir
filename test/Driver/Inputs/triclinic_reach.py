"""A narrow tilted cell that a barostat shrinks until the reach of the
neighbor structure is well beyond half of the least of a_x, b_y, c_z
(#258): the potential energy of every frame of `mdir run` against a sum in
NumPy over every image of every pair within the cutoff.

    triclinic_reach.py DIRECTORY MDIR TARGET STRUCTURE [SHAPE [REACH [PRUNED]]]

216 Lennard-Jones particles (argon, a plain cutoff of 9.2 Å, no correction
for the dispersion) at 300 K in a reduced cell with every tilt on its
bound, b_x = a_x/2, c_x = -a_x/2, c_y = b_y/2, and a_x = b_y = c_z = 22 Å.
The pairlist distance is 11 Å, half of the least of the diagonal at the
start. Stochastic cell rescaling at 20,000 bar takes the diagonal to about
19.2 Å in 400 steps, so the lists reach past half of it throughout, by a
seventh at the end, while the cutoff stays below it (I2). The frames, in
H5MD with the positions in f64, and the rows of the energy file are written
every 5 steps, those of the coupling.

A pair that a list lacks when it comes within the cutoff takes 0.0024
kcal/mol from the potential, and no pair adds to it: the potential of a
frame that lacks one differs from the sum by 240 times the tolerance, 1e-5
kcal/mol, which is ten times the last digit of the energy file. A build
of the matrix that tests the image of the one pass alone gives 9 such
frames of the 79 on the CPU.

SHAPE is `tilted` (the default) or `cube`, the same diagonal with no tilt;
REACH is the pairlist distance in nm, 1.1 by default, and PRUNED the reach
of the inner list of a dual list. The groups keep an entry for each image
within their reach; in the cube with a reach of 1.7 nm, 0.77 to 0.89 of
the edge, lists that took the nearest image and the one on the other side
of the boundary along each axis alone lost pairs (#263).

With `python` for MDIR the run is a Python simulation of the same model
(D[python-triclinic-npt]), whose reporters write the two files: the neighbor
matrix, which is the structure of every Python simulation (#270), under the
barostat of a simulation. The Python model has no key for the work of the
barostat, so its run has the default, `work = "TROTTER"`, where the control
file here has `"FIRST_ORDER"`; the rows of the two are of other
configurations (see below).
"""
import itertools
import pathlib
import subprocess
import sys

import numpy as np
import mdir

work = pathlib.Path(sys.argv[1])
cli, target, structure = sys.argv[2:5]
shape = sys.argv[5] if len(sys.argv) > 5 else "tilted"
pruned = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0

SIGMA, EPSILON, MASS = 0.34, 0.996, 39.948  # nm, kJ/mol, amu
CUTOFF, EDGE, SIDE = 0.92, 2.2, 6
REACH = float(sys.argv[6]) if len(sys.argv) > 6 else 1.1
KCAL = 4.184
STEPS, PERIOD = 400, 5

H0 = EDGE * np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.5, 0.5, 1.0]])
if shape == "cube":
    H0 = EDGE * np.eye(3)
rng = np.random.default_rng(7)
grid = np.stack(np.meshgrid(*[np.arange(SIDE)] * 3, indexing="ij"), -1)
fractional = (grid.reshape(-1, 3) + 0.5 + rng.uniform(-0.03, 0.03, (SIDE ** 3, 3))) / SIDE
start = fractional @ H0
count = len(start)

(work / "argon.top").write_text(f"""[ defaults ]
  1  2  no  1.0  1.0
[ atomtypes ]
  AR  18  {MASS}  0.0  A  {SIGMA}  {EPSILON}
[ moleculetype ]
  AR  0
[ atoms ]
  1  AR  1  AR  AR  1  0.0  {MASS}
[ system ]
argon
[ molecules ]
AR  {count}
""")
with open(work / "argon.gro", "w") as gro:
    gro.write(f"argon\n{count}\n")
    for i, r in enumerate(start):
        gro.write("%5d%-5s%5s%5d%14.9f%14.9f%14.9f\n" % (i + 1, "AR", "AR", i + 1, *r))
    if shape == "cube":
        gro.write("%.9f %.9f %.9f\n" % (H0[0, 0], H0[1, 1], H0[2, 2]))
    else:
        gro.write("%.9f %.9f %.9f 0 0 %.9f 0 %.9f %.9f\n"
                  % (H0[0, 0], H0[1, 1], H0[2, 2], H0[1, 0], H0[2, 0], H0[2, 1]))
extra = f"pruned_distance = {10 * pruned}\n" if pruned else ""
(work / "run.toml").write_text(f"""[input]
topology    = "argon.top"
coordinates = "argon.gro"
format      = "GROMACS"

[output]
energy              = "run.energy"
energy_interval     = {PERIOD}
trajectory          = "run.h5md"
trajectory_interval = {PERIOD}

[energy]
cutoff                 = {10 * CUTOFF}
switch_distance        = {10 * CUTOFF}
pairlist_distance      = {10 * REACH}
{extra}electrostatics         = "CUTOFF"
lennard_jones_modifier = "NONE"
dispersion_correction  = "NONE"

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.004
steps      = {STEPS}
seed       = 11

[ensemble]
ensemble    = "NPT"
temperature = 300.0
pressure    = 20000.0

[thermostat]
method        = "V-RESCALE"
time_constant = 0.5
interval      = 5

[barostat]
method        = "C-RESCALE"
time_constant = 1.0
interval      = 5
work          = "FIRST_ORDER"

[boundary]
type = "PERIODIC"

[execution]
target             = "{target}"
precision          = "DOUBLE"
neighbor_structure = "{structure}"
""")


def run_python():
    """The model of `run.toml` in a Python simulation; the pressure of the
    control file is in atm."""
    assert structure == "MATRIX" and not pruned
    loaded = mdir.load_gromacs(str(work / "argon.top"), str(work / "argon.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.switch_distance, system.pairlist_distance = CUTOFF, CUTOFF, REACH
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.periodic = True
    state = state.draw_velocities(system, 300.0, 11)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 0.004
    ensemble.kind, ensemble.seed = mdir.EnsembleKind.NPT, 11
    ensemble.temperature, ensemble.tau_t = 300.0, 0.5
    ensemble.pressure, ensemble.tau_p = 20000.0 * 1.01325, 1.0
    ensemble.coupling_period = PERIOD
    execution.target = getattr(mdir.Target, target)
    execution.precision = mdir.Precision.Double
    simulation = mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                              mdir.Schedule()))
    for reporter in (mdir.EnergyReporter(str(work / "run.energy"), PERIOD),
                     mdir.H5MDReporter(str(work / "run.h5md"), PERIOD)):
        simulation.reporters.append(reporter)
    simulation.run(STEPS)
    simulation.close_reporters()


TROTTER = cli == "python"
if TROTTER:
    run_python()
else:
    ran = subprocess.run([cli, "run", "run.toml"], cwd=work, capture_output=True, text=True)
    if ran.returncode:
        print("mdir run failed:", (ran.stderr or ran.stdout).strip().splitlines()[-1])
        sys.exit(0)

lines = (work / "run.energy").read_text().splitlines()
names = lines[0].split()[1:]
rows = {int(float(line.split()[0])): float(line.split()[names.index("potential")])
        for line in lines[2:]}

IMAGES = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)
first, second = np.triu_indices(count, 1)


def potential(x, H):
    """The sum over every image of every pair within the cutoff, kcal/mol."""
    d = x[first] - x[second]
    for k in (2, 1, 0):
        d -= np.rint(d[:, k] / H[k, k])[:, None] * H[k]
    r2 = ((d[:, None, :] - (IMAGES @ H)[None, :, :]) ** 2).sum(-1)
    s6 = (SIGMA * SIGMA / r2[r2 < CUTOFF * CUTOFF]) ** 3
    return float((4 * EPSILON * (s6 * s6 - s6)).sum()) / KCAL


def cell_of(frame):
    diagonal, tilt = np.asarray(frame.cell.diagonal), np.asarray(frame.cell.tilt)
    return np.array([[diagonal[0], 0.0, 0.0], [tilt[0], diagonal[1], 0.0],
                     [tilt[1], tilt[2], diagonal[2]]])


# The energies are those of the steps of coupling. There the row has the
# potential of the step before the barostat scales, and the frame the
# positions and the cell of after it, x diag(mu) and H diag(mu): the
# configuration of the row is the frame taken back to the cell of the
# frame before, which no scaling has changed since. That is the row of
# `work = "FIRST_ORDER"`. With `"TROTTER"`, the work of a Python
# simulation, the scaling is within the drift of the step and the row has
# the potential of the frame itself, in the cell of the frame (`mdir run`
# without the key gives the same rows).
frames = mdir.read_h5md(str(work / "run.h5md"))
compared = wrong = beyond = 0
worst, least = 0.0, np.inf
before = None
for frame in frames:
    step, after = int(frame.step), cell_of(frame)
    if before is not None and step in rows:
        H = after if TROTTER else before
        x = np.asarray(frame.positions) * (np.diag(H) / np.diag(after))
        half = 0.5 * min(H[0, 0], H[1, 1], H[2, 2])
        assert half >= CUTOFF, (step, half)
        difference = abs(rows[step] - potential(x, H))
        compared += 1
        beyond += REACH > half
        wrong += difference > 1e-5
        worst, least = max(worst, difference), min(least, half)
    before = after
print("%d frames, %d with the reach beyond half of the least of the diagonal"
      % (compared, beyond))
print("the least of them: %.4f nm; half of the least of the diagonal %s the reach"
      % (least, "well below" if least < 0.92 * REACH else "not well below"))
print("frames whose potential is not the sum over all images: %d" % wrong)
if wrong:
    print("largest difference: %.6f kcal/mol" % worst)
