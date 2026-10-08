"""A row of the outputs at a step where the barostat couples is of the state
of that step (#218): with the barostat of Trotter type the step scales the
positions, and the potentials of `[output] observables` and of
`[free_energy]` must take the cell after the scaling.

    barostat_rows.py dipeptide-directory target mdir work

On the dipeptide in water under NPT (C-rescale every 10 steps, from a start
far from its pressure, so that each scaling is large), for a pair term, a
term over bonds, and a wall, in double and mixed precision:

  mdir run   the rows at steps 10 and 20 of the observables file and of the
             free-energy file against the row at the start of a run from
             the positions and the cell of the checkpoint of that step
             (seven decimals of Å), which is an evaluation of that state;
             under NVE as well, where nothing scales
  Python     the values of the state at steps 10 and 20 against those of
             `run(0, energy=True)` right after, which moves nothing
"""
import pathlib
import subprocess
import sys

import numpy as np
import mdir

root, target_name, cli = sys.argv[1], sys.argv[2], sys.argv[3]
work = pathlib.Path(sys.argv[4])
KJ = 4.184

TERMS = """[[energy.pair]]
name = "soft"
expression = "a*exp(-r/l)"
observe = ["l"]
a = 0.5
l = 0.5
[[energy.pair]]
name = "alch"
expression = "lambda_s*a*exp(-r/l)"
a = 0.25
l = 0.6
[[energy.bond]]
name = "flat"
expression = "k*max(0, r - r0)^2"
particles = [[2, 19], [5, 15]]
observe = ["r0"]
k = 10.0
r0 = 4.0
[[energy.external]]
name = "wall"
expression = "0.5*k*max(0, z0 - z)^2"
selection = ":WAT"
scaling = "NONE"
observe = ["z0"]
k = 10.0
z0 = 6.0
[free_energy]
[free_energy.lambdas]
s = [1.0]
"""


def control(name, kind, precision, steps, coordinates=None):
    coupling = ('[thermostat]\nmethod = "V-RESCALE"\ninterval = 10\n'
                '[barostat]\nmethod = "C-RESCALE"\n' if kind == "NPT" else "")
    com = "center_of_mass_interval = 10" if kind == "NPT" else ""
    (work / f"{name}.toml").write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{coordinates or root + '/dipeptide.inpcrd'}"
[output]
energy_interval = {steps}
observables = "{name}.obs"
free_energy = "{name}.dhdl"
checkpoint = "{name}.h5"
checkpoint_interval = {steps}
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "PME"
{TERMS}[dynamics]
time_step = 0.0005
steps = {steps}
seed = 7
{com}
[ensemble]
ensemble = "{kind}"
temperature = 300.0
{coupling}[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    subprocess.run([cli, "run", f"{name}.toml"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = {}
    for suffix in ("obs", "dhdl"):
        lines = (work / f"{name}.{suffix}").read_text().splitlines()
        names = lines[0].lstrip("#").split()
        for line in lines[2:]:
            values = line.split()
            rows.setdefault(int(values[0]), {}).update(zip(names[2:], map(float, values[2:])))
    return rows


def write_inpcrd(checkpoint, path):
    """The positions and the cell of a checkpoint as Amber coordinates."""
    c = mdir.read_checkpoint(str(checkpoint))
    x = (c.positions * 10.0).ravel()
    lines = ["the state of a checkpoint", f"{len(x) // 3:6d}"]
    for i in range(0, len(x), 6):
        lines.append("".join(f"{v:12.7f}" for v in x[i:i + 6]))
    cell = c.cell.vectors
    lines.append("".join(f"{v:12.7f}" for v in (cell[0][0] * 10, cell[1][1] * 10,
                                                cell[2][2] * 10, 90.0, 90.0, 90.0)))
    path.write_text("\n".join(lines) + "\n")
    return float(np.prod(np.diag(cell)))


def run_cli(precision):
    # The file prints six decimals, and the coordinates of the evaluation
    # seven of Å: 5e-6 and 5e-7 relative. With the cell of before the
    # scaling the pair terms were off by 5e-6 to 1.2e-5.
    worst = {}
    for kind, steps in (("NPT", 10), ("NPT", 20), ("NVE", 10)):
        name = f"{kind}-{steps}-{precision}".lower()
        row = control(name, kind, precision, steps)[steps]
        volume = write_inpcrd(work / f"{name}.h5", work / f"{name}.inpcrd")
        evaluation = control(name + "-point", "NVE", precision, 1,
                             str(work / f"{name}.inpcrd"))[0]
        assert set(row) == set(evaluation) and len(row) == 7, sorted(row)
        for key, value in row.items():
            gap = abs(value - evaluation[key])
            assert gap <= 5e-6 + 5e-7 * abs(value), (name, key, value, evaluation[key])
            if kind == "NPT" and key in ("soft.energy", "soft.d_l", "dHdl.s"):
                worst[key] = max(worst.get(key, 0.0), gap / abs(value))
        if kind == "NPT" and steps == 10:
            first = volume
    assert len(worst) == 3, worst
    print(f"{target_name} {precision} mdir run: the rows at steps 10 and 20 under NPT (the "
          f"volume changes by {abs(volume / first - 1):.1e} between them) and at step 10 "
          f"under NVE are those of an evaluation of their state; pair terms within "
          f"{max(worst.values()):.1e}")


def run_python(precision):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants, soft.observe = [("a", 0.5 * KJ), ("l", 0.05)], ["l"]
    flat = mdir.TupleTerm()
    flat.name, flat.expression, flat.arity = "flat", "k*max(0, r - r0)^2", 2
    flat.particles = np.array([[1, 18], [4, 14]], dtype=np.int64)
    flat.parameters = [("k", np.array([4184.0, 4184.0])), ("r0", np.array([0.4, 0.4]))]
    flat.observe = ["r0"]
    wall = mdir.ExternalTerm()
    wall.name, wall.expression, wall.selection = "wall", "0.5*k*max(0, z0 - z)^2", ":WAT"
    wall.constants, wall.observe = [("k", 4184.0), ("z0", 0.6)], ["z0"]
    wall.scaling = mdir.ExternalScaling.None_
    system.pair_terms, system.tuple_terms, system.external_terms = [soft], [flat], [wall]
    state = state.draw_velocities(system, 300.0, 7)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, 7
    ensemble.kind = mdir.EnsembleKind.NPT
    ensemble.com_period = ensemble.coupling_period = 10
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    sim = mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                       mdir.Schedule()))
    tolerance = 1e-12 if precision == "Double" else 1e-6
    worst, cells = 0.0, []
    for steps in (5, 5, 10):
        sim.run(steps, energy=True)
        row = sim.state()
        sim.run(0, energy=True)
        evaluation = sim.state()
        cells.append(float(np.prod(np.diag(row.cell.vectors))))
        assert np.array_equal(row.positions, evaluation.positions)
        for key, value in row.observables.items():
            gap = abs(value / evaluation.observables[key] - 1)
            assert gap < tolerance, (row.step, key, value, evaluation.observables[key])
            worst = max(worst, gap)
    assert cells[0] != cells[1] != cells[2]
    print(f"{target_name} {precision} Python: the values of the state at steps 5, 10, and 20 "
          f"are those of run(0, energy=True) at that state within {worst:.1e}")


for precision in ("Double", "Mixed"):
    run_cli(precision)
    run_python(precision)
print("barostat rows passed")
