"""An output leaves the run alone (#233): in the deterministic mode, the run
with an output ends in the state of the run without it, bit for bit.

    output_independence.py dipeptide-directory target mdir work precision
        [ensemble ...]

The potential of an output shares the neighbor structure of the steps
(`md-exec-reuse-neighbors`). Before the fix of #218 the free-energy file,
`[output] observables`, and the pull file were evaluated in the cell of
before the scaling at a step where the barostat of Trotter type scales:
their refresh of the structure tested the scaled positions against a
reference of another cell and built the structure where the steps would not
have, in that cell, and the steps then summed their forces in another order.
At the state of the step the refresh of an output repeats that of the step
and is removed (D[output-independence]), so that it does not count for
`rebuild_interval` either.

On the dipeptide in water with PME, SHAKE, and SETTLE, from a start far from
its pressure, with a pair term, a term over bonds, and a wall that observe, a
pair term of `[free_energy]` with three states, and a term over the centers
of two waters across the cell, the same in every run:

  mdir run   each ensemble (NVE; NVT; NPT with each `work` of the barostat)
             without an output but the rows of the log and the final
             checkpoint, against the run with the observables file, the
             free-energy file, and the pull file, each alone (under NPT with
             the scaling of Trotter type) and together, with an energy file
             and a trajectory: the positions, velocities, forces, and cell
             of the final checkpoint, and the rows of the log; the same with
             `rebuild_interval`, whose count of refreshes the outputs do not
             advance; and the rows of the log at the end only against every
             10 steps (D204), without the term over centers (#240)
  Python     a simulation whose parts are read between them (`state()`,
             `view()`), and one with reporters of energies, frames, and
             observables and a callback, against one that runs the steps in
             parts of the same
             lengths and reads nothing, under NPT with the scaling of
             Trotter type (where that ensemble is among those asked for)
"""
import pathlib
import subprocess
import sys

import numpy as np
import mdir

root, target_name, cli = sys.argv[1], sys.argv[2], sys.argv[3]
work = pathlib.Path(sys.argv[4])
precision, ensembles = sys.argv[5], sys.argv[6:]
KJ = 4.184
STEPS, PERIOD = 40, 10

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
state = 1
[free_energy.lambdas]
s = [0.0, 0.5, 1.0]
"""


def far_waters():
    """Two waters whose separation along x is more than half the cell, so
    that the vector between them is that to an image."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    state, topology = loaded.make_state(), loaded.topology
    x, edge = state.positions[:, 0], state.cell.vectors[0][0]
    first = {}
    for particle, residue in enumerate(topology.residue_indices):
        if topology.residue_names[residue] == "WAT":
            first.setdefault(int(residue), particle)
    residues = sorted(first)
    low = min(residues, key=lambda r: x[first[r]])
    high = max(residues, key=lambda r: x[first[r]])
    assert x[first[high]] - x[first[low]] > 0.6 * edge
    return low + 1, high + 1


LOW, HIGH = far_waters()
CENTERS = f"""[[energy.bond]]
name = "pull"
expression = "k*(r - r0)^2"
groups = [":{LOW}", ":{HIGH}"]
k = 0.01
r0 = 3.0
"""

# The ensembles: the keys of [ensemble], [thermostat], and [barostat].
THERMOSTAT = '[thermostat]\nmethod = "V-RESCALE"\ninterval = 10\n'
ENSEMBLES = {"NVE": ("NVE", "")}
ENSEMBLES["NVT"] = ("NVT", THERMOSTAT)
for kind in ("TROTTER", "TROTTER_FIRST_ORDER", "EXACT", "FIRST_ORDER"):
    ENSEMBLES["NPT-" + kind] = (
        "NPT", THERMOSTAT + f'[barostat]\nmethod = "C-RESCALE"\ntime_constant = 0.5\n'
                            f'work = "{kind}"\n')

FILES = ('observables = "{name}.obs"\nfree_energy = "{name}.dhdl"\n'
         'pull = "{name}.pull"\nenergy = "{name}.energy"\n'
         'trajectory = "{name}.dcd"\ntrajectory_interval = 20\n')
INTERVAL = "rebuild_interval = 4\n"

# The runs: the keys of [output] beside the checkpoint, the interval of the
# rows of the log, whether the term over centers is there, and more keys of
# [energy].
OUTPUTS = {
    "none": ("", PERIOD, True, ""),
    "observables": ('observables = "{name}.obs"\n', PERIOD, True, ""),
    "free_energy": ('free_energy = "{name}.dhdl"\n', PERIOD, True, ""),
    "pull": ('pull = "{name}.pull"\n', PERIOD, True, ""),
    "files": (FILES, PERIOD, True, ""),
    "interval-none": ("", PERIOD, True, INTERVAL),
    "interval-files": (FILES, PERIOD, True, INTERVAL),
    "rows": ("", PERIOD, False, ""),
    "rows-at-end": ("", STEPS, False, ""),
}
# What each run is compared with, and the ensembles of the comparison (all
# of them where none are named).
COMPARED = {
    "observables": ("none", ("NPT-TROTTER",)),
    "free_energy": ("none", ("NPT-TROTTER",)),
    "pull": ("none", ("NPT-TROTTER",)),
    "files": ("none", ()),
    "interval-files": ("interval-none", ("NPT-TROTTER",)),
    "rows-at-end": ("rows", ("NVE", "NPT-TROTTER", "NPT-EXACT")),
}


def run(ensemble, output, precision):
    """Runs `mdir run` and returns the arrays of the final checkpoint and
    the rows of the log."""
    kind, coupling = ENSEMBLES[ensemble]
    keys, interval, centers, energy = OUTPUTS[output]
    # A term that observes needs the file of its columns: without the file
    # the terms are the same but for that key.
    terms = TERMS if "observables" in keys else "".join(
        line + "\n" for line in TERMS.splitlines() if not line.startswith("observe"))
    if centers:
        terms += CENTERS
    name = f"{ensemble}-{output}-{precision}".lower()
    (work / f"{name}.toml").write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[output]
energy_interval = {interval}
checkpoint = "{name}.h5"
checkpoint_interval = {STEPS}
{keys.format(name=name)}[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "PME"
{energy}{terms}[dynamics]
time_step = 0.002
steps = {STEPS}
seed = 7
{"center_of_mass_interval = 10" if kind != "NVE" else ""}
[ensemble]
ensemble = "{kind}"
temperature = 300.0
{coupling}[constraints]
hydrogen_bonds = true
rigid_water = true
[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    done = subprocess.run([cli, "run", f"{name}.toml"], cwd=work, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert done.returncode == 0, (name, done.stderr)
    c = mdir.read_checkpoint(str(work / f"{name}.h5"))
    arrays = {"positions": np.array(c.positions), "velocities": np.array(c.velocities),
              "forces": np.array(c.forces), "cell": np.array(c.cell.vectors)}
    rows = [line for line in done.stdout.splitlines()
            if line.startswith("INFO:")]
    builds = [line.split("built")[1].split()[0] for line in done.stdout.splitlines()
              if "were built" in line]
    return arrays, rows, int(builds[0]) if builds else -1


def differences(a, b):
    """The arrays of two states that differ: name, count, largest gap."""
    return [(key, int((a[key] != b[key]).sum()), float(np.abs(a[key] - b[key]).max()))
            for key in a if not np.array_equal(a[key], b[key])]


def run_cli(precision, ensembles):
    failures = []
    for ensemble in ensembles:
        done, compared = {}, []
        for output, (base, where) in COMPARED.items():
            if where and ensemble not in where:
                continue
            for name in (base, output):
                if name not in done:
                    done[name] = run(ensemble, name, precision)
            (reference, rows, builds), (state, other, count) = done[base], done[output]
            gaps = differences(reference, state)
            if OUTPUTS[output][1] == OUTPUTS[base][1] and other != rows:
                gaps.append(("rows of the log", sum(x != y for x, y in zip(rows, other)), 0.0))
            if gaps or (OUTPUTS[output][3] and builds != count):
                failures.append((ensemble, output, base, gaps, builds, count))
            compared.append(output)
        cell = done["none"][0]["cell"]
        print(f"{target_name} {precision} {ensemble}: {done['none'][2]} builds of the neighbor "
              f"structure in {STEPS} steps, volume {float(np.prod(np.diag(cell))):.4f} nm^3; "
              f"compared {', '.join(compared)}")
    for ensemble, output, base, gaps, builds, count in failures:
        print(f"FAILED {target_name} {precision} {ensemble}: the run '{output}' "
              f"({count} builds) is not the run '{base}' ({builds} builds): {gaps}")
    assert not failures
    print(f"{target_name} {precision} mdir run: the outputs leave the final state and the "
          f"rows of the log as they are without them, in {len(ensembles)} ensembles")


def make_simulation(precision):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants, soft.observe = [("a", 0.5 * KJ), ("l", 0.05)], ["l"]
    system.pair_terms = [soft]
    state = state.draw_velocities(system, 300.0, 7)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, 7
    ensemble.kind = mdir.EnsembleKind.NPT
    ensemble.com_period = ensemble.coupling_period = 10
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                        mdir.Schedule()))


def final(sim):
    state = sim.state()
    return {"positions": np.array(state.positions), "velocities": np.array(state.velocities),
            "forces": np.array(state.forces), "cell": np.array(state.cell.vectors)}


def run_python(precision):
    parts = (20, 10, 30)
    plain = make_simulation(precision)
    for steps in parts:
        plain.run(steps)
    reference = final(plain)

    read = make_simulation(precision)
    for steps in parts:
        read.run(steps)
        read.state()
        with read.view() as view:
            assert view.step == read.step
    gaps = differences(reference, final(read))
    assert not gaps, ("state() and view() between the parts", gaps)

    reported = make_simulation(precision)
    calls = []
    reported.reporters.append(mdir.EnergyReporter(str(work / f"{precision}.dat"), period=10))
    reported.reporters.append(mdir.TrajectoryReporter(str(work / f"{precision}.dcd"), period=20))
    reported.reporters.append(mdir.ObservablesReporter(str(work / f"{precision}.obs"), 10))
    reported.reporters.append(mdir.CallbackReporter(
        lambda simulation, state: calls.append(dict(state.observables)), period=10))
    for steps in parts:
        reported.run(steps)
    reported.close_reporters()
    assert len(calls) == sum(parts) // 10 and all(calls), calls
    gaps = differences(reference, final(reported))
    assert not gaps, ("reporters", gaps)
    print(f"{target_name} {precision} Python: state() and view() between the parts and the "
          f"reporters of energies, frames, observables, and a callback leave the final state as it is "
          f"without them")


ensembles = ensembles or list(ENSEMBLES)
run_cli(precision, ensembles)
if "NPT-TROTTER" in ensembles:
    run_python(precision)
print("output independence passed")
