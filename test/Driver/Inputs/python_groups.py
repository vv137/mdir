"""The groups of 16 and the dual list in a Python simulation
(D[python-groups], #270, docs/python-model.md): the dipeptide in water with
PME, SHAKE, and SETTLE.

Usage: python_groups.py ROOT MDIR WORK SCENARIO [PRECISION]

  refusals     the refusals of `mdir.compile`, with the words of the
               control file; needs no device.
  energies     the energies and the forces at the start, and the energies
               of 100 steps, of the groups, of the groups whose lists begin
               with no room, and of the dual list against the neighbor
               matrix; `Program.plan` and the pipeline.
  cli          a Python simulation with the groups against `mdir run` with
               the groups, and `mdir run` against itself: the rows of the
               energy file and the state after 100 steps.
  reuse        the code that a Program keeps (D236) with the groups.
  borrow       a writable borrow: a commit of the cell and the positions
               against a simulation compiled there, and the refusal of a
               cell narrower than the reach of the groups (D242).
  tunables     an update of the tunables and `gradient()` with the groups
               against the matrix; the frame evaluator over frames far from
               each other and in other cells.
  checkpoints  Python -> `mdir run --continue`, `mdir run` -> Python, and
               Python with the groups -> Python with the matrix.

The groups are refused in the deterministic mode, and their atomic
additions have no fixed order, so nothing here is equal to the bit: each
comparison prints its largest difference to the standard error and holds
it to a tolerance stated where it is made.
"""
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import mdir

root, cli = sys.argv[1:3]
work = pathlib.Path(sys.argv[3])
scenario = sys.argv[4]
precision = sys.argv[5] if len(sys.argv) > 5 else "Double"
SEED = 2718
KCAL = 4.184
work.mkdir(parents=True, exist_ok=True)

MATRIX = dict(structure="Matrix")
GROUPS = dict(structure="Groups")
DUAL = dict(structure="Groups", pairlist=1.1, pruned=0.93)


def note(*words):
    print(*words, file=sys.stderr, flush=True)


def expect(error, call, *texts):
    try:
        call()
    except error as exc:
        for text in texts:
            assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__}")


def inputs(structure="Matrix", pairlist=1.0, pruned=0.0, kind="NVE", target="GPU",
           deterministic=False, capacity=None, tunables=None, start=None, period=10,
           run_precision=None):
    """The parts of the model: the system, the state, the integrator, the
    ensemble, and the execution."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, pairlist
    if pruned:
        system.pruned_distance = pruned
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.periodic = True
    system.pme_grid = [32, 32, 32]
    system.rigid_hydrogen_bonds = system.rigid_water = True
    if tunables:
        system.tunables = tunables
        system.tunable_gradient = True
    state = start if start is not None else state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep = 0.001
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.temperature, ensemble.seed = 300.0, SEED
    if kind != "NVE":
        ensemble.coupling_period = ensemble.com_period = period
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, run_precision or precision)
    if deterministic:
        execution.deterministic = True
    if capacity:
        execution.neighbor_capacity = capacity
    if structure != "Matrix":
        execution.neighbor_structure = getattr(mdir.NeighborStructure, structure)
    return system, state, integrator, ensemble, execution


def make(**options):
    schedule = mdir.Schedule()
    return mdir.compile(*inputs(**options), schedule)


def relative(a, b):
    """The largest difference of two dicts of energies, relative to the
    larger magnitude of each entry, and its name."""
    worst, name = 0.0, ""
    assert a.keys() == b.keys(), (sorted(a), sorted(b))
    for key in a:
        scale = max(abs(a[key]), abs(b[key]), 1.0)
        difference = abs(a[key] - b[key]) / scale
        if difference > worst:
            worst, name = difference, key
    return worst, name


def control(path, structure="GROUPS", pairlist=10.0, pruned=0.0, kind="NVE", steps=100,
            interval=20, start=None, checkpoint=None, period=10):
    """The control file of what `inputs` gives."""
    name = path.stem
    baths = (f'[thermostat]\nmethod = "V-RESCALE"\ninterval = {period}\n'
             if kind != "NVE" else "")
    baths += '[barostat]\nmethod = "C-RESCALE"\n' if kind == "NPT" else ""
    com = f"center_of_mass_interval = {period}" if kind != "NVE" else ""
    path.write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
{f'checkpoint = "{start}"' if start else ""}
[energy]
cutoff = 8.0
pairlist_distance = {pairlist}
{f"pruned_distance = {pruned}" if pruned else ""}
electrostatics = "PME"
[pme]
grid = [32, 32, 32]
[constraints]
hydrogen_bonds = true
rigid_water = true
[dynamics]
time_step = 0.001
steps = {steps}
seed = {SEED}
{com}
[output]
energy_interval = {interval}
energy = "{name}.dat"
checkpoint = "{name}.h5"
checkpoint_interval = {checkpoint or steps}
[ensemble]
ensemble = "{kind}"
temperature = 300.0
{baths}[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "{precision.upper()}"
neighbor_structure = "{structure}"
""")
    return path


def run_cli(path, *options):
    result = subprocess.run([cli, "run", *options, str(path)], cwd=path.parent,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    return result.stdout


def table(path):
    """The rows of an energy file: {step: values}, and the names."""
    lines = pathlib.Path(path).read_text().splitlines()
    names = lines[0].split()[1:]
    rows = {}
    for line in lines[1:]:
        if line.startswith("#") or not line.strip():
            continue
        fields = [float(x) for x in line.split()]
        rows[int(fields[0])] = np.array(fields[1:])
    return names, rows


def rows_differ(a, b):
    """The largest difference of two energy files over their rows, relative
    to the larger magnitude of each number (at least 1)."""
    names, first = table(a)
    names_b, second = table(b)
    assert names == names_b and first.keys() == second.keys(), (a, b)
    worst = 0.0
    for step in first:
        scale = np.maximum(np.maximum(np.abs(first[step]), np.abs(second[step])), 1.0)
        worst = max(worst, float((np.abs(first[step] - second[step]) / scale).max()))
    return worst


def refusals():
    def compiled(**options):
        return lambda: make(**options)
    words = "model: 'neighbor_structure = \"GROUPS\"' needs 'target = \"GPU\"' and not 'deterministic'"
    expect(mdir.InputError, compiled(structure="Groups", target="CPU"), words)
    expect(mdir.InputError, compiled(structure="Groups", deterministic=True), words)
    expect(mdir.InputError, compiled(pruned=0.9),
           "model: 'pruned_distance' keeps a dual list, which needs "
           "'neighbor_structure = \"GROUPS\"'")
    between = "model: 'pruned_distance' is not between 'cutoff' and 'pairlist_distance'"
    for pruned in (0.7, 0.8, 1.0, 1.2):
        expect(mdir.InputError, compiled(structure="Groups", pruned=pruned), between)
    for pruned in (-0.9, float("nan"), float("inf")):
        expect(mdir.InputError, compiled(structure="Groups", pruned=pruned),
               "System.pruned_distance must be positive, or 0 for one list")
    # The builder's: the reach of the groups against the least edge, 25.397 A.
    expect(mdir.InputError, compiled(structure="Groups", pairlist=2.6),
           "'pairlist_distance', 26 Å, exceeds 0.999 of the least edge of the cell, "
           "25.3716 Å, the most that the groups take")
    # The fields: their defaults, their types, and a unit quantity.
    system, execution = mdir.System(), mdir.Execution()
    assert execution.neighbor_structure == mdir.NeighborStructure.Matrix
    assert system.pruned_distance == 0.0
    for wrong in ("GROUPS", 1, None):
        try:
            execution.neighbor_structure = wrong
        except TypeError:
            continue
        raise AssertionError(f"neighbor_structure took {wrong!r}")
    expect(TypeError, lambda: setattr(system, "pruned_distance", "0.9"))
    # The matrix, as before, on the CPU: the plan says what was asked.
    plan = make(target="CPU").plan
    assert plan["neighbor_structure"] == "matrix" and plan["pruned_distance"] == 0.0, plan
    print("refusals passed")


def energies():
    reference = make(**MATRIX)
    assert reference.plan["neighbor_structure"] == "matrix"
    assert "md-exec-choose-neighbors" not in reference.pipeline
    base = mdir.Simulation(reference)
    base.run(0, energy=True)
    first = base.state()
    series = []
    for _ in range(5):
        base.run(20, energy=True)
        series.append(base.state().energies)
    # Double: the sums of the two structures differ in their order alone.
    # Mixed: the forces are f32, and the trajectories part as two runs of
    # one structure do.
    start_tolerance = 1e-10 if precision == "Double" else 2e-5
    run_tolerance = 1e-7 if precision == "Double" else 5e-3
    force_tolerance = 1e-7 if precision == "Double" else 0.5
    cases = [("groups", GROUPS), ("groups, capacity 1", dict(GROUPS, capacity=1)),
             ("dual", DUAL)]
    for label, options in cases:
        program = make(**options)
        plan = program.plan
        assert plan["neighbor_structure"] == "groups", plan
        assert abs(plan["pruned_distance"] - options.get("pruned", 0.0)) < 1e-15, plan
        assert "md-exec-choose-neighbors{kind=groups}" in program.pipeline
        simulation = mdir.Simulation(program)
        simulation.run(0, energy=True)
        at = simulation.state()
        difference, name = relative(at.energies, first.energies)
        force = float(np.abs(np.asarray(at.forces) - np.asarray(first.forces)).max())
        note(f"{precision} {label}: energies at the start differ by {difference:.2e} ({name}), "
             f"forces by {force:.2e} kJ/mol/nm of {np.abs(np.asarray(first.forces)).max():.0f}")
        assert difference < start_tolerance, (label, difference, name)
        assert force < force_tolerance, (label, force)
        worst = 0.0
        for row in series:
            simulation.run(20, energy=True)
            difference, name = relative(simulation.state().energies, row)
            worst = max(worst, difference)
        note(f"{precision} {label}: energies over 100 steps differ by up to {worst:.2e}")
        assert worst < run_tolerance, (label, worst)
        print(f"{precision} {label}: energies and forces of the matrix")
    print(f"energies {precision} passed")


def against_cli():
    here = work / precision
    here.mkdir(parents=True, exist_ok=True)
    for label, options, keys in (("groups", GROUPS, {}),
                                 ("dual", DUAL, dict(pairlist=11.0, pruned=9.3))):
        first = control(here / f"{label}-a.toml", **keys)
        second = control(here / f"{label}-b.toml", **keys)
        log = run_cli(first)
        run_cli(second)
        if label == "dual":
            assert "the inner lists were pruned" in log, log
        simulation = mdir.Simulation(make(**options))
        simulation.reporters.append(mdir.EnergyReporter(str(here / f"{label}-python.dat"), 20))
        simulation.run(100)
        simulation.close_reporters()
        names, mine = table(here / f"{label}-python.dat")
        _, theirs = table(here / f"{label}-a.dat")
        # Step 0: the same configuration; the rows have 6 decimals of
        # kcal/mol, of numbers of up to 2e4.
        start = float(np.abs(mine[0] - theirs[0]).max())
        between = rows_differ(here / f"{label}-python.dat", here / f"{label}-a.dat")
        itself = rows_differ(here / f"{label}-a.dat", here / f"{label}-b.dat")
        note(f"{precision} {label}: rows at step 0 differ by {start:.1e} in the units of the file; "
             f"rows over 100 steps by {between:.2e} relative; mdir run against itself {itself:.2e}")
        assert start < (5e-6 if precision == "Double" else 0.5), start
        assert between < (1e-7 if precision == "Double" else 5e-3), between
        assert itself < (1e-7 if precision == "Double" else 5e-3), itself

        def read(field):
            out = subprocess.run([cli, "checkpoint", f"--print={field}", f"{label}-a.h5"],
                                 cwd=here, check=True, stdout=subprocess.PIPE, text=True).stdout
            return np.array([[float(x) for x in line.split()[2:]] for line in out.splitlines()])
        state = simulation.state()
        moved = float(np.abs(np.asarray(state.positions) - read("positions")).max())
        note(f"{precision} {label}: positions after 100 steps differ by {moved:.2e} nm")
        assert moved < (1e-8 if precision == "Double" else 1e-3), moved
        print(f"{precision} {label}: a Python simulation follows mdir run as mdir run follows itself")
    print(f"cli {precision} passed")


def reuse():
    program = make(**DUAL)
    first = mdir.Simulation(program)
    second = mdir.Simulation(program)
    assert first.compile_stats["program_reused"] == 0, first.compile_stats
    assert second.compile_stats["program_reused"] == 1, second.compile_stats
    energies_of = []
    for simulation in (first, second):
        simulation.run(20, energy=True)
        energies_of.append(simulation.state().energies)
    difference, name = relative(*energies_of)
    note(f"{precision} reuse: the two simulations differ by {difference:.2e} ({name})")
    assert difference < (1e-7 if precision == "Double" else 5e-3), difference
    # Another structure is another pipeline and another module: nothing of
    # the dual list is taken by the one list or by the matrix.
    for options in (GROUPS, MATRIX):
        other = make(**options)
        assert other.pipeline != program.pipeline or other.ir != program.ir
        simulation = mdir.Simulation(other)
        assert simulation.compile_stats["program_reused"] == 0
    print(f"reuse {precision} passed")


def borrow():
    import torch

    def edit(buffer, change):
        tensor = torch.from_dlpack(buffer)
        values = change(tensor.cpu().numpy().copy())
        tensor.copy_(torch.as_tensor(values).to(tensor.device))
        del tensor
        return values

    # A reach of 2 nm, so that it binds before twice the cutoff, 1.6 nm.
    options = dict(structure="Groups", pairlist=2.0)
    program = make(**options)
    simulation = mdir.Simulation(program)
    simulation.run(8)
    at = simulation.state()
    with simulation.borrow() as lease:
        edges = edit(lease.cell, lambda c: c * 1.01)
        edit(lease.positions, lambda x: x * 1.01)
        assert lease.commit() == ("positions", "cell")
    committed = simulation.state()
    assert np.array_equal(committed.cell.diagonal, edges)
    start = mdir.InitialState.from_state(committed)
    fresh = mdir.Simulation(make(start=start, **options))
    fresh.run(0, energy=True)
    difference, name = relative(committed.energies, fresh.state().energies)
    force = float(np.abs(np.asarray(committed.forces) - np.asarray(fresh.state().forces)).max())
    note(f"{precision} borrow: energies at the commit differ by {difference:.2e} ({name}), "
         f"forces by {force:.2e}")
    assert difference < (1e-10 if precision == "Double" else 2e-5), difference
    assert force < (1e-7 if precision == "Double" else 0.5), force
    simulation.run(12, energy=True)
    fresh.run(12, energy=True)
    difference, name = relative(simulation.state().energies, fresh.state().energies)
    note(f"{precision} borrow: 12 steps later by {difference:.2e} ({name})")
    assert difference < (1e-7 if precision == "Double" else 5e-3), difference
    # A cell of 0.75 of the first: its least edge, 1.905 nm, is below the
    # reach over 0.999 and above twice the cutoff. Nothing is committed.
    before = simulation.state()
    with simulation.borrow() as lease:
        edit(lease.cell, lambda c: c * (0.75 / 1.01))
        edit(lease.positions, lambda x: x * (0.75 / 1.01))
        expect(mdir.InputError, lease.commit,
               "the pairlist distance of the program, 2.000000 nm, exceeds 0.999 of "
               "the least edge of the cell written",
               "the most that the groups take; nothing is committed")
        lease.abandon()
    after = simulation.state()
    assert np.array_equal(after.positions, before.positions)
    assert np.array_equal(after.cell.diagonal, before.cell.diagonal)
    simulation.run(4, energy=True)
    # The matrix takes that cell (D241): the check is of the groups.
    matrix = mdir.Simulation(make(pairlist=2.0))
    matrix.run(8)
    with matrix.borrow() as lease:
        edit(lease.cell, lambda c: c * 0.75)
        edit(lease.positions, lambda x: x * 0.75)
        assert lease.commit() == ("positions", "cell")
    print(f"borrow {precision} passed")


def tunables():
    system = inputs()[0]
    count = system.particle_count
    # The charges of the first 22 particles, the dipeptide, each an entry,
    # and the well depth of the pair of the type of the oxygen of water.
    solute = np.zeros(count, dtype=np.int64) - 1
    solute[:22] = np.arange(22)
    top = system.topology
    names, pairs = list(top.type_names), np.array(top.type_pairs)
    water = names.index("OW")
    oxygen = np.where((pairs[:, 0] == water) & (pairs[:, 1] == water), 0, -1)
    declared = [mdir.Tunable("q", "charge", map=solute),
                mdir.Tunable("epsilon", "epsilon_pair", map=oxygen)]
    programs = {label: make(tunables=declared, **options)
                for label, options in (("matrix", MATRIX), ("groups", GROUPS), ("dual", DUAL))}
    simulations = {label: mdir.Simulation(program) for label, program in programs.items()}
    reference = simulations["matrix"]
    value_tolerance = 1e-10 if precision == "Double" else 2e-5
    # The derivative is a sum of terms of both signs in f32 in mixed
    # precision: its tolerance there is absolute, against its largest entry.
    for round_ in range(2):
        results = {}
        for label, simulation in simulations.items():
            if round_:
                simulation.tunables.update({"q": simulation.tunables["q"] * 1.05,
                                            "epsilon": simulation.tunables["epsilon"] * 0.9})
            simulation.run(0, energy=True)
            results[label] = (simulation.state().energies, simulation.tunables.gradient())
        for label in ("groups", "dual"):
            difference, name = relative(results[label][0], results["matrix"][0])
            assert difference < value_tolerance, (label, difference, name)
            worst = 0.0
            for key in ("q", "epsilon"):
                mine = np.asarray(results[label][1][key])
                theirs = np.asarray(results["matrix"][1][key])
                scale = max(float(np.abs(theirs).max()), 1.0)
                worst = max(worst, float(np.abs(mine - theirs).max()) / scale)
            note(f"{precision} tunables {label} round {round_}: energies differ by "
                 f"{difference:.2e}, the gradient by {worst:.2e} of its largest entry")
            assert worst < (1e-9 if precision == "Double" else 1e-3), (label, worst)
    print(f"{precision}: an update of the tunables and gradient() with the groups")

    # Frames far from each other: the start, the state 200 steps later, the
    # start again, and the later state in a cell 2% wider with its positions
    # scaled. Each frame begins an activation, whose lists are built for it.
    sampler = mdir.Simulation(programs["matrix"])
    sampler.run(200)
    later = sampler.state()
    x0 = np.asarray(reference.state().positions)
    x1 = np.asarray(later.positions)
    cell = np.asarray(later.cell.diagonal)
    positions = np.stack([x0, x1, x0, x1 * 1.02, x0])
    cells = np.stack([cell, cell, cell, cell * 1.02, cell])
    outs = {label: mdir.FrameEvaluator(program).evaluate(positions, cells=cells)
            for label, program in programs.items()}
    cotangent = np.array([1.0, -2.0, 0.5, 3.0, 1.5])
    theirs = outs["matrix"]
    for label in ("groups", "dual"):
        mine = outs[label]
        energy = float(np.abs((mine.energy - theirs.energy) / theirs.energy).max())
        virial = float(np.abs((mine.virial - theirs.virial)
                              / np.maximum(np.abs(theirs.virial), 1.0)).max())
        back = float(np.abs(mine.energy[[0, 2, 4]] - mine.energy[0]).max() / abs(mine.energy[0]))
        worst = 0.0
        g, h = mine.vjp(cotangent), theirs.vjp(cotangent)
        for key in ("q", "epsilon"):
            scale = max(float(np.abs(np.asarray(h[key])).max()), 1.0)
            worst = max(worst, float(np.abs(np.asarray(g[key]) - np.asarray(h[key])).max()) / scale)
        note(f"{precision} frames {label}: energy differs by {energy:.2e}, virial by {virial:.2e}, "
             f"vjp by {worst:.2e}; the same frame again by {back:.2e}")
        assert energy < value_tolerance and virial < (1e-8 if precision == "Double" else 1e-3)
        assert worst < (1e-9 if precision == "Double" else 1e-3), worst
        assert back < value_tolerance, back
    print(f"{precision}: the frame evaluator with the groups over frames far apart and in another cell")
    print(f"tunables {precision} passed")


def continued(program, checkpoint, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = mdir.Simulation(program, checkpoint=str(checkpoint), **options)
    return simulation, [str(w.message) for w in caught]


def checkpoints():
    here = work / precision
    here.mkdir(parents=True, exist_ok=True)
    keys = dict(pairlist=11.0, pruned=9.3, kind="NPT")
    options = dict(DUAL, kind="NPT")
    state_tolerance = 1e-8 if precision == "Double" else 1e-3
    # The run that does not stop, of `mdir run` and of Python.
    run_cli(control(here / "whole.toml", steps=40, **keys))
    straight = mdir.Simulation(make(**options))
    straight.run(40, energy=True)
    end = straight.state()

    def close(label, state, rows=None):
        moved = float(np.abs(np.asarray(state.positions) - np.asarray(end.positions)).max())
        cell = float(np.abs(np.asarray(state.cell.diagonal) - np.asarray(end.cell.diagonal)).max())
        note(f"{precision} checkpoints {label}: positions at step 40 differ by {moved:.2e} nm, "
             f"the cell by {cell:.2e} nm")
        assert state.step == 40 and moved < state_tolerance and cell < state_tolerance, label

    # Python -> mdir run --continue.
    first = mdir.Simulation(make(**options))
    first.run(20)
    first.save_checkpoint(str(here / "python.h5"))
    path = control(here / "python.toml", steps=40, checkpoint=20, **keys)
    run_cli(path, "--continue")
    read = mdir.read_checkpoint(str(here / "python.h5"))
    assert read.step == 40, read.step
    close("Python -> mdir run", read)
    print("Python -> mdir run --continue with the dual list")

    # mdir run -> Python: the checkpoint of step 20 of a run of 20 steps.
    run_cli(control(here / "cli.toml", steps=20, **keys))
    simulation, notes = continued(make(**options), here / "cli.h5")
    assert not notes, notes
    assert simulation.step == 20
    simulation.run(20, energy=True)
    close("mdir run -> Python", simulation.state())
    print("mdir run -> Python with the dual list, without a note")

    # Python with the dual list -> Python with one list and with the
    # matrix: other execution, the same run, with a note that names the keys.
    first = mdir.Simulation(make(**options))
    first.run(20)
    first.save_checkpoint(str(here / "dual.h5"))
    for label, other, named in (("one list", dict(GROUPS, kind="NPT"),
                                 ("pruned_distance", "pairlist_distance")),
                                ("the matrix", dict(MATRIX, kind="NPT"),
                                 ("neighbor_structure", "pruned_distance"))):
        simulation, notes = continued(make(**other), here / "dual.h5")
        text = " ".join(notes)
        note(f"{precision} checkpoints dual -> {label}: {text}")
        for key in named:
            assert key in text, (key, text)
        simulation.run(20, energy=True)
        close(f"dual -> {label}", simulation.state())
    print("Python with the dual list -> Python with one list, and with the matrix: a note")
    print(f"checkpoints {precision} passed")


{"refusals": refusals, "energies": energies, "cli": against_cli, "reuse": reuse,
 "borrow": borrow, "tunables": tunables, "checkpoints": checkpoints}[scenario]()
