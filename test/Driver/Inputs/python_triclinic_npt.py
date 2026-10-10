"""A Python simulation at constant pressure in a triclinic cell against
`mdir run` of the same system and settings (D[python-triclinic-npt],
docs/python-segments.md): 403 rigid waters in the rhombic dodecahedron of
`Inputs/triclinic`, PME on 28^3, velocity Verlet at 2 fs, stochastic
velocity rescaling and isotropic stochastic cell rescaling every 10 steps,
in the deterministic mode. The barostat scales the tilts with the cell
(D127), so every comparison of a cell here is one of its tilts as well.

Usage: python_triclinic_npt.py ROOT TARGET PRECISION MDIR WORK SCENARIO

  run          200 steps in one part, in parts of 10, and in parts of 30,
               70, and 100 against `mdir run` of 200 steps: the state, the
               cell, the energy file, and the DCD trajectory, whose record
               of the cell is of each frame.
  checkpoints  `mdir run` -> Python, Python -> `mdir run --continue`, and
               Python -> Python at step 100 against `mdir run` of 200
               steps.
  reporters    the cells of the frames in DCD, XTC, and H5MD against the
               cell of the state at each frame.
  frames       writes the H5MD and the DCD trajectory of 400 steps, 20
               frames, for the frame evaluator and the commits of
               `python_frames.py triclinic` and `python_dlpack_tilts.py
               frames`.
"""
import pathlib
import shutil
import struct
import subprocess
import sys
import warnings

import numpy as np
import mdir

root, target_name, precision, cli = sys.argv[1:5]
work = pathlib.Path(sys.argv[5])
scenario = sys.argv[6]
SEED = 2024
STEPS, PERIOD, TIMESTEP = 200, 10, 0.002
FIELDS = ("positions", "velocities", "forces")
label = f"{target_name} {precision}"


def control(path, checkpoint=STEPS):
    """The control file of what `model` gives, with a checkpoint every
    `checkpoint` steps. A checkpoint ends a segment of `mdir run`, as it
    ends the activation of a simulation (D215, D218): the neighbor
    structures are built anew after it, and the runs that are equal to the
    bit are those with the checkpoints at the same steps."""
    name = path.stem
    path.write_text(f"""[input]
topology = "{root}/water.top"
coordinates = "{root}/dodecahedron.gro"
format = "GROMACS"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "PME"
[pme]
grid = [28, 28, 28]
[constraints]
rigid_water = true
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = {TIMESTEP}
steps = {STEPS}
seed = {SEED}
center_of_mass_interval = {PERIOD}
[output]
energy_interval = {PERIOD}
energy = "{name}.dat"
trajectory = "{name}.dcd"
trajectory_interval = {PERIOD}
checkpoint = "{name}.h5"
checkpoint_interval = {checkpoint}
[ensemble]
ensemble = "NPT"
temperature = 300.0
[thermostat]
method = "V-RESCALE"
interval = {PERIOD}
[barostat]
method = "C-RESCALE"
[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    return path


def model():
    loaded = mdir.load_gromacs(root + "/water.top", root + "/dodecahedron.gro")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [28] * 3
    system.periodic = True
    system.rigid_water = True
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = TIMESTEP
    ensemble.seed = SEED
    ensemble.kind = mdir.EnsembleKind.NPT
    ensemble.temperature = 300.0
    ensemble.com_period = ensemble.coupling_period = PERIOD
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def run_cli(path, *options):
    result = subprocess.run([cli, "run", *options, str(path)], cwd=path.parent,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    return result.stdout


def continued(program, checkpoint):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = mdir.Simulation(program, checkpoint=str(checkpoint))
    assert not caught, [str(w.message) for w in caught]
    return simulation


def compare(what, state, reference):
    """To the bit: the state and the cell with its tilts."""
    for field in FIELDS:
        mine, theirs = getattr(state, field), getattr(reference, field)
        differing = np.count_nonzero(mine.view(np.uint64) != theirs.view(np.uint64))
        assert differing == 0, (what, field, differing, np.abs(mine - theirs).max())
    for field in ("diagonal", "tilt"):
        mine, theirs = getattr(state.cell, field), getattr(reference.cell, field)
        assert np.array_equal(mine, theirs), (what, field, mine, theirs)
    print(f"{what}: positions, velocities, forces, and the cell with its tilts equal")


def shape(cell):
    """How far the cell is from the dodecahedron: b_x, c_x/a_x - 1/2,
    c_y/b_y - 1/2."""
    d, t = cell.diagonal, cell.tilt
    return max(abs(t[0]), abs(t[1] / d[0] - 0.5), abs(t[2] / d[1] - 0.5))


def reference_run(checkpoint=STEPS):
    """`mdir run` of 200 steps: `run.h5` holds step 200 and, with a
    checkpoint every 100 steps, `run.h5.prev` step 100."""
    whole = work / "whole"
    whole.mkdir(parents=True, exist_ok=True)
    run_cli(control(whole / "run.toml", checkpoint))
    reference = mdir.read_checkpoint(str(whole / "run.h5"))
    assert reference.step == STEPS and reference.front_end == "mdir run"
    start = mdir.load_gromacs(root + "/water.top", root + "/dodecahedron.gro").make_state().cell
    moved = float(np.abs(reference.cell.tilt - start.tilt).max())
    assert moved > 1e-5, moved
    assert shape(reference.cell) < 1e-14, shape(reference.cell)
    return whole, reference, moved


def lower(record):
    """The lower-triangular cell of a DCD record, in nm."""
    a, cg, b, cb, ca, c = record
    bx, by = b * cg, b * np.sqrt(1.0 - cg * cg)
    cx = c * cb
    cy = c * (ca - cb * cg) / np.sqrt(1.0 - cg * cg)
    cz = np.sqrt(c * c - cx * cx - cy * cy)
    return np.array([a, by, cz, bx, cx, cy]) / 10.0


def dcd_cells(path):
    data, cells, i = pathlib.Path(path).read_bytes(), [], 0
    while i < len(data):
        n = int.from_bytes(data[i:i + 4], "little")
        if n == 48:
            cells.append(lower(np.frombuffer(data[i + 4:i + 52], dtype=np.float64)))
        i += n + 8
    return np.array(cells)


def xtc_cells(path):
    """The box of every frame of an XTC file (nm, f32): rows a, b, c."""
    data, cells, i = pathlib.Path(path).read_bytes(), [], 0
    while i < len(data):
        magic, count = struct.unpack_from(">ii", data, i)
        assert magic == 1995 and count > 9
        box = struct.unpack_from(">9f", data, i + 16)
        cells.append([box[0], box[4], box[8], box[3], box[6], box[7]])
        # Header 16, box 36, count 4, precision 4, ranges 24, smallest 4,
        # then the length of the packed positions and the bytes, padded.
        size = struct.unpack_from(">i", data, i + 88)[0]
        i += 92 + (size + 3) // 4 * 4
    return np.array(cells)


def with_reporters(simulation, directory, name="run"):
    simulation.reporters = [mdir.EnergyReporter(str(directory / f"{name}.dat"), PERIOD),
                            mdir.TrajectoryReporter(str(directory / f"{name}.dcd"), PERIOD)]


def same_files(directory, whole):
    for name in ("run.dat", "run.dcd"):
        assert (directory / name).read_bytes() == (whole / name).read_bytes(), name


def run():
    whole, reference, moved = reference_run()
    program = model()
    fingerprint = None
    for parts in ([STEPS], [PERIOD] * (STEPS // PERIOD), [30, 70, 100]):
        here = work / ("parts-" + "-".join(str(p) for p in parts[:3]))
        here.mkdir()
        simulation = mdir.Simulation(program)
        with_reporters(simulation, here)
        for part in parts:
            simulation.run(part)
        state = simulation.state()
        simulation.save_checkpoint(str(here / "run.h5"))
        simulation.close_reporters()
        fingerprint = mdir.read_checkpoint(str(here / "run.h5")).fingerprint
        assert fingerprint == reference.fingerprint, set(fingerprint) ^ set(reference.fingerprint)
        what = "one part" if len(parts) == 1 else "parts of " + (
            "10" if len(parts) > 3 else "30, 70, and 100")
        compare(f"{label}: {STEPS} steps in {what} against mdir run", state, reference)
        same_files(here, whole)
    cells = dcd_cells(whole / "run.dcd")
    assert len(cells) == STEPS // PERIOD and len(np.unique(cells[:, 4])) > STEPS // PERIOD // 2
    print(f"{label}: the energy file and the DCD trajectory equal those of mdir run byte for "
          f"byte; the cell of the {len(cells)} frames follows the barostat")
    print(f"{label}: the barostat moves the tilts by {moved:.1e} nm and keeps the shape of "
          f"the dodecahedron within {shape(reference.cell):.0e}")
    print(f"triclinic npt run {label} passed")


def checkpoints():
    whole, reference, _ = reference_run(STEPS // 2)
    program = model()
    middle = mdir.read_checkpoint(str(whole / "run.h5.prev"))
    assert middle.step == STEPS // 2
    assert np.abs(middle.cell.tilt - reference.cell.tilt).max() > 0.0

    # `mdir run` -> Python.
    part = work / "cli-to-python"
    part.mkdir()
    shutil.copy(whole / "run.h5.prev", part / "run.h5")
    for name in ("run.dat", "run.dcd"):
        shutil.copy(whole / name, part / name)
    simulation = continued(program, part / "run.h5")
    assert simulation.step == STEPS // 2
    assert np.array_equal(simulation.state().cell.tilt, middle.cell.tilt)
    with_reporters(simulation, part)
    simulation.run(STEPS // 2)
    compare(f"{label}: mdir run -> Python at step {STEPS // 2}", simulation.state(), reference)
    simulation.close_reporters()
    same_files(part, whole)

    # Python -> `mdir run --continue`, and Python -> Python.
    part = work / "python-to-cli"
    part.mkdir()
    path = control(part / "run.toml", STEPS // 2)
    first = mdir.Simulation(program)
    with_reporters(first, part)
    first.reporters.append(mdir.CheckpointReporter(str(part / "run.h5"), STEPS // 2))
    first.run(STEPS // 2)
    first.close_reporters()
    written = mdir.read_checkpoint(str(part / "run.h5"))
    assert written.step == STEPS // 2 and written.front_end == "python"
    assert np.array_equal(written.cell.tilt, middle.cell.tilt), (written.cell.tilt,
                                                                 middle.cell.tilt)
    assert np.array_equal(written.cell.diagonal, middle.cell.diagonal)
    assert written.fingerprint == reference.fingerprint
    python = continued(model(), part / "run.h5")
    python.run(STEPS // 2)
    compare(f"{label}: Python -> Python at step {STEPS // 2}", python.state(), reference)
    # The simulation that wrote the checkpoint goes on as well.
    first.run(STEPS // 2)
    compare(f"{label}: the simulation that wrote the checkpoint, {STEPS // 2} steps on",
            first.state(), reference)
    log = run_cli(path, "--continue")
    assert "other physics" not in log, log
    state = mdir.read_checkpoint(str(part / "run.h5"))
    assert state.step == STEPS
    compare(f"{label}: Python -> mdir run at step {STEPS // 2}", state, reference)
    same_files(part, whole)
    print(f"{label}: the energy files and the trajectories of the continued runs equal "
          f"those of mdir run byte for byte")
    print(f"triclinic npt checkpoints {label} passed")


def reporters():
    # A simulation takes one trajectory: three simulations of the program,
    # whose runs are the same to the bit, one for each format.
    program = model()
    cells = None
    for reporter in (mdir.H5MDReporter(str(work / "run.h5md"), PERIOD),
                     mdir.TrajectoryReporter(str(work / "run.dcd"), PERIOD),
                     mdir.TrajectoryReporter(str(work / "run.xtc"), PERIOD)):
        simulation = mdir.Simulation(program)
        simulation.reporters = [reporter]
        found, positions = [], []
        for _ in range(STEPS // PERIOD):
            simulation.run(PERIOD)
            state = simulation.state()
            found.append(np.concatenate([state.cell.diagonal, state.cell.tilt]))
            positions.append(state.positions)
        simulation.close_reporters()
        assert cells is None or np.array_equal(cells, found)
        cells = np.array(found)
    count = len(cells)
    assert len(np.unique(cells[:, 4])) > count // 2, "the barostat did not change the tilts"
    dcd, xtc = dcd_cells(work / "run.dcd"), xtc_cells(work / "run.xtc")
    assert len(dcd) == count and len(xtc) == count, (len(dcd), len(xtc))
    # DCD holds lengths and cosines in f64, XTC the vectors in f32.
    in_dcd, in_xtc = float(np.abs(dcd - cells).max()), float(np.abs(xtc - cells).max())
    assert in_dcd < 1e-12 and in_xtc < 5e-7, (in_dcd, in_xtc)
    frames = mdir.read_h5md(str(work / "run.h5md"))
    assert len(frames) == count and frames.dtype == np.float64
    for n, frame in enumerate(frames):
        assert int(frame.step) == (n + 1) * PERIOD
        assert np.array_equal(frame.cell.diagonal, cells[n, :3]), n
        assert np.array_equal(frame.cell.tilt, cells[n, 3:]), n
        assert np.array_equal(np.asarray(frame.positions), positions[n]), n
        edges = tuple(frame)[1]
        assert edges.shape == (3, 3) and edges[2, 0] == cells[n, 4]
    print(f"{label}: the cell of each of {count} frames against the state of its step: DCD "
          f"within {in_dcd:.0e} nm, XTC within {in_xtc:.0e} nm, H5MD (box/edges, (3, 3) per "
          f"frame) and its positions equal to the bit")
    print(f"triclinic npt reporters {label} passed")


def frames():
    program = model()
    for reporter in (mdir.H5MDReporter(str(work / "npt.h5md"), 20),
                     mdir.TrajectoryReporter(str(work / "npt.dcd"), 20)):
        simulation = mdir.Simulation(program)
        simulation.reporters = [reporter]
        simulation.run(400)
        simulation.close_reporters()
    print(f"{label}: 20 frames of a Python run at constant pressure written")


{"run": run, "checkpoints": checkpoints, "reporters": reporters, "frames": frames}[scenario]()
