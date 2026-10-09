"""Frames without loss in H5MD (D[h5md-reporter]): the frames that
`H5MDReporter` and `[output] trajectory = "x.h5md"` write, read back with
`read_h5md`, against the states of the run, bit for bit; continuation, a
killed run, backups, and refusals."""
import os
import pathlib
import signal
import subprocess
import sys

import numpy as np
import mdir

root = pathlib.Path(sys.argv[1])
target_name = sys.argv[2]
target = getattr(mdir.Target, target_name)
cli = sys.argv[3]
work = pathlib.Path(sys.argv[4])
mode = sys.argv[5] if len(sys.argv) > 5 else "all"
SEED = 271828


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__}")


def bits(a, b):
    """Whether two arrays have the same type, shape, and bits."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def program(name, kind, precision, tunables=None, leapfrog=False):
    if name == "triclinic":
        loaded = mdir.load_gromacs(str(root / "triclinic/water.top"),
                                   str(root / "triclinic/dodecahedron.gro"), defines=["FLEXIBLE"])
    else:
        loaded = mdir.load_amber(str(root / "dipeptide/dipeptide.prmtop"),
                                 str(root / "dipeptide/dipeptide.inpcrd"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    if name == "triclinic" or kind == "NPT":
        system.electrostatics = mdir.Electrostatics.PME
        system.pme_grid = [28] * 3 if name == "triclinic" else [32] * 3
    else:
        system.electrostatics = mdir.Electrostatics.Cutoff
    if tunables:
        system.tunables = [mdir.Tunable("q", "charge", map=np.arange(system.particle_count))]
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
    if leapfrog:
        integrator.method = mdir.IntegratorMethod.Leapfrog
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.coupling_period = 10
    if kind != "NVE":
        ensemble.com_period = 10
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def run(prog, path, steps=50, period=10, checkpoint=None, **options):
    """Runs a simulation with an H5MD reporter and a callback at the same
    steps; returns the states of the callback."""
    states = []
    sim = mdir.Simulation(prog)
    sim.reporters.append(mdir.H5MDReporter(str(path), period, **options))
    if checkpoint:
        sim.reporters.append(mdir.CheckpointReporter(str(checkpoint), 20))
    sim.reporters.append(mdir.CallbackReporter(lambda s, state: states.append(state), period))
    assert sim.run(steps) == steps
    sim.close_reporters()
    return states


def compare(path, states, single=False, velocities=False, forces=False, triclinic=False,
            periodic=True):
    """The frames of `path` against `states`: equal bits in f64, and the f32
    rounding of the state in f32."""
    kind = np.float32 if single else np.float64
    with mdir.read_h5md(str(path)) as frames:
        assert len(frames) == len(states), (len(frames), len(states))
        assert frames.dtype == np.dtype(kind)
        assert frames.has_velocities == velocities and frames.has_forces == forces
        assert frames.units["position"] == "nm" and frames.units["time"] == "ps"
        assert frames.creator.startswith("MDIR ")
        assert bits(frames.steps, np.array([s.step for s in states], dtype=np.int64))
        assert bits(frames.times, np.array([s.time for s in states]))
        for k, (frame, state) in enumerate(zip(frames, states)):
            assert frame.step == state.step and frame.time == state.time
            assert bits(frame.positions, state.positions.astype(kind)), (path, k)
            if velocities:
                assert bits(frame.velocities, state.velocities.astype(kind)), (path, k)
            else:
                assert frame.velocities is None
            if forces:
                assert bits(frame.forces, state.forces.astype(kind)), (path, k)
            else:
                assert frame.forces is None
            positions, cell = frame
            assert bits(positions, frame.positions)
            if not periodic:
                assert frame.cell is None and cell is None
                continue
            assert bits(frame.cell.diagonal, state.cell.diagonal), (path, k)
            assert bits(frame.cell.tilt, state.cell.tilt), (path, k)
            assert bits(frame.cell_vectors, state.cell.vectors), (path, k)
            if triclinic:
                assert bits(cell, state.cell.vectors)
            else:
                assert bits(cell, state.cell.diagonal)
            assert frame.potential_energy == state.energies["potential"], (path, k)
        assert bits(frames[-1].positions, states[-1].positions.astype(kind))
        expect(IndexError, lambda: frames[len(states)])
    expect(mdir.InputError, lambda: frames[0], "is closed")


def same_files(a, b):
    """Whether two trajectories hold the same frames, to the bit."""
    with mdir.read_h5md(str(a)) as x, mdir.read_h5md(str(b)) as y:
        assert len(x) == len(y), (len(x), len(y))
        assert bits(x.steps, y.steps) and bits(x.times, y.times)
        for f, g in zip(x, y):
            assert bits(f.positions, g.positions) and bits(f.cell_vectors, g.cell_vectors)
            for u, v in ((f.velocities, g.velocities), (f.forces, g.forces)):
                assert (u is None) == (v is None) and (u is None or bits(u, v))
            assert f.potential_energy == g.potential_energy
            assert f.tunables_version == g.tunables_version
        return len(x)


if mode == "kill":
    # A run that is killed between frames: the parent reads the file.
    prog = program("dipeptide", "NVE", "Double")
    sim = mdir.Simulation(prog)
    sim.reporters.append(mdir.H5MDReporter(str(work / "killed.h5md"), 10))

    def die(simulation, state):
        if state.step == 35:
            os.kill(os.getpid(), signal.SIGKILL)
    sim.reporters.append(mdir.CallbackReporter(die, 5))
    sim.run(50)
    raise SystemExit("the run was not killed")

if mode == "reference":
    # Files for readers that are not MDIR (h5md_independent.py), with the
    # states of the run beside them.
    data = {}

    def keep(name, prog, steps=50, **options):
        states = []
        sim = mdir.Simulation(prog)
        sim.reporters.append(mdir.H5MDReporter(str(work / f"{name}.h5md"), 10, **options))
        sim.reporters.append(mdir.CallbackReporter(lambda s, state: states.append(state), 10))
        sim.run(steps)
        sim.close_reporters()
        sim.save_checkpoint(str(work / f"{name}-state.h5"))
        data[f"{name}/masses"] = mdir.read_checkpoint(str(work / f"{name}-state.h5")).masses
        data[f"{name}/steps"] = np.array([s.step for s in states], dtype=np.int64)
        data[f"{name}/times"] = np.array([s.time for s in states])
        data[f"{name}/positions"] = np.array([s.positions for s in states])
        data[f"{name}/velocity"] = np.array([s.velocities for s in states])
        data[f"{name}/force"] = np.array([s.forces for s in states])
        data[f"{name}/cells"] = np.array([s.cell.vectors for s in states])
        data[f"{name}/potential"] = np.array([s.energies["potential"] for s in states])

    nve = program("dipeptide", "NVE", "Double")
    keep("ref-nve", nve)
    keep("ref-nve-f32", nve, positions="f32")
    keep("ref-nve-full", nve, velocities=True, forces=True)
    keep("ref-npt", program("dipeptide", "NPT", "Double"))
    keep("ref-tri", program("triclinic", "NVE", "Double"), steps=30, velocities=True)
    np.savez(work / "reference.npz", **data)
    killed = subprocess.run([sys.executable, __file__, str(root), target_name, cli, str(work), "kill"],
                            capture_output=True)
    assert killed.returncode == -signal.SIGKILL, (killed.returncode, killed.stderr.decode())
    print(f"{target_name}: 5 files and the states of their runs for independent readers")
    sys.exit(0)

plain = {}
for precision in ("Double", "Mixed"):
    tag = f"{target_name} {precision}"
    low = precision.lower()

    # Positions alone, written inside the parts of a run; f64, then f32.
    nve = program("dipeptide", "NVE", precision)
    states = run(nve, work / f"nve-{low}.h5md")
    plain[precision] = states
    compare(work / f"nve-{low}.h5md", states)
    assert [s.step for s in states] == [10, 20, 30, 40, 50]
    assert np.all(states[0].cell.tilt == 0)
    again = run(nve, work / f"nve-{low}-f32.h5md", positions="f32")
    assert all(bits(a.positions, b.positions) for a, b in zip(states, again))
    compare(work / f"nve-{low}-f32.h5md", again, single=True)
    print(f"{tag} NVE: 5 frames of positions and the cell equal to the states to the bit in "
          f"f64, and to their f32 rounding in f32")

    # With the velocities and the forces, written from the state at part
    # boundaries; the frames do not change the run in the deterministic mode.
    full = run(nve, work / f"nve-{low}-full.h5md", velocities=True, forces=True)
    compare(work / f"nve-{low}-full.h5md", full, velocities=True, forces=True)
    assert all(bits(a.positions, b.positions) and bits(a.velocities, b.velocities)
               for a, b in zip(states, full))
    half = run(nve, work / f"nve-{low}-full-f32.h5md", positions="f32", forces=True)
    compare(work / f"nve-{low}-full-f32.h5md", half, single=True, forces=True)
    print(f"{tag} NVE: velocities and forces equal to the states to the bit; the run is the "
          f"same with them")

    # A barostat: every frame has the cell of its step.
    npt = program("dipeptide", "NPT", precision)
    states = run(npt, work / f"npt-{low}.h5md")
    compare(work / f"npt-{low}.h5md", states)
    edges = np.array([s.cell.diagonal for s in states])
    assert len({e.tobytes() for e in edges}) > 1, "the barostat did not change the cell"
    states = run(npt, work / f"npt-{low}-full.h5md", velocities=True, forces=True)
    compare(work / f"npt-{low}-full.h5md", states, velocities=True, forces=True)
    print(f"{tag} NPT: {len({e.tobytes() for e in edges})} cells in 5 frames, each equal to "
          f"its state to the bit")

    # A triclinic cell: the cell vectors in rows. (A simulation takes no
    # barostat in a triclinic cell yet; `mdir run` below does.)
    for kind in ("NVE",):
        tri = program("triclinic", kind, precision)
        states = run(tri, work / f"tri-{kind}-{low}.h5md", steps=30)
        assert np.any(states[0].cell.tilt != 0)
        compare(work / f"tri-{kind}-{low}.h5md", states, triclinic=True)
        states = run(tri, work / f"tri-{kind}-{low}-full.h5md", steps=30, velocities=True)
        compare(work / f"tri-{kind}-{low}-full.h5md", states, velocities=True, triclinic=True)
        cells = len({s.cell.vectors.tobytes() for s in states})
        print(f"{tag} triclinic {kind}: 3 frames with {cells} "
              f"{'cell' if cells == 1 else 'cells'} equal to the states to the bit")

# Leapfrog: the velocities are half a step behind, and have their times.
leap = program("dipeptide", "NVE", "Double", leapfrog=True)
states = run(leap, work / "leap.h5md", velocities=True)
compare(work / "leap.h5md", states, velocities=True)
assert states[0].velocity_offset == -0.5
print(f"{target_name} leapfrog: velocities of the half step equal to the states to the bit")

# Tunables: every frame has the version of the values its forces were of.
tuned = program("dipeptide", "NVE", "Double", tunables=True)
sim = mdir.Simulation(tuned)
sim.reporters.append(mdir.H5MDReporter(str(work / "tuned.h5md"), 10))
sim.run(20)
sim.tunables["q"] = sim.tunables["q"] * 0.99
sim.run(20)
sim.close_reporters()
with mdir.read_h5md(str(work / "tuned.h5md")) as frames:
    assert frames.has_tunables_version
    versions = [f.tunables_version for f in frames]
assert versions == [0, 0, 1, 1], versions
with mdir.read_h5md(str(work / "nve-double.h5md")) as frames:
    assert not frames.has_tunables_version and frames[0].tunables_version is None
print(f"{target_name} tunables: versions {versions} at steps 10 to 40")

# A checkpoint between frames, a continuation, and the frames past the
# checkpoint removed (D130): the file of the run that did not stop, with
# the same checkpoints (a part begins anew after each, D223).
nve = program("dipeptide", "NVE", "Double")
whole = run(nve, work / "whole.h5md", steps=60, checkpoint=work / "whole.h5")
for options, name in (({"velocities": True, "forces": True}, "part-full.h5md"), ({}, "part.h5md")):
    sim = mdir.Simulation(nve)
    sim.reporters.append(mdir.H5MDReporter(str(work / name), 10, **options))
    sim.reporters.append(mdir.CheckpointReporter(str(work / "part.h5"), 20))
    sim.run(30)
    del sim
    assert len(mdir.read_h5md(str(work / name))) == 3
    assert mdir.read_checkpoint(str(work / "part.h5")).frames == 2
    sim = mdir.Simulation(nve, checkpoint=str(work / "part.h5"))
    sim.reporters.append(mdir.H5MDReporter(str(work / name), 10, **options))
    sim.reporters.append(mdir.CheckpointReporter(str(work / "part-next.h5"), 20))
    assert sim.run(40) == 40
    sim.close_reporters()
    if options:
        reference = run(nve, work / "whole-full.h5md", steps=60, checkpoint=work / "whole.h5",
                        **options)
        assert same_files(work / "whole-full.h5md", work / name) == 6
    else:
        assert same_files(work / "whole.h5md", work / name) == 6
# Another type, other elements, another period, or a file that is not a
# trajectory: refused by name.
for options, period, text in (({"positions": "f32"}, 10, "in f64, and the run writes f32"),
                              ({"velocities": True}, 10, "does not hold velocities"),
                              ({}, 20, "a frame every 10 steps")):
    sim = mdir.Simulation(nve, checkpoint=str(work / "part.h5"))
    sim.reporters.append(mdir.H5MDReporter(str(work / "part.h5md"), period, **options))
    expect(mdir.InputError, lambda: sim.run(10), text)
    del sim
# append=False: the frames that follow go to a part of their own.
sim = mdir.Simulation(nve, checkpoint=str(work / "part.h5"), append=False)
sim.reporters.append(mdir.H5MDReporter(str(work / "part.h5md"), 10))
sim.run(20)
sim.close_reporters()
assert list(mdir.read_h5md(str(work / "part.part0002.h5md")).steps) == [30, 40]
assert len(mdir.read_h5md(str(work / "part.h5md"))) == 6
# A file of the same name whose frames are not those of the checkpoint.
(work / "other").mkdir()
(work / "other" / "part.h5md").write_bytes((work / "part.part0002.h5md").read_bytes())
sim = mdir.Simulation(nve, checkpoint=str(work / "part.h5"))
sim.reporters.append(mdir.H5MDReporter(str(work / "other" / "part.h5md"), 10))
expect(mdir.InputError, lambda: sim.run(10), "holds a frame of step 40 among the 2")
del sim
print(f"{target_name} continuation: a checkpoint at step 20 of 60, the frame of step 30 removed, "
      f"6 frames equal to those of the run that did not stop")

# A run killed with SIGKILL between frames leaves a file that opens and
# holds the frames flushed.
killed = subprocess.run([sys.executable, __file__, str(root), target_name, cli, str(work), "kill"],
                        capture_output=True)
assert killed.returncode == -signal.SIGKILL, (killed.returncode, killed.stderr.decode())
with mdir.read_h5md(str(work / "killed.h5md")) as frames:
    assert list(frames.steps) == [10, 20, 30], frames.steps
    for frame, state in zip(frames, plain["Double"]):
        assert bits(frame.positions, state.positions)
        assert bits(frame.cell.diagonal, state.cell.diagonal)
print(f"{target_name} killed at step 35: the file opens and holds the 3 frames written, equal to "
      f"the states to the bit")

# `mdir run`: the trajectory of the control file. The last frame is the
# state of the checkpoint of the same step, and the DCD of the same run
# holds its f32 rounding in Å.
def control(name, trajectory, steps=100, extra="", precision="DOUBLE", triclinic=False,
            npt=False):
    path = work / f"{name}.toml"
    files = (f'topology = "{root}/triclinic/water.top"\ncoordinates = "{root}/triclinic/dodecahedron.gro"\n'
             'format = "GROMACS"' if triclinic else
             f'topology = "{root}/dipeptide/dipeptide.prmtop"\n'
             f'coordinates = "{root}/dipeptide/dipeptide.inpcrd"')
    coupling = ('pressure = 1.0\n[thermostat]\nmethod = "V-RESCALE"\ntime_constant = 0.5\n'
                'interval = 10\n[barostat]\nmethod = "C-RESCALE"\ntime_constant = 1.0\ninterval = 10')
    path.write_text(f"""[input]
{files}
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "{'PME' if triclinic else 'CUTOFF'}"
{'[pme]' if triclinic else ''}
{'grid = [28, 28, 28]' if triclinic else ''}
{'[constraints]' if triclinic else ''}
{'rigid_water = true' if triclinic else ''}
[dynamics]
time_step = 0.0005
steps = {steps}
seed = {SEED}
[output]
energy_interval = 10
trajectory_interval = 20
trajectory = "{trajectory}"
checkpoint = "{name}.h5"
checkpoint_interval = 20
{extra}
[ensemble]
ensemble = "{'NPT' if npt else 'NVE'}"
temperature = 300.0
{coupling if npt else ''}
[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision}"
deterministic = true
""")
    return path


def mdir_run(path, *flags, check=True):
    return subprocess.run([cli, "run", *flags, str(path)], cwd=work, check=check,
                          capture_output=True, text=True)


for precision in ("DOUBLE", "MIXED"):
    low = precision.lower()
    for triclinic, npt in ((False, False), (True, False), (True, True)):
        name = f"cli-{'tri-' if triclinic else ''}{'npt-' if npt else ''}{low}"
        mdir_run(control(name, f"{name}.h5md", precision=precision, triclinic=triclinic, npt=npt))
        last = mdir.read_checkpoint(str(work / f"{name}.h5"))
        with mdir.read_h5md(str(work / f"{name}.h5md")) as frames:
            assert list(frames.steps) == [20, 40, 60, 80, 100]
            assert frames.dtype == np.float64 and frames.has_potential_energy
            frame = frames[-1]
            assert frame.step == last.step and frame.time == last.time
            assert bits(frame.positions, last.positions)
            assert bits(frame.cell.diagonal, last.cell.diagonal)
            assert bits(frame.cell.tilt, last.cell.tilt)
            assert triclinic == bool(np.any(frame.cell.tilt != 0))
            exact = [f.positions for f in frames]
            energies = [f.potential_energy for f in frames]
            cells = len({f.cell_vectors.tobytes() for f in frames})
            assert (cells > 1) == npt, cells
        if triclinic:
            continue
        # The f32 file, and the DCD of the same run; the potential energy
        # of each frame is that of the row of the energy file, to the
        # digits of the file.
        mdir_run(control(f"{name}-f32", f"{name}-f32.h5md", precision=precision,
                         extra='trajectory_precision = "SINGLE"'))
        with mdir.read_h5md(str(work / f"{name}-f32.h5md")) as frames:
            assert frames.dtype == np.float32
            assert all(bits(f.positions, x.astype(np.float32)) for f, x in zip(frames, exact))
        mdir_run(control(f"{name}-dcd", f"{name}-dcd.dcd", precision=precision,
                         extra=f'energy = "{name}-dcd.dat"'))
        data, records, i = (work / f"{name}-dcd.dcd").read_bytes(), [], 0
        while i < len(data):
            n = int.from_bytes(data[i:i + 4], "little")
            records.append(data[i + 4:i + 4 + n])
            i += n + 8
        count = exact[0].shape[0]
        coordinates = [np.frombuffer(r, dtype=np.float32) for r in records if len(r) == 4 * count]
        assert len(coordinates) == 15
        for k, x in enumerate(exact):
            for axis in range(3):
                assert bits(coordinates[3 * k + axis], (x[:, axis] * 10.0).astype(np.float32))
        rows = [line.split() for line in (work / f"{name}-dcd.dat").read_text().splitlines()
                if not line.startswith("#")]
        header = (work / f"{name}-dcd.dat").read_text().splitlines()[0].lstrip("# ").split()
        column = header.index("potential")
        by_step = {int(r[0]): float(r[column]) for r in rows}
        for step, energy in zip((20, 40, 60, 80, 100), energies):
            assert abs(energy / 4.184 - by_step[step]) < 2e-6, (energy / 4.184, by_step[step])
    print(f"{target_name} mdir run {low}: the last frame equal to the checkpoint to the bit "
          f"(orthorhombic, triclinic, and triclinic with a barostat); f32 is the rounding of f64; the DCD is the f32 of the "
          f"frames in Å")

# `mdir run --continue`: stopped at every checkpoint, and once more from
# the checkpoint before the last, whose later frames are removed.
mdir_run(control("stop", "stop.h5md"))
stops = 0
while True:
    result = mdir_run(control("go", "go.h5md"), "--continue", "--max-walltime", "0.000001",
                      check=False)
    if result.returncode == 0:
        break
    assert result.returncode == 75, (result.returncode, result.stderr)
    stops += 1
    assert stops < 10
assert stops >= 2 and same_files(work / "stop.h5md", work / "go.h5md") == 5
os.replace(work / "go.h5.prev", work / "go.h5")
result = mdir_run(control("go", "go.h5md"), "--continue")
assert "removed 1 frames past the checkpoint" in result.stdout, result.stdout
assert same_files(work / "stop.h5md", work / "go.h5md") == 5
os.replace(work / "go.h5.prev", work / "go.h5")
result = mdir_run(control("go", "go.h5md", extra='trajectory_precision = "SINGLE"'), "--continue",
                  check=False)
assert result.returncode != 0 and "in f64, and the run writes f32" in result.stderr, result.stderr
print(f"{target_name} mdir run --continue: {stops} stops and a cut of 1 frame give the 5 frames "
      f"of the run that did not stop")

# The format by the extension or by name, and the refusals.
mdir_run(control("named", "frames.h5", extra='trajectory_format = "H5MD"', steps=20))
assert len(mdir.read_h5md(str(work / "frames.h5"))) == 1
result = mdir_run(control("bad", "bad-frames.h5", steps=20), check=False)
assert result.returncode != 0 and "'.h5md'" in result.stderr, result.stderr
result = mdir_run(control("bad", "bad.dcd", steps=20, extra='trajectory_precision = "SINGLE"'),
                  check=False)
assert result.returncode != 0 and "trajectory_precision" in result.stderr, result.stderr
# A run that is not continued keeps the file of an earlier one (D149).
mdir_run(control("named", "frames.h5", extra='trajectory_format = "H5MD"', steps=20))
assert (work / "#frames.h5.1#").exists()
run(nve, work / "nve-double.h5md", steps=10)
assert len(mdir.read_h5md(str(work / "#nve-double.h5md.1#"))) == 5
assert len(mdir.read_h5md(str(work / "nve-double.h5md"))) == 1

sim = mdir.Simulation(nve)
sim.reporters = [mdir.TrajectoryReporter(str(work / "a.dcd"), 10), mdir.H5MDReporter(str(work / "a.h5md"), 10)]
expect(mdir.InputError, lambda: sim.run(10), "one TrajectoryReporter or one H5MDReporter")
expect(mdir.InputError, lambda: mdir.TrajectoryReporter(str(work / "a.h5md"), 10), "H5MDReporter")
expect(mdir.InputError, lambda: mdir.H5MDReporter(str(work / "a.h5md"), 10, positions="f16"), '"f64" or "f32"')
expect(mdir.InputError, lambda: mdir.H5MDReporter(str(work / "a.h5md"), 0), "positive")
expect(mdir.InputError, lambda: mdir.H5MDReporter("", 10), "name of a file")
expect(mdir.InputError, lambda: mdir.read_h5md(str(work / "missing.h5md")), "cannot be read")
expect(mdir.InputError, lambda: mdir.read_h5md(__file__), "cannot be read as an HDF5 file")
expect(mdir.InputError, lambda: mdir.read_h5md(str(work / "nve-double.h5md"), group="solvent"),
       "has no group '/particles/solvent'")
# A checkpoint is an H5MD file with one frame of its own layout.
state = mdir.read_h5md(str(work / "part.h5"))
assert len(state) == 1 and state.has_velocities
assert bits(state[0].positions, mdir.read_checkpoint(str(work / "part.h5")).positions)
print("formats, backups, and refusals passed")
