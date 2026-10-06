"""Reporters of a Python simulation (D[python-reporters]): the energy file
and the trajectory of `mdir run` written inside the parts of a run, Python
callbacks at their steps, the schedule of OpenMM, backups, and the state
unchanged by the reports in the deterministic mode."""
import pathlib
import subprocess
import sys

import numpy as np
import mdir

root = sys.argv[1]
target_name = sys.argv[2]
target = getattr(mdir.Target, target_name)
cli = sys.argv[3]
work = pathlib.Path(sys.argv[4])
SEED = 271828


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def control(name, kind, precision, steps, energy, frames, fmt, constraints=False):
    thermostat = '[thermostat]\nmethod = "V-RESCALE"\ninterval = 10\n' if kind == "NVT" else ""
    path = work / f"{name}.toml"
    path.write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "CUTOFF"
[constraints]
hydrogen_bonds = {str(constraints).lower()}
rigid_water = {str(constraints).lower()}
[dynamics]
time_step = 0.0005
steps = {steps}
seed = {SEED}
[output]
energy_interval = {energy}
energy = "{name}.dat"
trajectory_interval = {frames}
trajectory = "{name}.{fmt}"
[ensemble]
ensemble = "{kind}"
temperature = 300.0
{thermostat}[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    return path


def simulation(kind, precision, constraints=False):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.rigid_hydrogen_bonds = system.rigid_water = constraints
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.coupling_period = 10
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                        mdir.Schedule()))


def rows(path):
    return [line for line in path.read_text().splitlines() if not line.startswith("#")]


def dcd_frames(path):
    """The coordinate records of a DCD file (Fortran records of 4 N bytes),
    as float32 arrays."""
    data, records, i = path.read_bytes(), [], 0
    while i < len(data):
        n = int.from_bytes(data[i:i + 4], "little")
        records.append(data[i + 4:i + 4 + n])
        i += n + 8
    size = max(len(r) for r in records)
    return np.array([np.frombuffer(r, dtype=np.float32) for r in records if len(r) == size])


# Files equal to those of `mdir run` with the same intervals: the energy
# rows and the frames, written inside one part (K report intervals). Where
# the two programs round differently in mixed precision (their loops have
# another shape; #105), rows and frames agree within three times the
# difference of `mdir run` in mixed and in double (E), as D196 bounds them.
cli_double = {}
for kind in ("NVE", "NVT"):
    for precision in ("Double", "Mixed"):
        for fmt in ("dcd", "xtc"):
            name = f"{kind}-{precision}-{fmt}".lower()
            path = control(name, kind, precision, 100, 10, 50, fmt)
            subprocess.run([cli, "run", str(path)], cwd=work, check=True, stdout=subprocess.DEVNULL)
            sim = simulation(kind, precision)
            sim.reporters.append(mdir.EnergyReporter(str(work / f"py-{name}.dat"), 10))
            sim.reporters.append(mdir.TrajectoryReporter(str(work / f"py-{name}.{fmt}"), 50))
            assert sim.run(100) == 100
            sim.close_reporters()
            mine, theirs = rows(work / f"py-{name}.dat"), rows(work / f"{name}.dat")
            ours, cli_file = work / f"py-{name}.{fmt}", work / f"{name}.{fmt}"
            if precision == "Double":
                cli_double[kind, fmt] = theirs, cli_file
            if mine == theirs and ours.read_bytes() == cli_file.read_bytes():
                print(f"{target_name} {kind} {precision} {fmt}: {len(mine)} rows and the "
                      f"trajectory equal to mdir run's, byte for byte")
                continue
            assert precision == "Mixed", name
            double_rows, double_file = cli_double[kind, fmt]
            assert len(mine) == len(theirs)
            worst = 0.0
            for a, b, d in zip(mine, theirs, double_rows):
                a, b, d = (np.array([float(x) for x in r.split()]) for r in (a, b, d))
                bound = 1e-6 + 3 * np.abs(b - d)
                assert np.all(np.abs(a - b) <= bound), (a, b, d)
                worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))))
            if fmt == "dcd":
                x, y, z = dcd_frames(ours), dcd_frames(cli_file), dcd_frames(double_file)
                assert x.shape == y.shape
                gap, bound = np.abs(x - y).max(), 1e-5 + 3 * np.abs(y - z).max()
                assert gap <= bound, (gap, bound)
                frames = f"{x.shape[0] // 3} frames within {gap:.1e} A (3 E: {bound:.1e} A)"
            else:
                frames = "XTC frames: " + str(ours.read_bytes().count(bytes([0, 0, 7, 203])))
            print(f"{target_name} {kind} {precision} {fmt}: {len(mine)} rows within 3 E "
                  f"(largest relative difference {worst:.1e}); {frames}")

# With SHAKE and SETTLE the steps of the rows measure K_half from the half
# steps and the temperature is the optimal estimate (D203); the steps of
# energy that are not rows keep no temperatures of the solvent. The rows
# equal those of `mdir run`, where every step of energy is a row, and where
# rows are every other step of energy (a frame period of 10 under an
# energy period of 20).
for energy, frames in ((10, 50), (20, 10)):
    name = f"optimal-{energy}-{frames}"
    path = control(name, "NVT", "Double", 100, energy, frames, "dcd", constraints=True)
    subprocess.run([cli, "run", str(path)], cwd=work, check=True, stdout=subprocess.DEVNULL)
    sim = simulation("NVT", "Double", constraints=True)
    sim.reporters.append(mdir.EnergyReporter(str(work / f"py-{name}.dat"), energy))
    sim.reporters.append(mdir.TrajectoryReporter(str(work / f"py-{name}.dcd"), frames))
    assert sim.run(100) == 100
    sim.close_reporters()
    mine, theirs = rows(work / f"py-{name}.dat"), rows(work / f"{name}.dat")
    assert mine == theirs, (mine[:2], theirs[:2])
    print(f"{target_name} constraints, energies every {energy}, frames every {frames}: "
          f"{len(mine)} rows with the optimal temperature equal to mdir run's")

# The schedule of OpenMM: periods 7 and 11 over 25 steps (NVE, so that any
# period is a report interval), callbacks with the state of their steps.
sim = simulation("NVE", "Double")
seen = []
sim.reporters.append(mdir.CallbackReporter(lambda s, state: seen.append((7, state.step)), 7))
sim.reporters.append(mdir.CallbackReporter(lambda s, state: seen.append((11, state.step)), 11))
sim.reporters.append(mdir.EnergyReporter(str(work / "sched.dat"), 7))
sim.reporters.append(mdir.TrajectoryReporter(str(work / "sched.dcd"), 11))
assert sim.run(25) == 25
assert seen == [(7, 7), (11, 11), (7, 14), (7, 21), (11, 22)], seen
steps = [int(r.split()[0]) for r in rows(work / "sched.dat")]
assert steps == [0, 7, 14, 21], steps  # the start, as mdir run writes it
# Reporters added later report after the step they were added at; the
# remainder of a run gets a report only if it is due.
assert sim.run(3) == 3 and seen[-1] == (7, 28)
sim.close_reporters()
assert [int(r.split()[0]) for r in rows(work / "sched.dat")] == [0, 7, 14, 21, 28]
print("schedule: callbacks at", seen)

# A callback that raises ends the run there; the steps taken count.
sim = simulation("NVE", "Double")


def fail(s, state):
    raise RuntimeError("callback failed")


sim.reporters.append(mdir.CallbackReporter(fail, 5))
expect(RuntimeError, lambda: sim.run(20), "callback failed")
assert sim.step == 5, sim.step
print("a callback error ends the run at step", sim.step)

# The deterministic mode: the reports change no bit of the state (cutoff,
# no constraints: D201).
plain, reported = simulation("NVT", "Double"), simulation("NVT", "Double")
reported.reporters.append(mdir.EnergyReporter(str(work / "det.dat"), 10))
reported.reporters.append(mdir.TrajectoryReporter(str(work / "det.dcd"), 20))
plain.run(60)
reported.run(60)
for field in ("positions", "velocities"):
    assert np.array_equal(getattr(plain.state(), field), getattr(reported.state(), field)), field
print("reports change no bit of the state in the deterministic mode")

# Backups of existing files, as `mdir run` makes them (D149), and refusals.
sim = simulation("NVE", "Double")
sim.reporters.append(mdir.EnergyReporter(str(work / "det.dat"), 10))
sim.run(10)
sim.close_reporters()
assert any(p.name != "det.dat" and "det.dat" in p.name for p in work.iterdir()), "no backup"
expect(mdir.InputError, lambda: mdir.EnergyReporter("x.dat", 0), "positive")
expect(mdir.InputError, lambda: mdir.TrajectoryReporter("x.txt", 10), "format")
sim.reporters = [mdir.EnergyReporter(str(work / "a.dat"), 10), mdir.EnergyReporter(str(work / "b.dat"), 10)]
expect(mdir.InputError, lambda: sim.run(10), "one EnergyReporter")
print("backups and refusals passed")
