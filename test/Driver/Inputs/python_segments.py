"""Persistent simulations (D196): segmented runs against
uninterrupted ones, the phase of the coupling across segment boundaries,
the energies of a final step of energy against `mdir run`, errors and stops
that leave Python alive, and the ownership of simulations.

Usage: python_segments.py ROOT TARGET MDIR WORK SCENARIO. Each scenario is
one test file (SCENARIOS at the end), so that the files run side by side;
most of their time is the compilation of each simulation."""
import faulthandler
import gc
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import mdir

root = sys.argv[1]
target = getattr(mdir.Target, sys.argv[2])
cli = sys.argv[3]
work = pathlib.Path(sys.argv[4])
scenario = sys.argv[5]
# A run that hangs ends with the stacks of its threads.
faulthandler.dump_traceback_later(900, exit=True)
PARTS = (1, 7, 13)  # boundaries inside the periods of 10 steps
PME_ORDER = 4


def compile_program(kind="NVE", precision="Double", method="VelocityVerlet",
                    timestep=0.001, period=10, deterministic=True):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_order = PME_ORDER
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.method = getattr(mdir.IntegratorMethod, method)
    integrator.timestep = timestep
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.temperature, ensemble.coupling_period = 300, period
    execution = mdir.Execution()
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.reorder, execution.deterministic = False, deterministic
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def run(program, parts):
    simulation = mdir.Simulation(program)
    for n in parts:
        assert simulation.run(n) == n
    return simulation, simulation.state()


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


QUANTITIES = ("positions", "velocities", "forces")


def segmented_runs(cases):
    # Segmented against uninterrupted runs, in both precisions. A part
    # continues the activation of the entry that the last part left, with its
    # order of the particles and its neighbor structures (D[resident-buffers]),
    # so the deterministic mode gives the run in parts to the bit: the same
    # state, cell, and energies.
    if cases[0][0] != "NVE":
        # The uncoupled reference of the coupling check below; the NVE case
        # itself is checked in its own scenario.
        _, nve = run(compile_program("NVE", "Double"), [sum(PARTS)])
    for kind, method in cases:
        references = {}
        for precision in ("Double", "Mixed"):
            program = compile_program(kind, precision, method)
            _, whole = run(program, [sum(PARTS)])
            _, again = run(program, [sum(PARTS)])
            # The deterministic mode repeats a run bit for bit.
            for q in QUANTITIES:
                assert np.array_equal(getattr(whole, q), getattr(again, q)), q
            for parts in (PARTS, (1,) * 5 + (3, 3, 10)):
                segmented_simulation, segmented = run(program, parts)
                assert segmented.step == whole.step == sum(PARTS)
                assert abs(segmented.time - sum(PARTS) * 0.001) < 1e-15
                assert segmented_simulation.step == sum(PARTS)
                for q in QUANTITIES:
                    assert np.array_equal(getattr(segmented, q), getattr(whole, q)), \
                        (kind, method, precision, parts, q,
                         np.abs(getattr(segmented, q) - getattr(whole, q)).max())
                assert np.array_equal(segmented.cell.vectors, whole.cell.vectors)
                if method == "Leapfrog":
                    assert segmented.velocity_offset == -0.5
            print(f"{kind} {method} {precision}: parts of {'+'.join(map(str, PARTS))} "
                  f"and of 1x5+3x2+10 equal {sum(PARTS)} steps in one part to the bit")
            references[precision] = whole
        if kind != "NVE" and method == "VelocityVerlet":
            # The coupling changes the velocities, so a coupling at other steps
            # would be seen.
            coupled = np.abs(references["Double"].velocities - nve.velocities).max()
            assert coupled > 1e-3, coupled
            print(f"{kind}: the coupling moves the velocities by {coupled:.2e} nm/ps")
        if kind == "NVE":
            nve = references["Double"]


def errors():
    # The state before a run, and one that runs no steps.
    simulation = mdir.Simulation(compile_program())
    state = simulation.state()
    assert state.step == 0 and state.time == 0.0 and state.forces is None
    assert state.positions.shape == (1168, 3) and not state.positions.flags.writeable
    assert simulation.run(0) == 0 and simulation.step == 0
    expect(mdir.InputError, lambda: simulation.run(-1), "nonnegative")

    # The two steps that close a period of the barostat of Trotter type are not
    # split: an end between them is refused before any step is taken.
    npt = mdir.Simulation(compile_program("NPT"))
    expect(mdir.InputError, lambda: npt.run(9), "between the two steps")
    assert npt.step == 0
    assert npt.run(8) == 8
    expect(mdir.InputError, lambda: npt.run(1), "between the two steps")
    assert npt.run(2) == 2 and npt.step == 10

    # Programs that a simulation does not run, and a stale program.
    expect(mdir.UnsupportedError, lambda: mdir.Simulation(compile_program("NPT", period=1)),
           "every step")
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    integrator, execution = mdir.Integrator(), mdir.Execution()
    execution.target = target
    integrator.minimize = True
    # A minimization is a simulation of its own (D202): it
    # takes minimize(), not run(n).
    minimizer = mdir.Simulation(mdir.compile(system, state, integrator, mdir.Ensemble(),
                                             execution, mdir.Schedule()))
    expect(mdir.InputError, lambda: minimizer.run(1), "minimize")
    del minimizer
    integrator.minimize = False
    program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution, mdir.Schedule())
    integrator.timestep = 0.002
    expect(mdir.StaleProgramError, lambda: mdir.Simulation(program))

    # A run whose state stops being numbers fails; Python stays alive, and the
    # simulation keeps the state from before the part that failed. Two
    # particles that are not bonded, on top of one another, make it so.
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    positions = state.positions.copy()
    positions[1000] = positions[0]
    state.positions = positions
    execution = mdir.Execution()
    execution.target = target
    blown = mdir.Simulation(mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                                         execution, mdir.Schedule()))
    blown.part_seconds = 0.05
    expect(mdir.SimulationError, lambda: blown.run(500), "not numbers")
    assert blown.failed
    kept = blown.state()
    assert np.isfinite(kept.positions).all() and kept.step == blown.step == 0
    expect(mdir.SimulationError, lambda: blown.run(1), "failed earlier")
    print("a failed run leaves Python alive and keeps step", kept.step)


def stops():
    # A stop requested from another thread ends the run after the part under
    # way; the parts are short, so the stop comes soon.
    stopping = mdir.Simulation(compile_program("NVT", deterministic=False))
    stopping.run(1); stopping.run(1)  # both programs compiled
    stopping.part_seconds = 0.05
    requested = []
    def stop_later():
        time.sleep(0.5)
        requested.append(time.monotonic())
        stopping.request_stop()
    thread = threading.Thread(target=stop_later)
    thread.start()
    taken = stopping.run(10**8)
    latency = time.monotonic() - requested[0]
    thread.join()
    assert 0 < taken < 10**8 and stopping.step == taken + 2
    assert latency < 5.0, latency
    print(f"stop: {taken} steps taken, returned {latency:.3f} s after the request")

    # Another operation on a simulation that is running is refused.
    busy = mdir.Simulation(compile_program(deterministic=False))
    busy.run(1); busy.run(1)
    busy.part_seconds = 0.05
    thread = threading.Thread(target=lambda: busy.run(10**8))
    thread.start()
    time.sleep(0.5)
    expect(mdir.SimulationError, busy.state, "under way")
    expect(mdir.SimulationError, lambda: busy.run(1), "under way")
    busy.request_stop()
    thread.join()
    assert busy.state().step > 2

    # Ctrl-C ends a run after its part, with the steps taken recorded.
    # lit workers ignore SIGINT, and Python preserves that inherited disposition.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    interrupted = mdir.Simulation(compile_program(deterministic=False))
    interrupted.run(1); interrupted.run(1)
    interrupted.part_seconds = 0.05
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    try:
        interrupted.run(10**8)
        raise AssertionError("expected KeyboardInterrupt")
    except KeyboardInterrupt:
        pass
    assert interrupted.state().step == interrupted.step > 2
    interrupted.run(3)


def ownership():
    # Simulations run one after another are independent: interleaving two
    # changes neither, and creating and destroying them leaves nothing behind.
    program = compile_program("NVT")
    _, alone = run(program, [10, 11])
    first, second = mdir.Simulation(program), mdir.Simulation(compile_program("NPT"))
    first.run(10); second.run(10); first.run(11); second.run(11)
    for q in QUANTITIES:
        assert np.array_equal(getattr(first.state(), q), getattr(alone, q)), q
    for _ in range(5):
        transient = mdir.Simulation(program)
        transient.run(3); transient.run(2)
        del transient
        gc.collect()


def write_start():
    # A run that ends with a step of energy leaves the energies of the row that
    # `mdir run` writes at that step. The same coordinates with velocities feed
    # both, and the control file is the model's (python_compile.py).
    lines = (pathlib.Path(root) / "dipeptide.inpcrd").read_text().splitlines()
    count = int(lines[1].split()[0])
    coordinates = [float(l[i:i + 12]) for l in lines[2:-1] for i in range(0, len(l), 12)
                   if l[i:i + 12].strip()]
    speeds = np.random.default_rng(7).normal(0.0, 0.2, 3 * count)
    def block(values):
        return ["".join("%12.7f" % v for v in values[i:i + 6]) for i in range(0, len(values), 6)]
    (work / "start.rst7").write_text("\n".join([lines[0], "%6d" % count] + block(coordinates) +
                                          block(speeds) + [lines[-1]]) + "\n")
# The units of the columns of the file of energies per those of MDIR.
COLUMNS = {"total": 4.184, "potential": 4.184, "kinetic": 4.184, "temperature": 1.0,
           "virial": 4.184, "pressure": 1.01325, "conserved": 4.184, "volume": 1e-3}


def cli_rows(kind, precision):
    thermostat = '[thermostat]\nmethod = "V-RESCALE"\ninterval = 10' if kind != "NVE" else ""
    barostat = '[barostat]\nmethod = "C-RESCALE"\ninterval = 10' if kind == "NPT" else ""
    name = f"{kind}-{precision}".lower()
    (work / f"{name}.toml").write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "start.rst7"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "CUTOFF"
[constraints]
hydrogen_bonds = false
rigid_water = false
[dynamics]
time_step = 0.0005
steps = 20
[output]
energy_interval = 10
energy = "{name}.dat"
[ensemble]
ensemble = "{kind}"
temperature = 300.0
{thermostat}
{barostat}
[boundary]
type = "PERIODIC"
[execution]
target = "{sys.argv[2]}"
precision = "{precision.upper()}"
deterministic = true
""")
    subprocess.run([cli, "run", str(work / f"{name}.toml")], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    text = (work / f"{name}.dat").read_text().splitlines()
    names = text[0].lstrip("#").split()
    return {int(row.split()[0]): dict(zip(names, row.split())) for row in text[2:]}


def energy_program(kind, precision):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", str(work / "start.rst7"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature = 0.0005, 300
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    if kind != "NVE":
        ensemble.com_period = 10
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def compare(energies, row):
    """Every column at every printed digit."""
    for column, written in row.items():
        if column not in COLUMNS:
            continue
        mine = energies[column] / COLUMNS[column]
        assert "%.6f" % mine == written, (column, "%.6f" % mine, written)


def energies(precision):
    write_start()
    for kind in ("NVE", "NVT", "NPT"):
        rows = cli_rows(kind, precision)
        program = energy_program(kind, precision)
        # One part to step 20 is the run of `mdir run`: the rows agree to
        # every digit.
        whole = mdir.Simulation(program)
        whole.run(20, energy=True)
        compare(whole.state().energies, rows[20])
        # Parts to steps 10 and 20: the second continues the first, with its
        # order and its neighbor structures (D[resident-buffers]), and agrees
        # to every digit as well.
        parts = mdir.Simulation(program)
        parts.run(10, energy=True)
        compare(parts.state().energies, rows[10])
        parts.run(10, energy=True)
        compare(parts.state().energies, rows[20])
        # A run that ends between the periods of coupling ends with a plain
        # step of energy; one without energy=True leaves none at its step.
        parts.run(3, energy=True)
        assert parts.state().energies is not None and parts.state().step == 23
        parts.run(2)
        assert parts.state().energies is None
        print(f"energies {kind} {precision}: the rows of mdir run at steps 10 and 20 "
              f"to every printed digit, in one part and in two")


SCENARIOS = {
    "nve": lambda: segmented_runs([("NVE", "VelocityVerlet")]),
    "nvt": lambda: segmented_runs([("NVT", "VelocityVerlet"), ("NVT", "Leapfrog")]),
    "npt": lambda: segmented_runs([("NPT", "VelocityVerlet")]),
    "errors-stops": lambda: (errors(), stops()),
    "ownership": ownership,
    "energies-double": lambda: energies("Double"),
    "energies-mixed": lambda: energies("Mixed"),
}
if scenario not in SCENARIOS:
    raise SystemExit(f"unknown scenario '{scenario}'; expected one of {', '.join(SCENARIOS)}")
SCENARIOS[scenario]()
print(f"segments {scenario} passed")
