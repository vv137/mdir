"""Device-resident buffers of a persistent simulation (D215,
docs/python-segments.md): one activation of the entry runs every part, and
the state stays where the program keeps it between parts.

Usage: python_resident.py ROOT TARGET SCENARIO

Scenarios:
  failures  a part that fails after parts that succeeded (a barostat that
            takes the cell below twice the cutoff) leaves the state and the
            cell at the end of the last of them (the snapshot), bit for bit
            against a simulation that stops there; a first part that fails leaves the
            state that the simulation was created with
  updates   updates of tunables, evaluations, and reads of the state between
            parts: any partition of the steps after them gives the same state
            to the bit, on velocity Verlet and leapfrog, in double and mixed
            precision
  threads   an activation resumed on threads other than the one it began
            on, with OpenMP threads on the CPU, gives the state of a run on
            one thread to the bit
  sanitize  a short life of three simulations for compute-sanitizer
"""
import sys

import numpy as np
import mdir

root, target, scenario = sys.argv[1], getattr(mdir.Target, sys.argv[2]), sys.argv[3]
QUANTITIES = ("positions", "velocities", "forces")


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__} ({text})")


def compile_program(precision="Double", method="VelocityVerlet", timestep=0.001,
                    kind="NVE", tunable=False, threads=1, pressure=1.0,
                    cutoff=0.8):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = cutoff, cutoff + (0.1 if cutoff < 1 else 0.005)
    system.switch_distance = cutoff - (0.1 if cutoff < 1 else 0.04)
    system.electrostatics = mdir.Electrostatics.PME
    if tunable:
        system.tunables = [mdir.Tunable("q", "charge")]
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = getattr(mdir.IntegratorMethod, method)
    integrator.timestep = timestep
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.temperature, ensemble.coupling_period = 300, 10
    ensemble.pressure, ensemble.tau_p = pressure, 0.1
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    execution.threads = threads
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule()), state


def same(a, b, label):
    for q in QUANTITIES:
        x, y = getattr(a, q), getattr(b, q)
        assert (x is None) == (y is None), (label, q)
        if x is not None:
            assert np.array_equal(x, y), (label, q, np.abs(x - y).max())
    assert a.step == b.step, (label, a.step, b.step)
    assert np.array_equal(a.cell.vectors, b.cell.vectors), label


def failures():
    for precision in ("Double", "Mixed"):
        # A cutoff of 1.24 nm in a cell of 2.54 nm along z, and a barostat
        # at 1000 bar, which shrinks the cell below twice the cutoff after a
        # few periods of coupling, with the state still finite. Parts of 10
        # steps (one period) run until one fails.
        program, _ = compile_program(precision, kind="NPT", pressure=1e3, cutoff=1.24)
        sim = mdir.Simulation(program)
        before = None
        for _ in range(1000):
            try:
                sim.run(10)
            except mdir.SimulationError as error:
                assert "barostat" in str(error) or "not numbers" in str(error), str(error)
                break
            before = sim.state()
        else:
            raise AssertionError("the run did not fail")
        assert sim.failed and before is not None and before.step > 0
        kept = sim.state()
        same(kept, before, "the state after the failure")
        # A simulation that runs the same steps in one part and stops there.
        again = mdir.Simulation(program)
        again.run(before.step)
        same(kept, again.state(), "a run that stops before the part that fails")
        expect(mdir.SimulationError, lambda: sim.run(1), "failed earlier")
        # The same without reading the state between parts: the state comes
        # from the copy made at the end of the last part that succeeded.
        unread = mdir.Simulation(program)
        expect(mdir.SimulationError, lambda: [unread.run(10) for _ in range(1000)])
        same(unread.state(), before, "the state after the failure, not read before")
        print(f"{precision}: a part that fails after step {kept.step} keeps the state "
              f"of that step, to the bit")

        # A first part that fails keeps the state of the start: with a
        # cutoff of 1.26 nm the first coupling takes the cell below twice it.
        # (States that blow up are the cases of python_position_guard.py.)
        program, start = compile_program(precision, kind="NPT", pressure=1e3, cutoff=1.26)
        blown = mdir.Simulation(program)
        expect(mdir.SimulationError, lambda: blown.run(10), "barostat")
        kept = blown.state()
        assert blown.failed and kept.step == 0 and kept.forces is None
        assert np.array_equal(kept.positions, start.positions)
        assert not kept.velocities.any()
        assert np.array_equal(kept.cell.vectors, start.cell.vectors)
        print(f"{precision}: a first part that fails keeps the state it began from")


def updates():
    for precision in ("Double", "Mixed"):
        for method in ("VelocityVerlet", "Leapfrog"):
            program, _ = compile_program(precision, method, kind="NVT", tunable=True)
            charges = None

            def history(parts, read=False):
                nonlocal charges
                sim = mdir.Simulation(program)
                sim.run(8)
                if charges is None:
                    charges = sim.tunables["q"] * 0.9
                sim.tunables.update({"q": charges})
                for n in parts:
                    sim.run(n)
                    if read:
                        sim.state()
                sim.run(0, energy=True)
                for n in parts:
                    sim.run(n)
                return sim.state()

            # Before the first run, an evaluation and then steps are the run
            # that takes its steps at once: the start ends at a boundary in
            # both.
            direct, evaluated = mdir.Simulation(program), mdir.Simulation(program)
            direct.run(20)
            evaluated.run(0, energy=True)
            evaluated.run(20)
            same(evaluated.state(), direct.state(), "an evaluation before the first run")
            whole = history([12])
            same(history([5, 7]), whole, "an update, then parts")
            same(history([1, 1, 10], read=True), whole, "reads of the state between parts")
            assert whole.tunables_version == 1
            print(f"{precision} {method}: after an update and an evaluation, parts of "
                  f"12, 5+7, and 1+1+10 steps give the same state to the bit; an "
                  f"evaluation before the first run changes nothing")


def threads():
    import threading
    for precision in ("Double", "Mixed"):
        for count in ((1, 4) if target == mdir.Target.CPU else (1,)):
            program, _ = compile_program(precision, kind="NVT", threads=count)
            alone = mdir.Simulation(program)
            for n in (3, 4, 5, 9):
                alone.run(n)
            moved = mdir.Simulation(program)
            moved.run(3)
            # The activation begun here takes its next parts on other
            # threads, then here again.
            for n in (4, 5):
                thread = threading.Thread(target=moved.run, args=(n,))
                thread.start()
                thread.join()
            moved.run(9)
            same(moved.state(), alone.state(), "parts on other threads")
            print(f"{precision}, Execution.threads {count}: parts on three Python threads "
                  f"give the state of one, to the bit")


def sanitize():
    # Short and in mixed precision, for compute-sanitizer: parts that
    # continue an activation, copies of the state, an update that ends one
    # activation and begins another, a part that fails, and simulations
    # that end with their activations.
    program, _ = compile_program("Mixed", kind="NVT", tunable=True)
    sim = mdir.Simulation(program)
    sim.run(3)
    sim.run(4, energy=True)
    sim.state()
    sim.tunables.update({"q": sim.tunables["q"] * 0.9})
    sim.run(5)
    other = mdir.Simulation(program)
    other.run(2)
    sim.run(1)
    del other
    program, _ = compile_program("Mixed", kind="NPT", pressure=1e3, cutoff=1.24)
    blown = mdir.Simulation(program)
    expect(mdir.SimulationError, lambda: [blown.run(10) for _ in range(10)], "barostat")
    print(f"steps {sim.step}, a part that failed, three simulations")


SCENARIOS = {"failures": failures, "updates": updates, "threads": threads,
             "sanitize": sanitize}
if scenario not in SCENARIOS:
    raise SystemExit(f"unknown scenario '{scenario}'")
SCENARIOS[scenario]()
print(f"resident {scenario} passed")
