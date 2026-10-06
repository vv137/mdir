"""Device-resident buffers of a persistent simulation (D[resident-buffers],
docs/python-segments.md): one activation of the entry runs every part, and
the state stays where the program keeps it between parts.

Usage: python_resident.py ROOT TARGET SCENARIO

Scenarios:
  failures  a part that fails after parts that succeeded leaves the state at
            the end of the last of them (the snapshot), bit for bit against a
            simulation that stops there; a first part that fails leaves the
            state that the simulation was created with
  updates   updates of tunables, evaluations, and reads of the state between
            parts: any partition of the steps after them gives the same state
            to the bit, on velocity Verlet and leapfrog, in double and mixed
            precision
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
                    kind="NVE", tunable=False, overlap=False):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    if tunable:
        system.tunables = [mdir.Tunable("q", "charge")]
    if overlap:
        # Two particles that are not bonded, on top of one another.
        positions = state.positions.copy()
        positions[1000] = positions[0]
        state.positions = positions
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = getattr(mdir.IntegratorMethod, method)
    integrator.timestep = timestep
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.temperature, ensemble.coupling_period = 300, 10
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
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
        # 4 fs without constraints: the hydrogens fly apart (after 53 steps
        # in double and 5 in mixed precision on the CPU). Parts of one step
        # run until one fails.
        program, _ = compile_program(precision, timestep=0.004)
        sim = mdir.Simulation(program)
        before = None
        for _ in range(1000):
            try:
                sim.run(1)
            except mdir.SimulationError as error:
                assert "not numbers" in str(error), str(error)
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
        expect(mdir.SimulationError, lambda: [unread.run(1) for _ in range(1000)],
               "not numbers")
        same(unread.state(), before, "the state after the failure, not read before")
        print(f"{precision}: a part that fails after step {kept.step} keeps the state "
              f"of that step, to the bit")

        # A first part that fails keeps the state of the start.
        program, start = compile_program(precision, overlap=True)
        blown = mdir.Simulation(program)
        expect(mdir.SimulationError, lambda: blown.run(5), "not numbers")
        kept = blown.state()
        assert blown.failed and kept.step == 0 and kept.forces is None
        assert np.array_equal(kept.positions, start.positions)
        assert not kept.velocities.any()
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

            whole = history([12])
            same(history([5, 7]), whole, "an update, then parts")
            same(history([1, 1, 10], read=True), whole, "reads of the state between parts")
            assert whole.tunables_version == 1
            print(f"{precision} {method}: after an update and an evaluation, parts of "
                  f"12, 5+7, and 1+1+10 steps give the same state to the bit")


SCENARIOS = {"failures": failures, "updates": updates}
if scenario not in SCENARIOS:
    raise SystemExit(f"unknown scenario '{scenario}'")
SCENARIOS[scenario]()
print(f"resident {scenario} passed")
