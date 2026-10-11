"""Runs whose positions or cell blow up (#168, D225): a
part that fails raises SimulationError, keeps the state from before it,
and leaves the process, and the device, able to run another simulation.

Usage: python_position_guard.py ROOT TARGET SCENARIO

Scenarios, each in double and mixed precision:
  restraint  a positional restraint of 1e300 kJ/mol/nm^2 on one atom sends
             it to about 1e190 nm, finite, within two steps
  overlap    two particles that are not bonded, on top of one another
  barostat   a barostat at 5e4 bar with a cutoff of 0.4 nm, whose pressure
             blows up and whose coupling scales the cell by about 1e10
  sanitize   one short case of each, in mixed precision, for
             compute-sanitizer
"""
import sys

import numpy as np
import mdir

root, target, scenario = sys.argv[1], getattr(mdir.Target, sys.argv[2]), sys.argv[3]


def compile_program(precision, kind="NVE", pressure=1.0, cutoff=0.8, restraint=None,
                    overlap=False, deterministic=True):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    # The defaults of before D[python-defaults], with which this was written.
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance = cutoff, cutoff + 0.1
    system.switch_distance = cutoff - 0.1
    system.electrostatics = mdir.Electrostatics.PME
    if restraint:
        system.restraints = [mdir.Restraint("@1", restraint)]
    if overlap:
        positions = state.positions.copy()
        positions[1000] = positions[0]
        state.positions = positions
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    ensemble.com_period = 0  # the default of before D[python-defaults]
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.temperature, ensemble.coupling_period = 300, 10
    ensemble.pressure, ensemble.tau_p = pressure, 0.1
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def same(a, b, label):
    for q in ("positions", "velocities", "forces"):
        x, y = getattr(a, q), getattr(b, q)
        assert (x is None) == (y is None), (label, q)
        if x is not None:
            assert np.array_equal(x, y), (label, q)
    assert a.step == b.step, (label, a.step, b.step)
    assert np.array_equal(a.cell.vectors, b.cell.vectors), label


def fails(program, precision, part, words, label, parts=200):
    simulation = mdir.Simulation(program)
    before = simulation.state()
    for _ in range(parts):
        try:
            simulation.run(part)
        except mdir.SimulationError as error:
            message = str(error)
            break
        before = simulation.state()
    else:
        raise AssertionError(f"{label}: the run did not fail")
    assert any(word in message for word in words), message
    assert simulation.failed
    kept = simulation.state()
    same(kept, before, label)
    assert np.isfinite(kept.positions).all() and np.isfinite(kept.velocities).all()
    try:
        simulation.run(1)
        raise AssertionError(f"{label}: a failed simulation ran")
    except mdir.SimulationError as error:
        assert "failed earlier" in str(error), str(error)
    # The process and the device run another simulation.
    healthy = mdir.Simulation(compile_program(precision))
    healthy.run(5)
    assert np.isfinite(healthy.state().positions).all()
    print(f"{label}: fails after step {kept.step} and keeps its state; another "
          f"simulation runs: {message.split(': ', 1)[1].split(';')[0]}")


def restraint(precision):
    fails(compile_program(precision, restraint=1e300), precision, 1,
          ("far outside the cell", "not numbers"), f"restraint {precision}")


def overlap(precision, deterministic=True):
    fails(compile_program(precision, overlap=True, deterministic=deterministic), precision, 1,
          ("far outside the cell", "not numbers"),
          f"overlap {precision}{'' if deterministic else ' groups'}")


def barostat(precision):
    fails(compile_program(precision, "NPT", pressure=5e4, cutoff=0.4), precision, 10,
          ("the barostat has scaled the cell", "the barostat has made the cell"),
          f"barostat {precision}")


if scenario == "restraint":
    for precision in ("Double", "Mixed"):
        restraint(precision)
elif scenario == "overlap":
    for precision in ("Double", "Mixed"):
        overlap(precision)
        if target == mdir.Target.GPU:
            overlap(precision, deterministic=False)
elif scenario == "barostat":
    for precision in ("Double", "Mixed"):
        barostat(precision)
elif scenario == "sanitize":
    restraint("Mixed")
    overlap("Mixed")
    barostat("Mixed")
else:
    raise SystemExit(f"unknown scenario {scenario}")
print(f"position guard {scenario} passed")
