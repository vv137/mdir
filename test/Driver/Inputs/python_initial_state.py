"""InitialState.from_state: a new stage from the state that a simulation
reached (copies of its positions, velocities, and cell), against the same
state built field by field; the velocities left out on request or when the
state has none; and the refusal of the velocities of a leapfrog state,
which are half a step behind its positions.

Usage: python_initial_state.py ROOT TARGET"""
import sys

import numpy as np
import mdir

root, target = sys.argv[1], getattr(mdir.Target, sys.argv[2])
loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, initial = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7


def simulate(start, method="VelocityVerlet", steps=10):
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.method = getattr(mdir.IntegratorMethod, method)
    integrator.timestep = 0.001
    ensemble.kind = mdir.EnsembleKind.NVE
    execution = mdir.Execution()
    execution.target, execution.precision = target, mdir.Precision.Double
    execution.reorder, execution.deterministic = False, True
    simulation = mdir.Simulation(mdir.compile(system, start, integrator, ensemble,
                                              execution, mdir.Schedule()))
    simulation.run(steps, energy=True)
    return simulation.state()


moving = initial.draw_velocities(system, 300.0, 7)
reached = simulate(moving)

# The copy holds the state's positions, velocities, and cell.
start = mdir.InitialState.from_state(reached)
assert np.array_equal(start.positions, reached.positions)
assert np.array_equal(start.velocities, reached.velocities)
assert np.array_equal(start.cell.diagonal, reached.cell.diagonal)
assert np.array_equal(start.cell.tilt, reached.cell.tilt)

# It is the state that the fields give one by one, and a stage from it runs
# as one from that state, bit for bit.
manual = mdir.InitialState()
manual.positions, manual.velocities, manual.cell = (reached.positions, reached.velocities,
                                                    reached.cell)
a, b = simulate(start), simulate(manual)
assert np.array_equal(a.positions, b.positions) and np.array_equal(a.velocities, b.velocities)
assert a.energies == b.energies

# A new initial state is its own: setting it leaves the state unchanged.
start.positions = reached.positions + 0.01
assert np.array_equal(reached.positions, mdir.InitialState.from_state(reached).positions)

# velocities=False, as after a minimization, and the draw that follows.
still = mdir.InitialState.from_state(reached, velocities=False)
assert still.velocities.shape == (0, 3)
drawn = still.draw_velocities(system, 300.0, 11)
plain = mdir.InitialState()
plain.positions, plain.cell = reached.positions, reached.cell
assert np.array_equal(drawn.velocities, plain.draw_velocities(system, 300.0, 11).velocities)


# The velocities of a leapfrog state are half a step behind its positions.
leapfrog = simulate(moving, method="Leapfrog")
assert leapfrog.velocity_offset == -0.5
try:
    mdir.InitialState.from_state(leapfrog)
except mdir.InputError as error:
    assert "leapfrog" in str(error) and "velocities=False" in str(error), str(error)
else:
    raise AssertionError("expected InputError for the velocities of a leapfrog state")
assert mdir.InitialState.from_state(leapfrog, velocities=False).velocities.shape == (0, 3)
print("initial state from state passed")
