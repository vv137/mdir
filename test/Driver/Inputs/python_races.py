"""Accesses of another thread at the points where an operation runs
without the GIL (#621), forced with `Simulation._set_pause_point`: the
operation stops there until the test lets it go, and the other thread's
access runs in that window. No repetition: each interleaving happens once,
by construction.

Usage: python_races.py ROOT TARGET SCENARIO

  tunables  (CPU) an update of a tunable, paused after it has built and
            evaluated its new values and before it publishes them: the
            readers of another thread (`tunables[name]`, `values()`,
            `version`, `history`, `snapshot()`) see the values, version,
            and history of before, all of one snapshot; once it goes on,
            those of after
  export    (GPU) the export of a buffer to a consumer stream, paused at
            the hand-off without the GIL: another thread releases the view
            there, and a run is refused, since the export holds a lease of
            its own; once the capsule is deleted the run goes on
"""
import sys
import threading

import numpy as np
import mdir

root, target_name, scenario = sys.argv[1], sys.argv[2], sys.argv[3]
WAIT = 120  # seconds; a bound, not a timing: each wait ends when its event is set


def simulation(tunables):
    data = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = data.make_system(), data.make_state()
    system.periodic = True
    if tunables:
        system.tunables = [mdir.Tunable("q", "charge")]
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = mdir.Precision.Double
    sim = mdir.Simulation(mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                                       execution, mdir.Schedule()))
    sim.run(2)
    return sim


class Pause:
    """Stops the operation at `point` until `go` is set."""

    def __init__(self, point):
        self.point, self.reached, self.go = point, threading.Event(), threading.Event()

    def __call__(self, point):
        if point == self.point:
            self.reached.set()
            assert self.go.wait(WAIT), "the test did not let the operation go on"


def tunables():
    sim = simulation(tunables=True)
    before_q = sim.tunables["q"].copy()
    before_version, before_history = sim.tunables.version, sim.tunables.history
    new_q = before_q * 0.5
    pause = Pause("tunables")
    sim._set_pause_point(pause)
    errors = []

    def update():
        try:
            sim.tunables.update(q=new_q)
        except Exception as exc:  # reported by the main thread
            errors.append(exc)

    worker = threading.Thread(target=update)
    worker.start()
    assert pause.reached.wait(WAIT), "the update did not reach its point"
    # The update has replaced the values of the program and not published
    # them: every reader sees the snapshot of before, whole.
    values, version, history = sim.tunables.snapshot()
    seen = {
        "item": sim.tunables["q"],
        "values": sim.tunables.values()[0],
        "snapshot": values["q"],
    }
    for name, array in seen.items():
        assert np.array_equal(array, before_q), f"{name}: the values of the update before it ended"
    assert sim.tunables.version == version == before_version, (sim.tunables.version, version)
    assert sim.tunables.history == history == before_history, (sim.tunables.history, history)
    print(f"during the update: the values, version {version}, and history of before")
    pause.go.set()
    worker.join(WAIT)
    assert not worker.is_alive() and not errors, errors
    sim._set_pause_point(None)
    values, version, history = sim.tunables.snapshot()
    assert np.array_equal(values["q"], new_q) and np.array_equal(sim.tunables["q"], new_q)
    assert version == before_version + 1 and history[-1] == (2, version), (version, history)
    print(f"after the update: its values, version {version}, history {history}")


def export():
    sim = simulation(tunables=False)
    view = sim.view()
    pause = Pause("hand-off")
    sim._set_pause_point(pause)
    taken, errors = [], []

    def take():
        try:
            taken.append(view.positions.__dlpack__(stream=1))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=take)
    worker.start()
    assert pause.reached.wait(WAIT), "the export did not reach the hand-off"
    # The window: the exporting thread is without the GIL. The view's own
    # lease ends here; the export's must keep the buffers.
    view.release()
    try:
        sim.run(1)
    except mdir.SimulationError as exc:
        assert "lease of a view" in str(exc), str(exc)
        print("during the hand-off, with the view released: run refused:", str(exc).split(":")[0])
    else:
        raise AssertionError("a run passed while an export was handing the buffers off")
    pause.go.set()
    worker.join(WAIT)
    assert not worker.is_alive() and not errors and len(taken) == 1, errors
    sim._set_pause_point(None)
    taken.clear()  # the capsule, not taken by a consumer, releases its lease
    sim.run(1)
    print("after the capsule is deleted: the run goes on, step", sim.state().step)


{"tunables": tunables, "export": export}[scenario]()
print(f"races {scenario} passed")
