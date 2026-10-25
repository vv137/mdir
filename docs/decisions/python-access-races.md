# D[python-access-races]

Recorded at its merge.

**The readers of the tunables read a snapshot that a successful operation publishes whole, and an export of a buffer takes its own lease before it releases the GIL** (#621).

Both defects were found by an external review that read the code; neither was reproduced as a crash.

**The tunables.**
- *Before.* `sim.tunables.update()` released the GIL and ran `Simulation::updateTunables`. That function replaces the values of the program (a move of the vectors), evaluates the forces at them, and then advances the version and the history. The Python readers took a `const&` to those vectors and copied them, outside the update's `busy` guard. A second Python thread could therefore read the new values with the old version, or read a vector while it was being replaced.
- *Now.* The simulation keeps a separate copy of the values, the version, and the history (`Simulation::TunableSnapshot`) under a lock of its own. Once an update, a commit of a borrow, or a continuation has succeeded, `publishTunables` copies the three there. Every reader takes that copy (`getTunableSnapshot`); the getters that returned references to the program's vectors are gone.
  - A reader during an update sees the snapshot of before, whole.
  - An update that fails, and is undone, publishes nothing.
  - `TunableValues.snapshot()` gives Python the three at once.
- *The alternative not taken.* Readers could have refused while an update runs. But the update's `busy` flag is also set by `run` and the other operations, so reading the tunables from a monitoring thread during a run would have become an error. That is a change of what works today, for no gain over a snapshot.
- *Cost.* A copy of the values (one double per entry of each tunable) at each successful update, and one at each read, as before.

**The export.**
- *Before.* `__dlpack__` checked the lease of the parent view or borrow, released the GIL, and called `Simulation::handOff`. Only afterwards did it take the export's own lease. If another thread called `View.release()` in that window, the simulation had no lease left: a run or an update passed its check of the leases and could end the activation, whose buffers the export was about to hand out.
- *Now.* The export takes its own lease first (`views::Lease::take`, RAII), while it holds the GIL and the parent's lease is alive. Then it releases the GIL for the hand-off. The lease moves into the capsule on success, and is released if the export fails.
- `Lease::take` is the one way to take a further lease. A lease adopted from `takeView` or `takeBorrow` keeps its constructor.

**Tests: forced interleavings, not repetition.**
- `Simulation.pausePoint` (Python: `Simulation._set_pause_point`, private, for tests) calls a function, on the operation's thread and without the GIL, at named points. At `"tunables"` an update has built and evaluated its new values but not yet published them. At `"hand-off"` the hand-off of an export begins. The test's function waits there on an event, and the other thread's access runs in that window, once, by construction.
- `test/Driver/python-races.test` (CPU): during the paused update, `tunables[name]`, `values()`, `snapshot()`, `version`, and `history` all give the snapshot of before; after it, those of after.
- `test/Driver/python-races-gpu.test`: during the paused hand-off, the view is released and a run is refused, because the export's lease is alive. Once the capsule is deleted, the run goes on.
- Both tests fail when the defect is put back in a temporary build: readers on the program's vectors, or the lease taken after the hand-off.
