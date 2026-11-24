"""Repeat a fixed PME frame, checking both bonded words and full observables."""
from pathlib import Path
import sys

import mdir
import numpy as np

inputs = Path(sys.argv[1])
topology = mdir.load_amber(str(inputs / "dipeptide.prmtop"),
                           str(inputs / "dipeptide.inpcrd"))
model, state = topology.make_system(), topology.make_state()
model.electrostatics = mdir.Electrostatics.PME
model.pme_grid = [32, 32, 32]
model.pme_alpha = 4.0
model.cutoff, model.pairlist_distance = 0.8, 1.0
model.tunables = [mdir.Tunable("q", "charge")]
model.tunable_gradient = True
positions = np.ascontiguousarray(state.positions)[None]
for deterministic in (False, True):
    run = mdir.Execution()
    run.target = mdir.Target.GPU
    run.precision = mdir.Precision.Mixed
    run.deterministic = deterministic
    evaluator = mdir.FrameEvaluator(mdir.compile(
        model, state, mdir.Integrator(), mdir.Ensemble(), run, mdir.Schedule()))
    reference = None
    for _ in range(3):
        frame = evaluator.evaluate(positions, gradients=("positions", "strain"))
        result = tuple(a.tobytes() for a in (frame.energy, frame.gradients["positions"],
                                             frame.gradients["strain"]))
        if reference is None:
            reference = result
        assert result == reference
    print("identical frames", deterministic)
