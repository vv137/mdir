"""A force-loss derivative uses the continuous evaluator with either selector."""
import os
from pathlib import Path
import sys

import mdir
import numpy as np

inputs = Path(sys.argv[1])
topology = mdir.load_amber(str(inputs / "dipeptide.prmtop"),
                           str(inputs / "dipeptide.inpcrd"))
model, state = topology.make_system(), topology.make_state()
model.electrostatics = mdir.Electrostatics.Cutoff
model.cutoff, model.pairlist_distance = 0.8, 1.0
model.tunables = [mdir.Tunable("q", "charge")]
model.tunable_gradient = True
positions = np.ascontiguousarray(state.positions)[None]
run = mdir.Execution()
run.target = mdir.Target.GPU
run.precision = mdir.Precision.Mixed
run.deterministic = True
direction = np.zeros_like(positions)
direction[0, 1, 0] = 1.
reference = None
for mode in ("", "workunits-check"):
    os.environ["MDIR_TUPLE_SUMS"] = mode
    program = mdir.compile(model, state, mdir.Integrator(), mdir.Ensemble(),
                           run, mdir.Schedule())
    assert "bonded-units=" not in program.pipeline
    evaluator = mdir.FrameEvaluator(program)
    frame = evaluator.evaluate(positions, gradients=("positions", "tunables"))
    product = frame.vjp(positions=direction, gradients=("positions", "tunables"))
    values = (frame.energy, frame.gradients["positions"], product.positions, product["q"])
    for value in values:
        assert np.all(np.isfinite(value))
    if reference is None:
        reference = values
    else:
        for a, b in zip(reference, values):
            np.testing.assert_array_equal(a, b)
print("continuous force, force-loss gradient and Hessian product agree")
