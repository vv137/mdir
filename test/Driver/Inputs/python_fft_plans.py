"""Several grids of PME in one process on a device (#323): the runtime keeps a
plan of cuFFT for each grid, direction, and precision of the process, and a
program in mixed precision with tunable charges takes four a grid (both
directions, f32 and f64). The table held 8, and the ninth plan ended the
process. Frame evaluators of the dipeptide in water at 64 and 128 points and
of the water in a rhombic dodecahedron at 32 points, the case of the issue,
take 12; each gives a finite energy and gradients, and the first gives the
same bits again after the others have made their plans.

usage: python_fft_plans.py DRIVER_INPUTS PRECISION"""
import sys

import numpy as np
import mdir

inputs, precision = sys.argv[1:3]


def model(kind, points):
    if kind == "water":
        loaded = mdir.load_gromacs(inputs + "/triclinic/water.top",
                                   inputs + "/triclinic/dodecahedron.gro",
                                   defines=["FLEXIBLE"])
    else:
        loaded = mdir.load_amber(inputs + "/dipeptide/dipeptide.prmtop",
                                 inputs + "/dipeptide/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_alpha, system.pme_grid = 3.5, [points] * 3
    system.tunables = [mdir.Tunable("q", "charge")]
    system.tunable_gradient = True
    execution = mdir.Execution()
    execution.target = mdir.Target.GPU
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), execution,
                           mdir.Schedule())
    return mdir.FrameEvaluator(program), state


def evaluate(evaluator, state, kind):
    x = np.array(state.positions)[None]
    tilts = np.array(state.cell.tilt)[None] if kind == "water" else None
    result = evaluator.evaluate(x, np.array(state.cell.diagonal)[None], tilts,
                                gradients=("positions", "strain"))
    energy = np.array(result.energy, dtype=np.float64)
    g, s = result.gradients["positions"][0], result.gradients["strain"][0]
    assert np.all(np.isfinite(energy)) and np.all(np.isfinite(g)) and np.all(np.isfinite(s))
    return energy, np.array(g), np.array(s)


cases = [("amber", 64), ("amber", 128), ("water", 32)]
made = []
for kind, points in cases:
    evaluator, state = model(kind, points)
    made.append((evaluator, state, kind, evaluate(evaluator, state, kind)))
    print(f"{precision}: {kind} at {points} points: finite energy and gradients")
evaluator, state, kind, first = made[0]
again = evaluate(evaluator, state, kind)
for a, b in zip(first, again):
    assert a.tobytes() == b.tobytes()
print(f"{precision}: the first evaluator again, after {len(cases)} grids: the same bits")
print("fft plans passed")
