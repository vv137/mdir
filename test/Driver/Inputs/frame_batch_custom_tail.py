"""General pair-tail device quadrature against the independent host path."""
from pathlib import Path
import sys

import mdir
import numpy as np
import torch

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
(work / "system.top").write_text(
    "[ defaults ]\n1 2 no 1.0 1.0\n[ atomtypes ]\nA 0 40.0 0.0 A 0.0 0.0\n"
    "[ moleculetype ]\nA 1\n[ atoms ]\n1 A 1 A A 1 0.0 40.0\n"
    "[ system ]\ncustom tail\n[ molecules ]\nA 2\n")
x = np.array([[0., 0., 0.], [2., 0., 0.]])
(work / "system.gro").write_text("custom tail\n2\n" + "".join(
    f"{i:5d}{'A':<5}{'A':>5}{i:5d}{a:8.3f}{b:8.3f}{c:8.3f}\n"
    for i, (a, b, c) in enumerate(x, 1)) + "4.0 4.0 4.0\n")
loaded = mdir.load_gromacs(str(work / "system.top"), str(work / "system.gro"))
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance, system.switch_distance = 1., 1.15, 1.
system.electrostatics = mdir.Electrostatics.Cutoff
system.truncation = mdir.Truncation.None_
system.dispersion = mdir.DispersionCorrection.EnergyPressure
term = mdir.PairTerm()
exponential = len(sys.argv) > 2 and sys.argv[2] == "exponential"
term.name = "tail"
term.expression = "a*exp(-r/l)" if exponential else "-c/r^6"
term.constants = [("a", 2.), ("l", .125)] if exponential else [("c", 0.0078125)]
names = [name for name, _ in term.constants]
system.pair_terms = [term]
system.tunables = [mdir.Tunable(name, name, term="tail") for name in names]
system.tunable_gradient = True
execution = mdir.Execution()
execution.target, execution.precision = mdir.Target.GPU, mdir.Precision.Double
execution.deterministic, execution.reorder, execution.fast_math = True, False, False
ensemble = mdir.Ensemble()
ensemble.com_period = 0
program = mdir.compile(system, state, mdir.Integrator(), ensemble, execution, mdir.Schedule())
evaluator = mdir.FrameEvaluator(program)
frames = np.stack((x, x))
cells = np.array([[4., 4., 4.], [5., 5., 5.]])
positions = torch.tensor(frames, device="cuda", dtype=torch.float64)
device_cells = torch.tensor(np.column_stack((cells, np.zeros_like(cells))), device="cuda")
theta = torch.empty(len(names), device="cuda", dtype=torch.float64)
energy = torch.empty(2, device="cuda", dtype=torch.float64)
gradient = torch.empty_like(positions)
strain = torch.empty((2, 3, 3), device="cuda", dtype=torch.float64)
rows = torch.empty((2, len(names)), device="cuda", dtype=torch.float64)
points = [(2., .125), (2.25, .15625)] if exponential else [(0.0078125,), (0.01171875,)]
for coefficients in points:
    theta.copy_(torch.tensor(coefficients, device="cuda", dtype=torch.float64))
    evaluator.tunables.update({name: np.array([value]) for name, value in zip(names, coefficients)})
    reference = evaluator.evaluate(frames, cells=cells, gradients=("positions", "strain", "tunables"))
    evaluator._evaluate_device_batch(positions, theta, energy, cells=device_cells,
        position_gradient=gradient, strain=strain, rows=rows)
    assert evaluator._simulation._device_batch_copy_bytes[:2] == [0, 0]
    failed = False
    for name, tensor, expected, exact in (
            ("energy", energy, reference.energy, False),
            ("positions", gradient, reference.gradients["positions"], True),
            ("strain", strain, reference.gradients["strain"], False),
            ("rows", rows, np.concatenate([reference.jacobian[name] for name in names], axis=1), False)):
        actual = tensor.cpu().numpy()
        relative = np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-300)
        mismatch = actual.view(np.uint64) != expected.view(np.uint64) if exact else (
            ~np.isfinite(actual) | (relative > 1e-14))
        print("coefficients", coefficients, name, "device", actual.tolist(),
              "host", expected.tolist(), "relative", relative.tolist(), flush=True)
        for index in zip(*np.where(mismatch)):
            a, b = actual[index], expected[index]
            print("MISMATCH", name, index, "reference", b, "device", a,
                  "ULP", abs(int(a.view(np.uint64)) - int(b.view(np.uint64))), flush=True)
            failed = True
    if failed:
        raise AssertionError("general pair-tail comparison differs outside the accepted bounds")
print("general pair-tail device quadrature passed", flush=True)
