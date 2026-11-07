"""Device combining rules and charge maps against the host frame entry."""
from pathlib import Path
import sys

import mdir
import numpy as np
import torch

work = Path(sys.argv[1])
electrostatics = sys.argv[2] if len(sys.argv) > 2 else "Cutoff"
overrides = len(sys.argv) > 3 and sys.argv[3] == "overrides"
work.mkdir(parents=True, exist_ok=True)
(work / "system.top").write_text(
    "[ defaults ]\n1 2 no 1.0 1.0\n[ atomtypes ]\n"
    "A 6 12.0 0.0 A 0.30 0.60\nB 8 16.0 0.0 A 0.25 0.40\n" +
    ("[ nonbond_params ]\nA B 1 0.28 0.45\n" if overrides else "") +
    "[ moleculetype ]\nM 0\n[ atoms ]\n"
    "1 A 1 M A1 1 0.2 12.0\n2 B 1 M B2 2 -0.3 16.0\n"
    "3 A 1 M A3 3 0.1 12.0\n[ system ]\nmixing\n[ molecules ]\nM 1\n")
x = np.array([[1., 1., 1.], [1.4, 1.1, 1.], [1.2, 1.5, 1.2]])
(work / "system.gro").write_text("mixing\n3\n" + "".join(
    f"{1:5d}{'M':<5}{'X'+str(i):>5}{i:5d}{a:8.3f}{b:8.3f}{c:8.3f}\n"
    for i, (a, b, c) in enumerate(x, 1)) + "3.0 3.0 3.0\n")
frames = np.stack((x, x.copy()))
frames[1, 2, 2] += 0.015625

def compare(name, actual, expected, relative=False):
    actual = actual.cpu().numpy()
    mismatch = (~np.isfinite(actual) |
        (np.abs(actual - expected) > 1e-14 * np.maximum(np.abs(expected), 1e-300))) if relative else (
        actual.view(np.uint64) != expected.view(np.uint64))
    if np.any(mismatch):
        for index in zip(*np.where(mismatch)):
            a, b = actual[index], expected[index]
            print(name, index, "device", a, "host", b,
                  "relative", abs(a - b) / max(abs(b), 1e-300),
                  "ULP", abs(int(a.view(np.uint64)) - int(b.view(np.uint64))), flush=True)
        raise AssertionError(name)
    return actual.copy()

for precision in (mdir.Precision.Double, mdir.Precision.Mixed):
    for mixing in ("arithmetic", "geometric"):
        loaded = mdir.load_gromacs(str(work / "system.top"), str(work / "system.gro"))
        system, state = loaded.make_system(), loaded.make_state()
        system.cutoff, system.pairlist_distance = 0.8, 0.9
        system.electrostatics = getattr(mdir.Electrostatics, electrostatics)
        system.dispersion = mdir.DispersionCorrection.None_
        system.tunables = [mdir.Tunable("sigma", "sigma", mixing=mixing),
                           mdir.Tunable("epsilon", "epsilon"), mdir.Tunable("q", "charge")]
        if overrides:
            taken = np.array([-1, 0, -1], dtype=np.int64)
            system.tunables += [mdir.Tunable("sigma_pair", "sigma_pair", map=taken),
                                mdir.Tunable("epsilon_pair", "epsilon_pair", map=taken)]
        system.tunable_gradient = True
        execution = mdir.Execution()
        execution.target, execution.precision = mdir.Target.GPU, precision
        execution.deterministic, execution.reorder, execution.fast_math = True, False, False
        ensemble = mdir.Ensemble()
        ensemble.com_period = 0
        program = mdir.compile(system, state, mdir.Integrator(), ensemble, execution, mdir.Schedule())
        evaluator = mdir.FrameEvaluator(program)
        names = ("sigma", "epsilon", "q") + (("sigma_pair", "epsilon_pair") if overrides else ())
        initial = {name: np.array(evaluator.tunables[name]) for name in names}
        position = torch.tensor(frames, device="cuda", dtype=torch.float64)
        count = sum(initial[name].size for name in names)
        theta = torch.empty(count, device="cuda", dtype=torch.float64)
        energy = torch.empty(2, device="cuda", dtype=torch.float64)
        gradient = torch.empty_like(position)
        strain = torch.empty((2, 3, 3), device="cuda", dtype=torch.float64)
        rows = torch.empty((2, count), device="cuda", dtype=torch.float64)
        previous = None
        for call, factor in enumerate((1., 1., 1.03125)):
            values = {name: value * factor for name, value in initial.items()}
            evaluator.tunables.update(values)
            theta.copy_(torch.tensor(np.concatenate([values[name] for name in names]), device="cuda"))
            reference = evaluator.evaluate(frames, gradients=("tunables", "positions", "strain"))
            evaluator._evaluate_device_batch(position, theta, energy,
                position_gradient=gradient, strain=strain, rows=rows)
            transfers = evaluator._simulation._device_batch_copy_bytes
            assert transfers[:2] == [0, 0], (precision, mixing, call, transfers)
            actual = [compare("energy", energy, reference.energy),
                      compare("positions", gradient, reference.gradients["positions"]),
                      compare("strain", strain, reference.gradients["strain"]),
                      compare("rows", rows, np.concatenate([reference.jacobian[n] for n in names], axis=1), True)]
            if call == 1:
                assert all(np.array_equal(a.view(np.uint64), b.view(np.uint64))
                           for a, b in zip(actual, previous)), "deterministic repeat differs"
            previous = actual
            print(precision, mixing, "call", call, "passed", flush=True)
print("device combining rules and charge maps passed", flush=True)
