"""Internal device batch entry against the host frame entry, without tails."""
from pathlib import Path
import sys

import mdir
import numpy as np
import torch

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
(work / "system.top").write_text(
    "[ defaults ]\n1 2 no 0.5 0.8333333333333333\n"
    "[ atomtypes ]\nC 6 12.0 0.0 A 0.0 0.0\n"
    "[ moleculetype ]\nTORSION 0\n[ atoms ]\n" +
    "".join(f"{i} C 1 TOR C{i} {i} 0.0 12.0\n" for i in range(1, 5)) +
    "[ dihedrals ]\n1 2 3 4 1 20 1.5 1\n"
    "[ system ]\ntorsion\n[ molecules ]\nTORSION 1\n")
x = np.array([[1., 1., 1.], [1.2, 1.1, 1.],
              [1.3, 1.2, 1.25], [1.5, 1.1, 1.4]])
(work / "system.gro").write_text("torsion\n4\n" + "".join(
    f"{1:5d}{'TOR':<5}{'C'+str(i):>5}{i:5d}{a:8.3f}{b:8.3f}{c:8.3f}\n"
    for i, (a, b, c) in enumerate(x, 1)) + "3.0 3.0 3.0\n")
loaded = mdir.load_gromacs(str(work / "system.top"), str(work / "system.gro"))
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = 0.8, 0.9
system.dispersion = mdir.DispersionCorrection.None_
system.tunables = [mdir.Tunable("k", "dihedral_k")]
system.tunable_gradient = True
execution = mdir.Execution()
execution.target, execution.precision = mdir.Target.GPU, mdir.Precision.Double
execution.deterministic, execution.reorder, execution.fast_math = True, False, False
integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
integrator.timestep, ensemble.com_period = 0.0001, 0
program = mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())
evaluator = mdir.FrameEvaluator(program)
frames = np.stack((x, x.copy()))
frames[1, 3, 2] += 0.015625
positions = torch.tensor(frames, device="cuda", dtype=torch.float64)
theta = torch.tensor(evaluator.tunables["k"], device="cuda", dtype=torch.float64)
energy = torch.empty(2, device="cuda", dtype=torch.float64)
gradient = torch.empty_like(positions)
strain = torch.empty((2, 3, 3), device="cuda", dtype=torch.float64)
rows = torch.empty((2, theta.numel()), device="cuda", dtype=torch.float64)
tangent = torch.tensor([0.5], device="cuda", dtype=torch.float64)
cotangent = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.float64)
jvp = torch.empty(2, device="cuda", dtype=torch.float64)
vjp = torch.empty(theta.numel(), device="cuda", dtype=torch.float64)
aux = torch.empty((2, 2), device="cuda", dtype=torch.float64)
assert evaluator._simulation._device_batch_aux_names == ["virial", "volume"]
start_counts, previous = None, None
for call, coefficient in enumerate((1.5, 1.5, 2.25)):
    theta.fill_(coefficient)
    evaluator.tunables["k"] = np.array([coefficient])
    reference = evaluator.evaluate(frames, gradients=("tunables", "positions", "strain"))
    evaluator._evaluate_device_batch(positions, theta, energy,
        position_gradient=gradient, strain=strain, rows=rows if call != 1 else None,
        tangent=tangent, cotangent=cotangent, jvp=jvp, vjp=vjp, aux=aux,
        stream=torch.cuda.current_stream().cuda_stream or 1)
    actual_values = []
    for name, actual, expected in (
            ("energy", energy.cpu().numpy(), reference.energy),
            ("positions", gradient.cpu().numpy(), reference.gradients["positions"]),
            ("strain", strain.cpu().numpy(), reference.gradients["strain"]),
            ("rows", rows.cpu().numpy(), reference.jacobian["k"]),
            ("jvp", jvp.cpu().numpy(), reference.jacobian["k"] @ np.array([0.5])),
            ("vjp", vjp.cpu().numpy(), np.array([0.25, -0.5]) @ reference.jacobian["k"]),
            ("aux", aux.cpu().numpy(), np.stack((reference.virial, reference.volume), axis=1))):
        mismatch = actual.view(np.uint64) != expected.view(np.uint64)
        if np.any(mismatch):
            for index in zip(*np.where(mismatch)):
                a, b = actual[index], expected[index]
                print(name, index, "device", a, "host", b,
                      "ULP", abs(int(a.view(np.uint64)) - int(b.view(np.uint64))), flush=True)
            raise AssertionError(name)
        actual_values.append(actual.copy())
    counts = evaluator._simulation._device_batch_start_arguments
    if start_counts is None:
        start_counts = counts
    assert counts == start_counts, "theta update restarted the activation"
    if call == 1:
        assert all(np.array_equal(a.view(np.uint64), b.view(np.uint64))
                   for a, b in zip(previous, actual_values)), "device repeat differs"
    previous = actual_values
    print(f"call {call}: all 46 results bitwise identical to host", flush=True)
    print(f"call {call}: JVP/VJP bitwise identical; rows requested: {call != 1}", flush=True)
print("device frame entry: rows, repeated calls and theta updates passed", flush=True)

def expect_input_error(call, text):
    try:
        call()
    except mdir.InputError as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError(f"expected InputError: {text}")

bad = positions.clone()
bad[1, 0, 0] = float("nan")
expect_input_error(lambda: evaluator._evaluate_device_batch(bad, theta, energy), "frame 1")
theta.fill_(float("nan"))
expect_input_error(lambda: evaluator._evaluate_device_batch(positions, theta, energy), "theta")
theta.fill_(2.25)
evaluator._evaluate_device_batch(positions, theta, energy)
assert np.array_equal(energy.cpu().numpy().view(np.uint64), reference.energy.view(np.uint64))
expect_input_error(lambda: evaluator._evaluate_device_batch(positions, energy[:1], energy), "overlap")
vjp.fill_(float("nan"))
evaluator._evaluate_device_batch(positions[:0], theta, energy[:0],
    cotangent=cotangent[:0], vjp=vjp)
assert torch.equal(vjp, torch.zeros_like(vjp)), "empty batch VJP must be zero"
print("device frame entry: auxiliary outputs, input errors, recovery and empty batch passed", flush=True)
