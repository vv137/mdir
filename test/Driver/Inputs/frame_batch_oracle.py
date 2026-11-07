"""Finite differences of an independent NumPy torsion, CPU and device frames."""
import json
from pathlib import Path
import runpy
import sys

import mdir
import numpy as np
import torch

# Reuse the small topology and the already checked internal API fixture.
fixture = runpy.run_path(str(Path(__file__).with_name("frame_batch_entry.py")))
work = Path(sys.argv[1])
system, state = fixture["system"], fixture["state"]
coefficient = 2.25

def oracle(frames, k=coefficient):
    a, b, c, d = (frames[:, i] for i in range(4))
    b0, b1, b2 = a - b, c - b, d - c
    unit = b1 / np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.sum(b0 * unit, axis=1)[:, None] * unit
    w = b2 - np.sum(b2 * unit, axis=1)[:, None] * unit
    sine = np.sum(np.cross(unit, v) * w, axis=1)
    cosine = np.sum(v * w, axis=1)
    phi = np.arctan2(sine, cosine)
    return k * (1 + np.cos(phi - np.deg2rad(20.)))

def richardson(function, h=1e-4):
    coarse = (function(h) - function(-h)) / (2 * h)
    fine = (function(h / 2) - function(-h / 2)) / h
    return (4 * fine - coarse) / 3

results = []
for precision, deterministic, dtype in ((mdir.Precision.Double, True, torch.float64),
                                      (mdir.Precision.Mixed, True, torch.float64),
                                      (mdir.Precision.Mixed, False, torch.float32)):
    frames = fixture["frames"].astype(np.float32 if dtype == torch.float32 else np.float64).astype(np.float64)
    expected_u = oracle(frames)
    expected_g = np.empty_like(frames)
    expected_s = np.empty((2, 3, 3))
    for atom in range(4):
        for coordinate in range(3):
            direction = np.zeros_like(frames)
            direction[:, atom, coordinate] = 1
            expected_g[:, atom, coordinate] = richardson(lambda h: oracle(frames + h * direction))
    for i in range(3):
        for j in range(3):
            direction = np.zeros((3, 3))
            direction[i, j] = 1
            expected_s[:, i, j] = richardson(lambda h: oracle(frames @ (np.eye(3) + h * direction).T))
    expected_theta = richardson(lambda h: oracle(frames, coefficient + h))[:, None]
    execution = mdir.Execution()
    execution.target, execution.precision = mdir.Target.GPU, precision
    execution.deterministic, execution.reorder, execution.fast_math = deterministic, False, False
    program = mdir.compile(system, state, fixture["integrator"], fixture["ensemble"], execution, mdir.Schedule())
    evaluator = mdir.FrameEvaluator(program)
    theta = torch.tensor([coefficient], device="cuda", dtype=torch.float64)
    position = torch.tensor(frames, device="cuda", dtype=dtype)
    energy = torch.empty(2, device="cuda", dtype=torch.float64)
    gradient = torch.empty_like(position)
    strain = torch.empty((2, 3, 3), device="cuda", dtype=torch.float64)
    rows = torch.empty((2, 1), device="cuda", dtype=torch.float64)
    evaluator._evaluate_device_batch(position, theta, energy,
        position_gradient=gradient, strain=strain, rows=rows)
    assert evaluator._simulation._device_batch_copy_bytes[:2] == [0, 0]
    execution.target = mdir.Target.CPU
    cpu = mdir.FrameEvaluator(mdir.compile(system, state, fixture["integrator"], fixture["ensemble"], execution, mdir.Schedule()))
    cpu.tunables["k"] = np.array([coefficient])
    host = cpu.evaluate(frames, gradients=("positions", "strain", "tunables"))
    # Determinism fixes repeatability; it does not turn Mixed into Double.
    # The Mixed comparison uses the evaluator bounds of the PR 1 handoff.
    tolerances = (1e-12, 5e-7, 5e-7, 1e-6) if precision == mdir.Precision.Double else (5e-3, 1e-4, 1e-4, 1e-6)
    entry = {"precision": str(precision), "deterministic": deterministic, "input_dtype": str(dtype)}
    for name, actual, reference, host_value, tolerance in zip(
            ("energy", "positions", "strain", "theta"),
            (energy.cpu().numpy(), gradient.cpu().numpy(), strain.cpu().numpy(), rows.cpu().numpy()),
            (expected_u, expected_g, expected_s, expected_theta),
            (host.energy, host.gradients["positions"], host.gradients["strain"], host.jacobian["k"]), tolerances):
        device_error = float(np.max(np.abs(actual - reference)))
        cpu_error = float(np.max(np.abs(host_value - reference)))
        entry[name] = {"device_error": device_error, "cpu_error": cpu_error, "tolerance": tolerance}
        print(entry["precision"], deterministic, name, entry[name], flush=True)
        assert np.isfinite(actual).all() and device_error <= tolerance and cpu_error <= tolerance, entry
    results.append(entry)
(work / "finite-difference-results.json").write_text(json.dumps(results, indent=2) + "\n")
print("independent frame finite differences: CPU and GPU, Double and Mixed passed", flush=True)
