"""Independent LJ energy/strain correction bound; all other outputs bitwise."""
import json
from pathlib import Path
import sys

import mdir
import numpy as np
import torch

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
(work / "system.top").write_text(
    "[ defaults ]\n1 2 no 1.0 1.0\n[ atomtypes ]\n"
    "A 0 40.0 0.0 A 0.300000 0.6363864\n"
    "[ moleculetype ]\nA 1\n[ atoms ]\n1 A 1 A A 1 0.0 40.0\n"
    "[ system ]\ntwo neutral particles\n[ molecules ]\nA 2\n")
x = np.array([[0., 0., 0.], [2., 0., 0.]])
(work / "system.gro").write_text("two neutral particles\n2\n" + "".join(
    f"{i:5d}{'A':<5}{'A':>5}{i:5d}{a:8.3f}{b:8.3f}{c:8.3f}\n"
    for i, (a, b, c) in enumerate(x, 1)) + "4.0 4.0 4.0\n")
loaded = mdir.load_gromacs(str(work / "system.top"), str(work / "system.gro"))
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance, system.switch_distance = 1.0, 1.15, 1.0
system.electrostatics = mdir.Electrostatics.Cutoff
system.truncation = getattr(mdir.Truncation, sys.argv[2]) if len(sys.argv) > 2 else mdir.Truncation.None_
system.dispersion = mdir.DispersionCorrection.EnergyPressure
system.tunables = [mdir.Tunable("sigma", "sigma_pair"),
                   mdir.Tunable("epsilon", "epsilon_pair")]
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
device_cells = torch.tensor(np.column_stack((cells, np.zeros_like(cells))),
                            device="cuda", dtype=torch.float64)
theta = torch.empty(2, device="cuda", dtype=torch.float64)
energy = torch.empty(2, device="cuda", dtype=torch.float64)
gradient = torch.empty_like(positions)
strain = torch.empty((2, 3, 3), device="cuda", dtype=torch.float64)
rows = torch.empty((2, 2), device="cuda", dtype=torch.float64)
maximum = maximum_strain = 0.0
energy_ulp = strain_ulp = 0
diagonal = np.eye(3, dtype=bool)[None].repeat(2, axis=0)
def ulps(actual, expected):
    return max((abs(int(a) - int(b)) for a, b in
                zip(actual.view(np.uint64).flat, expected.view(np.uint64).flat)), default=0)

for sigma in (0.15, 0.20, 0.25, 0.30, 0.3150752406575124,
              0.3399669508423535, 0.35, 0.40):
    theta.copy_(torch.tensor([sigma, 0.6363864], device="cuda", dtype=torch.float64))
    evaluator.tunables.update({"sigma": np.array([sigma]), "epsilon": np.array([0.6363864])})
    reference = evaluator.evaluate(frames, cells=cells, gradients=("positions", "strain", "tunables"))
    evaluator._evaluate_device_batch(positions, theta, energy, cells=device_cells,
        position_gradient=gradient, strain=strain, rows=rows,
        stream=torch.cuda.current_stream().cuda_stream or 1)
    actual_u, actual_g, actual_s = (value.cpu().numpy() for value in (energy, gradient, strain))
    relative = np.abs(actual_u - reference.energy) / np.maximum(np.abs(reference.energy), 1e-300)
    maximum = max(maximum, float(relative.max()))
    energy_ulp = max(energy_ulp, ulps(actual_u, reference.energy))
    print("sigma", sigma, "energy device", actual_u.tolist(), "host", reference.energy.tolist(),
          "relative", relative.tolist(), flush=True)
    failed = not bool(np.all(np.isfinite(actual_u) & (relative <= 1e-14)))
    # Every direct interaction is outside the cutoff and the charges are
    # zero: only the independent dispersion correction enters the diagonal.
    actual_d = actual_s[diagonal]
    expected_d = reference.gradients["strain"][diagonal]
    strain_relative = np.abs(actual_d - expected_d) / np.maximum(np.abs(expected_d), 1e-300)
    maximum_strain = max(maximum_strain, float(strain_relative.max()))
    strain_ulp = max(strain_ulp, ulps(actual_d, expected_d))
    print("strain correction relative", strain_relative.tolist(), flush=True)
    if not np.all(np.isfinite(actual_d) & (strain_relative <= 1e-14)):
        failed = True
        print("strain correction device", actual_d.tolist(), "host", expected_d.tolist(), flush=True)
    for name, actual, expected in (("positions", actual_g, reference.gradients["positions"]),
                                   ("strain off diagonal", actual_s[~diagonal],
                                    reference.gradients["strain"][~diagonal])):
        mismatch = actual.view(np.uint64) != expected.view(np.uint64)
        if np.any(mismatch):
            failed = True
            for index in zip(*np.where(mismatch)):
                a, b = actual[index], expected[index]
                print(name, index, "device", a, "host", b,
                      "ULP", abs(int(a.view(np.uint64)) - int(b.view(np.uint64))), flush=True)
    if failed:
        raise AssertionError("integrated LJ entry differs outside the accepted contracts")
    actual_rows = rows.cpu().numpy()
    expected_rows = np.concatenate([reference.jacobian[n] for n in ("sigma", "epsilon")], axis=1)
    relative_rows = np.abs(actual_rows - expected_rows) / np.maximum(np.abs(expected_rows), 1e-300)
    print("rows device", actual_rows.tolist(), "host", expected_rows.tolist(),
          "relative", relative_rows.tolist(), flush=True)
    if not np.all(np.isfinite(actual_rows) & (relative_rows <= 1e-14)):
        raise AssertionError("LJ parameter rows differ outside the accepted bound")
    first = [value.copy() for value in (actual_u, actual_g, actual_s)]
    evaluator._evaluate_device_batch(positions, theta, energy, cells=device_cells,
        position_gradient=gradient, strain=strain,
        stream=torch.cuda.current_stream().cuda_stream or 1)
    for name, value, expected in zip(("energy", "positions", "strain"),
                                    (energy, gradient, strain), first):
        actual = value.cpu().numpy()
        if not np.array_equal(actual.view(np.uint64), expected.view(np.uint64)):
            raise AssertionError("deterministic device repeat differs: " + name)
result = dict(energies=16, strain_correction_values=48, exact_gradients=96,
              exact_strain_off_diagonals=96, deterministic_repeat_values=256,
              energy_max_relative=maximum, energy_max_ulp=energy_ulp,
              strain_max_relative=maximum_strain, strain_max_ulp=strain_ulp)
(work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print("integrated LJ entry passed", json.dumps(result), flush=True)
