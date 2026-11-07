"""Tuple fields, observed scalar parameters and empty theta in device frames."""
from pathlib import Path
import runpy
import sys

import mdir
import numpy as np
import torch

fixture = runpy.run_path(str(Path(__file__).with_name("frame_batch_entry.py")))
frames = fixture["frames"]
for precision, empty in ((mdir.Precision.Double, False), (mdir.Precision.Mixed, False),
                         (mdir.Precision.Double, True)):
    system, state = fixture["loaded"].make_system(), fixture["loaded"].make_state()
    system.cutoff, system.pairlist_distance = .8, .9
    system.dispersion = mdir.DispersionCorrection.None_
    spring = mdir.TupleTerm()
    spring.name, spring.expression, spring.arity = "spring", ".5*k*(r-r0)^2", 2
    spring.particles = np.array([[0, 1], [2, 3]], dtype=np.int64)
    spring.parameters = [("k", np.array([5., 7.])), ("r0", np.array([.2, .25]))]
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants, soft.observe = [("a", .125), ("l", .25)], ["a"]
    system.tuple_terms, system.pair_terms = [spring], [soft]
    system.tunables = [] if empty else [mdir.Tunable("k", "k", term="spring"),
        mdir.Tunable("r0", "r0", term="spring"), mdir.Tunable("a", "a", term="soft")]
    system.tunable_gradient = not empty
    execution = mdir.Execution()
    execution.target, execution.precision = mdir.Target.GPU, precision
    execution.deterministic, execution.reorder, execution.fast_math = True, False, False
    program = mdir.compile(system, state, fixture["integrator"],
        fixture["ensemble"], execution, mdir.Schedule())
    if empty:
        # The existing FrameEvaluator contract requires declared tunables.
        # The private device exercise entry does not change that contract.
        try:
            mdir.FrameEvaluator(program)
        except mdir.InputError as error:
            assert "declares no tunable parameters" in str(error), str(error)
        else:
            raise AssertionError("an evaluator without tunables was accepted")
        print("empty theta evaluator refusal passed", flush=True)
        continue
    evaluator = mdir.FrameEvaluator(program)
    names = [] if empty else ["k", "r0", "a"]
    initial = {name: np.array(evaluator.tunables[name]) for name in names}
    size = sum(value.size for value in initial.values())
    theta = torch.empty(size, device="cuda", dtype=torch.float64)
    positions = torch.tensor(frames, device="cuda", dtype=torch.float64)
    energy = torch.empty(2, device="cuda", dtype=torch.float64)
    gradient, strain = torch.empty_like(positions), torch.empty((2, 3, 3), device="cuda", dtype=torch.float64)
    rows = torch.empty((2, size), device="cuda", dtype=torch.float64)
    aux_names = evaluator._simulation._device_batch_aux_names
    aux = torch.empty((2, len(aux_names)), device="cuda", dtype=torch.float64)
    previous = None
    for call, factor in enumerate((1., 1., 1.03125)):
        values = {name: value * factor for name, value in initial.items()}
        if values:
            evaluator.tunables.update(values)
            theta.copy_(torch.tensor(np.concatenate([values[name] for name in names]), device="cuda"))
        reference = evaluator.evaluate(frames,
            gradients=("positions", "strain") if empty else ("positions", "strain", "tunables"))
        evaluator._evaluate_device_batch(positions, theta, energy,
            position_gradient=gradient, strain=strain, rows=None if empty else rows, aux=aux)
        expected_aux = np.stack([reference.virial, reference.volume] +
            [reference.observables[name] for name in aux_names[2:]], axis=1)
        actual_values = []
        for name, value, expected in (("energy", energy, reference.energy),
                ("positions", gradient, reference.gradients["positions"]),
                ("strain", strain, reference.gradients["strain"]), ("aux", aux, expected_aux)):
            actual = value.cpu().numpy()
            mismatch = actual.view(np.uint64) != expected.view(np.uint64)
            if np.any(mismatch):
                for index in zip(*np.where(mismatch)):
                    a, b = actual[index], expected[index]
                    print(precision, empty, name, index, "device", a, "host", b,
                        "ULP", abs(int(a.view(np.uint64)) - int(b.view(np.uint64))), flush=True)
                raise AssertionError(name)
            actual_values.append(actual.copy())
        if not empty:
            expected = np.concatenate([reference.jacobian[name] for name in names], axis=1)
            actual = rows.cpu().numpy()
            relative = np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-300)
            print(precision, "tuple/scalar rows maximum relative", float(relative.max()), flush=True)
            assert np.all(np.isfinite(actual) & (relative <= 1e-14)), (actual, expected, relative)
            actual_values.append(actual.copy())
        assert evaluator._simulation._device_batch_copy_bytes[:2] == [0, 0]
        if call == 1:
            assert all(np.array_equal(a.view(np.uint64), b.view(np.uint64))
                       for a, b in zip(actual_values, previous)), "deterministic repeat differs"
        previous = actual_values
        print(precision, "empty theta", empty, "call", call, "passed", flush=True)
print("tuple fields, observed scalar parameters and empty theta refusal passed", flush=True)
