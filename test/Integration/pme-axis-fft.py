"""Compare the generated PME convolution with cuFFT and an independent DFT."""
import contextlib
import importlib.util
from pathlib import Path
import os
import subprocess
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("gaussian", Path(__file__).with_name("reciprocal-grid.py"))
gaussian = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gaussian)
directory, libraries, mdir_opt = sys.argv[1:]
work = Path(directory)
work.mkdir(parents=True, exist_ok=True)


def run(grid, mixed, axis, force_only=False, tilted=False):
    gaussian.GRID = gaussian.reference.GRID = grid
    source = work / "input.mlir"
    with source.open("w") as out, contextlib.redirect_stdout(out):
        gaussian.emit(mixed=mixed, force_only=force_only, tilted=tilted)
    args = [mdir_opt, str(source), "--md-inline", "--canonicalize",
            "--convert-md-to-md-exec", "--md-exec-assign-algebra",
            "--md-exec-assign-storage=memory=device pme-axis-fft=" + ("on" if axis else "off"),
            "--convert-md-exec-to-gpu", "--md-alg-transform-collections",
            "--convert-md-alg-to-gpu", "--symbol-dce"]
    ir = subprocess.check_output(args)
    selected = b"call @mdrt_gpu_pme_axis_fft_" in ir
    assert selected == (axis and mixed), (grid, selected)
    binary = subprocess.check_output(
        ["mlir-opt", f"--gpu-lower-to-nvvm-pipeline=cubin-format=isa cubin-chip={os.environ['MDIR_LIT_GPU_CHIP']}",
         "--convert-ub-to-llvm", "--reconcile-unrealized-casts"], input=ir)
    result = subprocess.check_output(
        ["mlir-runner", "-e", "main", "--entry-point-result=void",
         "--shared-libs=" + libraries], input=binary)
    return np.asarray([float(word) for word in result.split()])


def compare(label, actual, expected, tol):
    assert actual.shape == expected.shape
    for name, selection in (("energy", slice(0, 1)), ("forces", slice(1, 10)),
                            ("virial", slice(10, 19))):
        delta = np.max(np.abs(actual[selection] - expected[selection]))
        scale = np.max(np.abs(expected[selection]))
        print(f"{label} {name} max |reference| {scale:.9g}, max |difference| {delta:.9g}, bound {tol*scale:.9g}")
        assert delta <= tol * scale


for grid in ((15, 7, 10), (8, 9, 12)):
    emitter = run(grid, True, True)
    force_only = run(grid, True, True, True)
    baseline = run(grid, True, False)
    double = run(grid, False, True)[:19]
    reference = gaussian.expected()[:19]
    print("mesh", grid)
    compare("double cuFFT / direct DFT", double, reference, 2e-7)
    compare("axis / direct DFT", emitter, reference, 3e-5)
    compare("axis / mixed cuFFT", emitter, baseline, 3e-5)
    np.testing.assert_allclose(force_only, reference[1:10], rtol=0,
                               atol=3e-5 * np.max(np.abs(reference[1:10])))

# The Trp-cage cells are tilted. Check their mesh sizes with the Gaussian
# split, including the 11-point register stage in the last X transform.
# Keep the Nyquist modes appreciable to exercise the real inverse end planes.
gaussian.BETA = 6.0
for grid in ((28, 28, 30), (32, 32, 32), (18, 20, 20), (22, 24, 24)):
    print("triclinic mesh", grid)
    emitter = run(grid, True, True, tilted=True)
    baseline = run(grid, True, False, tilted=True)
    double = run(grid, False, False, tilted=True)[:19]
    compare("axis / double cuFFT", emitter, double, 3e-5)
    compare("axis / mixed cuFFT", emitter, baseline, 3e-5)
