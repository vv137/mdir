#!/usr/bin/env python3
"""Alternate semantic FFT host/device strategies on one scheduler-owned GPU.

Invoke compare through gpu-run --idle with MDIR_GPU pinned. The internal
rtclock interval excludes module compilation and three warmup compositions.
The host strategy remains the default regardless of this experiment.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def emit(size, iterations, precision):
    scalar = "f32" if precision == "mixed" else "f64"
    real = f"tensor<{size}x{size}x{size}x{scalar}>"
    half = f"tensor<{size}x{size}x{size // 2 + 1}xcomplex<{scalar}>>"
    wide = "  %wide = arith.extf %value : f32 to f64\n" if scalar == "f32" else ""
    value = "%wide" if scalar == "f32" else "%value"
    return f'''func.func private @rtclock() -> f64
func.func private @mdrtPrintFFTValue(f64)
func.func @roundtrip() -> f64 {{
  %input = arith.constant dense<1.0> : {real}
  %f = md.linear_transform %input kind("fft3d") direction("forward")
    adjoint("backward") pairing("real_hermitian_half") : {real} -> {half}
  %b = md.linear_transform %f kind("fft3d") direction("backward")
    adjoint("forward") pairing("real_hermitian_half") : {half} -> {real}
  %c0 = arith.constant 0 : index
  %value = tensor.extract %b[%c0, %c0, %c0] : {real}
{wide}  return {value} : f64
}}
func.func @main() {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %warm = arith.constant 3 : index
  %count = arith.constant {iterations} : index
  %zero = arith.constant 0.0 : f64
  scf.for %i = %c0 to %warm step %c1 {{
    %unused = func.call @roundtrip() : () -> f64
  }}
  %start = func.call @rtclock() : () -> f64
  %sum = scf.for %i = %c0 to %count step %c1 iter_args(%acc = %zero) -> (f64) {{
    %result = func.call @roundtrip() : () -> f64
    %next = arith.addf %acc, %result : f64
    scf.yield %next : f64
  }}
  %end = func.call @rtclock() : () -> f64
  %elapsed = arith.subf %end, %start : f64
  func.call @mdrtPrintFFTValue(%elapsed) : (f64) -> ()
  func.call @mdrtPrintFFTValue(%sum) : (f64) -> ()
  return
}}
'''


def command(args, source=None, env=None):
    return subprocess.run([str(a) for a in args], input=source, text=True,
                          capture_output=True, check=True, env=env)


def compare(args):
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    llvm_bin = (Path(args.llvm_bin) if args.llvm_bin
                else Path(args.runner_utils).parent.parent / "bin")
    report = []
    for precision in args.precision:
        source = emit(args.size, args.iterations, precision)
        programs = {}
        for label, tree, strategy in (("main-host", args.main_build, "host"),
                                       ("branch-device", args.branch_build, "device")):
            tree = Path(tree)
            bridge = ("--convert-linear-transform-values" if strategy == "host"
                      else "--convert-linear-transform-values=memory=device")
            lowered = command([tree / "bin/mdir-opt", bridge,
                "--one-shot-bufferize", "--buffer-deallocation-pipeline",
                "--convert-linalg-to-loops", "--md-exec-place-linear-transform-device",
                "--convert-md-exec-to-gpu", "--convert-complex-to-llvm"], source).stdout
            llvm = command([llvm_bin / "mlir-opt", "--gpu-lower-to-nvvm-pipeline=cubin-format=isa",
                            "--reconcile-unrealized-casts"], lowered).stdout
            path = scratch / f"{precision}-{label}.mlir"
            path.write_text(llvm)
            programs[label] = (tree, path)
        for round_index in range(args.rounds):
            order = list(programs)
            if round_index % 2:
                order.reverse()
            for label in order:
                tree, path = programs[label]
                libraries = ",".join((args.runner_utils, str(tree / "lib/libmdrt.so"),
                                      str(tree / "lib/libmdrt_cuda.so")))
                env = dict(os.environ, MDIR_DEVICE_MEMORY="1")
                result = command([llvm_bin / "mlir-runner", path, "-e", "main",
                    "--entry-point-result=void", f"--shared-libs={libraries}"], env=env)
                (scratch / f"{precision}-{label}-{round_index}.log").write_text(
                    result.stdout + result.stderr)
                seconds, checksum = map(float, result.stdout.split())
                expected = args.size ** 3 * args.iterations
                tolerance = 1e-6 * expected if precision == "mixed" else 2e-11 * expected
                if abs(checksum - expected) > tolerance:
                    raise ValueError(f"{label}: checksum {checksum}, expected {expected}")
                memory = re.search(r"peak ([\d.]+) MiB allocated, ([\d.]+) MiB with the pool",
                                   result.stderr)
                if not memory:
                    raise ValueError("missing peak-memory report")
                row = dict(precision=precision, strategy=label, round=round_index,
                    ms_per_composition=seconds * 1000 / args.iterations,
                    peak_live_mib=float(memory[1]), peak_reserved_mib=float(memory[2]),
                    reference=expected, difference=checksum - expected, tolerance=tolerance)
                report.append(row)
                print(json.dumps(row), flush=True)
        for label in programs:
            values = [r["ms_per_composition"] for r in report
                      if r["precision"] == precision and r["strategy"] == label]
            print(json.dumps(dict(precision=precision, strategy=label,
                                  mean_ms=statistics.mean(values),
                                  stdev_ms=statistics.stdev(values) if len(values) > 1 else 0)),
                  flush=True)
    (scratch / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("emit", "compare"))
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--precision", nargs="+", choices=("mixed", "double"),
                        default=["mixed", "double"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--main-build")
    parser.add_argument("--branch-build")
    parser.add_argument("--runner-utils")
    parser.add_argument("--llvm-bin")
    parser.add_argument("--scratch")
    args = parser.parse_args()
    if args.size < 2 or args.iterations < 1 or args.rounds < 1:
        parser.error("positive iteration/round counts and size >= 2 required")
    if args.action == "emit":
        print(emit(args.size, args.iterations, args.precision[0]))
    else:
        if not all((args.main_build, args.branch_build, args.runner_utils, args.scratch)):
            parser.error("compare requires both builds, runner-utils, and scratch")
        compare(args)


if __name__ == "__main__":
    main()
