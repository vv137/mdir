"""Verify valid staged tensor/linalg input reaches segmented carry rewriting."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("scratch", type=Path)
parser.add_argument("mdir")
parser.add_argument("printer")
parser.add_argument("kind", choices=("tensor-helper", "exporter-tensor-linalg"))
args = parser.parse_args()
root, place = args.root.resolve(), args.scratch.resolve()
place.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "program_dialect_order", root / "scripts/builder-equivalence/order.py")
order = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = order
spec.loader.exec_module(order)
controls, _ = order.controls(root, place)
verified = 0
for precision in ("DOUBLE", "MIXED"):
    original = next(path for path in controls
                    if path.name == f"model-{args.kind}-{precision}.toml")
    text = original.read_text()
    assert text.count("[execution]\n") == 1
    control = place / (precision + ".toml")
    control.write_text(text.replace("[execution]\n", '[execution]\ntarget = "GPU"\n'))
    for stage in ("module", "lowered"):
        emitted = subprocess.run(
            [args.mdir, "emit", "--segments", "--stage=" + stage, str(control)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        assert emitted.returncode == 0, emitted.stderr.decode()
        assert b"mdir.carries" not in emitted.stdout, "carry rewrite was skipped"
        output = place / (precision + "-" + stage + ".mlir")
        output.write_bytes(emitted.stdout)
        result = subprocess.run(
            [args.printer, "-", "-o", os.devnull], input=emitted.stdout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        assert result.returncode == 0, result.stderr.decode()
    verified += 1
assert verified == 2
print(f"{args.kind}: {verified} segmented modules and lowered programs verified")
