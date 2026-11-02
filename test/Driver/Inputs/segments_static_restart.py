#!/usr/bin/env python3
"""Prepare the static-restart controls used by the refusal test and its oracle.

The model and transformation follow Driver/restart.test; tiny checkpoints are
created by the supplied executable, never committed as binary fixtures.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("work", type=Path)
parser.add_argument("program")
args = parser.parse_args()
source, work = args.source.resolve(), args.work.resolve()
work.mkdir(parents=True, exist_ok=True)
program = shutil.which(args.program)
assert program, args.program
text = source.read_text().replace("Inputs/", str(source.parent / "Inputs") + "/")
text, steps = re.subn(r"(?m)^steps\s*=\s*200$", "steps = 1", text)
text, rows = re.subn(r"(?m)^energy_interval\s*=\s*50$", "energy_interval = 1", text)
assert steps == rows == 1
for precision in ("double", "mixed"):
    first = "first-" + precision
    control = text + '\ncheckpoint = "' + first + '.h5"\ncheckpoint_interval = 1\n'
    control += '[execution]\nthreads = 4\nprecision = "' + precision + '"\ntarget = "CPU"\n'
    (work / (first + ".toml")).write_text(control)
    result = subprocess.run([program, "run", first + ".toml"], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (work / (first + ".stdout")).write_bytes(result.stdout)
    (work / (first + ".stderr")).write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    static = control.replace('checkpoint = "' + first + '.h5"',
                             'checkpoint = "unused-' + precision + '.h5"')
    static, count = re.subn(r"(?m)^(coordinates\s*=.*)$",
        lambda match: match[1] + '\ncheckpoint = "' + first + '.h5"', static)
    assert count == 1
    (work / ("segments-static-restart-" + precision + ".toml")).write_text(static)
