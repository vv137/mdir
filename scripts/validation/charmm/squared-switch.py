#!/usr/bin/env python3
"""Compare VSWITCH on the existing Trp-cage or two-POPC validation inputs.

The directory holds energy.inp and native.toml (Trp-cage), or popc.toml.
Use a private copy of the input directory. Generate Trp-cage through
run.py's build, convert, and terms phases first. CHARMM files and force
fields are external compatibility inputs, never copied into the repo.
GPU invocations must run under the device lock with no other device job.
"""

import argparse
from pathlib import Path
import re
import subprocess


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("work", type=Path)
parser.add_argument("--mdir", required=True)
parser.add_argument("--charmm", default="charmm")
parser.add_argument("--target", choices=["CPU", "GPU"], default="CPU")
parser.add_argument("--precision", choices=["DOUBLE", "MIXED"], default="DOUBLE")
args = parser.parse_args()
work = args.work.resolve()
energy = (work / "energy.inp").read_text().replace("vfswitch", "vswitch")
(work / "energy-squared-switch.inp").write_text(energy)
with (work / "energy-squared-switch.out").open("w") as log:
    subprocess.run([args.charmm, "-i", "energy-squared-switch.inp"],
                   cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
text = (work / "energy-squared-switch.out").read_text()
match = re.search(r"^TERMS VDW (\S+) IMNB (\S+)", text, re.M)
if not match:
    raise RuntimeError("CHARMM did not report VDW and IMNB energies")
reference = float(match[1]) + float(match[2])
source = work / "native.toml"
if not source.exists():
    source = work / "popc.toml"
control = source.read_text().replace("POWER_FORCE_SWITCH", "SQUARED_DISTANCE_SWITCH")
control = re.sub(r'(?m)^target\s*=.*$', f'target = "{args.target}"', control)
control = re.sub(r'(?m)^precision\s*=.*$', f'precision = "{args.precision}"', control)
if args.target == "GPU":
    control += 'neighbor_structure = "GROUPS"\n'
name = f"squared-switch-{args.target.lower()}-{args.precision.lower()}"
(work / f"{name}.toml").write_text(control)
text = subprocess.check_output([args.mdir, "run", str(work / f"{name}.toml")],
                               text=True)
(work / f"{name}.log").write_text(text)
energies = [float(re.search(r"MDIR:\s+" + re.escape(term) + r"\s+(\S+)",
                           text)[1])
            for term in ["Lennard-Jones", "Lennard-Jones 1-4"]]
actual = sum(energies)
difference = abs(actual-reference)
tolerance = 1e-5 if args.precision == "DOUBLE" else 5e-3
print(f"{args.target}/{args.precision}: LJ + LJ-14 (kcal/mol): "
      f"CHARMM {reference:.12g}, MDIR {actual:.12g}, "
      f"difference {difference:.3g}, tolerance {tolerance:.3g}")
assert difference < tolerance
