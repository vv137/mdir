"""Runs one distributed CPU input for two numbers of steps and checks that
the longer run's peak resident memory exceeds the shorter one's by at most
BOUND MiB, with the same particles and output.

Usage: distributed_memory.py <mdir> <directory> <control file> SHORT LONG BOUND

The procedure of a domain is called at every step; before its buffers were
freed at its end (md-dist-release-domain-buffers), each call kept the local
rows it packed and the storage of its fields, and the two domains of the
mixture of charged.toml kept about 0.09 MiB a step. It prints both peaks.
"""
import os
from pathlib import Path
import re
import subprocess
import sys

mdir = sys.argv[1]
root = Path(sys.argv[2]).resolve()
control_file = Path(sys.argv[3]).resolve()
short, long, bound = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
coordinates = control_file.parent / "Inputs" / "mixture.pdb"
base = control_file.read_text()
base = re.sub(r"(?m)^coordinates\s*=.*$", f'coordinates = "{coordinates}"', base)
base = re.sub(r"(?m)^energy_interval\s*=.*$", "energy_interval = 50", base)
base = re.sub(r"(?ms)^\[execution\].*", "", base)
base = re.sub(r"(?ms)^\[\[energy.pair_override\]\].*?(?=^\[\[energy.type\]\])", "", base)
base = re.sub(r"(?m)^dispersion_correction.*\n", "", base)
base = base.replace("[dynamics]", "[dynamics]\nseed = 314159")
base += """
[execution]
target = "CPU"
threads = 2
precision = "DOUBLE"
deterministic = false
[distribute]
domains = 2
transport = "local"
traversal = "midpoint"
"""


def peak(steps):
    path = root / str(steps)
    path.mkdir(parents=True, exist_ok=True)
    text = re.sub(r"(?m)^steps\s*=.*$", f"steps = {steps}", base)
    (path / "run.toml").write_text(text)
    with open(path / "run.log", "w") as log:
        process = subprocess.Popen([mdir, "run", "run.toml"], cwd=path,
                                   stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(process.pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        sys.exit(f"{path}: mdir run failed:\n{(path / 'run.log').read_text()}")
    return usage.ru_maxrss / 1024.0  # KiB on Linux


low, high = peak(short), peak(long)
print(f"peak {low:.1f} MiB after {short} steps, {high:.1f} MiB after {long} steps")
if high - low > bound:
    sys.exit(f"the peak grew by {high - low:.1f} MiB, more than {bound} MiB")
print("bounded")
