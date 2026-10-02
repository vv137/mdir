#!/usr/bin/env python3
"""Compares semi-isotropic with isotropic stochastic cell rescaling on a
box of OPC water (D119). A liquid has no shape of its own, so the two must
sample the same volume: the same mean density and the same compressibility
from the fluctuations of the volume. With a frozen height (the
compressibility of z 0) the area alone takes those fluctuations, and the
height must not move.

    scripts/validation/barostat-semi/water.py WORK MDIR [--ns 2]

WORK holds the box and the equilibrated state of the tests of the
ensembles (scripts/validation/ensembles/run.py: opc.prmtop, opc.inpcrd,
equil.h5); each run starts from equil.h5 at 300 K and 1 atm, on the GPU
in mixed precision, with the settings of those tests. Needs numpy.
"""
import argparse
import math
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ensembles"))
from analyze import inefficiency  # noqa: E402

CONTROL = """[input]
topology    = "opc.prmtop"
coordinates = "opc.inpcrd"
checkpoint  = "equil.h5"

[output]
energy_interval = 500
checkpoint = "{name}.h5"
checkpoint_interval = {steps}

[energy]
cutoff            = 9.0
pairlist_distance = 12.0
pruned_distance   = 9.6
electrostatics    = "PME"

[pme]
tolerance   = 1e-05
max_spacing = 1.0

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {steps}
seed       = {seed}

[ensemble]
ensemble    = "NPT"
temperature = 300.0
pressure    = 1.0

[thermostat]
method        = "V-RESCALE"
time_constant = 1.0
interval      = 25

[barostat]
method        = "C-RESCALE"
time_constant = 2.0
interval      = 25
{coupling}
[constraints]
rigid_water = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
neighbor_structure = "GROUPS"
"""

RUNS = {
    "isotropic": "",
    "semi": 'coupling = "SEMI_ISOTROPIC"\n',
    "frozen": 'coupling = "SEMI_ISOTROPIC"\ncompressibility_z = 0.0\n',
}


def run(args, name, steps, seed):
    with open(os.path.join(args.work, name + ".toml"), "w") as file:
        file.write(CONTROL.format(name=name, steps=steps, seed=seed,
                                  coupling=RUNS[name]))
    with open(os.path.join(args.work, name + ".log"), "w") as out:
        result = subprocess.run([args.mdir, "run", name + ".toml"],
                                cwd=args.work, stdout=out,
                                stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"{name} failed; see {os.path.join(args.work, name + '.log')}")


def volumes(path):
    """The volumes of the rows of a log after the first tenth, in Å^3."""
    rows = []
    for line in open(path):
        fields = line.split()
        if line.startswith("INFO:") and fields[1].isdigit():
            rows.append(float(fields[-1]))
    rows = np.array(rows)
    return rows[len(rows) // 10:]


def box(mdir, work, name):
    """The edges of the cell at the end of a run, in nm."""
    text = subprocess.run([mdir, "checkpoint", name + ".h5"],
                          cwd=work, capture_output=True, text=True).stdout
    return [float(v) for v in
            re.search(r"box: +(\S+) (\S+) (\S+) nm", text).groups()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("mdir")
    parser.add_argument("--ns", type=float, default=2.0)
    args = parser.parse_args()
    args.work = os.path.abspath(args.work)
    steps = int(args.ns * 1000 / 0.002)
    for seed, name in enumerate(RUNS, start=21):
        run(args, name, steps, seed)

    # The compressibility from the fluctuations of the volume,
    # beta_T = <dV^2> / (kB T <V>): with V in Å^3, var 1e-30 / (kT mean) is
    # in 1/Pa, and 1e10 times that in 1e-5/bar.
    kT = 1.380649e-23 * 300.0  # J
    mass = 1039 * 18.01528 / 6.02214076e23  # g
    print(f"{'run':10s} {'density, g/cm^3':>24s} {'beta_T, 1e-5/bar':>22s}"
          f" {'g':>6s} {'edges at the end, nm':>26s}")
    for name in RUNS:
        v = volumes(os.path.join(args.work, name + ".log"))
        g = inefficiency(v)
        n = len(v) / g
        mean = v.mean()
        density = mass / (mean * 1e-24)
        density_error = density * v.std(ddof=1) / math.sqrt(n) / mean
        var = v.var(ddof=1)
        beta = var * 1e-30 / (kT * mean) * 1e10
        beta_error = beta * math.sqrt(2.0 / (n - 1))
        edges = box(args.mdir, args.work, name)
        print(f"{name:10s} {density:13.5f} ± {density_error:8.5f}"
              f" {beta:12.2f} ± {beta_error:6.2f} {g:6.1f}"
              f"   {edges[0]:.4f} {edges[1]:.4f} {edges[2]:.4f}")


if __name__ == "__main__":
    main()
