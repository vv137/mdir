#!/usr/bin/env python3
"""Compares the GPU with the CPU at constant temperature: the dipeptide in
OPC of pme-settle.test, in double precision on both, equilibrated for
EQUIL ps at 300 K and 1 atm on the GPU (the box of tleap holds the water
at about 0.6 g/cm^3, which at constant volume relaxes for hundreds of ps),
then PS ps at constant volume on each from that checkpoint, with seeds of
the thermostat of their own, so that the two are independent samples (with
one seed the noise of the thermostat is common to both, and the errors of
the difference below, which treat them as independent, would be too
large). What is compared is the distribution: the means of the potential
energy, the temperature, and the total energy, and the standard deviation
of the potential energy, each with the error of its samples spaced by
their statistical inefficiency.

    scripts/validation/gpu-cpu/nvt.py WORK MDIR [--ps 500] [--equil 100]
        [--threads 64]

Needs numpy. Both run with velocity Verlet at 2 fs, SHAKE on the bonds of
hydrogen and SETTLE, particle mesh Ewald with a cutoff of 9 Å, and
stochastic velocity rescaling every 10 steps (tau_T = 1 ps); the GPU is
the one that CUDA_VISIBLE_DEVICES leaves visible.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ensembles"))
from analyze import inefficiency  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "..")

CONTROL = """[input]
topology   = "opc.prmtop"
coordinates = "opc.inpcrd"
{checkpoint}
[output]
energy_interval = {interval}
checkpoint = "{name}.h5"
checkpoint_interval = {steps}

[energy]
cutoff        = 9.0
pairlist_distance = 10.0
electrostatics = "PME"

[dynamics]
integrator    = "VELOCITY_VERLET"
time_step     = 0.002
steps = {steps}
seed = {seed}

[ensemble]
ensemble    = "{ensemble}"
temperature = 300.0
{pressure}
[thermostat]
method = "V-RESCALE"
interval = 10
time_constant = 1.0
{barostat}
[constraints]
hydrogen_bonds = true
rigid_water = true

[boundary]
type = "PERIODIC"

[execution]
target = "{target}"
precision = "DOUBLE"
{threads}"""


def run(mdir, work, name, text):
    with open(os.path.join(work, name + ".toml"), "w") as file:
        file.write(text)
    with open(os.path.join(work, name + ".log"), "w") as out:
        result = subprocess.run([mdir, "run", name + ".toml"], cwd=work,
                                stdout=out, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"{name} failed; see {os.path.join(work, name + '.log')}")


def rows(path):
    """The rows of the energies after the first one: time, total,
    potential, temperature."""
    out = []
    for line in open(path):
        fields = line.split()
        if line.startswith("INFO:") and fields[1].isdigit():
            out.append([float(fields[2]), float(fields[3]), float(fields[4]),
                        float(fields[6])])
    return np.array(out[1:])


def estimate(x):
    """The mean and the standard deviation of x with their errors, and the
    statistical inefficiency and the number of independent samples."""
    g = inefficiency(x)
    n = len(x) / g
    sd = np.std(x, ddof=1)
    return np.mean(x), sd / math.sqrt(n), sd, sd / math.sqrt(2 * (n - 1)), g, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("mdir")
    parser.add_argument("--ps", type=float, default=500.0)
    parser.add_argument("--equil", type=float, default=100.0)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    for name in ("opc.prmtop", "opc.inpcrd"):
        shutil.copy(os.path.join(ROOT, "test", "Driver", "Inputs", "opc", name),
                    work)

    equil = int(args.equil / 0.002)
    run(args.mdir, work, "equil",
        CONTROL.format(checkpoint="", interval=equil // 10, name="equil",
                       steps=equil, target="GPU", threads="",
                       ensemble="NPT", pressure="pressure    = 1.0\n", seed=11,
                       barostat='\n[barostat]\nmethod = "C-RESCALE"\n'
                                'time_constant = 2.0\n'))
    steps = int(args.ps / 0.002)
    for target in ("GPU", "CPU"):
        name = target.lower()
        run(args.mdir, work, name,
            CONTROL.format(checkpoint='checkpoint = "equil.h5"', interval=50,
                           name=name, steps=steps, target=target,
                           threads=f"threads = {args.threads}\n"
                           if target == "CPU" else "", ensemble="NVT",
                           pressure="", barostat="",
                           seed=12 if target == "GPU" else 13))

    gpu, cpu = rows(os.path.join(work, "gpu.log")), rows(os.path.join(work, "cpu.log"))
    print(f"{len(gpu)} and {len(cpu)} rows of 0.1 ps after {args.equil:g} ps "
          f"at 300 K and 1 atm on the GPU")
    print(f"{'':22s} {'GPU':>22s} {'CPU':>22s} {'difference':>12s}  g (GPU, CPU)")
    for label, column in (("potential energy", 2), ("temperature", 3),
                          ("total energy", 1)):
        a, b = estimate(gpu[:, column]), estimate(cpu[:, column])
        z = (a[0] - b[0]) / math.hypot(a[1], b[1])
        print(f"{'mean ' + label:22s} {a[0]:13.3f} ± {a[1]:6.3f} "
              f"{b[0]:13.3f} ± {b[1]:6.3f} {z:+8.2f} SE  {a[4]:.1f}, {b[4]:.1f}")
        if column == 2:
            z = (a[2] - b[2]) / math.hypot(a[3], b[3])
            print(f"{'sd ' + label:22s} {a[2]:13.3f} ± {a[3]:6.3f} "
                  f"{b[2]:13.3f} ± {b[3]:6.3f} {z:+8.2f} SE")
    for name in ("gpu", "cpu"):
        text = open(os.path.join(work, name + ".log")).read()
        rate = [l for l in text.splitlines() if "from step" in l]
        print(f"{name}: {rate[-1].split(', ')[-1] if rate else '?'}")


if __name__ == "__main__":
    main()
