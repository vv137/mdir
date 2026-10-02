#!/usr/bin/env python3
"""Runs a dataset of MDBench (https://mdbench.ace-net.ca/mdbench/datasets/)
with pmemd.cuda, GROMACS, OpenMM, and MDIR on one GPU, each from the input
that MDBench gives for it, and reports the rates and what the runs sampled.

    scripts/benchmarks/mdbench/run.py DATA WORK --mdir MDIR [--steps 30000]
        [--engines pmemd,gromacs,openmm,mdir] [--pmemd PMEMD] [--gmx GMX]
        [--openmm-python PYTHON]

DATA holds benchmark_<name>_{pmemd,gromacs,openmm} as MDBench's tarballs
unpack them. Each engine runs STEPS steps of its MDBench input in WORK/<engine>,
with the step count changed and nothing else: pmemd.cuda with
pmemd_prod.in (Langevin at 300 K, Berendsen's barostat, grid 128^3),
GROMACS with gromacs_production.mdp (V-rescale, Parrinello-Rahman, grid
144^3; -nb gpu -pme gpu -update gpu -bonded cpu as submit.cuda.sh, over
all steps with its tuning of PME, as MDBench times it), OpenMM with openmm_input.py (Langevin, a Monte
Carlo barostat every 25 steps, ewaldErrorTolerance 4e-4, mixed). All take
1 fs, a cutoff of 8 Å, and the bonds of hydrogen constrained. MDIR runs
the Amber files of the dataset with the nearest of its own couplings:
stochastic velocity and cell rescaling every 25 steps, PME with
erfc(beta r_c) = 1e-5, groups with a dual list (11 / 8.6 Å), mixed
precision, timed over the second half, past its compilation. The rates
are each program's own; the temperature and the density are means over
the second half of each run.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

MDIR_CONTROL = """[input]
topology    = "prmtop.parm7"
coordinates = "restart.rst7"

[output]
energy_interval = {interval}

[energy]
cutoff            = 8.0
pairlist_distance = 11.0
pruned_distance   = 8.6
electrostatics    = "PME"

[pme]
tolerance   = 1e-05
max_spacing = 1.0

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.001
steps      = {steps}

[ensemble]
ensemble    = "NPT"
temperature = 300.0
pressure    = 0.986923

[thermostat]
method        = "V-RESCALE"
time_constant = 1.0
interval      = 25

[barostat]
method        = "C-RESCALE"
time_constant = 2.0
interval      = 25

[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
neighbor_structure = "GROUPS"
"""


def run(command, cwd, log, env=None):
    with open(os.path.join(cwd, log), "w") as out:
        result = subprocess.run(command, cwd=cwd, stdout=out,
                                stderr=subprocess.STDOUT,
                                env=dict(os.environ, **(env or {})))
    if result.returncode != 0:
        sys.exit(f"{' '.join(command)} failed; see {os.path.join(cwd, log)}")


def stage(source, work):
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(source, work)


def density(volume, work):
    """g/cm^3 from a volume in Å^3 and the masses of the topology."""
    masses, reading = [], False
    for line in open(os.path.join(work, "prmtop.parm7")):
        if line.startswith("%FLAG"):
            reading = line.split()[1] == "MASS"
        elif reading and not line.startswith("%"):
            masses += [float(v) for v in line.split()]
    return sum(masses) / 6.02214076e23 / (volume * 1e-24)


def pmemd(args, data, work):
    stage(os.path.join(data, "pmemd"), work)
    path = os.path.join(work, "pmemd_prod.in")
    text = re.sub(r"nstlim=\d+", f"nstlim={args.steps}", open(path).read())
    open(path, "w").write(text)
    run([args.pmemd, "-O", "-i", "pmemd_prod.in", "-p", "prmtop.parm7", "-c",
         "restart.rst7", "-o", "mdout", "-r", "restrt", "-inf", "mdinfo"],
        work, "pmemd.out")
    text = open(os.path.join(work, "mdout")).read()
    rate = re.findall(r"ns/day =\s+([0-9.]+)", text)
    averages = text.split("A V E R A G E S")[1] if "A V E R A G E S" in text \
        else ""
    temperature = re.search(r"TEMP\(K\) =\s+([0-9.]+)", averages)
    density = re.search(r"Density\s+=\s+([0-9.]+)", averages)
    return {"rate": float(rate[-1]) if rate else None,
            "temperature": float(temperature.group(1)) if temperature else None,
            "density": float(density.group(1)) if density else None}


def gromacs(args, data, work):
    stage(os.path.join(data, "gromacs"), work)
    path = os.path.join(work, "gromacs_production.mdp")
    text = re.sub(r"(?m)^nsteps\s*=.*$", f"nsteps = {args.steps}",
                  open(path).read())
    text = re.sub(r"(?m)^nstxout-compressed\s*=.*$", "nstxout-compressed = 0",
                  text)
    open(path, "w").write(text)
    run([args.gmx, "grompp", "-p", "topol.top", "-c", "restart.gro", "-t",
         "restart.trr", "-f", "gromacs_production.mdp", "-o", "topol.tpr",
         "-maxwarn", "2"], work, "grompp.log")
    run([args.gmx, "mdrun", "-s", "topol.tpr", "-ntmpi", "1", "-nb", "gpu",
         "-pme", "gpu", "-update", "gpu", "-bonded", "cpu"],
        work, "mdrun.out", {"GMX_MAXBACKUP": "-1"})
    log = open(os.path.join(work, "md.log")).read()
    rate = re.search(r"Performance:\s+([0-9.]+)", log)
    with open(os.path.join(work, "energy.log"), "w") as out:
        subprocess.run([args.gmx, "energy", "-f", "ener.edr", "-o",
                        "energy.xvg"], cwd=work, stdout=out,
                       stderr=subprocess.STDOUT, input="Temperature\nDensity\n\n",
                       text=True)
    rows = [[float(v) for v in line.split()]
            for line in open(os.path.join(work, "energy.xvg"))
            if not line.startswith(("#", "@"))]
    half = rows[len(rows) // 2:]
    return {"rate": float(rate.group(1)) if rate else None,
            "temperature": sum(r[1] for r in half) / len(half),
            "density": sum(r[2] for r in half) / len(half) / 1000.0}


def openmm(args, data, work):
    stage(os.path.join(data, "openmm"), work)
    path = os.path.join(work, "openmm_input.py")
    text = re.sub(r"(?m)^nsteps=\d+", f"nsteps={args.steps}",
                  open(path).read())
    open(path, "w").write(text)
    run([args.openmm_python, "openmm_input.py"], work, "openmm.out")
    out = open(os.path.join(work, "openmm.out")).read()
    rate = re.search(r"Benchmark time:\s+([0-9.]+)", out)
    rows = [line.split(",") for line in out.splitlines()
            if re.match(r"^\d+,", line)]
    # ParmEd's reporter: step, potential, kinetic, total, temperature,
    # volume (Å^3).
    half = rows[len(rows) // 2:]
    return {"rate": float(rate.group(1)) if rate else None,
            "temperature": sum(float(r[4]) for r in half) / len(half),
            "density": density(sum(float(r[5]) for r in half) / len(half),
                               work)}


def mdir(args, data, work):
    stage(os.path.join(data, "pmemd"), work)
    with open(os.path.join(work, "mdir.toml"), "w") as file:
        file.write(MDIR_CONTROL.format(steps=args.steps,
                                       interval=1000 if args.steps % 1000 == 0
                                       else args.steps // 10))
    run([args.mdir, "run", "mdir.toml"], work, "mdir.log")
    log = open(os.path.join(work, "mdir.log")).read()
    rate = re.search(r"([0-9.]+) ns per day", log)
    rows = [l.split() for l in log.splitlines()
            if l.startswith("INFO:") and l.split()[1].isdigit()]
    half = rows[len(rows) // 2:]
    return {"rate": float(rate.group(1)) if rate else None,
            "temperature": sum(float(r[6]) for r in half) / len(half),
            "density": density(sum(float(r[-1]) for r in half) / len(half),
                               work)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("work")
    parser.add_argument("--mdir", required=True)
    parser.add_argument("--pmemd", default="pmemd.cuda")
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--openmm-python", default="python3")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--engines", default="pmemd,gromacs,openmm,mdir")
    args = parser.parse_args()
    data = os.path.abspath(args.data)
    name = os.path.basename(data.rstrip("/"))
    # The directories of the tarballs, benchmark_<name>_<engine>.
    links = os.path.join(os.path.abspath(args.work), "inputs")
    os.makedirs(links, exist_ok=True)
    for entry in os.listdir(data):
        m = re.match(r"benchmark_.+_(pmemd|gromacs|openmm)$", entry)
        if m and os.path.isdir(os.path.join(data, entry)):
            target = os.path.join(links, m.group(1))
            if not os.path.exists(target):
                os.symlink(os.path.join(data, entry), target)
    engines = {"pmemd": pmemd, "gromacs": gromacs, "openmm": openmm,
               "mdir": mdir}
    with open(os.path.join(links, "pmemd", "prmtop.parm7")) as file:
        lines = file.read().split("%FLAG POINTERS")[1].splitlines()
    atoms = int(lines[2].split()[0])
    print(f"{name}: {atoms:,} atoms", flush=True)
    for engine in args.engines.split(","):
        result = engines[engine](args, links,
                                 os.path.join(os.path.abspath(args.work),
                                              engine))
        print(f"{name} ({atoms:,} atoms) {engine:8s} " + " ".join(
            f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
            for k, v in result.items()), flush=True)


if __name__ == "__main__":
    main()
