#!/usr/bin/env python3
"""Compares the mean volume of systems of the Amber benchmark suite at
constant pressure across programs and barostats, as one model (D134).

    scripts/validation/suite/density.py --engines ENGINE,... [--mdir MDIR]
        [--pmemd PMEMD] [--openmm PYTHON] [--gmx GMX] [--suite DIR]
        [--steps N] [--no-dispersion-correction] [SYSTEM ...]

ENGINE is one of:
  mdir             MDIR with the settings of scripts/benchmarks/amber/bench.py:
                   stochastic velocity and cell rescaling (the pressure of the
                   virial), groups of 16 with a dual list;
  pmemd            pmemd.cuda with the input of the suite (mdin.GPU):
                   Berendsen's thermostat (tau 10 ps) and the Monte Carlo
                   barostat, which accepts a scaling by the change of the energy;
  pmemd-berendsen  the same with Berendsen's barostat (tau_p 2 ps), from the
                   pressure of the virial;
  openmm           OpenMM: the Langevin middle integrator (1/ps) and its Monte
                   Carlo barostat every 25 steps, PME of the same beta and grid,
                   from the topology that bench.py writes again in the
                   current format (system.parm7);
  gromacs          GROMACS with CUDA: stochastic velocity and cell rescaling,
                   from the topology that bench.py converts.

With --no-dispersion-correction, MDIR and OpenMM run without the correction
for the dispersion. Each run takes N steps of 2 fs from the files of the
suite; the outputs go into density-ENGINE* in the directory of each system of
the prepared suite. Prints the mean volume over the second half of each run
with the error of ten blocks, and the mean temperature. The device is the
one that CUDA_VISIBLE_DEVICES leaves visible.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "benchmarks", "amber"))
import bench  # noqa: E402

OPENMM = """
import sys
import numpy as np
import openmm as mm, openmm.app as app, openmm.unit as u
steps, alpha, nx, ny, nz, dispersion = sys.argv[1:7]
prm = app.AmberPrmtopFile("../system.parm7")
crd = app.AmberInpcrdFile("../system.rst7")
system = prm.createSystem(nonbondedMethod=app.PME,
                          nonbondedCutoff=0.8 * u.nanometer,
                          constraints=app.HBonds, rigidWater=True)
for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
        force.setPMEParameters(float(alpha), int(nx), int(ny), int(nz))
        force.setUseDispersionCorrection(dispersion == "1")
        force.setUseSwitchingFunction(False)
system.addForce(mm.MonteCarloBarostat(1.0 * u.bar, 300 * u.kelvin, 25))
integrator = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond,
                                         0.002 * u.picoseconds)
simulation = app.Simulation(prm.topology, system, integrator,
                            mm.Platform.getPlatformByName("CUDA"),
                            {"Precision": "mixed"})
simulation.context.setPositions(crd.positions)
simulation.context.setPeriodicBoxVectors(*crd.boxVectors)
simulation.context.setVelocities(crd.velocities)
with open("volumes.txt", "w") as out:
    for _ in range(int(steps) // 500):
        simulation.step(500)
        state = simulation.context.getState(getEnergy=True)
        volume = state.getPeriodicBoxVolume().value_in_unit(u.angstrom**3)
        kinetic = state.getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
        out.write(f"{volume} {kinetic}\\n")
"""


def mean_and_error(values):
    half = values[len(values) // 2:]
    blocks = np.array_split(half, 10)
    means = np.array([b.mean() for b in blocks])
    return half.mean(), means.std(ddof=1) / np.sqrt(len(means))


def suite_files(args, name):
    return os.path.join(args.suite, "Amber24_Benchmark_Suite", "PME",
                        bench.SYSTEMS[name]["directory"])


def run_mdir(args, name, target, tag):
    path = bench.write_mdir(name, bench.SYSTEMS[name], target,
                            steps=args.steps, path=f"density-{tag}.toml",
                            neighbor_structure="GROUPS", skin=3.0,
                            prune_skin=0.6, energy_interval=500)
    if args.no_dispersion_correction:
        text = open(path).read().replace(
            'electrostatics    = "PME"',
            'electrostatics    = "PME"\ndispersion_correction = "NONE"')
        open(path, "w").write(text)
    log = os.path.join(target, f"density-{tag}.log")
    with open(log, "w") as out:
        subprocess.run([os.path.abspath(args.mdir), "run",
                        os.path.basename(path)], cwd=target, stdout=out,
                       stderr=subprocess.STDOUT, check=True)
    rows = [line.split() for line in open(log)
            if line.startswith("INFO:") and line.split()[1].isdigit()]
    return (np.array([float(f[-1]) for f in rows[1:]]),
            np.array([float(f[6]) for f in rows[1:]]))


def run_pmemd(args, name, target, tag, berendsen):
    work = os.path.join(target, f"density-{tag}")
    os.makedirs(work, exist_ok=True)
    for f in ("prmtop", "inpcrd"):
        shutil.copy(os.path.join(suite_files(args, name), f), work)
    text = open(os.path.join(suite_files(args, name), "mdin.GPU")).read()
    text = re.sub(r"nstlim=\d+", f"nstlim={args.steps}", text)
    text = re.sub(r"ntpr=\d+", "ntpr=500", text)
    text = re.sub(r"ntwx=\d+", "ntwx=0", text)
    text = re.sub(r"ntwr=\d+", f"ntwr={args.steps}", text)
    if berendsen:
        text = text.replace("barostat=2", "barostat=1, taup=2.0")
    with open(os.path.join(work, "mdin"), "w") as f:
        f.write(text)
    subprocess.run([args.pmemd, "-O", "-i", "mdin", "-p", "prmtop", "-c",
                    "inpcrd", "-o", "mdout", "-r", "restrt", "-inf",
                    "mdinfo"], cwd=work, check=True)
    text = open(os.path.join(work, "mdout")).read()
    text = text.split("A V E R A G E S")[0]
    volume = re.findall(r"VOLUME\s+=\s+([\d.]+)", text)
    temperature = re.findall(r"TEMP\(K\)\s+=\s+([\d.]+)", text)
    return (np.array([float(v) for v in volume]),
            np.array([float(t) for t in temperature]))


def run_openmm(args, name, target, tag):
    work = os.path.join(target, f"density-{tag}")
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "run.py"), "w") as f:
        f.write(OPENMM)
    system = bench.SYSTEMS[name]
    alpha = bench.ewald_beta(system) * 10.0   # 1/nm
    subprocess.run([args.openmm, "run.py", str(args.steps), repr(alpha)] +
                   [str(n) for n in system["grid"]] +
                   ["0" if args.no_dispersion_correction else "1"],
                   cwd=work, check=True)
    data = np.loadtxt(os.path.join(work, "volumes.txt"))
    return data[:, 0], None


def run_gromacs(args, name, target, tag):
    work = os.path.join(target, f"density-{tag}")
    os.makedirs(work, exist_ok=True)
    mdp = open(bench.write_mdp(name, bench.SYSTEMS[name], work)).read()
    mdp = re.sub(r"nsteps\s*=.*", f"nsteps           = {args.steps}", mdp)
    mdp = re.sub(r"nstenergy\s*=.*", "nstenergy        = 500", mdp)
    with open(os.path.join(work, "density.mdp"), "w") as f:
        f.write(mdp)
    subprocess.run([args.gmx, "grompp", "-f", "density.mdp", "-c",
                    "../system.gro", "-p", "../system.top", "-o",
                    "density.tpr", "-maxwarn", "5"], cwd=work, check=True,
                   capture_output=True)
    subprocess.run([args.gmx, "mdrun", "-s", "density.tpr", "-deffnm",
                    "density", "-nb", "gpu", "-pme", "gpu", "-bonded", "gpu",
                    "-update", "gpu", "-ntmpi", "1", "-ntomp", "8"], cwd=work,
                   check=True, capture_output=True)
    subprocess.run([args.gmx, "energy", "-f", "density.edr", "-o",
                    "density.xvg"], cwd=work, input=b"Volume\nTemperature\n\n",
                   check=True, capture_output=True)
    # gmx energy writes the terms in the order of the energy file, not in
    # that of the request: the legends name the columns.
    lines = open(os.path.join(work, "density.xvg")).read().split("\n")
    legends = re.findall(r'^@ s(\d+) legend "(.*)"', "\n".join(lines), re.M)
    column = {name: int(index) + 1 for index, name in legends}
    rows = [line.split() for line in lines
            if line and not line.startswith(("#", "@"))]
    return (np.array([float(r[column["Volume"]]) * 1000.0 for r in rows]),
            np.array([float(r[column["Temperature"]]) for r in rows]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", default="mdir,pmemd")
    parser.add_argument("--mdir")
    parser.add_argument("--pmemd")
    parser.add_argument("--openmm")
    parser.add_argument("--gmx")
    parser.add_argument("--suite", default=os.environ.get("MDIR_BENCH_DIR"))
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--no-dispersion-correction", action="store_true")
    parser.add_argument("systems", nargs="*",
                        default=["jac_npt", "factorix_npt"])
    args = parser.parse_args()
    suffix = "-nolrc" if args.no_dispersion_correction else ""
    for name in args.systems:
        target = os.path.join(args.suite, name)
        for engine in args.engines.split(","):
            tag = engine + suffix
            if engine == "mdir":
                volume, temperature = run_mdir(args, name, target, tag)
            elif engine in ("pmemd", "pmemd-berendsen"):
                volume, temperature = run_pmemd(args, name, target, tag,
                                                engine == "pmemd-berendsen")
            elif engine == "openmm":
                volume, temperature = run_openmm(args, name, target, tag)
            elif engine == "gromacs":
                volume, temperature = run_gromacs(args, name, target, tag)
            else:
                sys.exit(f"unknown engine {engine}")
            mean, error = mean_and_error(volume)
            t = (f", T {temperature[len(temperature) // 2:].mean():.2f} K"
                 if temperature is not None else "")
            print(f"{name} {tag}: V {mean:.1f} ± {error:.1f} Å³{t}",
                  flush=True)


if __name__ == "__main__":
    main()
