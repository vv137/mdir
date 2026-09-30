#!/usr/bin/env python3
"""Runs the explicit-solvent systems of the Amber GPU benchmark suite with
MDIR and with GROMACS, under the settings of their Amber inputs.

    bench.py prepare [--work DIR] [--parmed-python PYTHON]
    bench.py run mdir [SYSTEM ...] [--work DIR] [--mdir MDIR]
    bench.py run gromacs [SYSTEM ...] [--work DIR] [--gmx GMX]
    bench.py report [--work DIR]

The inputs are not part of MDIR: `prepare` downloads the suite from
ambermd.org into the work directory, writes each topology again with
ParmEd (two of them are in the format before Amber 7, which MDIR does not
read), and converts it for GROMACS. See README.md."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import urllib.request

SUITE_URL = "https://ambermd.org/Amber24_Benchmark_Suite.tar.gz"
# The archive as it was downloaded on 2026-09-30.
SUITE_SHA256 = "0c23da4d8b72ca76e764a82c4bfadd983feeeaa5a3931b0d4f7f93bac273a148"

# The systems, from the directories of the suite. `steps` is how many a
# timing takes; `pmemd` is the rate of pmemd.cuda on an RTX 3090 that
# ambermd.org/GPUPerformance.php reports, in ns/day, where it reports one.
SYSTEMS = {
    "jac_nve": dict(directory="JAC_production_NVE", ensemble="NVE",
                    timestep=0.002, steps=10000, pmemd=632.19),
    "jac_nve_4fs": dict(directory="JAC_production_NVE_4fs", ensemble="NVE",
                        timestep=0.004, steps=10000, pmemd=1196.50),
    "jac_npt": dict(directory="JAC_production_NPT", ensemble="NPT",
                    timestep=0.002, steps=10000),
    "jac_npt_4fs": dict(directory="JAC_production_NPT_4fs", ensemble="NPT",
                        timestep=0.004, steps=10000),
    "factorix_nve": dict(directory="FactorIX_production_NVE", ensemble="NVE",
                         timestep=0.002, steps=4000, pmemd=264.78),
    "factorix_npt": dict(directory="FactorIX_production_NPT", ensemble="NPT",
                         timestep=0.002, steps=4000),
    "cellulose_nve": dict(directory="Cellulose_production_NVE",
                          ensemble="NVE", timestep=0.002, steps=1000,
                          pmemd=63.23),
    "cellulose_npt": dict(directory="Cellulose_production_NPT",
                          ensemble="NPT", timestep=0.002, steps=1000),
    "stmv_npt_4fs": dict(directory="STMV_production_NPT_4fs", ensemble="NPT",
                         timestep=0.004, steps=500, pmemd=38.65),
}

# The settings of the Amber inputs of the suite (mdin.GPU): a cutoff of
# 8 Å, SHAKE of the bonds of hydrogen and rigid water, particle mesh Ewald
# with the tolerance of the direct sum 1e-6 for NVE and the default 1e-5
# for NPT, the correction for the dispersion; NPT at 300 K and 1 bar. The
# couplings differ: Amber uses the thermostat of Berendsen and a Monte
# Carlo barostat, MDIR and GROMACS stochastic velocity rescaling and cell
# rescaling. The neighbor lists have the skin of pmemd, 2 Å.
CUTOFF = 8.0
SKIN = 2.0
GRID_SPACING = 1.0
TEMPERATURE = 300.0


def tolerance(system):
    return 1e-6 if system["ensemble"] == "NVE" else 1e-5


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    os.makedirs(args.work, exist_ok=True)
    archive = os.path.join(args.work, os.path.basename(SUITE_URL))
    if not os.path.exists(archive):
        print(f"downloading {SUITE_URL}")
        urllib.request.urlretrieve(SUITE_URL, archive)
    digest = sha256(archive)
    if digest != SUITE_SHA256:
        print(f"warning: the archive has changed since this script was "
              f"written (sha256 {digest})")
    suite = os.path.join(args.work, "Amber24_Benchmark_Suite")
    if not os.path.isdir(suite):
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers()
                       if not os.path.basename(m.name).startswith("._")]
            tar.extractall(args.work, members=members, filter="data")
    for name, system in SYSTEMS.items():
        source = os.path.join(suite, "PME", system["directory"])
        target = os.path.join(args.work, name)
        os.makedirs(target, exist_ok=True)
        if os.path.exists(os.path.join(target, "system.top")):
            continue
        print(f"converting {name}")
        # ParmEd writes the topology again in the current Amber format, and,
        # from that, for GROMACS, with the coordinates, the velocities, and
        # the cell of the restart file. (It cannot write coordinates of
        # GROMACS from a topology of the old format that it has read.)
        parm = os.path.join(target, "system.parm7")
        rst = os.path.join(target, "system.rst7")
        # ParmEd takes the elements of a topology without atomic numbers
        # from the masses, and a hydrogen of 3.024 amu, whose mass has been
        # repartitioned, for helium; the topology is then made again from
        # the corrected atoms, so that the bonds of hydrogen are those
        # that SHAKE constrains.
        script = f"""
import parmed
p = parmed.load_file({os.path.join(source, "prmtop")!r},
                     {os.path.join(source, "inpcrd")!r})
for atom in p.atoms:
    if atom.name.startswith("H") and 0.0 < atom.mass < 4.1:
        atom.atomic_number = 1
p = parmed.amber.AmberParm.from_structure(p)
p.save({parm!r}, format="amber", overwrite=True)
p.save({rst!r}, format="rst7", overwrite=True)
q = parmed.load_file({parm!r}, {rst!r})
q.save({os.path.join(target, "system.gro")!r}, overwrite=True)
q.save({os.path.join(target, "system.top")!r}, format="gromacs",
       overwrite=True)
"""
        subprocess.run([args.parmed_python, "-c", script], check=True)


def write_mdir(name, system, target):
    npt = system["ensemble"] == "NPT"
    ensemble = (f"""ensemble    = "NPT"
thermostat  = "BUSSI"
temperature = {TEMPERATURE}
tau_t       = 1.0
barostat    = "BERNETTI-BUSSI"
pressure    = 0.986923
tau_p       = 2.0
""" if npt else f"""ensemble    = "NVE"
temperature = {TEMPERATURE}
""")
    steps = system["steps"]
    text = f"""# {name}: from {system['directory']} of the Amber benchmark suite.
[input]
prmtopfile = "system.parm7"
ambcrdfile = "system.rst7"

[energy]
cutoffdist          = {CUTOFF}
pairlistdist        = {CUTOFF + SKIN}
electrostatic       = "PME"
pme_alpha_tol       = {tolerance(system)}
pme_max_spacing     = {GRID_SPACING}

[dynamics]
integrator    = "VVER"
timestep      = {system['timestep']}
nsteps        = {steps}
eneout_period = {steps}
{"thermostat_period = 10" if npt else ""}

[ensemble]
{ensemble}
[constraints]
rigid_bond = true
fast_water = true

[boundary]
type = "PBC"

[execution]
target    = "gpu"
precision = "mixed"
"""
    path = os.path.join(target, "mdir.toml")
    with open(path, "w") as file:
        file.write(text)
    return path


def write_mdp(name, system, target):
    npt = system["ensemble"] == "NPT"
    couplings = f"""tcoupl           = v-rescale
tc-grps          = System
tau-t            = 1.0
ref-t            = {TEMPERATURE}
pcoupl           = c-rescale
tau-p            = 2.0
ref-p            = 1.0
compressibility  = 4.5e-5
""" if npt else "tcoupl           = no\npcoupl           = no\n"
    text = f"""; {name}: from {system['directory']} of the Amber benchmark suite.
integrator       = md
dt               = {system['timestep']}
nsteps           = {system['steps']}
continuation     = yes
gen-vel          = no
cutoff-scheme    = Verlet
rcoulomb         = {CUTOFF / 10}
rvdw             = {CUTOFF / 10}
coulombtype      = PME
ewald-rtol       = {tolerance(system)}
fourierspacing   = {GRID_SPACING / 10}
pme-order        = 4
vdwtype          = Cut-off
DispCorr         = EnerPres
constraints      = h-bonds
nstcalcenergy    = 100
nstenergy        = 0
nstlog           = 0
{couplings}"""
    path = os.path.join(target, "gromacs.mdp")
    with open(path, "w") as file:
        file.write(text)
    return path


def record(args, engine, name, rate, log):
    path = os.path.join(args.work, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    results.setdefault(engine, {})[name] = {"ns_per_day": rate, "log": log}
    with open(path, "w") as file:
        json.dump(results, file, indent=2)


def run(args):
    names = args.systems or list(SYSTEMS)
    unknown = [name for name in names if name not in SYSTEMS]
    if unknown:
        sys.exit(f"unknown systems: {', '.join(unknown)}")
    for name in names:
        system = SYSTEMS[name]
        target = os.path.join(args.work, name)
        if not os.path.exists(os.path.join(target, "system.parm7")):
            sys.exit(f"{name} is not prepared; run 'bench.py prepare'")
        # The output of the commands; GROMACS writes its own log,
        # gromacs.log, where its rate is.
        log = os.path.join(target, f"{args.engine}.out")
        if args.engine == "mdir":
            control = write_mdir(name, system, target)
            with open(log, "w") as out:
                subprocess.run([args.mdir, "run", control], cwd=target,
                               stdout=out, stderr=subprocess.STDOUT)
            text = open(log).read()
            match = re.search(r"([0-9.]+) ns per day", text)
        else:
            mdp = write_mdp(name, system, target)
            tpr = os.path.join(target, "gromacs.tpr")
            with open(log, "w") as out:
                subprocess.run([args.gmx, "grompp", "-f", mdp, "-c",
                                "system.gro", "-p", "system.top", "-o", tpr,
                                "-maxwarn", "2"], cwd=target, stdout=out,
                               stderr=subprocess.STDOUT)
                # The update runs on the GPU where GROMACS can put it there,
                # and on the host otherwise (constraints in triangles, for
                # one).
                for update in ("gpu", "cpu"):
                    done = subprocess.run(
                        [args.gmx, "mdrun", "-s", tpr, "-deffnm", "gromacs",
                         "-nb", "gpu", "-pme", "gpu", "-bonded", "gpu",
                         "-update", update, "-ntmpi", "1", "-ntomp",
                         str(args.threads), "-pin", "on", "-notunepme",
                         "-resethway", "-noconfout"],
                        cwd=target, stdout=out, stderr=subprocess.STDOUT)
                    if done.returncode == 0:
                        break
            mdrun = os.path.join(target, "gromacs.log")
            text = open(mdrun).read() if os.path.exists(mdrun) else ""
            match = re.search(r"Performance:\s+([0-9.]+)", text)
        rate = float(match.group(1)) if match else None
        print(f"{args.engine:8s} {name:15s} "
              f"{rate if rate is not None else 'failed; see ' + log}")
        record(args, args.engine, name, rate, log)


def report(args):
    path = os.path.join(args.work, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    engines = [e for e in ("mdir", "gromacs") if e in results]
    print("| System | Atoms | " + " | ".join(engines) +
          " | pmemd.cuda (published) |")
    print("|---|---" + "|---" * len(engines) + "|---|")
    for name, system in SYSTEMS.items():
        cells = []
        for engine in engines:
            rate = results[engine].get(name, {}).get("ns_per_day")
            cells.append(f"{rate:.1f}" if rate else "")
        pmemd = system.get("pmemd")
        atoms = ""
        parm = os.path.join(args.work, name, "system.parm7")
        if os.path.exists(parm):
            with open(parm) as file:
                for line in file:
                    if line.startswith("%FLAG POINTERS"):
                        next(file)
                        atoms = next(file).split()[0]
                        break
        print(f"| {name} | {atoms} | " + " | ".join(cells) +
              f" | {pmemd if pmemd else ''} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", default=os.environ.get(
        "MDIR_BENCH_DIR", os.path.expanduser("~/mdir-benchmarks")),
        help="where the suite, the converted inputs, and the results go")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--parmed-python", default="python3",
                   help="a Python that has ParmEd (AmberTools)")
    p = commands.add_parser("run")
    p.add_argument("engine", choices=["mdir", "gromacs"])
    p.add_argument("systems", nargs="*", metavar="SYSTEM",
                   help="of " + ", ".join(SYSTEMS) + "; all by default")
    p.add_argument("--mdir", default="mdir")
    p.add_argument("--gmx", default="gmx")
    p.add_argument("--threads", type=int, default=8,
                   help="OpenMP threads of GROMACS")
    commands.add_parser("report")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
