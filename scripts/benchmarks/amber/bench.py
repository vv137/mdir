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
# rescaling. The neighbor lists reach 10 Å, a skin of 2 Å.
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


def write_mdir(name, system, target, steps=None, path="mdir.toml",
               skin=SKIN, neighbor_structure=None, barostat_work=None):
    npt = system["ensemble"] == "NPT"
    ensemble = (f"""ensemble    = "NPT"
temperature = {TEMPERATURE}
pressure    = 0.986923

[thermostat]
method        = "V-RESCALE"
time_constant = 1.0
interval      = 10

[barostat]
method        = "C-RESCALE"
time_constant = 2.0
""" + (f'work          = "{barostat_work}"\n' if barostat_work else "")
        if npt else f"""ensemble    = "NVE"
temperature = {TEMPERATURE}
""")
    steps = steps or system["steps"]
    text = f"""# {name}: from {system['directory']} of the Amber benchmark suite.
[input]
topology    = "system.parm7"
coordinates = "system.rst7"

[output]
energy_interval = {max(steps // 2, 1)}

[energy]
cutoff            = {CUTOFF}
pairlist_distance = {CUTOFF + skin}
electrostatics    = "PME"

[pme]
tolerance   = {tolerance(system)}
max_spacing = {GRID_SPACING}

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = {system['timestep']}
steps      = {steps}

[ensemble]
{ensemble}
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
""" + (f'neighbor_structure = "{neighbor_structure}"\n'
       if neighbor_structure else "")
    path = os.path.join(target, path)
    with open(path, "w") as file:
        file.write(text)
    return path


def smoke(args):
    """Runs each system for a few steps and checks that the run ends and
    that every number of the log is finite: the tier of short runs of large
    systems (docs/principles.md, Section 5)."""
    names = args.systems or list(SYSTEMS)
    failed = 0
    for name in names:
        system = SYSTEMS[name]
        target = os.path.join(args.work, name)
        control = write_mdir(name, system, target, steps=args.steps,
                             path="smoke.toml")
        done = subprocess.run([args.mdir, "run", control], cwd=target,
                              capture_output=True, text=True)
        rows = [line.split() for line in done.stdout.splitlines()
                if line.startswith("INFO:") and line.split()[1].isdigit()]
        finite = all(re.fullmatch(r"-?[0-9.]+", value)
                     for row in rows for value in row[1:])
        ok = done.returncode == 0 and len(rows) == 2 and finite
        failed += not ok
        print(f"{name}: {'ok' if ok else 'FAILED'}, {len(rows)} rows")
        if not ok:
            print(done.stdout[-2000:] + done.stderr[-2000:])
    sys.exit(1 if failed else 0)


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


def visible_bus_ids():
    """The PCI bus ids of the devices that a run can see."""
    listing = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    devices = {}
    for line in listing.splitlines():
        index, bus = [field.strip() for field in line.split(",")]
        devices[index] = bus
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return set(devices.values())
    return {devices[index] for index in visible.split(",") if index in devices}


def others_on(buses, own):
    """The processes other than `own` that compute on the devices `buses`."""
    listing = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    found = set()
    for line in listing.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3 and fields[0] in buses and int(fields[1]) != own:
            found.add((int(fields[1]), fields[2]))
    return found


def timed(command, cwd, out):
    """Runs `command` and watches its devices every two seconds; returns the
    processes of others that shared them, which make a rate meaningless."""
    buses = visible_bus_ids()
    shared = others_on(buses, own=-1)
    child = subprocess.Popen(command, cwd=cwd, stdout=out,
                             stderr=subprocess.STDOUT)
    while True:
        try:
            child.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            shared |= others_on(buses, own=child.pid)
    shared |= others_on(buses, own=child.pid)
    if shared:
        names = ", ".join(f"{name} ({pid})" for pid, name in sorted(shared))
        print(f"warning: the device was shared with {names}; "
              "the rate is not a measurement", file=sys.stderr)
    return child.returncode, shared


def record(args, engine, name, rate, log, shared=False):
    path = os.path.join(args.work, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    results.setdefault(engine, {})[name] = {"ns_per_day": rate, "log": log,
                                            "shared": shared}
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
            control = write_mdir(name, system, target, skin=args.skin,
                                 neighbor_structure=args.neighbor_structure,
                                 barostat_work=args.barostat_work)
            with open(log, "w") as out:
                _, shared = timed([args.mdir, "run", control], target, out)
            text = open(log).read()
            # The rate of the second half, past the set-up and the first
            # build, as GROMACS reports it with -resethway; that of the
            # whole run if the log has none.
            match = (re.search(r"from step [0-9]+ to step [0-9]+, [0-9.]+ "
                               r"ms per step, ([0-9.]+) ns per day", text)
                     or re.search(r"([0-9.]+) ns per day", text))
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
                    code, shared = timed(
                        [args.gmx, "mdrun", "-s", tpr, "-deffnm", "gromacs",
                         "-nb", "gpu", "-pme", "gpu", "-bonded", "gpu",
                         "-update", update, "-ntmpi", "1", "-ntomp",
                         str(args.threads), "-pin", "on", "-notunepme",
                         "-resethway", "-noconfout"], target, out)
                    if code == 0:
                        break
            mdrun = os.path.join(target, "gromacs.log")
            text = open(mdrun).read() if os.path.exists(mdrun) else ""
            match = re.search(r"Performance:\s+([0-9.]+)", text)
        rate = float(match.group(1)) if match else None
        print(f"{args.engine:8s} {name:15s} "
              f"{rate if rate is not None else 'failed; see ' + log}"
              f"{' (the device was shared)' if shared else ''}")
        record(args, args.engine, name, rate, log, bool(shared))


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
    p.add_argument("--skin", type=float, default=SKIN,
                   help="of the neighbor structures of MDIR (Å)")
    p.add_argument("--neighbor-structure", choices=["MATRIX", "GROUPS"],
                   help="of MDIR on the device; its default if absent")
    p.add_argument("--barostat-work",
                   choices=["TROTTER", "TROTTER_FIRST_ORDER", "EXACT",
                            "FIRST_ORDER"],
                   help="how MDIR's barostat counts a scaling; its default "
                        "if absent")
    p.add_argument("--gmx", default="gmx")
    p.add_argument("--threads", type=int, default=8,
                   help="OpenMP threads of GROMACS")
    commands.add_parser("report")
    p = commands.add_parser("smoke")
    p.add_argument("systems", nargs="*", metavar="SYSTEM")
    p.add_argument("--mdir", default="mdir")
    p.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report,
     "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
