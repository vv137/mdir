#!/usr/bin/env python3
"""Runs the explicit-solvent systems of the Amber GPU benchmark suite with
MDIR and with GROMACS, under the settings of their Amber inputs.

    bench.py prepare [--work DIR] [--parmed-python PYTHON]
    bench.py run mdir [SYSTEM ...] [--work DIR] [--mdir MDIR]
                 [--skin S|AUTO] [--prune-skin S|AUTO]
                 [--suite-protocol [--energies-only]] [--record FILE]
    bench.py run pmemd [SYSTEM ...] [--work DIR] [--pmemd PMEMD]
                 [--suite-protocol [--energies-only]] [--record FILE]
    bench.py run gromacs [SYSTEM ...] [--work DIR] [--gmx GMX]
    bench.py report [--work DIR]

The inputs are not part of MDIR: `prepare` downloads the suite from
ambermd.org into the work directory and converts each system for GROMACS
with ParmEd. MDIR and pmemd read the files of the suite as they are. See
README.md.

`--suite-protocol` runs MDIR and pmemd under one protocol, that of each
system's mdin.GPU (D[suite-symmetric]): `nstlim` steps; the energies every
`ntpr` steps, a trajectory every `ntwx` steps, and a restart every `ntwr`
steps; and the rate over one window, from the first report (step `ntpr`)
to the last (step `nstlim`). For pmemd that is its "Average timings for
last N steps"; for MDIR it is the clock of this script at the rows of
those two steps in MDIR's log, which MDIR prints once the device has
finished the step. `--energies-only` keeps the energies and drops the
trajectory and the restarts (pmemd still writes its restart at the end).
Without `--suite-protocol` MDIR runs the short runs of `steps` below, as
before. Every run records the wall time of its process."""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

SUITE_URL = "https://ambermd.org/Amber24_Benchmark_Suite.tar.gz"
# The archive as it was downloaded on 2026-09-30.
SUITE_SHA256 = "0c23da4d8b72ca76e764a82c4bfadd983feeeaa5a3931b0d4f7f93bac273a148"

# The systems, from the directories of the suite. `steps` is how many a
# timing takes; `pmemd` is the rate of pmemd.cuda on an RTX 3090 that
# ambermd.org/GPUPerformance.php reports, in ns/day, where it reports one.
SYSTEMS = {
    "jac_nve": dict(directory="JAC_production_NVE", ensemble="NVE",
                    timestep=0.002, steps=10000, pmemd=632.19,
                    grid=(64, 64, 64)),
    "jac_nve_4fs": dict(directory="JAC_production_NVE_4fs", ensemble="NVE",
                        timestep=0.004, steps=10000, pmemd=1196.50,
                        grid=(64, 64, 64)),
    "jac_npt": dict(directory="JAC_production_NPT", ensemble="NPT",
                    timestep=0.002, steps=10000, grid=(64, 64, 64)),
    "jac_npt_4fs": dict(directory="JAC_production_NPT_4fs", ensemble="NPT",
                        timestep=0.004, steps=10000, grid=(64, 64, 64)),
    "factorix_nve": dict(directory="FactorIX_production_NVE", ensemble="NVE",
                         timestep=0.002, steps=4000, pmemd=264.78,
                         grid=(144, 84, 80)),
    "factorix_npt": dict(directory="FactorIX_production_NPT", ensemble="NPT",
                         timestep=0.002, steps=4000, grid=(144, 84, 80)),
    "cellulose_nve": dict(directory="Cellulose_production_NVE",
                          ensemble="NVE", timestep=0.002, steps=1000,
                          pmemd=63.23, grid=(256, 128, 128)),
    "cellulose_npt": dict(directory="Cellulose_production_NPT",
                          ensemble="NPT", timestep=0.002, steps=1000,
                          grid=(256, 128, 128)),
    "stmv_npt_4fs": dict(directory="STMV_production_NPT_4fs", ensemble="NPT",
                         timestep=0.004, steps=500, pmemd=38.65,
                         grid=(224, 224, 240)),
}

# The settings of the Amber inputs of the suite (mdin.GPU): a cutoff of
# 8 Å, SHAKE of the bonds of hydrogen and rigid water, particle mesh Ewald
# with the tolerance of the direct sum 1e-6 for NVE and the default 1e-5
# for NPT, the correction for the dispersion; NPT at 300 K and 1 bar. All
# three programs take the beta of pmemd, erfc(beta r_c) / r_c = dsum_tol,
# and the grid of pmemd.cuda (`grid` of SYSTEMS, from its output), and MDIR
# the influence function of Amber ("OPTIMAL"), so that they compute one
# model (D134). The couplings differ: Amber uses the thermostat of Berendsen and a Monte
# Carlo barostat, MDIR and GROMACS stochastic velocity rescaling and cell
# rescaling. The neighbor lists reach 10 Å, a skin of 2 Å, unless `run`
# is given other skins or "AUTO", which leaves them to MDIR (D281).
CUTOFF = 8.0
SKIN = 2.0
TEMPERATURE = 300.0
# The steps between the actions of the thermostat and of the barostat,
# which act together: those that GROMACS takes for the pressure on these
# inputs (nstpcouple 25 with nstcalcenergy 100, tau-p 2 ps); Amber's Monte
# Carlo barostat acts every 100 steps.
COUPLING_INTERVAL = 25
# A skin that MDIR chooses (D281).
AUTO = "AUTO"


def skin_value(text):
    """A skin in Å, or AUTO."""
    return AUTO if text.upper() == AUTO else float(text)


def tolerance(system):
    """dsum_tol of the Amber input: erfc(beta r_c) / r_c, r_c in Å."""
    return 1e-6 if system["ensemble"] == "NVE" else 1e-5


def ewald_beta(system):
    """The beta of pmemd, in 1/Å: erfc(beta r_c) / r_c = dsum_tol."""
    low, high = 0.0, 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if math.erfc(middle * CUTOFF) / CUTOFF > tolerance(system):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def suite_directory(work, system):
    """The directory of a system in the suite as it was downloaded."""
    return os.path.join(work, "Amber24_Benchmark_Suite", "PME",
                        system["directory"])


def read_mdin(work, system):
    """`nstlim`, `ntpr`, `ntwx`, and `ntwr` of the mdin.GPU of a system."""
    text = open(os.path.join(suite_directory(work, system), "mdin.GPU")).read()
    found = {}
    for key in ("nstlim", "ntpr", "ntwx", "ntwr"):
        match = re.search(rf"\b{key}\s*=\s*([0-9]+)", text)
        if not match:
            sys.exit(f"mdin.GPU of {system['directory']} has no {key}")
        found[key] = int(match.group(1))
    return found


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
        # For GROMACS: ParmEd writes the topology again in the current Amber
        # format, and, from that, for GROMACS, with the coordinates, the
        # velocities, and the cell of the restart file. (It cannot write
        # coordinates of GROMACS from a topology of the old format that it
        # has read.)
        parm = os.path.join(target, "system.parm7")
        rst = os.path.join(target, "system.rst7")
        # ParmEd takes the elements of a topology without atomic numbers
        # from the masses, and a hydrogen of 3.024 amu, whose mass has been
        # repartitioned, for helium; the topology is then made again from
        # the corrected atoms, so that the bonds of hydrogen are those
        # that SHAKE constrains.
        #
        # A topology in the format before Amber 7 (Factor IX) gives a
        # dihedral of several terms with one entry: a negative periodicity
        # of its type continues to the next type. ParmEd keeps the sign but
        # not the order of the types when it writes them again, after which
        # pmemd takes wrong terms (11,203 kcal/mol for the dihedrals of
        # Factor IX instead of 2,259) and GROMACS only the first term
        # (1,339). The terms are made explicit first, as LEaP writes them
        # now: each its own entry, without a 1-4 pair.
        script = f"""
import parmed
from parmed.topologyobjects import Dihedral
p = parmed.load_file({os.path.join(source, "prmtop")!r},
                     {os.path.join(source, "inpcrd")!r})
types = list(p.dihedral_types)
terms = []
for d in p.dihedrals:
    t = d.type
    while t.per < 0:
        t = types[t.idx + 1]
        terms.append(Dihedral(d.atom1, d.atom2, d.atom3, d.atom4,
                              improper=d.improper, ignore_end=True, type=t))
p.dihedrals.extend(terms)
for t in types:
    t.per = abs(t.per)
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
               skin=SKIN, neighbor_structure=None, barostat_work=None,
               prune_skin=None, energy_interval=None, analytic_bonds=False,
               trajectory_interval=None, checkpoint_interval=None):
    npt = system["ensemble"] == "NPT"
    ensemble = (f"""ensemble    = "NPT"
temperature = {TEMPERATURE}
pressure    = 0.986923

[thermostat]
method        = "V-RESCALE"
time_constant = 1.0
interval      = {COUPLING_INTERVAL}

[barostat]
method        = "C-RESCALE"
time_constant = 2.0
""" + (f'work          = "{barostat_work}"\n' if barostat_work else "")
        if npt else f"""ensemble    = "NVE"
temperature = {TEMPERATURE}
""")
    steps = steps or system["steps"]
    source = os.path.relpath(
        os.path.join(os.path.dirname(target), "Amber24_Benchmark_Suite",
                     "PME", system["directory"]), target)
    # The reaches: a skin of "AUTO" writes the key as "AUTO", which is what
    # a key not given means (D281), so that the log states the reaches the
    # run took and their source; a skin of AUTO with no inner skin given
    # leaves the inner list to the run as well.
    if skin == AUTO:
        reaches = 'pairlist_distance = "AUTO"\n'
        if prune_skin is None:
            reaches += 'pruned_distance   = "AUTO"\n'
    else:
        reaches = f"pairlist_distance = {CUTOFF + skin}\n"
    if prune_skin == AUTO:
        reaches += 'pruned_distance   = "AUTO"\n'
    elif prune_skin:
        reaches += (f"pruned_distance   = "
                    f"{CUTOFF + prune_skin * system['timestep'] / 0.002:.2f}\n")
    # The output of a protocol of the suite: a trajectory in DCD, f32
    # positions and the cell of each frame without compression, the format
    # of MDIR closest in cost to pmemd's NetCDF trajectory (ioutfm=1), and
    # checkpoints in place of pmemd's restarts.
    files = ""
    if trajectory_interval:
        files += ('trajectory          = "mdir.dcd"\n'
                  f"trajectory_interval = {trajectory_interval}\n")
    if checkpoint_interval:
        files += ('checkpoint          = "mdir.h5"\n'
                  f"checkpoint_interval = {checkpoint_interval}\n")
    text = f"""# {name}: from {system['directory']} of the Amber benchmark suite.
[input]
format      = "AMBER"
topology    = "{source}/prmtop"
coordinates = "{source}/inpcrd"

[output]
energy_interval = {energy_interval or max(steps // 2, 1)}
{files}
[energy]
cutoff            = {CUTOFF}
""" + reaches + f"""electrostatics    = "PME"

[pme]
beta      = {ewald_beta(system)!r}
grid      = [{", ".join(str(n) for n in system["grid"])}]
order     = 4
influence = "OPTIMAL"

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = {system['timestep']}
steps      = {steps}

[ensemble]
{ensemble}
[constraints]
hydrogen_bonds = true
rigid_water    = true
analytic_bonds = {str(analytic_bonds).lower()}

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
                             path="smoke.toml", energy_interval=args.steps,
                             analytic_bonds=args.analytic_bonds)
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
ewald-rtol       = {math.erfc(ewald_beta(system) * CUTOFF)!r}
fourier-nx       = {system["grid"][0]}
fourier-ny       = {system["grid"][1]}
fourier-nz       = {system["grid"][2]}
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


def timed(command, cwd, out, rows=None, env=None):
    """Runs `command` and watches its devices every two seconds; returns the
    exit status, the processes of others that shared the devices, which
    make a rate meaningless, and the wall time of the process in seconds,
    from its start to its exit. With a dict `rows`, the output passes
    through this script on its way to `out`, and `rows` takes the time of
    the clock of the script at each row of energies of MDIR's log
    ("INFO: STEP ..."), by its step."""
    buses = visible_bus_ids()
    shared = others_on(buses, own=-1)
    start = time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, env=env,
                             stdout=out if rows is None else subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    reader = None
    if rows is not None:
        def read():
            for raw in iter(child.stdout.readline, b""):
                now = time.monotonic()
                fields = raw.split()
                if (len(fields) > 1 and fields[0] == b"INFO:"
                        and fields[1].isdigit()):
                    rows.setdefault(int(fields[1]), now)
                out.write(raw.decode(errors="replace"))
            out.flush()
        reader = threading.Thread(target=read)
        reader.start()
    # The exit is taken by a thread of its own, so that the wall time does
    # not include a query of the devices that was under way.
    ends = []
    waiter = threading.Thread(
        target=lambda: (child.wait(), ends.append(time.monotonic())))
    waiter.start()
    while True:
        waiter.join(timeout=2)
        if not waiter.is_alive():
            break
        shared |= others_on(buses, own=child.pid)
    wall = ends[0] - start
    if reader:
        reader.join()
    shared |= others_on(buses, own=child.pid)
    if shared:
        names = ", ".join(f"{name} ({pid})" for pid, name in sorted(shared))
        print(f"warning: the device was shared with {names}; "
              "the rate is not a measurement", file=sys.stderr)
    return child.returncode, shared, wall


def record(args, engine, name, rate, log, shared=False, details=None):
    """Keeps the rate in results.json of the work directory and, with
    --record, appends every number of the run as a line of JSON to that
    file, for scripts/paper/suite-table.py."""
    path = os.path.join(args.work, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    key = engine + (f"/{args.variant}" if args.variant else "")
    results.setdefault(key, {})[name] = {"ns_per_day": rate, "log": log,
                                         "shared": shared}
    with open(path, "w") as file:
        json.dump(results, file, indent=2)
    if args.record:
        entry = {"engine": engine, "variant": args.variant,
                 "repeat": args.repeat, "system": name, "ns_per_day": rate,
                 "shared": shared}
        entry.update(details or {})
        with open(args.record, "a") as file:
            file.write(json.dumps(entry) + "\n")


def remove_outputs(target):
    """The trajectory and the checkpoints of an earlier run of MDIR in the
    directory of a system, and the copies that a run keeps of them (D149),
    so that every run writes its files anew and none accumulates."""
    for pattern in ("mdir.dcd", "mdir.h5", "mdir.h5.prev", "#mdir.*#"):
        for path in glob.glob(os.path.join(target, pattern)):
            os.remove(path)


def pmemd_timings(text):
    """The numbers of the end of an output of pmemd: the steps, the seconds,
    and the ns/day of its "Average timings for last N steps" and of all
    steps, and its wall times of setup, of the rest, and in total (whole
    seconds, as pmemd prints them)."""
    found = {}
    last = re.search(r"Average timings for last\s+([0-9]+) steps:\s*\n"
                     r"\|\s+Elapsed\(s\) =\s+([0-9.]+).*\n"
                     r"\|\s+ns/day =\s+([0-9.]+)", text)
    if last:
        found.update(window_steps=int(last.group(1)),
                     window_seconds=float(last.group(2)),
                     window_ns_per_day=float(last.group(3)))
    every = re.search(r"Average timings for all steps:\s*\n"
                      r"\|\s+Elapsed\(s\) =\s+([0-9.]+).*\n"
                      r"\|\s+ns/day =\s+([0-9.]+)", text)
    if every:
        found.update(all_seconds=float(every.group(1)),
                     all_ns_per_day=float(every.group(2)))
    for key, label in (("setup_wall", "Setup"), ("nonsetup_wall", "NonSetup"),
                       ("total_wall", "Total")):
        match = re.search(rf"\|\s+{label} wall time:\s+([0-9.]+)", text)
        if match:
            found[key] = float(match.group(1))
    return found


def mdir_timings(text):
    """The numbers of the end of MDIR's log: the time of compilation, the
    time and the rate of all steps, and MDIR's own rate past the middle of
    the run, with its steps."""
    found = {}
    match = re.search(r"MDIR: compiled in ([0-9.]+) s", text)
    if match:
        found["compiled"] = float(match.group(1))
    match = re.search(r"MDIR: ran in ([0-9.]+) s(?:, [0-9.]+ ms per step, "
                      r"([0-9.]+) ns per day)?", text)
    if match:
        found["all_seconds"] = float(match.group(1))
        if match.group(2):
            found["all_ns_per_day"] = float(match.group(2))
    match = re.search(r"MDIR: from step ([0-9]+) to step ([0-9]+), [0-9.]+ ms "
                      r"per step, ([0-9.]+) ns per day", text)
    if match:
        found.update(half_from=int(match.group(1)),
                     half_to=int(match.group(2)),
                     half_ns_per_day=float(match.group(3)))
    match = re.search(r"^MDIR: skins: the reach ([0-9.]+) Å"
                      r"(?:, pruned to ([0-9.]+) Å)? \((\w+):", text, re.M)
    if match:
        found.update(reach=float(match.group(1)),
                     pruned=float(match.group(2)) if match.group(2) else None,
                     reach_source=match.group(3))
    return found


def checkpoint_interval(mdin):
    """The interval of MDIR's checkpoints for pmemd's restarts every `ntwr`
    steps: `ntwr` where it divides `nstlim`, else the largest multiple of
    `ntwx` below it that does, since MDIR's intervals divide the steps and
    one another (Cellulose: 6000 for 8000, five checkpoints against pmemd's
    three and the one at the end)."""
    for interval in range(mdin["ntwr"], 0, -1):
        if mdin["nstlim"] % interval == 0 and interval % mdin["ntwx"] == 0:
            return interval
    return mdin["nstlim"]


def ns_per_day(system, steps, seconds):
    return system["timestep"] * steps * 1e-3 * 86400.0 / seconds


def run(args):
    names = args.systems or list(SYSTEMS)
    unknown = [name for name in names if name not in SYSTEMS]
    if unknown:
        sys.exit(f"unknown systems: {', '.join(unknown)}")
    if args.energies_only and not args.suite_protocol:
        sys.exit("--energies-only needs --suite-protocol")
    for name in names:
        system = SYSTEMS[name]
        target = os.path.join(args.work, name)
        if not os.path.exists(os.path.join(target, "system.top")):
            sys.exit(f"{name} is not prepared; run 'bench.py prepare'")
        mdin = read_mdin(args.work, system)
        details = {"protocol": "suite" if args.suite_protocol else "short",
                   "energies_only": args.energies_only}
        # The output of the commands; GROMACS writes its own log,
        # gromacs.log, where its rate is.
        log = os.path.join(target, f"{args.engine}.out")
        if args.engine == "pmemd":
            # The input of the suite as it is (mdin.GPU), in a directory of
            # its own; with --energies-only a copy without the trajectory
            # (ntwx=0) and with a restart only at the end (ntwr=nstlim), in
            # another. The rate is that of all steps, from the timings at
            # the end of the output of pmemd, or, with --suite-protocol,
            # that of the steps after the first report.
            work = os.path.join(target, "pmemd-energies" if args.energies_only
                                else "pmemd")
            os.makedirs(work, exist_ok=True)
            source = suite_directory(args.work, system)
            for file in ("mdin.GPU", "prmtop", "inpcrd"):
                shutil.copy(os.path.join(source, file), work)
            if args.energies_only:
                path = os.path.join(work, "mdin.GPU")
                text = open(path).read()
                text = re.sub(r"\bntwx\s*=\s*[0-9]+", "ntwx=0", text)
                text = re.sub(r"\bntwr\s*=\s*[0-9]+", f"ntwr={mdin['nstlim']}",
                              text)
                open(path, "w").write(text)
            for file in ("mdout", "mdcrd", "restrt", "mdinfo"):
                if os.path.exists(os.path.join(work, file)):
                    os.remove(os.path.join(work, file))
            with open(log, "w") as out:
                code, shared, wall = timed(
                    [args.pmemd, "-O", "-i", "mdin.GPU", "-p", "prmtop", "-c",
                     "inpcrd", "-o", "mdout", "-r", "restrt", "-x", "mdcrd",
                     "-inf", "mdinfo"], work, out)
            mdout = os.path.join(work, "mdout")
            text = open(mdout).read() if os.path.exists(mdout) else ""
            details.update(pmemd_timings(text))
            details.update(steps=mdin["nstlim"], wall=wall, exit=code)
            if args.suite_protocol:
                rate = details.get("window_ns_per_day")
                if details.get("window_steps") not in (None, mdin["nstlim"] - mdin["ntpr"]):
                    print(f"warning: {name}: pmemd timed the last "
                          f"{details['window_steps']} steps, not "
                          f"{mdin['nstlim'] - mdin['ntpr']}", file=sys.stderr)
            else:
                rates = re.findall(r"ns/day =\s+([0-9.]+)", text)
                rate = float(rates[-1]) if rates else None
        elif args.engine == "mdir":
            if args.suite_protocol:
                remove_outputs(target)
                files = ({} if args.energies_only else
                         dict(trajectory_interval=mdin["ntwx"],
                              checkpoint_interval=checkpoint_interval(mdin)))
                control = write_mdir(name, system, target, skin=args.skin,
                                     neighbor_structure=args.neighbor_structure,
                                     barostat_work=args.barostat_work,
                                     prune_skin=args.prune_skin,
                                     steps=mdin["nstlim"],
                                     energy_interval=mdin["ntpr"],
                                     path="mdir-suite.toml", **files)
                steps = mdin["nstlim"]
            else:
                control = write_mdir(name, system, target, skin=args.skin,
                                     neighbor_structure=args.neighbor_structure,
                                     barostat_work=args.barostat_work,
                                     prune_skin=args.prune_skin)
                steps = system["steps"]
            rows = {}
            with open(log, "w") as out:
                code, shared, wall = timed([args.mdir, "run", control], target,
                                           out, rows=rows)
            text = open(log).read()
            details.update(mdir_timings(text))
            details.update(steps=steps, wall=wall, exit=code)
            # The window of the suite's protocol, by the clock of this script
            # at the rows of the first report and of the last step; and the
            # same clock over MDIR's own window, as a check of the first.
            first, final = mdin["ntpr"], steps
            if first in rows and final in rows and rows[final] > rows[first]:
                details.update(window_steps=final - first,
                               window_seconds=rows[final] - rows[first])
                details["window_ns_per_day"] = ns_per_day(
                    system, final - first, details["window_seconds"])
            a, b = details.get("half_from"), details.get("half_to")
            if a in rows and b in rows and rows[b] > rows[a]:
                details["half_ns_per_day_by_rows"] = ns_per_day(
                    system, b - a, rows[b] - rows[a])
            if args.suite_protocol:
                rate = details.get("window_ns_per_day")
            else:
                # The rate of the second half, past the set-up and the first
                # build, as GROMACS reports it with -resethway; that of the
                # whole run if the log has none.
                rate = details.get("half_ns_per_day",
                                   details.get("all_ns_per_day"))
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
                    code, shared, wall = timed(
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
            details.update(wall=wall, exit=code)
        print(f"{args.engine:8s} {name:15s} "
              f"{rate if rate is not None else 'failed; see ' + log}"
              f"{' (the device was shared)' if shared else ''}")
        record(args, args.engine, name, rate, log, bool(shared), details)


def report(args):
    path = os.path.join(args.work, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    engines = [e for e in ("mdir", "gromacs", "pmemd") if e in results]
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
        coordinates = os.path.join(args.work, "Amber24_Benchmark_Suite", "PME",
                                   system["directory"], "inpcrd")
        if os.path.exists(coordinates):
            with open(coordinates) as file:
                next(file)
                atoms = next(file).split()[0]
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
    p.add_argument("engine", choices=["mdir", "gromacs", "pmemd"])
    p.add_argument("systems", nargs="*", metavar="SYSTEM",
                   help="of " + ", ".join(SYSTEMS) + "; all by default")
    p.add_argument("--mdir", default="mdir")
    p.add_argument("--skin", type=skin_value, default=SKIN,
                   help="of the neighbor structures of MDIR (Å), or AUTO: "
                        "pairlist_distance = \"AUTO\", and pruned_distance "
                        "as well unless --prune-skin is given, which is what "
                        "keys not given mean: the reaches of an earlier run "
                        "of the same key cached, else the defaults, or a "
                        "search where the run is long enough (D281)")
    p.add_argument("--prune-skin", type=skin_value,
                   help="of the inner list of a dual list of MDIR (Å) at a "
                        "step of 2 fs, in proportion to the step (D114), or "
                        "AUTO; one list if absent, unless --skin is AUTO")
    p.add_argument("--neighbor-structure", choices=["AUTO", "MATRIX", "GROUPS"],
                   help="of MDIR on the device; its default if absent")
    p.add_argument("--barostat-work",
                   choices=["TROTTER", "TROTTER_FIRST_ORDER", "EXACT",
                            "FIRST_ORDER"],
                   help="how MDIR's barostat counts a scaling; its default "
                        "if absent")
    p.add_argument("--gmx", default="gmx")
    p.add_argument("--pmemd", default="pmemd.cuda",
                   help="pmemd.cuda of Amber, run on the input of the suite "
                        "as it is")
    p.add_argument("--threads", type=int, default=8,
                   help="OpenMP threads of GROMACS")
    p.add_argument("--suite-protocol", action="store_true",
                   help="MDIR and pmemd: the steps and the output of each "
                        "system's mdin.GPU, and the rate from its first "
                        "report to its last step (D[suite-symmetric])")
    p.add_argument("--energies-only", action="store_true",
                   help="with --suite-protocol: no trajectory and no "
                        "restarts (pmemd: ntwx=0, ntwr=nstlim)")
    p.add_argument("--record", metavar="FILE",
                   help="appends the numbers of each run to FILE, one line "
                        "of JSON a run")
    p.add_argument("--variant", default="",
                   help="a name of the settings, kept with the numbers")
    p.add_argument("--repeat", type=int, default=0,
                   help="the number of the repeat, kept with the numbers")
    commands.add_parser("report")
    p = commands.add_parser("smoke")
    p.add_argument("systems", nargs="*", metavar="SYSTEM")
    p.add_argument("--mdir", default="mdir")
    p.add_argument("--steps", type=int, default=2 * COUPLING_INTERVAL)
    p.add_argument("--analytic-bonds", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report,
     "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
