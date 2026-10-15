"""One system of the size sweep of #345, both engines, in one job.

    sweep.py NAME [ROUNDS]

NAME is a directory under $SWEEP_DIR with sys.prmtop, sys.inpcrd (tleap),
gmx.top and start.gro (ParmEd), or with npt.h5 (MDIR), eq.gro and eq.cpt
(GROMACS) of an equilibrated box; or jac, whose directory holds
sys.prmtop, sys.inpcrd, gmx.top, and eq.gro (JAC of the Amber suite,
converted with ParmEd). Equilibrates with each engine when not yet
done (MDIR: minimization, NVT 10 ps, NPT 40 ps; GROMACS: steepest descent,
NPT 20 ps), then times the variants, alternated round by round (round 0
dropped), then runs each once under nsys. Writes $SWEEP_DIR/NAME/results.json,
which fit.py reads.

The environment: SWEEP_DIR (required), GMX, NSYS, MDIR (programs; by
default the bare names), MDIR_NEW (the build with the EXPERIMENT variants
of the branch feature/pair-loop-staging; MDIR by default). Times are on
one kind of device; compare only within one job.

Variants:
  gmx       GROMACS 2026.3, default list policy (verlet-buffer-tolerance)
  m96       MDIR, groups, dual list 12 / 9.6 Å (cutoff + 3, cutoff + 0.6)
  m93, m100 the same with an inner reach of cutoff + 0.3 and + 1.0 Å
  p96       m96 with the prototype kernel (MDIR_EXP_PAIR_LOOP=bcast)
  s96       m96 with PME on a second stream (MDIR_EXP_PME_STREAM=1)
Same cutoff, Ewald tolerance (1e-5), spline order (4), and grid for both
engines: GROMACS picks the grid for a spacing of 1 Å, MDIR is given it.
"""
import csv
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

if not os.environ.get("SWEEP_DIR"):
    print("sweep.py: SWEEP_DIR, the directory of the systems, is not set", file=sys.stderr)
    sys.exit(2)
S = os.environ["SWEEP_DIR"]
GMX = os.environ.get("GMX", "gmx")
NSYS = os.environ.get("NSYS", "nsys")
MAIN = os.environ.get("MDIR", "mdir")
NEW = os.environ.get("MDIR_NEW", MAIN)
name = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
JAC = name == "jac"
d = f"{S}/{name}"
os.makedirs(d, exist_ok=True)
os.chdir(d)
cut = 8.0 if JAC else 9.0
env0 = dict(os.environ, MDIR_COMPILE_CACHE="off")


def sh(cmd, log, env=None, cwd=None):
    with open(log, "w") as o:
        r = subprocess.run(cmd, stdout=o, stderr=subprocess.STDOUT, env=env or env0, cwd=cwd)
    if r.returncode:
        print("FAILED", " ".join(cmd), "see", log, flush=True)
        print(open(log).read()[-3000:], flush=True)
        sys.exit(1)


# ------------------------------------------------------------ GROMACS
mdp = f"""integrator      = md
dt              = 0.002
cutoff-scheme   = Verlet
rcoulomb        = {cut / 10}
rvdw            = {cut / 10}
coulombtype     = PME
coulomb-modifier = Potential-shift
vdw-modifier    = None
DispCorr        = EnerPres
ewald-rtol      = 1e-5
pme-order       = 4
fourierspacing  = 0.1
constraints     = h-bonds
constraint-algorithm = lincs
tcoupl          = v-rescale
tc-grps         = System
tau-t           = 1.0
ref-t           = 300
nsttcouple      = 25
nstcalcenergy   = 100
nstenergy       = 0
nstlog          = 0
verlet-buffer-tolerance = 0.005
"""
npt = """pcoupl          = c-rescale
tau-p           = 2.0
ref-p           = 1.01325
compressibility = 4.5e-5
nstpcouple      = 25
"""
if not os.path.exists("eq.gro"):
    open("em.mdp", "w").write(mdp.replace("integrator      = md", "integrator      = steep\nnsteps = 1000")
                              .replace("tcoupl          = v-rescale", "tcoupl = no"))
    sh([GMX, "grompp", "-f", "em.mdp", "-c", "start.gro", "-p", "gmx.top", "-o", "em.tpr", "-maxwarn", "5"],
       "grompp-em.log")
    sh([GMX, "mdrun", "-s", "em.tpr", "-deffnm", "em", "-ntmpi", "1", "-ntomp", "16"], "mdrun-em.log")
    open("eq.mdp", "w").write(mdp + npt + "gen-vel = yes\ngen-temp = 300\ncontinuation = no\nnsteps = 10000\n")
    sh([GMX, "grompp", "-f", "eq.mdp", "-c", "em.gro", "-p", "gmx.top", "-o", "eq.tpr", "-maxwarn", "5"],
       "grompp-eq.log")
    sh([GMX, "mdrun", "-s", "eq.tpr", "-deffnm", "eq", "-ntmpi", "1", "-ntomp", "16", "-nb", "gpu",
        "-pme", "gpu"], "mdrun-eq.log")
natoms = int(open("eq.gro").read().splitlines()[1])
STEPS = 4000 if natoms > 150000 else 10000 if natoms > 40000 else 20000
if JAC:
    run_mdp = mdp + "pcoupl = no\ngen-vel = no\ncontinuation = yes\n" + f"nsteps = {STEPS}\n"
    start = ["-c", "eq.gro"]
else:
    run_mdp = mdp + npt + "gen-vel = no\ncontinuation = yes\n" + f"nsteps = {STEPS}\n"
    start = ["-c", "eq.gro", "-t", "eq.cpt"]
open("gmx.mdp", "w").write(run_mdp)
sh([GMX, "grompp", "-f", "gmx.mdp", *start, "-p", "gmx.top", "-o", "gmx.tpr", "-maxwarn", "5"],
   "grompp-run.log")
dump = subprocess.run([GMX, "dump", "-s", "gmx.tpr"], capture_output=True, text=True).stdout
grid = [int(re.search(rf"fourier-n{a}\s*=\s*(\d+)", dump).group(1)) for a in "xyz"]
print(f"{name}: {natoms} sites, grid {grid}, {STEPS} steps", flush=True)

# ------------------------------------------------------------ MDIR
energy = f"""[energy]
cutoff            = {cut}
electrostatics    = "PME"
coulomb_modifier  = "POTENTIAL_SHIFT"

[pme]
tolerance = 1e-05
grid      = [{grid[0]}, {grid[1]}, {grid[2]}]
order     = 4

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
lists = "pairlist_distance = {o}\npruned_distance = {i}\n"
if not JAC and not os.path.exists("npt.h5"):
    head = '[input]\ntopology = "sys.prmtop"\ncoordinates = "sys.inpcrd"\n'
    open("1-min.toml", "w").write(head + '[output]\ncheckpoint = "min.h5"\nenergy_interval = 500\n'
                                  '[minimize]\nmethod = "STEEPEST_DESCENT"\nsteps = 2000\n' + energy)
    open("2-nvt.toml", "w").write(head + 'checkpoint = "min.h5"\n[output]\ncheckpoint = "nvt.h5"\n'
                                  'checkpoint_interval = 10000\nenergy_interval = 5000\n[dynamics]\n'
                                  'integrator = "VELOCITY_VERLET"\ntime_step = 0.001\nsteps = 10000\n'
                                  '[ensemble]\nensemble = "NVT"\ntemperature = 300.0\n[thermostat]\n'
                                  'method = "V-RESCALE"\ntime_constant = 0.5\ninterval = 10\n' + energy)
    open("3-npt.toml", "w").write(head + 'checkpoint = "nvt.h5"\n[output]\ncheckpoint = "npt.h5"\n'
                                  'checkpoint_interval = 20000\nenergy_interval = 5000\n[dynamics]\n'
                                  'integrator = "VELOCITY_VERLET"\ntime_step = 0.002\nsteps = 20000\n'
                                  '[ensemble]\nensemble = "NPT"\ntemperature = 300.0\npressure = 1.0\n'
                                  '[thermostat]\nmethod = "V-RESCALE"\ntime_constant = 0.5\ninterval = 10\n'
                                  '[barostat]\nmethod = "C-RESCALE"\ntime_constant = 2.0\n' + energy)
    for stage in ("1-min", "2-nvt", "3-npt"):
        text = open(f"{stage}.toml").read().replace(
            'coulomb_modifier  = "POTENTIAL_SHIFT"\n',
            'coulomb_modifier  = "POTENTIAL_SHIFT"\n' + lists.format(o=cut + 3, i=cut + 0.6))
        open(f"{stage}.toml", "w").write(text)
        sh([MAIN, "run", f"{stage}.toml"], f"{stage}.log")
    print(f"{name}: MDIR equilibrated", flush=True)

if JAC:
    run_head = f"""[input]
format      = "AMBER"
topology    = "sys.prmtop"
coordinates = "sys.inpcrd"
[output]
energy_interval = {STEPS // 2}
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {STEPS}
[ensemble]
ensemble    = "NVT"
temperature = 300.0
[thermostat]
method        = "V-RESCALE"
time_constant = 1.0
interval      = 25
"""
else:
    run_head = f"""[input]
topology    = "{d}/sys.prmtop"
coordinates = "{d}/sys.inpcrd"
checkpoint  = "{d}/npt.h5"
[output]
energy_interval = {STEPS // 2}
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {STEPS}
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
"""
if JAC:
    run_head = run_head.replace('"sys.prmtop"', f'"{d}/sys.prmtop"').replace('"sys.inpcrd"', f'"{d}/sys.inpcrd"')
mdir_variants = {
    "m96": (MAIN, {}, cut + 0.6),
    "m93": (MAIN, {}, cut + 0.3),
    "m100": (MAIN, {}, cut + 1.0),
    "p96": (NEW, {"MDIR_EXP_PAIR_LOOP": "bcast"}, cut + 0.6),
    "s96": (NEW, {"MDIR_EXP_PME_STREAM": "1"}, cut + 0.6),
}
for v, (_, _, rin) in mdir_variants.items():
    text = run_head + energy.replace('coulomb_modifier  = "POTENTIAL_SHIFT"\n',
                                     'coulomb_modifier  = "POTENTIAL_SHIFT"\n' + lists.format(o=cut + 3, i=rin))
    open(f"{v}.toml", "w").write(text)
gmx_args = ["-ntmpi", "1", "-ntomp", "16", "-nb", "gpu", "-pme", "gpu", "-notunepme", "-resethway", "-noconfout"]


def run(v, tag, nsys=False):
    w = f"{d}/runs/{v}-{tag}"
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    if v == "gmx":
        cmd = [GMX, "mdrun", "-s", f"{d}/gmx.tpr", "-deffnm", "gmx"] + gmx_args
        env = env0
    else:
        binary, extra, _ = mdir_variants[v]
        cmd = [binary, "run", f"{d}/{v}.toml"]
        env = dict(env0, **extra)
    if nsys:
        cmd = [NSYS, "profile", "-t", "cuda", "-o", f"{w}/prof", "--force-overwrite", "true"] + cmd
    with open(f"{w}/out.log", "w") as o:
        subprocess.run(cmd, cwd=w, env=env, stdout=o, stderr=subprocess.STDOUT)
    if v == "gmx":
        m = re.search(r"Performance:\s+([0-9.]+)", open(f"{w}/gmx.log").read())
        return 172.8 / float(m.group(1)) * 1000 if m else None
    m = re.search(r"from step [0-9]+ to step [0-9]+, ([0-9.]+) ms per step", open(f"{w}/out.log").read())
    return float(m.group(1)) * 1000 if m else None


names = ["gmx"] + list(mdir_variants)
times = {v: [] for v in names}
for r in range(rounds):
    for v in (names if r % 2 == 0 else names[::-1]):
        t = run(v, f"r{r}")
        print(f"{name} round {r} {v}: {t} µs/step", flush=True)
        if r > 0 and t:
            times[v].append(t)


def kernels(report):
    out = subprocess.run([NSYS, "stats", "-r", "cuda_gpu_kern_sum", "-f", "csv", "-q", report],
                         capture_output=True, text=True).stdout
    if "Time (%)" not in out:
        return []
    return [{"name": r["Name"], "ns": float(r["Total Time (ns)"]), "instances": int(r["Instances"])}
            for r in csv.DictReader(io.StringIO(out[out.index("Time (%)"):]))]


results = {"name": name, "sites": natoms, "grid": grid, "steps": STEPS, "cutoff": cut, "variants": {}}
for v in names:
    run(v, "nsys", nsys=True)
    w = f"{d}/runs/{v}-nsys"
    log = open(f"{w}/gmx.log" if v == "gmx" else f"{w}/out.log").read()
    info = {"times": times[v], "kernels": kernels(f"{w}/prof.nsys-rep"),
            "r_in": None if v == "gmx" else mdir_variants[v][2]}
    if v == "gmx":
        m = re.search(r"inner list: updated every\s+(\d+) steps, buffer [0-9.]+ nm, rlist ([0-9.]+) nm", log)
        o = re.search(r"outer list: updated every\s+(\d+) steps, buffer [0-9.]+ nm, rlist ([0-9.]+) nm", log)
        if m and o:
            info.update(prune_interval=int(m.group(1)), r_in=float(m.group(2)) * 10,
                        build_interval=int(o.group(1)), r_out=float(o.group(2)) * 10)
        nb = re.search(r"NxN Ewald Elec\. \+ LJ \[F\]\s+([0-9.]+)", log)
        info["nxn_mnumber"] = float(nb.group(1)) if nb else None
    else:
        b = re.search(r"built (\d+) times, every ([0-9.]+) steps", log)
        p = re.search(r"pruned (\d+) times(?:, every ([0-9.]+) steps)?", log)
        if b:
            info.update(builds=int(b.group(1)), build_interval=float(b.group(2)))
        if p:
            info.update(prunes=int(p.group(1)), prune_interval=float(p.group(2)) if p.group(2) else None)
    results["variants"][v] = info
    print(f"{name} {v}: nsys done", flush=True)
for v in names:
    t = times[v]
    if t:
        se = statistics.stdev(t) / len(t) ** 0.5 if len(t) > 1 else 0.0
        print(f"{name} {v}: {statistics.mean(t):.1f} ± {se:.1f} µs/step (standard error, n={len(t)})", flush=True)
json.dump(results, open(f"{d}/results.json", "w"), indent=1)
print(f"{name}: wrote {d}/results.json", flush=True)
