"""CPU timings of MDIR, GROMACS, and pmemd, alternated in one job (#359, #552).

    bench.py SYSTEM ROUNDS VARIANT[,VARIANT...] [--tag T] [--steps-scale X]
        [--ensemble nve|coupled] [--cutoff ANGSTROM] [--min-steps N]
        [--grid NX,NY,NZ]

SYSTEM is a directory under $CPU_BENCH_DIR with the files of the size sweep
of #345 (scripts/benchmarks/pair-loop/sweep.py of PR #353 prepares them):
sys.prmtop and sys.inpcrd (tleap), npt.h5 (MDIR's equilibrated state),
gmx.top, eq.gro, and eq.cpt (GROMACS's); or `jac`, with sys.prmtop,
sys.inpcrd, gmx.top, and eq.gro, which runs NVT from the coordinates.
Both engines: cutoff 9 A (JAC 8 A), PME with an Ewald tolerance of 1e-5,
order 4, the grid that GROMACS picks for a spacing of 1 A (MDIR is given it),
rigid water and constrained bonds to hydrogen, 2 fs, V-rescale and
C-rescale every 25 steps.

VARIANT:
  m<T>         MDIR mixed, target CPU, T threads, the default list
  d<T>         MDIR double, target CPU, T threads, the default list
  m<T>r<D>     the same with pairlist_distance D (A)
  ms<T>        one run of m<T> under the sampler (sampler.c), not timed
  m<T>@NAME    (and ms<T>@NAME) the same with the program NAME of --builds
  g<T>         GROMACS -nb cpu -pme cpu, T OpenMP threads, its default list policy
  x<T>         GROMACS with an exact buffer: verlet-buffer-tolerance -1,
               rlist = cutoff + 3 A, nstlist 10
  p1           serial pmemd 26, double precision, one core (NVE only)
The rate of MDIR is that of the second half of its run ("from step"), that of
GROMACS with -resethway. Writes $CPU_BENCH_DIR/SYSTEM/runs/TAG/<variant>-r<round>/
and appends a JSON line per run to runs/TAG/results.jsonl; summarize.py
reads them. --builds NAME=PATH,NAME=PATH names programs of MDIR (frozen
copies of builds), so that builds are compared in one job. Round 0 is to be dropped; the order alternates from round to round.

The environment: CPU_BENCH_DIR (required); MDIR, GMX, PMEMD (programs;
by default the bare names); SAMPLER (the library built from sampler.c).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

from timing import pmemd_all_step_ms

if not os.environ.get("CPU_BENCH_DIR"):
    sys.exit("bench.py: CPU_BENCH_DIR, the directory of the systems, is not set")
S = os.environ["CPU_BENCH_DIR"]
MDIR = os.environ.get("MDIR", "mdir")
GMX = os.environ.get("GMX", "gmx")
PMEMD = os.environ.get("PMEMD", "pmemd")
SAMPLER = os.environ.get("SAMPLER", "")

system = sys.argv[1]
rounds = int(sys.argv[2])
variants = sys.argv[3].split(",")
scale = 1.0
ensemble = "coupled"
cutoff = None
min_steps = 100
forced_grid = None
BUILDS = {}
tag = "t" + time.strftime("%m%d-%H%M%S")
args = sys.argv[4:]
while args:
    if args[0] == "--steps-scale":
        scale = float(args[1]); args = args[2:]
    elif args[0] == "--builds":
        BUILDS = dict(b.split("=", 1) for b in args[1].split(",")); args = args[2:]
    elif args[0] == "--tag":
        tag = args[1]; args = args[2:]
    elif args[0] == "--ensemble":
        ensemble = args[1]; args = args[2:]
    elif args[0] == "--cutoff":
        cutoff = float(args[1]); args = args[2:]
    elif args[0] == "--min-steps":
        min_steps = int(args[1]); args = args[2:]
    elif args[0] == "--grid":
        forced_grid = [int(value) for value in args[1].split(",")]
        args = args[2:]
    else:
        sys.exit("unknown " + args[0])
D = f"{S}/{system}"
JAC = system == "jac"
if ensemble not in ("nve", "coupled"):
    sys.exit("--ensemble must be nve or coupled")
if min_steps < 2 or min_steps % 2:
    sys.exit("--min-steps must be a positive even number")
if forced_grid is not None and (len(forced_grid) != 3 or min(forced_grid) < 1):
    sys.exit("--grid must have three positive dimensions")
if "p1" in variants and ensemble != "nve":
    sys.exit("p1 requires --ensemble nve")
cut = cutoff if cutoff is not None else (8.0 if JAC or system.startswith("cellulose") else 9.0)
natoms = int(open(D + "/eq.gro").read().splitlines()[1])
OUT = f"{D}/runs/{tag}"
os.makedirs(OUT, exist_ok=True)
MODEL = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")

GMX_MDP = f"""integrator      = md
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
""" + ("pcoupl = no\n" if JAC else """pcoupl          = c-rescale
tau-p           = 2.0
ref-p           = 1.01325
compressibility = 4.5e-5
nstpcouple      = 25
""") + "gen-vel = no\ncontinuation = yes\nnsteps = 1000\n"
if ensemble == "nve":
    GMX_MDP = re.sub(r"tcoupl\s*=\s*v-rescale", "tcoupl = no", GMX_MDP)
    GMX_MDP = re.sub(r"pcoupl\s*=\s*c-rescale", "pcoupl = no", GMX_MDP)
if forced_grid is not None:
    GMX_MDP = GMX_MDP.replace("fourierspacing  = 0.1", "\n".join(
        f"fourier-n{axis} = {length}" for axis, length in zip("xyz", forced_grid)))


def gmx_start():
    start = ["-c", f"{D}/eq.gro"]
    if ensemble != "nve" and os.path.exists(f"{D}/eq.cpt"):
        start += ["-t", f"{D}/eq.cpt"]
    return start


def grid():
    """The grid that GROMACS picks for the box (fourierspacing 0.1 nm)."""
    d = f"{OUT}/grid"
    os.makedirs(d, exist_ok=True)
    open(d + "/run.mdp", "w").write(GMX_MDP)
    subprocess.run([GMX, "grompp", "-f", "run.mdp", *gmx_start(), "-p", f"{D}/gmx.top", "-o", "run.tpr",
                    "-maxwarn", "5"], cwd=d, capture_output=True, check=True)
    dump = subprocess.run([GMX, "dump", "-s", "run.tpr"], cwd=d, capture_output=True, text=True).stdout
    return [int(re.search(rf"fourier-n{a}\s*=\s*(\d+)", dump).group(1)) for a in "xyz"]


GRID = forced_grid if forced_grid is not None else grid()


def steps_for(threads, kind):
    # About 20 s of steps for MDIR and 15 s for GROMACS, from the rates of
    # main at 16 threads (MDIR about 1 us a site and step, GROMACS 1/18 of
    # that) and a step about half serial: MDIR at least 100 steps, a multiple
    # of 100 (the coupling interval, 25, divides half the steps); GROMACS at
    # least 1000.
    rate = 0.95e-6 * natoms * (16.0 / threads) ** 0.5
    if kind == "g":
        rate = rate / 18 * (16.0 / threads) ** 0.5
        minimum = 1000 if ensemble == "coupled" else max(200, min_steps)
        return max(minimum, min(40000, int(15.0 / rate * scale / 100) * 100))
    return max(min_steps, min(20000, int(20.0 / rate * scale / 100) * 100))


def mdir_control(threads, steps, reach=None, precision="MIXED"):
    if JAC or ensemble == "nve":
        head = f'[input]\nformat      = "AMBER"\ntopology    = "{D}/sys.prmtop"\ncoordinates = "{D}/sys.inpcrd"\n'
        ensemble_text = ('[ensemble]\nensemble    = "NVE"\n' if ensemble == "nve" else
                         '[ensemble]\nensemble    = "NVT"\ntemperature = 300.0\n')
    else:
        head = (f'[input]\ntopology    = "{D}/sys.prmtop"\ncoordinates = "{D}/sys.inpcrd"\n'
                f'checkpoint  = "{D}/npt.h5"\n')
        ensemble_text = ('[ensemble]\nensemble    = "NPT"\ntemperature = 300.0\npressure    = 1.0\n'
                    '[barostat]\nmethod        = "C-RESCALE"\ntime_constant = 2.0\n')
    return (head + f"[output]\nenergy_interval = {steps // 2}\n"
            f'[dynamics]\nintegrator = "VELOCITY_VERLET"\ntime_step  = 0.002\nsteps      = {steps}\n'
            + ensemble_text +
            ('' if ensemble == "nve" else
             '[thermostat]\nmethod        = "V-RESCALE"\ntime_constant = 1.0\ninterval      = 25\n') +
            f'[energy]\ncutoff            = {cut}\nelectrostatics    = "PME"\n'
            'coulomb_modifier  = "POTENTIAL_SHIFT"\n'
            + (f"pairlist_distance = {reach}\n" if reach else "") +
            f"[pme]\ntolerance = 1e-05\ngrid      = [{GRID[0]}, {GRID[1]}, {GRID[2]}]\norder     = 4\n"
            '[constraints]\nhydrogen_bonds = true\nrigid_water    = true\n'
            '[boundary]\ntype = "PERIODIC"\n'
            f'[execution]\ntarget    = "CPU"\nthreads   = {threads}\nprecision = "{precision}"\n')


def run_mdir(variant, d):
    sampled = variant[1] == "s"
    m = re.match(r"[md]s?(\d+)(?:r([0-9.]+))?(?:@(\w+))?$", variant)
    threads, reach = int(m.group(1)), m.group(2)
    program = BUILDS[m.group(3)] if m.group(3) else MDIR
    steps = steps_for(threads, "m")
    precision = "DOUBLE" if variant[0] == "d" else "MIXED"
    open(d + "/mdir.toml", "w").write(mdir_control(threads, steps, reach, precision))
    env = dict(os.environ, MDIR_COMPILE_CACHE="off", OMP_PROC_BIND="close", OMP_PLACES="cores")
    if sampled:
        os.makedirs(d + "/prof", exist_ok=True)
        env.update(LD_PRELOAD=SAMPLER, SAMPLER_OUT=d + "/prof")
    t0 = time.time()
    out = subprocess.run([program, "run", "mdir.toml"], cwd=d, capture_output=True, text=True,
                         env=env)
    log = out.stdout + out.stderr
    open(d + "/run.log", "w").write(log)
    rate = re.search(r"from step [0-9]+ to step [0-9]+, ([0-9.]+) ms per step", log)
    b = re.search(r"built (\d+) times, every ([0-9.]+) steps", log)
    return dict(rate_ms=float(rate.group(1)) if rate else None, steps=steps, threads=threads, reach=reach,
                precision=precision, ensemble=ensemble,
                program=program,
                builds=int(b.group(1)) if b else None, build_interval=float(b.group(2)) if b else None,
                wall_s=time.time() - t0, ok=out.returncode == 0)


def run_gmx(variant, d):
    threads = int(variant[1:])
    exact = variant[0] == "x"
    steps = steps_for(threads, "g")
    mdp = GMX_MDP.replace("nsteps = 1000", f"nsteps = {steps}")
    if exact:
        mdp = mdp.replace("verlet-buffer-tolerance = 0.005",
                          f"verlet-buffer-tolerance = -1\nrlist = {(cut + 3) / 10:.2f}\nnstlist = 10")
    open(d + "/run.mdp", "w").write(mdp)
    g = subprocess.run([GMX, "grompp", "-f", "run.mdp", *gmx_start(), "-p", f"{D}/gmx.top", "-o", "run.tpr",
                        "-maxwarn", "5"], cwd=d, capture_output=True, text=True)
    if g.returncode:
        open(d + "/grompp.log", "w").write(g.stdout + g.stderr)
        return dict(ok=False)
    t0 = time.time()
    out = subprocess.run([GMX, "mdrun", "-s", "run.tpr", "-nb", "cpu", "-pme", "cpu", "-bonded", "cpu",
                          "-update", "cpu", "-ntmpi", "1", "-ntomp", str(threads), "-pin", "on",
                          "-resethway", "-noconfout", "-g", "md.log"],
                         cwd=d, capture_output=True, text=True)
    open(d + "/mdrun.out", "w").write(out.stdout + out.stderr)
    log = open(d + "/md.log").read() if os.path.exists(d + "/md.log") else ""
    m = re.search(r"Performance:\s+([0-9.]+)\s+([0-9.]+)", log)
    rate = 0.002 * 86400.0 / float(m.group(1)) if m else None  # ms a step of 2 fs, from ns/day
    return dict(rate_ms=rate, steps=steps, threads=threads, exact=exact, ensemble=ensemble,
                wall_s=time.time() - t0, ok=out.returncode == 0)


def run_pmemd(d):
    """Serial NVE reference on the same Amber topology and coordinates."""
    steps = steps_for(1, "m")
    with open(D + "/sys.inpcrd") as restart_file:
        restart_file.readline()
        count = int(restart_file.readline().split()[0])
        remaining = sum(1 for _ in restart_file)
    # Formatted Amber restart files use six coordinates or velocities per
    # line, followed by one box line. A topology without velocities starts
    # pmemd with a generated velocity field instead of pretending to restart.
    has_velocities = remaining > (3 * count + 5) // 6 + 1
    mdin = f"""CPU baseline, NVE
 &cntrl
  imin=0, ntx={5 if has_velocities else 1}, irest={1 if has_velocities else 0},
  nstlim={steps}, dt=0.002, ntpr={steps}, ntwx=0, ntwr={steps},
  ntc=2, ntf=2, tol=0.000001, cut={cut},
  ntt=0, ntb=1, ntp=0, tempi=300, ig=12345, ioutfm=1,
 /
 &ewald
  dsum_tol={1e-5 / cut:.12g}, nfft1={GRID[0]}, nfft2={GRID[1]}, nfft3={GRID[2]}, order=4,
 /
"""
    with open(d + "/mdin", "w") as file:
        file.write(mdin)
    command = [PMEMD, "-O", "-i", "mdin", "-p", D + "/sys.prmtop",
               "-c", D + "/sys.inpcrd", "-o", "mdout", "-r", "restrt"]
    t0 = time.time()
    out = subprocess.run(command, cwd=d, capture_output=True, text=True)
    with open(d + "/run.log", "w") as file:
        file.write(out.stdout + out.stderr)
    output = open(d + "/mdout").read() if os.path.exists(d + "/mdout") else ""
    rate = pmemd_all_step_ms(output) if out.returncode == 0 else None
    return dict(rate_ms=rate, steps=steps, threads=1, precision="DOUBLE",
                ensemble=ensemble, has_velocities=has_velocities,
                wall_s=time.time() - t0, ok=out.returncode == 0)


def run(variant, r):
    d = f"{OUT}/{variant}-r{r}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    res = (run_mdir(variant, d) if variant[0] in "md" else
           run_pmemd(d) if variant == "p1" else run_gmx(variant, d))
    res.update(system=system, sites=natoms, variant=variant, round=r, dir=d, tag=tag,
               host=os.uname().nodename, cpu=MODEL, grid=GRID, cutoff=cut,
               affinity=sorted(os.sched_getaffinity(0)))
    with open(f"{OUT}/results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)
    if not res.get("ok") or res.get("rate_ms") is None:
        raise RuntimeError(f"{variant} failed or produced no rate; see its run files")


print(f"{system}: {natoms} sites; grid {GRID}; host {os.uname().nodename}; cpu {MODEL}; "
      f"affinity {sorted(os.sched_getaffinity(0))}", flush=True)
timed = [v for v in variants if v[1] != "s"]
for r in range(rounds):
    for v in (timed if r % 2 == 0 else timed[::-1]):
        run(v, r)
for v in variants:
    if v[1] == "s":
        run(v, 0)
