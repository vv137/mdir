#!/usr/bin/env python3
"""Compares the semi-isotropic stochastic cell rescaling of MDIR (D119)
with that of GROMACS on a bilayer of POPC with Lipid21 in TIP3P water.

    scripts/validation/barostat-semi/membrane.py WORK --mdir MDIR --gmx GMX
        [--ns 20] [--phases build,equil,terms,runs,analyze]
        [--mdir-gpu 0] [--gmx-gpu 1]

The phases, in WORK:

  - build: packmol-memgen writes the input of packmol for 63 lipids in a
    leaflet on a square of 65 Å with 17.5 Å of water on each side
    (packmol.inp; it does not run packmol itself here), packmol packs it,
    tleap makes popc.prmtop and popc.inpcrd with Lipid21 and TIP3P in a
    cell of 66 x 66 x 83 Å, and ParmEd writes popc.top and popc.gro for
    GROMACS. AmberTools on PATH.
  - equil: MDIR minimizes 5000 steps, runs 100 ps at 1 fs, then 10 ns at
    2 fs, at 303 K and 1 bar, semi-isotropic, a frame every 20 ps
    (equil.dcd, which scripts/render/movie.py renders).
  - terms: the energy terms of the end of equil, MDIR on the CPU in double
    precision against a rerun of GROMACS with the same grid, to show that
    the two read the same model.
  - runs: NS ns of each from the end of equil, MDIR from its checkpoint and
    GROMACS from its last frame with velocities drawn at 303 K, at once on
    two devices, the cell every 20 ps.
  - analyze: the area per lipid, the height and the volume of the cell,
    with errors from the statistical inefficiency, and the modulus of the
    area from its fluctuations, K_A = k_B T <A> / var(A), over the runs
    after their first nanosecond.

Both take a cutoff of 10 Å with the correction for the dispersion, PME
with erfc(beta r_c) = 1e-5 and a spacing of at most 1 Å, bonds to
hydrogens and water rigid, stochastic velocity rescaling (tau_T = 1 ps)
and semi-isotropic stochastic cell rescaling (tau_P = 5 ps, 4.5e-5 /bar
in both directions) every 25 steps, in mixed precision. Needs numpy.
"""
import argparse
import math
import os
import re
import shutil
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ensembles"))
from analyze import inefficiency  # noqa: E402

LIPIDS_PER_LEAFLET = 63

MDIR_CONTROL = """[input]
topology    = "popc.prmtop"
coordinates = "popc.inpcrd"
{checkpoint}
[output]
energy_interval     = {energy}
trajectory          = "{name}.dcd"
trajectory_interval = {frames}
checkpoint          = "{name}.h5"
checkpoint_interval = {steps}

[energy]
cutoff            = 10.0
pairlist_distance = 13.0
pruned_distance   = 10.6
electrostatics    = "PME"

[pme]
tolerance   = 1e-05
max_spacing = 1.0

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = {dt}
steps      = {steps}
seed       = {seed}

[ensemble]
ensemble    = "NPT"
temperature = 303.0
pressure    = 0.986923

[thermostat]
method        = "V-RESCALE"
time_constant = 1.0
interval      = 25

[barostat]
method        = "C-RESCALE"
time_constant = 5.0
interval      = 25
coupling      = "SEMI_ISOTROPIC"

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

MINIMIZE = """[input]
topology    = "popc.prmtop"
coordinates = "popc.inpcrd"

[output]
energy_interval = 500
checkpoint      = "min.h5"

[energy]
cutoff            = 10.0
pairlist_distance = 12.0
electrostatics    = "PME"

[pme]
tolerance   = 1e-05
max_spacing = 1.0

[minimize]
method       = "STEEPEST_DESCENT"
steps        = 5000
initial_step = 0.01

[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
"""

# ParmEd writes one fudgeLJ for every pair 1-4, but Lipid21 scales the
# Lennard-Jones term of a pair 1-4 by 1/2 for most torsions and by 1/6 for
# some (SCNB_SCALE_FACTOR 2.0 and 6.0), which no single factor holds; it
# wrote fudgeLJ 1, and GROMACS took those terms unscaled, 3845 kcal/mol
# against 1854. Each pair of the lipid gets its parameters here, sigma the
# mean and epsilon the geometric mean over the factor of the first torsion
# that counts it, as Amber takes them; the charges keep fudgeQQ 1/1.2,
# which is the same for all.
CONVERT = """import math
import parmed

s = parmed.load_file("popc.prmtop", "popc.inpcrd")
s.save("popc.top", format="gromacs", overwrite=True)
s.save("popc.gro", overwrite=True)
first = s.residues[0].atoms[0].idx
lipid = set(a.idx for r in s.residues[:3] for a in r.atoms)
scale = {}
for d in s.dihedrals:
    if d.ignore_end or d.atom1.idx not in lipid:
        continue
    key = tuple(sorted((d.atom1.idx, d.atom4.idx)))
    scale.setdefault(key, d.type.scnb)
lines = open("popc.top").read().split("\\n")
out, molecule, section = [], None, None
for line in lines:
    stripped = line.strip()
    if stripped.startswith("["):
        section = stripped.strip("[] ")
    elif section == "moleculetype" and stripped and not stripped.startswith(";"):
        molecule = stripped.split()[0]
    elif (section == "pairs" and molecule != "WAT" and stripped
          and not stripped.startswith(";")):
        i, j = (int(v) - 1 + first for v in stripped.split()[:2])
        a, b = s.atoms[i], s.atoms[j]
        sigma = 0.05 * (a.sigma_14 + b.sigma_14)
        epsilon = math.sqrt(a.epsilon_14 * b.epsilon_14) * 4.184
        epsilon /= scale[tuple(sorted((i, j)))]
        line = f"{i + 1 - first:7d}{j + 1 - first:7d}     1 {sigma:.8f} {epsilon:.8f}"
    out.append(line)
open("popc.top", "w").write("\\n".join(out))
"""

MDP_MODEL = """cutoff-scheme   = Verlet
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = PME
coulomb-modifier = None
vdw-modifier    = None
DispCorr        = EnerPres
ewald-rtol      = 1e-5
fourierspacing  = 0.1
pme-order       = 4
"""


def run(command, cwd, log, env=None):
    with open(os.path.join(cwd, log), "w") as out:
        result = subprocess.run(command, cwd=cwd, stdout=out,
                                stderr=subprocess.STDOUT,
                                env=dict(os.environ, **(env or {})))
    if result.returncode != 0:
        sys.exit(f"{' '.join(command)} failed; see {os.path.join(cwd, log)}")


def write(path, text):
    with open(path, "w") as file:
        file.write(text)


def build(w):
    run(["packmol-memgen", "-l", "POPC", "-r", "1", "--distxy_fix", "65",
         "--notrun", "--noprogress", "--overwrite", "-o", "popc.pdb"], w,
        "memgen.log")
    with open(os.path.join(w, "packmol.inp")) as source, \
            open(os.path.join(w, "packmol.out"), "w") as out:
        subprocess.run(["packmol"], cwd=w, stdin=source, stdout=out,
                       check=True)
    write(os.path.join(w, "leap.in"), """source leaprc.lipid21
source leaprc.water.tip3p
SYS = loadpdb popc.pdb
set SYS box {66.0 66.0 83.0}
saveamberparm SYS popc.prmtop popc.inpcrd
quit
""")
    run(["tleap", "-f", "leap.in"], w, "leap.log")
    write(os.path.join(w, "convert.py"), CONVERT)
    run([shutil.which("python3") or "python3", "convert.py"], w,
        "convert.log")


def equilibrate(args, w):
    env = {"CUDA_VISIBLE_DEVICES": args.mdir_gpu}
    write(os.path.join(w, "min.toml"), MINIMIZE)
    run([args.mdir, "run", "min.toml"], w, "min.log", env)
    write(os.path.join(w, "heat.toml"), MDIR_CONTROL.format(
        checkpoint='checkpoint  = "min.h5"', energy=1000, name="heat",
        frames=2000, steps=100000, dt=0.001, seed=30))
    run([args.mdir, "run", "heat.toml"], w, "heat.log", env)
    write(os.path.join(w, "equil.toml"), MDIR_CONTROL.format(
        checkpoint='checkpoint  = "heat.h5"', energy=5000, name="equil",
        frames=10000, steps=5000000, dt=0.002, seed=31))
    run([args.mdir, "run", "equil.toml"], w, "equil.log", env)


def read_dcd(path):
    """The cells (Å) and the positions (Å) of the frames of a DCD."""
    data = open(path, "rb").read()
    blocks, at = [], 0
    while at + 4 <= len(data):
        (length,) = struct.unpack_from("<i", data, at)
        if at + 8 + length > len(data):
            break
        blocks.append((at + 4, length))
        at += 8 + length
    count = struct.unpack_from("<i", data, blocks[2][0])[0]
    cells, last = [], None
    frames = blocks[3:]
    for start in range(0, len(frames) - 3, 4):
        v = struct.unpack_from("<6d", data, frames[start][0])
        cells.append([v[0], v[2], v[5]])
        last = start
    xyz = [np.frombuffer(data, dtype="<f4", count=count, offset=o)
           for o, _ in frames[last + 1:last + 4]]
    return np.array(cells), np.stack(xyz, axis=1).astype(np.float64)


def write_gro(w, positions, cell, target):
    """The last frame of a DCD as a .gro with the names of popc.gro."""
    lines = open(os.path.join(w, "popc.gro")).read().splitlines()
    count = int(lines[1])
    out = ["the end of the equilibration of MDIR", f"{count:5d}"]
    for line, x in zip(lines[2:2 + count], positions / 10.0):
        out.append(f"{line[:20]}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}")
    out.append("".join(f"{v / 10.0:10.5f}" for v in cell))
    write(os.path.join(w, target), "\n".join(out) + "\n")


def write_inpcrd(w, positions, cell, target):
    """The same frame as Amber coordinates, which MDIR reads with
    popc.prmtop."""
    out = ["the end of the equilibration of MDIR", f"{len(positions):6d}"]
    values = positions.ravel()
    for k in range(0, len(values), 6):
        out.append("".join(f"{v:12.7f}" for v in values[k:k + 6]))
    out.append("".join(f"{v:12.7f}" for v in list(cell) + [90.0] * 3))
    write(os.path.join(w, target), "\n".join(out) + "\n")


def gromacs_run(args, w, name, steps):
    write(os.path.join(w, f"{name}.mdp"), MDP_MODEL + f"""integrator = md
dt = 0.002
nsteps = {steps}
constraints = h-bonds
constraint-algorithm = lincs
tcoupl = V-rescale
tc-grps = System
tau-t = 1.0
ref-t = 303
nsttcouple = 25
pcoupl = C-rescale
pcoupltype = semiisotropic
tau-p = 5.0
ref-p = 1.0 1.0
compressibility = 4.5e-5 4.5e-5
nstpcouple = 25
gen-vel = yes
gen-temp = 303
gen-seed = 41
nstcalcenergy = 25
nstenergy = 10000
nstlog = 50000
""")
    run([args.gmx, "grompp", "-f", f"{name}.mdp", "-c", "start.gro", "-p",
         "popc.top", "-o", f"{name}.tpr", "-maxwarn", "2"], w,
        f"{name}-grompp.log")
    return [args.gmx, "mdrun", "-deffnm", name, "-ntmpi", "1", "-ntomp",
            "16", "-nb", "gpu", "-pme", "gpu", "-update", "gpu"]


def terms(args, w):
    cells, x = read_dcd(os.path.join(w, "equil.dcd"))
    write_gro(w, x, cells[-1], "start.gro")
    # The rerun and MDIR take the positions as the .gro has them, rounded to
    # 0.01 Å, so that both evaluate the same configuration.
    lines = open(os.path.join(w, "start.gro")).read().splitlines()
    rounded = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])]
                        for l in lines[2:2 + len(x)]]) * 10.0
    write_inpcrd(w, rounded, cells[-1], "start.inpcrd")
    write(os.path.join(w, "terms.mdp"), MDP_MODEL + """integrator = md
nsteps = 0
constraints = none
nstcalcenergy = 1
nstenergy = 1
""")
    run([args.gmx, "grompp", "-f", "terms.mdp", "-c", "start.gro", "-p",
         "popc.top", "-o", "terms.tpr", "-maxwarn", "2"], w,
        "terms-grompp.log")
    run([args.gmx, "mdrun", "-s", "terms.tpr", "-rerun", "start.gro",
         "-deffnm", "terms", "-ntmpi", "1", "-ntomp", "16", "-nb", "cpu"], w,
        "terms-mdrun.log", {"GMX_NBNXN_EWALD_TABLE": "1"})
    names = ["Bond", "Angle", "Proper-Dih.", "Per.-Imp.-Dih.",
             "Improper-Dih.", "LJ-14",
             "Coulomb-14", "LJ-(SR)", "Disper.-corr.", "Coulomb-(SR)",
             "Coul.-recip.", "Potential"]
    with open(os.path.join(w, "terms-energy.log"), "w") as out:
        subprocess.run([args.gmx, "energy", "-f", "terms.edr", "-o",
                        "terms.xvg"], cwd=w, stdout=out,
                       stderr=subprocess.STDOUT, check=True,
                       input="\n".join(names) + "\n\n", text=True)
    log = open(os.path.join(w, "terms.log")).read()
    grid = re.search(r"fourier-nx\s*=\s*([0-9]+).*?fourier-ny\s*=\s*([0-9]+)"
                     r".*?fourier-nz\s*=\s*([0-9]+)", log, re.S)
    nx, ny, nz = (int(v) for v in grid.groups())
    write(os.path.join(w, "terms.toml"), f"""[input]
topology    = "popc.prmtop"
coordinates = "start.inpcrd"

[output]
energy_interval = 0

[energy]
cutoff            = 10.0
pairlist_distance = 11.0
electrostatics    = "PME"

[pme]
tolerance = 1e-05
order     = 4
grid      = [{nx}, {ny}, {nz}]

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = 0

[ensemble]
ensemble    = "NVE"
temperature = 303.0

[constraints]
rigid_water = true

[boundary]
type = "PERIODIC"

[execution]
target    = "CPU"
precision = "DOUBLE"
threads   = 16
""")
    run([args.mdir, "run", "terms.toml"], w, "terms-mdir.log")
    ours = {}
    for line in open(os.path.join(w, "terms-mdir.log")):
        m = re.match(r"MDIR:   (.+?)\s+(-?[0-9.]+)$", line.rstrip())
        if m:
            ours[m.group(1)] = float(m.group(2))
    legends, values = [], None
    for line in open(os.path.join(w, "terms.xvg")):
        if line.startswith("@ s") and "legend" in line:
            legends.append(line.split('"')[1])
        elif not line.startswith(("#", "@")):
            values = [float(v) for v in line.split()[1:]]
    theirs = {k: v / 4.184 for k, v in zip(legends, values)}

    def get(*keys):
        return sum(theirs.get(k, 0.0) for k in keys)
    rows = [
        ("bonds", ours.get("bonds"), get("Bond")),
        ("angles", ours.get("angles"), get("Angle")),
        ("dihedrals", ours.get("dihedrals"),
         get("Proper Dih.", "Per. Imp. Dih.", "Improper Dih.")),
        ("Lennard-Jones 1-4", ours.get("Lennard-Jones 1-4"), get("LJ-14")),
        ("Coulomb 1-4", ours.get("Coulomb 1-4"), get("Coulomb-14")),
        ("Lennard-Jones", ours.get("Lennard-Jones"), get("LJ (SR)")),
        ("dispersion", ours.get("dispersion"), get("Disper. corr.")),
        ("Coulomb, all but 1-4",
         sum(ours.get(k, 0.0) for k in ("Coulomb", "Coulomb excluded",
                                        "Coulomb reciprocal", "Coulomb self")),
         get("Coulomb (SR)", "Coul. recip.")),
        ("total", ours.get("total"), get("Potential")),
    ]
    text = [f"{'term':24s} {'MDIR':>16s} {'GROMACS':>16s} {'relative':>10s}"]
    for name, a, b in rows:
        rel = (a - b) / abs(b) if a is not None and b else float("nan")
        text.append(f"{name:24s} {a if a is not None else float('nan'):16.4f}"
                    f" {b:16.4f} {rel:10.1e}")
    write(os.path.join(w, "terms.txt"), "\n".join(text) + "\n")
    print("\n".join(text))


def runs(args, w):
    steps = int(round(args.ns * 1000 / 0.002))
    cells, x = read_dcd(os.path.join(w, "equil.dcd"))
    write_gro(w, x, cells[-1], "start.gro")
    write(os.path.join(w, "prod-mdir.toml"), MDIR_CONTROL.format(
        checkpoint='checkpoint  = "equil.h5"', energy=10000,
        name="prod-mdir", frames=10000, steps=steps, dt=0.002, seed=32))
    gmx = gromacs_run(args, w, "prod-gmx", steps)
    with open(os.path.join(w, "prod-mdir.log"), "w") as a, \
            open(os.path.join(w, "prod-gmx.out"), "w") as b:
        ours = subprocess.Popen(
            [args.mdir, "run", "prod-mdir.toml"], cwd=w, stdout=a,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.mdir_gpu))
        theirs = subprocess.Popen(
            gmx, cwd=w, stdout=b, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gmx_gpu))
        if ours.wait() != 0 or theirs.wait() != 0:
            sys.exit("a production run failed; see prod-mdir.log and "
                     "prod-gmx.log")


def gromacs_cells(args, w):
    with open(os.path.join(w, "prod-gmx-energy.log"), "w") as out:
        subprocess.run([args.gmx, "energy", "-f", "prod-gmx.edr", "-o",
                        "prod-gmx-box.xvg"], cwd=w, stdout=out,
                       stderr=subprocess.STDOUT, check=True,
                       input="Box-X\nBox-Y\nBox-Z\n\n", text=True)
    rows = [[float(v) for v in line.split()]
            for line in open(os.path.join(w, "prod-gmx-box.xvg"))
            if not line.startswith(("#", "@"))]
    rows = np.array(rows)
    return rows[:, 0], rows[:, 1:4] * 10.0


def estimate(x):
    g = inefficiency(x)
    return x.mean(), x.std(ddof=1) * math.sqrt(g / len(x)), g


def analyze(args, w):
    cells, _ = read_dcd(os.path.join(w, "prod-mdir.dcd"))
    times = np.arange(1, len(cells) + 1) * 20.0
    sets = {"MDIR": (times, cells)}
    if os.path.exists(os.path.join(w, "prod-gmx.edr")):
        sets["GROMACS"] = gromacs_cells(args, w)
    kT = 1.380649e-23 * 303.0
    print(f"{'':8s} {'frames':>6s} {'area per lipid, Å²':>22s} "
          f"{'height, Å':>18s} {'volume, nm³':>20s} {'sd of A, Å²':>12s} "
          f"{'K_A, mN/m':>14s}")
    for label, (t, c) in sets.items():
        keep = t > 1000.0
        c = c[keep]
        area = c[:, 0] * c[:, 1]
        apl = area / LIPIDS_PER_LEAFLET
        a, ae, g = estimate(apl)
        h, he, _ = estimate(c[:, 2])
        v, ve, _ = estimate(np.prod(c, axis=1) / 1000.0)
        sd = area.std(ddof=1)
        n = len(area) / g
        modulus = kT * area.mean() / area.var(ddof=1) * 1e20 * 1e3
        modulus_error = modulus * math.sqrt(2.0 / max(n - 1, 1))
        print(f"{label:8s} {len(c):6d} {a:12.2f} ± {ae:5.2f} "
              f"{h:10.2f} ± {he:4.2f} {v:12.2f} ± {ve:4.2f} {sd:12.1f} "
              f"{modulus:8.0f} ± {modulus_error:3.0f}   (g {g:.1f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("--mdir", required=True)
    parser.add_argument("--gmx", required=True)
    parser.add_argument("--ns", type=float, default=20.0)
    parser.add_argument("--mdir-gpu", default="0")
    parser.add_argument("--gmx-gpu", default="1")
    parser.add_argument("--phases", default="build,equil,terms,runs,analyze")
    args = parser.parse_args()
    w = os.path.abspath(args.work)
    os.makedirs(w, exist_ok=True)
    phases = args.phases.split(",")
    if "build" in phases:
        build(w)
    if "equil" in phases:
        equilibrate(args, w)
    if "terms" in phases:
        terms(args, w)
    if "runs" in phases:
        runs(args, w)
    if "analyze" in phases:
        analyze(args, w)


if __name__ == "__main__":
    main()
