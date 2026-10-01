#!/usr/bin/env python3
"""Compares MDIR with GROMACS on a protein in water: ubiquitin (PDB 1UBQ)
with ff19SB in OPC, from the port of GROMACS (amber19sb.ff), which both
programs read.

    scripts/validation/protein/run.py WORK --mdir MDIR --gmx GMX
        [--steps 60000]

GMX is a GROMACS with CUDA. WORK receives:

  - ubq.top and ubq.gro from pdb2gmx (amber19sb, OPC; 1UBQ without its
    waters, the hydrogens added again), editconf (a rectangular cell 1.2
    nm from the protein), and solvate (the four sites of tip4p.gro, which
    SETTLE brings to the geometry of OPC); ubiquitin is neutral;
  - em, equil: GROMACS, a minimization by steepest descent, then NPT at
    300 K and 1 bar for 200 ps, h-bonds and rigid water;
  - terms: the energy terms of the equilibrated state, MDIR on the CPU in
    double precision against GROMACS (a rerun), with the same beta, grid,
    and order of particle mesh Ewald, and the same cutoff of 9 Å;
  - nve-mdir.log, nve-gmx.log, npt-mdir.log, npt-gmx.log: STEPS steps of
    2 fs from the equilibrated state on the device, the rates over the
    second half (GROMACS with -resethway, after its tuning of PME) and the
    conservation of the energy.

Both run with a cutoff of 9 Å, particle mesh Ewald with erfc(beta r_c) =
1e-5 and a spacing of at most 1 Å, the correction for the dispersion,
SHAKE on the bonds of hydrogen and rigid water, mixed precision, and at
constant pressure stochastic velocity and cell rescaling every 25 steps.
GROMACS chooses its own pair list (verlet-buffer-tolerance; nstlist 80
at constant energy, where it does not raise it itself), MDIR the dual
list with an outer reach of 12 Å and an inner one of 9.6 Å, within 0.4%
of the best of 11 to 12 Å and 9.4 to 9.8 Å. GROMACS updates
on the CPU: its update on the GPU does not take virtual sites, which the
four sites of OPC are.
"""
import argparse
import os
import re
import subprocess
import sys

TOP = ""

MDP_COMMON = """cutoff-scheme   = Verlet
rlist           = 0.9
rcoulomb        = 0.9
rvdw            = 0.9
coulombtype     = PME
coulomb-modifier = None
vdw-modifier    = None
DispCorr        = EnerPres
ewald-rtol      = 1e-5
fourierspacing  = 0.1
pme-order       = 4
constraints     = h-bonds
constraint-algorithm = lincs
"""


def run(command, cwd, log, stdin=None, env=None):
    with open(os.path.join(cwd, log), "w") as out:
        result = subprocess.run(command, cwd=cwd, stdout=out, input=stdin,
                                text=True, stderr=subprocess.STDOUT,
                                env=dict(os.environ, **(env or {})))
    if result.returncode != 0:
        sys.exit(f"{' '.join(command)} failed; see {os.path.join(cwd, log)}")


def write(path, text):
    with open(path, "w") as file:
        file.write(text)


def mdir_control(name, steps, ensemble, target, precision, grid=None,
                 energy_interval=1000):
    # beta from erfc(beta r_c) = 1e-5, as GROMACS solves for it; the width
    # that GROMACS logs has six digits, which moves the self term and the
    # excluded pairs by 0.6 kcal/mol here.
    pme = f"\n[pme]\ntolerance = 1e-05\norder = 4\ngrid = {grid}\n" if grid \
        else "\n[pme]\ntolerance = 1e-05\nmax_spacing = 1.0\n"
    couple = ""
    if ensemble == "NPT":
        couple = """
[thermostat]
method        = "V-RESCALE"
time_constant = 1.0
interval      = 25

[barostat]
method        = "C-RESCALE"
time_constant = 2.0
interval      = 25
"""
    return f"""[input]
topology    = "ubq.top"
coordinates = "equil.gro"
format      = "GROMACS"
include_paths = ["{TOP}"]

[output]
energy_interval = {energy_interval}

[energy]
cutoff            = 9.0
pairlist_distance = {'12.0' if target == 'GPU' else '10.0'}
{'pruned_distance   = 9.6' if target == 'GPU' else ''}
electrostatics    = "PME"
{pme}
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {steps}

[ensemble]
ensemble    = "{ensemble}"
temperature = 300.0
{'pressure    = 0.986923' if ensemble == 'NPT' else ''}
{couple}
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "{target}"
precision = "{precision}"
{'neighbor_structure = "GROUPS"' if target == 'GPU' else 'threads = 16'}
"""


def gromacs_top(gmx):
    """The directory of the force fields of GROMACS."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(gmx))),
                        "share", "gromacs", "top")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("--mdir", required=True)
    parser.add_argument("--gmx", required=True)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--pdb", default="1ubq.pdb")
    parser.add_argument("--phases", default="prepare,equil,terms,rates",
                        help="which of prepare, equil, terms, rates to run")
    args = parser.parse_args()
    phases = args.phases.split(",")
    w = args.work
    global TOP
    TOP = gromacs_top(args.gmx)
    os.makedirs(w, exist_ok=True)
    pdb = os.path.join(w, "1ubq.pdb")
    if not os.path.exists(pdb):
        subprocess.run(["curl", "-s", "-o", pdb,
                        "https://files.rcsb.org/download/1UBQ.pdb"],
                       check=True)
    gmx = [args.gmx]
    if "prepare" in phases:
        prepare(w, gmx)
    if "equil" in phases:
        equilibrate(w, gmx)
    if "terms" in phases:
        terms(args, w, gmx)
    if "rates" in phases:
        rates(args, w, gmx)


def prepare(w, gmx):
    lines = [l for l in open(os.path.join(w, "1ubq.pdb"))
             if l.startswith(("ATOM", "TER", "END"))]
    write(os.path.join(w, "1ubq-protein.pdb"), "".join(lines))
    run(gmx + ["pdb2gmx", "-f", "1ubq-protein.pdb", "-o", "protein.gro",
               "-p", "ubq.top", "-ff", "amber19sb", "-water", "opc",
               "-ignh"], w, "pdb2gmx.log")
    run(gmx + ["editconf", "-f", "protein.gro", "-o", "box.gro", "-bt",
               "triclinic", "-d", "1.2"], w, "editconf.log")
    run(gmx + ["solvate", "-cp", "box.gro", "-cs", "tip4p.gro", "-o",
               "ubq.gro", "-p", "ubq.top"], w, "solvate.log")


def equilibrate(w, gmx):
    """Equilibration by GROMACS, from the box of solvate after a
    minimization by steepest descent, which removes its close contacts."""
    write(os.path.join(w, "em.mdp"), MDP_COMMON.replace(
        "constraints     = h-bonds", "constraints     = none") + """integrator = steep
nsteps = 5000
emtol = 500
emstep = 0.01
""")
    run(gmx + ["grompp", "-f", "em.mdp", "-c", "ubq.gro", "-p", "ubq.top",
               "-o", "em.tpr", "-maxwarn", "2"], w, "em-grompp.log")
    run(gmx + ["mdrun", "-deffnm", "em", "-ntmpi", "1", "-nb", "gpu"], w,
        "em-mdrun.log")
    write(os.path.join(w, "equil.mdp"), MDP_COMMON + """integrator = md
dt = 0.002
nsteps = 100000
tcoupl = V-rescale
tc-grps = System
tau-t = 1.0
ref-t = 300
pcoupl = C-rescale
tau-p = 2.0
ref-p = 1.0
compressibility = 4.5e-5
gen-vel = yes
gen-temp = 300
gen-seed = 7
nstcalcenergy = 100
nstenergy = 1000
nstxout-compressed = 0
""")
    run(gmx + ["grompp", "-f", "equil.mdp", "-c", "em.gro", "-p",
               "ubq.top", "-o", "equil.tpr", "-maxwarn", "2"], w,
        "equil-grompp.log")
    run(gmx + ["mdrun", "-deffnm", "equil", "-ntmpi", "1", "-nb", "gpu",
               "-pme", "gpu", "-update", "cpu"], w, "equil-mdrun.log")


def terms(args, w, gmx):
    # The terms of the equilibrated state: GROMACS by a rerun, MDIR on the
    # CPU in double precision with the beta and the grid of GROMACS. The
    # sites of solvate are those of TIP4P, which a rerun of GROMACS keeps
    # and MDIR places again from the weights of OPC; after the dynamics of
    # GROMACS they are those of OPC.
    write(os.path.join(w, "terms.mdp"), MDP_COMMON.replace(
        "constraints     = h-bonds", "constraints     = none") + """integrator = md
nsteps = 0
nstcalcenergy = 1
nstenergy = 1
""")
    run(gmx + ["grompp", "-f", "terms.mdp", "-c", "equil.gro", "-p",
               "ubq.top", "-o", "terms.tpr", "-maxwarn", "2"], w,
        "terms-grompp.log")
    # A rerun takes the sites as the file has them, rounded to 0.001 nm;
    # MDIR places them again from their atoms. GROMACS gets them placed
    # the same way, from the weights of the OPC of amber19sb.ff.
    write_sites(w, "equil.gro", "equil-sites.g96", 0.14772234, 0.14772234)
    # The SIMD kernels of GROMACS take the Ewald correction from an
    # analytical approximation in single precision, whose error is the same
    # for the excluded pairs of every rigid water and adds up (0.55 kcal/mol
    # over 5700 waters); a second rerun with tables of the correction shows
    # it.
    names = ["Bond", "Angle", "Proper-Dih.", "Per.-Imp.-Dih.",
             "Improper-Dih.", "CMAP-Dih.", "LJ-14", "Coulomb-14", "LJ-(SR)",
             "Disper.-corr.", "Coulomb-(SR)", "Coul.-recip.", "Potential"]
    for name, env in (("terms", {}),
                      ("terms-table", {"GMX_NBNXN_EWALD_TABLE": "1"})):
        run(gmx + ["mdrun", "-s", "terms.tpr", "-rerun", "equil-sites.g96",
                   "-deffnm", name, "-ntmpi", "1", "-nb", "cpu"], w,
            name + "-mdrun.log", env=env)
        run(gmx + ["energy", "-f", name + ".edr", "-o", name + ".xvg"], w,
            name + "-energy.log", stdin="\n".join(names) + "\n\n")
    log = open(os.path.join(w, "terms.log")).read()
    grid = re.search(r"grid ([0-9]+) ([0-9]+) ([0-9]+)", log)
    if grid is None:
        grid = re.search(r"fourier-nx\s*=\s*([0-9]+).*?fourier-ny\s*=\s*([0-9]+)"
                         r".*?fourier-nz\s*=\s*([0-9]+)", log, re.S)
    nx, ny, nz = (int(v) for v in grid.groups())
    control = mdir_control("terms", 0, "NVE", "CPU", "DOUBLE",
                           grid=f"[{nx}, {ny}, {nz}]",
                           energy_interval=0).replace(
        "hydrogen_bonds = true", "hydrogen_bonds = false")
    write(os.path.join(w, "terms.toml"), control)
    run([args.mdir, "run", "terms.toml"], w, "terms-mdir.log")
    compare_terms(w)


def rates(args, w, gmx):
    """MDIR and GROMACS on the device, from the equilibrated state."""
    for ensemble in ("NVE", "NPT"):
        name = ensemble.lower()
        write(os.path.join(w, f"{name}-mdir.toml"),
              mdir_control(name, args.steps, ensemble, "GPU", "MIXED"))
        run([args.mdir, "run", f"{name}-mdir.toml"], w, f"{name}-mdir.log")
        couple = ("tcoupl = V-rescale\ntc-grps = System\ntau-t = 1.0\n"
                  "ref-t = 300\npcoupl = C-rescale\ntau-p = 2.0\n"
                  "ref-p = 1.0\ncompressibility = 4.5e-5\nnsttcouple = 25\n"
                  "nstpcouple = 25\n" if ensemble == "NPT" else
                  # GROMACS keeps nstlist at 10 at constant energy (its
                  # rate falls from 806 to 690 ns/day); 80 is what it
                  # chooses at constant pressure here.
                  "tcoupl = no\npcoupl = no\nnstlist = 80\n")
        write(os.path.join(w, f"{name}-gmx.mdp"), MDP_COMMON + f"""integrator = md
dt = 0.002
nsteps = {args.steps}
continuation = yes
nstcalcenergy = 100
nstenergy = {args.steps // 10}
nstlog = {args.steps // 10}
{couple}""")
        run(gmx + ["grompp", "-f", f"{name}-gmx.mdp", "-c", "equil.gro", "-t",
                   "equil.cpt", "-p", "ubq.top", "-o", f"{name}-gmx.tpr",
                   "-maxwarn", "2"], w, f"{name}-grompp.log")
        run(gmx + ["mdrun", "-deffnm", f"{name}-gmx", "-ntmpi", "1", "-nb",
                   "gpu", "-pme", "gpu", "-update", "cpu", "-resethway"], w,
            f"{name}-gmx.out")
        ours = re.search(r"from step [0-9]+ to step [0-9]+, [0-9.]+ ms per step, "
                         r"([0-9.]+) ns per day",
                         open(os.path.join(w, f"{name}-mdir.log")).read())
        theirs = re.search(r"Performance:\s+([0-9.]+)",
                           open(os.path.join(w, f"{name}-gmx.log")).read())
        changed = re.search(r"energy changed by ([0-9.e+-]+)",
                            open(os.path.join(w, f"{name}-mdir.log")).read())
        print(f"{ensemble}: MDIR {ours.group(1)} ns/day, GROMACS "
              f"{theirs.group(1)} ns/day; MDIR's energy changed by "
              f"{changed.group(1) if changed else '?'}")


def write_sites(w, source, target, a, b):
    """`source` as a .g96 of nine decimals with each site MW placed from
    the three atoms before it, MW = O + a (H1 - O) + b (H2 - O)."""
    lines = open(os.path.join(w, source)).read().split("\n")
    count = int(lines[1])
    atoms = lines[2:2 + count]
    xyz = [[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in atoms]
    names = [l[10:15].strip() for l in atoms]
    for i, name in enumerate(names):
        if name == "MW":
            o, h1, h2 = xyz[i - 3], xyz[i - 2], xyz[i - 1]
            xyz[i] = [o[k] + a * (h1[k] - o[k]) + b * (h2[k] - o[k])
                      for k in range(3)]
    out = ["TITLE", f"{source} with its sites placed from their atoms",
           "END", "POSITION"]
    for i, line in enumerate(atoms):
        out.append(f"{int(line[0:5]):5d} {line[5:10].strip():5s} "
                   f"{names[i]:5s}{i + 1:7d}{xyz[i][0]:15.9f}"
                   f"{xyz[i][1]:15.9f}{xyz[i][2]:15.9f}")
    box = [float(v) for v in lines[2 + count].split()[:3]]
    out += ["END", "BOX", f"{box[0]:15.9f}{box[1]:15.9f}{box[2]:15.9f}",
            "END"]
    write(os.path.join(w, target), "\n".join(out) + "\n")


def read_xvg(path):
    """The last frame of an energy file of GROMACS, in kcal/mol."""
    legends, values = [], None
    for line in open(path):
        if line.startswith("@ s") and "legend" in line:
            legends.append(line.split('"')[1])
        elif not line.startswith(("#", "@")):
            values = [float(v) for v in line.split()[1:]]
    return {k: v / 4.184 for k, v in zip(legends, values)}


def compare_terms(w):
    """The terms of MDIR (kcal/mol) against those of GROMACS (kJ/mol), with
    its SIMD kernels and with tables of the Ewald correction."""
    ours = {}
    for line in open(os.path.join(w, "terms-mdir.log")):
        m = re.match(r"MDIR:   (.+?)\s+(-?[0-9.]+)$", line.rstrip())
        if m:
            ours[m.group(1)] = float(m.group(2))
    print(f"{'term':24s} {'MDIR':>16s} {'GROMACS':>16s} {'relative':>10s} "
          f"{'with tables':>16s} {'relative':>10s}")
    simd = rows_of(ours, read_xvg(os.path.join(w, "terms.xvg")))
    table = rows_of(ours, read_xvg(os.path.join(w, "terms-table.xvg")))
    for (name, a, b), (_, _, c) in zip(simd, table):
        if a is None:
            continue
        rel = (a - b) / abs(b) if b else float("nan")
        rel2 = (a - c) / abs(c) if c else float("nan")
        print(f"{name:24s} {a:16.4f} {b:16.4f} {rel:10.1e} {c:16.4f} "
              f"{rel2:10.1e}")


def rows_of(ours, theirs):
    def get(*keys):
        return sum(theirs.get(k, 0.0) for k in keys)
    return [
        ("bonds", ours.get("bonds"), get("Bond")),
        ("angles", ours.get("angles"), get("Angle")),
        ("dihedrals", ours.get("dihedrals"),
         get("Proper Dih.", "Per. Imp. Dih.", "Improper Dih.")),
        ("CMAP", ours.get("CMAP"), get("CMAP Dih.")),
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


if __name__ == "__main__":
    main()
