#!/usr/bin/env python3
"""Runs MDIR on a box of OPC water for the tests of the ensembles that
analyze.py makes, and GROMACS on the same box for its density.

    scripts/validation/ensembles/run.py WORK --mdir MDIR [--gmx GMX]
        [--ns 5] [--tleap TLEAP] [--parmed-python PYTHON]

WORK receives the box, made by tleap from leaprc.water.opc (1039 waters,
solvateBox with a buffer of 14.5 Å), and the logs:

  equil.log      NPT, 300 K, 1 atm, 200 ps, from the box of tleap
  nvt-300.log    NVT, 300 K, from the end of equil
  nvt-306.log    NVT, 306 K, from the same state
  npt-1.log      NPT, 300 K, 1 atm, from the same state
  npt-300.log    NPT, 300 K, 300 atm, from the same state
  gmx-npt.xvg    GROMACS with its own OPC (amber19sb.ff/opc.itp), whose
                 update runs on the CPU (virtual sites): volume,
                 density, and temperature of a run at 300 K and 1 atm of the
                 same length, after 200 ps of equilibration from the box

MDIR runs in mixed precision on the GPU with groups and the dual list,
SETTLE (M-SHAKE below double precision), particle mesh Ewald, a cutoff of
9 Å with the correction for the dispersion, 2 fs, stochastic velocity
rescaling and cell rescaling every 25 steps (tau_T 1 ps, tau_P 2 ps).
"""
import argparse
import os
import shutil
import subprocess
import sys

LEAP = """source leaprc.water.opc
m = copy OPC
solvateBox m OPCBOX 14.5 iso
saveAmberParm m opc.prmtop opc.inpcrd
quit
"""

CONTROL = """[input]
topology    = "opc.prmtop"
coordinates = "opc.inpcrd"
{checkpoint_in}
[output]
energy_interval = {energy_interval}
checkpoint = "{name}.h5"
checkpoint_interval = {steps}

[energy]
cutoff            = 9.0
pairlist_distance = 12.0
pruned_distance   = 9.6
electrostatics    = "PME"

[pme]
tolerance   = 1e-05
max_spacing = 1.0

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {steps}
seed       = {seed}

[ensemble]
ensemble    = "{ensemble}"
temperature = {temperature}
{pressure}
[thermostat]
method        = "V-RESCALE"
time_constant = 1.0
interval      = 25
{barostat}
[constraints]
rigid_water = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
neighbor_structure = "GROUPS"
"""

BAROSTAT = """
[barostat]
method        = "C-RESCALE"
time_constant = 2.0
interval      = 25
"""

MDP = """integrator      = md
dt              = 0.002
nsteps          = {steps}
nstcalcenergy   = 25
nstenergy       = 100
nstlog          = 50000
cutoff-scheme   = Verlet
rlist           = 0.9
rcoulomb        = 0.9
rvdw            = 0.9
coulombtype     = PME
fourierspacing  = 0.1
ewald-rtol      = 1e-5
vdw-modifier    = None
DispCorr        = EnerPres
tcoupl          = V-rescale
tc-grps         = System
tau-t           = 1.0
ref-t           = 300
pcoupl          = C-rescale
pcoupltype      = isotropic
tau-p           = 2.0
ref-p           = 1.01325
compressibility = 4.5e-5
nsttcouple      = 25
nstpcouple      = 25
constraints     = h-bonds
continuation    = yes
gen-vel         = no
"""


def run(command, cwd, log):
    with open(os.path.join(cwd, log), "w") as out:
        result = subprocess.run(command, cwd=cwd, stdout=out,
                                stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"{' '.join(command)} failed; see {log}")


COUPLINGS = {
    # Semi-isotropic coupling (D119): the volume, not the shape, of a
    # liquid has a distribution, so the test of two pressures holds for it;
    # with the height held the area takes the change of the volume.
    "semi": 'coupling      = "SEMI_ISOTROPIC"\n',
    "held": 'coupling      = "SEMI_ISOTROPIC"\ncompressibility_z = 0.0\n',
}


def mdir_run(args, name, ensemble, temperature, pressure, steps, seed,
             checkpoint=None, energy_interval=50, coupling=""):
    text = CONTROL.format(
        checkpoint_in=f'checkpoint  = "{checkpoint}"\n' if checkpoint else "",
        energy_interval=energy_interval, name=name, steps=steps, seed=seed,
        ensemble=ensemble, temperature=temperature,
        pressure=f"pressure    = {pressure}\n" if pressure else "",
        barostat=BAROSTAT + coupling if ensemble == "NPT" else "")
    path = os.path.join(args.work, name + ".toml")
    with open(path, "w") as file:
        file.write(text)
    run([args.mdir, "run", name + ".toml"], args.work, name + ".log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("--mdir", required=True)
    parser.add_argument("--gmx")
    parser.add_argument("--tleap", default="tleap")
    parser.add_argument("--ns", type=float, default=5.0)
    parser.add_argument("--equil-steps", type=int, default=100000)
    parser.add_argument("--gmx-only", action="store_true",
                        help="run GROMACS alone, on the box already made")
    parser.add_argument("--semi-only", default="",
                        help="run the two pressures with the couplings "
                        "named (semi, held; comma-separated) from equil.h5")
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)
    steps = int(round(args.ns * 500000))
    if args.gmx_only:
        gromacs(args, steps)
        return
    if args.semi_only:
        for tag in args.semi_only.split(","):
            coupling = COUPLINGS[tag]
            mdir_run(args, f"npt-1-{tag}", "NPT", 300.0, 1.0, steps, 41,
                     "equil.h5", coupling=coupling)
            mdir_run(args, f"npt-300-{tag}", "NPT", 300.0, 300.0, steps, 42,
                     "equil.h5", coupling=coupling)
        return
    with open(os.path.join(args.work, "box.leap"), "w") as file:
        file.write(LEAP)
    run([args.tleap, "-f", "box.leap"], args.work, "tleap.log")

    mdir_run(args, "equil", "NPT", 300.0, 1.0, args.equil_steps, 11)
    start = "equil.h5"
    mdir_run(args, "nvt-300", "NVT", 300.0, None, steps, 21, start)
    mdir_run(args, "nvt-306", "NVT", 306.0, None, steps, 22, start)
    mdir_run(args, "npt-1", "NPT", 300.0, 1.0, steps, 31, start)
    mdir_run(args, "npt-300", "NPT", 300.0, 300.0, steps, 32, start)

    if args.gmx:
        gromacs(args, steps)


def write_gro(work):
    """opc.gro from opc.inpcrd: the waters of tleap (O, H1, H2, EPW) as the
    SOL of GROMACS (OW, HW1, HW2, MW), in nm."""
    lines = open(os.path.join(work, "opc.inpcrd")).read().split("\n")
    count = int(lines[1].split()[0])
    numbers = []
    for line in lines[2:]:
        numbers += [float(line[i:i + 12]) for i in range(0, len(line), 12)
                    if line[i:i + 12].strip()]
    xyz = numbers[:3 * count]
    box = numbers[3 * count:3 * count + 3]
    names = ["OW", "HW1", "HW2", "MW"]
    out = ["OPC water from tleap", f"{count:5d}"]
    for i in range(count):
        residue = i // 4 + 1
        x, y, z = (v / 10.0 for v in xyz[3 * i:3 * i + 3])
        out.append(f"{residue % 100000:5d}{'SOL':<5}{names[i % 4]:>5}"
                   f"{(i + 1) % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}")
    out.append(f"{box[0] / 10:10.5f}{box[1] / 10:10.5f}{box[2] / 10:10.5f}")
    with open(os.path.join(work, "opc.gro"), "w") as file:
        file.write("\n".join(out) + "\n")
    return count // 4


def gromacs(args, steps):
    """The same box with the OPC of GROMACS (amber19sb.ff/opc.itp): an
    equilibration of 200 ps at constant pressure from the box of tleap,
    and a run of the length of those of MDIR from it."""
    waters = write_gro(args.work)
    with open(os.path.join(args.work, "topol.top"), "w") as file:
        file.write('#include "amber19sb.ff/forcefield.itp"\n'
                   '#include "amber19sb.ff/opc.itp"\n\n'
                   "[ system ]\nOPC water\n\n"
                   f"[ molecules ]\nSOL {waters}\n")
    equil = MDP.format(steps=100000).replace(
        "continuation    = yes\ngen-vel         = no",
        "continuation    = no\ngen-vel         = yes\ngen-temp        = 300\n"
        "gen-seed        = 41")
    with open(os.path.join(args.work, "gmx-equil.mdp"), "w") as file:
        file.write(equil)
    with open(os.path.join(args.work, "gmx-npt.mdp"), "w") as file:
        file.write(MDP.format(steps=steps))
    gmx = [args.gmx]
    run(gmx + ["grompp", "-f", "gmx-equil.mdp", "-c", "opc.gro", "-p",
               "topol.top", "-o", "gmx-equil.tpr", "-maxwarn", "1"],
        args.work, "gmx-equil-grompp.log")
    run(gmx + ["mdrun", "-deffnm", "gmx-equil", "-ntmpi", "1", "-nb", "gpu",
               "-pme", "gpu", "-update", "cpu"], args.work, "gmx-equil.out")
    run(gmx + ["grompp", "-f", "gmx-npt.mdp", "-c", "gmx-equil.gro", "-t",
               "gmx-equil.cpt", "-p", "topol.top", "-o", "gmx-npt.tpr"],
        args.work, "gmx-npt-grompp.log")
    run(gmx + ["mdrun", "-deffnm", "gmx-npt", "-ntmpi", "1", "-nb", "gpu",
               "-pme", "gpu", "-update", "cpu"], args.work, "gmx-npt.out")
    with open(os.path.join(args.work, "gmx-energy.log"), "w") as out:
        subprocess.run(gmx + ["energy", "-f", "gmx-npt.edr", "-o",
                              "gmx-npt.xvg"], cwd=args.work,
                       input="Volume\nDensity\nTemperature\n\n", text=True,
                       stdout=out, stderr=subprocess.STDOUT, check=True)

if __name__ == "__main__":
    main()
