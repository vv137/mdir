#!/bin/bash
# Compares the forces of MDIR with those of sander and GROMACS on the inputs
# of test/Driver (scripts/validation/forces/compare.py):
#
#   - alanine dipeptide in TIP3P, ff14SB, a plain cutoff of 9 Å, against
#     sander (Inputs/dipeptide);
#   - the dipeptide in OPC with particle mesh Ewald of the same beta, grid,
#     and order, against sander (Inputs/opc, as amber-pme.test);
#   - the dipeptide in TIP3P from its topology for GROMACS, with particle
#     mesh Ewald of the same beta, grid, and order, against GROMACS
#     (Inputs/dipeptide/dipeptide.top) in mixed precision, which computes
#     again at the positions of MDIR.
#
# For sander, the charges of the topology that MDIR reads are scaled by
# sqrt(332.0522173 / 332.0637133), so that its Coulomb constant (CODATA
# 2018) gives the energies of Amber's, and sander keeps the net force of
# particle mesh Ewald (netfrc=0), which it removes by default.
#
#   scripts/validation/forces/run.sh WORK MDIR ENGINES
#
# ENGINES is the prefix of AmberTools and GROMACS (bin/sander, bin/gmx,
# bin/python with parmed, scipy, and h5py).
set -e
work=$1; mdir=$2; engines=$3
here=$(cd "$(dirname "$0")" && pwd)
inputs=$here/../../../test/Driver/Inputs
python=$engines/bin/python
mkdir -p "$work"
cd "$work"

scale() {  # scale IN OUT: the charges of a prmtop to Amber's constant
  "$python" - "$1" "$2" <<'PY'
import math, sys, parmed
p = parmed.load_file(sys.argv[1])
factor = math.sqrt(332.0522173 / 332.0637133)
for atom in p.atoms:
    atom.charge *= factor
p.save(sys.argv[2], overwrite=True)
PY
}

mdir_forces() {  # mdir_forces NAME TOPOLOGY COORDINATES EXTRA...
  local name=$1 top=$2 crd=$3; shift 3
  cat > "$name.toml" <<TOML
[input]
topology    = "$top"
coordinates = "$crd"
[output]
energy_interval = 0
checkpoint = "$name.h5"
checkpoint_interval = 1
[energy]
cutoff = 9.0
pairlist_distance = 10.0
$*
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 1e-9
steps      = 1
[ensemble]
ensemble    = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
[execution]
target    = "CPU"
precision = "DOUBLE"
threads   = 8
TOML
  "$mdir" run "$name.toml" > "$name.log"
}

sander_forces() {  # sander_forces NAME PRMTOP INPCRD EWALD
  cat > "$1.in" <<IN
Forces at the input coordinates
 &cntrl
  imin=0, ntx=1, irest=0, tempi=0.0, ig=1, nstlim=1, dt=0.000000001,
  ntb=1, cut=9.0, ntc=1, ntf=1, ntpr=1, ntwx=1, ntwf=1, ioutfm=1, ntt=0,
 /
 &ewald
  $4
 /
IN
  "$engines/bin/sander" -O -i "$1.in" -p "$2" -c "$3" -o "$1.out" \
    -x "$1.nc" -frc "$1.frc.nc" -r "$1.rst"
}

# 1. Plain cutoff, TIP3P.
cp "$inputs/dipeptide/dipeptide.prmtop" "$inputs/dipeptide/dipeptide.inpcrd" .
scale dipeptide.prmtop dipeptide-scaled.prmtop
mdir_forces cutoff dipeptide-scaled.prmtop dipeptide.inpcrd \
  'electrostatics = "CUTOFF"'
sander_forces sander-cutoff dipeptide.prmtop dipeptide.inpcrd \
  "eedmeth=4, vdwmeth=0,"
echo "== plain cutoff, against sander"
"$python" "$here/compare.py" cutoff.h5 sander-cutoff.frc.nc --kind sander

# 2. Particle mesh Ewald, OPC.
cp "$inputs/opc/opc.prmtop" "$inputs/opc/opc.inpcrd" .
scale opc.prmtop opc-scaled.prmtop
mdir_forces pme opc-scaled.prmtop opc.inpcrd 'electrostatics = "PME"
[pme]
beta = 0.347045919371208278
order = 4
influence = "OPTIMAL"
grid = [30, 30, 24]'
sander_forces sander-pme opc.prmtop opc.inpcrd \
  "nfft1=30, nfft2=30, nfft3=24, order=4, ew_coeff=0.347045919371208278, vdwmeth=0, netfrc=0,"
echo "== particle mesh Ewald, against sander"
"$python" "$here/compare.py" pme.h5 sander-pme.frc.nc --kind sander

# 3. Particle mesh Ewald from the topology of GROMACS: both solve
# erfc(beta r_c) = ewald-rtol for beta; the grid is the same.
cp "$inputs/dipeptide/dipeptide.top" "$inputs/dipeptide/dipeptide.gro" .
cat > rerun.mdp <<MDP
integrator      = md
nsteps          = 0
cutoff-scheme   = Verlet
rlist           = 0.9
rcoulomb        = 0.9
rvdw            = 0.9
coulombtype     = PME
coulomb-modifier = None
vdw-modifier    = None
DispCorr        = no
ewald-rtol      = 1e-5
pme-order       = 4
fourier-nx      = 28
fourier-ny      = 30
fourier-nz      = 26
constraints     = none
nstfout         = 1
nstcalcenergy   = 1
nstenergy       = 1
MDP
"$engines/bin/gmx" grompp -f rerun.mdp -c dipeptide.gro -p dipeptide.top \
  -o rerun.tpr -maxwarn 2 > grompp.log 2>&1
"$engines/bin/gmx" mdrun -s rerun.tpr -rerun dipeptide.gro -deffnm rerun \
  -nb cpu -ntmpi 1 -ntomp 8 > mdrun.log 2>&1
echo 0 | "$engines/bin/gmx" traj -f rerun.trr -s rerun.tpr -of gmx-forces.xvg \
  > traj.log 2>&1
mdir_forces gromacs dipeptide.top dipeptide.gro "electrostatics = \"PME\"
[pme]
tolerance = 1e-5
order = 4
grid = [28, 30, 26]
[constraints]
rigid_water = true"
# SETTLE puts the waters of the .gro, rounded to 0.001 nm, at their
# geometry in the step of 1e-9 ps, so GROMACS computes again at the
# positions of MDIR, written with nine decimals.
"$python" - <<'PY'
import numpy as np, h5py
with h5py.File("gromacs.h5") as f:
    x = f["particles/all/position/value"][0]
    ids = f["particles/all/id"][:]
    box = f["particles/all/box/edges"][:]
x = x[np.argsort(ids)]
lines = open("dipeptide.gro").read().split("\n")[2:2 + len(x)]
out = ["TITLE", "positions of MDIR after a step of 1e-9 ps", "END", "POSITION"]
for k, (line, p) in enumerate(zip(lines, x)):
    out.append(f"{int(line[0:5]):5d} {line[5:10].strip():5s} "
               f"{line[10:15].strip():5s}{k + 1:7d}"
               f"{p[0]:15.9f}{p[1]:15.9f}{p[2]:15.9f}")
out += ["END", "BOX", f"{box[0]:15.9f}{box[1]:15.9f}{box[2]:15.9f}", "END"]
open("mdir-x.g96", "w").write("\n".join(out) + "\n")
PY
"$engines/bin/gmx" mdrun -s rerun.tpr -rerun mdir-x.g96 -deffnm at-mdir \
  -nb cpu -ntmpi 1 -ntomp 8 > mdrun-at-mdir.log 2>&1
echo 0 | "$engines/bin/gmx" traj -f at-mdir.trr -s rerun.tpr -of gmx-forces.xvg \
  > traj-at-mdir.log 2>&1
echo "== particle mesh Ewald, against GROMACS"
"$python" "$here/compare.py" gromacs.h5 gmx-forces.xvg --kind gromacs
