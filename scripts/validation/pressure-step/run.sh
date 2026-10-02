#!/usr/bin/env bash
# The pressure of one cell of TIP3P water in MDIR, GROMACS, and pmemd.cuda
# against the step: 2972 waters in a cube of 44.87 Å that GROMACS
# equilibrates at 303 K and 1 bar, then 0.5 ns at 2, 1, 0.5, and 0.25 fs
# in that cell in each program, from the same positions, with a cutoff of
# 10 Å, PME (erfc(beta r_c) = 1e-5), the correction for the dispersion,
# and rigid water. MDIR holds the cell with semi-isotropic coupling of a
# compressibility of 1e-12 and reports the pressures it took; pmemd.cuda
# with Berendsen's barostat of a time constant of 1e6 ps, which prints
# the pressure. The estimators of the pressure of the integrators differ
# at 2 fs and agree as the step goes to zero (D119).
#
#   run.sh WORK MDIR GMX PMEMD          (AmberTools on PATH)
set -euo pipefail
work=$1 mdir=$2 gmx=$3 pmemd=$4
top=$(dirname "$(dirname "$(readlink -f "$gmx")")")/share/gromacs/top
mkdir -p "$work" && cd "$work"
"$gmx" solvate -cs spc216.gro -box 4.5 4.5 4.5 -o water.gro > solvate.log 2>&1
n=$(( $(sed -n 2p water.gro) / 3 ))
printf '#include "amber99sb.ff/forcefield.itp"\n#include "amber99sb.ff/tip3p.itp"\n\n[ system ]\nTIP3P water\n\n[ molecules ]\nSOL %d\n' "$n" > water.top
cat > common.mdp <<'MDP'
cutoff-scheme = Verlet
rcoulomb = 1.0
rvdw = 1.0
coulombtype = PME
coulomb-modifier = None
vdw-modifier = None
DispCorr = EnerPres
ewald-rtol = 1e-5
fourierspacing = 0.1
pme-order = 4
tcoupl = V-rescale
tc-grps = System
tau-t = 1.0
ref-t = 303
nsttcouple = 25
nstcalcenergy = 25
nstenergy = 25
MDP
(grep -v -E '^(tcoupl|tc-grps|tau-t|ref-t|nsttcouple)' common.mdp; printf 'integrator = steep\nnsteps = 2000\nemtol = 500\n') > em.mdp
(cat common.mdp; printf 'integrator = md\ndt = 0.002\nnsteps = 25000\npcoupl = C-rescale\ntau-p = 2.0\nref-p = 1.0\ncompressibility = 4.5e-5\nnstpcouple = 25\ngen-vel = yes\ngen-temp = 303\n') > equil.mdp
"$gmx" grompp -f em.mdp -c water.gro -p water.top -o em.tpr -maxwarn 2 > em.log 2>&1
"$gmx" mdrun -deffnm em -ntmpi 1 > em.out 2>&1
"$gmx" grompp -f equil.mdp -c em.gro -p water.top -o equil.tpr -maxwarn 2 > equil-grompp.log 2>&1
"$gmx" mdrun -deffnm equil -ntmpi 1 -nb gpu -pme gpu -update gpu > equil.out 2>&1
edge=$(tail -1 equil.gro | awk '{printf "%.4f", $1 * 10}')
# The same waters for tleap, with the names of Amber.
awk -v n="$(sed -n 2p equil.gro)" 'NR > 2 && NR <= n + 2 {
  name = substr($0, 11, 5); gsub(/ /, "", name)
  name = (name == "OW") ? "O" : (name == "HW1") ? "H1" : "H2"
  i = NR - 2; printf "ATOM  %5d %-4s WAT %5d    %8.3f%8.3f%8.3f  1.00  0.00\n", i % 100000, name, int((i - 1) / 3) + 1 % 10000, substr($0, 21, 8) * 10, substr($0, 29, 8) * 10, substr($0, 37, 8) * 10
  if (name == "H2") print "TER" }' equil.gro > water.pdb
printf 'source leaprc.water.tip3p\nw = loadpdb water.pdb\nset w box {%s %s %s}\nsaveamberparm w water.prmtop water.rst7\nquit\n' "$edge" "$edge" "$edge" > water.leap
tleap -f water.leap > leap.log 2>&1
for dt in 0.002 0.001 0.0005 0.00025; do
  steps=$(python3 -c "print(round(0.5 / $dt * 1000))")
  tag=${dt#0.}
  (cat common.mdp; printf 'integrator = md\ndt = %s\nnsteps = %d\npcoupl = no\ncontinuation = yes\n' "$dt" "$steps") > fixed-$tag.mdp
  "$gmx" grompp -f fixed-$tag.mdp -c equil.gro -t equil.cpt -p water.top -o fixed-$tag.tpr -maxwarn 2 > fixed-$tag-grompp.log 2>&1
  "$gmx" mdrun -deffnm fixed-$tag -ntmpi 1 -nb gpu -pme gpu -update gpu > fixed-$tag.out 2>&1
  printf 'Pressure\n\n' | "$gmx" energy -f fixed-$tag.edr -o pressure-$tag.xvg > /dev/null 2>&1
  cat > fixed-mdir-$tag.toml <<TOML
[input]
topology    = "water.top"
coordinates = "equil.gro"
format      = "GROMACS"
include_paths = ["$top"]

[output]
energy_interval = 5000

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
time_step  = $dt
steps      = $steps

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
time_constant = 2.0
interval      = 25
coupling      = "SEMI_ISOTROPIC"
compressibility = 1.0e-12
compressibility_z = 0.0

[constraints]
rigid_water = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
neighbor_structure = "GROUPS"
TOML
  "$mdir" run fixed-mdir-$tag.toml > fixed-mdir-$tag.log 2>&1
  cat > fixed-amber-$tag.in <<IN
TIP3P in one cell; Berendsen's barostat of a time constant of 1e6 ps prints the pressure
 &cntrl
  imin=0, irest=0, ntx=1, tempi=303.0, ig=11, nstlim=$steps, dt=$dt,
  ntc=2, ntf=2, cut=10.0, ntpr=$(python3 -c "print(round(0.05 / $dt))"), ntwx=0, ntwr=$steps,
  ntb=2, ntp=1, barostat=1, taup=1000000.0, pres0=1.01325,
  ntt=3, gamma_ln=1.0, temp0=303.0,
 /
IN
  "$pmemd" -O -i fixed-amber-$tag.in -p water.prmtop -c water.rst7 -o fixed-amber-$tag.out -r fixed-amber-$tag.rst -inf fixed-amber-$tag.info
done
python3 - <<'PY'
import math, re
def blocks(x, n):
    x = x[len(x) // 20:]
    m = len(x) // n
    b = [sum(x[i * n:(i + 1) * n]) / n for i in range(m)]
    mean = sum(x) / len(x)
    var = sum((v - sum(b) / m) ** 2 for v in b) / (m - 1)
    return mean, math.sqrt(var / m)
print(f"{'dt, fs':7s} {'MDIR':>14s} {'GROMACS':>14s} {'pmemd.cuda':>14s}")
for tag, dt in (("002", 2), ("001", 1), ("0005", 0.5), ("00025", 0.25)):
    m = re.search(r"x and y ([-0-9.]+) ± ([0-9.]+), z ([-0-9.]+) ± ([0-9.]+)", open(f"fixed-mdir-{tag}.log").read())
    lat, el, z, ez = map(float, m.groups())
    a = ((2 * lat + z) / 3, math.sqrt(4 * el ** 2 + ez ** 2) / 3)
    g = blocks([float(l.split()[1]) for l in open(f"pressure-{tag}.xvg") if not l.startswith(("#", "@"))], 200)
    body = open(f"fixed-amber-{tag}.out").read().split("A V E R A G E S")[0]
    p = blocks([float(v) for v in re.findall(r"PRESS =\s*(-?[0-9.]+)", body)], 50)
    print(f"{dt:<7g} {a[0]:7.1f} ± {a[1]:3.1f} {g[0]:7.1f} ± {g[1]:3.1f} {p[0]:7.1f} ± {p[1]:3.1f}")
PY
