#!/bin/bash
# Conservation of energy at constant energy against sander: alanine
# dipeptide in flexible TIP3P water (test/Driver/Inputs/dipeptide), particle
# mesh Ewald, no constraints, 2 ps at 0.5 and 0.25 fs, both programs from the
# same positions and velocities (a step of sander at 300 K, written in
# ASCII). MDIR runs on the CPU in double precision from the topology with
# its charges scaled to Amber's Coulomb constant; both take the same beta,
# grid, order, and influence function, and no correction for the
# dispersion. Prints, for each program and step, the change of the total
# energy, its standard deviation, and the slope of a line fitted to it.
#
#   scripts/validation/nve/run.sh WORK MDIR ENGINES
#
# ENGINES is the prefix of AmberTools (bin/sander, and bin/python with
# numpy).
set -e
work=$1; mdir=$2; engines=$3
here=$(cd "$(dirname "$0")" && pwd)
inputs=$here/../../../test/Driver/Inputs/dipeptide
mkdir -p "$work"; cd "$work"
cp "$inputs/dipeptide.prmtop" "$inputs/dipeptide.inpcrd" .
"$engines/bin/python" "$here/../suite/run.py" --scale-only dipeptide.prmtop scaled.prmtop
ewald=" &ewald
  nfft1=30, nfft2=30, nfft3=30, order=4, ew_coeff=0.3470459193712083,
  vdwmeth=0, netfrc=0,
 /"
cat > start.in <<IN
Velocities at 300 K
 &cntrl
  imin=0, ntx=1, irest=0, tempi=300.0, ig=2026, nstlim=1, dt=0.0005,
  ntb=1, cut=9.0, ntc=1, ntf=1, ntpr=1, ntt=0, ntxo=1,
 /
$ewald
IN
"$engines/bin/sander" -O -i start.in -p dipeptide.prmtop -c dipeptide.inpcrd \
  -o start.out -r start.rst7
for dt in 0.0005 0.00025; do
  steps=$(python3 -c "print(round(2.0 / $dt))")
  every=$(python3 -c "print(round(0.005 / $dt))")
  cat > sander-$dt.in <<IN
Constant energy, flexible water
 &cntrl
  imin=0, ntx=5, irest=1, nstlim=$steps, dt=$dt, ntb=1, cut=9.0,
  ntc=1, ntf=1, ntpr=$every, ntt=0, ntwr=$steps, ntwx=0,
 /
$ewald
IN
  "$engines/bin/sander" -O -i sander-$dt.in -p dipeptide.prmtop \
    -c start.rst7 -o sander-$dt.out -r sander-$dt.rst -inf mdinfo-$dt &
  cat > mdir-$dt.toml <<TOML
[input]
format      = "AMBER"
topology    = "scaled.prmtop"
coordinates = "start.rst7"
[output]
energy_interval = $every
[energy]
cutoff                = 9.0
pairlist_distance     = 10.0
electrostatics        = "PME"
dispersion_correction = "NONE"
[pme]
beta      = 0.3470459193712083
grid      = [30, 30, 30]
order     = 4
influence = "OPTIMAL"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = $dt
steps      = $steps
[ensemble]
ensemble    = "NVE"
temperature = 300.0
[boundary]
type = "PERIODIC"
[execution]
target    = "CPU"
precision = "DOUBLE"
threads   = 4
TOML
  "$mdir" run mdir-$dt.toml > mdir-$dt.log
done
wait
"$engines/bin/python" - <<'PY'
import re
import numpy as np


def sander(path):
    rows = re.findall(r"NSTEP =\s+\d+\s+TIME\(PS\) =\s+([\d.]+).*?"
                      r"Etot\s+=\s+(-?[\d.]+)", open(path).read(), re.S)
    rows = rows[:-2]  # the averages and the fluctuations
    return np.array([[float(t), float(e)] for t, e in rows])


def mdir(path):
    rows = [line.split() for line in open(path)]
    return np.array([[float(f[2]), float(f[3])] for f in rows
                     if len(f) > 4 and f[0] == "INFO:" and f[1].isdigit()])


for dt in ("0.0005", "0.00025"):
    for name, data in (("sander", sander(f"sander-{dt}.out")),
                       ("MDIR", mdir(f"mdir-{dt}.log"))):
        t, e = data[:, 0], data[:, 1]
        slope = np.polyfit(t - t[0], e, 1)[0]
        print(f"{float(dt) * 1000:.2f} fs {name:6s} change {e[-1] - e[0]:+8.4f}"
              f"  std {e.std():.4f}  slope {slope:+.3f} kcal/mol/ps")
PY
