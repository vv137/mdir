#!/bin/bash
# Runs the dipeptide in OPC of pme-settle.test (particle mesh Ewald, SETTLE,
# extra points) in double precision on the CPU and on the GPU for STEPS
# steps of 1 fs, from the same state, and compares them
# (scripts/validation/gpu-cpu/compare.py): how far apart the rows of the
# two logs drift, which the chaos of the dynamics amplifies from the
# rounding of sums in different orders; the means of the potential energy
# and of the temperature, with their standard errors; and the conservation
# of the energy of each.
#
#   scripts/validation/gpu-cpu/run.sh WORK MDIR [STEPS]
set -e
work=$1; mdir=$2; steps=${3:-20000}
here=$(cd "$(dirname "$0")" && pwd)
root=$here/../../..
mkdir -p "$work"
cp "$root/test/Driver/Inputs/opc/opc.prmtop" "$root/test/Driver/Inputs/opc/opc.inpcrd" "$work"
cd "$work"
sed -n '/^#--- run.toml/,/^#--- end/p' "$root/test/Driver/pme-settle.test" | sed '1d;$d' \
  | sed -e "s/^steps *=.*/steps = $steps/" -e 's/^energy_interval *=.*/energy_interval = 100/' > base.toml
for target in cpu gpu; do
  cp base.toml $target.toml
  printf '\n[execution]\ntarget = "%s"\nprecision = "double"\n' $target >> $target.toml
  [ $target = cpu ] && echo 'threads = 16' >> $target.toml
  "$mdir" run $target.toml > $target.log
done
python3 "$here/compare.py" cpu.log gpu.log
