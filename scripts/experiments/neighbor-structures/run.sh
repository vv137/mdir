#!/usr/bin/env bash
# Runs the prototypes of the loop over pairs and of the search on JAC and
# writes their times to results/<date>-<device>.csv. See README.md.
#
#   MDIR_BENCH_DIR=~/opt/benchmarks/amber CUDA_VISIBLE_DEVICES=0 ./run.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bench=${MDIR_BENCH_DIR:?set MDIR_BENCH_DIR to the prepared Amber suite}
nvcc=${NVCC:-nvcc}
arch=${CUDA_ARCH:-sm_86}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

python3 "$here/prep.py" "$bench/jac_nve/system.parm7" "$bench/jac_nve/system.rst7" "$work/jac.bin"
for program in pairs search; do
  "$nvcc" -O3 -arch="$arch" -std=c++17 -diag-suppress 1650 -Xcompiler -Wno-unused-result \
    -o "$work/$program" "$here/$program.cu"
done

device=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | tr ' ' '-')
out="$here/results/$(date +%F)-$device.csv"
echo "experiment,variant,reach,parameter,microseconds" > "$out"
for reach in 8.05 8.5 9.0 9.5 10.0 11.0 12.0; do
  "$work/pairs" "$work/jac.bin" "$reach" | grep '^csv,' | cut -d, -f2- >> "$out"
done
for divisor in 1 2 3 4; do
  "$work/search" "$work/jac.bin" 10.0 "$divisor" | grep '^csv,' | cut -d, -f2- >> "$out"
done
for reach in 8.5 9.0 11.0 12.0; do
  "$work/search" "$work/jac.bin" "$reach" 2 | grep '^csv,' | cut -d, -f2- >> "$out"
done
echo "wrote $out"

# The search in an octree (octree.cu, D[octane]), with OCTREE=1: JAC,
# Cellulose, and argon (1,000,000 atoms at the density of the liquid and at
# that of [Toutouni2026], equilibrated by the prototype itself), and the
# loops and builds of groups.cu, search.cu, and build.cu beside them. The
# README says how the rows of M and Gr in the results are put together.
if [ "${OCTREE:-0}" = 1 ]; then
  flags="-O3 -arch=$arch -std=c++17 -DFAST_ERFC -diag-suppress 1650 -Xcompiler -Wno-unused-result"
  for program in octree groups build; do "$nvcc" $flags -o "$work/$program" "$here/$program.cu"; done
  "$nvcc" $flags -DCUTOFF=10.122f -DLJ_ONLY -o "$work/groups-argon" "$here/groups.cu"
  python3 "$here/prep.py" "$bench/cellulose_nve/system.parm7" "$bench/cellulose_nve/system.rst7" "$work/cellulose.bin"
  "$work/octree" gen "$work/liquid0.bin" lattice 100 362.46 0.2 1
  "$work/octree" gen "$work/vapor0.bin" lattice 100 1033 3.0 2
  "$work/octree" gen "$work/uniform.bin" random 100 1033 0 5
  "$work/octree" "$work/liquid0.bin" 10.122 levels=5 divisors=2 usteps=2 equil=3000 steps=1000 out="$work/argon-liquid.bin" > /dev/null
  "$work/octree" "$work/vapor0.bin" 10.122 levels=6 divisors=2 usteps=2 equil=3000 steps=1000 out="$work/argon-vapor.bin" > /dev/null
  out="$here/results/$(date +%F)-octree-$device.csv"
  echo "system,order,variant,parameter,quantity,value" > "$out"
  for order in input morton; do
    "$work/octree" "$work/jac.bin" 8 check=1 reach=9 steps=100 order=$order
    "$work/octree" "$work/cellulose.bin" 8 reach=9 steps=100 order=$order
    "$work/octree" "$work/argon-liquid.bin" 10.122 reach=11.122 steps=3000 order=$order
    "$work/octree" "$work/argon-vapor.bin" 10.122 levels=5,6,7 reach=11.122 steps=3000 order=$order
  done | grep '^csv,' | cut -d, -f2- >> "$out"
  "$work/octree" "$work/uniform.bin" 10.122 levels=5,6,7 divisors=2 usteps=1 | grep '^csv,' | cut -d, -f2- >> "$out"
  for system in jac cellulose; do
    "$work/groups" "$work/$system.bin" 9 compact 0 50; "$work/search" "$work/$system.bin" 9.0 2; ORDER=gpu "$work/build" "$work/$system.bin" 9
  done
  for system in argon-liquid argon-vapor; do
    "$work/groups-argon" "$work/$system.bin" 11.122 compact 0 50; "$work/search" "$work/$system.bin" 11.122 2; ORDER=gpu "$work/build" "$work/$system.bin" 11.122
  done
  echo "wrote $out"
fi
