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
