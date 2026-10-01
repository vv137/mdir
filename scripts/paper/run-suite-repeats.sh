#!/bin/bash
# Runs the Amber suite with MDIR and with pmemd.cuda, alternately, N times
# (3 by default) on one device, and writes a log of each run to LOGS:
#   mdir-1.log, pmemd-1.log, mdir-2.log, ...
# The settings of MDIR are those of the paper (Section 10): groups of 16, a
# dual list with a skin of 3 Å and an inner skin of 0.6 Å at 2 fs.
#
#   scripts/paper/run-suite-repeats.sh MDIR PMEMD LOGS [N]
#
# MDIR_BENCH_DIR names the prepared suite (scripts/benchmarks/amber); the
# device is the one that CUDA_VISIBLE_DEVICES leaves visible.
set -e
mdir=$1; pmemd=$2; logs=$3; repeats=${4:-3}
here=$(cd "$(dirname "$0")" && pwd)
bench="$here/../benchmarks/amber/bench.py"
mkdir -p "$logs"
for r in $(seq 1 "$repeats"); do
  python3 "$bench" run mdir --mdir "$mdir" --neighbor-structure GROUPS \
    --skin 3.0 --prune-skin 0.6 > "$logs/mdir-$r.log" 2>&1
  python3 "$bench" run pmemd --pmemd "$pmemd" > "$logs/pmemd-$r.log" 2>&1
done
