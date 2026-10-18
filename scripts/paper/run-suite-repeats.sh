#!/bin/bash
# Runs the Amber suite with MDIR and with pmemd.cuda, alternately, N times
# (3 by default) on one device, and writes a log of each run to LOGS:
#   mdir-1.log, pmemd-1.log, mdir-2.log, ...
# and keeps the output of each system of each repeat, which the next repeat
# overwrites: out-mdir-N/SYSTEM.out and out-pmemd-N/SYSTEM.mdout.
#
# The settings of MDIR are those of the paper (Section 10): what a user
# gets with neither `neighbor_structure` nor the reaches of the lists
# given. The structure is AUTO (the groups with a dual list, D261, for
# every system of the suite), and the reaches are "AUTO" (D281), chosen by
# a search that a run makes only when it is long (5e12 particles times
# steps). So that every timed run takes the reaches such a run would
# choose, the script first runs each system once, untimed, with
# MDIR_SKIN_SEARCH=always, which measures the candidates and caches its
# choice (search.log, out-search/SYSTEM.out); the timed runs then take the
# cached choice, with MDIR_SKIN_SEARCH=never, so that none searches. The
# script stops if a timed run did not take its reaches from the cache.
#
#   scripts/paper/run-suite-repeats.sh MDIR PMEMD LOGS [N]
#
# MDIR_BENCH_DIR names the prepared suite (scripts/benchmarks/amber); the
# device is the one that CUDA_VISIBLE_DEVICES leaves visible. The cache of
# the skins is MDIR_SKIN_CACHE_DIR, by default LOGS/skins; an entry already
# there for a key is taken without a search.
set -e
mdir=$1; pmemd=$2; repeats=${4:-3}
# Absolute: bench.py runs MDIR in the directory of each system.
mkdir -p "$3"; logs=$(cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
bench="$here/../benchmarks/amber/bench.py"
work=${MDIR_BENCH_DIR:-$HOME/mdir-benchmarks}
systems="jac_nve jac_nve_4fs jac_npt jac_npt_4fs factorix_nve factorix_npt
         cellulose_nve cellulose_npt stmv_npt_4fs"
export MDIR_SKIN_CACHE_DIR=${MDIR_SKIN_CACHE_DIR:-$logs/skins}
unset MDIR_SKIN_CACHE
mkdir -p "$logs/out-search"
for s in $systems; do
  MDIR_SKIN_SEARCH=always python3 "$bench" run mdir "$s" --mdir "$mdir" \
    --skin AUTO >> "$logs/search.log" 2>&1
  cp "$work/$s/mdir.out" "$logs/out-search/$s.out"
done
for r in $(seq 1 "$repeats"); do
  MDIR_SKIN_SEARCH=never python3 "$bench" run mdir --mdir "$mdir" \
    --skin AUTO > "$logs/mdir-$r.log" 2>&1
  mkdir -p "$logs/out-mdir-$r"
  for s in $systems; do
    cp "$work/$s/mdir.out" "$logs/out-mdir-$r/$s.out"
    grep -q "^MDIR: skins: the reach .*(cached:" "$logs/out-mdir-$r/$s.out" || {
      echo "run-suite-repeats.sh: $s of repeat $r did not take cached skins" >&2
      exit 1
    }
  done
  python3 "$bench" run pmemd --pmemd "$pmemd" > "$logs/pmemd-$r.log" 2>&1
  mkdir -p "$logs/out-pmemd-$r"
  for s in $systems; do cp "$work/$s/pmemd/mdout" "$logs/out-pmemd-$r/$s.mdout"; done
done
