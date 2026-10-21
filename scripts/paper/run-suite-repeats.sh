#!/bin/bash
# Runs the Amber suite with MDIR and with pmemd.cuda, alternately, N times
# (3 by default) on one device, under the protocol of each system's
# mdin.GPU for both (bench.py --suite-protocol, D[suite-symmetric]): its
# steps (nstlim), its energies (ntpr), its trajectory (ntwx), and its
# restarts (ntwr), and a rate over the same steps, from the first report to
# the last step. Each repeat runs, one after the other:
#
#   searched           MDIR, the reaches of the search, from the cache
#   output             pmemd.cuda on mdin.GPU as it is
#   default            MDIR, the default reaches (no search, no cache)
#   searched-energies  MDIR as "searched", energies only
#   energies           pmemd.cuda with ntwx=0 and ntwr=nstlim
#
# and writes the numbers of every run to LOGS/runs.jsonl (scripts/paper/
# suite-table.py reads them), a summary of each variant to
# LOGS/VARIANT-N.log, and the output of each system to
# LOGS/out-VARIANT-N/SYSTEM.out (pmemd: SYSTEM.mdout).
#
# The settings of MDIR are otherwise those a user gets with neither
# `neighbor_structure` nor the reaches of the lists given: the structure is
# AUTO (the groups with a dual list, D261, for every system of the suite),
# and the reaches are "AUTO" (D281), chosen by a search that a run makes
# only when it is long (5e12 particles times steps). So that the
# "searched" runs take the reaches such a run would choose, the script
# first runs each system once, untimed and short (bench.py's own steps;
# the key of the cache has no step count), with MDIR_SKIN_SEARCH=always,
# which measures the candidates and caches its choice (search.log,
# out-search/SYSTEM.out); the timed runs then take the cached choice, with
# MDIR_SKIN_SEARCH=never, so that none searches. The "default" runs read
# no cache (MDIR_SKIN_CACHE=off) and take the defaults of D261. The script
# stops if a timed run did not take its reaches as its variant says.
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
record="$logs/runs.jsonl"
mkdir -p "$logs/out-search"
for s in $systems; do
  MDIR_SKIN_SEARCH=always python3 "$bench" run mdir "$s" --mdir "$mdir" \
    --skin AUTO >> "$logs/search.log" 2>&1
  cp "$work/$s/mdir.out" "$logs/out-search/$s.out"
done

# MDIR in the variant $1 of repeat $2, with the remaining arguments for
# bench.py; checks that every system took its reaches from the source $3.
run_mdir() {
  variant=$1; r=$2; source=$3; shift 3
  python3 "$bench" run mdir --mdir "$mdir" --skin AUTO --suite-protocol \
    --record "$record" --variant "$variant" --repeat "$r" "$@" \
    > "$logs/$variant-$r.log" 2>&1
  mkdir -p "$logs/out-$variant-$r"
  for s in $systems; do
    cp "$work/$s/mdir.out" "$logs/out-$variant-$r/$s.out"
    grep -q "^MDIR: skins: the reach .*($source:" "$logs/out-$variant-$r/$s.out" || {
      echo "run-suite-repeats.sh: $s of $variant $r did not take $source skins" >&2
      exit 1
    }
  done
}

# pmemd.cuda in the variant $1 of repeat $2, with the remaining arguments.
run_pmemd() {
  variant=$1; r=$2; shift 2
  python3 "$bench" run pmemd --pmemd "$pmemd" --suite-protocol \
    --record "$record" --variant "$variant" --repeat "$r" "$@" \
    > "$logs/$variant-$r.log" 2>&1
  mkdir -p "$logs/out-$variant-$r"
  directory=pmemd; [ "$variant" = energies ] && directory=pmemd-energies
  for s in $systems; do
    cp "$work/$s/$directory/mdout" "$logs/out-$variant-$r/$s.mdout"
  done
}

for r in $(seq 1 "$repeats"); do
  MDIR_SKIN_SEARCH=never run_mdir searched "$r" cached
  run_pmemd output "$r"
  MDIR_SKIN_SEARCH=never MDIR_SKIN_CACHE=off run_mdir default "$r" default
  MDIR_SKIN_SEARCH=never run_mdir searched-energies "$r" cached --energies-only
  run_pmemd energies "$r" --energies-only
done
