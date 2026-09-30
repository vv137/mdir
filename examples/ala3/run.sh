#!/usr/bin/env bash
# Runs the four stages of the example in a directory of its own:
#
#   examples/ala3/run.sh [<directory>] [<mdir>]
#
# The directory is ala3-run by default, and mdir is found on PATH or given.
# Each stage begins from the checkpoint of the one before.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=${1:-ala3-run}
mdir=${2:-mdir}
mkdir -p "$work"
cp "$here"/ala3.prmtop "$here"/ala3.inpcrd "$here"/*.toml "$work"
cd "$work"
for stage in 1-min 2-nvt 3-npt 4-md; do
  echo "== $stage"
  "$mdir" run "$stage.toml" | tee "$stage.log"
done
