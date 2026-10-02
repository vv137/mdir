#!/usr/bin/env bash
# Builds ubiquitin in OPC water with tleap and runs the four stages of the
# tutorial in a directory of its own:
#
#   examples/ubiquitin/run.sh [<directory>] [<mdir>]
#
# The directory is ubiquitin-run by default, and mdir is found on PATH or
# given. tleap of AmberTools must be on PATH. Each stage begins from the
# checkpoint of the one before; a stage that ended is skipped.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=${1:-ubiquitin-run}
mdir=${2:-mdir}
mkdir -p "$work"
cp "$here"/1ubq-protein.pdb "$here"/ubq.leap "$here"/*.toml "$work"
cd "$work"
if [ ! -f ubq.prmtop ]; then
  echo "== 0-build"
  tleap -f ubq.leap > leap.log
  grep -E "^(Error|FATAL)" leap.log && exit 1 || true
fi
for name in 1-min 2-nvt 3-npt 4-md; do
  # A stage is done when it ended, not when its checkpoint exists: a run
  # writes checkpoints as it goes, and one that stopped has one too.
  [ -f "$name.done" ] && { echo "== $name (done)"; continue; }
  echo "== $name"
  "$mdir" run "$name.toml" | tee "$name.log"
  touch "$name.done"
done
