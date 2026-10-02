#!/usr/bin/env bash
# Builds ubiquitin in OPC water with tleap and runs the four stages of the
# tutorial in a directory of its own:
#
#   examples/ubiquitin/run.sh [<directory>] [<mdir>]
#
# The directory is ubiquitin-run by default, and mdir is found on PATH or
# given. tleap of AmberTools must be on PATH. Each stage begins from the
# checkpoint of the one before; run again, the script continues a stage
# that stopped and skips one that is complete.
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
  # A stage that stopped goes on from its checkpoint, and one that is
  # complete says so (docs/driver-m0.md, Section 2.7).
  echo "== $name"
  "$mdir" run --continue "$name.toml" | tee -a "$name.log"
done
