#!/usr/bin/env bash
# Builds the POPC bilayer and runs the four stages of the tutorial in a
# directory of its own:
#
#   examples/popc-bilayer/run.sh [<directory>] [<mdir>]
#
# The directory is popc-run by default, and mdir is found on PATH or given.
# packmol-memgen, packmol, and tleap of AmberTools must be on PATH. Each
# stage begins from the checkpoint of the one before; run again, the
# script continues a stage that stopped and skips one that is complete.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=${1:-popc-run}
mdir=${2:-mdir}
mkdir -p "$work"
cp "$here"/popc.leap "$here"/*.toml "$work"
cd "$work"
if [ ! -f popc.prmtop ]; then
  echo "== 0-build"
  "$here"/build.sh
fi
for name in 1-min 2-nvt 3-npt 4-md; do
  # A stage that stopped goes on from its checkpoint, and one that is
  # complete says so (docs/driver-m0.md, Section 2.7).
  echo "== $name"
  "$mdir" run --continue "$name.toml" | tee -a "$name.log"
done
python3 "$here"/area.py md.dcd --skip 10000
