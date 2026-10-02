#!/usr/bin/env bash
# Builds the POPC bilayer and runs the four stages of the tutorial in a
# directory of its own:
#
#   examples/popc-bilayer/run.sh [<directory>] [<mdir>]
#
# The directory is popc-run by default, and mdir is found on PATH or given.
# packmol-memgen, packmol, and tleap of AmberTools must be on PATH. Each
# stage begins from the checkpoint of the one before; a stage whose
# checkpoint exists is skipped.
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
for stage in 1-min:min 2-nvt:nvt 3-npt:npt 4-md:md; do
  name=${stage%%:*} checkpoint=${stage##*:}.h5
  [ -f "$checkpoint" ] && { echo "== $name (done)"; continue; }
  echo "== $name"
  "$mdir" run "$name.toml" | tee "$name.log"
done
python3 "$here"/area.py md.dcd --skip 10000
