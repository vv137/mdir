#!/usr/bin/env bash
# Builds a bilayer of 126 POPC lipids of Lipid21 in TIP3P water: 63 a
# leaflet on a square of 65 Å with 17.5 Å of water on each side, 31,680
# atoms. Needs packmol-memgen, packmol, and tleap of AmberTools on PATH.
#
#   build.sh            (in the directory of the run)
#
# packmol-memgen writes the input of packmol (packmol.inp) and does not run
# it (--notrun); packmol packs it into popc.pdb, which takes about ten
# minutes, and tleap makes popc.prmtop and popc.inpcrd. packmol may end
# "without perfect packing" with some hydrogens of different lipids almost
# on top of each other; the minimization of stage 1 takes them apart.
set -euo pipefail
# packmol-memgen exits with 1 when it does not run packmol, after writing
# its input.
packmol-memgen -l POPC -r 1 --distxy_fix 65 --notrun --noprogress \
  --overwrite -o popc.pdb > memgen.log 2>&1 || true
[ -s packmol.inp ] || { echo "packmol-memgen wrote no packmol.inp; see memgen.log"; exit 1; }
packmol < packmol.inp > packmol.log 2>&1 || true
[ -s popc.pdb ] || { echo "packmol wrote no popc.pdb; see packmol.log"; exit 1; }
tleap -f popc.leap > leap.log 2>&1
grep -E "^(Error|FATAL)" leap.log && exit 1 || true
echo "built popc.prmtop and popc.inpcrd"
