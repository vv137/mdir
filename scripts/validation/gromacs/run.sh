#!/bin/bash
# Compares the terms of the potential of MDIR with those of GROMACS for the
# peptide of peptide.leap in water, with each force field that GROMACS
# ships and MDIR should read, and shows what MDIR rejects for the others.
#
# Usage: scripts/validation/gromacs/run.sh <work directory>
#
# Needs tleap of AmberTools and gmx of GROMACS on PATH or in ENGINES/bin,
# and mdir in MDIR (default: ~/build/mdir/bin/mdir). The energies of
# GROMACS are those of a run of zero steps with single.mdp: flexible water
# (-DFLEXIBLE), Lennard-Jones cut at 0.9 nm with no shift, and the
# correction for the dispersion. Coulomb (SR) is not compared: GROMACS has
# no plain cutoff for it.
set -e
here=$(cd "$(dirname "$0")" && pwd)
work=${1:?usage: run.sh <work directory>}
ENGINES=${ENGINES:-$HOME/opt/engines}
export PATH="$ENGINES/bin:$PATH"
export MDIR=${MDIR:-$HOME/build/mdir/bin/mdir}
top=$(dirname "$(dirname "$(command -v gmx)")")/share/gromacs/top
mkdir -p "$work" && cd "$work"
tleap -f "$here/peptide.leap" > leap.log 2>&1

for spec in amber99sb-ildn:tip3p amber99sb:tip3p amber03:tip3p amber14sb:tip3p \
            amber19sb:tip3p charmm27:tip3p oplsaa:tip3p gromos54a7:spc; do
  ff=${spec%%:*}; water=${spec##*:}
  rm -rf "$ff" && mkdir "$ff" && pushd "$ff" > /dev/null
  gmx pdb2gmx -f ../peptide.pdb -o conf.gro -p topol.top -ff "$ff" \
      -water "$water" -ignh > pdb2gmx.log 2>&1
  gmx editconf -f conf.gro -o box.gro -c -d 0.7 -bt cubic > editconf.log 2>&1
  gmx solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top \
      > solvate.log 2>&1
  gmx grompp -f "$here/single.mdp" -c solv.gro -p topol.top -o run.tpr \
      -maxwarn 5 > grompp.log 2>&1
  gmx mdrun -s run.tpr -deffnm run -nt 1 > mdrun.log 2>&1
  printf 'Bond\nAngle\nProper-Dih.\nPer.-Imp.-Dih.\nImproper-Dih.\nLJ-14\nCoulomb-14\nLJ-(SR)\nDisper.-corr.\nCoulomb-(SR)\nPotential\n\n' \
      | gmx energy -f run.edr -o energy.xvg > energy.log 2>&1 || true
  sed "s|@TOP@|$top|" "$here/mdir.toml.in" > mdir.toml
  popd > /dev/null
  python3 "$here/compare.py" "$work/$ff"
done
