#!/usr/bin/env bash
# Compares the terms of the potential of MDIR with those of GROMACS for the
# peptide of peptide.leap in water, with each force field that GROMACS
# ships and MDIR should read.
#
# Usage: scripts/validation/gromacs/run.sh <work directory>
#
# Needs tleap of AmberTools and gmx of GROMACS, on PATH, in ENGINES/bin, or
# named by TLEAP and GMX,
# and mdir, on PATH or named by MDIR (docs/environment.md, Scripts; the
# file that MDIR_ENV_FILE names is read). It stops at its start if one of
# the three is not found. A gmx built for a GPU runs its zero steps on the
# device that CUDA_VISIBLE_DEVICES leaves visible. The energies of
# GROMACS are those of a run of zero steps with single.mdp: flexible water
# (-DFLEXIBLE), Lennard-Jones cut at 0.9 nm with no shift, the correction
# for the dispersion, and the reaction field with a permittivity of 78.5
# beyond the cutoff (D140).
set -e
here=$(cd "$(dirname "$0")" && pwd)
work=${1:?usage: run.sh <work directory>}
. "$here/../../environment.sh"
if [[ -n ${ENGINES:-} ]]; then
  need_directory ENGINES "a prefix whose bin holds tleap and gmx"
  PATH="$ENGINES/bin:$PATH"
fi
need_program TLEAP tleap "tleap of AmberTools, on PATH or in ENGINES/bin"
need_program GMX gmx "gmx of GROMACS, on PATH or in ENGINES/bin"
need_program MDIR mdir "the mdir program to compare"
need_check
export PATH MDIR
top=$(dirname "$(dirname "$GMX")")/share/gromacs/top
mkdir -p "$work" && cd "$work"
tleap -f "$here/peptide.leap" > leap.log 2>&1

for spec in amber99sb-ildn:tip3p amber99sb:tip3p amber03:tip3p amber14sb:tip3p \
            amber99sb-ildn:tip4pew amber19sb:tip3p charmm27:tip3p \
            oplsaa:tip3p gromos54a7:spc; do
  ff=${spec%%:*}; water=${spec##*:}
  # Water of four sites takes the configuration of four sites.
  box=spc216.gro
  case $water in tip4p*) box=tip4p.gro; ff=$ff-$water ;; esac
  rm -rf "$ff" && mkdir "$ff" && pushd "$ff" > /dev/null
  "$GMX" pdb2gmx -f ../peptide.pdb -o conf.gro -p topol.top -ff "${spec%%:*}" \
      -water "$water" -ignh > pdb2gmx.log 2>&1
  "$GMX" editconf -f conf.gro -o box.gro -c -d 0.7 -bt cubic > editconf.log 2>&1
  "$GMX" solvate -cp box.gro -cs "$box" -o solv.gro -p topol.top \
      > solvate.log 2>&1
  "$GMX" grompp -f "$here/single.mdp" -c solv.gro -p topol.top -o run.tpr \
      -maxwarn 5 > grompp.log 2>&1
  "$GMX" mdrun -s run.tpr -deffnm run -nt 1 > mdrun.log 2>&1
  printf 'Bond\nG96Bond\nAngle\nG96Angle\nU-B\nProper-Dih.\nPer.-Imp.-Dih.\nImproper-Dih.\nCMAP-Dih.\nRyckaert-Bell.\nLJ-14\nCoulomb-14\nLJ-(SR)\nDisper.-corr.\nCoulomb-(SR)\nPotential\n\n' \
      | "$GMX" energy -f run.edr -o energy.xvg > energy.log 2>&1 || true
  sed "s|@TOP@|$top|" "$here/mdir.toml.in" > mdir.toml
  popd > /dev/null
  python3 "$here/compare.py" "$work/$ff"
done
