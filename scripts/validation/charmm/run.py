#!/usr/bin/env python3
"""Compares MDIR with CHARMM, term by term, on a system of CHARMM36m.

    scripts/validation/charmm/run.py WORK --mdir MDIR [--charmm CHARMM]
        [--toppar DIR] [--gmx GMX] [--packmol PACKMOL]
        [--parmed-python PYTHON] [--phases pair,build,convert,terms,hexagonal]

The topology and parameter files of CHARMM are read from DIR (default
$CHARMM_TOPPAR); nothing of CHARMM is copied into the repository. The phases:

  pair     Two particles at 3 to 12.5 Å in CHARMM, with VFSWITCH and VSWITCH
           (10 to 12 Å), and the Coulomb constant from two unit charges 10 Å
           apart; the curves against the force switch of Steinbach and
           Brooks, the potential switch of CHARMM, and the force switch of
           GROMACS.
  build    Trp-cage (NLYIQWLKDGGPSSGRPPPS) from internal coordinates in
           CHARMM, minimized; 3900 TIP3P waters, 6 Na+ and 7 Cl- packed around
           it by packmol in a cell of 76 x 40 x 40 Å; minimized in CHARMM with
           PME and VFSWITCH (1000 steps). Writes system.psf and
           system-charmm.crd.
  convert  The PSF and the parameters to a topology of GROMACS with ParmEd,
           with the special 1-4 Lennard-Jones of CHARMM added as
           [ pairtypes ], which ParmEd leaves out; the coordinates of the CRD
           in a .gro at full precision.
  terms    The energy of the coordinates as written, term by term: CHARMM
           (PME with kappa 0.34, grid 80 x 40 x 40, order 6; VFSWITCH),
           GROMACS 2026 by a rerun (force-switch), and MDIR on the CPU in
           double precision: from the PSF and the files of CHARMM
           (POWER_FORCE_SWITCH), and from the topology of GROMACS with each
           switch.
  hexagonal  The waters of the build within a hexagonal cell of CHARMM, a =
           b = 36 Å, c = 38 Å, γ = 120°, in its symmetric frame, the rows of
           G^(1/2) for the metric G; minimized in CHARMM with PME (300
           steps); the energy of the coordinates as written, term by term,
           in CHARMM and in MDIR from the PSF and the CRD with the cell
           given by its lengths and angles (docs/triclinic-m2.md).
"""
import argparse
import math
import os
import re
import subprocess
import sys

SEQUENCE = ("ASN LEU TYR ILE GLN TRP LEU LYS ASP GLY GLY PRO SER SER GLY ARG "
            "PRO PRO PRO SER")
CELL = (76.0, 40.0, 40.0)
WATERS, SODIUM, CHLORIDE = 3900, 6, 7
KAPPA, GRID, ORDER = 0.34, (80, 40, 40), 6
CUTOFF, SWITCH = 12.0, 10.0


def run(command, cwd, log, stdin=None):
    with open(os.path.join(cwd, log), "w") as out:
        result = subprocess.run(command, cwd=cwd, stdout=out,
                                stderr=subprocess.STDOUT,
                                stdin=open(os.path.join(cwd, stdin))
                                if stdin else None)
    if result.returncode != 0:
        sys.exit(f"{' '.join(command)} failed; see {os.path.join(cwd, log)}")


def write(path, text):
    with open(path, "w") as file:
        file.write(text)


def toppar(args):
    return (f"read rtf card name {args.toppar}/top_all36_prot.rtf\n"
            f"read param card flex name {args.toppar}/par_all36m_prot.prm\n"
            f"stream {args.toppar}/toppar_water_ions.str\n")


# --------------------------------------------------------------------------
# pair
# --------------------------------------------------------------------------

PAIR = """* two particles: the Lennard-Jones energy under the switches, and the
* Coulomb constant
*
read rtf card
* a synthetic residue
*
36 1
MASS 1 XA 39.948
RESI XA 0.0
ATOM XA XA 0.0
END

read param card
* synthetic parameters
*
NONBONDED nbxmod 5 atom cdiel fshift vdwr vswitch cutnb 14.0 ctofnb 12.0 -
  ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5
XA 0.0 -0.238 1.8800
END

read sequence XA 2
generate A setup noangle nodihedral
coor set xdir 0.0 ydir 0.0 zdir 0.0 select bynum 1 end
set r 3.0
label loop
  coor set xdir @r ydir 0.0 zdir 0.0 select bynum 2 end
  energy atom cdiel vdw vfswitch cutnb 14.0 ctofnb 12.0 ctonnb 10.0
  echo VFSW @r ?VDW
  energy atom cdiel vdw vswitch cutnb 14.0 ctofnb 12.0 ctonnb 10.0
  echo VSWI @r ?VDW
  incr r by 0.25
if r .lt. 12.6 goto loop

scalar charge set 1.0 select bynum 1 end
scalar charge set -1.0 select bynum 2 end
coor set xdir 10.0 ydir 0.0 zdir 0.0 select bynum 2 end
energy atom cdiel switch vdw vswitch cutnb 990.0 ctofnb 980.0 ctonnb 970.0
echo COULOMB ?ELEC
stop
"""


def pair_curves(r, eps=0.238, rmin=3.76, ron=SWITCH, roff=CUTOFF):
    """kcal/mol: the force switch of Steinbach and Brooks, the potential
    switch of CHARMM, and the force switch of GROMACS."""
    a, b = eps * rmin ** 12, 2 * eps * rmin ** 6
    if r >= roff:
        return 0.0, 0.0, 0.0
    if r <= ron:
        sb = a * (r ** -12 - (ron * roff) ** -6) - b * (r ** -6 - (ron * roff) ** -3)
    else:
        sb = (a * roff ** 6 / (roff ** 6 - ron ** 6) * (r ** -6 - roff ** -6) ** 2
              - b * roff ** 3 / (roff ** 3 - ron ** 3) * (r ** -3 - roff ** -3) ** 2)
    plain = a / r ** 12 - b / r ** 6
    switch = 1.0 if r <= ron else ((roff ** 2 - r ** 2) ** 2
                                   * (roff ** 2 + 2 * r ** 2 - 3 * ron ** 2)
                                   / (roff ** 2 - ron ** 2) ** 3)

    def phi(n):
        an = -n * ((n + 4) * roff - (n + 1) * ron) / (roff ** (n + 2) * (roff - ron) ** 2)
        bn = n * ((n + 3) * roff - (n + 1) * ron) / (roff ** (n + 2) * (roff - ron) ** 3)
        cn = roff ** -n - an / 3 * (roff - ron) ** 3 - bn / 4 * (roff - ron) ** 4
        t = max(r - ron, 0.0)
        return r ** -n - an / 3 * t ** 3 - bn / 4 * t ** 4 - cn
    return sb, plain * switch, a * phi(12) - b * phi(6)


def pair(args, work):
    w = os.path.join(work, "pair")
    os.makedirs(w, exist_ok=True)
    write(os.path.join(w, "pair.inp"), PAIR)
    run([args.charmm, "-i", "pair.inp"], w, "pair.out")
    out = open(os.path.join(w, "pair.out")).read()
    rows = re.findall(r"^\s*(VFSW|VSWI) (\S+) (\S+)", out, re.M)
    coulomb = float(re.search(r"^\s*COULOMB (\S+)", out, re.M).group(1))
    print(f"The Coulomb constant of CHARMM: {-10.0 * coulomb:.4f} kcal/mol Å/e²")
    worst = {"VFSW": [0.0, 0.0], "VSWI": [0.0]}
    for kind, r, value in rows:
        r, value = float(r), float(value)
        sb, switch, gromacs = pair_curves(r)
        if value == 0.0:
            continue
        if kind == "VFSW":
            worst["VFSW"][0] = max(worst["VFSW"][0], abs(sb - value) / abs(value))
            worst["VFSW"][1] = max(worst["VFSW"][1], abs(gromacs - value) / abs(value))
        else:
            worst["VSWI"][0] = max(worst["VSWI"][0], abs(switch - value) / abs(value))
    print("Largest relative difference over 3 to 12.5 Å:")
    print(f"  VFSWITCH against Steinbach and Brooks  {worst['VFSW'][0]:.1e}")
    print(f"  VFSWITCH against GROMACS force-switch  {worst['VFSW'][1]:.1e}")
    print(f"  VSWITCH against CHARMM's switch        {worst['VSWI'][0]:.1e}")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(args, work):
    w = os.path.join(work, "trpcage")
    os.makedirs(w, exist_ok=True)
    write(os.path.join(w, "build.inp"), f"""* Trp-cage in CHARMM36m, from internal coordinates
*
{toppar(args)}
read sequence card
* trp-cage
*
20
{SEQUENCE}
generate PROA setup first NTER last CTER
ic param
ic seed 1 N 1 CA 1 C
ic build
hbuild select hydrogen end
nbonds atom cdie eps 1.0 switch vswitch cutnb 99 ctofnb 98 ctonnb 97
mini sd nstep 500 nprint 100
mini abnr nstep 500 nprint 100
coor orient
write psf card name peptide.psf
write coor pdb name peptide.pdb
* trp-cage
*
stop
""")
    run([args.charmm, "-i", "build.inp"], w, "build.out")

    # The peptide in the middle of the cell, and packmol around it.
    lines = [l for l in open(os.path.join(w, "peptide.pdb"))
             if l.startswith("ATOM")]
    xyz = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines]
    center = [(min(p[k] for p in xyz) + max(p[k] for p in xyz)) / 2
              for k in range(3)]
    with open(os.path.join(w, "peptide-centered.pdb"), "w") as out:
        for l, p in zip(lines, xyz):
            out.write(l[:30] + "".join(f"{p[k] - center[k] + CELL[k] / 2:8.3f}"
                                       for k in range(3)) + l[54:])
        out.write("END\n")
    half = math.radians(104.52) / 2
    write(os.path.join(w, "tip3.pdb"),
          "ATOM      1  OH2 TIP3    1       0.000   0.000   0.000\n"
          f"ATOM      2  H1  TIP3    1    {0.9572 * math.sin(half):8.3f}"
          f"{0.9572 * math.cos(half):8.3f}   0.000\n"
          f"ATOM      3  H2  TIP3    1    {-0.9572 * math.sin(half):8.3f}"
          f"{0.9572 * math.cos(half):8.3f}   0.000\nEND\n")
    for ion in ("SOD", "CLA"):
        write(os.path.join(w, f"{ion.lower()}.pdb"),
              f"ATOM      1  {ion} {ion}     1       0.000   0.000   0.000\n"
              "END\n")
    box = f"inside box 1. 1. 1. {CELL[0] - 1} {CELL[1] - 1} {CELL[2] - 1}"
    write(os.path.join(w, "pack.inp"), f"""tolerance 2.0
filetype pdb
output packed.pdb
seed 2026
structure peptide-centered.pdb
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure
structure tip3.pdb
  number {WATERS}
  {box}
end structure
structure sod.pdb
  number {SODIUM}
  {box}
end structure
structure cla.pdb
  number {CHLORIDE}
  {box}
end structure
""")
    run([args.packmol], w, "pack.out", stdin="pack.inp")

    # A CRD of the whole system in the order of the PSF that CHARMM generates.
    packed = [l for l in open(os.path.join(w, "packed.pdb"))
              if l.startswith(("ATOM", "HETATM"))]
    rows = []
    for p, a in zip(lines, packed):
        rows.append((p[17:21].strip(), p[12:16].strip(), a, "PROA",
                     int(p[22:26])))
    rest = iter(packed[len(lines):])
    for wtr in range(WATERS):
        for name in ("OH2", "H1", "H2"):
            rows.append(("TIP3", name, next(rest), "SOLV", wtr + 1))
    for ion, count in (("SOD", SODIUM), ("CLA", CHLORIDE)):
        for k in range(count):
            rows.append((ion, ion, next(rest), ion, k + 1))
    resno, last, out = 0, None, []
    for i, (res, name, a, seg, resid) in enumerate(rows):
        if (seg, resid) != last:
            resno, last = resno + 1, (seg, resid)
        x, y, z = float(a[30:38]), float(a[38:46]), float(a[46:54])
        out.append(f"{i + 1:10d}{resno:10d}  {res:<8s}  {name:<8s}{x:20.10f}"
                   f"{y:20.10f}{z:20.10f}  {seg:<8s}  {resid:<8d}"
                   f"{0.0:20.10f}")
    write(os.path.join(w, "system.crd"), "* trp-cage in water\n*\n"
          f"{len(out):10d}  EXT\n" + "\n".join(out) + "\n")

    write(os.path.join(w, "setup.inp"), f"""* Trp-cage, {WATERS} TIP3P, {SODIUM} Na+, {CHLORIDE} Cl-: assembled and minimized
*
{toppar(args)}
read psf card name peptide.psf
read sequence TIP3 {WATERS}
generate SOLV setup noangle nodihedral
read sequence SOD {SODIUM}
generate SOD setup noangle nodihedral
read sequence CLA {CHLORIDE}
generate CLA setup noangle nodihedral
read coor card name system.crd
write psf card name system.psf
{images()}
mini sd nstep 500 nprint 100
mini abnr nstep 500 nprint 100
write coor card name system-charmm.crd
stop
""")
    run([args.charmm, "-i", "setup.inp"], w, "setup.out")


def images():
    a, b, c = CELL
    return f"""crystal define orthorhombic {a} {b} {c} 90.0 90.0 90.0
crystal build cutoff 16.0 noper 0
image byseg xcen {a / 2} ycen {b / 2} zcen {c / 2} select segid PROA end
image byres xcen {a / 2} ycen {b / 2} zcen {c / 2} select .not. segid PROA end
nbonds atom vatom cdie eps 1.0 vfswitch cutnb 16.0 cutim 16.0 -
  ctofnb {CUTOFF} ctonnb {SWITCH} -
  ewald pmewald kappa {KAPPA} spline order {ORDER} -
  fftx {GRID[0]} ffty {GRID[1]} fftz {GRID[2]} inbfrq -1 imgfrq -1
"""


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------

CONVERT = r'''
import itertools, sys
import parmed as pmd
from parmed.charmm import CharmmPsfFile, CharmmParameterSet, CharmmCrdFile
toppar, cell = sys.argv[1], [float(v) for v in sys.argv[2:5]]
params = CharmmParameterSet(f"{toppar}/top_all36_prot.rtf",
                            f"{toppar}/par_all36m_prot.prm",
                            f"{toppar}/toppar_water_ions.str")
psf = CharmmPsfFile("system.psf")
psf.load_parameters(params)
psf.coordinates = CharmmCrdFile("system-charmm.crd").coordinates
psf.box = cell + [90.0, 90.0, 90.0]
pmd.gromacs.GromacsTopologyFile.from_structure(psf).write("system-parmed.top")
# The special 1-4 parameters of CHARMM, which ParmEd's topology leaves out.
types = {a.atom_type.name: a.atom_type for a in psf.atoms}
lines = ["[ pairtypes ]", "; i j func sigma epsilon: the 1-4 parameters of CHARMM"]
for a, b in itertools.combinations_with_replacement(sorted(types), 2):
    ta, tb = types[a], types[b]
    rmin = ta.rmin_14 + tb.rmin_14
    eps = (abs(ta.epsilon_14) * abs(tb.epsilon_14)) ** 0.5
    lines.append(f"{a} {b} 1 {rmin / 10 / 2 ** (1 / 6):.12f} {eps * 4.184:.12f}")
top = open("system-parmed.top").read()
top = top.replace("[ nonbond_params ]", "\n".join(lines) + "\n\n[ nonbond_params ]", 1)
open("system.top", "w").write(top)
psf.save("system-parmed.gro", overwrite=True)
'''


def convert(args, work):
    w = os.path.join(work, "trpcage")
    write(os.path.join(w, "convert.py"), CONVERT)
    run([args.parmed_python, "convert.py", args.toppar] + [str(v) for v in CELL],
        w, "convert.log")
    # The coordinates of the CRD (5 decimals in Å) in a .gro whose fields are
    # wide enough to keep them.
    lines = open(os.path.join(w, "system-charmm.crd")).read().splitlines()
    i = 0
    while lines[i].startswith("*"):
        i += 1
    count = int(lines[i].split()[0])
    crd = lines[i + 1:i + 1 + count]
    gro = open(os.path.join(w, "system-parmed.gro")).read().splitlines()
    out = [gro[0], f"{count:5d}"]
    for row, line in zip(crd, gro[2:2 + count]):
        x, y, z = (float(v) / 10 for v in row.split()[4:7])
        out.append(line[:20] + f"{x:12.7f}{y:12.7f}{z:12.7f}")
    out.append("".join(f"{v / 10:12.7f}" for v in CELL))
    write(os.path.join(w, "system.gro"), "\n".join(out) + "\n")


# --------------------------------------------------------------------------
# terms
# --------------------------------------------------------------------------

MDIR = """[input]
topology    = "system.top"
coordinates = "system.gro"
format      = "GROMACS"
defines     = ["FLEXIBLE"]

[output]
energy_interval = 0

[energy]
cutoff            = {cutoff}
switch_distance   = {switch}
lennard_jones_modifier = "{modifier}"
dispersion_correction  = "NONE"
pairlist_distance = {pairlist}
electrostatics    = "PME"

[pme]
beta  = {kappa}
order = {order}
grid  = [{grid}]

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.001
steps      = 0

[ensemble]
ensemble = "NVE"

[constraints]
hydrogen_bonds = false
rigid_water    = false

[boundary]
type = "PERIODIC"

[execution]
target    = "CPU"
precision = "DOUBLE"
"""

GROMACS = """define = -DFLEXIBLE
integrator = md
nsteps = 0
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = {rc}
vdwtype = Cut-off
vdw-modifier = Force-switch
rvdw-switch = {rs}
rvdw = {rc}
DispCorr = no
fourier-nx = {nx}
fourier-ny = {ny}
fourier-nz = {nz}
pme-order = {order}
ewald-rtol = {rtol}
nstenergy = 1
nstcalcenergy = 1
constraints = none
verlet-buffer-tolerance = -1
rlist = {rc}
"""


def terms(args, work):
    w = os.path.join(work, "trpcage")
    # CHARMM.
    write(os.path.join(w, "energy.inp"), f"""* Trp-cage: the energy of the coordinates as written, term by term
*
{toppar(args)}
read psf card name system.psf
read coor card name system-charmm.crd
{images()}
energy
echo TERMS BOND ?BOND ANGL ?ANGL UREY ?UREY DIHE ?DIHE IMPR ?IMPR CMAP ?CMAP
echo TERMS VDW ?VDW IMNB ?IMNB ELEC ?ELEC IMEL ?IMEL EWKS ?EWKS EWSE ?EWSE EWEX ?EWEX
stop
""")
    run([args.charmm, "-i", "energy.inp"], w, "energy.out")
    text = open(os.path.join(w, "energy.out")).read()
    c = {}
    for line in re.findall(r"^\s*TERMS (.*)$", text, re.M):
        fields = line.split()
        c.update({fields[k]: float(fields[k + 1])
                  for k in range(0, len(fields), 2)})
    charmm = {
        "bonds": c["BOND"], "angles": c["ANGL"], "Urey-Bradley": c["UREY"],
        "dihedrals": c["DIHE"], "impropers": c["IMPR"], "CMAP": c["CMAP"],
        "Lennard-Jones": c["VDW"] + c["IMNB"],
        "Coulomb": c["ELEC"] + c["IMEL"] + c["EWKS"] + c["EWSE"] + c["EWEX"]}

    # GROMACS, a rerun.
    rtol = math.erfc(KAPPA * 10.0 * CUTOFF / 10.0)
    write(os.path.join(w, "rerun.mdp"), GROMACS.format(
        rc=CUTOFF / 10, rs=SWITCH / 10, nx=GRID[0], ny=GRID[1], nz=GRID[2],
        order=ORDER, rtol=rtol))
    run([args.gmx, "grompp", "-f", "rerun.mdp", "-c", "system.gro", "-p",
         "system.top", "-o", "rerun.tpr", "-maxwarn", "5"], w, "grompp.log")
    run([args.gmx, "mdrun", "-s", "rerun.tpr", "-rerun", "system.gro", "-nt",
         "8", "-deffnm", "rerun"], w, "rerun.out")
    write(os.path.join(w, "energy-select.txt"),
          "Bond\nU-B\nProper-Dih.\nImproper-Dih.\nCMAP-Dih.\nLJ-14\n"
          "Coulomb-14\nLJ-(SR)\nCoulomb-(SR)\nCoul.-recip.\n\n")
    run([args.gmx, "energy", "-f", "rerun.edr", "-o", "rerun.xvg"], w,
        "energy.log", stdin="energy-select.txt")
    legend, values = {}, None
    for line in open(os.path.join(w, "rerun.xvg")):
        m = re.match(r'@ s(\d+) legend "(.*)"', line)
        if m:
            legend[m.group(2)] = int(m.group(1)) + 1
        elif not line.startswith(("#", "@")):
            values = [float(v) for v in line.split()]
    g = {name: values[column] / 4.184 for name, column in legend.items()}
    gromacs = {
        "bonds": g["Bond"], "angles+Urey-Bradley": g["U-B"],
        "dihedrals": g["Proper Dih."], "impropers": g["Improper Dih."],
        "CMAP": g["CMAP Dih."], "Lennard-Jones": g["LJ-14"] + g["LJ (SR)"],
        "Coulomb": g["Coulomb-14"] + g["Coulomb (SR)"] + g["Coul. recip."]}

    # MDIR, from the files of CHARMM and from the topology of GROMACS with
    # each force switch.
    mdir = {}
    native = MDIR.format(
        cutoff=CUTOFF, switch=SWITCH, modifier="POWER_FORCE_SWITCH",
        pairlist=CUTOFF + 1.0, kappa=KAPPA, order=ORDER,
        grid=", ".join(str(v) for v in GRID))
    files = ", ".join(f'"{args.toppar}/{f}"' for f in
                      ("top_all36_prot.rtf", "par_all36m_prot.prm",
                       "toppar_water_ions.str"))
    native = native.replace(
        'topology    = "system.top"\ncoordinates = "system.gro"\n'
        'format      = "GROMACS"\ndefines     = ["FLEXIBLE"]\n',
        'topology    = "system.psf"\ncoordinates = "system-charmm.crd"\n'
        f'format      = "CHARMM"\nparameters  = [{files}]\n')
    native = native.replace('type = "PERIODIC"\n',
                            'type = "PERIODIC"\nbox  = [%s]\n' %
                            ", ".join(str(v) for v in CELL))
    write(os.path.join(w, "native.toml"), native)
    for modifier in ("NATIVE", "POWER_FORCE_SWITCH", "FORCE_SWITCH"):
        name = modifier.lower()
        if modifier != "NATIVE":
            write(os.path.join(w, f"{name}.toml"), MDIR.format(
                cutoff=CUTOFF, switch=SWITCH, modifier=modifier,
                pairlist=CUTOFF + 1.0, kappa=KAPPA, order=ORDER,
                grid=", ".join(str(v) for v in GRID)))
        run([args.mdir, "run", f"{name}.toml"], w, f"{name}.log")
        t = {m.group(1): float(m.group(2)) for m in re.finditer(
            r"^MDIR:   (.+?)\s+(-?[0-9.]+)$",
            open(os.path.join(w, f"{name}.log")).read(), re.M)}
        mdir[modifier] = {
            "bonds": t["bonds"], "angles": t["angles"],
            "Urey-Bradley": t["Urey-Bradley"],
            "dihedrals": t["dihedrals"], "impropers": t["harmonic impropers"],
            "CMAP": t["CMAP"],
            "Lennard-Jones": t["Lennard-Jones"] + t["Lennard-Jones 1-4"],
            "Coulomb": t["Coulomb"] + t["Coulomb 1-4"] + t["Coulomb excluded"]
            + t["Coulomb reciprocal"] + t["Coulomb self"]}

    ours = mdir["NATIVE"]
    print("kcal/mol; MDIR from the PSF and the files of CHARMM against "
          "CHARMM (VFSWITCH),")
    print("MDIR from the topology of GROMACS with FORCE_SWITCH against "
          "GROMACS (force-switch)")
    print(f"{'term':16s} {'CHARMM':>14s} {'MDIR':>14s} {'relative':>9s}"
          f" {'GROMACS':>14s} {'MDIR':>14s} {'relative':>9s}")
    theirs = mdir["FORCE_SWITCH"]
    for name in ("bonds", "angles", "Urey-Bradley", "dihedrals", "impropers",
                 "CMAP", "Lennard-Jones", "Coulomb"):
        a, b = charmm[name], ours[name]
        if name in ("angles", "Urey-Bradley"):
            gname = "angles+Urey-Bradley"
            x = gromacs[gname] if name == "angles" else None
            y = theirs["angles"] + theirs["Urey-Bradley"]
        else:
            x, y = gromacs[name], theirs[name]
        line = f"{name:16s} {a:14.6f} {b:14.6f} {(b - a) / abs(a):9.1e}"
        if x is not None:
            line += f" {x:14.6f} {y:14.6f} {(y - x) / abs(x):9.1e}"
        print(line)
    print("Coulomb: CHARMM's constant is 332.0716 kcal/mol Å/e², that of the "
          "topology of GROMACS 138.935458 kJ/mol nm/e² (332.0637), 2.38e-5 "
          "less.")


# --------------------------------------------------------------------------
# hexagonal
# --------------------------------------------------------------------------

HEXAGONAL = (36.0, 36.0, 38.0, 90.0, 90.0, 120.0)
HEXAGONAL_GRID = (36, 36, 40)


def symmetric_cell(a, b, c, alpha, beta, gamma):
    """The rows of G^(1/2), the cell of CHARMM in its symmetric frame, for
    the metric G of the lengths and angles, by the iteration of Denman and
    Beavers."""
    ca, cb, cg = (math.cos(math.radians(t)) for t in (alpha, beta, gamma))
    g = [[a * a, a * b * cg, a * c * cb],
         [a * b * cg, b * b, b * c * ca],
         [a * c * cb, b * c * ca, c * c]]

    def inverse(m):
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        return [[(m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3]
                  - m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3])
                 / det for j in range(3)] for i in range(3)]

    y, z = g, [[float(i == j) for j in range(3)] for i in range(3)]
    for _ in range(60):
        yi, zi = inverse(y), inverse(z)
        y, z = ([[0.5 * (y[i][j] + zi[i][j]) for j in range(3)]
                 for i in range(3)],
                [[0.5 * (z[i][j] + yi[i][j]) for j in range(3)]
                 for i in range(3)])
    return y, inverse(y)


def hexagonal(args, work):
    w = os.path.join(work, "hexagonal")
    os.makedirs(w, exist_ok=True)
    cell, inv = symmetric_cell(*HEXAGONAL)
    center = [0.5 * sum(cell[k][i] for k in range(3)) for i in range(3)]
    origin = [0.5 * CELL[i] - center[i] for i in range(3)]
    # The waters of the build whose oxygens are within the cell.
    residues = {}
    for line in open(os.path.join(work, "trpcage",
                                  "system-charmm.crd")).read().splitlines()[4:]:
        f = line.split()
        if len(f) >= 9 and f[2] == "TIP3":
            residues.setdefault((f[7], f[8]), []).append(f)
    kept = []
    for members in residues.values():
        o = next(m for m in members if m[3] == "OH2")
        x = [float(o[4 + i]) - origin[i] for i in range(3)]
        frac = [sum(x[k] * inv[k][i] for k in range(3)) for i in range(3)]
        if all(0.0 <= v < 1.0 for v in frac):
            kept.append(members)
    lines = ["* TIP3P WATERS IN A HEXAGONAL CELL", "*",
             f"{3 * len(kept):10d}  EXT"]
    atom = 0
    for number, members in enumerate(kept, 1):
        for m in members:
            atom += 1
            x = [float(m[4 + i]) - origin[i] for i in range(3)]
            lines.append(f"{atom:10d}{number:10d}  {'TIP3':<8s}  {m[3]:<8s}"
                         f"{x[0]:20.10f}{x[1]:20.10f}{x[2]:20.10f}  "
                         f"{'WAT':<8s}  {number:<8d}{0.0:20.10f}")
    write(os.path.join(w, "water.crd"), "\n".join(lines) + "\n")
    crystal = ("crystal define hexagonal %.1f %.1f %.1f %.1f %.1f %.1f\n"
               % HEXAGONAL
               + "crystal build cutoff 16.0 noper 0\n"
               "image byres xcen 0.0 ycen 0.0 zcen 0.0 select all end\n"
               "nbonds atom vatom cdie eps 1.0 vfswitch cutnb 16.0 cutim 16.0 "
               "ctofnb 12.0 ctonnb 10.0 -\n"
               f"  ewald pmewald kappa {KAPPA} spline order {ORDER} "
               f"fftx {HEXAGONAL_GRID[0]} ffty {HEXAGONAL_GRID[1]} "
               f"fftz {HEXAGONAL_GRID[2]} inbfrq -1 imgfrq -1\n")
    write(os.path.join(w, "build.inp"), f"""* waters in a hexagonal cell, minimized with PME
*
{toppar(args)}
read sequence TIP3 {len(kept)}
generate WAT setup noangle nodihedral
read coor card name water.crd
{crystal}
mini sd nstep 300 nprint 100
write psf card name hexa.psf
* waters in a hexagonal cell
*
write coor card name hexa.crd
* waters in a hexagonal cell
*
stop
""")
    run([args.charmm, "-i", "build.inp"], w, "build.out")
    write(os.path.join(w, "energy.inp"), f"""* waters in a hexagonal cell: the energy of the coordinates as written
*
{toppar(args)}
read psf card name hexa.psf
read coor card name hexa.crd
{crystal}
energy
echo TERMS BOND ?BOND ANGL ?ANGL VDW ?VDW IMNB ?IMNB ELEC ?ELEC IMEL ?IMEL
echo TERMS EWKS ?EWKS EWSE ?EWSE EWEX ?EWEX
stop
""")
    run([args.charmm, "-i", "energy.inp"], w, "energy.out")
    c = {}
    for line in re.findall(r"^\s*TERMS (.*)$",
                           open(os.path.join(w, "energy.out")).read(), re.M):
        f = line.split()
        c.update({f[k]: float(f[k + 1]) for k in range(0, len(f), 2)})
    charmm = {"bonds": c["BOND"], "angles": c["ANGL"],
              "Lennard-Jones": c["VDW"] + c["IMNB"],
              "Coulomb real": c["ELEC"] + c["IMEL"], "Coulomb excluded":
              c["EWEX"], "Coulomb reciprocal": c["EWKS"],
              "Coulomb self": c["EWSE"]}
    files = ", ".join(f'"{args.toppar}/{f}"' for f in
                      ("top_all36_prot.rtf", "par_all36m_prot.prm",
                       "toppar_water_ions.str"))
    write(os.path.join(w, "hexa.toml"), f"""[input]
format      = "CHARMM"
topology    = "hexa.psf"
coordinates = "hexa.crd"
parameters  = [{files}]

[output]
energy_interval = 0

[energy]
cutoff            = {CUTOFF}
switch_distance   = {SWITCH}
lennard_jones_modifier = "POWER_FORCE_SWITCH"
dispersion_correction  = "NONE"
pairlist_distance = {CUTOFF + 1.0}
electrostatics    = "PME"

[pme]
beta  = {KAPPA}
order = {ORDER}
grid  = [{", ".join(str(v) for v in HEXAGONAL_GRID)}]

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.001
steps      = 0

[ensemble]
ensemble = "NVE"

[constraints]
hydrogen_bonds = false
rigid_water    = false

[boundary]
type = "PERIODIC"
box  = [{", ".join(str(v) for v in HEXAGONAL)}]

[execution]
target    = "CPU"
precision = "DOUBLE"
""")
    run([args.mdir, "run", "hexa.toml"], w, "hexa.log")
    t = {m.group(1): float(m.group(2)) for m in re.finditer(
        r"^MDIR:   (.+?)\s+(-?[0-9.]+)$",
        open(os.path.join(w, "hexa.log")).read(), re.M)}
    ours = {"bonds": t["bonds"], "angles": t["angles"],
            "Lennard-Jones": t["Lennard-Jones"], "Coulomb real": t["Coulomb"],
            "Coulomb excluded": t["Coulomb excluded"],
            "Coulomb reciprocal": t["Coulomb reciprocal"],
            "Coulomb self": t["Coulomb self"]}
    ratio = 332.0716 / 332.0637
    print(f"{len(kept)} waters in a hexagonal cell; kcal/mol, MDIR from the "
          "PSF and the CRD against CHARMM; the electrostatics over the ratio "
          f"of the Coulomb constants, {ratio:.7f}")
    print(f"{'term':20s} {'CHARMM':>16s} {'MDIR':>16s} {'relative':>9s}")
    for name, a in charmm.items():
        b = ours[name] * (ratio if name.startswith("Coulomb") else 1.0)
        print(f"{name:20s} {a:16.6f} {b:16.6f} {(b - a) / abs(a):9.1e}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("work")
    parser.add_argument("--mdir", required=True)
    parser.add_argument("--charmm", default="charmm")
    parser.add_argument("--toppar", default=os.environ.get("CHARMM_TOPPAR"))
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--packmol", default="packmol")
    parser.add_argument("--parmed-python", default="python3")
    parser.add_argument("--phases",
                        default="pair,build,convert,terms,hexagonal")
    args = parser.parse_args()
    if not args.toppar:
        sys.exit("give --toppar or CHARMM_TOPPAR: the directory of "
                 "top_all36_prot.rtf, par_all36m_prot.prm, and "
                 "toppar_water_ions.str")
    args.toppar = os.path.abspath(args.toppar)
    args.mdir = os.path.abspath(args.mdir)
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    phases = args.phases.split(",")
    for phase, function in (("pair", pair), ("build", build),
                            ("convert", convert), ("terms", terms),
                            ("hexagonal", hexagonal)):
        if phase in phases:
            function(args, work)


if __name__ == "__main__":
    main()
