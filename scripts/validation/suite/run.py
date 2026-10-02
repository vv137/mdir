#!/usr/bin/env python3
"""Compares the energy terms and the forces of MDIR at the start of each
system of the Amber benchmark suite with those of pmemd or sander.

    scripts/validation/suite/run.py WORK --mdir MDIR --amber PROGRAM
        [--suite DIR] [--threads N] [SYSTEM ...]

DIR is the work directory of scripts/benchmarks/amber/bench.py (default
$MDIR_BENCH_DIR), which holds the suite as it is distributed
(Amber24_Benchmark_Suite/PME). Both programs read its files, prmtop and
inpcrd, as they are: two of the topologies are in the format before Amber
7. PROGRAM is pmemd on the CPU or sander, which evaluate in double
precision; sander refuses Factor IX, whose dihedrals of several terms
exceed its MAXDUP of 2000. The script needs numpy, scipy, and h5py.

Both programs evaluate the positions of the restart file without
constraints, with a cutoff of 8 Å, particle mesh Ewald of the same beta
(erfc(beta r_c) = 1e-5), grid (the least even numbers of points of factors
2, 3, and 5 at a spacing of at most 1 Å, which pmemd needs), and order 4,
the influence function of Amber ('influence = "OPTIMAL"'), and no
correction for the dispersion, which the two programs average differently
(white paper, Section 5.4). MDIR runs on the CPU in double precision from
a copy of the topology whose charges are scaled by
sqrt(332.0522173 / 332.0637133), so that its Coulomb constant (CODATA 2018)
gives the energies of Amber's; Amber keeps the net force of particle mesh
Ewald (netfrc=0).

The energies are those of step 0. The forces are those that MDIR writes
into a checkpoint after one step of 1e-9 ps, and that Amber writes after
the same step (ntwf=1); the positions move by less than 1e-8 Å between.
WORK/SYSTEM receives the inputs and outputs of both; WORK/results.json the
differences.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

# The systems and their directories in the suite.
SYSTEMS = {
    "jac_nve": "JAC_production_NVE",
    "jac_npt": "JAC_production_NPT",
    "jac_nve_4fs": "JAC_production_NVE_4fs",
    "jac_npt_4fs": "JAC_production_NPT_4fs",
    "factorix_nve": "FactorIX_production_NVE",
    "factorix_npt": "FactorIX_production_NPT",
    "cellulose_nve": "Cellulose_production_NVE",
    "cellulose_npt": "Cellulose_production_NPT",
    "stmv_npt_4fs": "STMV_production_NPT_4fs",
}
CUTOFF = 8.0
TOLERANCE = 1e-5
KCAL_A = 1.0 / 4.184 / 10.0   # kJ/(mol nm) to kcal/(mol Å)

# The terms of sander's step 0 and the terms of MDIR that sum to each.
TERMS = [
    ("BOND", ["bonds"]),
    ("ANGLE", ["angles"]),
    ("DIHED", ["dihedrals"]),
    ("1-4 NB", ["Lennard-Jones 1-4"]),
    ("1-4 EEL", ["Coulomb 1-4"]),
    ("VDWAALS", ["Lennard-Jones"]),
    ("EELEC", ["Coulomb", "Coulomb excluded", "Coulomb reciprocal",
               "Coulomb self"]),
]


def grid_size(length):
    """The least even number of points >= length (1 Å apart) of factors 2,
    3, and 5."""
    n = math.ceil(length - 1e-9)
    while True:
        m = n
        if m % 2:
            n += 1
            continue
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def beta_of(cutoff, tolerance):
    from scipy.special import erfc
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if erfc(mid * cutoff) > tolerance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def read_box(rst7):
    with open(rst7) as f:
        last = f.read().rstrip("\n").split("\n")[-1]
    return [float(v) for v in last.split()[:3]]


def scale_charges(src, dst):
    """Copies a topology with the section of the charges scaled. In the
    format before Amber 7 the charges follow the title, 30 pointers in
    lines of 12, and the names of the atoms in lines of 20."""
    lines = open(src).read().split("\n")
    if lines[0].startswith("%VERSION"):
        flags = [line.rstrip() for line in lines]
        start = flags.index("%FLAG CHARGE") + 2
        assert lines[start - 1].startswith("%FORMAT(5E16.8)")
        natom = int(lines[flags.index("%FLAG POINTERS") + 2][:8])
    else:
        natom = int(lines[1][:6])
        start = 1 + 3 + (natom + 19) // 20
    factor = math.sqrt(332.0522173 / 332.0637133)
    values = []
    for line in lines[start:start + (natom + 4) // 5]:
        values += [float(line[i:i + 16]) for i in range(0, len(line), 16)
                   if line[i:i + 16].strip()]
    assert len(values) == natom
    scaled = [f"{v * factor:16.8E}" for v in values]
    rows = ["".join(scaled[i:i + 5]) for i in range(0, natom, 5)]
    lines[start:start + len(rows)] = rows
    open(dst, "w").write("\n".join(lines))


def write_mdir(path, beta, grid, threads):
    with open(path, "w") as f:
        f.write(f"""[input]
format      = "AMBER"
topology    = "scaled.prmtop"
coordinates = "inpcrd"
[output]
energy_interval = 0
checkpoint = "mdir.h5"
checkpoint_interval = 1
[energy]
cutoff            = {CUTOFF}
pairlist_distance = {CUTOFF + 1.0}
electrostatics    = "PME"
dispersion_correction = "NONE"
[pme]
beta      = {beta!r}
grid      = [{grid[0]}, {grid[1]}, {grid[2]}]
order     = 4
influence = "OPTIMAL"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 1e-9
steps      = 1
[ensemble]
ensemble    = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
[execution]
target    = "CPU"
precision = "DOUBLE"
threads   = {threads}
""")


def write_sander(path, beta, grid):
    with open(path, "w") as f:
        f.write(f"""Terms and forces at the input coordinates
 &cntrl
  imin=0, ntx=1, irest=0, tempi=0.0, ig=1, nstlim=1, dt=0.000000001,
  ntb=1, cut={CUTOFF}, ntc=1, ntf=1, ntpr=1, ntwx=1, ntwf=1, ioutfm=1,
  ntt=0,
 /
 &ewald
  nfft1={grid[0]}, nfft2={grid[1]}, nfft3={grid[2]}, order=4,
  ew_coeff={beta!r}, vdwmeth=0, netfrc=0,
 /
""")


def mdir_terms(log):
    terms = {}
    inside = False
    for line in open(log):
        if "the terms at the start" in line:
            inside = True
            continue
        if inside:
            m = re.match(r"MDIR:\s+(\S.*?)\s+(-?\d+\.\d+)\s*$", line)
            if not m:
                break
            terms[m.group(1)] = float(m.group(2))
    return terms


def sander_terms(mdout):
    text = open(mdout).read()
    block = text.split("NSTEP =        0", 1)[1].split("-" * 20, 1)[0]
    return {k.strip(): float(v) for k, v in
            re.findall(r"([A-Z0-9][A-Z0-9 \-]*?)\s*=\s*(-?\d+\.\d+)", block)}


def compare_forces(h5, frc):
    import h5py
    from scipy.io import netcdf_file
    with h5py.File(h5, "r") as f:
        ours = f["particles/all/force/value"][0] * KCAL_A
        masses = f["particles/all/mass"][:]
        ids = f["particles/all/id"][:]
    order = np.argsort(ids)
    ours, masses = ours[order], masses[order]
    with netcdf_file(frc, "r", mmap=False) as f:
        theirs = np.array(f.variables["forces"][0], dtype=float)
    real = masses > 0
    diff = np.linalg.norm(ours[real] - theirs[real], axis=1)
    rms = math.sqrt(np.mean(np.sum(theirs[real] ** 2, axis=1)))
    return {"rms_force": rms,
            "rms_difference": math.sqrt(np.mean(diff ** 2)) / rms,
            "largest_difference": float(diff.max()) / rms}


def run(name, args):
    here = os.path.join(args.work, name)
    os.makedirs(here, exist_ok=True)
    source = os.path.join(args.suite, "Amber24_Benchmark_Suite", "PME",
                          SYSTEMS[name])
    for f in ("prmtop", "inpcrd"):
        target = os.path.join(here, f)
        if not os.path.exists(target):
            os.symlink(os.path.join(source, f), target)
    box = read_box(os.path.join(here, "inpcrd"))
    grid = [grid_size(length) for length in box]
    beta = beta_of(CUTOFF, TOLERANCE)

    scale_charges(os.path.join(here, "prmtop"),
                  os.path.join(here, "scaled.prmtop"))
    write_sander(os.path.join(here, "sander.in"), beta, grid)
    sander = subprocess.Popen(
        [args.amber, "-O", "-i", "sander.in",
         "-p", "prmtop", "-c", "inpcrd", "-o", "sander.out",
         "-x", "sander.nc", "-frc", "sander.frc.nc", "-r", "sander.rst"],
        cwd=here)
    write_mdir(os.path.join(here, "mdir.toml"), beta, grid, args.threads)
    with open(os.path.join(here, "mdir.log"), "w") as log:
        subprocess.run([args.mdir, "run", "mdir.toml"], cwd=here, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    if sander.wait() != 0:
        sys.exit(f"{name}: {args.amber} failed")

    ours = mdir_terms(os.path.join(here, "mdir.log"))
    theirs = sander_terms(os.path.join(here, "sander.out"))
    result = {"box": box, "grid": grid, "beta": beta,
              "terms": {}}
    for key, parts in TERMS:
        a = sum(ours[p] for p in parts)
        b = theirs[key]
        result["terms"][key] = {"mdir": a, "sander": b,
                                "relative": abs(a - b) / max(abs(b), 1e-12)}
    result["forces"] = compare_forces(os.path.join(here, "mdir.h5"),
                                      os.path.join(here, "sander.frc.nc"))
    return result


def main():
    # `run.py --scale-only SOURCE TARGET` only writes the topology with its
    # charges scaled, for scripts/validation/nve.
    if len(sys.argv) == 4 and sys.argv[1] == "--scale-only":
        scale_charges(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("--mdir", required=True)
    parser.add_argument("--amber", required=True)
    parser.add_argument("--suite", default=os.environ.get("MDIR_BENCH_DIR"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("systems", nargs="*", default=list(SYSTEMS))
    args = parser.parse_args()
    args.work = os.path.abspath(args.work)
    args.mdir = os.path.abspath(args.mdir)
    args.amber = os.path.abspath(args.amber)
    os.makedirs(args.work, exist_ok=True)

    path = os.path.join(args.work, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for name in args.systems:
        results[name] = run(name, args)
        with open(path, "w") as f:
            json.dump(results, f, indent=1)
        r = results[name]
        print(f"== {name}: grid {r['grid']}, beta {r['beta']:.6f}")
        for key, t in r["terms"].items():
            print(f"  {key:8s} {t['mdir']:16.4f} {t['sander']:16.4f} "
                  f"{t['relative']:.1e}")
        f = r["forces"]
        print(f"  forces: rms {f['rms_force']:.3f} kcal/mol/Å, difference "
              f"rms {f['rms_difference']:.1e}, largest "
              f"{f['largest_difference']:.1e} of the rms force", flush=True)


if __name__ == "__main__":
    main()
