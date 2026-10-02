#!/usr/bin/env python3
"""Compares the structure that the engines of run.py sample on 6n4o
(Argonaute2 with miR-122): the RMSD of the backbone of the protein, the
RMSD of the phosphates of the RNA in the frame of the protein, the radius
of gyration of the complex, and the RMSF by residue.

    scripts/benchmarks/mdbench/structure.py WORK INPUTS [--cpptraj CPPTRAJ]
        [--engines pmemd,gromacs,openmm,mdir] [--plot FILE]
        [--solute OUT]

WORK holds the directories of run.py with their trajectories (run.py
--frames 10000 --steps 2000000 for 200 frames over 2 ns);
INPUTS is benchmark_6n4o_pmemd, whose prmtop.parm7 and restart.rst7 every
engine starts from, so the atoms are in one order. Each engine writes
whole molecules but not in the same images (MDIR keeps the image of the
sorted state, about 178 Å from the protein for the RNA); cpptraj images
the other molecules of the solute nearest to the protein, fits the
backbone of the protein to the restart, and measures.
The RMSF is over the whole run and over each half; the correlation of the
halves of one engine is the noise against which the correlation of two
engines is read. With --solute, the fitted solute of each engine is
written to OUT/<engine>.dcd with OUT/solute.parm7, for a movie.
"""
import argparse
import math
import os
import subprocess
import sys

TRAJECTORIES = {"pmemd": "mdcrd.nc", "gromacs": "traj_comp.xtc",
                "openmm": "trajectory.nc", "mdir": "md.dcd"}
PROTEIN = ":1-859"
RNA = ":860-901"

SCRIPT = """parm {prmtop}
reference {restart} [start]
trajin {trajectory} {first} last
autoimage anchor {protein}
rms backbone {protein}@CA,C,N ref [start] out rmsd.dat
rms rna {rna}@P ref [start] nofit out rmsd.dat
radgyr complex ({protein},{rna})&!@H= out rg.dat
atomicfluct ca {protein}@CA byres out rmsf-protein.dat
atomicfluct ca1 {protein}@CA byres out rmsf-protein.dat stop {half}
atomicfluct ca2 {protein}@CA byres out rmsf-protein.dat start {next}
atomicfluct p {rna}@P byres out rmsf-rna.dat
atomicfluct p1 {rna}@P byres out rmsf-rna.dat stop {half}
atomicfluct p2 {rna}@P byres out rmsf-rna.dat start {next}
{solute}run
"""

# The restart has the RNA in another image too; image it as the frames are.
START = """parm {prmtop}
trajin {restart}
autoimage anchor {protein}
trajout start.rst7 restart
run
"""

SOLUTE = """strip !({protein},{rna}) parmout {out}/solute.parm7
trajout {out}/{engine}.dcd
"""


def frames(cpptraj, prmtop, trajectory):
    """The number of frames of a trajectory, as cpptraj reads it."""
    result = subprocess.run([cpptraj, "-p", prmtop, "-y", trajectory,
                             "-tl"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("Frames:"):
            return int(line.split()[1])
    sys.exit(f"cpptraj cannot read {trajectory}:\n{result.stdout}"
             f"{result.stderr}")


def table(path):
    """The columns of a data file of cpptraj, by the names of its header."""
    lines = open(path).read().splitlines()
    names = lines[0].lstrip("#").split()
    columns = {name: [] for name in names}
    for line in lines[1:]:
        for name, value in zip(names, line.split()):
            columns[name].append(float(value))
    return columns


def mean(values):
    return sum(values) / len(values)


def correlation(a, b):
    ma, mb = mean(a), mean(b)
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    return sab / math.sqrt(saa * sbb)


def cpptraj(args, script, out):
    open(os.path.join(out, "cpptraj.in"), "w").write(script)
    with open(os.path.join(out, "cpptraj.log"), "w") as log:
        result = subprocess.run([args.cpptraj, "-i", "cpptraj.in"], cwd=out,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"cpptraj failed; see {os.path.join(out, 'cpptraj.log')}")


def start(args):
    """The restart with the solute imaged nearest to the protein."""
    os.makedirs(args.out, exist_ok=True)
    cpptraj(args, START.format(prmtop=os.path.join(args.inputs,
                                                   "prmtop.parm7"),
                               restart=os.path.join(args.inputs,
                                                    "restart.rst7"),
                               protein=PROTEIN), args.out)


def analyze(args, engine):
    work = os.path.join(args.work, engine)
    trajectory = os.path.join(work, TRAJECTORIES[engine])
    prmtop = os.path.join(args.inputs, "prmtop.parm7")
    count = frames(args.cpptraj, prmtop, trajectory)
    # GROMACS writes the first step as a frame, the others do not: start
    # where the count leaves the same steps for all.
    first = count - args.count + 1
    if first < 1:
        sys.exit(f"{engine}: {count} frames, fewer than {args.count}")
    half = args.count // 2
    solute = SOLUTE.format(protein=PROTEIN, rna=RNA, engine=engine,
                           out=os.path.abspath(args.solute)) \
        if args.solute else ""
    script = SCRIPT.format(prmtop=prmtop,
                           restart=os.path.join(args.out, "start.rst7"),
                           trajectory=trajectory, first=first,
                           protein=PROTEIN, rna=RNA, half=half,
                           next=half + 1, solute=solute)
    out = os.path.join(args.out, engine)
    os.makedirs(out, exist_ok=True)
    if args.solute:
        os.makedirs(args.solute, exist_ok=True)
    cpptraj(args, script, out)
    rmsd = table(os.path.join(out, "rmsd.dat"))
    return {"frames": count,
            "backbone": rmsd["backbone"], "rna": rmsd["rna"],
            "rg": table(os.path.join(out, "rg.dat"))["complex"],
            "protein": table(os.path.join(out, "rmsf-protein.dat")),
            "nucleic": table(os.path.join(out, "rmsf-rna.dat"))}


def report(results, count):
    engines = list(results)
    print(f"Over the last {count} frames of each run; second half for the "
          "means of the series.")
    print(f"{'':10}{'frames':>7}{'bb RMSD':>9}{'RNA P':>8}{'Rg':>8}"
          f"{'RMSF CA':>9}{'RMSF P':>8}{'halves CA':>10}{'halves P':>10}")
    for engine, r in results.items():
        second = slice(len(r["backbone"]) // 2, None)
        print(f"{engine:10}{r['frames']:7d}"
              f"{mean(r['backbone'][second]):9.2f}"
              f"{mean(r['rna'][second]):8.2f}{mean(r['rg'][second]):8.2f}"
              f"{mean(r['protein']['ca']):9.2f}{mean(r['nucleic']['p']):8.2f}"
              f"{correlation(r['protein']['ca1'], r['protein']['ca2']):10.2f}"
              f"{correlation(r['nucleic']['p1'], r['nucleic']['p2']):10.2f}")
    print("\nThe correlation of the RMSF by residue between engines "
          "(CA of the protein above the diagonal, P of the RNA below):")
    print(f"{'':10}" + "".join(f"{e:>9}" for e in engines))
    for i, a in enumerate(engines):
        row = f"{a:10}"
        for j, b in enumerate(engines):
            if i == j:
                row += f"{'':>9}"
            else:
                key, column = ("protein", "ca") if j > i else ("nucleic", "p")
                row += f"{correlation(results[a][key][column], results[b][key][column]):9.2f}"
        print(row)


def plot(results, path, count, interval):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for engine, r in results.items():
        time = [(i + 1) * interval for i in range(count)]
        axes[0, 0].plot(time, r["backbone"], label=engine, lw=1)
        axes[0, 1].plot(time, r["rg"], label=engine, lw=1)
        axes[1, 0].plot(r["protein"]["Res"], r["protein"]["ca"],
                        label=engine, lw=0.8)
        axes[1, 1].plot(r["nucleic"]["Res"], r["nucleic"]["p"],
                        label=engine, lw=1, marker=".")
    axes[0, 0].set(xlabel="time (ns)", ylabel="RMSD of the backbone (Å)")
    axes[0, 1].set(xlabel="time (ns)", ylabel="radius of gyration (Å)")
    axes[1, 0].set(xlabel="residue", ylabel="RMSF of CA (Å)")
    axes[1, 1].set(xlabel="residue", ylabel="RMSF of P (Å)")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("inputs")
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--engines", default="pmemd,gromacs,openmm,mdir")
    parser.add_argument("--count", type=int, default=200,
                        help="frames per run, without a frame of step 0")
    parser.add_argument("--interval", type=float, default=0.01,
                        help="ns between frames, for the plot")
    parser.add_argument("--out", default=None,
                        help="where cpptraj writes (default WORK/structure)")
    parser.add_argument("--plot")
    parser.add_argument("--solute")
    args = parser.parse_args()
    args.work = os.path.abspath(args.work)
    args.inputs = os.path.abspath(args.inputs)
    args.out = os.path.abspath(args.out or os.path.join(args.work,
                                                        "structure"))
    start(args)
    results = {e: analyze(args, e) for e in args.engines.split(",")}
    report(results, args.count)
    if args.plot:
        plot(results, args.plot, args.count, args.interval)


if __name__ == "__main__":
    main()
