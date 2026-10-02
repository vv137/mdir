#!/usr/bin/env python3
"""Times analysis programs on one trajectory and compares what they compute.

    scripts/benchmarks/analysis/tools.py PRMTOP DCD [--cpptraj CPPTRAJ]
        [--tools cpptraj,mdtraj,mdanalysis] [--protein 1-859]

Each program reads every atom of every frame, fits the backbone (CA, C, N)
of the protein to the first frame, and computes the RMSD of the backbone
per frame, the RMSF of each CA, and the radius of gyration of the heavy
atoms of the protein, without weights. The protein must be whole in the
frames (no imaging is done). MDTraj and MDAnalysis run in the Python of
this script; cpptraj as a program. The table gives the time of each and
the largest difference of each quantity from cpptraj's.
"""
import argparse
import os
import subprocess
import tempfile
import time

import numpy as np


def cpptraj(args, first, last):
    out = tempfile.mkdtemp()
    mask = f":{first}-{last}"
    script = f"""parm {args.prmtop}
trajin {args.dcd}
rms bb {mask}@CA,C,N first out rmsd.dat
atomicfluct ca {mask}@CA byres out rmsf.dat
radgyr rg {mask}&!@H= out rg.dat
run
"""
    open(os.path.join(out, "in"), "w").write(script)
    start = time.perf_counter()
    subprocess.run([args.cpptraj, "-i", "in"], cwd=out, check=True,
                   capture_output=True)
    elapsed = time.perf_counter() - start

    def column(name, index):
        rows = [l.split() for l in open(os.path.join(out, name))
                if not l.startswith("#")]
        return np.array([float(r[index]) for r in rows])
    return elapsed, column("rmsd.dat", 1), column("rmsf.dat", 1), \
        column("rg.dat", 1)


def mdtraj(args, first, last):
    import mdtraj as md
    start = time.perf_counter()
    top = md.load_prmtop(args.prmtop)
    traj = md.load_dcd(args.dcd, top=top)
    protein = top.select(f"resid {first - 1} to {last - 1}")
    backbone = top.select(f"resid {first - 1} to {last - 1} and "
                          "(name CA or name C or name N)")
    ca = top.select(f"resid {first - 1} to {last - 1} and name CA")
    heavy = [i for i in protein if top.atom(i).element.symbol != "H"]
    traj.superpose(traj, 0, atom_indices=backbone)
    rmsd = md.rmsd(traj, traj, 0, atom_indices=backbone, precentered=False,
                   superpose=False) * 10.0
    xyz = traj.xyz[:, ca, :]
    rmsf = np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(-1).mean(0)) * 10.0
    rg = md.compute_rg(traj.atom_slice(heavy)) * 10.0
    return time.perf_counter() - start, rmsd, rmsf, rg


def mdanalysis(args, first, last):
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms
    start = time.perf_counter()
    u = mda.Universe(args.prmtop, args.dcd, topology_format="PARM7")
    sel = f"resid {first}-{last}"
    align.AlignTraj(u, u, select=f"{sel} and name CA C N",
                    in_memory=True).run()
    backbone = u.select_atoms(f"{sel} and name CA C N")
    ca = u.select_atoms(f"{sel} and name CA")
    heavy = u.select_atoms(f"{sel} and not name H*")
    reference = None
    rmsd, rg, xyz = [], [], []
    for ts in u.trajectory:
        position = backbone.positions.copy()
        if reference is None:
            reference = position
        rmsd.append(np.sqrt(((position - reference) ** 2).sum(-1).mean()))
        center = heavy.positions.mean(0)
        rg.append(np.sqrt(((heavy.positions - center) ** 2).sum(-1).mean()))
        xyz.append(ca.positions.copy())
    xyz = np.array(xyz)
    rmsf = np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(-1).mean(0))
    return time.perf_counter() - start, np.array(rmsd), rmsf, np.array(rg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prmtop")
    parser.add_argument("dcd")
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--tools", default="cpptraj,mdtraj,mdanalysis")
    parser.add_argument("--protein", default="1-859")
    args = parser.parse_args()
    args.prmtop = os.path.abspath(args.prmtop)
    args.dcd = os.path.abspath(args.dcd)
    first, last = (int(v) for v in args.protein.split("-"))
    runs = {"cpptraj": cpptraj, "mdtraj": mdtraj, "mdanalysis": mdanalysis}
    results = {t: runs[t](args, first, last) for t in args.tools.split(",")}
    base = results.get("cpptraj")
    print(f"{'tool':12s} {'seconds':>9s} {'RMSD (A)':>9s} {'RMSF':>7s} "
          f"{'Rg':>8s}   largest difference from cpptraj: RMSD, RMSF, Rg")
    for tool, (seconds, rmsd, rmsf, rg) in results.items():
        line = (f"{tool:12s} {seconds:9.1f} {rmsd[len(rmsd) // 2:].mean():9.4f} "
                f"{rmsf.mean():7.4f} {rg.mean():8.4f}")
        if base is not None and tool != "cpptraj":
            line += (f"   {np.abs(rmsd - base[1]).max():.1e} "
                     f"{np.abs(rmsf - base[2]).max():.1e} "
                     f"{np.abs(rg - base[3]).max():.1e}")
        print(line)


if __name__ == "__main__":
    main()
