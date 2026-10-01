#!/usr/bin/env python3
"""Compares the forces of MDIR on every particle with those of sander and
of GROMACS, on the same coordinates.

    scripts/validation/forces/compare.py MDIR.h5 OTHER [--kind sander|gromacs]
        [--scale-coulomb S]

MDIR.h5 is a checkpoint of MDIR taken after one step of 1e-9 ps from zero
velocities, so its positions and forces are those of the input to 1e-15
relative. OTHER is the NetCDF file of the forces of sander (ntwf=1,
ioutfm=1, written the same way) or the xvg of `gmx traj -of` of a rerun of
GROMACS. Particles without mass (extra points) are left out: each program
moves their forces onto the atoms.

Prints the largest and the root-mean-square difference of the forces,
absolute (kcal/mol/Å) and relative to the root-mean-square force, and the
particle with the largest difference.
"""
import argparse

import h5py
import numpy as np

KCAL_A = 1.0 / 4.184 / 10.0   # kJ/(mol nm) to kcal/(mol Å)


def read_mdir(path):
    with h5py.File(path, "r") as f:
        forces = f["particles/all/force/value"][0] * KCAL_A
        masses = f["particles/all/mass"][:]
        ids = f["particles/all/id"][:]
    order = np.argsort(ids)
    return forces[order], masses[order]


def read_sander(path):
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as f:
        return np.array(f.variables["forces"][0], dtype=float)


def read_gromacs(path):
    rows = [line.split() for line in open(path)
            if not line.startswith(("#", "@"))]
    values = np.array([float(v) for v in rows[0][1:]])
    return values.reshape(-1, 3) * KCAL_A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mdir")
    parser.add_argument("other")
    parser.add_argument("--kind", choices=["sander", "gromacs"],
                        default="sander")
    args = parser.parse_args()
    ours, masses = read_mdir(args.mdir)
    theirs = (read_sander(args.other) if args.kind == "sander"
              else read_gromacs(args.other))
    real = masses > 0
    a, b = ours[real], theirs[real]
    diff = np.linalg.norm(a - b, axis=1)
    rms_force = np.sqrt(np.mean(np.sum(b * b, axis=1)))
    worst = int(np.argmax(diff))
    print(f"{real.sum()} particles with mass; rms force {rms_force:.4f} "
          f"kcal/mol/Å")
    print(f"largest difference {diff.max():.3e} kcal/mol/Å "
          f"({diff.max() / rms_force:.2e} of the rms force), particle "
          f"{np.flatnonzero(real)[worst] + 1}")
    print(f"rms difference {np.sqrt(np.mean(diff ** 2)):.3e} kcal/mol/Å "
          f"({np.sqrt(np.mean(diff ** 2)) / rms_force:.2e} of the rms force)")
    rel = diff / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    print(f"median relative difference per particle {np.median(rel):.2e}")


if __name__ == "__main__":
    main()
