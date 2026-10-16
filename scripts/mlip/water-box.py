#!/usr/bin/env python3
"""Writes a box of water for the tests and the validation of the model term of
D270 (docs/metatomic-ml1.md, Section 7): a GROMACS topology of
flexible waters without classical terms (no bonds, no charges, and Lennard-Jones
of zero), with the elements in [ atomtypes ], and its coordinates, so that the
potential of a run is the model's alone.

The configurations are those of the feasibility study (Section 3.2): n^3
waters on a jittered lattice of spacing a (Angstrom) with random orientations,
in a cubic cell or a triclinic one of the given tilts; or liquid water at
29.9 A^3 per molecule (`--liquid`, Section 4.1).

With `--reference` it also evaluates NequIP's own calculator (the oracle of
ML1, Section 1) on the configuration and writes the energy (kJ/mol), the
forces (kJ/mol/nm), and the virial W = -dE/d(strain) (kJ/mol) to
reference.npz. That needs the environment of the plugin (torch, nequip, ase).
With `--minimize STEPS` the configuration is first relaxed with ASE's FIRE
on NequIP's calculator, in the fixed cell: ML1 refuses [minimize] with a
model term, and its runs of constant energy start from such a configuration.

usage: water-box.py OUTDIR [--n 3] [--spacing 5.0] [--tilt bx cx cy]
                           [--liquid N] [--seed 1]
                           [--reference PACKAGE.nequip.zip] [--device cuda]
                           [--minimize STEPS]
"""
import argparse
import os

import numpy as np

KJ_PER_EV = 96.48533212331
H2O = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                [0.0, -0.763239, -0.477047]])  # Angstrom, O H H


def lattice(n, spacing, seed, tilt):
    rng = np.random.default_rng(seed)
    water = H2O - H2O[0]
    positions = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                center = (np.array([i, j, k]) + 0.5) * spacing + rng.normal(0, 0.15, 3)
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                positions.extend(center + water @ q.T)
    length = n * spacing
    cell = np.diag([length] * 3)
    if tilt is not None:  # rows a, b, c, lower triangular
        cell = np.array([[length, 0, 0], [tilt[0] * length, length, 0],
                         [tilt[1] * length, tilt[2] * length, length]])
    return np.array(positions), cell


def liquid(count, seed):
    """`count` waters at 29.9 A^3 each on a jittered cubic lattice."""
    side = int(np.ceil(count ** (1.0 / 3.0)))
    length = (29.9 * count) ** (1.0 / 3.0)
    spacing = length / side
    rng = np.random.default_rng(seed)
    water = H2O - H2O[0]
    sites = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    chosen = rng.permutation(len(sites))[:count]
    positions = []
    for s in sorted(chosen):
        center = (np.array(sites[s]) + 0.5) * spacing + rng.normal(0, 0.1, 3)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        positions.extend(center + water @ q.T)
    return np.array(positions), np.diag([length] * 3)


def wrap(positions, cell):
    fraction = positions @ np.linalg.inv(cell)
    return (fraction - np.floor(fraction)) @ cell


def write(outdir, positions, cell, velocities=None):
    count = len(positions) // 3
    with open(os.path.join(outdir, "system.top"), "w") as f:
        f.write("; Flexible water without classical terms, for a learned potential.\n\n"
                "[ defaults ]\n  1  2  no  1.0  1.0\n\n"
                "[ atomtypes ]\n"
                "; name  at.num  mass     charge  ptype  sigma  epsilon\n"
                "  O     8       15.9994  0.0     A      0.0    0.0\n"
                "  H     1       1.0080   0.0     A      0.0    0.0\n\n"
                "[ moleculetype ]\n  MOL  1\n\n"
                "[ atoms ]\n"
                "  1  O  1  MOL  O   1  0.0  15.9994\n"
                "  2  H  1  MOL  H1  1  0.0   1.0080\n"
                "  3  H  1  MOL  H2  1  0.0   1.0080\n\n"
                "[ system ]\nwater for a learned potential\n\n"
                f"[ molecules ]\nMOL {count}\n")
    nm = positions / 10.0
    box = cell / 10.0
    with open(os.path.join(outdir, "system.gro"), "w") as f:
        f.write("water for a learned potential\n%d\n" % len(nm))
        names = ["O", "H1", "H2"]
        for i, x in enumerate(nm):
            f.write("%5d%-5s%5s%5d%12.7f%12.7f%12.7f" % (
                i // 3 + 1, "MOL", names[i % 3], (i + 1) % 100000, x[0], x[1], x[2]))
            if velocities is not None:
                f.write("%12.7f%12.7f%12.7f" % tuple(velocities[i]))
            f.write("\n")
        f.write("%12.7f %12.7f %12.7f %12.7f %12.7f %12.7f %12.7f %12.7f %12.7f\n" % (
            box[0, 0], box[1, 1], box[2, 2], box[0, 1], box[0, 2], box[1, 0],
            box[1, 2], box[2, 0], box[2, 1]))


def calculator(package, device):
    from nequip.integrations.ase import NequIPCalculator

    return NequIPCalculator._from_saved_model(package, device=device)


def minimize(package, positions, cell, device, steps):
    from ase import Atoms
    from ase.optimize import FIRE

    count = len(positions) // 3
    atoms = Atoms(["O", "H", "H"] * count, positions=positions, cell=cell, pbc=True)
    atoms.calc = calculator(package, device)
    FIRE(atoms, logfile=None).run(fmax=0.01, steps=steps)
    return atoms.get_positions()


def reference(package, positions, cell, device):
    from ase import Atoms

    count = len(positions) // 3
    atoms = Atoms(["O", "H", "H"] * count, positions=positions, cell=cell, pbc=True)
    atoms.calc = calculator(package, device)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    virial = -stress * atoms.get_volume()
    return energy * KJ_PER_EV, forces * KJ_PER_EV * 10.0, virial * KJ_PER_EV


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("outdir")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=5.0)
    parser.add_argument("--tilt", type=float, nargs=3)
    parser.add_argument("--liquid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimize", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="writes velocities of this temperature (K) to the .gro, "
                        "without the motion of the center of mass")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.liquid:
        positions, cell = liquid(args.liquid, args.seed)
    else:
        positions, cell = lattice(args.n, args.spacing, args.seed, args.tilt)
    if args.minimize:
        positions = minimize(args.reference, positions, cell, args.device, args.minimize)
    positions = wrap(positions, cell)
    # The coordinates are written to 1e-7 nm; the reference takes them so.
    positions = np.round(positions / 10.0, 7) * 10.0
    velocities = None
    if args.temperature > 0:
        # Maxwell-Boltzmann in nm/ps: sqrt(kT/m), kT in kJ/mol, m in amu.
        masses = np.array([15.9994, 1.008, 1.008] * (len(positions) // 3))
        rng = np.random.default_rng(args.seed + 1000)
        velocities = rng.normal(size=positions.shape) * np.sqrt(
            0.0083144626 * args.temperature / masses)[:, None]
        velocities -= (masses[:, None] * velocities).sum(0) / masses.sum()
        velocities = np.round(velocities, 7)
    write(args.outdir, positions, cell, velocities)
    if args.reference:
        e, f, w = reference(args.reference, positions, cell, args.device)
        np.savez(os.path.join(args.outdir, "reference.npz"), energy=e, forces=f,
                 virial=w, positions=positions / 10.0, cell=cell / 10.0)
        print("reference energy %.6f kJ/mol, %d particles" % (e, len(positions)))


if __name__ == "__main__":
    main()
