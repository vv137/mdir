#!/usr/bin/env python3
"""Checks the virial of the model term of D270 against central
differences of the energy under strain (docs/metatomic-ml1.md, Section 10):
W_ab = -dE/d(eps_ab), the nine components, through MDIR's Python model on a
GPU. Each strained configuration, x' = x (I + eps) and H' = H (I + eps) with
the edge vectors as the rows of H, is turned back into MDIR's reduced,
lower-triangular frame by a rotation, which changes no energy, and is
evaluated by a Simulation of its own; the virial at zero strain is the one
that the run reports, the model's and that of the program's pressure.

The directory holds system.top and system.gro (water-box.py). Prints one JSON
object: the virial of the run, the one of the differences, their largest
difference, and the scale. The differences at h and h/2 are combined by
Richardson's extrapolation, (4 D(h/2) - D(h)) / 3, which removes the error
of order h^2; a smaller h would not do, as the model's own sums are not
deterministic (about 1e-3 kJ/mol in the energy).

usage: virial-strain.py DIR ARTIFACT [--precision Double|Mixed] [--h 5e-3]
"""
import argparse
import json
import os

import torch  # noqa: F401  the plugin takes the libtorch of this module
import numpy as np

import mdir


def evaluate(loaded, artifact, precision, positions=None, cell=None):
    """The potential energy (kJ/mol), the model's virial and the program's
    (kJ/mol, row by row) at `positions` (nm) in `cell` (rows, nm)."""
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = 0.2
    system.pairlist_distance = 0.25
    system.models = [mdir.ModelTerm("allegro", artifact)]
    if positions is not None:
        state.positions = np.ascontiguousarray(positions)
        given = mdir.Cell()
        given.diagonal = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
        given.tilt = np.array([cell[1, 0], cell[2, 0], cell[2, 1]])
        state.cell = given
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target = mdir.Target.GPU
    execution.precision = getattr(mdir.Precision, precision)
    sim = mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                       mdir.Schedule()))
    sim.run(0, energy=True)
    reached = sim.state()
    model = sim.model_energies()
    return (reached.energies["potential"], np.asarray(model["virial"]),
            reached.energies["virial"], np.asarray(reached.positions))


def lower(positions, cell):
    """`positions` and `cell` rotated so that the rows of the cell are lower
    triangular with a positive diagonal."""
    q, r = np.linalg.qr(cell.T)
    signs = np.sign(np.diag(r))
    q, r = q * signs, (r.T * signs).T
    return positions @ q, r.T


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("directory")
    p.add_argument("artifact")
    p.add_argument("--precision", default="Double")
    p.add_argument("--h", type=float, default=5e-3)
    a = p.parse_args()

    loaded = mdir.load_gromacs(os.path.join(a.directory, "system.top"),
                               os.path.join(a.directory, "system.gro"))
    state = loaded.make_state()
    x = np.asarray(state.positions)
    cell = np.asarray(state.cell.vectors)
    energy, model, trace, _ = evaluate(loaded, a.artifact, a.precision)

    def central(h):
        result = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                values = []
                for sign in (1.0, -1.0):
                    strain = np.eye(3)
                    strain[i, j] += sign * h
                    xs, hs = lower(x @ strain, cell @ strain)
                    values.append(evaluate(loaded, a.artifact, a.precision, xs, hs)[0])
                result[i, j] = -(values[0] - values[1]) / (2.0 * h)
        return result

    coarse, fine = central(a.h), central(a.h / 2.0)
    differences = (4.0 * fine - coarse) / 3.0
    print(json.dumps({
        "precision": a.precision, "h": a.h, "energy": energy,
        "virial": model.tolist(), "virial_differences": differences.tolist(),
        "max_difference_h": float(np.abs(model - coarse).max()),
        "max_difference_h2": float(np.abs(model - fine).max()),
        "max_difference": float(np.abs(model - differences).max()),
        "virial_scale": float(np.abs(model).max()),
        "trace": float(np.trace(model)), "program_trace": trace,
        "cell": cell.tolist()}))


if __name__ == "__main__":
    main()
