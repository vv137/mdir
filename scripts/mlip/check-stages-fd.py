#!/usr/bin/env python3
"""Checks the forces and the virial of a model in stages (D[ml4-painn-stages],
docs/ml4-ir.md, Section 8) against central differences of its own energy
through MDIR's Python model on a GPU: `md-check-derivatives` (D184) runs on
the CPU, and the tiles of batched stages run on a device only.

For `--coordinates` particles' coordinates (the largest forces first), the
energy at x +- h and at x +- 2h gives the derivative to h^4 (Richardson);
for the trace of the virial, the energy of the system scaled isotropically
by 1 +- e (positions and cell). Prints one JSON object with the largest
difference of each against the analytic value, relative to the largest
force (or to the virial's trace).

usage: check-stages-fd.py DIR MODEL.mlir [--precision Double] [--h 1e-4]
           [--coordinates 6]
DIR holds system.top and system.gro (water-box.py).
"""
import argparse
import json
import os

import numpy as np
import mdir


def simulation(directory, model, precision, positions=None, scale=1.0):
    loaded = mdir.load_gromacs(os.path.join(directory, "system.top"),
                               os.path.join(directory, "system.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = 0.2
    system.pairlist_distance = 0.25
    system.models = [mdir.ModelTerm("model", model, backend="STAGES")]
    if positions is not None:
        state.positions = positions
    if scale != 1.0:
        state.positions = np.asarray(state.positions) * scale
        cell = state.cell
        cell.diagonal = np.asarray(cell.diagonal) * scale
        cell.tilt = np.asarray(cell.tilt) * scale
        state.cell = cell
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep = 0.0005
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target = mdir.Target.GPU
    execution.precision = getattr(mdir.Precision, precision)
    sim = mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                       mdir.Schedule()))
    sim.run(0, energy=True)
    return sim.state()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("directory")
    p.add_argument("model")
    p.add_argument("--precision", default="Double")
    p.add_argument("--h", type=float, default=1e-4, help="nm")
    p.add_argument("--strain", type=float, default=1e-4)
    p.add_argument("--coordinates", type=int, default=6)
    a = p.parse_args()
    model = os.path.abspath(a.model)
    base = simulation(a.directory, model, a.precision)
    x0 = np.asarray(base.positions, dtype=np.float64)
    forces = np.asarray(base.forces)
    energy = lambda x: simulation(a.directory, model, a.precision, positions=x).energies["potential"]
    order = np.argsort(-np.abs(forces).ravel())[:a.coordinates]
    worst = 0.0
    rows = []
    for flat in order:
        i, c = divmod(int(flat), 3)
        e = {}
        for k in (-2, -1, 1, 2):
            x = x0.copy()
            x[i, c] += k * a.h
            e[k] = energy(x)
        d1 = (e[1] - e[-1]) / (2 * a.h)
        d2 = (e[2] - e[-2]) / (4 * a.h)
        derivative = (4 * d1 - d2) / 3
        rows.append({"particle": i, "component": c, "force": float(forces[i, c]),
                     "difference": float(-derivative)})
        worst = max(worst, abs(-derivative - forces[i, c]))
    scale = float(np.abs(forces).max())
    # dU/d(eps) of x -> (1 + eps) x with the cell is -trace(W).
    plus = simulation(a.directory, model, a.precision, scale=1 + a.strain).energies["potential"]
    minus = simulation(a.directory, model, a.precision, scale=1 - a.strain).energies["potential"]
    trace = -(plus - minus) / (2 * a.strain)
    print(json.dumps({"precision": a.precision, "coordinates": rows,
                      "force_difference": worst / scale, "force_scale": scale,
                      "virial_trace": base.energies["virial"],
                      "virial_trace_difference": abs(trace - base.energies["virial"]),
                      "virial_trace_difference_relative":
                          abs(trace - base.energies["virial"]) / abs(base.energies["virial"])}))


if __name__ == "__main__":
    main()
