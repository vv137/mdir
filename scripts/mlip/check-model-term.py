#!/usr/bin/env python3
"""Checks the model term of D270 (docs/metatomic-ml1.md,
Section 7) through MDIR's Python model on a GPU: the energy, the forces, and
the virial tensor of a metatomic artifact, or an AOTInductor package of the
`nequip` package (`.nequip.pt2`, D273), evaluated by a Python simulation
against a reference (that of the `nequip` package's ASE calculator, written by water-box.py
--reference), and optionally a short run of constant energy.

The directory holds system.top, system.gro, and reference.npz (water-box.py).
Prints one JSON object: the largest differences, the reference's scale, and,
for `--steps`, the total energy of every step.

usage: check-model-term.py DIR ARTIFACT [--precision Double|Mixed]
           [--backend METATOMIC|AOTI|STAGES (by default AOTI for a .pt2)]
           [--centers-per-call N] [--steps N] [--timestep 0.0005]
           [--temperature 300] [--forces OUT.npy]
"""
import argparse
import json
import os
import time

import torch  # noqa: F401  the plugin takes the libtorch of this module
import numpy as np

import mdir


def build(directory, artifact, backend, precision, centers, timestep, temperature):
    loaded = mdir.load_gromacs(os.path.join(directory, "system.top"),
                               os.path.join(directory, "system.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    # No classical term: the topology has no charges, no bonds, and
    # Lennard-Jones of zero; the cutoff only has to fit the cell.
    system.cutoff = 0.2
    system.pairlist_distance = 0.25
    system.models = [mdir.ModelTerm("allegro", artifact, backend=backend,
                                     centers_per_call=centers)]
    # The velocities of the .gro, if it has them, else drawn.
    if temperature > 0:
        state = state.draw_velocities(system, temperature, 7)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep = timestep
    ensemble.kind = mdir.EnsembleKind.NVE
    ensemble.temperature = max(temperature, 1.0)
    execution.target = mdir.Target.GPU
    execution.precision = getattr(mdir.Precision, precision)
    return mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                        mdir.Schedule()))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("directory")
    p.add_argument("artifact")
    p.add_argument("--backend")
    p.add_argument("--precision", default="Double")
    p.add_argument("--centers-per-call", type=int, default=0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.0005)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--forces")
    a = p.parse_args()

    backend = a.backend or ("AOTI" if a.artifact.endswith(".pt2") else "METATOMIC")
    sim = build(a.directory, a.artifact, backend, a.precision, a.centers_per_call, a.timestep,
                a.temperature)
    sim.run(0, energy=True)
    state = sim.state()
    # A model in stages is a potential of the program: the run gives the
    # trace of its virial, not the tensor (D[ml4-painn-stages]).
    model = sim.model_energies() if backend != "STAGES" else None
    result = {"precision": a.precision, "centers_per_call": a.centers_per_call,
              "particles": int(len(state.positions)),
              "energy": state.energies["potential"],
              "virial": model["virial"].tolist() if model else None}
    if a.forces:
        np.save(a.forces, state.forces)
    reference = os.path.join(a.directory, "reference.npz")
    if os.path.exists(reference):
        r = np.load(reference)
        result.update(
            energy_difference=abs(state.energies["potential"] - float(r["energy"])),
            energy_reference=float(r["energy"]),
            force_difference=float(np.abs(state.forces - r["forces"]).max()),
            force_scale=float(np.abs(r["forces"]).max()),
            virial_difference=(float(np.abs(model["virial"] - r["virial"]).max())
                               if model else None),
            virial_scale=float(np.abs(r["virial"]).max()),
            virial_trace_difference=abs(state.energies["virial"]
                                        - float(np.trace(r["virial"]))))
    if a.steps:
        totals, begin = [], time.perf_counter()
        for _ in range(a.steps):
            sim.run(1, energy=True)
            totals.append(sim.state().energies["total"])
        result["seconds_per_step"] = (time.perf_counter() - begin) / a.steps
        result["total_energy"] = totals
    print(json.dumps(result))


if __name__ == "__main__":
    main()
