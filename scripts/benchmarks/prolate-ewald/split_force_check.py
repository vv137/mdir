"""Compare one PME force term with displaced energies on fixed-cell Amber frames.

Use a separate diagnostic build of the Python extension. In Builder.cpp's
emitTopologyPotential("energy", AllTerms) call, select Coulomb (real) or
CoulombReciprocal (mesh) in place of AllTerms. This script verifies the
generated potential against the corresponding term before evaluating it.

Arguments are the production control, term, precision and optional .npy
frames (input atom order, nm). The starting coordinates are included. Pipe
RESULT lines to force_difference.py to select displacements and estimate
numerical uncertainty. Requires NumPy, SciPy and CUDA-enabled PyTorch.
"""

import argparse
import json
from pathlib import Path
import re
import tomllib

import mdir
import numpy as np
from scipy.spatial import cKDTree
import torch


def potential(ir, name):
    match = re.search(r"^( *)md\.potential @" + name + r"(\(.*?^\1\})",
                      ir, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"missing potential {name}")
    return match.group(2)


def directions(x, box, cutoff, guard, seed=938):
    """Keep each swept pair on the same side of the cutoff for every h."""
    wrapped = x % box
    tree = cKDTree(wrapped, boxsize=box)
    rng = np.random.default_rng(seed)
    used = set()
    hmax, margin = 2.0**-9, 1e-6
    for _ in range(3):
        if not guard:
            yield (rng.choice(len(x), size=64, replace=False),
                   rng.integers(3, size=64), rng.choice([-1, 1], size=64))
            continue
        atoms, axes, signs = [], [], []
        for atom in rng.permutation(len(x)):
            if atom in used:
                continue
            if atoms:
                delta = wrapped[atoms] - wrapped[atom]
                delta -= box * np.rint(delta / box)
                if np.any(np.linalg.norm(delta, axis=1) < cutoff + 2*hmax):
                    continue
            axis = rng.integers(3)
            neighbors = tree.query_ball_point(wrapped[atom], cutoff + hmax + margin)
            delta = wrapped[neighbors] - wrapped[atom]
            delta -= box * np.rint(delta / box)
            transverse = np.sum(np.delete(delta, axis, axis=1)**2, axis=1)
            axial = np.abs(delta[:, axis])
            near = transverse + np.maximum(axial - hmax, 0)**2
            far = transverse + (axial + hmax)**2
            if np.any((near < (cutoff + margin)**2) & (far > (cutoff - margin)**2)):
                continue
            used.add(atom)
            atoms.append(atom)
            axes.append(axis)
            signs.append(rng.choice([-1, 1]))
            if len(atoms) == 32:
                break
        if len(atoms) != 32:
            raise ValueError("could not find 32 separated probes clear of the cutoff")
        yield tuple(map(np.array, (atoms, axes, signs)))


def sample(sim, x, force=False):
    with sim.borrow() as lease:
        ids = torch.from_dlpack(lease.ids)
        buffer = torch.from_dlpack(lease.positions)
        buffer.copy_(torch.from_numpy(x).to(buffer.device)[ids.long()])
        torch.cuda.synchronize(buffer.device)
        del ids, buffer
        lease.commit()
    if force:
        sim.run(1, energy=False)
        state = sim.state()
        if np.max(abs(x - state.positions)) >= 1e-7:
            raise ValueError("the force-only step moved the frame")
        return state.forces
    sim.run(0, energy=True)
    return sim.state().energies["potential"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("part", choices=["real", "mesh"])
    parser.add_argument("precision", choices=["Mixed", "Double"])
    parser.add_argument("frames", type=Path, nargs="*")
    parser.add_argument("--guard-cutoff", action="store_true")
    args = parser.parse_args()
    control = tomllib.loads(args.control.read_text())
    inputs, pme = control["input"], control["pme"]
    files = [str(args.control.parent / inputs[key]) for key in ("topology", "coordinates")]
    loaded = mdir.load_amber(*files)
    system, state = loaded.make_system(), loaded.make_state()
    box = np.asarray(state.cell.diagonal)
    if np.any(state.cell.tilt):
        raise ValueError("the probes require an orthorhombic cell")
    system.cutoff = float(control["energy"]["cutoff"]) / 10
    system.pairlist_distance = float(control["energy"]["pairlist_distance"]) / 10
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    system.pme_grid = pme["grid"]
    if pme.get("order", 4) != 4:
        raise ValueError("expected an order-four mesh")
    kernel = pme.get("kernel", "GAUSSIAN")
    if kernel == "PROLATE_C2":
        system.pme_kernel = mdir.PMEKernel.ProlateC2
        system.pme_bandwidth = pme["bandwidth"]
    elif kernel == "GAUSSIAN" and pme.get("influence") == "OPTIMAL":
        system.pme_alpha = 10 * pme["beta"]
        system.pme_influence = mdir.PMEInfluence.Optimal
    else:
        raise ValueError("expected PROLATE_C2 or Gaussian with explicit optimal influence")
    exe = mdir.Execution()
    exe.target, exe.precision = mdir.Target.GPU, getattr(mdir.Precision, args.precision)
    integrator = mdir.Integrator()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 1e-12
    initial = np.asarray(state.positions)
    state.velocities = np.zeros_like(initial)
    program = mdir.compile(system, state, integrator, mdir.Ensemble(), exe, mdir.Schedule())
    ir = program.ir
    name = "term_coulomb" if args.part == "real" else "term_reciprocal"
    if potential(ir, "energy") != potential(ir, name):
        raise ValueError(f"the diagnostic build must select {name} as its energy")
    sim = mdir.Simulation(program)
    sim.run(0, energy=True)
    powers = range(9, 15) if args.precision == "Mixed" else range(13, 19)
    for frame, path in enumerate([None, *args.frames]):
        x = initial if path is None else np.load(path)
        f = sample(sim, x, force=True)
        for group, (atoms, axes, signs) in enumerate(
                directions(x, box, system.cutoff, args.guard_cutoff)):
            projected = float(np.sum(f[atoms, axes]*signs, dtype=float))
            samples = []
            for power in powers:
                h = 2.0**-power
                for _ in range(2):
                    energies = []
                    for sign in (1, -1):
                        displaced = x.copy()
                        displaced[atoms, axes] += sign*h*signs
                        energies.append(sample(sim, displaced))
                    samples.append([h, *energies])
            print("RESULT", json.dumps(dict(frame=frame, group=group, part=args.part,
                  kernel=kernel, precision=args.precision, force=projected,
                  samples=samples)), flush=True)


if __name__ == "__main__":
    main()
