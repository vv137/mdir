"""The forces and energies at the start, and the energies over a few steps,
of a system's settings through the Python model (#357).

    forces.py SYSTEM OUT.npz [PRECISION]

The module `mdir` is the one on PYTHONPATH: run it once with main's and
once with the branch's, twice each, and compare with `compare` below; the
two runs of one module give the noise of the order of the atomic
additions. The inputs are those of steps.py (in/SYSTEM/m96.toml or
bench.toml under ROLLING_PRUNE_DIR); the topology and coordinates of the
control file, its cutoff, reaches, and PME, constraints and the groups
with the dual list, at constant energy, from its checkpoint where it names
one.

    forces.py compare A.npz B.npz
"""
import os
import sys
import tomllib

import numpy as np

STEPS, ROWS = 20, 5

if sys.argv[1] == "compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    scale = np.abs(a["forces"]).max()
    force = np.abs(a["forces"] - b["forces"]).max()
    rms = np.sqrt(((a["forces"] - b["forces"]) ** 2).mean())
    start = np.abs(a["start"] - b["start"]) / np.maximum(np.abs(a["start"]), 1.0)
    rows = np.abs(a["rows"] - b["rows"]) / np.maximum(np.abs(a["rows"]), 1.0)
    print(f"forces: largest difference {force:.3e} kJ/mol/nm (rms {rms:.3e}) of {scale:.0f}; "
          f"energies at the start: {start.max():.2e} relative ({a['names'][start.argmax()]}); "
          f"over {STEPS * ROWS} steps: {rows.max():.2e}")
    sys.exit(0)

import mdir  # noqa: E402

name, out = sys.argv[1], sys.argv[2]
precision = sys.argv[3] if len(sys.argv) > 3 else "Mixed"
S = os.environ.get("ROLLING_PRUNE_DIR", os.getcwd())
src = f"{S}/in/{name}"
path = f"{src}/{'bench' if name == 'ubq' else 'm96'}.toml"
control = tomllib.load(open(path, "rb"))
here = lambda p: p if p.startswith("/") else f"{src}/{p}"
loaded = mdir.load_amber(here(control["input"]["topology"]), here(control["input"]["coordinates"]))
system, state = loaded.make_system(), loaded.make_state()
energy, pme = control["energy"], control.get("pme", {})
system.cutoff = energy["cutoff"] / 10
system.pairlist_distance = energy["pairlist_distance"] / 10
system.pruned_distance = energy["pruned_distance"] / 10
system.truncation = mdir.Truncation.None_
system.electrostatics = mdir.Electrostatics.PME
system.periodic = True
if "grid" in pme:
    system.pme_grid = pme["grid"]
if "beta" in pme:
    system.pme_alpha = pme["beta"] * 10
if "tolerance" in pme:
    system.pme_tolerance = pme["tolerance"]
if "max_spacing" in pme:
    system.pme_spacing = pme["max_spacing"] / 10
system.rigid_hydrogen_bonds = system.rigid_water = True
state = state.draw_velocities(system, 300.0, 2026)
integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
integrator.timestep = control["dynamics"]["time_step"]
ensemble.kind = mdir.EnsembleKind.NVE
execution.target = mdir.Target.GPU
execution.precision = getattr(mdir.Precision, precision)
execution.neighbor_structure = mdir.NeighborStructure.Groups
program = mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())
# The equilibrated start of the control file, where it names one.
start = control["input"].get("checkpoint")
simulation = (mdir.Simulation(program, checkpoint=here(start), stage=True) if start
              else mdir.Simulation(program))
simulation.run(0, energy=True)
first = simulation.state()
names = sorted(first.energies)
rows = []
for _ in range(ROWS):
    simulation.run(STEPS, energy=True)
    rows.append([simulation.state().energies[k] for k in names])
np.savez(out, forces=np.asarray(first.forces), names=np.array(names),
         start=np.array([first.energies[k] for k in names]), rows=np.array(rows))
print(f"{name} {precision}: {len(np.asarray(first.forces))} particles, written {out}")
