"""The TOWERS input kernel keeps the cell geometry of the last cell (D[towers-input-geometry]).

Thread 0 of the per-atom input kernel recomputes the geometry only when the
lattice words change. The oracle is a fresh process: a run continued from a
checkpoint starts with no geometry in memory, so its first evaluation
computes the geometry of the checkpoint's cell, and every later scaling
must be seen again. A run of 200 steps under a barostat that compresses
the cell by several percent is compared with 100 steps continued from the
checkpoint of the same run at step 100, within ten times the spread of
two uninterrupted runs. A geometry kept from an earlier cell moves the
cell offsets of the force kernel by the compression and fails the bound by
orders of magnitude (checked by keeping the first geometry).
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np

inputs, work = (Path(p).resolve() for p in sys.argv[1:3])
mdir = sys.argv[3]
deck = Path(__file__).with_name("step-graph.toml.in").read_text()

# The dipeptide in a cell tilted by 2 degrees: a triclinic cell, whose
# geometry takes the cross products, divisions and square roots.
tilted = work / "tilted"
tilted.mkdir()
shutil.copy(inputs / "dipeptide.prmtop", tilted)
lines = (inputs / "dipeptide.inpcrd").read_text().splitlines()
box = lines[-1].split()
box[5] = "88.0"
lines[-1] = "".join(f"{float(v):12.7f}" for v in box)
(tilted / "dipeptide.inpcrd").write_text("\n".join(lines) + "\n")


def run(name, mode, steps, source, checkpoint=None, barostat=True):
    directory = work / name
    directory.mkdir()
    control = deck.format(inputs=source, mode=mode, steps=steps, period=10)
    control = control.replace("checkpoint_interval = 10", "checkpoint_interval = 100")
    if barostat:
        control = control.replace('ensemble    = "NVT"', 'ensemble    = "NPT"\npressure    = 500.0')
        control += '\n[barostat]\nmethod = "C-RESCALE"\ntime_constant = 0.1\n'
    if checkpoint:
        control = control.replace("[input]", f'[input]\ncheckpoint = "{checkpoint}"')
    (directory / "control.toml").write_text(control)
    proc = subprocess.run([mdir, "run", "control.toml"], cwd=directory, text=True,
                          capture_output=True, env=dict(os.environ))
    (directory / "log").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, (name, proc.stdout[-3000:], proc.stderr[-3000:])
    with h5py.File(directory / "state.h5") as f:
        group = f["particles/all"]
        edges = group["box/edges"]
        edges = (edges["value"] if isinstance(edges, h5py.Group) else edges)[()]
        return {"position": group["position/value"][()], "velocity": group["velocity/value"][()],
                "edges": np.asarray(edges, dtype=float).ravel(),
                "step": int(np.asarray(group["position/step"][()]).ravel()[-1])}


def compare(whole, again, resumed, label):
    assert whole["step"] == resumed["step"] == 200, (label, whole["step"], resumed["step"])
    for key in ("position", "velocity"):
        spread = float(np.linalg.norm(whole[key] - again[key]) / np.linalg.norm(whole[key]))
        error = float(np.linalg.norm(whole[key] - resumed[key]) / np.linalg.norm(whole[key]))
        bound = max(10 * spread, 1e-5)
        print(f"{label} {key}: continued {error:.3e}, spread {spread:.3e}, bound {bound:.3e}")
        assert error <= bound, (label, key, error, bound)
    cell = np.abs(whole["edges"] - resumed["edges"]).max()
    spread = np.abs(whole["edges"] - again["edges"]).max()
    assert cell <= max(10 * spread, 1e-6), (label, cell, spread)


for label, source, mode in (("rectangular", inputs, "LAUNCH"), ("triclinic", tilted, "LAUNCH"),
                            ("triclinic graph", tilted, "GRAPH")):
    tag = label.replace(" ", "-")
    start = run(f"{tag}-start", "LAUNCH", 200, source)
    again = run(f"{tag}-again", "LAUNCH", 200, source)
    first = run(f"{tag}-first", "LAUNCH", 100, source)
    resumed = run(f"{tag}-resumed", mode, 100, source, checkpoint=work / f"{tag}-first" / "state.h5")
    # The barostat moved the cell: the edges at step 200 differ from those
    # at step 100 by more than a percent of volume.
    change = abs(np.prod(start["edges"][[0, 4, 8]] if start["edges"].size == 9 else start["edges"][:3])
                 / np.prod(first["edges"][[0, 4, 8]] if first["edges"].size == 9 else first["edges"][:3]) - 1)
    print(f"{label}: volume change from step 100 to 200 {change:.3%}")
    assert change > 0.01, (label, change)
    compare(start, again, resumed, label)
print("towers input geometry: ok")
