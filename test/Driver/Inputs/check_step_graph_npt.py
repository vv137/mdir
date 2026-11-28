"""Graph replay of TOWERS steps under stochastic cell rescaling (D450).

The oracle is the same program submitted with LAUNCH. TOWERS has no
deterministic mode, so the bound on GRAPH is ten times the spread of two
LAUNCH runs, with a floor (D448, GROUPS). The barostat compresses the cell
by about 6% in 200 steps, so a chunk that replayed the cell of its capture
would fail the bound by orders of magnitude; MDRT_STEP_MIRROR_FAULT=1 makes
the runtime keep the first cell and checks that the test sees it.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

inputs, work = (Path(p).resolve() for p in sys.argv[1:3])
mdir = sys.argv[3]
deck = Path(__file__).with_name("step-graph.toml.in").read_text()

# The dipeptide in a cell tilted by 2 degrees, for the six edges of a
# triclinic cell.
tilted = work / "tilted"
tilted.mkdir()
shutil.copy(inputs / "dipeptide.prmtop", tilted)
lines = (inputs / "dipeptide.inpcrd").read_text().splitlines()
box = lines[-1].split()
box[5] = "88.0"
lines[-1] = "".join(f"{float(v):12.7f}" for v in box)
(tilted / "dipeptide.inpcrd").write_text("\n".join(lines) + "\n")


def run(name, mode, steps=200, energy=10, barostat="", pressure=500.0, tau=0.1,
        source=inputs, checkpoint=None, env=None, fail=False, cutoff=8.0,
        timestep=0.001):
    directory = work / name
    directory.mkdir()
    control = deck.format(inputs=source, mode=mode, steps=steps, period=10)
    control = control.replace("cutoff            = 8.0", f"cutoff = {cutoff}")
    control = control.replace("time_step  = 0.001", f"time_step = {timestep}")
    control = control.replace("energy_interval = 10", f"energy_interval = {energy}")
    control = control.replace("checkpoint_interval = 10", f"checkpoint_interval = {energy}")
    control = control.replace('ensemble    = "NVT"', f'ensemble    = "NPT"\npressure    = {pressure}')
    control += f'\n[barostat]\nmethod = "C-RESCALE"\ntime_constant = {tau}\n{barostat}'
    if checkpoint:
        control = control.replace("[input]", f'[input]\ncheckpoint = "{checkpoint}"')
    (directory / "control.toml").write_text(control)
    proc = subprocess.run([mdir, "run", "control.toml"], cwd=directory, text=True,
                          capture_output=True, env={**os.environ, **(env or {})})
    (directory / "log").write_text(proc.stdout + proc.stderr)
    if fail:
        assert proc.returncode, name
        return proc.stdout + proc.stderr
    assert proc.returncode == 0, (name, proc.stdout[-3000:], proc.stderr[-3000:])
    with h5py.File(directory / "state.h5") as f:
        group = f["particles/all"]
        edges = group["box/edges"]
        edges = (edges["value"] if isinstance(edges, h5py.Group) else edges)[()]
        state = {"position": group["position/value"][()], "velocity": group["velocity/value"][()],
                 "edges": np.asarray(edges, dtype=float).ravel(),
                 "step": group["position/step"][()], "time": group["position/time"][()]}
    rows = [row.split() for row in re.findall(r"^INFO: +(\d+ .+)$", proc.stdout, re.M)]
    report = re.search(r"TOWERS step graph: (\d+) chunk graphs, (\d+) chunks, (\d+) cell writes,",
                       proc.stderr)
    return {"state": state, "rows": np.array(rows, dtype=float), "report": report,
            "log": proc.stdout + proc.stderr}


def difference(a, b, key):
    x, y = a["state"][key], b["state"][key]
    return float(np.linalg.norm(x - y) / np.linalg.norm(x))


def compare(launch, again, graph, label):
    """GRAPH against LAUNCH within ten times the LAUNCH spread."""
    for key in ("step", "time"):
        assert graph["state"][key].tobytes() == launch["state"][key].tobytes(), (label, key)
    for key in ("position", "velocity"):
        spread = difference(launch, again, key)
        bound = max(10 * spread, 1e-5)
        error = difference(launch, graph, key)
        print(f"{label} {key}: GRAPH {error:.3e}, LAUNCH spread {spread:.3e}, bound {bound:.3e}")
        assert error <= bound, (label, key, error, bound)
    cell = np.abs(launch["state"]["edges"] - graph["state"]["edges"]).max()
    spread = np.abs(launch["state"]["edges"] - again["state"]["edges"]).max()
    assert cell <= max(10 * spread, 1e-6), (label, cell, spread)
    # The same rows: the same steps and times, the energies within the spread.
    l, g = launch["rows"], graph["rows"]
    assert l.shape == g.shape, (label, l.shape, g.shape)
    assert (l[:, :2] == g[:, :2]).all(), label
    spread = np.abs(l - again["rows"]).max(axis=0)
    np.testing.assert_array_less(np.abs(l - g), 10 * spread + 0.05 + 1e-4 * np.abs(l), err_msg=label)
    # The cell moved, and the chunks read each new cell from the mirror
    # without a graph of their own. A scaling may leave the binary32 edges
    # as they were, and then the words are not written again.
    assert abs(l[-1, -2] / l[0, -2] - 1) > 0.01 or label != "isotropic", l[:, -2]
    graphs, chunks, writes = map(int, graph["report"].groups())
    # The chunk graphs are keyed by their banks, as at a fixed cell (D439);
    # the cell adds no key.
    assert graphs <= chunks and chunks >= 9 and chunks - 2 <= writes <= chunks, (label, graphs, chunks, writes)


launch = run("launch", "LAUNCH")
again = run("launch-again", "LAUNCH")
graph = run("graph", "GRAPH")
assert launch["report"] is None and graph["report"]
compare(launch, again, graph, "isotropic")

# Five coupling periods per row, as in production schedules: the outer
# loops of the schedule permute the banks over more chunks, and the chunk
# graphs repeat.
long = [run(f"long-{name}", mode, steps=400, energy=50)
        for name, mode in (("launch", "LAUNCH"), ("again", "LAUNCH"), ("graph", "GRAPH"))]
compare(*long, "five periods per row")

# A graph that kept its first cell: the cell is written once and the test
# sees the replay.
stale = run("stale", "GRAPH", env={"MDRT_STEP_MIRROR_FAULT": "1"})
assert int(stale["report"].group(3)) == 1, stale["report"].group(0)
for key in ("position", "velocity"):
    assert difference(launch, stale, key) > 100 * max(10 * difference(launch, again, key), 1e-5), key

# A continuation from the GRAPH checkpoint at step 200: the barostat's state
# and the cell come from the checkpoint.
checkpoint = work / "graph" / "state.h5"
resumed = [run(f"resume-{name}", mode, steps=100, energy=20, checkpoint=checkpoint)
           for name, mode in (("launch", "LAUNCH"), ("again", "LAUNCH"), ("graph", "GRAPH"))]
assert resumed[2]["state"]["step"].item() == 300
assert abs(resumed[2]["state"]["time"].item() - 0.300) < 1e-12
steps = resumed[2]["rows"][:, 0]
assert steps[-1] == 300 and (steps[steps > 200] == np.arange(220, 301, 20)).all(), steps
compare(*resumed, "continuation")
# A row interval that does not divide the restart step: the rows fall on the
# steps of LAUNCH (#1014).
resumed = [run(f"resume30-{name}", mode, steps=90, energy=30, checkpoint=checkpoint)
           for name, mode in (("launch", "LAUNCH"), ("again", "LAUNCH"), ("graph", "GRAPH"))]
compare(*resumed, "continuation, rows every 30 steps")

# The other couplings and a triclinic cell.
for label, keys, source in (("semi-isotropic", 'coupling = "SEMI_ISOTROPIC"\n', inputs),
                            ("anisotropic", 'coupling = "ANISOTROPIC"\n', inputs),
                            ("triclinic", "", tilted)):
    runs = [run(f"{label}-{name}", mode, steps=100, barostat=keys, source=source)
            for name, mode in (("launch", "LAUNCH"), ("again", "LAUNCH"), ("graph", "GRAPH"))]
    compare(*runs, label)

# At 7.35 A cutoff, y starts at 29.718 A and takes four bins. The barostat
# crosses the 29.4 A boundary, so the next force and replay use three bins.
crossing = [run(f"crossing-{name}", mode, cutoff=7.35, timestep=0.002)
            for name, mode in (("launch", "LAUNCH"), ("again", "LAUNCH"), ("graph", "GRAPH"))]
for result in crossing:
    marker = "MDIR: TOWERS launch shape "
    shape = next(json.loads(line[len(marker):]) for line in result["log"].splitlines()
                 if line.startswith(marker))
    assert shape["adaptive_grid"] and shape["grid_bins"] == [3, 4, 3], shape
    final_y = result["state"]["edges"][1]
    assert 3 * 0.735 < final_y < 4 * 0.735, final_y
compare(*crossing, "fine-grid boundary")

# A cell compressed until its faces no longer cover the cutoff. The
# scaling is followed by a step of the host's launches, whose check of the
# faces stops the run before any chunk takes the cell: GRAPH stops where
# LAUNCH does, after the same rows.
stops = []
for mode in ("LAUNCH", "GRAPH"):
    log = run(f"crushed-{mode}", mode, steps=200, pressure=5.0e4, fail=True)
    kernel = re.search(r"the last kernel launched was (\w+)", log)
    assert kernel, log[-2000:]
    stops.append((kernel[1], len(re.findall(r"^INFO: +\d+ ", log, re.M))))
assert stops[0] == stops[1], stops
print("step graph npt: ok")
