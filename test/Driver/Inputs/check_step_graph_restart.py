"""Continue TOWERS graph replay at a step that is not a multiple of the
intervals and compare every observation with the launched steps (#1014).

A checkpoint at step 20 is continued with several couplings, energy,
trajectory and checkpoint intervals and numbers of steps. For each, the
steps of the energy rows, of the frames and of the checkpoint, the energy
rows themselves and the final state of `step_graph = "GRAPH"` must be those
of `step_graph = "LAUNCH"`, and the rows and frames are those due, counted
from the start step.
"""
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np

inputs, work = (Path(p).resolve() for p in sys.argv[1:3])
mdir = sys.argv[3]
deck = Path(__file__).with_name("step-graph.toml.in").read_text()


def control(mode, steps, couple, energy, frame, checkpoint, restart=None):
    text = deck.format(inputs=inputs, mode=mode, steps=steps, period=energy)
    text = re.sub(r"^checkpoint_interval = .*$", f"checkpoint_interval = {checkpoint}", text,
                  flags=re.M)
    text = re.sub(r"^interval = .*$", f"interval = {couple}", text, flags=re.M)
    text = text.replace("[output]\n", '[output]\nenergy = "energy.dat"\n'
                        f'trajectory = "frames.h5md"\ntrajectory_interval = {frame}\n', 1)
    if restart:
        text = text.replace("[input]", f'[input]\ncheckpoint = "{restart}"', 1)
    return text


def run(name, mode, steps, couple, energy, frame, checkpoint, restart=None):
    directory = work / name
    directory.mkdir()
    (directory / "control.toml").write_text(
        control(mode, steps, couple, energy, frame, checkpoint, restart))
    done = subprocess.run([mdir, "run", "control.toml"], cwd=directory, text=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if done.returncode:
        raise RuntimeError(f"{mode} failed: {done.stderr.strip().splitlines()[-1:]}")
    rows = [line.split() for line in (directory / "energy.dat").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    steps_of_rows = [int(float(r[0])) for r in rows]
    values = np.array([[float(x) for x in r[1:]] for r in rows])
    with h5py.File(directory / "frames.h5md") as f:
        frames = f["particles/all/position/step"][()].ravel().tolist()
    with h5py.File(directory / "state.h5") as f:
        state = {k: f["particles/all/" + k][()] for k in
                 ("position/value", "velocity/value", "position/step", "position/time")}
    return steps_of_rows, values, frames, state


start = 20
run("start", "LAUNCH", start, 4, 4, 4, start)
restart = work / "start" / "state.h5"
failures = 0
# (steps, coupling, energy, trajectory, checkpoint): phases of the start
# step within the coupling and within the energy interval, frames more and
# less frequent than energies, fewer steps than one energy interval.
cases = [(12, 6, 6, 6, 12), (18, 3, 6, 3, 18), (32, 4, 8, 16, 32), (24, 8, 8, 8, 24),
         (30, 5, 15, 30, 30), (30, 3, 30, 30, 30), (16, 4, 16, 8, 16), (32, 4, 4, 4, 8)]
for case in cases:
    steps, couple, energy, frame, checkpoint = case
    tag = "-".join(map(str, case))
    try:
        launch = run("launch-" + tag, "LAUNCH", steps, couple, energy, frame, checkpoint, restart)
        graph = run("graph-" + tag, "GRAPH", steps, couple, energy, frame, checkpoint, restart)
    except RuntimeError as error:
        print(f"start {start} steps {steps}: {error}")
        failures += 1
        continue
    # Intervals count from the step the run starts at.
    due = list(range(start + energy, start + steps + 1, energy))
    frames = list(range(start + frame, start + steps + 1, frame))
    problems = []
    if launch[0] != due:
        problems.append(f"LAUNCH rows at {launch[0]}, due {due}")
    if launch[2] != frames:
        problems.append(f"LAUNCH frames at {launch[2]}, due {frames}")
    if graph[0] != launch[0]:
        problems.append(f"GRAPH rows at {graph[0]}, LAUNCH at {launch[0]}")
    elif len(due) and not np.allclose(graph[1], launch[1], rtol=1e-4, atol=0.05):
        problems.append("GRAPH energies differ from LAUNCH")
    if graph[2] != launch[2]:
        problems.append(f"GRAPH frames at {graph[2]}, LAUNCH at {launch[2]}")
    for key in launch[3]:
        x, y = launch[3][key], graph[3][key]
        if key.endswith("value"):
            error = np.linalg.norm(x - y) / np.linalg.norm(x)
            if not error < 1e-3:
                problems.append(f"{key} differs by {error:.2e}")
        elif x.tobytes() != y.tobytes():
            problems.append(f"{key}: GRAPH {y.ravel()}, LAUNCH {x.ravel()}")
    print(f"start {start} steps {steps} coupling {couple} energy {energy} "
          f"trajectory {frame} checkpoint {checkpoint}: "
          + ("same" if not problems else "; ".join(problems)))
    failures += bool(problems)
sys.exit(1 if failures else 0)
