"""D[concurrent-terms]: TOWERS ordinary steps with the force terms on side
streams, launched and replayed as a captured graph (D439), against one
stream launched step by step. 20 steps of the dipeptide in water: the
positions, velocities and forces agree within a relative L2 difference of
1e-3 and the energies within 1e-4 (the tolerances of check_step_graph.py;
the runs differ by the order of floating-point atomics and of the sum of
the forces), the steps and times to the bit. The lowered program of a
graph forks both side streams inside the captured step. One launched step
with TERMS gives the forces of SINGLE within a relative L2 of 1e-5.

    check_term_streams_graph.py INPUTS WORK MDIR
"""
import json
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np

inputs, work = (Path(p).resolve() for p in sys.argv[1:3])
mdir = sys.argv[3]
deck = Path(__file__).with_name("step-graph.toml.in").read_text()


def run(name, mode, streams):
    directory = work / name
    directory.mkdir()
    control = deck.format(inputs=inputs, mode=mode, steps=20, period=4)
    control += f'term_streams = "{streams}"\n'
    (directory / "control.toml").write_text(control)
    log = subprocess.check_output([mdir, "run", "control.toml"], cwd=directory, text=True)
    start = json.loads((directory / "choices.jsonl").read_text().splitlines()[0])
    choices = start["resolved_choices"]
    assert choices["step_graph"]["value"] == mode, choices["step_graph"]
    assert choices["term_streams"]["value"] == streams, choices["term_streams"]
    with h5py.File(directory / "state.h5") as f:
        fields = {k: f["particles/all/" + k][()] for k in
                  ("position/value", "velocity/value", "force/value", "position/time", "position/step")}
    energy = [float(row.split()[1]) for row in re.findall(r"^INFO: +\d+ +(.+)$", log, re.M)]
    return directory, fields, energy


def compare(name, a, b):
    worst = 0.0
    for key in a[1]:
        x, y = a[1][key], b[1][key]
        if key.endswith("value"):
            error = np.linalg.norm(x - y) / np.linalg.norm(x)
            assert error < 1e-3, (name, key, error)
            worst = max(worst, error)
        else:
            assert x.tobytes() == y.tobytes(), (name, key)
    np.testing.assert_allclose(a[2], b[2], rtol=1e-4, atol=0.05)
    print(f"{name}: agrees, largest relative L2 difference {worst:.1e}")


def forces(name, streams):
    """The forces after one step of 1e-6 ps, launched, TOWERS."""
    directory = work / name
    directory.mkdir()
    control = deck.format(inputs=inputs, mode="LAUNCH", steps=1, period=1)
    control = control.replace("time_step  = 0.001", "time_step  = 1e-6")
    control += f'term_streams = "{streams}"\n'
    (directory / "control.toml").write_text(control)
    subprocess.check_output([mdir, "run", "control.toml"], cwd=directory, text=True)
    with h5py.File(directory / "state.h5") as f:
        return f["particles/all/force/value"][()]


# TOWERS forces of one step, TERMS against SINGLE: the order of the sum of
# the forces of the terms and the atomics differ, within the rounding of
# f32 (relative L2 1e-5).
a, b = forces("one-single", "SINGLE"), forces("one-terms", "TERMS")
error = np.linalg.norm(a - b) / np.linalg.norm(a)
assert a.size and error < 1e-5, error
print(f"one step, TERMS against SINGLE: forces agree, relative L2 {error:.1e}")

reference = run("launch-single", "LAUNCH", "SINGLE")
for mode, streams in (("LAUNCH", "TERMS"), ("GRAPH", "RECIPROCAL"), ("GRAPH", "TERMS")):
    compare(f"{mode} {streams}", reference, run(f"{mode.lower()}-{streams.lower()}", mode, streams))

directory = work / "graph-terms"
lowered = subprocess.run([mdir, "emit", "control.toml", "--stage=lowered"], cwd=directory,
                         text=True, capture_output=True, check=True).stdout
capture = lowered.find("mdrtStepCapture")
end = lowered.find("mdrtStepEnd", capture)
window = lowered[capture:end]
assert capture >= 0 and end > capture
for name in ("mdrtSideBegin", "mdrtSide2Begin", "mdrtSide2End", "mdrtSideJoin"):
    assert f"@{name}" in window, name
print("the captured step forks and joins both side streams")
