"""Compare runs whose work between host boundaries is recorded and replayed
as CUDA graphs (step_graph = "SPANS") with runs that launch each kernel
(D[step-no-host-wait])."""
import json
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np

inputs, work, mdir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]

DECK = """[input]
format      = "AMBER"
topology    = "{inputs}/dipeptide.prmtop"
coordinates = "{inputs}/dipeptide.inpcrd"

[output]
manifest = "choices.jsonl"
energy_interval = 20
checkpoint = "state.h5"
checkpoint_interval = {steps}

[energy]
cutoff            = 8.0
pairlist_distance = 9.0
electrostatics    = "PME"

[pme]
grid  = [32, 32, 32]
order = 4

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {steps}
seed       = 7

[ensemble]
ensemble    = "{ensemble}"
temperature = 300.0
{pressure}{coupling}
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "{precision}"
neighbor_structure = "{structure}"
deterministic = {deterministic}
{mode}
"""
NVT = """
[thermostat]
method        = "V-RESCALE"
time_constant = 0.5
interval      = 10
"""
NPT = NVT + """
[barostat]
method        = "C-RESCALE"
time_constant = 2.0
interval      = 10
"""


def run(name, mode, precision="MIXED", structure="MATRIX", deterministic=True,
        ensemble="NVT", steps=300, env=None):
    directory = work / name
    directory.mkdir(parents=True)
    coupling = {"NVT": NVT, "NPT": NPT, "NVE": ""}[ensemble]
    line = f'step_graph = "{mode}"' if mode else ""
    (directory / "control.toml").write_text(DECK.format(
        inputs=inputs, steps=steps, ensemble=ensemble, coupling=coupling,
        pressure="pressure    = 1.0\n" if ensemble == "NPT" else "",
        precision=precision, structure=structure,
        deterministic=str(deterministic).lower(), mode=line))
    environment = dict(__import__("os").environ)
    environment.pop("MDRT_SPANS", None)
    environment.update(env or {})
    log = subprocess.run([mdir, "run", "control.toml"], cwd=directory, text=True,
                         env=environment, capture_output=True)
    assert log.returncode == 0, log.stdout + log.stderr
    start = json.loads((directory / "choices.jsonl").read_text().splitlines()[0])
    choice = start["resolved_choices"]["step_graph"]
    spans = re.search(r"^MDIR: spans: (\d+) issued as CUDA graphs and (\d+) as recorded, "
                      r"(\d+) graphs made, (\d+) captures refused, (\d+) graphs released, "
                      r"(\d+) kernel nodes updated", log.stdout, re.M)
    with h5py.File(directory / "state.h5") as f:
        state = {k: f["particles/all/" + k][()] for k in
                 ("position/value", "velocity/value", "force/value", "box/edges/value")
                 if "particles/all/" + k in f}
    rows = re.findall(r"^INFO: +\d+ .+$", log.stdout, re.M)
    return dict(choice=choice, spans=spans and tuple(map(int, spans.groups())),
                state=state, rows=rows)


def bitwise(a, b, what):
    for key in a["state"]:
        assert a["state"][key].tobytes() == b["state"][key].tobytes(), (what, key)
    assert a["rows"] == b["rows"], what


# The default is SPANS on the GPU, a rule of the target.
default = run("default", None)
assert default["choice"]["requested"] == "SPANS", default["choice"]
assert default["choice"]["value"] == "SPANS" and default["choice"]["category"] == "rule"
graphs, recorded, made, refused, released, updated = default["spans"]
assert refused == 0 and released == 0, default["spans"]
assert graphs > 0 and made > 0, default["spans"]
# Most of the work of the steps goes out as graphs.
assert graphs > recorded, default["spans"]

# Deterministic mode: the same kernels, arguments and order, so the
# checkpoint and every row of the log are bitwise those of LAUNCH, in
# mixed and double precision, in NVT and NPT.
for precision in ("MIXED", "DOUBLE"):
    for ensemble in ("NVT", "NPT"):
        tag = f"{precision.lower()}-{ensemble.lower()}"
        launch = run(f"launch-{tag}", "LAUNCH", precision, ensemble=ensemble)
        assert launch["spans"] is None, launch["spans"]
        spans = run(f"spans-{tag}", "SPANS", precision, ensemble=ensemble)
        assert spans["choice"]["value"] == "SPANS" and spans["spans"][2] > 0
        # Graphs that took the arguments of another span of their shape.
        assert spans["spans"][5] > 0, spans["spans"]
        bitwise(launch, spans, tag)
        print(f"{tag}: bitwise, {spans['spans'][0]} graph launches")

# Spans that end only at the waits of the host (MDRT_SPANS_SPLIT=0): the
# flag's event is then issued by the wait itself.
whole = run("no-split", "SPANS", env={"MDRT_SPANS_SPLIT": "0"})
assert whole["spans"][2] > 0, whole["spans"]
bitwise(default, whole, "no split")
print("no split: bitwise")

# A cache of four graphs releases graphs as it goes, and the results stay
# bitwise.
small = run("small-cache", "SPANS", env={"MDRT_SPANS_CACHE": "4"})
assert small["spans"][4] > 0, small["spans"]
bitwise(default, small, "small cache")
print(f"small cache: bitwise, {small['spans'][4]} graphs released")

# MDRT_SPANS=0 overrides the key; nothing is recorded.
off = run("override-off", "SPANS", env={"MDRT_SPANS": "0"})
assert off["spans"] is None, off["spans"]
bitwise(default, off, "override")

# The groups with a dual list are not deterministic, so two runs of LAUNCH
# differ too: over 20 steps the difference of SPANS from LAUNCH stays within
# ten times that of two LAUNCH runs (and 1e-5 relative).
def difference(a, b):
    return max(np.linalg.norm(a["state"][k] - b["state"][k]) / np.linalg.norm(a["state"][k])
               for k in ("position/value", "velocity/value"))


launch = run("launch-groups", "LAUNCH", structure="GROUPS", deterministic=False, steps=20)
again = run("launch-groups-again", "LAUNCH", structure="GROUPS", deterministic=False, steps=20)
spans = run("spans-groups", "SPANS", structure="GROUPS", deterministic=False, steps=20)
assert spans["spans"][2] > 0, spans["spans"]
bound = max(10 * difference(launch, again), 1e-5)
assert difference(launch, spans) <= bound, (difference(launch, spans), bound)
print("groups: close")
