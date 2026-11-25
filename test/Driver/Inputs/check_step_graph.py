"""Compare the state, energy and clock across TOWERS ordinary-step chunks."""
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import h5py
import numpy as np

from manifest_schema import validate

inputs, work = (Path(p).resolve() for p in sys.argv[1:3])
deck = Path(__file__).with_name("step-graph.toml.in").read_text()
schema = json.loads((Path(__file__).resolve().parents[3] / "docs/run-manifest.schema.json").read_text())


def run(name, mode, steps, period, checkpoint=None):
    directory = work / name
    directory.mkdir()
    control = deck.format(inputs=inputs, mode=mode, steps=steps, period=period)
    if checkpoint:
        control = control.replace("[input]", f'[input]\ncheckpoint = "{checkpoint}"')
    (directory / "control.toml").write_text(control)
    log = subprocess.check_output([sys.argv[3], "run", "control.toml"], cwd=directory, text=True)
    start, end = map(json.loads, (directory / "choices.jsonl").read_text().splitlines())
    assert end["step"] == (20 if checkpoint else 0) + steps, end
    choice = start["resolved_choices"]["step_graph"]
    assert choice["requested"] == mode
    if mode == "AUTO":
        validate(start, schema)
        for path in ("launch_setup", "graph_setup"):
            timing = choice[path]
            for field in ("compile_s", "warm_s", "timed_s"):
                assert timing[field] > 0, timing
        assert choice["graph_setup"]["instantiate_s"] > 0
        assert choice["launch_setup"]["instantiate_s"] == 0
        assert choice["graph_setup"]["shared_lowering"]
        pairs = choice["paired_gain_us"]
        assert len(pairs) == 4 and choice["setup_ms"] > 0
        gain = statistics.mean(pairs)
        noise = statistics.stdev(pairs)
        np.testing.assert_allclose([choice["launch_us"] - choice["graph_us"],
                                    choice["margin_us"], choice["noise_us"]],
                                   [gain, abs(gain), noise], atol=1e-10)
        assert choice["noise_factor"] == 2
        limit = 2 * noise
        winner = "GRAPH" if gain > limit else "LAUNCH"
        assert choice["value"] == winner
        assert choice["below_noise"] == (gain <= limit)
        if choice["below_noise"]:
            assert choice["reason"] == "within noise"
        logged = re.search(r"^MDIR: step_graph (.+)$", log, re.M)
        assert logged and json.loads(logged[1]) == choice
        pin = subprocess.check_output([sys.argv[3], "pin", "choices.jsonl"], cwd=directory, text=True)
        assert f'step_graph = "{winner}"' in pin and "towers_shape = {" in pin
    with h5py.File(directory / "state.h5") as f:
        fields = {k: f["particles/all/" + k][()] for k in
                  ("position/value", "velocity/value", "force/value", "position/time", "position/step")}
    energy = [float(row.split()[1]) for row in re.findall(r"^INFO: +\d+ +(.+)$", log, re.M)]
    return fields, energy


def compare(a, b):
    for key in a[0]:
        x, y = a[0][key], b[0][key]
        if key.endswith("value"):
            error = np.linalg.norm(x - y) / np.linalg.norm(x)
            assert error < 1e-3, (key, error)
        else:
            assert x.tobytes() == y.tobytes(), key
    np.testing.assert_allclose(a[1], b[1], rtol=1e-4, atol=0.05)


launch = run("launch", "LAUNCH", 20, 4)
compare(launch, run("graph", "GRAPH", 20, 4))
compare(launch, run("auto", "AUTO", 20, 4))
checkpoint = work / "graph" / "state.h5"
compare(run("resume-launch", "LAUNCH", 12, 4, checkpoint),
        run("resume-graph", "GRAPH", 12, 4, checkpoint))
fields, _ = run("partial", "GRAPH", 12, 6, checkpoint)
assert fields["position/step"].item() == 30
assert abs(fields["position/time"].item() - 0.030) < 1e-12
