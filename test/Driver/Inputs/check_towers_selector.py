"""Check pinned, cached, and cross-cell TOWERS setup records."""

import json
import os
from pathlib import Path
import subprocess
import sys

mdir, example, work = sys.argv[1:]
work = Path(work).resolve()
example = Path(example).resolve()
template = (example / "argon.toml").read_text()
template = template.replace('"argon.pdb"', json.dumps(str(example / "argon.pdb")))
template = template.replace('"CPU"', '"GPU"').replace('"DOUBLE"', '"MIXED"')
template = template.replace("= 2000", "= 100")
environment = {**os.environ, "MDIR_COMPILE_CACHE_DIR": str(work / "cache"),
               "MDIR_COMPILE_CACHE": "on"}
pins = "threads=128, blocks_per_sm=1, panel=32, stripe=32, split=1"


def run(height=None, text=template):
    control = work / "run.toml"
    settings = pins + f', cell_height="{height}"' if height else pins.replace(", split=1", "")
    control.write_text(text.replace("[output]", '[output]\nmanifest = "run.jsonl"') +
                       f'\ntowers_shape = {{{settings}}}\n')
    output = subprocess.check_output(
        [mdir, "run", str(control)], cwd=work, env=environment,
        stderr=subprocess.STDOUT, text=True,
    )
    prefix = "MDIR: TOWERS launch shape "
    record = json.loads(next(line.split(prefix, 1)[1] for line in output.splitlines()
                             if prefix in line))
    manifest = next(json.loads(line) for line in (work / "run.jsonl").read_text().splitlines()
                    if json.loads(line)["event"] == "start")
    assert manifest["effective"]["towers_shape"] == record
    return record


for height in ["HALF", "FULL"]:
    first = run(height)
    assert first["selection"] == "setup trial" and first["timed_candidates"] == 2, first
    finalists = first["finalists"]
    assert len(finalists) == 2 and all(x["median_us"] > 0 for x in finalists), first
    assert first["effective"] == min(finalists, key=lambda x: x["median_us"])["shape"]
    assert {x["shape"]["records"] for x in finalists} == {16, 32}
    assert all(x["shape"]["cell_height"] == height for x in finalists)
    second = run(height)
    assert second["selection"] == "grid cache" and second["effective"] == first["effective"]

auto = run()
cells = auto["cells"]
assert {(c["shape"]["cell_height"], c["shape"]["split"]) for c in cells} == {
    (height, split) for height in ("FULL", "HALF") for split in (1, 2)
}
ranked = sorted(cells, key=lambda cell: cell["median_us"])
assert auto["winner"] == auto["effective"] == ranked[0]["shape"]
assert ranked[1]["median_us"] - ranked[0]["median_us"] == auto["margin_us"]
assert isinstance(auto["below_noise"], bool) and auto["trial_cost_ms"] > 0

pressure = template.replace('"NVE"', '"NPT"').replace("[execution]", """
[thermostat]
method = "V-RESCALE"
time_constant = 1.0
[barostat]
method = "C-RESCALE"
time_constant = 2.0
[execution]
""")
pressure_record = run("FULL", pressure)
assert pressure_record["force_kernel"] == "towers_force_virial"
assert pressure_record["cells"][0]["median_us"] > 0
