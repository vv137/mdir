"""A fixed-height launch trial honors partial pins and reuses a grid choice."""

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


def run(height):
    control = work / "run.toml"
    control.write_text(template + f'\ntowers_shape = {{{pins}, cell_height="{height}"}}\n')
    output = subprocess.check_output(
        [mdir, "run", str(control)], cwd=work, env=environment,
        stderr=subprocess.STDOUT, text=True,
    )
    prefix = "MDIR: TOWERS launch shape "
    return json.loads(next(line.split(prefix, 1)[1] for line in output.splitlines()
                           if prefix in line))


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
