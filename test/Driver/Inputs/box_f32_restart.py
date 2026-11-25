"""Stage an unrounded checkpoint, then restore the rounded checkpoint again."""
import json
from pathlib import Path
import subprocess
import sys

import mdir
import numpy as np

binary, fixture, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
template = fixture.read_text().split("\n[input]\n", 1)[1]
template = "[input]\n" + template.replace("Inputs/", str(fixture.parent / "Inputs") + "/")
template = template.replace("0.004", "0.002").replace("= 200\n", "= 2\n").replace("= 50\n", "= 2\n")
template += 'checkpoint = "state.h5"\ncheckpoint_interval = 2\nmanifest = "run.jsonl"\n'
for mode in ("single", "mixed"):
    previous = None
    for step, precision in enumerate(("double", mode, mode)):
        directory = root / mode / str(step)
        directory.mkdir(parents=True, exist_ok=True)
        control = template + f'[execution]\nprecision = "{precision}"\nthreads = 1\n'
        if previous:
            control = control.replace("[input]", f'[input]\ncheckpoint = "{previous}"')
        (directory / "control.toml").write_text(control)
        result = subprocess.run([binary, "run", "control.toml"], cwd=directory, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        previous = directory / "state.h5"
        checkpoint = mdir.read_checkpoint(str(previous))
        edges = np.asarray(checkpoint.cell.diagonal)
        effective = json.loads((directory / "run.jsonl").read_text().splitlines()[0])["effective"]
        if step == 0:
            raw = edges.copy()
            assert np.all(raw != raw.astype(np.float32))
            assert "box_rounding_delta_nm" not in effective
        else:
            assert np.array_equal(edges, raw.astype(np.float32))
            assert effective["input_box_rounding_delta_nm"] == [*(edges - raw), 0, 0, 0]
            delta = edges - raw if step == 1 else np.zeros(3)
            assert effective["box_rounding_delta_nm"] == [*delta, 0, 0, 0]
            assert ("checkpoint cell was rounded" in result.stdout) == (step == 1)
