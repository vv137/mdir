"""What the log and the manifest of `mdir run` record of the grid of the
prolate kernel with the B-spline window (D[prolate-bspline-grid]):
`aliasing_estimate`, `aliasing_bound`, and `grid_source`, for the automatic
grid, a given grid, and `max_spacing`; the bound eta for the constrained
density (continuity 2, eight terms by default) and eta/2 for the prolate's
own transform (continuity 0, no `basis_terms`); and a given grid whose
estimate is far above the bound, which runs."""
import json
from pathlib import Path
import re
import subprocess
import sys

from manifest_schema import validate

work, executable = Path(sys.argv[1]).resolve(), sys.argv[2]
schema = json.loads((Path(__file__).resolve().parents[3] /
                     "docs/run-manifest.schema.json").read_text())
base = (work / "base.toml").read_text()


def run(name, pme):
    """The [pme] of the start record, and the line of the log."""
    text = base.replace("#PME", 'kernel = "PROLATE"\n' + pme)
    text = text.replace("[output]\n", f'[output]\nmanifest = "{name}.jsonl"\n', 1)
    (work / f"{name}.toml").write_text(text)
    done = subprocess.run([executable, "run", f"{name}.toml"], cwd=work, text=True,
                          capture_output=True)
    output = done.stdout + done.stderr
    assert done.returncode == 0, (name, output)
    rows = [json.loads(line) for line in (work / f"{name}.jsonl").read_text().splitlines()]
    for row in rows:
        validate(row, schema)
    start = next(row for row in rows if row["event"] == "start")
    line = re.search(r"MDIR: a grid of (\d+) x (\d+) x (\d+) \(([a-z_ ]+)\), the mesh error "
                     r"estimate (\S+), the bound of the automatic grid (\S+) "
                     r"\(D\[prolate-bspline-grid\]\)", output)
    assert line, (name, output)
    pme = start["effective"]["pme"]
    grid, source = [int(k) for k in line.groups()[:3]], line.group(4)
    assert pme["window"] == "BSPLINE" and pme["grid"] == grid, (name, pme, grid)
    # The log prints two digits of each.
    assert line.group(5) == f"{pme['aliasing_estimate']:.2g}", (name, pme, line.group(5))
    assert line.group(6) == f"{pme['aliasing_bound']:.2g}", (name, pme, line.group(6))
    print(f"{name}: grid {grid}, {source}, grid_source {pme['grid_source']}, estimate "
          f"{pme['aliasing_estimate']:.4g}, bound {pme['aliasing_bound']:g}", flush=True)
    return pme, source


automatic, source = run("automatic", "continuity = 2\ntolerance = 1e-3")
assert source == "automatic" and automatic["grid_source"] == "automatic", automatic
assert automatic["basis_terms"] == 8 and automatic["aliasing_bound"] == 1e-3, automatic
assert automatic["aliasing_estimate"] <= 1e-3, automatic

# The automatic grid given: the same estimate, from the grid.
text = ", ".join(map(str, automatic["grid"]))
given, source = run("given", f"continuity = 2\ntolerance = 1e-3\ngrid = [{text}]")
assert source == "given" and given["grid_source"] == "grid", given
assert given["grid"] == automatic["grid"], given
assert given["aliasing_estimate"] == automatic["aliasing_estimate"], given

# A coarser grid given: its estimate is above the bound, and it runs.
coarse, source = run("coarse", "continuity = 2\ntolerance = 1e-3\ngrid = [12, 12, 12]")
assert source == "given" and coarse["grid_source"] == "grid", coarse
assert coarse["aliasing_estimate"] > 2 * coarse["aliasing_bound"], coarse

spaced, source = run("spaced", "continuity = 2\ntolerance = 1e-3\nmax_spacing = 1.5")
assert source == "from max_spacing" and spaced["grid_source"] == "max_spacing", spaced
assert spaced["aliasing_bound"] == 1e-3, spaced

# Continuity 0 with the B-spline window and no basis_terms: the transform
# of the prolate within its band, whose bound is eta/2.
prolate, source = run("continuity0", 'continuity = 0\nwindow = "BSPLINE"\ntolerance = 1e-3')
assert source == "automatic" and prolate["grid_source"] == "automatic", prolate
assert "basis_terms" not in prolate and prolate["aliasing_bound"] == 5e-4, prolate
assert prolate["aliasing_estimate"] <= 5e-4, prolate
print("passed")
